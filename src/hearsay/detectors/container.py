"""Container / metadata detector (brief rubric, technique 1): embedded header facts only.

Reads, with ffprobe: container format, codec, sample rate, channels, bit depth, duration and
every embedded tag (encoder, software, comment, creation_time, ...). Never filesystem MAC
times and never the filename: git, Docker and unzip rewrite those, and the brief lists them as
spoofable.

Two jobs:
1. Routing facts for the orchestrator (`features`): lossy codec, PCM WAV, native sample rate,
   FFmpeg-written header, tag count. Compression forensics, for one, only makes sense when
   the file is lossy or was re-muxed.
2. A rule-based score, never learned. 0.5 ("no class evidence in the container") unless an
   embedded tag names a speech-synthesis tool (0.9) or claims synthesis (0.75). It is not
   learned on purpose: on the training data the container *is* the label (LibriSpeech = FLAC
   16 kHz, LJ = 22 kHz WAV, DiffSSD = 22/24/44.1 kHz WAV or MP3), so any fitted model would be
   a shortcut, and the NSA test set defeats it: all 1,671 files are PCM16 16 kHz mono WAV
   with one identical tag (encoder=Lavf58.29.100). See docs/reports/ for the measurement.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from hearsay.detectors.base import REGISTRY, ClipContext, DetectorResult, register

NAME = "container"
LOSSY = {"mp3", "aac", "opus", "vorbis", "wmav1", "wmav2", "wmapro", "ac3", "eac3", "amr_nb",
         "amr_wb", "gsm", "gsm_ms", "adpcm_ima_wav", "adpcm_ms", "g722", "g726"}  # fmt: skip
TAG_VALUES = ("encoder", "software", "comment", "title", "artist", "date", "creation_time",
              "handler_name", "encoded_by", "tool", "major_brand")  # fmt: skip
AI_TOOL_TERMS = (
    "elevenlabs", "eleven labs", "play.ht", "playht", "resemble", "descript", "murf",
    "speechify", "tortoise", "coqui", "xtts", "bark", "suno", "polly", "wellsaid", "lovo",
    "respeecher", "voicebox", "openvoice", "styletts", "tacotron", "fastspeech", "wavegrad",
    "diffwave", "hifigan", "hifi-gan", "text-to-speech", "text to speech", "tts",
)
SYNTH_TERMS = ("synthetic", "synthesized", "synthesised", "generated", "cloned", "deepfake",
               "ai voice", "ai-generated")  # fmt: skip


def probe_container(path: str | Path) -> dict:
    """ffprobe facts + lower-cased tags for one file. Never raises; `probe_ok` says whether
    ffprobe found an audio stream."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
           "-select_streams", "a:0", str(path)]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, check=False)
    try:
        info = json.loads(proc.stdout or b"{}")
    except json.JSONDecodeError:
        info = {}
    streams, fmt = info.get("streams") or [], info.get("format", {})
    s = streams[0] if streams else {}
    tags = {str(k).lower(): str(v) for k, v in {**fmt.get("tags", {}), **s.get("tags", {})}.items()}
    row = {
        "probe_ok": proc.returncode == 0 and bool(streams),
        "format": fmt.get("format_name"),
        "codec": s.get("codec_name"),
        "sample_rate": int(s["sample_rate"]) if s.get("sample_rate") else None,
        "channels": s.get("channels"),
        "bits": s.get("bits_per_sample") or s.get("bits_per_raw_sample"),
        "sample_fmt": s.get("sample_fmt"),
        "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        "duration_s": float(fmt["duration"]) if fmt.get("duration") else None,
        "nb_streams": fmt.get("nb_streams"),
        "n_tags": len(tags),
        "tag_keys": "|".join(sorted(tags)),
        "tags": tags,
        "error": "" if proc.returncode == 0 else proc.stderr.decode(errors="replace")[:200],
    }
    for k in TAG_VALUES:
        row[f"tag_{k}"] = tags.get(k)
    return row


def _term_hit(blob: str, terms: tuple[str, ...]) -> str | None:
    for t in terms:
        if t == "tts":
            if re.search(r"\btts\b", blob):
                return t
        elif t in blob:
            return t
    return None


def classify_tags(tags: dict[str, str]) -> tuple[float, str]:
    """Rule-based score from embedded tag values (keys are not evidence)."""
    blob = " ".join(tags.values()).lower()
    hit = _term_hit(blob, AI_TOOL_TERMS)
    if hit:
        return 0.9, f"an embedded tag names a speech-synthesis tool ({hit!r})"
    hit = _term_hit(blob, SYNTH_TERMS)
    if hit:
        return 0.75, f"an embedded tag claims synthesis ({hit!r})"
    return 0.5, "no class evidence in the container"


class ContainerDetector:
    """Header facts for routing plus a tag-rule score; runs on every file (it is cheap)."""

    name = NAME

    def applies(self, ctx: ClipContext) -> bool:
        return True

    def run(self, ctx: ClipContext) -> DetectorResult:
        info = ctx.memo("container.probe", lambda: probe_container(ctx.path))
        if not info["probe_ok"]:
            raise RuntimeError(f"ffprobe found no audio stream: {info['error'] or 'unknown'}")
        codec, fmt = info["codec"] or "?", info["format"] or "?"
        lossy = codec in LOSSY
        enc = info["tags"].get("encoder", "")
        ffmpeg_written = enc.lower().startswith("lavf")
        score, why = classify_tags(info["tags"])
        feats = {
            "lossy": float(lossy),
            "is_pcm_wav": float(fmt == "wav" and codec.startswith("pcm_")),
            "sample_rate": float(info["sample_rate"] or 0),
            "channels": float(info["channels"] or 0),
            "bits": float(info["bits"] or 0),
            "duration_s": float(info["duration_s"] or 0.0),
            "n_tags": float(info["n_tags"]),
            "has_encoder_tag": float(bool(enc)),
            "ffmpeg_written": float(ffmpeg_written),
            "rule_score": score,
        }
        ch = {1: "mono", 2: "stereo"}.get(info["channels"], f"{info['channels']}ch")
        bits = f" {info['bits']}-bit" if info["bits"] else ""
        tag_txt = ", ".join(f"{k}={v}" for k, v in sorted(info["tags"].items())[:4]) or "none"
        notes = [why]
        if ffmpeg_written:
            notes.append("written by FFmpeg (libavformat), so the file was re-muxed or "
                         "transcoded at least once")  # fmt: skip
        if lossy:
            notes.append(f"lossy codec ({codec}); compression forensics apply")
        evidence = (f"{fmt}/{codec} {info['sample_rate'] or '?'} Hz {ch}{bits}, "
                    f"{info['n_tags']} tag(s) [{tag_txt}]: " + "; ".join(notes))  # fmt: skip
        return DetectorResult(self.name, score, evidence, feats)


DETECTOR = ContainerDetector()
if NAME not in REGISTRY.names():
    register(DETECTOR)
