"""Container / metadata detector against the contract; files are generated with ffmpeg."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from hearsay.detectors import base
from hearsay.detectors.base import ClipContext, safe_run
from hearsay.detectors.container import (
    ContainerDetector,
    classify_tags,
    probe_container,
)

SINE = "sine=frequency=440:sample_rate=16000:duration=1"


def make(path, *args, codec=None):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", SINE, "-ac", "1"]
    if codec:
        cmd += ["-acodec", codec]
    subprocess.run([*cmd, *args, str(path)], check=True)
    return path


def has_encoder(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                         check=False)
    return name in out.stdout


@pytest.fixture
def det():
    return ContainerDetector()


def test_plain_pcm_wav_is_neutral_with_routing_facts(det, tmp_path):
    r = safe_run(det, ClipContext(make(tmp_path / "a.wav")))
    assert r.status == "ok" and r.score == 0.5 and r.name == "container"
    f = r.features
    assert f["is_pcm_wav"] == 1.0 and f["lossy"] == 0.0 and f["sample_rate"] == 16000.0
    assert f["channels"] == 1.0 and f["bits"] == 16.0 and f["rule_score"] == 0.5
    assert f["ffmpeg_written"] == 1.0 and f["has_encoder_tag"] == 1.0  # ffmpeg writes ISFT
    assert "no class evidence" in r.evidence and "wav/pcm_s16le 16000 Hz mono 16-bit" in r.evidence
    assert "FFmpeg" in r.evidence


@pytest.mark.skipif(not has_encoder("libmp3lame"), reason="ffmpeg lacks libmp3lame")
def test_lossy_codec_is_flagged_for_routing_but_stays_neutral(det, tmp_path):
    r = safe_run(det, ClipContext(make(tmp_path / "a.mp3", "-b:a", "64k", codec="libmp3lame")))
    assert r.status == "ok" and r.score == 0.5
    assert r.features["lossy"] == 1.0 and r.features["is_pcm_wav"] == 0.0
    assert "compression forensics apply" in r.evidence


def test_tag_naming_a_synthesis_tool_scores_high(det, tmp_path):
    p = make(tmp_path / "a.wav", "-metadata", "comment=made with ElevenLabs v2")
    r = safe_run(det, ClipContext(p))
    assert r.status == "ok" and r.score == 0.9 and "elevenlabs" in r.evidence
    assert r.features["rule_score"] == 0.9 and r.features["n_tags"] >= 2


def test_tag_claiming_synthesis_scores_medium(det, tmp_path):
    p = make(tmp_path / "a.wav", "-metadata", "title=cloned voice sample")
    r = safe_run(det, ClipContext(p))
    assert r.status == "ok" and r.score == 0.75 and "cloned" in r.evidence


@pytest.mark.parametrize(
    ("tags", "score"),
    [
        ({"encoder": "Lavf58.29.100"}, 0.5),
        ({}, 0.5),
        ({"comment": "Watts per channel"}, 0.5),  # 'tts' inside a word does not match
        ({"software": "Coqui TTS 0.22"}, 0.9),
        ({"artist": "ai-generated narrator"}, 0.75),
        ({"tts": "not a value hit"}, 0.5),  # keys are not evidence
    ],
)
def test_tag_rules(tags, score):
    assert classify_tags(tags)[0] == score


def test_unreadable_file_is_an_error_result_not_a_crash(det, tmp_path):
    p = tmp_path / "junk.wav"
    p.write_bytes(b"RIFF....WAVEjunk")
    r = safe_run(det, ClipContext(p))
    assert r.status == "error" and r.score == 0.5 and "ffprobe" in r.error


def test_deterministic_memoized_and_filename_blind(det, tmp_path):
    a = make(tmp_path / "a.wav")
    b = tmp_path / "elevenlabs_synthetic_generated.wav"
    shutil.copy(a, b)
    ca, cb = ClipContext(a), ClipContext(b)
    ra, rb = safe_run(det, ca), safe_run(det, cb)
    assert ra == rb and ra == safe_run(det, ca)  # same bytes, any name -> same verdict
    assert "container.probe" in ca.cache


def test_probe_reports_no_filesystem_fields(tmp_path):
    info = probe_container(make(tmp_path / "a.wav"))
    assert info["probe_ok"] and info["codec"] == "pcm_s16le"
    assert not any(k in info for k in ("mtime", "ctime", "atime", "filename", "path"))


def test_registered_under_its_name():
    assert "container" in base.REGISTRY.names()
