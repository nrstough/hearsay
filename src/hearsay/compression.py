"""Compression forensics features (brief rubric, technique 5), CPU only.

The NSA test set is PCM16 16 kHz WAV throughout, so the container says nothing about codec
history; only the signal can. These features look for what a lossy codec leaves behind after
decoding and resampling to 16 kHz:
- spectral holes: MP3/AAC zero out scale-factor bands the psychoacoustic model calls masked.
  Measured two ways per frame in 3-7 kHz (the test-set band match empties 7.25-8 kHz for
  every clip, so that band would swamp the statistic): cells far below the clip peak (deep
  holes) and cells far below the frame's own band level (local holes); plus how the hole mask
  flickers frame to frame ("birdies") and how many hole runs a frame has.
- floor depth: how far the quietest cells sit below the clip peak. Codec zeros and clean
  vocoder output sit deeper than a microphone's noise floor.
- effective bandwidth and high-band tilt: the highest 250 Hz band still within 40 dB of the
  1-3 kHz level (telephony 3.4-4 kHz, low-bitrate lowpass, band-limited laundering) and the
  band levels above 4 kHz relative to 1-3 kHz.
Same clip preparation as the other engineered detectors (`hearsay.handcrafted._crop`:
test-set band match, trim, optional test-length crop, 8 s cap, RMS normalization).

`launder` re-encodes a clip through MP3 or AAC at a chosen bitrate and sample rate with
ffmpeg and decodes it back to 16 kHz PCM. Training applies it to a random half of *both*
classes so a learned model cannot key on "MP3 = one particular generator" (in DiffSSD only
ElevenLabs and PlayHT are MP3, and every real clip is lossless).
"""

from __future__ import annotations

import subprocess

import numpy as np

from hearsay import SR
from hearsay.audio import DecodeError, load_audio
from hearsay.handcrafted import _crop

N_FFT = 1024
HOP = 256
DEEP_HOLE_DB = -75.0  # below the clip peak
LOCAL_HOLE_DB = -30.0  # below the frame's 3-8 kHz median
BW_DB = 40.0
HOLE_BAND = (3000.0, 7000.0)  # below the band-match cutoff
LAUNDER_CODECS = ("mp3", "mp3", "mp3", "aac")  # weights: MP3 is the common case
LAUNDER_KBPS = (24, 32, 48, 64, 96, 128)
LAUNDER_SRS = (16000, 22050, 44100)


def features(
    x: np.ndarray, crop_s: float | None = None, seed: int | None = None,
    crop_mode: str = "segment", band_match: bool = True,
) -> dict[str, float]:  # fmt: skip
    import librosa

    x = _crop(x, crop_s, seed, crop_mode, band_match).astype(np.float32)
    S = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    db = 10 * np.log10(S + 1e-12)
    rel_clip = db - db.max()
    lt = 10 * np.log10(S.mean(axis=1) + 1e-12)  # long-term spectrum
    ref = float(lt[(freqs >= 1000) & (freqs <= 3000)].mean())
    f: dict[str, float] = {}

    bw = 3000.0
    for lo in np.arange(3000, 8000, 250):
        m = (freqs >= lo) & (freqs < lo + 250)
        if np.median(lt[m]) >= ref - BW_DB:
            bw = float(lo + 250)
        else:
            break
    f["bw_hz"] = bw
    for lo, hi in ((4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000)):
        m = (freqs >= lo) & (freqs < hi)
        f[f"band_{lo // 1000}_{hi // 1000}k_db"] = float(lt[m].mean() - ref)
    m = (freqs >= 2000) & (freqs < 8000)
    f["hf_slope_db_per_khz"] = float(np.polyfit(freqs[m] / 1000, lt[m], 1)[0])

    band = (freqs >= HOLE_BAND[0]) & (freqs < HOLE_BAND[1])
    rb = rel_clip[band]
    deep = rb < DEEP_HOLE_DB
    local = rb < (np.median(rb, axis=0, keepdims=True) + LOCAL_HOLE_DB)
    for name, holes in (("deep", deep), ("local", holes_local := local)):
        hpf = holes.mean(axis=0)
        f[f"{name}_hole_frac"] = float(hpf.mean())
        f[f"{name}_hole_flicker"] = float(np.abs(np.diff(hpf)).mean()) if hpf.size > 1 else 0.0
        if holes.shape[1] > 1:
            a, b = holes[:, :-1], holes[:, 1:]
            inter, union = (a & b).sum(axis=0), (a | b).sum(axis=0)
            f[f"{name}_hole_persistence"] = float(
                np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0))
            )
        else:
            f[f"{name}_hole_persistence"] = 1.0
        edges = np.diff(np.pad(holes.astype(np.int8), ((1, 1), (0, 0))), axis=0)
        f[f"{name}_hole_runs"] = float((edges == 1).sum(axis=0).mean())
    del holes_local
    f["floor_p2_db"] = float(np.percentile(rb, 2, axis=0).mean())
    f["floor_p10_db"] = float(np.percentile(rb, 10, axis=0).mean())
    f["valley_depth_db"] = float(
        (np.percentile(rb, 50, axis=0) - np.percentile(rb, 2, axis=0)).mean()
    )
    low = rel_clip[(freqs >= 300) & (freqs < 3000)]
    f["floor_p2_lowband_db"] = float(np.percentile(low, 2, axis=0).mean())
    f["clip_floor_db"] = float(np.percentile(rel_clip, 1))
    return {k: (v if np.isfinite(v) else 0.0) for k, v in f.items()}


def launder(x: np.ndarray, codec: str = "mp3", kbps: int = 64, sr_enc: int = 44100) -> np.ndarray:
    """16 kHz float32 -> ffmpeg encode (codec at kbps, resampled to sr_enc) -> decode back to
    16 kHz float32. Raises DecodeError if either ffmpeg pass yields nothing."""
    enc = {"mp3": ["-acodec", "libmp3lame", "-f", "mp3"], "aac": ["-acodec", "aac", "-f", "adts"]}
    if codec not in enc:
        raise ValueError(f"codec must be one of {sorted(enc)}, got {codec!r}")
    p1 = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-f", "f32le", "-ar", str(SR),
         "-ac", "1", "-i", "-", "-ar", str(sr_enc), "-b:a", f"{kbps}k", *enc[codec], "-"],
        input=np.ascontiguousarray(x, dtype=np.float32).tobytes(), capture_output=True,
        check=False,
    )  # fmt: skip
    if p1.returncode != 0 or not p1.stdout:
        raise DecodeError(f"launder encode failed ({codec} {kbps}k @ {sr_enc}): "
                          f"{p1.stderr.decode(errors='replace')[-200:]}")  # fmt: skip
    p2 = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-f", codec if codec == "mp3"
         else "aac", "-i", "-", "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le",
         "-acodec", "pcm_f32le", "-"],
        input=p1.stdout, capture_output=True, check=False,
    )  # fmt: skip
    y = np.frombuffer(p2.stdout, dtype="<f4")
    if y.size == 0:
        raise DecodeError(f"launder decode failed: {p2.stderr.decode(errors='replace')[-200:]}")
    return np.ascontiguousarray(np.nan_to_num(y), dtype=np.float32)


def draw_laundering(rng: np.random.Generator, frac: float) -> str:
    """"" (clean) with probability 1 - frac, else a spec like "mp3-64k@44100"."""
    if rng.random() >= frac:
        return ""
    codec = LAUNDER_CODECS[int(rng.integers(len(LAUNDER_CODECS)))]
    kbps = LAUNDER_KBPS[int(rng.integers(len(LAUNDER_KBPS)))]
    sr = LAUNDER_SRS[int(rng.integers(len(LAUNDER_SRS)))]
    return f"{codec}-{kbps}k@{sr}"


def parse_laundering(spec: str) -> tuple[str, int, int] | None:
    if not spec:
        return None
    codec, rest = spec.split("-", 1)
    kbps, sr = rest.split("@")
    return codec, int(kbps.rstrip("k")), int(sr)


def features_for_path(
    path: str, crop_s: float | None = None, seed: int | None = None,
    crop_mode: str = "segment", launder_spec: str = "", band_match: bool = True,
) -> dict[str, float] | None:  # fmt: skip
    try:
        x = load_audio(path)
        spec = parse_laundering(launder_spec)
        if spec:
            x = launder(x, *spec)  # codec history first, then the test-set band match
        return features(x, crop_s, seed, crop_mode, band_match)
    except Exception:  # noqa: BLE001 - one bad file becomes a missing row, reported by caller
        return None
