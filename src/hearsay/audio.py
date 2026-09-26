"""The one audio path: any container/codec -> 16 kHz mono float32.

Train, validation, test, and live audio all go through `load_audio`, so no model ever sees a
second resampler or decoder. It never denoises or loudness-normalizes: codec artifacts are signal.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from hearsay import SR


class DecodeError(RuntimeError):
    """FFmpeg could not produce any audio samples from the file."""


def load_audio(path: str | Path) -> np.ndarray:
    """Decode the first audio stream of `path` to a 1-D float32 array at 16 kHz mono.

    Channels are averaged by FFmpeg's downmix. Non-finite samples (NaN/inf in float WAVs) are
    zeroed. Raises DecodeError if FFmpeg fails or yields no samples.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error",
        "-i", str(path),
        "-map", "0:a:0", "-ac", "1", "-ar", str(SR),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, check=False)
    x = np.frombuffer(proc.stdout, dtype="<f4")
    if x.size == 0:
        err = proc.stderr.decode(errors="replace").strip().splitlines()
        raise DecodeError(f"{path}: no samples (ffmpeg rc={proc.returncode}: {err[-1:]})")
    # A truncated file can decode partially with rc != 0; keep what decoded.
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(x, dtype=np.float32)


def probe_audio(path: str | Path) -> dict:
    """Container/codec facts for inventory: format, codec, sample_rate, channels, duration_s.

    Returns {"error": ...} instead of raising, so an inventory never stops on one bad file.
    """
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-select_streams", "a:0", str(path),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, check=False)
    try:
        info = json.loads(proc.stdout or b"{}")
    except json.JSONDecodeError:
        info = {}
    streams = info.get("streams") or []
    if proc.returncode != 0 or not streams:
        return {"error": proc.stderr.decode(errors="replace").strip()[:200] or "no audio stream"}
    s, fmt = streams[0], info.get("format", {})
    dur = s.get("duration") or fmt.get("duration")
    return {
        "format": fmt.get("format_name"),
        "codec": s.get("codec_name"),
        "sample_rate": int(s["sample_rate"]) if s.get("sample_rate") else None,
        "channels": s.get("channels"),
        "duration_s": float(dur) if dur else None,
        "bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
    }
