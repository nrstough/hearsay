"""M5 bundle: every training/test clip decoded once through the project loader, silence-trimmed
exactly as at test time, and stored as 16 kHz PCM16 FLAC with its trivial-shortcut features and a
PCM hash (cross-corpus dedup). Core and test rows are never dropped; extras may be.

The bundle is versioned (bundle/v1) and identified by a tree sha over every file in it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from hearsay import SR
from hearsay.audio import DecodeError, load_audio, trim_silence

MAX_CLIP_S = 15.0
PEAK = 0.999


def leading_silence_s(x: np.ndarray, top_db: float = 35.0, frame: int = SR // 50) -> float:
    """Seconds before the first 20 ms frame within `top_db` of the loudest frame."""
    n = x.size // frame
    if n < 2:
        return 0.0
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    keep = np.where(db > -top_db)[0]
    return float(keep[0] * frame / SR) if keep.size else 0.0


def prepare_clip(x: np.ndarray, max_s: float = MAX_CLIP_S) -> np.ndarray:
    """trim_silence -> cap -> peak-scale to PEAK if above (harmless: training normalizes)."""
    x = trim_silence(x)[: int(max_s * SR)]
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > PEAK:
        x = x * (PEAK / peak)
    return np.ascontiguousarray(x, dtype=np.float32)


def write_clip(src: str | Path, dst: str | Path, *, max_s: float = MAX_CLIP_S,
               min_s: float | None = None) -> dict:  # fmt: skip
    """Decode, prepare and write one clip. Returns a stats dict, {"dropped": "short"} when the
    trimmed clip is under `min_s` (extras only), or {"error": ...} on a decode failure."""
    try:
        raw = load_audio(src)
    except DecodeError as e:
        return {"error": str(e)[:200]}
    x = prepare_clip(raw, max_s)
    if min_s is not None and x.size < int(min_s * SR):
        return {"dropped": "short", "duration": round(x.size / SR, 4)}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, x, SR, subtype="PCM_16", format="FLAC")
    pcm16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    return {
        "file": dst.name,
        "duration": round(x.size / SR, 4),
        "raw_duration": round(raw.size / SR, 4),
        "peak": round(float(np.abs(x).max()) if x.size else 0.0, 5),
        "lead_s": round(leading_silence_s(x), 4),
        "pcm_sha256": hashlib.sha256(pcm16.tobytes()).hexdigest(),
    }


def read_clip(path: str | Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR, (path, sr)
    return np.ascontiguousarray(x, dtype=np.float32)


def clip_ok(path: Path, expected_frames: int | None = None) -> bool:
    """A resumable build reuses an existing FLAC only if it opens and matches the manifest."""
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError, sf.LibsndfileError):
        return False
    if info.samplerate != SR or info.channels != 1:
        return False
    return expected_frames is None or info.frames == expected_frames


def source_stamp(src: str | Path) -> str:
    """Identity of a source file for resumable builds: size and mtime (a changed source at the
    same path must be re-decoded, never reused)."""
    st = Path(src).stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def tree_sha(root: str | Path,
             exclude: tuple[str, ...] = ("TREE_SHA", "bundle_meta.json")) -> str:
    """sha256 over sorted (relative path, file sha256) of every file under root, except the
    two files that record the sha itself."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file() and q.name not in exclude
                    and q.relative_to(root).parts[0] not in exclude):
        h.update(str(p.relative_to(root)).encode())
        h.update(b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\n")
    return h.hexdigest()


def write_meta(root: Path, **fields) -> None:
    (root / "bundle_meta.json").write_text(json.dumps(fields, indent=2, sort_keys=True))
