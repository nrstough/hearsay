"""On-box codec round-trips for a seeded subset of training clips and every holdout clip.

Each selected clip gets ONE random codec (mp3 / opus / aac / amrnb / mulaw), encoded with the
box's ffmpeg, decoded back through hearsay.audio.load_audio (16 kHz mono), re-trimmed with the
same trim_silence, and stored as <bundle>/codecs/<id>.<codec>.flac. The subset is chosen from
the manifest without looking at the label (class-blind by construction).

Writes <bundle>/codecs/codec_manifest.csv (id, codec, file, duration, lag_ms, lead_s).
Usage: python scripts/cloud/box_codecs.py --bundle /root/m5/bundle --frac 0.3 --workers 30
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from hearsay import SR
from hearsay.audio import DecodeError, load_audio
from hearsay.m5_bundle import leading_silence_s, prepare_clip, read_clip

CODECS = {
    "mp3": (["-c:a", "libmp3lame", "-b:a", "48k"], ".mp3"),
    "opus": (["-c:a", "libopus", "-b:a", "20k"], ".ogg"),
    "aac": (["-c:a", "aac", "-b:a", "40k"], ".m4a"),
    "amrnb": (["-ar", "8000", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"], ".amr"),
    "mulaw": (["-ar", "8000", "-c:a", "pcm_mulaw"], ".wav"),
}


def available_codecs() -> list[str]:
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                         check=False).stdout
    out = []
    for name, (args, _) in CODECS.items():
        codec = args[args.index("-c:a") + 1]
        if codec in enc:
            out.append(name)
    return out


def roundtrip(src: str, codec: str, dst: str) -> dict:
    args, ext = CODECS[codec]
    with tempfile.TemporaryDirectory() as td:
        enc = Path(td) / f"x{ext}"
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, *args, str(enc)],
                           capture_output=True, check=False)
        if r.returncode != 0:
            return {"error": r.stderr.decode(errors="replace")[-200:]}
        try:
            y = load_audio(enc)
        except DecodeError as e:
            return {"error": str(e)[:200]}
    x = read_clip(src)
    y = prepare_clip(y, max_s=15.0)
    # lag of the decoded copy vs the source (codec delay that survived the re-trim)
    n = min(x.size, y.size, 4 * SR)
    if n > SR:
        c = np.correlate(y[:n], x[: n // 2], mode="valid")
        lag_ms = (int(np.argmax(c)) - 0) / SR * 1000.0
    else:
        lag_ms = float("nan")
    sf.write(dst, y, SR, subtype="PCM_16", format="FLAC")
    return {"duration": round(y.size / SR, 4), "src_duration": round(x.size / SR, 4),
            "lag_ms": round(lag_ms, 1), "lead_s": round(leading_silence_s(y), 4)}  # fmt: skip


def _job(a):
    id_, src, codec, dst = a
    try:
        r = roundtrip(src, codec, dst)
    except (OSError, RuntimeError, ValueError) as e:
        r = {"error": f"{type(e).__name__}: {e}"[:200]}
    return {"id": id_, "codec": codec, "file": Path(dst).name, **r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--codecs", default=None, help="comma list; default: all available")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    codecs = args.codecs.split(",") if args.codecs else available_codecs()
    missing = sorted(set(CODECS) - set(codecs))
    print(f"codecs: {codecs}; missing encoders: {missing}")
    m = pd.read_csv(args.bundle / "manifest.csv")
    rng = np.random.default_rng(args.seed)
    is_hold = m.fold.astype(str) == "holdout"
    pick = is_hold | (rng.random(len(m)) < args.frac)  # label never consulted
    sel = m[pick]
    if args.limit:
        sel = sel.iloc[: args.limit]
    out = args.bundle / "codecs"
    out.mkdir(exist_ok=True)
    jobs = []
    for r in sel.itertuples():
        codec = codecs[int(rng.integers(len(codecs)))]
        src = args.bundle / ("core" if r.train_scope == "core" else "extra") / r.file
        jobs.append((r.id, str(src), codec, str(out / f"{r.id}.{codec}.flac")))
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=8)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(jobs)} ({(i + 1) / (time.time() - t0):.0f}/s)", flush=True)
    df = pd.DataFrame(rows)
    errs = df[df.get("error", pd.Series(dtype=object)).notna()] if "error" in df else df.iloc[:0]
    ok = df[~df.index.isin(errs.index)]
    ok.to_csv(out / "codec_manifest.csv", index=False)
    errs.to_csv(out / "codec_errors.csv", index=False)
    print(f"{len(ok)} variants ({len(errs)} errors) in {time.time() - t0:.0f}s; "
          f"per codec: {ok.codec.value_counts().to_dict()}; "
          f"lag p95 {ok.lag_ms.quantile(.95):.1f} ms; lead p95 {ok.lead_s.quantile(.95):.3f} s")


if __name__ == "__main__":
    main()
