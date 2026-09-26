"""Hash index for leakage dedupe (plan.md: hash-dedupe NSA train/test against public sets).

Two hashes per file:
  file_sha256  exact byte match
  pcm_sha256   sha256 of the 16 kHz mono float32 PCM from hearsay.audio.load_audio, so the same
               audio re-wrapped in another container/sample rate still matches when the decode is
               bit-identical (it will not catch lossy re-encodes; that needs embedding similarity)

Usage:
  uv run python scripts/hash_audio.py --root data/in_the_wild/release_in_the_wild --name itw
  uv run python scripts/hash_audio.py --root data/nsa --name nsa
  uv run python scripts/hash_audio.py --compare outputs/hashes/nsa.csv outputs/hashes/itw.csv
"""

from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from hearsay.audio import DecodeError, load_audio

REPO = Path(__file__).resolve().parents[1]
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm", ".amr", ".mp4"}


def hash_one(p: str) -> tuple[str, str, str, int]:
    fh = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    try:
        x = load_audio(p)
        return p, fh, hashlib.sha256(x.tobytes()).hexdigest(), int(x.size)
    except DecodeError:
        return p, fh, "", 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--compare", nargs="+", type=Path, help="first CSV vs each of the others")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    if args.compare:
        a = pd.read_csv(args.compare[0])
        for other in args.compare[1:]:
            b = pd.read_csv(other)
            for col in ("file_sha256", "pcm_sha256"):
                hits = a[a[col].isin(set(b[col].dropna()) - {""})]
                print(f"{args.compare[0].name} vs {other.name} [{col}]: {len(hits)} matches")
                if len(hits):
                    print(hits.head(20).to_string(index=False))
        return

    files = sorted(str(p) for p in args.root.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    print(f"hashing {len(files)} files under {args.root} with {args.workers} workers")
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(hash_one, files, chunksize=64))
    df = pd.DataFrame(rows, columns=["path", "file_sha256", "pcm_sha256", "n_samples"])
    df["path"] = [str(Path(p).relative_to(REPO)) if p.startswith(str(REPO)) else p for p in df.path]
    out = REPO / "outputs" / "hashes" / f"{args.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    dup = df[df.duplicated("pcm_sha256", keep=False) & (df.pcm_sha256 != "")]
    print(f"wrote {out}: {len(df)} rows, {int((df.pcm_sha256 == '').sum())} decode failures, "
          f"{len(dup)} rows in within-set PCM duplicate groups")


if __name__ == "__main__":
    main()
