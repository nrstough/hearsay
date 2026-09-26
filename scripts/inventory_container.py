"""Container / metadata inventory (technique 1): what ffprobe sees in every file, per class.

Writes outputs/inventory/container_<name>.csv, one row per file: format, codec, sample rate,
channels, bits per sample, bit rate, duration, byte size, number of tags, tag keys, and the
encoder/software/comment tag values when present. Embedded fields only: no filesystem
timestamps and no filenames (git, Docker and unzip rewrite those).

Then prints how uniform the set is and, when labels exist, the per-field class split. On the
training data that split *is* the container shortcut (LibriSpeech = FLAC, LJ = 22 kHz WAV,
DiffSSD = 22 kHz WAV or MP3); on the NSA test set every file is the same PCM16 16 kHz WAV.

Usage: uv run python scripts/inventory_container.py --manifest outputs/manifests/nsa_test.csv \
    --name nsa_test [--per-generator 300] [--workers 4]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from hearsay.detectors.container import probe_container

REPO = Path(__file__).resolve().parents[1]


def probe(path: str) -> dict:
    row = probe_container(REPO / path if not Path(path).is_absolute() else path)
    row.pop("tags", None)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--per-generator", type=int, help="keep the first N rows per generator")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    if args.per_generator and "generator" in m:
        m = m.groupby("generator", sort=False).head(args.per_generator).reset_index(drop=True)
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(probe, m.path))
    out = pd.concat([m, pd.DataFrame(rows)], axis=1)
    dest = REPO / "outputs" / "inventory" / f"container_{args.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)

    fields = ["format", "codec", "sample_rate", "channels", "bits", "bit_rate", "n_tags",
              "tag_keys", "tag_encoder", "tag_software"]  # fmt: skip
    print(f"{args.name}: {len(out)} files, probe failures {(~out.probe_ok).sum()}")
    print("distinct values per field:")
    for c in fields:
        vals = out[c].fillna("<none>").astype(str).value_counts()
        head = ", ".join(f"{k}×{v}" for k, v in vals.head(6).items())
        print(f"  {c:12s} {len(vals):4d} distinct: {head}")
    if "label" in out:
        print("\nper-field split by label/source (the container shortcut):")
        key = out.source.astype(str) + "/" + out.label.astype(str)
        for c in ("format", "codec", "sample_rate", "bits"):
            print(pd.crosstab(key, out[c].fillna("<none>").astype(str)).to_string(), "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
