"""Handcrafted forensic features for a manifest (CPU, parallel) -> parquet-free CSV.

Usage: uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_train_sample.csv \
    --name nsa_train_sample [--workers 6]
Output: outputs/handcrafted/<name>.csv (manifest columns + one column per feature; failures
flagged in `hc_flag`).
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from hearsay.handcrafted import features_for_path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(features_for_path, m.path, chunksize=32))
    feats = pd.DataFrame([r or {} for r in rows])
    out = pd.concat([m.reset_index(drop=True), feats], axis=1)
    out["hc_flag"] = ["" if r else "feature_error" for r in rows]
    dest = REPO / "outputs" / "handcrafted" / f"{args.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}: {len(out)} rows, {feats.shape[1]} features, "
          f"{(out.hc_flag != '').sum()} failures, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
