"""Compression-forensics features for a manifest (CPU, parallel) -> CSV.

Usage:
  training (v2 crops as for the handcrafted features; a random half of BOTH classes is
  laundered through MP3/AAC at a random bitrate and sample rate, then decoded back to 16 kHz):
    uv run python scripts/extract_compression.py --manifest outputs/manifests/nsa_train_sample.csv \
        --name nsa_train_sample --crop test --seed 0 --launder-frac 0.5 --workers 6
  test (whole clip, never laundered):
    uv run python scripts/extract_compression.py --manifest outputs/manifests/nsa_test.csv \
        --name nsa_test --workers 6

Laundering is drawn per row from numpy.default_rng(seed + row) so a re-run reproduces it, and
the draw is recorded in the `cmp_launder` column ("" = clean). Output:
outputs/compression/<name>.csv (manifest columns + hc_crop_s + cmp_launder + features;
failures flagged in `cmp_flag`) and outputs/compression/<name>.meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.compression import draw_laundering, features_for_path
from hearsay.handcrafted import CROP_MODES

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--crop", choices=["none", "test"], default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop-mode", choices=CROP_MODES, default="segment")
    ap.add_argument("--band-match", action=argparse.BooleanOptionalAction, default=True,
                    help="apply hearsay.handcrafted.band_limit (NSA test-set roll-off; v3)")
    ap.add_argument("--launder-frac", type=float, default=0.0,
                    help="fraction of rows re-encoded through a lossy codec (training only)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    if args.limit:
        m = m.iloc[: args.limit].reset_index(drop=True)
    if args.crop == "test":
        from hearsay.embed import test_duration_sampler

        draw = test_duration_sampler(args.seed)
        crop_s = [draw() for _ in range(len(m))]
    else:
        crop_s = [None] * len(m)
    seeds = [args.seed + r for r in range(len(m))]
    modes = [args.crop_mode] * len(m)
    bands = [args.band_match] * len(m)
    launder = [draw_laundering(np.random.default_rng(args.seed + r), args.launder_frac)
               for r in range(len(m))]  # fmt: skip

    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(features_for_path, m.path, crop_s, seeds, modes, launder, bands,
                           chunksize=32))  # fmt: skip
    feats = pd.DataFrame([r or {} for r in rows])
    out = pd.concat([m, feats], axis=1)
    out.insert(len(m.columns), "hc_crop_s", [float("nan") if c is None else c for c in crop_s])
    out.insert(len(m.columns) + 1, "cmp_launder", launder)
    out["cmp_flag"] = ["" if r else "feature_error" for r in rows]
    dest = REPO / "outputs" / "compression" / f"{args.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    meta = {"manifest": str(args.manifest), "crop_mode": args.crop_mode, "crop": args.crop,
            "band_match": args.band_match, "seed": args.seed,
            "launder_frac": args.launder_frac, "workers": args.workers,
            "rows": len(out), "n_laundered": int((out.cmp_launder != "").sum()),
            "n_features": int(feats.shape[1]), "failures": int((out.cmp_flag != "").sum()),
            "seconds": round(time.time() - t0)}  # fmt: skip
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {dest}: {len(out)} rows, {feats.shape[1]} features, "
          f"{meta['n_laundered']} laundered, {meta['failures']} failures, {meta['seconds']}s")


if __name__ == "__main__":
    main()
