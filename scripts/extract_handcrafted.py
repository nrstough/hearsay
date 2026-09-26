"""Handcrafted forensic features for a manifest (CPU, parallel) -> CSV.

Usage:
  training (v2: random crop to NSA test-duration lengths, one draw per manifest row):
    uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_train_sample.csv \
        --name nsa_train_sample_v2 --crop test --seed 0 --workers 6
  test (whole clip, no crop, 8 s cap):
    uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_test.csv \
        --name nsa_test_v2 --workers 6

Crop lengths are drawn exactly as scripts/extract_embeddings.py --segment --crop test --seed S
draws them: hearsay.embed.test_duration_sampler(S), one draw per manifest row in order, and the
per-row crop-offset seed is S + row. So the handcrafted and deep detectors see the same audio.
--crop-mode first4s reproduces v1 (first 4 s after the silence trim).

Output: outputs/handcrafted/<name>.csv (manifest columns + hc_crop_s + one column per feature;
failures flagged in `hc_flag`) and outputs/handcrafted/<name>.meta.json.
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

from hearsay.handcrafted import CROP_MODES, draw_augment, features_for_path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--crop", choices=["none", "test"], default="none",
                    help="test: random crop to NSA test-duration lengths (training manifests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop-mode", choices=CROP_MODES, default="segment",
                    help="segment = hearsay.embed.prepare_segment (v2); first4s = v1")
    ap.add_argument("--band-match", action=argparse.BooleanOptionalAction, default=True,
                    help="apply hearsay.handcrafted.band_limit (NSA test-set roll-off; v3)")
    ap.add_argument("--families", default="",
                    help="comma-separated hearsay.hc_v4 families to append (v4), e.g. lfcc,phase")
    ap.add_argument("--launder-frac", type=float, default=0.0,
                    help="training only: share of rows re-encoded through MP3/AAC first (v5)")
    ap.add_argument("--tilt-frac", type=float, default=0.0,
                    help="training only: share of rows given a random spectral tilt + low-pass (v5)")
    args = ap.parse_args()
    families = tuple(n for n in args.families.split(",") if n)

    m = pd.read_csv(args.manifest)
    if args.crop == "test":
        from hearsay.embed import test_duration_sampler

        draw = test_duration_sampler(args.seed)
        crop_s = [draw() for _ in range(len(m))]
    else:
        crop_s = [None] * len(m)
    seeds = [args.seed + r for r in range(len(m))]
    modes = [args.crop_mode] * len(m)
    bands = [args.band_match] * len(m)
    fams = [families] * len(m)
    augs = [draw_augment(np.random.default_rng(args.seed + r), args.launder_frac, args.tilt_frac)
            if (args.launder_frac or args.tilt_frac) else "" for r in range(len(m))]  # fmt: skip
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(features_for_path, m.path, crop_s, seeds, modes, bands, fams, augs,
                           chunksize=32))  # fmt: skip
    feats = pd.DataFrame([r or {} for r in rows])
    out = pd.concat([m.reset_index(drop=True), feats], axis=1)
    out.insert(len(m.columns), "hc_crop_s", [float("nan") if c is None else c for c in crop_s])
    out.insert(len(m.columns) + 1, "hc_augment", augs)
    out["hc_flag"] = ["" if r else "feature_error" for r in rows]
    dest = REPO / "outputs" / "handcrafted" / f"{args.name}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    meta = {"manifest": str(args.manifest), "crop_mode": args.crop_mode, "crop": args.crop,
            "band_match": args.band_match, "families": list(families), "seed": args.seed,
            "launder_frac": args.launder_frac, "tilt_frac": args.tilt_frac,
            "n_augmented": int(sum(1 for a in augs if a)), "workers": args.workers,
            "rows": len(out),
            "n_features": int(feats.shape[1]), "failures": int((out.hc_flag != "").sum()),
            "seconds": round(time.time() - t0)}  # fmt: skip
    dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {dest}: {len(out)} rows, {feats.shape[1]} features, "
          f"{meta['failures']} failures, {meta['seconds']}s")


if __name__ == "__main__":
    main()
