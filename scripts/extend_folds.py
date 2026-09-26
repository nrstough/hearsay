"""Extend the shared fold file with extra training-only rows (never the holdout).

Extra rows are assigned to inner folds 0..k-1 with StratifiedGroupKFold, grouped by speaker
(bona fide) or generator (spoof), prefixed by source so groups never collide with NSA groups.
The shared splits/nsa_folds.csv is read, never modified; the outer holdout is unchanged, so
holdout readouts stay comparable across data mixes.

Usage: uv run python scripts/extend_folds.py --extra outputs/manifests/asv19_addon.csv \
    --out splits/nsa_folds_plus_asv19.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=REPO / "splits" / "nsa_folds.csv")
    ap.add_argument("--extra", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    x = pd.read_csv(args.extra)
    key = x.generator.where(x.label == "spoof", x.speaker)
    x["group"] = x.source + ":" + key.astype(str)
    y = (x.label == "spoof").astype(int)
    x["fold"] = ""
    for i, (_, te) in enumerate(StratifiedGroupKFold(args.k, shuffle=True, random_state=args.seed)
                                .split(x, y, x.group)):  # fmt: skip
        x.loc[x.index[te], "fold"] = str(i)
    assert not set(x.path) & set(base.path), "extra rows overlap the base fold file"
    out = pd.concat([base, x[base.columns.intersection(x.columns)]], ignore_index=True)
    assert (out.groupby("group").fold.nunique() == 1).all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby(["fold", "label"]).size().unstack(fill_value=0).to_string())
    print(f"wrote {args.out}: base {len(base)} + extra {len(x)} (holdout unchanged: "
          f"{(out.fold == 'holdout').sum()} rows)")


if __name__ == "__main__":
    main()
