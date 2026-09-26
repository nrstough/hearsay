"""Fold file for NSA training (docs/plan.md "Data"): nested, grouped, class-checked.

Groups: spoof -> generator (whole generators held out); bona fide -> speaker, except the
single-speaker LJ corpus, which is grouped by source chapter (LJ id prefix, e.g. "LJ025").
Outer holdout: the --holdout-generators (default one LJ-voice model and one commercial
cloning service) plus ~--holdout-frac of the bona fide groups. The rest gets --k inner folds
(StratifiedGroupKFold). Every fold must hold >= --min-per-class clips of each class.

Limits stated in the output (plan X1): DiffSSD's multi-speaker generators share 10 cloned
target speakers, so a held-out generator still shares *voices* (not synthesis) with training;
LJ-voice generators share the LJ voice with LJ bona fide. Both are reported, not hidden.

Usage: uv run python scripts/make_folds.py --manifest outputs/manifests/nsa_train_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[1]


def group_key(row: pd.Series) -> str:
    if row.label == "spoof":
        return f"gen:{row.generator}"
    if row.source == "ljspeech":
        return f"lj:{str(row.utt)[:5]}"
    return f"spk:{row.speaker}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--holdout-generators", default="wavegrad2,playht")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-per-class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "splits" / "nsa_folds.csv")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    m["group"] = m.apply(group_key, axis=1)
    m["fold"] = ""
    held_gens = set(args.holdout_generators.split(","))
    assert held_gens <= set(m.generator), held_gens - set(m.generator)

    rng = np.random.default_rng(args.seed)
    bona_groups = sorted(m.loc[m.label == "bonafide", "group"].unique())
    n_hold = max(1, round(args.holdout_frac * len(bona_groups)))
    held_bona = set(rng.choice(bona_groups, size=n_hold, replace=False))
    hold = m.generator.isin(held_gens) | m.group.isin(held_bona)
    m.loc[hold, "fold"] = "holdout"

    inner = m[~hold]
    y = (inner.label == "spoof").to_numpy(int)
    sgkf = StratifiedGroupKFold(args.k, shuffle=True, random_state=args.seed)
    for i, (_, te) in enumerate(sgkf.split(inner, y, inner.group)):
        m.loc[inner.index[te], "fold"] = str(i)

    counts = m.groupby(["fold", "label"]).size().unstack(fill_value=0)
    print(counts.to_string())
    low = counts[(counts < args.min_per_class).any(axis=1)]
    assert low.empty, f"folds below {args.min_per_class} per class:\n{low}"
    # Groups never straddle folds.
    assert (m.groupby("group").fold.nunique() == 1).all()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    m[["path", "label", "generator", "speaker", "source", "group", "fold"]].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {m.group.nunique()} groups; holdout generators {sorted(held_gens)}, "
          f"{len(held_bona)} bona fide groups held out")
    print("limits: multi-speaker generators share 10 cloned target voices across folds; "
          "LJ-voice generators share the LJ voice with LJ bona fide (grouped by chapter).")


if __name__ == "__main__":
    main()
