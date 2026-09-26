"""Build the M5 training manifest: fold file (+ the main chat's ASV19 extension) plus
training-only extra rows from the rest of the NSA manifest and MLAAD.

Writes outputs/manifests/m5_manifest.csv and prints per-fold / per-scope class counts.
The shared fold files are read-only and sha-pinned (hearsay.m5_data.check_folds_file).

Usage: uv run python scripts/m5_build_manifest.py [--no-mlaad] [--extra-spoof-cap 1000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hearsay.m5_data import (
    FOLDS,
    FOLDS_PLUS,
    FOLDS_PLUS_SHA256,
    FOLDS_SHA256,
    build_manifest,
    check_folds_file,
    training_rows,
    validation_rows,
)  # fmt: skip

REPO = Path(__file__).resolve().parents[1]
MAN = REPO / "outputs" / "manifests"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=Path, default=FOLDS_PLUS)
    ap.add_argument("--full", type=Path, default=MAN / "nsa_train_full.csv")
    ap.add_argument("--mlaad", type=Path, nargs="*",
                    default=[MAN / "mlaad_en.csv", MAN / "mlaad_en_named.csv"])
    ap.add_argument("--no-mlaad", action="store_true")
    ap.add_argument("--extra-spoof-cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=MAN / "m5_manifest.csv")
    args = ap.parse_args()

    check_folds_file(FOLDS, FOLDS_SHA256)
    if args.folds == FOLDS_PLUS:
        check_folds_file(FOLDS_PLUS, FOLDS_PLUS_SHA256)
    folds = pd.read_csv(args.folds)
    if args.folds != FOLDS:  # the extended file must agree with the shared fold file on every core row
        base = pd.read_csv(FOLDS).set_index("path")
        core = folds[folds.path.isin(base.index)].set_index("path")
        assert len(core) == len(base), (len(core), len(base))
        bad = (core.fold.astype(str) != base.fold.astype(str).reindex(core.index)).sum()
        assert bad == 0, f"{bad} core rows have a different fold in {args.folds}"
    full = pd.read_csv(args.full)
    mlaad = None
    if not args.no_mlaad:
        mlaad = pd.concat([pd.read_csv(p) for p in args.mlaad], ignore_index=True)
    m = build_manifest(folds, full, mlaad, extra_spoof_cap=args.extra_spoof_cap, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)

    print(m.groupby(["train_scope", "label"]).size().unstack(fill_value=0).to_string())
    print()
    print(m[m.train_scope == "core"].groupby(["fold", "label"]).size().unstack(fill_value=0)
          .to_string())
    print()
    for k in ["0", "1", "2", "3", "4", None]:
        tr = training_rows(m, k)
        line = f"model fold={k or 'full'}: train {len(tr)} " \
               f"(spoof {int((tr.label == 'spoof').sum())}, bona {int((tr.label == 'bonafide').sum())})"
        if k is not None:
            va = validation_rows(m, k)
            line += f"; val {len(va)} ({sorted(set(va.generator) - {'bonafide'})})"
        print(line)
    print(f"\nwrote {args.out}: {len(m)} rows")


if __name__ == "__main__":
    main()
