"""Export an M1-family probe's scores in the fusion format (CPU only).

outputs/detector_scores/<name>.csv with columns path, fold, split, score, logit:
- inner_oof: out-of-fold LLR on inner rows, predefined folds from the fold file used to train
  the probe (same layer and C; each fold refit from scratch, Platt from the saved probe),
- holdout: the saved probe (fit on all inner rows) on outer-holdout rows,
- test: the saved probe on the test embedding set,
- itw: the saved probe on an eval-only stress set (optional).
Only rows present in --keep-folds (default: the shared NSA fold file) are exported, so every
detector's export has the same NSA rows; extra training rows (e.g. ASV19) stay internal.
score = sigmoid(logit), logit = prior-neutral LLR.

Usage:
  uv run python scripts/export_probe_scores.py --probe models/m1_...-0521 --name m1b_v3 \
      --train nsa_train_sample_v3,asv19_addon_v3 --folds splits/nsa_folds_plus_asv19.csv \
      --test nsa_test_v3 --stress itw_stress_v3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.metrics import sigmoid
from hearsay.probe import Probe, load_embeddings, oof_scores

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--keep-folds", type=Path, default=REPO / "splits" / "nsa_folds.csv")
    ap.add_argument("--test", required=True)
    ap.add_argument("--stress")
    ap.add_argument("--c", type=float, default=1.0)
    args = ap.parse_args()

    probe = Probe.load(args.probe)
    parts = [load_embeddings(probe.backbone, n) for n in args.train.split(",")]
    X = np.concatenate([p[0] for p in parts])[:, probe.layer]
    m = pd.concat([p[1] for p in parts], ignore_index=True)
    m = m.join(pd.read_csv(args.folds).set_index("path")[["fold"]], on="path")
    assert m.fold.notna().all()
    y = (m.label == "spoof").to_numpy(int)
    inner = (m.fold != "holdout").to_numpy()

    llr = np.zeros(len(m))
    z = oof_scores(X[inner], y[inner], None, args.c, folds=m.fold[inner].astype(int).to_numpy())
    llr[inner] = probe.platt_a * z + probe.platt_b
    llr[~inner] = probe.llr(np.concatenate([p[0] for p in parts])[~inner])
    rows = pd.DataFrame({"path": m.path, "fold": m.fold,
                         "split": np.where(inner, "inner_oof", "holdout"), "logit": llr})  # fmt: skip
    keep = set(pd.read_csv(args.keep_folds).path)
    rows = rows[rows.path.isin(keep)]

    Xt, mt = load_embeddings(probe.backbone, args.test)
    out = [rows, pd.DataFrame({"path": mt.path, "fold": "", "split": "test",
                               "logit": probe.llr(Xt)})]  # fmt: skip
    if args.stress:
        Xs, ms = load_embeddings(probe.backbone, args.stress)
        out.append(pd.DataFrame({"path": ms.path, "fold": "", "split": "itw",
                                 "logit": probe.llr(Xs)}))  # fmt: skip
    df = pd.concat(out, ignore_index=True)
    df["score"] = sigmoid(df.logit.to_numpy())
    assert df.logit.notna().all() and not df.path.duplicated().any()
    dest = REPO / "outputs" / "detector_scores" / f"{args.name}.csv"
    df[["path", "fold", "split", "score", "logit"]].to_csv(dest, index=False)
    print(f"wrote {dest}: {df.split.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
