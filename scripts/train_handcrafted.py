"""Handcrafted-feature detector (spectral + prosody) on the fold file.

1. Per-feature AUC on the inner (training) rows: which cues separate real from synthetic.
2. Logistic regression (standardized, class-balanced) and LightGBM, each scored by
   out-of-fold minDCF on the predefined inner folds; the better one is refit on all inner rows.
3. One readout on the outer holdout (held-out generators + bona fide groups), all metric
   readings, per generator and per bona fide source.

Usage: uv run python scripts/train_handcrafted.py --train nsa_train_sample \
    --folds splits/nsa_folds.csv
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hearsay.metrics import PI_SYNTH, eer, min_cost, report

REPO = Path(__file__).resolve().parents[1]
META_COLS = {"path", "label", "generator", "speaker", "utt", "source", "hc_flag", "filename",
             "group", "fold"}  # fmt: skip


def models() -> dict:
    return {
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
        ),
        "lgbm": lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
            verbose=-1, random_state=0,
        ),
    }


def decision(model, X):
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--pi-synth", type=float, default=PI_SYNTH)
    args = ap.parse_args()

    t0 = time.time()
    d = pd.read_csv(REPO / "outputs" / "handcrafted" / f"{args.train}.csv")
    f = pd.read_csv(args.folds).set_index("path")[["group", "fold"]]
    d = d.join(f, on="path")
    assert d.fold.notna().all(), "rows missing from the fold file"
    d = d[d.hc_flag.fillna("") == ""].reset_index(drop=True)
    feat_cols = [c for c in d.columns if c not in META_COLS]
    X = d[feat_cols].to_numpy(np.float32)
    y = (d.label == "spoof").to_numpy(int)
    tr, va = (d.fold != "holdout").to_numpy(), (d.fold == "holdout").to_numpy()
    folds = d.loc[tr, "fold"].astype(int).to_numpy()

    auc = {c: round(float(roc_auc_score(y[tr], d.loc[tr, c])), 4) for c in feat_cols}
    top = sorted(auc.items(), key=lambda kv: -abs(kv[1] - 0.5))[:15]
    print("top single-feature AUCs (inner rows; >0.5 = higher on synthetic):")
    for k, v in top:
        print(f"  {k:22s} {v:.3f}")

    cv = {}
    for name, mk in models().items():
        oof = np.zeros(tr.sum())
        for a, b in PredefinedSplit(folds).split():
            oof[b] = decision(mk().fit(X[tr][a], y[tr][a]), X[tr][b])
        cv[name] = {"min_dcf": round(min_cost(y[tr], oof, args.pi_synth), 4),
                    "eer": round(eer(y[tr], oof), 4)}  # fmt: skip
        print(f"CV {name}: {cv[name]}", flush=True)
    best = min(cv, key=lambda k: (cv[k]["min_dcf"], cv[k]["eer"]))
    model = models()[best]().fit(X[tr], y[tr])

    s = decision(model, X[va])
    yv, dv = y[va], d[va].reset_index(drop=True)
    val = {f"val_{k}": v for k, v in report(yv, s, args.pi_synth).items()}
    val["val_auc"] = round(float(roc_auc_score(yv, s)), 4)
    bona = s[yv == 0]
    per_gen = {g: {"min_dcf": round(min_cost(np.r_[np.zeros(len(bona)), np.ones(m.sum())],
                                             np.r_[bona, s[m]], args.pi_synth), 4)}
               for g in sorted(set(dv.generator) - {"bonafide"})
               for m in [(dv.generator == g).to_numpy()]}  # fmt: skip
    per_src = {src: {"min_dcf": round(min_cost(np.r_[np.zeros(m.sum()), np.ones((yv == 1).sum())],
                                                np.r_[s[m], s[yv == 1]], args.pi_synth), 4)}
               for src in sorted(set(dv.loc[yv == 0, "source"]))
               for m in [((dv.source == src) & (yv == 0)).to_numpy()]}  # fmt: skip

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    out = REPO / "models" / f"hc_{best}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": feat_cols, "kind": best}, out / "model.joblib")
    meta = {"rung": "D-track (handcrafted spectral+prosody)", "train": args.train,
            "folds": str(args.folds), "model": best, "cv": cv, "n_features": len(feat_cols),
            **val, "val_by_generator": per_gen, "val_by_bonafide_source": per_src,
            "feature_auc_top15": dict(top), "seconds": round(time.time() - t0)}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "feature_auc_top15"}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
