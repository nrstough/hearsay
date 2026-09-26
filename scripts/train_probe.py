"""M1: frozen embeddings -> per-layer logistic probe.

1. For every layer, generator-grouped 5-fold CV EER on the TRAIN embeddings.
2. Best layer (lowest CV EER) -> Platt map fit on its out-of-fold scores (still train only).
3. Refit on all of train; score the held-out VAL embeddings once; report AUC/EER.

Usage:
  uv run python scripts/train_probe.py --train asv19_train_probe --val asv19_eval_probe
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hearsay.probe import Probe, cv_groups, eer, load_embeddings, make_clf, oof_scores

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wav2vec2-xls-r-300m")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--c", type=float, default=1.0)
    args = ap.parse_args()

    t0 = time.time()
    Xtr, mtr = load_embeddings(args.model, args.train)
    Xva, mva = load_embeddings(args.model, args.val)
    ytr, yva = (mtr.label == "spoof").to_numpy(int), (mva.label == "spoof").to_numpy(int)
    groups = cv_groups(mtr)
    print(f"train {Xtr.shape} spoof={ytr.mean():.2f}  val {Xva.shape} spoof={yva.mean():.2f}")

    cv = {}
    for layer in range(Xtr.shape[1]):
        cv[layer] = eer(ytr, oof_scores(Xtr[:, layer], ytr, groups, args.c))
        print(f"  layer {layer:2d}  CV EER {cv[layer]:.4f}", flush=True)
    best = min(cv, key=cv.get)

    oof = oof_scores(Xtr[:, best], ytr, groups, args.c)
    platt = LogisticRegression(class_weight="balanced").fit(oof[:, None], ytr)
    clf = make_clf(args.c).fit(Xtr[:, best], ytr)
    probe = Probe(args.model, best, clf, float(platt.coef_[0, 0]), float(platt.intercept_[0]))

    p = probe.p_synthetic(Xva)
    val = {
        "val_auc": round(float(roc_auc_score(yva, p)), 4),
        "val_eer": round(eer(yva, p), 4),
        "val_mean_p_spoof": round(float(p[yva == 1].mean()), 4),
        "val_mean_p_bonafide": round(float(p[yva == 0].mean()), 4),
    }
    per_gen = {
        g: round(eer(np.r_[np.zeros((yva == 0).sum()), np.ones(len(i))],
                     np.r_[p[yva == 0], p[i]]), 4)
        for g in sorted(set(mva.generator) - {"bonafide"})
        for i in [np.where(mva.generator.to_numpy() == g)[0]]
    }  # fmt: skip
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    out = REPO / "models" / f"m1_{args.model}_L{best}_{stamp}"
    meta = {
        "rung": "M1", "backbone": args.model, "train": args.train, "val": args.val,
        "layer": best, "C": args.c, "cv_eer_by_layer": {k: round(v, 4) for k, v in cv.items()},
        "cv_eer_best": round(cv[best], 4), "platt": [probe.platt_a, probe.platt_b],
        **val, "val_eer_by_generator": per_gen, "seconds": round(time.time() - t0),
    }  # fmt: skip
    probe.save(out, meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "cv_eer_by_layer"}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
