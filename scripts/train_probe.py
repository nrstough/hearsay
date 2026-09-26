"""M1: frozen embeddings -> per-layer logistic probe.

1. For every layer, generator-grouped 5-fold CV on the TRAIN embeddings: normalized
   minDCF (NSA C_FA = 4, hearsay.metrics) and EER. Layer = lowest CV minDCF.
2. Platt map (class-balanced -> prior-neutral LLR) fit on that layer's out-of-fold scores.
3. Refit on all of train; score the held-out VAL embeddings once; report minDCF + EER.

Usage:
  uv run python scripts/train_probe.py --train asv19_train_probe --val asv19_eval_probe \
      [--pi-synth 0.5]
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

from hearsay.metrics import eer, min_cost, report
from hearsay.probe import Probe, cv_groups, load_embeddings, make_clf, oof_scores

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wav2vec2-xls-r-300m")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--pi-synth", type=float, default=0.5, help="prior used in the cost metric")
    args = ap.parse_args()

    t0 = time.time()
    Xtr, mtr = load_embeddings(args.model, args.train)
    Xva, mva = load_embeddings(args.model, args.val)
    ytr, yva = (mtr.label == "spoof").to_numpy(int), (mva.label == "spoof").to_numpy(int)
    groups = cv_groups(mtr)
    print(f"train {Xtr.shape} spoof={ytr.mean():.2f}  val {Xva.shape} spoof={yva.mean():.2f}")

    cv = {}
    for layer in range(Xtr.shape[1]):
        oof = oof_scores(Xtr[:, layer], ytr, groups, args.c)
        cv[layer] = {"min_dcf": round(min_cost(ytr, oof, args.pi_synth), 4),
                     "eer": round(eer(ytr, oof), 4)}  # fmt: skip
        print(f"  layer {layer:2d}  CV {cv[layer]}", flush=True)
    best = min(cv, key=lambda k: (cv[k]["min_dcf"], cv[k]["eer"]))

    oof = oof_scores(Xtr[:, best], ytr, groups, args.c)
    platt = LogisticRegression(class_weight="balanced").fit(oof[:, None], ytr)
    clf = make_clf(args.c).fit(Xtr[:, best], ytr)
    probe = Probe(args.model, best, clf, float(platt.coef_[0, 0]), float(platt.intercept_[0]))

    llr = probe.llr(Xva)
    auc = float(roc_auc_score(yva, llr))
    assert auc > 0.5, f"val AUC {auc:.3f}: score does not increase with synthetic likelihood"
    val = {f"val_{k}": v for k, v in report(yva, llr, args.pi_synth).items()}
    val["val_auc"] = round(auc, 4)
    bona = llr[yva == 0]
    per_gen = {}
    for g in sorted(set(mva.generator) - {"bonafide"}):
        sp = llr[(mva.generator == g).to_numpy()]
        yy = np.r_[np.zeros(len(bona)), np.ones(len(sp))]
        per_gen[g] = {"eer": round(eer(yy, np.r_[bona, sp]), 4),
                      "min_dcf": round(min_cost(yy, np.r_[bona, sp], args.pi_synth), 4)}  # fmt: skip

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    out = REPO / "models" / f"m1_{args.model}_L{best}_{stamp}"
    meta = {
        "rung": "M1", "backbone": args.model, "train": args.train, "val": args.val,
        "layer": best, "C": args.c, "selection": "CV normalized minDCF (C_FA=4)",
        "cv_by_layer": cv, "cv_best": cv[best], "platt": [probe.platt_a, probe.platt_b],
        **val, "val_by_generator": per_gen, "seconds": round(time.time() - t0),
    }  # fmt: skip
    probe.save(out, meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "cv_by_layer"}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
