"""M1: frozen embeddings -> per-layer logistic probe.

1. For every layer, generator-grouped 5-fold CV on the TRAIN embeddings: normalized
   minDCF (NSA C_FA = 4, hearsay.metrics) and EER. Layer = lowest CV minDCF.
2. Platt map (class-balanced -> prior-neutral LLR) fit on that layer's out-of-fold scores.
3. Refit on all of train; score the held-out VAL embeddings once; report minDCF + EER.

Usage:
  uv run python scripts/train_probe.py --train asv19_train_probe --val asv19_eval_probe \
      [--pi-synth 0.3]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hearsay.metrics import cost_at, eer, min_cost, report
from hearsay.probe import Probe, cv_groups, load_embeddings, make_clf, oof_scores

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wav2vec2-xls-r-300m")
    ap.add_argument("--train", required=True, help="embedding set name")
    ap.add_argument("--val", help="separate embedding set name (public-data mode)")
    ap.add_argument("--folds", type=Path, help="fold file: holdout = val, inner folds = CV")
    ap.add_argument("--max-train", type=int, help="subsample inner rows (speed)")
    ap.add_argument("--stress", help="eval-only embedding set (e.g. In-the-Wild): never fit on")
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--pi-synth", type=float, default=0.3, help="NSA prior (70/30 real/synth)")
    args = ap.parse_args()

    t0 = time.time()
    folds = None
    if args.folds:
        parts = [load_embeddings(args.model, n) for n in args.train.split(",")]
        X = np.concatenate([p[0] for p in parts])
        m = pd.concat([p[1] for p in parts], ignore_index=True)
        f = pd.read_csv(args.folds).set_index("path")
        m = m.join(f[["group", "fold"]], on="path")
        assert m.fold.notna().all(), "embedding rows missing from the fold file"
        tr, va = (m.fold != "holdout").to_numpy(), (m.fold == "holdout").to_numpy()
        Xtr, mtr, Xva, mva = X[tr], m[tr].reset_index(drop=True), X[va], m[va].reset_index(drop=True)
        if args.max_train and len(mtr) > args.max_train:
            keep = mtr.groupby("label", group_keys=False).sample(
                frac=args.max_train / len(mtr), random_state=0).index.sort_values()
            Xtr, mtr = Xtr[keep], mtr.loc[keep].reset_index(drop=True)
        folds = mtr.fold.astype(int).to_numpy()
        groups = mtr.group.to_numpy()
    else:
        assert args.val, "--val or --folds is required"
        Xtr, mtr = load_embeddings(args.model, args.train)
        Xva, mva = load_embeddings(args.model, args.val)
        groups = cv_groups(mtr)
    ytr, yva = (mtr.label == "spoof").to_numpy(int), (mva.label == "spoof").to_numpy(int)
    print(f"train {Xtr.shape} spoof={ytr.mean():.2f}  val {Xva.shape} spoof={yva.mean():.2f}")

    cv = {}
    for layer in range(Xtr.shape[1]):
        oof = oof_scores(Xtr[:, layer], ytr, groups, args.c, folds=folds)
        cv[layer] = {"min_dcf": round(min_cost(ytr, oof, args.pi_synth), 4),
                     "eer": round(eer(ytr, oof), 4)}  # fmt: skip
        print(f"  layer {layer:2d}  CV {cv[layer]}", flush=True)
    best = min(cv, key=lambda k: (cv[k]["min_dcf"], cv[k]["eer"]))

    oof = oof_scores(Xtr[:, best], ytr, groups, args.c, folds=folds)
    platt = LogisticRegression(class_weight="balanced").fit(oof[:, None], ytr)
    clf = make_clf(args.c).fit(Xtr[:, best], ytr)
    probe = Probe(args.model, best, clf, float(platt.coef_[0, 0]), float(platt.intercept_[0]))
    meta_path = REPO / "outputs" / "embeddings" / args.model / args.train / "extract_meta.json"
    if meta_path.exists():
        em = json.loads(meta_path.read_text())
        probe.segment = em.get("mode") == "segment"
        probe.max_windows = em.get("max_windows", probe.max_windows)

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
    per_src = {}
    if "source" in mva:
        spoof_llr = llr[yva == 1]
        for src in sorted(set(mva.loc[yva == 0, "source"])):
            b = llr[((mva.source == src) & (yva == 0)).to_numpy()]
            yy = np.r_[np.zeros(len(b)), np.ones(len(spoof_llr))]
            per_src[src] = {"n": len(b), "eer": round(eer(yy, np.r_[b, spoof_llr]), 4),
                            "min_dcf": round(min_cost(yy, np.r_[b, spoof_llr], args.pi_synth), 4)}  # fmt: skip

    stress = {}
    if args.stress:
        Xs, ms = load_embeddings(args.model, args.stress)
        ys = (ms.label == "spoof").to_numpy(int)
        ss = probe.llr(Xs)
        oof_llr = probe.platt_a * oof + probe.platt_b
        thr = min(np.unique(oof_llr), key=lambda t: cost_at(ytr, oof_llr, t, args.pi_synth))
        stress = {"set": args.stress, "n_bona": int((ys == 0).sum()), "n_spoof": int((ys == 1).sum()),
                  "min_dcf": round(min_cost(ys, ss, args.pi_synth), 4), "eer": round(eer(ys, ss), 4),
                  "inner_threshold_llr": round(float(thr), 4),
                  "p_fa_at_inner_thr": round(float(np.mean(ss[ys == 0] >= thr)), 4),
                  "p_miss_at_inner_thr": round(float(np.mean(ss[ys == 1] < thr)), 4),
                  "act_dcf_at_inner_thr": round(cost_at(ys, ss, thr, args.pi_synth), 4)}  # fmt: skip
        print("stress:", stress)

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    out = REPO / "models" / f"m1_{args.model}_L{best}_{stamp}"
    meta = {
        "rung": "M1", "backbone": args.model, "train": args.train, "val": args.val or "holdout",
        "folds": str(args.folds) if args.folds else None,
        "layer": best, "C": args.c, "selection": "CV normalized minDCF (C_FA=4)",
        "cv_by_layer": cv, "cv_best": cv[best], "platt": [probe.platt_a, probe.platt_b],
        **val, "val_by_generator": per_gen, "val_by_bonafide_source": per_src, "stress": stress, "seconds": round(time.time() - t0),
    }  # fmt: skip
    probe.save(out, meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "cv_by_layer"}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
