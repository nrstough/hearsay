"""Score a feature file with a saved learned-detector bundle and read it out against labels,
including the false-alarm rate at the threshold that minimizes the bundle's own inner
out-of-fold DCF. Eval only: nothing is fitted. Built for the In-the-Wild stress readout
(outputs/manifests/itw_stress.csv, 3,000 clips, never trained on) but works for any feature
file with a `label` column.

Usage: uv run python scripts/eval_bundle.py --bundle models/hc_lgbm_<stamp> \
    --features outputs/handcrafted/itw_stress_v4.csv --oof outputs/detector_scores/handcrafted_v4.csv \
    [--name itw_stress] [--pi-synth 0.3]
Writes <bundle>/eval_<name>.json and prints it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from hearsay.detectors._learned import predictor
from hearsay.metrics import C_FA, C_MISS, PI_SYNTH, eer, min_cost, report

REPO = Path(__file__).resolve().parents[1]


def best_threshold(y: np.ndarray, s: np.ndarray, pi: float) -> float:
    """Score threshold (call synthetic when s >= thr) minimizing normalized DCF."""
    fpr, tpr, thr = roc_curve(y, s)
    c = C_FA * fpr * (1 - pi) + C_MISS * (1 - tpr) * pi
    return float(thr[int(np.argmin(c))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--oof", type=Path, required=True,
                    help="the bundle's fusion export; its inner_oof rows set the threshold")
    ap.add_argument("--name", default="itw_stress")
    ap.add_argument("--pi-synth", type=float, default=PI_SYNTH)
    args = ap.parse_args()

    b = joblib.load(args.bundle / "model.joblib")
    d = pd.read_csv(args.features)
    flag = next((c for c in d.columns if c.endswith("_flag")), None)
    ok = (d[flag].fillna("") == "") if flag else pd.Series(True, index=d.index)
    d = d[ok].reset_index(drop=True)
    missing = [c for c in b["features"] if c not in d.columns]
    assert not missing, f"feature file lacks {missing[:5]}"
    X = d[b["features"]].to_numpy(np.float32)
    p = np.clip(predictor(b).predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    s = np.log(p / (1 - p))
    y = (d.label == "spoof").to_numpy(int)

    oof = pd.read_csv(args.oof)
    oof = oof[(oof.split == "inner_oof") & oof.logit.notna()]
    lab = pd.read_csv(REPO / "splits" / "nsa_folds.csv")[["path", "label"]]
    oof = oof.merge(lab, on="path")
    thr = best_threshold((oof.label == "spoof").to_numpy(int), oof.logit.to_numpy(), args.pi_synth)

    out = {"bundle": args.bundle.name, "features": str(args.features), "n": len(d),
           "n_bonafide": int((y == 0).sum()), "n_spoof": int((y == 1).sum()),
           "dropped_rows": int((~ok).sum()),
           **{k: v for k, v in report(y, s, args.pi_synth).items()},
           "auc": round(float(roc_auc_score(y, s)), 4),
           "inner_oof_threshold_logit": round(thr, 4),
           "p_fa_at_inner_threshold": round(float((s[y == 0] >= thr).mean()), 4),
           "p_miss_at_inner_threshold": round(float((s[y == 1] < thr).mean()), 4),
           "dcf_at_inner_threshold": round(float(
               (C_FA * (s[y == 0] >= thr).mean() * (1 - args.pi_synth)
                + C_MISS * (s[y == 1] < thr).mean() * args.pi_synth)
               / min(C_FA * (1 - args.pi_synth), C_MISS * args.pi_synth)), 4),
           "score_mean_bonafide": round(float(p[y == 0].mean()), 4),
           "score_mean_spoof": round(float(p[y == 1].mean()), 4)}  # fmt: skip
    if "source" in d:
        out["by_source"] = {}
        for src, g in d.groupby("source"):
            yy = (g.label == "spoof").to_numpy(int)
            ss = s[g.index.to_numpy()]
            entry = {"n": len(g)}
            if 0 < yy.sum() < len(yy):
                entry |= {"min_dcf": round(min_cost(yy, ss, args.pi_synth), 4),
                          "eer": round(eer(yy, ss), 4)}  # fmt: skip
            if (yy == 0).any():
                entry["p_fa_at_inner_threshold"] = round(float((ss[yy == 0] >= thr).mean()), 4)
            out["by_source"][str(src)] = entry
    (args.bundle / f"eval_{args.name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
