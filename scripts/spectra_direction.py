"""Spectra-AASIST on a labeled sample: assert score direction, then fit a provisional Platt map.

Input manifest CSV: columns `path` and `label` (spoof | bonafide; "bona-fide" is accepted).
Writes raw per-file scores to outputs/spectra/<name>_scores.csv and, with --fit-platt, the
Platt parameters to outputs/calibration/spectra_platt_<name>.json.

Direction contract: synth_logit must increase with synthetic likelihood. The script exits
non-zero if AUC(synth_logit, is_spoof) < 0.5 or if the spoof median is not above the bona fide
median, so a flipped stream can never feed a CSV.

The Platt fit is class-balanced (prior-neutral: p = 0.5 at the EER-ish crossing), because the
NSA test prior is unknown. Never fit it on the NSA validation split or on In-the-Wild.

Usage:
  uv run python scripts/spectra_direction.py --manifest outputs/manifests/asv19_dev_sample.csv \
      --name asv19dev --fit-platt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from hearsay.audio import DecodeError, load_audio
from hearsay.spectra import load_spectra, spectra_logits, synth_logit

REPO = Path(__file__).resolve().parents[1]


def eer(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fit-platt", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    m["label"] = m["label"].str.lower().str.replace("-", "", regex=False)
    assert set(m["label"]) <= {"spoof", "bonafide"}, set(m["label"])

    model = load_spectra()
    rows, t0 = [], time.time()
    for i, r in enumerate(m.itertuples()):
        try:
            lg = spectra_logits(model, load_audio(r.path))
            rows.append((lg[0], lg[1], synth_logit(lg), ""))
        except DecodeError:
            rows.append((np.nan, np.nan, np.nan, "decode_error"))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(m)} ({time.time() - t0:.0f}s)", flush=True)
    m[["logit_spoof", "logit_bonafide", "synth_logit", "flag"]] = rows

    out_dir = REPO / "outputs" / "spectra"
    out_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(out_dir / f"{args.name}_scores.csv", index=False)

    ok = m[m.flag == ""]
    y = (ok.label == "spoof").to_numpy(int)
    s = ok.synth_logit.to_numpy()
    auc = roc_auc_score(y, s)
    e, thr = eer(y, s)
    med_spoof, med_bona = np.median(s[y == 1]), np.median(s[y == 0])
    summary = {
        "name": args.name, "n": len(m), "n_ok": len(ok),
        "n_spoof": int(y.sum()), "n_bonafide": int((1 - y).sum()),
        "auc": round(auc, 4), "eer": round(e, 4), "eer_threshold_synth_logit": round(thr, 4),
        "median_synth_spoof": round(float(med_spoof), 3),
        "median_synth_bonafide": round(float(med_bona), 3),
        "sec_per_clip": round((time.time() - t0) / len(m), 3),
    }  # fmt: skip
    print(json.dumps(summary, indent=2))

    if auc < 0.5 or med_spoof <= med_bona:
        sys.exit("DIRECTION FAILED: synth_logit does not increase with synthetic likelihood")
    print("direction OK: synth_logit increases with synthetic likelihood")

    if args.fit_platt:
        lr = LogisticRegression(class_weight="balanced", C=1e6).fit(s[:, None], y)
        a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        assert a > 0, "Platt slope must be positive"
        p = lr.predict_proba(s[:, None])[:, 1]
        cal = {
            "stream": "spectra_aasist.synth_logit", "fit_on": args.name,
            "p_synthetic": "1 / (1 + exp(-(a * synth_logit + b)))", "a": a, "b": b,
            "class_weight": "balanced", "provisional": True, **summary,
            "mean_p_spoof": round(float(p[y == 1].mean()), 4),
            "mean_p_bonafide": round(float(p[y == 0].mean()), 4),
        }  # fmt: skip
        cal_dir = REPO / "outputs" / "calibration"
        cal_dir.mkdir(parents=True, exist_ok=True)
        path = cal_dir / f"spectra_platt_{args.name}.json"
        path.write_text(json.dumps(cal, indent=2))
        print(f"Platt a={a:.4f} b={b:.4f} -> {path}")


if __name__ == "__main__":
    main()
