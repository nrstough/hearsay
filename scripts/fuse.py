"""M4 fusion v0: pre-declared fusion rules over detector exports, read out honestly.

Inputs: outputs/detector_scores/<name>.csv (path, fold, split, score, logit) for each
--detectors name, plus optional In-the-Wild rows (split "itw" in the export, or --itw
name=path CSV with path + logit columns). Labels come from splits/nsa_folds.csv (inner,
holdout) and outputs/manifests/itw_stress.csv (ITW).

Each detector's logit is standardized with its inner_oof mean/std. Pre-declared rules:
  alone:<d>    one detector
  zmean        mean of standardized logits over all detectors
  rankmean     mean of per-detector ECDF ranks against its inner_oof distribution
  stack_nonlj  logistic stacker fit on inner_oof rows with LJ bona fide excluded (LJ is one
               speaker on both sides of every fold, so its OOF scores are optimistic);
               reported with its weights; its inner number is in-sample, so it is not shown
Readouts (normalized minDCF at pi = 0.3, C_FA = 4): inner OOF (rules without fitting),
holdout, per bona fide source on the holdout, ITW minDCF and real P_FA at the threshold that
minimizes inner DCF, and the test share above 0.5 after a prior-shifted Platt map fit on
inner rows. Nothing is fit on holdout, ITW or test rows.

Usage: uv run python scripts/fuse.py --detectors m1b_v3 handcrafted_v4 spectra_aasist \
    [--itw handcrafted_v4=outputs/detector_scores/_itw_handcrafted_v4.csv] [--write zmean]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from hearsay.metrics import PI_SYNTH, cost_at, eer, min_cost, sigmoid

REPO = Path(__file__).resolve().parents[1]
SCORES = REPO / "outputs" / "detector_scores"


def norm_path(p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else (REPO / q).resolve())


def load(name: str, itw_override: str | None) -> pd.DataFrame:
    d = pd.read_csv(SCORES / f"{name}.csv")
    d["split"] = d.split.replace({"stress": "itw"})
    if itw_override:
        x = pd.read_csv(itw_override)
        x = x.assign(split="itw", fold="")[["path", "fold", "split", "logit"]]
        d = pd.concat([d[d.split != "itw"], x], ignore_index=True)
    d["path"] = d.path.map(norm_path)
    assert not d.path.duplicated().any(), f"{name}: duplicate paths"
    return d.set_index("path")


def best_thr(y, s, pi):
    return min(np.unique(s), key=lambda t: cost_at(y, s, t, pi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", required=True)
    ap.add_argument("--itw", nargs="*", default=[], help="name=path overrides for ITW scores")
    ap.add_argument("--pi-synth", type=float, default=PI_SYNTH)
    ap.add_argument("--write", nargs="*", default=[], help="rules to export for TSV building")
    args = ap.parse_args()
    pi = args.pi_synth
    over = dict(kv.split("=", 1) for kv in args.itw)

    lab = pd.read_csv(REPO / "splits" / "nsa_folds.csv")
    lab["path"] = lab.path.map(norm_path)
    itw = pd.read_csv(REPO / "outputs" / "manifests" / "itw_stress.csv")
    itw["path"] = itw.path.map(norm_path)
    labels = pd.concat([lab[["path", "label", "source"]], itw[["path", "label", "source"]]])
    labels = labels.set_index("path")

    dets = {n: load(n, over.get(n)) for n in args.detectors}
    base = dets[args.detectors[0]]
    idx = {s: base.index[base.split == s] for s in ("inner_oof", "holdout", "test", "itw")}
    for n, d in dets.items():
        for s in ("inner_oof", "holdout", "test"):
            miss = idx[s].difference(d.index[d.split == s])
            assert miss.empty, f"{n}: {len(miss)} {s} rows missing"
    has_itw = all((d.split == "itw").any() for d in dets.values())
    if not has_itw:
        idx["itw"] = pd.Index([])  # a detector lacks ITW rows: skip the ITW readout
    else:
        common = set(idx["itw"])
        for d in dets.values():
            common &= set(d.index[d.split == "itw"])
        idx["itw"] = pd.Index(sorted(common))

    Z, R = {}, {}
    consts: dict = {"detectors": args.detectors, "pi_synth": pi, "standardize": {}, "rules": {}}
    for n, d in dets.items():
        ref = d.loc[idx["inner_oof"], "logit"].to_numpy()
        mu, sd = ref.mean(), ref.std() + 1e-9
        consts["standardize"][n] = {"mean": float(mu), "std": float(sd),
                                    "inner_oof_sorted": np.sort(ref).tolist()}  # fmt: skip
        srt = np.sort(ref)
        Z[n] = {s: (d.loc[ix, "logit"].to_numpy() - mu) / sd for s, ix in idx.items() if len(ix)}
        R[n] = {s: np.searchsorted(srt, d.loc[ix, "logit"].to_numpy()) / len(srt)
                for s, ix in idx.items() if len(ix)}  # fmt: skip

    y = {s: (labels.loc[ix, "label"] == "spoof").to_numpy(int) for s, ix in idx.items()
         if s != "test" and len(ix)}  # fmt: skip
    src = {s: labels.loc[ix, "source"].to_numpy() for s, ix in idx.items() if s != "test"}

    rules: dict[str, dict[str, np.ndarray]] = {}
    for n in args.detectors:
        rules[f"alone:{n}"] = Z[n]
    if len(args.detectors) > 1:
        rules["zmean"] = {s: np.mean([Z[n][s] for n in args.detectors], axis=0) for s in Z[args.detectors[0]]}
        rules["rankmean"] = {s: np.mean([R[n][s] for n in args.detectors], axis=0) for s in R[args.detectors[0]]}
        keep = ~((src["inner_oof"] == "ljspeech") & (y["inner_oof"] == 0))
        Xi = np.column_stack([Z[n]["inner_oof"] for n in args.detectors])
        st = LogisticRegression(C=1.0, class_weight="balanced").fit(Xi[keep], y["inner_oof"][keep])
        rules["stack_nonlj"] = {s: st.decision_function(np.column_stack([Z[n][s] for n in args.detectors]))
                                for s in Z[args.detectors[0]]}  # fmt: skip
        print("stack_nonlj weights:", dict(zip(args.detectors, st.coef_[0].round(3), strict=True)))
        consts["rules"]["stack_nonlj"] = {"weights": st.coef_[0].tolist(),
                                          "intercept": float(st.intercept_[0])}  # fmt: skip

    report = {}
    for name, sc in rules.items():
        r = {}
        if name != "stack_nonlj":
            r["inner_mindcf"] = round(min_cost(y["inner_oof"], sc["inner_oof"], pi), 4)
        r["holdout_mindcf"] = round(min_cost(y["holdout"], sc["holdout"], pi), 4)
        r["holdout_eer"] = round(eer(y["holdout"], sc["holdout"]), 4)
        hs = sc["holdout"][y["holdout"] == 1]
        for s_ in ("ljspeech", "librispeech"):
            b = sc["holdout"][(y["holdout"] == 0) & (src["holdout"] == s_)]
            r[f"holdout_{s_}"] = round(min_cost(np.r_[np.zeros(len(b)), np.ones(len(hs))],
                                                np.r_[b, hs], pi), 4)  # fmt: skip
        thr = best_thr(y["inner_oof"], sc["inner_oof"], pi)
        if has_itw:
            yi, si = y["itw"], sc["itw"]
            r["itw_mindcf"] = round(min_cost(yi, si, pi), 4)
            r["itw_pfa_at_inner_thr"] = round(float(np.mean(si[yi == 0] >= thr)), 4)
            r["itw_pmiss_at_inner_thr"] = round(float(np.mean(si[yi == 1] < thr)), 4)
        platt = LogisticRegression(class_weight="balanced").fit(sc["inner_oof"][:, None], y["inner_oof"])
        p_test = sigmoid(platt.decision_function(sc["test"][:, None]) + math.log(pi / (1 - pi)))
        r["test_share_gt_0.5"] = round(float(np.mean(p_test > 0.5)), 4)
        consts["rules"].setdefault(name, {})["platt"] = {
            "a": float(platt.coef_[0, 0]), "b": float(platt.intercept_[0]),
            "prior_shift": math.log(pi / (1 - pi)),
            "p": "sigmoid(a * fused + b + prior_shift)",
        }
        report[name] = r
        if name in args.write:
            dest = REPO / "outputs" / "fusion" / f"{name.replace(':', '_')}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"path": idx["test"], "p": p_test, "fused": sc["test"]}).to_csv(dest, index=False)
            print(f"wrote {dest}")

    df = pd.DataFrame(report).T
    pd.set_option("display.width", 220)
    print(df.to_string())
    consts["how"] = {
        "z": "(logit - standardize[d].mean) / standardize[d].std, per detector",
        "zmean": "mean of z over detectors",
        "rankmean": "mean over detectors of searchsorted(inner_oof_sorted, logit) / len",
        "stack_nonlj": "weights . z + intercept",
        "alone:<d>": "z of that detector",
    }
    cdir = REPO / "models" / "fusion_v0"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "constants.json").write_text(json.dumps(consts))
    out = REPO / "outputs" / "fusion" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"detectors": args.detectors, "pi_synth": pi, "rules": report}, indent=2))


if __name__ == "__main__":
    main()
