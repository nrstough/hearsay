"""Pre-declared fusion sweep (docs/reports/2026-09-26_fusion-sweep-predeclared.md).

Candidates A-E over M1b v3 + handcrafted v5, M3 only as false-alarm suppression. Readouts
under two costs:
- brief: pi_synth 0.3, C_FA 4          (normalized 9.33*P_FA + P_miss)
- miss-averse (sponsor code semantics): pi 0.5, C_FA 1, C_miss 4 (normalized P_FA + 4*P_miss)
Selection applies the pre-declared rule mechanically. With --write it also emits the winner's
test scores in both polarities, pinned-block ready (determinate scores in [0.001, 1]).

Usage: uv run python scripts/fuse_sweep.py [--write]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from hearsay.metrics import cost_at, min_cost, sigmoid

REPO = Path(__file__).resolve().parents[1]
S = REPO / "outputs" / "detector_scores"
PI = 0.3


def norm(p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else (REPO / q).resolve())


def load(name: str, itw: str | None = None) -> pd.Series:
    d = pd.read_csv(S / f"{name}.csv")
    d["split"] = d.split.replace({"stress": "itw"})
    if itw:
        x = pd.read_csv(itw).assign(split="itw")
        d = pd.concat([d[d.split != "itw"], x[["path", "split", "logit"]]], ignore_index=True)
    d["path"] = d.path.map(norm)
    return d.drop_duplicates("path").set_index("path")


def brief(y, s):
    return min_cost(y, s, PI)


def averse(y, s):
    return min_cost(y, s, 0.5, c_fa=1.0, c_miss=4.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    m1 = load("m1b_v3")
    hc = load("handcrafted_v5", str(S / "_itw_handcrafted_v5.csv"))
    m3 = load("spectra_aasist", str(REPO / "outputs/spectra/itw_stress/scores.csv"))
    lab = pd.read_csv(REPO / "splits" / "nsa_folds.csv")
    itwm = pd.read_csv(REPO / "outputs" / "manifests" / "itw_stress.csv")
    labels = pd.concat([lab[["path", "label", "source"]], itwm[["path", "label", "source"]]])
    labels["path"] = labels.path.map(norm)
    labels = labels.set_index("path")

    idx = {}
    for s in ("inner_oof", "holdout", "test", "itw"):
        ix = m1.index[m1.split == s]
        ix = ix.intersection(hc.index[hc.split == s]).intersection(m3.index[m3.split == s])
        idx[s] = ix
    y = {s: (labels.loc[ix, "label"] == "spoof").to_numpy(int) for s, ix in idx.items() if s != "test"}
    src = {s: labels.loc[ix, "source"].to_numpy() for s, ix in idx.items() if s != "test"}

    def rank(det: pd.Series, s: str) -> np.ndarray:
        ref = np.sort(det.loc[idx["inner_oof"], "logit"].to_numpy())
        return np.searchsorted(ref, det.loc[idx[s], "logit"].to_numpy()) / len(ref)

    def z(det: pd.Series, s: str) -> np.ndarray:
        ref = det.loc[idx["inner_oof"], "logit"].to_numpy()
        return (det.loc[idx[s], "logit"].to_numpy() - ref.mean()) / ref.std()

    R1 = {s: rank(m1, s) for s in idx}
    RH = {s: rank(hc, s) for s in idx}
    cands: dict[str, dict[str, np.ndarray]] = {}
    for a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        cands[f"A_alpha{a:.1f}"] = {s: (1 - a) * R1[s] + a * RH[s] for s in idx}
    cands["B_min"] = {s: np.minimum(R1[s], RH[s]) for s in idx}
    cands["B_max"] = {s: np.maximum(R1[s], RH[s]) for s in idx}
    lo, hi = np.percentile(R1["inner_oof"], [40, 90])
    cands["C_cascade"] = {s: np.where((R1[s] >= lo) & (R1[s] <= hi), (R1[s] + RH[s]) / 2, R1[s])
                          for s in idx}  # fmt: skip
    keep = ~((src["inner_oof"] == "ljspeech") & (y["inner_oof"] == 0))
    Zi = np.column_stack([z(m1, "inner_oof"), z(hc, "inner_oof")])
    w = np.clip(LogisticRegression(class_weight="balanced").fit(Zi[keep], y["inner_oof"][keep]).coef_[0], 0, None)
    w = w / w.sum() if w.sum() > 0 else np.array([0.5, 0.5])
    w = 0.5 * w + 0.5 * np.array([0.5, 0.5])
    cands["D_nonneg_shrunk"] = {s: w[0] * z(m1, s) + w[1] * z(hc, s) for s in idx}

    def readout(sc):
        r = {"inner_brief": round(brief(y["inner_oof"], sc["inner_oof"]), 4),
             "holdout_brief": round(brief(y["holdout"], sc["holdout"]), 4),
             "holdout_averse": round(averse(y["holdout"], sc["holdout"]), 4)}  # fmt: skip
        hs = sc["holdout"]
        t = min(np.unique(hs), key=lambda t_: cost_at(y["holdout"], hs, t_, PI))
        r["holdout_argmin_FA_miss"] = [int(np.sum(hs[y["holdout"] == 0] >= t)),
                                       int(np.sum(hs[y["holdout"] == 1] < t))]  # fmt: skip
        spoof = hs[y["holdout"] == 1]
        for s_ in ("ljspeech", "librispeech"):
            b = hs[(y["holdout"] == 0) & (src["holdout"] == s_)]
            r[f"holdout_{s_}"] = round(brief(np.r_[np.zeros(len(b)), np.ones(len(spoof))], np.r_[b, spoof]), 4)
        r["itw_brief"] = round(brief(y["itw"], sc["itw"]), 4)
        r["itw_averse"] = round(averse(y["itw"], sc["itw"]), 4)
        ti = min(np.unique(sc["inner_oof"]), key=lambda t_: cost_at(y["inner_oof"], sc["inner_oof"], t_, PI))
        r["itw_pfa_at_inner_thr"] = round(float(np.mean(sc["itw"][y["itw"] == 0] >= ti)), 4)
        r["itw_pmiss_at_inner_thr"] = round(float(np.mean(sc["itw"][y["itw"] == 1] < ti)), 4)
        return r

    rep = {k: readout(v) for k, v in cands.items()}
    best_inner = min(r["inner_brief"] for r in rep.values())
    adm = {k: r for k, r in rep.items()
           if r["inner_brief"] <= best_inner + 0.03 and r["itw_averse"] <= 0.45}  # fmt: skip
    if adm:
        order = sorted(adm, key=lambda k: (adm[k]["itw_brief"], k))
        top = order[0]
        ties = [k for k in order if adm[k]["itw_brief"] <= adm[top]["itw_brief"] + 0.01]
        alphas = [k for k in ties if k.startswith("A_alpha")]
        winner = min(alphas, key=lambda k: float(k[7:])) if alphas else top
    else:
        winner = "A_alpha0.0"

    # E: M3 as FA suppression on top of the winner (rank domain)
    base = cands[winner]
    if winner.startswith("D_"):
        ref = np.sort(base["inner_oof"])
        base = {s: np.searchsorted(ref, v) / len(ref) for s, v in base.items()}
    m3l = {s: m3.loc[idx[s], "logit"].to_numpy() for s in idx}
    e = {s: np.where((m3l[s] < -3) & (base[s] > 0.5), base[s] * 0.5, base[s]) for s in idx}
    rep["E_on_" + winner] = readout(e)
    rw, re_ = readout(base), rep["E_on_" + winner]
    use_e = (re_["itw_brief"] <= rw["itw_brief"] - 0.01 and re_["holdout_brief"] <= rw["holdout_brief"] + 0.01
             and re_["inner_brief"] <= rw["inner_brief"] + 0.01)  # fmt: skip
    final = ("E_on_" + winner) if use_e else winner
    final_sc = e if use_e else cands[winner]

    pd.set_option("display.width", 250)
    print(pd.DataFrame(rep).T.to_string())
    print(f"\nadmissible: {sorted(adm)}\nwinner (A-D): {winner}; apply E: {use_e}; FINAL: {final}")
    out = REPO / "outputs" / "fusion"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep_report.json").write_text(json.dumps(
        {"candidates": rep, "admissible": sorted(adm), "winner_AD": winner, "apply_E": use_e,
         "final": final, "D_weights": w.tolist(), "cascade_band": [float(lo), float(hi)]}, indent=2))

    if args.write:
        pl = LogisticRegression(class_weight="balanced").fit(final_sc["inner_oof"][:, None], y["inner_oof"])
        p = sigmoid(pl.decision_function(final_sc["test"][:, None]) + math.log(PI / (1 - PI)))
        det = 0.001 + 0.999 * p  # determinate scores in [0.001, 1]; pinned block would sit below
        pd.DataFrame({"path": idx["test"], "p": det, "p_flipped": 0.001 + 0.999 * (1 - p),
                      "fused": final_sc["test"]}).to_csv(out / "sweep_final_test.csv", index=False)  # fmt: skip
        print(f"wrote {out / 'sweep_final_test.csv'}")
        refs = {n: np.sort(d.loc[idx["inner_oof"], "logit"].to_numpy()).tolist()
                for n, d in (("m1b_v3", m1), ("handcrafted_v5", hc))}  # fmt: skip
        alpha = float(winner[7:]) if winner.startswith("A_alpha") else None
        cdir = REPO / "models" / "fusion_v1"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "constants.json").write_text(json.dumps({
            "final": final, "pi_synth": PI,
            "rank_ref_inner_oof_sorted": refs,
            "alpha_handcrafted": alpha,
            "e_rule": {"applied": use_e, "m3_logit_below": -3.0, "base_rank_above": 0.5,
                       "multiply_by": 0.5} if use_e else {"applied": False},
            "platt": {"a": float(pl.coef_[0, 0]), "b": float(pl.intercept_[0]),
                      "prior_shift": math.log(PI / (1 - PI))},
            "determinate_map": "0.001 + 0.999 * sigmoid(a*fused + b + prior_shift); gated files below 0.001",
            "how": "rank_d = searchsorted(rank_ref[d], logit_d)/len; base = (1-alpha)*rank_m1b + alpha*rank_hc; "
                   "if m3_logit < -3 and base > 0.5: base *= 0.5; p = determinate_map(base)",
        }))
        print(f"wrote {cdir / 'constants.json'}")


if __name__ == "__main__":
    main()
