"""Pre-declared M5 fusion candidates F and A3 (addendum in
docs/reports/2026-09-26_fusion-sweep-predeclared.md). Reuses fuse_sweep's loaders and
readouts; never rewrites the frozen fusion_v1 constants.

Always writes outputs/fusion/sweep_m5_report.json and, when a candidate qualifies,
outputs/fusion/sweep_m5_final_test.csv. Nothing under models/ is written unless --write is
given: then the winner's constants go to models/fusion_v2/constants.json (the runner's file:
`weights` over the three ranked columns, their inner-OOF rank references, the M3 step, the
Platt map, `how`, and the M5 checkpoint's hashes so the runner can refuse another checkpoint).

Usage: uv run python scripts/fuse_sweep_m5.py [--write]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fs", REPO / "scripts" / "fuse_sweep.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write models/fusion_v2/constants.json (the runner's file); without it nothing under models/ is written")  # fmt: skip
    args = ap.parse_args()
    S = fs.S
    m1 = fs.load("m1b_v3")
    hc = fs.load("handcrafted_v5", str(S / "_itw_handcrafted_v5.csv"))
    m3 = fs.load("spectra_aasist", str(REPO / "outputs/spectra/itw_stress/scores.csv"))
    m5 = fs.load("m5_xlsr_ft", str(S / "_itw_m5.csv"))
    lab = pd.read_csv(REPO / "splits" / "nsa_folds.csv")
    itwm = pd.read_csv(REPO / "outputs" / "manifests" / "itw_stress.csv")
    labels = pd.concat([lab[["path", "label", "source"]], itwm[["path", "label", "source"]]])
    labels["path"] = labels.path.map(fs.norm)
    labels = labels.set_index("path")
    idx = {}
    for s in ("inner_oof", "holdout", "test", "itw"):
        ix = m1.index[m1.split == s]
        for d in (hc, m3, m5):
            ix = ix.intersection(d.index[d.split == s])
        idx[s] = ix
    y = {s: (labels.loc[ix, "label"] == "spoof").to_numpy(int) for s, ix in idx.items() if s != "test"}
    src = {s: labels.loc[ix, "source"].to_numpy() for s, ix in idx.items() if s != "test"}
    print({s: len(ix) for s, ix in idx.items()})

    def rank(det, s):
        ref = np.sort(det.loc[idx["inner_oof"], "logit"].to_numpy())
        return np.searchsorted(ref, det.loc[idx[s], "logit"].to_numpy()) / len(ref)

    def lg(det, s):
        return det.loc[idx[s], "logit"].to_numpy()

    R1, RH, R5 = ({s: rank(d, s) for s in idx} for d in (m1, hc, m5))

    def e_step(base, s):
        return np.where((lg(m3, s) < -3) & (base > 0.5), base * 0.5, base)

    inner_spoof_m5 = lg(m5, "inner_oof")[y["inner_oof"] == 1]
    t_f = float(np.percentile(inner_spoof_m5, 1))
    real_cov = float(np.mean(lg(m5, "inner_oof")[y["inner_oof"] == 0] < t_f))
    print(f"t_F = {t_f:.4f} (1% of inner spoof below); inner real coverage {real_cov:.3f}")

    current = {s: e_step(0.8 * R1[s] + 0.2 * RH[s], s) for s in idx}
    cands = {"CURRENT_E_on_A0.2": current,
             "F_m5_suppress": {s: np.where((lg(m5, s) < t_f) & (current[s] > 0.5), current[s] * 0.5,
                                           current[s]) for s in idx}}  # fmt: skip
    for w in (0.1, 0.2):
        blend = {s: (0.8 - w) * R1[s] + 0.2 * RH[s] + w * R5[s] for s in idx}
        cands[f"A3_w{w:.1f}"] = blend
        cands[f"A3_w{w:.1f}_E"] = {s: e_step(blend[s], s) for s in idx}

    # reuse fuse_sweep's readout by rebuilding its closure inputs
    def readout(sc):
        r = {"inner_brief": round(fs.brief(y["inner_oof"], sc["inner_oof"]), 4),
             "holdout_brief": round(fs.brief(y["holdout"], sc["holdout"]), 4),
             "holdout_averse": round(fs.averse(y["holdout"], sc["holdout"]), 4)}  # fmt: skip
        hs = sc["holdout"]
        t = min(np.unique(hs), key=lambda t_: fs.cost_at(y["holdout"], hs, t_, fs.PI))
        r["holdout_argmin_FA_miss"] = [int(np.sum(hs[y["holdout"] == 0] >= t)),
                                       int(np.sum(hs[y["holdout"] == 1] < t))]  # fmt: skip
        spoof = hs[y["holdout"] == 1]
        for s_ in ("ljspeech", "librispeech"):
            b = hs[(y["holdout"] == 0) & (src["holdout"] == s_)]
            r[f"holdout_{s_}"] = round(fs.brief(np.r_[np.zeros(len(b)), np.ones(len(spoof))], np.r_[b, spoof]), 4)
        r["itw_brief"] = round(fs.brief(y["itw"], sc["itw"]), 4)
        r["itw_averse"] = round(fs.averse(y["itw"], sc["itw"]), 4)
        ti = min(np.unique(sc["inner_oof"]), key=lambda t_: fs.cost_at(y["inner_oof"], sc["inner_oof"], t_, fs.PI))
        r["itw_pfa_at_inner_thr"] = round(float(np.mean(sc["itw"][y["itw"] == 0] >= ti)), 4)
        r["itw_pmiss_at_inner_thr"] = round(float(np.mean(sc["itw"][y["itw"] == 1] < ti)), 4)
        return r

    rep = {k: readout(v) for k, v in cands.items()}
    cur = rep["CURRENT_E_on_A0.2"]
    ok = {k: r for k, r in rep.items() if k != "CURRENT_E_on_A0.2"
          and r["itw_brief"] <= cur["itw_brief"] - 0.01 and r["itw_averse"] <= cur["itw_averse"] + 0.01
          and r["holdout_brief"] <= cur["holdout_brief"] + 0.05 and r["inner_brief"] <= cur["inner_brief"] + 0.03}
    winner = min(ok, key=lambda k: ok[k]["itw_brief"]) if ok else None
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rep).T.to_string())
    print(f"\nqualifying: {sorted(ok)}; decision: {'REPLACE with ' + winner if winner else 'KEEP current rule (frozen)'}")
    if winner:
        import math

        from sklearn.linear_model import LogisticRegression

        from hearsay.metrics import sigmoid

        sc = cands[winner]
        pl = LogisticRegression(class_weight="balanced").fit(sc["inner_oof"][:, None], y["inner_oof"])
        p = sigmoid(pl.decision_function(sc["test"][:, None]) + math.log(fs.PI / (1 - fs.PI)))
        pd.DataFrame({"path": idx["test"], "p": 0.001 + 0.999 * p, "p_flipped": 0.001 + 0.999 * (1 - p),
                      "fused": sc["test"]}).to_csv(REPO / "outputs/fusion/sweep_m5_final_test.csv", index=False)
        print("wrote outputs/fusion/sweep_m5_final_test.csv")
        if args.write:
            from hearsay.pipeline import M5_DIR

            w = float(winner.split("_w")[1][:3])
            hashes = json.loads((M5_DIR / "hashes.json").read_text())
            cdir = REPO / "models" / "fusion_v2"
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / "constants.json").write_text(json.dumps({
                "final": winner, "pi_synth": fs.PI,
                "rank_ref_inner_oof_sorted": {n: np.sort(d.loc[idx["inner_oof"], "logit"].to_numpy()).tolist()
                                              for n, d in (("m1b_v3", m1), ("handcrafted_v5", hc), ("m5_xlsr_ft", m5))},
                "weights": {"m1b_v3": round(0.8 - w, 2), "handcrafted_v5": 0.2, "m5_xlsr_ft": w},
                "e_rule": {"applied": winner.endswith("_E"), "m3_logit_below": -3.0, "base_rank_above": 0.5,
                           "multiply_by": 0.5},
                "platt": {"a": float(pl.coef_[0, 0]), "b": float(pl.intercept_[0]),
                          "prior_shift": math.log(fs.PI / (1 - fs.PI))},
                "determinate_map": "0.001 + 0.999 * sigmoid(a*fused + b + prior_shift), applied once",
                "how": "rank_d = searchsorted(rank_ref[d], logit_d)/len; base = sum(weights[d] * rank_d) over "
                       "m1b_v3, handcrafted_v5, m5_xlsr_ft, accumulated in that order; "
                       "if m3_logit < -3 and base > 0.5: base *= 0.5; p = determinate_map(base)",
                "m5_checkpoint": {"dir": str(M5_DIR.relative_to(REPO)),
                                  **{k: hashes[k] for k in ("backbone_sha256", "head_sha256", "config_hash")}},
            }))
            print(f"wrote {cdir / 'constants.json'}")
    (REPO / "outputs" / "fusion" / "sweep_m5_report.json").write_text(json.dumps(
        {"t_F": t_f, "t_F_inner_real_coverage": real_cov, "candidates": rep, "qualifying": sorted(ok),
         "winner": winner}, indent=2))


if __name__ == "__main__":
    main()
