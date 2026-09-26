"""Post-draft heads diagnostics for a new probe export (reads exports only; writes nothing).

1. Reproduce the shipped rule (A3 w0.2 + E) on In-the-Wild; its misses at the ITW brief argmin
   must be exactly 15 FA / 158 miss before anything is counted.
2. For each column: inner threshold = argmin of the brief cost over its own inner_oof logits;
   count the shipped rule's ITW misses the column alone flags (logit >= threshold).
3. Spearman of the new column vs m1b_v3 on holdout, test and In-the-Wild (Spectra vs M1b on test
   for reference).
4. Column-alone brief and averse minDCF on inner_oof, holdout and In-the-Wild (new column and M1b).

Usage: uv run python scripts/heads_diagnostics.py wavlm_l
(docs/specs/2026-09-26_post-draft-heads.md, D4)
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fs", REPO / "scripts" / "fuse_sweep.py")
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)
S = fs.S


def main() -> None:
    NEW = sys.argv[1] if len(sys.argv) > 1 else None

    m1 = fs.load("m1b_v3")
    hc = fs.load("handcrafted_v5", str(S / "_itw_handcrafted_v5.csv"))
    m3 = fs.load("spectra_aasist", str(REPO / "outputs/spectra/itw_stress/scores.csv"))
    m5 = fs.load("m5_xlsr_ft", str(S / "_itw_m5.csv"))
    cols = {"m1b_v3": m1, "handcrafted_v5": hc, "m5_xlsr_ft": m5, "spectra_aasist": m3}
    if NEW:
        cols[NEW] = fs.load(NEW)
    lab = pd.read_csv(REPO / "splits" / "nsa_folds.csv")
    itwm = pd.read_csv(REPO / "outputs" / "manifests" / "itw_stress.csv")
    labels = pd.concat([lab[["path", "label"]], itwm[["path", "label"]]])
    labels["path"] = labels.path.map(fs.norm)
    labels = labels.set_index("path")

    idx = {}
    for s in ("inner_oof", "holdout", "test", "itw"):
        ix = m1.index[m1.split == s]
        for d in (hc, m3, m5):
            ix = ix.intersection(d.index[d.split == s])
        idx[s] = ix
    y = {
        s: (labels.loc[ix, "label"] == "spoof").to_numpy(int)
        for s, ix in idx.items()
        if s != "test"
    }

    def rank(det, s):
        ref = np.sort(det.loc[idx["inner_oof"], "logit"].to_numpy())
        return np.searchsorted(ref, det.loc[idx[s], "logit"].to_numpy()) / len(ref)

    def lg(det, s, ix=None):
        return det.loc[idx[s] if ix is None else ix, "logit"].to_numpy()

    base = 0.6 * rank(m1, "itw") + 0.2 * rank(hc, "itw") + 0.2 * rank(m5, "itw")
    fused = np.where((lg(m3, "itw") < -3) & (base > 0.5), base * 0.5, base)
    yi = y["itw"]
    t = min(np.unique(fused), key=lambda t_: fs.cost_at(yi, fused, t_, fs.PI))
    fa, miss = int(np.sum(fused[yi == 0] >= t)), int(np.sum(fused[yi == 1] < t))
    print(f"shipped rule ITW brief {fs.brief(yi, fused):.4f}: FA {fa}, miss {miss}")
    assert (fa, miss) == (15, 158), "shipped rule does not reproduce; stop"
    miss_ix = idx["itw"][(yi == 1) & (fused < t)]

    out = {"shipped_itw": {"fa": fa, "miss": miss}, "flags_of_158": {}, "inner_thr": {}}
    for name, d in cols.items():
        oi = lg(d, "inner_oof")
        thr = float(min(np.unique(oi), key=lambda t_: fs.cost_at(y["inner_oof"], oi, t_, fs.PI)))
        out["inner_thr"][name] = round(thr, 4)
        out["flags_of_158"][name] = int(np.sum(lg(d, "itw", miss_ix) >= thr))
    if NEW:
        new = cols[NEW]
        out["spearman_vs_m1b"] = {}
        for s in ("holdout", "test", "itw"):
            ix = idx[s].intersection(new.index[new.split == s])
            out["spearman_vs_m1b"][s] = round(float(spearmanr(lg(m1, s, ix), lg(new, s, ix))[0]), 4)
            out["spearman_vs_m1b"][f"n_{s}"] = len(ix)
    out["alone"] = {}
    for name in ["m1b_v3"] + ([NEW] if NEW else []):
        d = cols[name]
        r = {}
        for s in ("inner_oof", "holdout", "itw"):
            ix = idx[s].intersection(d.index[d.split == s])
            yy = (labels.loc[ix, "label"] == "spoof").to_numpy(int)
            r[f"{s}_brief"] = round(fs.brief(yy, lg(d, s, ix)), 4)
            r[f"{s}_averse"] = round(fs.averse(yy, lg(d, s, ix)), 4)
            r[f"n_{s}"] = len(ix)
        out["alone"][name] = r
    out["spectra_vs_m1b_test"] = round(float(spearmanr(lg(m1, "test"), lg(m3, "test"))[0]), 4)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
