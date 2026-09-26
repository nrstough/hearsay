"""The noise-twin diagnostic for handcrafted v6 (post-draft CPU lane): does training on 20 dB
white-noise twins recover the handcrafted model under 20 dB noise without costing clean rows?

Cohort and perturbations are the channel lane's, imported from scripts/m3_probes.py so nothing
is re-implemented: `holdout_rows(500)` (outer holdout, 125 per real source / spoof generator,
never trained on), `segment()` (trim + one seeded test-length crop) and `perturb()` (`none`,
`noise20` = white Gaussian at exactly 20 dB SNR). Features go through the runtime path
(`hearsay.handcrafted.features` with the bundle's crop mode, band match and families), once per
clip and kind, then every bundle scores the same rows.

Pre-declared bar (post-draft CPU handoff): v6 AUC >= 0.90 under noise20, and v6 clean AUC within
0.005 of 0.998 (v5b's clean AUC on this cohort). Self-check: v5b's logits here must match the
`handcrafted_v5` column of outputs/channel/m3_perturb.csv on the same keys.

Usage:
  uv run python scripts/hc_noise_probe.py --bundles models/hc_selected,models/hc_lgbm_<v6 stamp> [--workers 2]
Writes outputs/channel/hc_noise_features.csv (resumable cache) and outputs/channel/hc_noise.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "channel"
KINDS = ("none", "noise20")
BAR_NOISE_AUC, CLEAN_REF_AUC, CLEAN_TOL = 0.90, 0.998, 0.005


def _probes():
    spec = importlib.util.spec_from_file_location("m3_probes", REPO / "scripts" / "m3_probes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.TEST_DURS = mod._test_durs()
    return mod


def _features_for(job: tuple[str, int, dict]) -> list[dict]:
    from hearsay.handcrafted import features

    path, seed, kw = job
    mp = _probes()
    out = []
    try:
        base = mp.segment(path, seed)
    except Exception as e:  # noqa: BLE001 - reported per row
        return [{"key": f"{path}|{k}|{seed}", "error": repr(e)[:200]} for k in KINDS]
    for k in KINDS:
        try:
            f = features(mp.perturb(base, k, seed), **kw)
            out.append({"key": f"{path}|{k}|{seed}", **f})
        except Exception as e:  # noqa: BLE001
            out.append({"key": f"{path}|{k}|{seed}", "error": repr(e)[:200]})
    return out


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def main() -> None:
    from hearsay.detectors._learned import predictor
    from hearsay.metrics import C_FA, C_MISS, PI_SYNTH, min_cost

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", required=True, help="comma-separated bundle dirs")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    dirs = [Path(b) for b in args.bundles.split(",") if b]
    bundles = {d.resolve().name: joblib.load(d / "model.joblib") for d in dirs}
    kws = {(b["crop_mode"], b.get("band_match", False), tuple(b.get("families") or ())) for b in bundles.values()}
    assert len(kws) == 1, f"bundles disagree on the feature path: {kws}"
    crop_mode, band_match, families = kws.pop()
    kw = {"crop_mode": crop_mode, "band_match": band_match, "families": families}

    mp = _probes()
    rows = mp.holdout_rows(args.n)
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "hc_noise_features.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["key"])
    have = set(done.key)
    todo = [(p, int(s), kw) for p, s in zip(rows.path, rows.seed)
            if not all(f"{p}|{k}|{s}" in have for k in KINDS)]  # fmt: skip
    if todo:
        with ProcessPoolExecutor(args.workers) as ex:
            new = [r for rs in ex.map(_features_for, todo, chunksize=8) for r in rs]
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True).drop_duplicates("key", keep="last")
        done.to_csv(cache, index=False)

    frames = []
    for k in KINDS:
        frames.append(rows.assign(kind=k, key=[f"{p}|{k}|{s}" for p, s in zip(rows.path, rows.seed)]))
    d = pd.concat(frames, ignore_index=True).merge(done, on="key", how="left")
    err = d["error"].notna() if "error" in d else pd.Series(False, index=d.index)
    y_all = (d.label == "spoof").to_numpy(int)

    result: dict = {"n_clips": len(rows), "kinds": list(KINDS), "feature_errors": int(err.sum()),
                    "bar": {"noise20_auc_min": BAR_NOISE_AUC, "clean_auc_ref": CLEAN_REF_AUC,
                            "clean_tol": CLEAN_TOL}, "bundles": {}}  # fmt: skip
    for name, b in bundles.items():
        X = d[b["features"]].to_numpy(np.float32)
        ok = ~err.to_numpy() & np.isfinite(X).all(axis=1)
        p = np.clip(predictor(b).predict_proba(np.nan_to_num(X))[:, 1], 1e-6, 1 - 1e-6)
        s = np.log(p / (1 - p))
        d[f"logit_{name}"] = np.where(ok, s, np.nan)
        per = {}
        for k in KINDS:
            m = ok & (d.kind == k).to_numpy()
            y, sk = y_all[m], s[m]
            per[k] = {"n": int(m.sum()), "auc": round(auc(y, sk), 4),
                      "min_dcf_brief": round(float(min_cost(y, sk, PI_SYNTH)), 4),
                      "min_dcf_averse": round(float(min_cost(y, sk, 0.5, c_fa=1, c_miss=4)), 4)}  # fmt: skip
        result["bundles"][name] = per
    assert (C_FA, C_MISS) == (4, 1)

    # self-check against the channel lane's cached v5 logits on the same keys
    ref = pd.read_csv(OUT / "m3_perturb.csv")
    ref = ref[ref.kind.isin(KINDS)][["key", "handcrafted_v5"]].dropna()
    v5 = Path("models/hc_selected").resolve().name
    if v5 in bundles:
        j = d[["key", f"logit_{v5}"]].merge(ref, on="key").dropna()
        result["selfcheck_v5_vs_m3_perturb"] = {
            "n": len(j), "max_abs_diff": round(float((j[f"logit_{v5}"] - j.handcrafted_v5).abs().max()), 6)}

    d[["key", "kind", "label", *[c for c in d.columns if c.startswith("logit_")]]].to_csv(
        OUT / "hc_noise_scores.csv", index=False)
    (OUT / "hc_noise.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
