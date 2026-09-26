"""Action E of the channel-robustness rung: is M3 (Spectra-AASIST) memorising, and what does its
false-alarm-suppression step cost in the shipped rule? (run spec D6-D9, revised after the
Codex plan review).

Three probes:
- perturb: 500 outer-HOLDOUT clips (never trained on by M1b, M5 or handcrafted v5; M3's
  exposure is the open question) under none / MP3 round-trip / 20 dB SNR noise / +-2% speed /
  one-sample shift, scored through the runner's own paths (hearsay.pipeline.Models) and fused
  with the shipped rule (models/fusion_v2/constants.json). Readouts per model and perturbation:
  AUC, minDCF, EER, and for the fused rule with and without the M3 step.
- mlaad: 600 MLAAD English spoof clips (stratified by model), same scoring: each model's miss
  rate at its own inner-OOF threshold and the share of files the shipped rule actually damps
  (M3 logit < -3 AND fused base rank > 0.5), which is the step's miss exposure. M5 trained on
  MLAAD, so its number is in-sample.
- agreement: Spearman(M1b, M3) on holdout vs test rows of the exports. Descriptive only.

Usage:
  uv run python scripts/m3_probes.py perturb   [--n 500] [--threads 3]
  uv run python scripts/m3_probes.py mlaad     [--n 600] [--threads 3]
  uv run python scripts/m3_probes.py agreement
  uv run python scripts/m3_probes.py verdict
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "channel"
S = REPO / "outputs" / "detector_scores"
SR = 16000
PERTURBATIONS = ("none", "mp3", "noise20", "speed", "shift1")
MODELS = ("m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "spectra_aasist")
EXPORT = {"m1b_v3": "m1b_v3", "handcrafted_v5": "handcrafted_v5", "m5_xlsr_ft": "m5_xlsr_ft",
          "spectra_aasist": "spectra_aasist"}  # fmt: skip
M3_SUPPRESS_BELOW = -3.0
N_PERTURB, N_MLAAD = 500, 572  # the run spec's cohorts (MLAAD: 4 clips x 143 models)
MAX_LAG = 4000  # samples searched when aligning a codec round-trip to its input


# ------------------------------------------------------------------ perturbations


def align_to(y: np.ndarray, x: np.ndarray, max_lag: int = MAX_LAG) -> np.ndarray:
    """Shift y (a codec round-trip of x) so it lines up with x, then cut or zero-pad it to
    len(x). The lag is the cross-correlation peak within +-max_lag samples; codecs add encoder
    delay and padding that would otherwise change the clip's length and silence."""
    from scipy.signal import correlate

    n = x.size
    c = correlate(y.astype(np.float64), x.astype(np.float64), mode="full", method="fft")
    lags = np.arange(-n + 1, y.size)
    ok = np.abs(lags) <= max_lag
    lag = int(lags[ok][np.argmax(c[ok])])
    z = y[lag:] if lag >= 0 else np.r_[np.zeros(-lag, y.dtype), y]
    z = z[:n]
    return np.pad(z, (0, n - z.size)).astype(np.float32)


def perturb(x: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """Label-preserving perturbation of a raw 16 kHz segment. `speed` alternates +2% / -2% by
    seed parity (resampling: tempo and pitch both move, the Kaldi speed perturbation)."""
    x = np.asarray(x, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if kind == "none":
        return x.copy()
    if kind == "mp3":
        from hearsay.compression import launder

        return align_to(launder(x, "mp3", 64, SR), x)
    if kind == "noise20":
        p = float(np.mean(x.astype(np.float64) ** 2))
        return (x + rng.standard_normal(x.size) * math.sqrt(p / 100.0)).astype(np.float32)
    if kind == "speed":
        from scipy.signal import resample_poly

        up, down = (50, 51) if seed % 2 == 0 else (51, 50)  # 0.98x length (+2%) / 1.02x (-2%)
        return resample_poly(x, up, down).astype(np.float32)
    if kind == "shift1":
        return np.roll(x, 1)
    raise ValueError(f"unknown perturbation {kind!r}")


def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    n = noisy.astype(np.float64) - clean.astype(np.float64)
    return float(10 * np.log10(np.mean(clean.astype(np.float64) ** 2) / np.mean(n**2)))


# ------------------------------------------------------------------ metrics and verdict


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def delta_auc(y: np.ndarray, s_clean: np.ndarray, s_pert: np.ndarray) -> float:
    """AUC(perturbed) - AUC(clean) on the paired rows where both scores exist."""
    ok = np.isfinite(s_clean) & np.isfinite(s_pert)
    return auc(y[ok], s_pert[ok]) - auc(y[ok], s_clean[ok])


def inner_threshold(det: str) -> float:
    """The brief-cost argmin threshold on the detector's inner-OOF logits (train_probe.py's rule)."""
    from hearsay.metrics import cost_at

    d = pd.read_csv(S / f"{EXPORT[det]}.csv")
    d = d[d.split == "inner_oof"]
    lab = pd.read_csv(REPO / "splits" / "nsa_folds.csv")[["path", "label"]]
    d = d.merge(lab, on="path", how="inner")
    y, s = (d.label == "spoof").to_numpy(int), d.logit.to_numpy(float)
    grid = np.unique(np.quantile(s, np.linspace(0, 1, 2001)))
    return float(min(grid, key=lambda t: cost_at(y, s, t)))


def m3_verdict(m3_dauc: float, m1_dauc: float, mlaad_damped: float,
               e_hurts_perturbed: float) -> tuple[str, list[str]]:  # fmt: skip
    """Pre-declared (run spec D9, revised): the M3 suppression step is "at risk" if
    (a) M3's mean |dAUC| over perturbations > 2x M1b's and > 0.02, or
    (c) the shipped rule damps > 5% of MLAAD spoof (unseen by M1b and handcrafted; M3's training
        data is undisclosed), or
    (d) under some perturbation the step raises the fused holdout minDCF by > 0.01.
    Any NaN input makes the verdict "inconclusive", never "kept"."""
    vals = (m3_dauc, m1_dauc, mlaad_damped, e_hurts_perturbed)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals):
        return "inconclusive", ["a probe did not produce a number"]
    why = []
    if abs(m3_dauc) > 2 * abs(m1_dauc) and abs(m3_dauc) > 0.02:
        why.append(f"(a) M3 mean |dAUC| {abs(m3_dauc):.3f} > 2x M1b's {abs(m1_dauc):.3f}")
    if mlaad_damped > 0.05:
        why.append(f"(c) the shipped rule damps {mlaad_damped:.1%} of MLAAD spoof")
    if e_hurts_perturbed > 0.01:
        why.append(f"(d) the step raises fused minDCF by {e_hurts_perturbed:.3f} under a perturbation")
    return ("at risk" if why else "kept"), why


def verdict_from(p: dict, m: dict, n_perturb: int = N_PERTURB, n_mlaad: int = N_MLAAD) -> dict:
    """Gather every D9 input from the two probe readouts and apply m3_verdict. "inconclusive"
    when a probe is short of its cohort or any input (including any per-perturbation step
    effect) is missing or non-finite: Python's max() would otherwise skip a NaN silently."""
    named = p.get("e_step_effect_min_dcf", {})
    effects = list(named.values())
    inputs = {"m3_mean_abs_d_auc": p.get("mean_abs_d_auc", {}).get("spectra_aasist"),
              "m1b_mean_abs_d_auc": p.get("mean_abs_d_auc", {}).get("m1b_v3"),
              "mlaad_damped_share": m.get("e_applied_share"),
              "e_step_effects": effects, "n_perturb": p.get("n_paired"), "n_mlaad": m.get("n")}  # fmt: skip
    short = []
    if not (isinstance(p.get("n_paired"), int) and p["n_paired"] >= n_perturb):
        short.append(f"perturbation probe has {p.get('n_paired')} of {n_perturb} clips")
    if not (isinstance(m.get("n"), int) and m["n"] >= n_mlaad):
        short.append(f"MLAAD probe has {m.get('n')} of {n_mlaad} clips")
    fin = [isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in effects]
    if short or not all(fin) or set(named) != set(PERTURBATIONS):
        return {"verdict": "inconclusive", "reasons": short or ["a step-effect number is missing"], "inputs": inputs}
    verdict, why = m3_verdict(inputs["m3_mean_abs_d_auc"], inputs["m1b_mean_abs_d_auc"],
                              inputs["mlaad_damped_share"], max(effects))  # fmt: skip
    inputs["max_e_step_min_dcf_increase"] = max(effects)
    return {"verdict": verdict, "reasons": why, "inputs": inputs}


def spearman_gap(a_h, b_h, a_t, b_t, n: int = 1000, seed: int = 0) -> dict:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    rh, rt = spearmanr(a_h, b_h)[0], spearmanr(a_t, b_t)[0]
    gaps = []
    for _ in range(n):
        i, j = rng.integers(0, len(a_h), len(a_h)), rng.integers(0, len(a_t), len(a_t))
        gaps.append(spearmanr(a_h[i], b_h[i])[0] - spearmanr(a_t[j], b_t[j])[0])
    return {"holdout": round(float(rh), 4), "test": round(float(rt), 4), "gap": round(float(rh - rt), 4),
            "gap_ci95": [round(float(np.percentile(gaps, q)), 4) for q in (2.5, 97.5)]}  # fmt: skip


# ------------------------------------------------------------------ scoring


def segment(path: str, seed: int) -> np.ndarray:
    """Decode, trim silence, crop once to a test-length draw (seeded). No band match here: the
    runner's own paths apply it, as at test time."""
    from hearsay.audio import load_audio, trim_silence

    x = trim_silence(load_audio(path))
    crop = float(np.random.default_rng(seed).choice(TEST_DURS))
    n = int(crop * SR)
    if x.size > n:
        off = int(np.random.default_rng(seed + 1).integers(0, x.size - n + 1))
        x = x[off : off + n]
    return np.ascontiguousarray(x, dtype=np.float32)


TEST_DURS: np.ndarray = np.array([3.4])
THRESHOLDS_FROM_EXPORTS = True  # tests switch this off (no exports in a hermetic run)


def score_all(models, consts, x: np.ndarray) -> dict:
    from hearsay.detectors.base import ClipContext

    out = {"m1b_v3": models.m1_logit(x), "spectra_aasist": models.spectra_logit(x),
           "m5_xlsr_ft": models.m5_logit(x)}  # fmt: skip
    r = models.handcrafted.run(ClipContext.from_array(x))
    out["handcrafted_v5"] = float(r.features.get("hc_logit", float("nan"))) if r.status == "ok" else float("nan")
    fo = consts.fuse(out)
    out["fused_base"] = float(fo.detail["base"])
    out["fused"] = float(fo.fused)
    out["e_applied"] = float(bool(fo.detail["e_applied"]))
    return out


def _lambda_mod():
    spec = importlib.util.spec_from_file_location("channel_lambda", REPO / "scripts" / "channel_lambda.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def requested_keys(rows: pd.DataFrame, kinds: tuple[str, ...]) -> set[str]:
    return {f"{p}|{k}|{s}" for p, s in zip(rows.path, rows.seed) for k in kinds}


def select_requested(cache: pd.DataFrame, rows: pd.DataFrame, kinds: tuple[str, ...]) -> pd.DataFrame:
    """Only the cached rows this cohort asked for (a rerun with another --n reassigns seeds, and
    stale rows must never be pivoted in), one per key."""
    return cache[cache.key.isin(requested_keys(rows, kinds))].drop_duplicates("key", keep="last")


def run_scoring(rows: pd.DataFrame, kinds: tuple[str, ...], name: str, threads: int) -> pd.DataFrame:
    """rows: path, label, seed. Resumable cache keyed path|kind|seed; refuses a cache written by
    different models or constants."""
    global TEST_DURS
    from hearsay.pipeline import CONSTANTS_V2_PATH, FusionConstants, Models

    TEST_DURS = _test_durs()
    OUT.mkdir(parents=True, exist_ok=True)
    cache, meta_p = OUT / f"m3_{name}.csv", OUT / f"m3_{name}.meta.json"
    models = Models(device="cpu", load_m5=True, threads=threads)
    consts = FusionConstants.load(CONSTANTS_V2_PATH)
    ident = {"models": models.version(), "constants": str(CONSTANTS_V2_PATH.relative_to(REPO)),
             "constants_final": consts.final, "kinds": list(kinds)}  # fmt: skip
    _lambda_mod().check_cache_identity(meta_p, cache, ident)
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["key"])
    have = set(done.key)
    new, t0 = [], time.time()
    for i, r in enumerate(rows.itertuples(index=False)):
        try:
            base = segment(r.path, int(r.seed))
        except Exception as e:  # noqa: BLE001
            for k in kinds:
                new.append({"key": f"{r.path}|{k}|{r.seed}", "path": r.path, "kind": k, "label": r.label,
                            "error": repr(e)[:200]})  # fmt: skip
            continue
        for k in kinds:
            key = f"{r.path}|{k}|{r.seed}"
            if key in have:
                continue
            try:
                s = score_all(models, consts, perturb(base, k, int(r.seed)))
                new.append({"key": key, "path": r.path, "kind": k, "label": r.label, **s})
            except Exception as e:  # noqa: BLE001
                new.append({"key": key, "path": r.path, "kind": k, "label": r.label, "error": repr(e)[:200]})
        if (i + 1) % 25 == 0:
            pd.concat([done, pd.DataFrame(new)]).to_csv(cache, index=False)
            print(f"  {i + 1}/{len(rows)} clips, {time.time() - t0:.0f} s", flush=True)
    out = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
    out.to_csv(cache, index=False)
    return select_requested(out, rows, kinds)


def _test_durs() -> np.ndarray:
    from hearsay.embed import TEST_DURATIONS

    return pd.read_csv(TEST_DURATIONS).duration_s.to_numpy(float)


def holdout_rows(n: int) -> pd.DataFrame:
    f = pd.read_csv(REPO / "splits" / "nsa_folds.csv")
    h = f[f.fold == "holdout"]
    real = h[h.label == "bonafide"].groupby("source", group_keys=False).apply(
        lambda d: d.sample(n // 4, random_state=0))
    spoof = h[h.label == "spoof"].groupby("generator", group_keys=False).apply(
        lambda d: d.sample(n // 4, random_state=0))
    r = pd.concat([real, spoof], ignore_index=True)
    return pd.DataFrame({"path": r.path, "label": r.label, "seed": np.arange(len(r))})


def mlaad_rows(n: int) -> pd.DataFrame:
    m = pd.read_csv(REPO / "outputs" / "manifests" / "mlaad_en.csv")
    per = max(1, n // m.model_name.nunique())
    r = m.sample(frac=1.0, random_state=0).groupby("model_name").head(per)  # stratified, seeded
    r = r.sample(min(n, len(r)), random_state=0)
    return pd.DataFrame({"path": r.path, "label": "spoof", "seed": 10_000 + np.arange(len(r)),
                         "model_name": r.model_name.to_numpy()})  # fmt: skip


# ------------------------------------------------------------------ readouts


def perturb_readout(d: pd.DataFrame) -> dict:
    from hearsay.metrics import eer, min_cost

    d = d[d.get("error", pd.Series(index=d.index, dtype=object)).isna()] if "error" in d else d
    piv = {c: d.pivot_table(index="path", columns="kind", values=c, aggfunc="first")
           for c in (*MODELS, "fused_base", "fused", "e_applied")}  # fmt: skip
    lab = d.groupby("path").label.first()
    # A clip counts only if every detector, both fused scores and the step flag exist under
    # every perturbation: a missing M3 (or any) score must shrink n_paired, never vanish.
    need = (*MODELS, "fused_base", "fused", "e_applied")
    idx = piv["m1b_v3"].index
    for c in need:
        idx = idx.intersection(piv[c].index)
    paths = [p for p in idx
             if all(set(PERTURBATIONS) <= set(piv[c].columns)
                    and np.isfinite(piv[c].loc[p, list(PERTURBATIONS)].to_numpy(float)).all() for c in need)]  # fmt: skip
    y = (lab.loc[paths] == "spoof").to_numpy(int)
    res: dict = {"n_paired": len(paths), "n_real": int((y == 0).sum()), "n_spoof": int((y == 1).sum())}
    for c in (*MODELS, "fused_base", "fused"):
        m = piv[c].loc[paths]
        res[c] = {}
        for k in PERTURBATIONS:
            s = m[k].to_numpy(float)
            ok = np.isfinite(s)
            two = len(np.unique(y[ok])) == 2  # a one-class slice has no EER or minDCF
            res[c][k] = {"auc": round(auc(y, s), 4),
                         "min_dcf": round(min_cost(y[ok], s[ok]), 4) if two else float("nan"),
                         "eer": round(eer(y[ok], s[ok]), 4) if two else float("nan")}  # fmt: skip
            if k != "none":
                res[c][k]["d_auc"] = round(delta_auc(y, m["none"].to_numpy(float), s), 4)
                res[c][k]["spearman_vs_clean"] = round(float(pd.Series(s).corr(pd.Series(m["none"].to_numpy(float)),
                                                                                 method="spearman")), 4)  # fmt: skip
    thr = {c: inner_threshold(c) for c in MODELS} if THRESHOLDS_FROM_EXPORTS else {}
    res["at_inner_threshold"] = {
        c: {k: {"p_fa": round(float((piv[c].loc[paths][k].to_numpy()[y == 0] >= t).mean()), 4),
                "p_miss": round(float((piv[c].loc[paths][k].to_numpy()[y == 1] < t).mean()), 4)}
            for k in PERTURBATIONS} for c, t in thr.items()}  # fmt: skip
    ea = piv["e_applied"].loc[paths]
    res["e_applied_share"] = {k: {"real": round(float(ea[k].to_numpy()[y == 0].mean()), 4),
                                  "spoof": round(float(ea[k].to_numpy()[y == 1].mean()), 4)} for k in PERTURBATIONS}  # fmt: skip
    m3 = piv["spectra_aasist"].loc[paths]
    res["m3_below_suppress_share_spoof"] = {k: round(float((m3[k].to_numpy()[y == 1] < M3_SUPPRESS_BELOW).mean()), 4)
                                            for k in PERTURBATIONS}  # fmt: skip
    res["e_step_effect_min_dcf"] = {k: round(res["fused"][k]["min_dcf"] - res["fused_base"][k]["min_dcf"], 4)
                                    for k in PERTURBATIONS}  # fmt: skip
    res["mean_abs_d_auc"] = {c: round(float(np.mean([abs(res[c][k]["d_auc"]) for k in PERTURBATIONS[1:]])), 4)
                             for c in (*MODELS, "fused_base", "fused")}  # fmt: skip
    return res


def complete_rows(d: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Rows with no error and a finite value in every required column (NaN and +-inf dropped),
    so a missing score shrinks the count instead of being skipped by a mean."""
    d = d[d.error.isna()] if "error" in d else d
    ok = np.ones(len(d), bool)
    for c in cols:
        ok &= np.isfinite(pd.to_numeric(d[c], errors="coerce").to_numpy(float)) if c in d else False
    return d[ok]


def mlaad_readout(d: pd.DataFrame, rows: pd.DataFrame) -> dict:
    d = complete_rows(d, (*MODELS, "fused_base", "fused", "e_applied"))
    d = d.merge(rows[["path", "model_name"]], on="path", how="left")
    res: dict = {"n": len(d), "n_models": int(d.model_name.nunique())}
    for c in MODELS:
        thr = inner_threshold(c)
        res[c] = {"inner_threshold": round(thr, 4), "miss_rate": round(float((d[c] < thr).mean()), 4),
                  "mlaad_in_training": {"m5_xlsr_ft": "yes", "spectra_aasist": "unknown (undisclosed)"}.get(c, "no")}  # fmt: skip
    res["m3_below_suppress_share"] = round(float((d.spectra_aasist < M3_SUPPRESS_BELOW).mean()), 4)
    res["e_applied_share"] = round(float(d.e_applied.mean()), 4)
    worst = d.groupby("model_name").e_applied.mean().sort_values(ascending=False).head(8)
    res["e_applied_by_model_top"] = {k: round(float(v), 3) for k, v in worst.items()}
    return res


def agreement() -> dict:
    a = pd.read_csv(S / "m1b_v3.csv").set_index("path")
    b = pd.read_csv(S / "spectra_aasist.csv").set_index("path")
    j = a[["split", "logit"]].join(b[["logit"]], rsuffix="_m3", how="inner")
    h, t = j[j.split == "holdout"], j[j.split == "test"]
    return spearman_gap(h.logit.to_numpy(), h.logit_m3.to_numpy(), t.logit.to_numpy(), t.logit_m3.to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe", choices=["perturb", "mlaad", "agreement", "verdict"])
    ap.add_argument("--n", type=int)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--readout-only", action="store_true", help="recompute the readout from the cache")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    def scored(rows, kinds, name):
        if a.readout_only:  # never rewrites the identity; refuses a cache that has none or another one
            meta = OUT / f"m3_{name}.meta.json"
            if not meta.exists():
                raise SystemExit(f"{meta} missing: the cache has no model identity; rescore")
            stored = json.loads(meta.read_text())
            if stored.get("kinds") != list(kinds) or not stored.get("models"):
                raise SystemExit(f"{meta} does not match this probe's perturbations or has no models")
            print(f"readout-only: cache scored by {stored['models']}", flush=True)
            return select_requested(pd.read_csv(OUT / f"m3_{name}.csv"), rows, kinds)
        return run_scoring(rows, kinds, name, a.threads)

    if a.probe == "perturb":
        rows = holdout_rows(a.n or N_PERTURB)
        res = perturb_readout(scored(rows, PERTURBATIONS, "perturb"))
    elif a.probe == "mlaad":
        rows = mlaad_rows(a.n or 600)
        res = mlaad_readout(scored(rows, ("none",), "mlaad"), rows)
    elif a.probe == "agreement":
        res = agreement()
    else:
        res = verdict_from(json.loads((OUT / "m3_perturb.json").read_text()),
                           json.loads((OUT / "m3_mlaad.json").read_text()))  # fmt: skip
    res = _lambda_mod().json_safe(res)
    (OUT / f"m3_{a.probe}.json").write_text(json.dumps(res, indent=2, allow_nan=False))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
