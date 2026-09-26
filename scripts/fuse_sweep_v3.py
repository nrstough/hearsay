"""Post-draft gate sweep: the four pre-declared candidates against the shipped rule.

Implements docs/reports/2026-09-26_post-draft-manifest.md (141b254; clarifications b877760; P_wl
"with room" amendment ae9c4c4) as specified in docs/specs/2026-09-26_post-draft-gate-sweep.md.
Candidates T2 (second M3 tier), W4 (M5 weight 0.4; judged by its fresh-evidence bake-off), P_wl
(WavLM Large probe as a fourth column) and H_noise (handcrafted v6; withdrawn). A missing export is
NOT RUN; an export that fails validation is INVALID; neither stops the other candidates.

Always writes outputs/fusion/sweep_v3_report.json (with the sha256 of every input). Nothing under
models/ is written unless --write --ratified-by nathan --candidate NAME is given, the inputs hash
identically to the locked report, and NAME is that report's pick; then only
models/fusion_v3/constants.json (never overwritten; fusion_v2 is never touched).

Usage: uv run python scripts/fuse_sweep_v3.py [--write --ratified-by nathan --candidate NAME [--new-column-model DIR]]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from hearsay.metrics import cost_at, min_cost, sigmoid

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fs", REPO / "scripts" / "fuse_sweep.py")
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)

S = REPO / "outputs" / "detector_scores"
SHIPPED_TSV = REPO / "submissions" / "20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv"
SHIPPED_SHA_PREFIX = "fb783076"
FUSION_V2 = REPO / "models" / "fusion_v2" / "constants.json"
FUSION_V3 = REPO / "models" / "fusion_v3" / "constants.json"
PERTURB_CSV = REPO / "outputs" / "channel" / "m3_perturb.csv"
REPORT = REPO / "outputs" / "fusion" / "sweep_v3_report.json"
MANIFEST_COMMITS = ["141b254", "b877760", "ae9c4c4"]
PI = 0.3
PINNED_BELOW = 0.001
N_TEST, N_ITW = 1671, 3000

E_TIERS = ((-3.0, 0.5),)
T2_TIERS = ((-3.0, 0.5), (-6.0, 0.25))
CURRENT_W = {"m1b_v3": 0.6, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.2}
# manifest order; weights are accumulated in dict order (hearsay.pipeline's order for the shipped three)
CANDIDATES = {
    "T2": {"weights": CURRENT_W, "tiers": T2_TIERS, "new": None, "test": "gate"},
    "W4": {"weights": {"m1b_v3": 0.4, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.4}, "tiers": E_TIERS,
           "new": None, "test": "bakeoff"},
    "P_wl": {"weights": {"m1b_v3": 0.4, "wavlm_l": 0.2, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.2}, "tiers": E_TIERS,
             "new": ("wavlm_l", None), "test": "gate+room"},
    "H_noise": {"weights": {"m1b_v3": 0.6, "handcrafted_v6": 0.2, "m5_xlsr_ft": 0.2}, "tiers": E_TIERS,
                "new": ("handcrafted_v6", "_itw_handcrafted_v6.csv"), "test": "gate"},
}
EXPECTED_CURRENT = {"inner_brief": 0.1351, "holdout_brief": 0.0065, "holdout_averse": 0.0087, "itw_brief": 0.228,
                    "itw_averse": 0.2385, "itw_pfa_at_inner_thr": 0.013, "itw_pmiss_at_inner_thr": 0.122,
                    "holdout_argmin_FA_miss": [0, 13]}  # fmt: skip
EXPECTED_W4 = {"inner_brief": 0.1373, "inner_averse": 0.3366, "holdout_brief": 0.0065, "holdout_averse": 0.0088,
               "itw_brief": 0.1997, "itw_averse": 0.2075}  # fmt: skip
GATE = {"itw_brief_gain": 0.020, "itw_averse_reg": 0.002, "inner_brief_gain": 0.010, "holdout_reg": 0.010,
        "spearman_floor": 0.974}  # fmt: skip
STANDING = {"itw_brief_gain": 0.01, "itw_averse_reg": 0.01, "holdout_brief_reg": 0.05, "inner_brief_reg": 0.03}
ROOM = {"itw_brief_gain": 0.030, "itw_averse_gain": 0.030, "catches": 19}
PWL_MIN_CATCHES, CORRECTIVE_MIN = 16, 0.5
HNOISE = {"noise_auc_min": 0.90, "clean_auc_v5": 0.998, "clean_tol": 0.005}  # the CPU chat's pre-declared bar
BOOT_N, BOOT_SEED, PERTURB_REG, TRIPWIRE = 2000, 0, 0.010, 1e-9
POPS, COSTS = ("inner", "holdout", "itw"), ("brief", "averse")
CELLS = [f"{p}_{c}" for p in POPS for c in COSTS]
EPS = 1e-12  # absorbs float representation in comparisons of 4-decimal values


# ---------------------------------------------------------------- costs and fusion (pure)
def brief(y, s) -> float:
    """The brief's cost: pi_synth 0.3, C_FA 4, C_miss 1 (normalized 9.33*P_FA + P_miss)."""
    return min_cost(y, s, 0.3)


def averse(y, s) -> float:
    """The sponsor code's convention: pi 0.5, C_FA 1, C_miss 4 (normalized P_FA + 4*P_miss)."""
    return min_cost(y, s, 0.5, c_fa=1.0, c_miss=4.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_ok(path: Path, prefix: str = SHIPPED_SHA_PREFIX) -> bool:
    return sha256(path).startswith(prefix)


def rank_vs(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(ref_sorted, x) / len(ref_sorted)


def blend(ranks: dict, weights: dict) -> np.ndarray:
    b = 0.0
    for k, w in weights.items():  # accumulated in weights order, as hearsay.pipeline does
        b = b + w * ranks[k]
    return np.asarray(b, dtype=np.float64)


def apply_tiers(base: np.ndarray, m3: np.ndarray, tiers) -> np.ndarray:
    """Multiply base by the deepest satisfied tier's TOTAL multiplier when base > 0.5 and the M3
    logit is strictly below the tier threshold. Tiers must be ordered shallow to deep."""
    thrs = [t for t, _ in tiers]
    assert all(a > b for a, b in itertools.pairwise(thrs)), "tiers must have strictly decreasing thresholds"
    base, m3 = np.asarray(base, dtype=np.float64), np.asarray(m3, dtype=np.float64)
    mult = np.ones_like(base)
    for thr, m in tiers:
        mult = np.where((m3 < thr) & (base > 0.5), m, mult)
    return base * mult


def inner_threshold(y, s) -> float:
    """Brief-cost threshold on inner OOF (s >= t called synthetic); includes t = inf (all real)."""
    y, s = np.asarray(y), np.asarray(s)
    cands = np.r_[np.unique(s), np.inf]
    return float(min(cands, key=lambda t: cost_at(y, s, t, PI)))


def readout(y: dict, src: dict, sc: dict) -> dict:
    r = {"inner_brief": round(brief(y["inner_oof"], sc["inner_oof"]), 4),
         "inner_averse": round(averse(y["inner_oof"], sc["inner_oof"]), 4),
         "holdout_brief": round(brief(y["holdout"], sc["holdout"]), 4),
         "holdout_averse": round(averse(y["holdout"], sc["holdout"]), 4)}  # fmt: skip
    hs, yh = sc["holdout"], y["holdout"]
    t = min(np.unique(hs), key=lambda t_: cost_at(yh, hs, t_, PI))
    r["holdout_argmin_FA_miss"] = [int(np.sum(hs[yh == 0] >= t)), int(np.sum(hs[yh == 1] < t))]
    spoof = hs[yh == 1]
    for s_ in ("ljspeech", "librispeech"):
        b = hs[(yh == 0) & (src["holdout"] == s_)]
        r[f"holdout_{s_}"] = round(brief(np.r_[np.zeros(len(b)), np.ones(len(spoof))], np.r_[b, spoof]), 4)
    r["itw_brief"] = round(brief(y["itw"], sc["itw"]), 4)
    r["itw_averse"] = round(averse(y["itw"], sc["itw"]), 4)
    ti = inner_threshold(y["inner_oof"], sc["inner_oof"])
    r["itw_pfa_at_inner_thr"] = round(float(np.mean(sc["itw"][y["itw"] == 0] >= ti)), 4)
    r["itw_pmiss_at_inner_thr"] = round(float(np.mean(sc["itw"][y["itw"] == 1] < ti)), 4)
    return r


def six(r: dict) -> dict:
    return {c: r[c] for c in CELLS}


def deltas(cur: dict, cand: dict) -> dict:
    """CURRENT − candidate per cell, on 4-decimal values; > 0 means the candidate is better."""
    return {c: round(round(cur[c], 4) - round(cand[c], 4), 4) for c in CELLS}


# ---------------------------------------------------------------- the gate (pure)
def gate(d: dict, spearman: float, diag_ok: bool) -> dict:
    if not np.isfinite(spearman):
        raise ValueError("non-finite Spearman")
    doubled = spearman < GATE["spearman_floor"]
    k = 2.0 if doubled else 1.0
    rules = {
        "2a_itw_brief_gain": d["itw_brief"] >= k * GATE["itw_brief_gain"] - EPS,
        "2b_itw_averse_reg": d["itw_averse"] >= -GATE["itw_averse_reg"] - EPS,
        "3a_inner_brief_gain": d["inner_brief"] >= k * GATE["inner_brief_gain"] - EPS,
        "3b_inner_averse_no_reg": d["inner_averse"] >= -EPS,
        "4a_holdout_brief": d["holdout_brief"] >= -GATE["holdout_reg"] - EPS,
        "4b_holdout_averse": d["holdout_averse"] >= -GATE["holdout_reg"] - EPS,
        "5_brief_2of3": sum(d[f"{p}_brief"] > EPS for p in POPS) >= 2,
        "5_averse_2of3": sum(d[f"{p}_averse"] > EPS for p in POPS) >= 2,
        "5_diagnostic": bool(diag_ok),
    }
    failed = [r for r, ok in rules.items() if not ok]
    return {"rules": rules, "doubled": bool(doubled), "failed": failed, "verdict": "FAIL" if failed else "PASS"}


def room(d: dict, catches: int | None) -> dict:
    """P_wl's 'with room' amendment (ae9c4c4), on top of the gate."""
    rules = {"itw_brief_gain_0.030": d["itw_brief"] >= ROOM["itw_brief_gain"] - EPS,
             "itw_averse_gain_0.030": d["itw_averse"] >= ROOM["itw_averse_gain"] - EPS,
             "catches_19": catches is not None and catches >= ROOM["catches"]}  # fmt: skip
    failed = [r for r, ok in rules.items() if not ok]
    return {"rules": rules, "failed": failed, "verdict": "FAIL" if failed else "PASS"}


def standing_rule(d: dict) -> dict:
    rules = {"itw_brief_gain_0.01": d["itw_brief"] >= STANDING["itw_brief_gain"] - EPS,
             "itw_averse_reg_0.01": d["itw_averse"] >= -STANDING["itw_averse_reg"] - EPS,
             "holdout_brief_reg_0.05": d["holdout_brief"] >= -STANDING["holdout_brief_reg"] - EPS,
             "inner_brief_reg_0.03": d["inner_brief"] >= -STANDING["inner_brief_reg"] - EPS}  # fmt: skip
    return {"rules": rules, "ok": all(rules.values())}


def pareto_pick(cells: dict[str, dict]) -> str | None:
    """cells: passer -> six-cell costs on a common row set. The passer <= every other in all six and
    < in at least one; a lone passer is picked; otherwise None."""
    if not cells:
        return None
    if len(cells) == 1:
        return next(iter(cells))
    for a, ca in cells.items():
        if all(all(ca[c] <= cb[c] + EPS for c in CELLS) and any(ca[c] < cb[c] - EPS for c in CELLS)
               for b, cb in cells.items() if b != a):  # fmt: skip
            return a
    return None


# ---------------------------------------------------------------- diagnostics (pure)
def t2_fires(y, base, m3) -> dict:
    y, base, m3 = np.asarray(y), np.asarray(base), np.asarray(m3)
    hit = (m3 < -6.0) & (base > 0.5)
    return {"spoof": int(np.sum(hit & (y == 1))), "bonafide": int(np.sum(hit & (y == 0)))}


def new_catches(y, cur, thr_cur, col, thr_col) -> int:
    y, cur, col = np.asarray(y), np.asarray(cur), np.asarray(col)
    miss = (y == 1) & (cur < thr_cur)
    return int(np.sum(miss & (col >= thr_col)))


def corrective_share(y, cur, thr_cur, r_new, r_m1b) -> float:
    y, cur, r_new, r_m1b = map(np.asarray, (y, cur, r_new, r_m1b))
    fa, miss = (y == 0) & (cur >= thr_cur), (y == 1) & (cur < thr_cur)
    n = int(fa.sum() + miss.sum())
    if n == 0:
        return float("nan")
    fixed = np.sum(fa & (r_new < r_m1b)) + np.sum(miss & (r_new > r_m1b))
    return float(fixed / n)


def cluster_bootstrap(y, s_cur, s_cand, groups, n=BOOT_N, seed=BOOT_SEED, return_draws=False) -> dict:
    """Paired speaker-cluster bootstrap of Δ = cost(CURRENT) − cost(candidate) under both costs."""
    y, s_cur, s_cand = np.asarray(y), np.asarray(s_cur, float), np.asarray(s_cand, float)
    g = pd.Series(groups)
    if g.isna().any() or (g.astype(str).str.strip() == "").any():
        raise ValueError("bootstrap: missing speaker/group values")
    ids = pd.factorize(g)[0]
    members = [np.flatnonzero(ids == k) for k in range(ids.max() + 1)]
    rng = np.random.default_rng(seed)
    out = {"brief": [], "averse": []}
    draws, skipped = [], 0
    for _ in range(n):
        pick = rng.choice(len(members), len(members), replace=True)
        rows = np.concatenate([members[k] for k in pick])
        draws.append(pick)
        yy = y[rows]
        if yy.min() == yy.max():
            skipped += 1
            continue
        for name, f in (("brief", brief), ("averse", averse)):
            out[name].append(f(yy, s_cur[rows]) - f(yy, s_cand[rows]))
    res = {"brief_p5": float(np.percentile(out["brief"], 5)), "averse_p5": float(np.percentile(out["averse"], 5)),
           "n": n, "skipped_single_class": skipped, "n_groups": len(members)}  # fmt: skip
    if return_draws:
        res["draws"] = draws
    return res


def bakeoff_verdict(standing: dict, perturb_deltas: dict | None, tripwire: float | None, boot: dict | None) -> dict:
    parts = {"1_standing_rule": bool(standing["ok"])}
    if perturb_deltas is None or tripwire is None or not tripwire <= TRIPWIRE:
        return {"parts": parts, "verdict": "INVALID",
                "reason": f"perturbation tripwire {tripwire} > {TRIPWIRE} or perturbation cells missing"}
    parts["2_perturbation"] = all(v >= -PERTURB_REG - EPS for v in perturb_deltas.values())
    parts["3_itw_bootstrap"] = bool(boot and boot["brief_p5"] > 0 and boot["averse_p5"] > 0)
    failed = [k for k, v in parts.items() if not v]
    return {"parts": parts, "failed": failed, "verdict": "FAIL" if failed else "PASS"}


def fit_platt(s_inner, y_inner) -> tuple[float, float]:
    pl = LogisticRegression(class_weight="balanced").fit(np.asarray(s_inner)[:, None], y_inner)
    return float(pl.coef_[0, 0]), float(pl.intercept_[0])


def to_p(fused, a: float, b: float) -> np.ndarray:
    return 0.001 + 0.999 * sigmoid(a * np.asarray(fused) + b + math.log(PI / (1 - PI)))


def test_agreement(test_paths, fused, platt: tuple[float, float], shipped: pd.Series) -> dict:
    """shipped: cm-score indexed by filename. Requires complete, unique coverage of the shipped files;
    returns counts only (never file names)."""
    names = pd.Index([Path(p).name for p in test_paths])
    if names.has_duplicates:
        raise ValueError("test agreement: duplicate test basenames")
    if len(names) != len(shipped) or not names.isin(shipped.index).all():
        raise ValueError(f"test agreement: coverage {len(names)} of {len(shipped)} shipped files, or unmatched names")
    sh = shipped.loc[names].to_numpy()
    keep = sh >= PINNED_BELOW
    fused = np.asarray(fused)[keep]
    rho = float(spearmanr(fused, sh[keep]).statistic)
    p = to_p(fused, *platt)
    return {"spearman": rho, "crossings": int(np.sum((p >= 0.5) != (sh[keep] >= 0.5))),
            "n_nonpinned": int(keep.sum()), "n_pinned": int((~keep).sum())}  # fmt: skip


def validate_export(raw: pd.DataFrame, folds: pd.DataFrame, test_names: set, itw_paths: set) -> list[str]:
    """raw: the export as fs.load would assemble it, BEFORE dedup (normalized paths). folds: path -> fold."""
    probs = []
    if raw.path.duplicated().any():
        probs.append(f"{int(raw.path.duplicated().sum())} duplicate paths")
    if not np.isfinite(raw.logit.to_numpy(dtype=float)).all():
        probs.append("non-finite logits")
    for split, ok_fold in (("inner_oof", lambda f: f in {"0", "1", "2", "3", "4"}), ("holdout", lambda f: f == "holdout")):
        rows = raw[raw.split == split]
        f = rows.path.map(folds)
        if f.isna().any():
            probs.append(f"{split}: {int(f.isna().sum())} rows not in the fold file")
        bad = ~f.dropna().map(ok_fold)
        if bad.any():
            probs.append(f"{split}: {int(bad.sum())} rows whose fold disagrees with splits/nsa_folds.csv")
    tn = raw[raw.split == "test"].path.map(lambda p: Path(p).name)
    if tn.duplicated().any() or set(tn) != test_names:
        probs.append(f"test coverage {tn.nunique()} unique of {len(test_names)}")
    iw = set(raw[raw.split == "itw"].path)
    if iw != itw_paths:
        probs.append(f"itw coverage {len(iw & itw_paths)} of {len(itw_paths)}")
    return probs


def write_constants(path: Path, payload: dict) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------- data path
def load_raw(name: str, itw: str | None) -> pd.DataFrame:
    """fs.load without the dedup, so validation sees duplicates."""
    d = pd.read_csv(S / f"{name}.csv")
    missing = {"path", "split", "logit"} - set(d.columns)
    if missing:
        raise ValueError(f"{name}.csv lacks columns {sorted(missing)}")
    d["split"] = d.split.replace({"stress": "itw"})
    if itw:
        x = pd.read_csv(S / itw).assign(split="itw")
        d = pd.concat([d[d.split != "itw"], x[["path", "split", "logit"]]], ignore_index=True)
    d["path"] = d.path.map(fs.norm)
    return d


def resolve_itw(col: str, itw: str | None) -> str | None:
    """The export's separate ITW file: the declared one, else _itw_<col>.csv if present, else None (in-file)."""
    if itw is None and (S / f"_itw_{col}.csv").exists():
        return f"_itw_{col}.csv"
    return itw


def load_labels() -> tuple[pd.DataFrame, pd.Series]:
    lab = pd.read_csv(REPO / "splits" / "nsa_folds.csv", dtype={"fold": str})
    itwm = pd.read_csv(REPO / "outputs" / "manifests" / "itw_stress.csv")
    labels = pd.concat([lab[["path", "label", "source", "speaker"]], itwm[["path", "label", "source", "speaker"]]])
    labels["path"] = labels.path.map(fs.norm)
    folds = lab.assign(path=lab.path.map(fs.norm)).set_index("path").fold.astype(str)
    return labels.set_index("path"), folds


def build(cols: dict[str, pd.DataFrame], m3: pd.DataFrame, labels: pd.DataFrame) -> dict:
    idx = {}
    for s in ("inner_oof", "holdout", "test", "itw"):
        ix = m3.index[m3.split == s]
        for d in cols.values():
            ix = ix.intersection(d.index[d.split == s])
        idx[s] = ix
    ctx = {"idx": idx, "cols": list(cols)}
    ctx["y"] = {s: (labels.loc[ix, "label"] == "spoof").to_numpy(int) for s, ix in idx.items() if s != "test"}
    ctx["src"] = {s: labels.loc[ix, "source"].to_numpy() for s, ix in idx.items() if s != "test"}
    ctx["spk"] = labels.loc[idx["itw"], "speaker"].to_numpy()
    ctx["m3"] = {s: m3.loc[ix, "logit"].to_numpy(float) for s, ix in idx.items()}
    ctx["logit"] = {n: {s: d.loc[ix, "logit"].to_numpy(float) for s, ix in idx.items()} for n, d in cols.items()}
    ctx["ref"] = {n: np.sort(ctx["logit"][n]["inner_oof"]) for n in cols}
    ctx["rank"] = {n: {s: rank_vs(ctx["ref"][n], v) for s, v in ctx["logit"][n].items()} for n in cols}
    ctx["rows"] = {s: len(ix) for s, ix in idx.items()}
    return ctx


def score(ctx: dict, weights: dict, tiers) -> tuple[dict, dict]:
    base = {s: blend({n: ctx["rank"][n][s] for n in weights}, weights) for s in ctx["idx"]}
    return base, {s: apply_tiers(base[s], ctx["m3"][s], tiers) for s in ctx["idx"]}


def rule_eval(ctx: dict, weights: dict, tiers, shipped: pd.Series) -> dict:
    base, sc = score(ctx, weights, tiers)
    r = readout(ctx["y"], ctx["src"], sc)
    platt = fit_platt(sc["inner_oof"], ctx["y"]["inner_oof"])
    ta = test_agreement(ctx["idx"]["test"], sc["test"], platt, shipped)
    return {"base": base, "sc": sc, "readout": r, "platt": platt, "test": ta}


def perturb_eval(df: pd.DataFrame, consts: dict, rules: dict) -> dict:
    """rules: name -> (weights over the shipped three, tiers). Ranks from the shipped inner-OOF references."""
    ref = {k: np.asarray(v) for k, v in consts["rank_ref_inner_oof_sorted"].items()}
    ranks = {k: rank_vs(ref[k], df[k].to_numpy(float)) for k in ref}
    scores = {n: apply_tiers(blend(ranks, w), df["spectra_aasist"].to_numpy(float), t) for n, (w, t) in rules.items()}
    y = (df.label == "spoof").to_numpy(int)
    cells = {n: {f"{k}_{c}": round(f(y[(df.kind == k).to_numpy()], s[(df.kind == k).to_numpy()]), 4)
                 for k in df.kind.unique() for c, f in (("brief", brief), ("averse", averse))}
             for n, s in scores.items()}  # fmt: skip
    trip = float(np.max(np.abs(scores["CURRENT"] - df["fused"].to_numpy(float)))) if "CURRENT" in scores else None
    return {"cells": cells, "tripwire": trip, "scores": scores, "y": y}


def evaluate(hnoise_evidence: tuple[float, float] | None = None) -> dict:
    """Everything the report holds; no writes."""
    shipped_df = pd.read_csv(SHIPPED_TSV, sep="\t")
    shipped = shipped_df.set_index("filename")["cm-score"]
    labels, folds = load_labels()
    base_cols = {"m1b_v3": fs.load("m1b_v3"),
                 "handcrafted_v5": fs.load("handcrafted_v5", str(S / "_itw_handcrafted_v5.csv")),
                 "m5_xlsr_ft": fs.load("m5_xlsr_ft", str(S / "_itw_m5.csv"))}  # fmt: skip
    m3 = fs.load("spectra_aasist", str(REPO / "outputs/spectra/itw_stress/scores.csv"))
    inputs = {str(p.relative_to(REPO)): sha256(p) for p in (
        SHIPPED_TSV, FUSION_V2, PERTURB_CSV, S / "m1b_v3.csv", S / "handcrafted_v5.csv", S / "_itw_handcrafted_v5.csv",
        S / "m5_xlsr_ft.csv", S / "_itw_m5.csv", S / "spectra_aasist.csv", REPO / "outputs/spectra/itw_stress/scores.csv",
        REPO / "splits/nsa_folds.csv", REPO / "outputs/manifests/itw_stress.csv")}  # fmt: skip
    base = build(base_cols, m3, labels)
    cur = rule_eval(base, CURRENT_W, E_TIERS, shipped)
    sc_ok = {k: cur["readout"][k] == v for k, v in EXPECTED_CURRENT.items()}
    self_check = {"expected": EXPECTED_CURRENT, "got": {k: cur["readout"][k] for k in EXPECTED_CURRENT},
                  "fields_ok": sc_ok, "spearman": cur["test"]["spearman"],
                  "ok": all(sc_ok.values()) and cur["test"]["spearman"] >= 0.999}  # fmt: skip
    rep = {"manifest_commits": MANIFEST_COMMITS, "inputs": inputs, "rows_base": base["rows"], "self_check": self_check,
           "current": {"readout": cur["readout"], "test": cur["test"], "platt": cur["platt"]}, "candidates": {}}
    if not self_check["ok"]:
        return finalize(rep, self_check_ok=False)

    perturb = pd.read_csv(PERTURB_CSV)
    consts = json.loads(FUSION_V2.read_text())
    world = {"base": base, "base_cols": base_cols, "m3": m3, "labels": labels, "folds": folds, "shipped": shipped,
             "perturb": perturb, "consts": consts, "test_names": set(shipped.index), "itw_paths": set(base["idx"]["itw"]),
             "inputs": inputs, "hnoise_evidence": hnoise_evidence}  # fmt: skip
    rep["candidates"] = run_candidates(world)
    return finalize(rep, self_check_ok=True, ctxs={"base_cols": base_cols, "m3": m3, "labels": labels,
                                                   "shipped": shipped})  # fmt: skip


def run_candidates(world: dict) -> dict:
    """Every candidate in manifest order; any exception is that candidate's INVALID, never the run's end."""
    out = {}
    for name, spec in CANDIDATES.items():
        try:
            out[name] = eval_candidate(name, spec, **world)
        except Exception as e:  # noqa: BLE001 - a broken export must not kill the locked run
            out[name] = {"status": "INVALID", "reason": f"{type(e).__name__}: {e}"}
    return out


def hnoise_diag(evidence: tuple[float, float] | None) -> dict:
    """H_noise mechanism diagnostic (manifest): 20 dB noise AUC >= 0.90 on the channel cohort and clean AUC within
    0.005 of v5's 0.998, read from the CPU chat's report (the check needs audio). No evidence fails."""
    if evidence is None:
        return {"ok": False, "reason": "no noise-AUC evidence supplied (--hnoise-evidence NOISE_AUC,CLEAN_AUC)"}
    noise, clean_ = (float(v) for v in evidence)
    ok = bool(np.isfinite(noise) and np.isfinite(clean_) and noise >= HNOISE["noise_auc_min"] - EPS
              and abs(clean_ - HNOISE["clean_auc_v5"]) <= HNOISE["clean_tol"] + EPS)  # fmt: skip
    return {"ok": ok, "noise_auc": noise, "clean_auc": clean_, "bar": HNOISE}


def pwl_diag_ok(n_catch: int, corr: dict) -> bool:
    """P_wl mechanism diagnostic (manifest): >= 16 new catches AND corrective share > 0.5 on inner and ITW."""
    return bool(n_catch >= PWL_MIN_CATCHES and all(np.isfinite(v) and v > CORRECTIVE_MIN for v in corr.values()))


def eval_candidate(name, spec, base, base_cols, m3, labels, folds, shipped, perturb, consts, test_names, itw_paths,
                   inputs, hnoise_evidence=None) -> dict:  # fmt: skip
    out = {"test": spec["test"], "weights": spec["weights"], "tiers": [list(t) for t in spec["tiers"]]}
    ctx, cols = base, dict(base_cols)
    if spec["new"]:
        col, itw = spec["new"]
        if not (S / f"{col}.csv").exists():
            return {**out, "status": "NOT RUN", "reason": f"{col}.csv absent"}
        itw = resolve_itw(col, itw)
        if itw and not (S / itw).exists():
            return {**out, "status": "INVALID", "reason": f"{col}.csv present but its ITW file {itw} is absent"}
        raw = load_raw(col, itw)
        inputs[f"outputs/detector_scores/{col}.csv"] = sha256(S / f"{col}.csv")
        if itw:
            inputs[f"outputs/detector_scores/{itw}"] = sha256(S / itw)
        probs = validate_export(raw, folds, test_names, itw_paths)
        if probs:
            return {**out, "status": "INVALID", "reason": "; ".join(probs)}
        d = raw.drop_duplicates("path").set_index("path")
        cols[col] = d  # union: CURRENT's three plus the new column, so CURRENT is rebuilt on identical rows
        ctx = build(cols, m3, labels)
        auc = roc_auc_score(ctx["y"]["inner_oof"], ctx["logit"][col]["inner_oof"])
        out["new_column_inner_auc"] = round(float(auc), 4)
        if auc < 0.5:
            return {**out, "status": "INVALID", "reason": f"{col} inner-OOF AUC {auc:.4f} < 0.5 (direction inverted?)"}
    out["rows"] = ctx["rows"]
    out["rows_shrunk_vs_base"] = ctx["rows"] != base["rows"]
    cur = rule_eval(ctx, CURRENT_W, E_TIERS, shipped)
    cand = rule_eval(ctx, spec["weights"], spec["tiers"], shipped)
    d = deltas(cur["readout"], cand["readout"])
    for k in ("spearman",):
        if not np.isfinite(cand["test"][k]):
            return {**out, "status": "INVALID", "reason": "non-finite Spearman"}
    out.update({"readout": cand["readout"], "current_readout": cur["readout"], "deltas": d, "test_agreement": cand["test"],
                "platt": cand["platt"]})  # fmt: skip

    # mechanism diagnostics
    if name == "T2":
        fires = {s: t2_fires(ctx["y"][s], cand["base"][s], ctx["m3"][s]) for s in ("inner_oof", "holdout", "itw")}
        py = (perturb.label == "spoof").to_numpy(int)
        for k in perturb.kind.unique():
            m = (perturb.kind == k).to_numpy()
            fires[f"perturb_{k}"] = t2_fires(py[m], perturb.fused_base.to_numpy(float)[m],
                                             perturb.spectra_aasist.to_numpy(float)[m])  # fmt: skip
        diag_ok = all(v["spoof"] == 0 for v in fires.values())
        out["diagnostic"] = {"deeper_tier_fires": fires, "ok": diag_ok}
        out["known_number_check"] = {"six_identical_to_current": six(cand["readout"]) == six(cur["readout"])}
    elif name == "P_wl":
        col = spec["new"][0]
        thr_cur = inner_threshold(ctx["y"]["inner_oof"], cur["sc"]["inner_oof"])
        n_miss = int(np.sum((ctx["y"]["itw"] == 1) & (cur["sc"]["itw"] < thr_cur)))

        def catches(c, logit=None):
            lg = ctx["logit"][c] if logit is None else logit
            t = inner_threshold(ctx["y"]["inner_oof"], lg["inner_oof"])
            return new_catches(ctx["y"]["itw"], cur["sc"]["itw"], thr_cur, lg["itw"], t)

        m3l = {s: ctx["m3"][s] for s in ctx["idx"]}
        scale = {c: catches(c) for c in ("m1b_v3", "m5_xlsr_ft")}
        scale["spectra_aasist"] = catches(None, m3l)
        n_catch = catches(col)
        corr = {s: corrective_share(ctx["y"][s], cur["sc"][s], thr_cur, ctx["rank"][col][s], ctx["rank"]["m1b_v3"][s])
                for s in ("inner_oof", "itw")}  # fmt: skip
        if not all(np.isfinite(v) for v in corr.values()):
            return {**out, "status": "INVALID", "reason": f"non-finite corrective share {corr}"}
        diag_ok = pwl_diag_ok(n_catch, corr)
        rho_m1b = {s: round(float(spearmanr(ctx["logit"][col][s], ctx["logit"]["m1b_v3"][s]).statistic), 4)
                   for s in ("holdout", "test")}  # fmt: skip
        out["diagnostic"] = {"itw_misses_at_current_inner_thr": n_miss, "new_catches": n_catch,
                             "catches_scale": scale, "corrective_share": corr, "spearman_vs_m1b": rho_m1b,
                             "ok": bool(diag_ok)}  # fmt: skip
        out["room"] = room(d, n_catch)
    elif name == "H_noise":
        out["diagnostic"] = hnoise_diag(hnoise_evidence)
        diag_ok = out["diagnostic"]["ok"]
    else:  # W4: governed by its bake-off; the gate is recorded for the record only (no diagnostic defined)
        diag_ok = True
        out["diagnostic"] = {"ok": True, "reason": "none under the gate; W4 is governed by its bake-off"}
        out["known_number_check"] = {k: cand["readout"][k] == v for k, v in EXPECTED_W4.items()}

    out["gate"] = {**gate(d, cand["test"]["spearman"], diag_ok), "governing": name != "W4"}
    if name == "W4":
        st = standing_rule(d)
        pe = perturb_eval(perturb, consts, {"CURRENT": (CURRENT_W, E_TIERS), "W4": (spec["weights"], E_TIERS)})
        pdel = {k: round(pe["cells"]["CURRENT"][k] - pe["cells"]["W4"][k], 4) for k in pe["cells"]["CURRENT"]}
        boot = cluster_bootstrap(ctx["y"]["itw"], cur["sc"]["itw"], cand["sc"]["itw"], ctx["spk"])
        bv = bakeoff_verdict(st, pdel, pe["tripwire"], boot)
        out["bakeoff"] = {"standing_rule": st, "perturbation_cells": pe["cells"], "perturbation_deltas": pdel,
                          "tripwire_max_abs": pe["tripwire"], "bootstrap": boot, **bv}  # fmt: skip
        out["status"] = {"PASS": "BAKEOFF PASS", "FAIL": "BAKEOFF FAIL"}.get(bv["verdict"], "INVALID")
        out["ratifiable"] = bv["verdict"] == "PASS"
    elif name == "P_wl":
        g, rm = out["gate"]["verdict"], out["room"]["verdict"]
        out["status"] = "PASS WITH ROOM" if g == rm == "PASS" else ("GATE PASS, NO ROOM" if g == "PASS" else "FAIL")
        out["ratifiable"] = g == rm == "PASS"
    else:
        out["status"] = out["gate"]["verdict"]
        out["ratifiable"] = out["gate"]["verdict"] == "PASS"
    return out


def finalize(rep: dict, self_check_ok: bool, ctxs: dict | None = None) -> dict:
    if not self_check_ok:
        for name in CANDIDATES:
            rep["candidates"][name] = {"status": "INVALID", "reason": "self-check failed"}
        rep["decision"] = {"passers": [], "pick": None, "line": "SELF-CHECK FAIL: no candidate read"}
        return rep
    passers = [n for n, c in rep["candidates"].items() if c.get("ratifiable")]
    common = {}
    if len(passers) >= 2 and ctxs:  # rule 7 on the intersection of every passer's row set
        try:
            common = rule7_common(passers, ctxs, rep)
        except Exception as e:  # noqa: BLE001 - fall back to KEEP, never crash the locked run
            rep["rule7_error"] = f"{type(e).__name__}: {e}"
            common = {}
    elif passers:
        common = {passers[0]: six(rep["candidates"][passers[0]]["readout"])}
    pick = pareto_pick(common)
    if not passers:
        line = "KEEP (no candidate passes; submissions/CrossExam_predictions.tsv stays final)"
    elif pick:
        line = f"RECOMMEND {pick} for Nathan's ratification"
    elif "rule7_error" in rep:
        line = "KEEP (two or more pass; the rule-7 comparison failed, see rule7_error)"
    else:
        line = "KEEP (two or more pass, none Pareto-dominant)"
    rep["decision"] = {"passers": passers, "pick": pick, "line": line}
    return rep


def rule7_common(passers: list, ctxs: dict, rep: dict) -> dict:
    """Every passer re-scored on the intersection of all passers' row sets (clarification 5)."""
    cols = dict(ctxs["base_cols"])
    for n in passers:
        if CANDIDATES[n]["new"]:
            col, itw = CANDIDATES[n]["new"]
            cols[col] = load_raw(col, resolve_itw(col, itw)).drop_duplicates("path").set_index("path")
    cctx = build(cols, ctxs["m3"], ctxs["labels"])
    common = {}
    for n in passers:
        _, sc = score(cctx, CANDIDATES[n]["weights"], CANDIDATES[n]["tiers"])
        common[n] = six(readout(cctx["y"], cctx["src"], sc))
    rep["rule7_common_rows"] = {"rows": cctx["rows"], "cells": common}
    return common


def print_table(rep: dict) -> None:
    rows = {"CURRENT (base)": {**six(rep["current"]["readout"]), "spearman": rep["current"]["test"]["spearman"],
                               "crossings": rep["current"]["test"]["crossings"], "status": "shipped"}}  # fmt: skip
    for n, c in rep["candidates"].items():
        if "readout" in c:
            rows[n] = {**six(c["readout"]), "spearman": round(c["test_agreement"]["spearman"], 4),
                       "crossings": c["test_agreement"]["crossings"], "status": c["status"]}  # fmt: skip
        else:
            rows[n] = {"status": f"{c['status']}: {c.get('reason', '')}"}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(pd.DataFrame(rows).T.to_string())


def clean(x):
    """JSON-safe: numpy scalars to Python, non-finite floats to None (never bare NaN in the report)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.bool_ | bool):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return float(x) if np.isfinite(x) else None
    return x


def precheck_write(args) -> dict:
    """Cheap refusals before any evaluation: the locked report must exist, hash to --report-sha256, name
    --candidate as its pick; a new-column candidate needs --new-column-model DIR/meta.json, others refuse it."""
    if not REPORT.exists():
        sys.exit("no locked report; run without --write first")
    if sha256(REPORT) != args.report_sha256:
        sys.exit("the report on disk is not the locked one (sha256 differs from --report-sha256); refuse")
    locked = json.loads(REPORT.read_text())
    if locked["decision"]["pick"] != args.candidate:
        sys.exit(f"--candidate {args.candidate} is not the locked pick ({locked['decision']['pick']}); refuse")
    if CANDIDATES[args.candidate]["new"]:
        if not args.new_column_model or not (Path(args.new_column_model) / "meta.json").exists():
            sys.exit("--new-column-model DIR with a meta.json is required for a new column; refuse")
    elif args.new_column_model:
        sys.exit(f"--new-column-model is not used by {args.candidate}; refuse")
    return locked


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write models/fusion_v3/constants.json for the ratified pick")
    ap.add_argument("--ratified-by", help="must be 'nathan' with --write")
    ap.add_argument("--candidate", help="the ratified candidate (must equal the locked report's pick)")
    ap.add_argument("--new-column-model", help="P_wl: the wavlm_l probe directory (its meta.json is hashed)")
    ap.add_argument("--hnoise-evidence", metavar="NOISE_AUC,CLEAN_AUC",
                    help="H_noise diagnostic numbers from the CPU chat's report (20 dB noise AUC, clean AUC)")
    ap.add_argument("--report-sha256", help="with --write: the sha256 of the locked report, as recorded in the sweep doc")
    args = ap.parse_args(argv)
    if args.write and (args.ratified_by != "nathan" or not args.candidate or not args.report_sha256):
        ap.error("--write requires --ratified-by nathan, --candidate NAME and --report-sha256 HEX")
    if args.candidate and args.candidate not in CANDIDATES:
        ap.error(f"--candidate must be one of {list(CANDIDATES)}")
    if not sha_ok(SHIPPED_TSV):
        sys.exit(f"shipped TSV {SHIPPED_TSV.name} no longer hashes to {SHIPPED_SHA_PREFIX}…; stop")
    locked = precheck_write(args) if args.write else None
    hn = None
    if args.hnoise_evidence:
        try:
            hn = tuple(float(v) for v in args.hnoise_evidence.split(","))
            assert len(hn) == 2
        except (ValueError, AssertionError):
            ap.error("--hnoise-evidence takes NOISE_AUC,CLEAN_AUC")

    rep = evaluate(hn) if hn else evaluate()
    print(f"rows (base): {rep['rows_base']}")
    sc = rep["self_check"]
    print(f"self-check: {'ok' if sc['ok'] else 'FAIL'}; got {sc['got']}; Spearman vs shipped {sc['spearman']:.6f}")
    if args.write:
        return do_write(args, rep, locked)
    text = json.dumps(clean(rep), indent=2, allow_nan=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    stamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    (REPORT.parent / f"sweep_v3_report_{stamp}.json").write_text(text)  # every run kept; the lock is by sha256
    print(f"report sha256 {sha256(REPORT)} (archived as sweep_v3_report_{stamp}.json)")
    if not sc["ok"]:
        print("SELF-CHECK FAIL: no candidate read")
        sys.exit(2)
    print_table(rep)
    for n, c in rep["candidates"].items():
        if "gate" in c:
            print(f"{n}: gate {c['gate']['verdict']} failed={c['gate']['failed']} doubled={c['gate']['doubled']}")
        if "diagnostic" in c:
            print(f"  diagnostic: {json.dumps(c['diagnostic'], default=float)}")
        if "room" in c:
            print(f"  room: {c['room']['verdict']} failed={c['room']['failed']}")
        if "bakeoff" in c:
            b = c["bakeoff"]
            print(f"  bake-off: {b['verdict']} parts={b['parts']} tripwire={b['tripwire_max_abs']:.2e}")
            print(f"  perturbation Δ: {b['perturbation_deltas']}")
            print(f"  bootstrap: {b['bootstrap']}")
        if "known_number_check" in c:
            print(f"  known-number check: {c['known_number_check']}")
    print(f"\nDECISION: {rep['decision']['line']}")
    print(f"wrote {REPORT.relative_to(REPO)}")


def do_write(args, rep: dict, locked: dict) -> None:
    if locked.get("inputs") != rep["inputs"]:
        sys.exit("inputs changed since the locked report; refuse")
    if locked["decision"] != clean(rep["decision"]):
        sys.exit("recomputed decision differs from the locked report; refuse")
    # only here; hearsay.pipeline imports neither torch nor lightgbm
    from hearsay.pipeline import M5_DIR

    c, spec = rep["candidates"][args.candidate], CANDIDATES[args.candidate]
    labels, _ = load_labels()
    cols = {k: fs.load(k, str(S / i) if i else None) for k, i in
            (("m1b_v3", None), ("handcrafted_v5", "_itw_handcrafted_v5.csv"), ("m5_xlsr_ft", "_itw_m5.csv"))
            if k in spec["weights"]}  # fmt: skip
    new_model = None
    if spec["new"]:
        col, itw = spec["new"]
        cols[col] = load_raw(col, resolve_itw(col, itw)).drop_duplicates("path").set_index("path")
        new_model = {"name": col, "dir": str(Path(args.new_column_model)),
                     "meta_sha256": sha256(Path(args.new_column_model) / "meta.json")}  # fmt: skip
    ctx = build(cols, fs.load("spectra_aasist", str(REPO / "outputs/spectra/itw_stress/scores.csv")), labels)
    hashes = json.loads((M5_DIR / "hashes.json").read_text())
    a, b = c["platt"]
    tiers = spec["tiers"]
    payload = {
        "status": f"RATIFIED by {args.ratified_by} from the locked post-draft sweep (report sha256 {args.report_sha256})",
        "final": args.candidate, "ratified_by": args.ratified_by, "manifest_commits": MANIFEST_COMMITS,
        "inputs": rep["inputs"], "pi_synth": PI,
        "rank_ref_inner_oof_sorted": {n: ctx["ref"][n].tolist() for n in spec["weights"]},
        "weights": spec["weights"],
        "e_rule": {"applied": True, "m3_logit_below": tiers[0][0], "base_rank_above": 0.5, "multiply_by": tiers[0][1],
                   **({"tiers": [list(t) for t in tiers]} if len(tiers) > 1 else {})},
        "platt": {"a": a, "b": b, "prior_shift": math.log(PI / (1 - PI))},
        "determinate_map": "0.001 + 0.999 * sigmoid(a*fused + b + prior_shift), applied once",
        "how": "rank_d = searchsorted(rank_ref[d], logit_d)/len; base = sum(weights[d] * rank_d) accumulated in "
               "weights order; M3 tiers: deepest satisfied (m3_logit < thr and base > 0.5) sets the total multiplier; "
               "p = determinate_map(base)",
        "m5_checkpoint": {"dir": str(M5_DIR.relative_to(REPO)),
                          **{k: hashes[k] for k in ("backbone_sha256", "head_sha256", "config_hash")}},
        **({"new_column_model": new_model} if new_model else {}),
    }  # fmt: skip
    write_constants(FUSION_V3, payload)
    print(f"wrote {FUSION_V3.relative_to(REPO)}")


if __name__ == "__main__":
    main()
