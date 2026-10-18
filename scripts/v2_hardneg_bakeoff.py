"""V2-33 stage 0: example-weighting bake-off (hard vs easy vs random).

Spec: docs/specs/2026-10-13_v2-33-hardneg-stage0.md (design of record = its
"Phase 3 round 2 - FINAL DESIGN" section, plus the Codex round-1 dispositions).
Plan: docs/reports/2026-10-13_v2-33-hardneg-stage0-plan.md

A DIAGNOSTIC on the legacy split, not a validation: splits/nsa_folds_plus_asv19.csv has known
speaker overlap (41 of 113 speakers span >1 fold, 19,699 of 23,790 rows), so no validation
conclusion is drawn here. V2-33 stays open pending a V2-01 benchmark row on generator- AND
speaker-disjoint folds.

CPU only. Reads saved embeddings; never extracts, never touches torch. Writes only under
outputs/v2/hardneg/.

    uv run python scripts/v2_hardneg_bakeoff.py fit [--jobs 4] [--layers 6,7,8,20]
    uv run python scripts/v2_hardneg_bakeoff.py csweep
    uv run python scripts/v2_hardneg_bakeoff.py gate
    uv run python scripts/v2_hardneg_bakeoff.py report
    uv run python scripts/v2_hardneg_bakeoff.py log      # one experiments/log.csv row
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from hearsay.metrics import cost_at, eer, min_cost

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "v2" / "hardneg"
MODEL = "wav2vec2-xls-r-300m"
TRAIN_SETS = ("nsa_train_sample_v3", "asv19_addon_v3")
FOLDS = REPO / "splits" / "nsa_folds_plus_asv19.csv"
BAKEOFF = REPO / "outputs" / "v2" / "bakeoff"

LAYERS = (6, 7, 8, 20)
PRIMARY_LAYER = 7
C_GRID = (1.0, 8.0, 64.0)
C_SWEEP = (0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0)
POOLS = ("pool_full", "pool_nsa")

FRAC = 0.05
MULT = 5.0
MIN_MINED = 20
FOCAL_GAMMA = 2.0
FOCAL_GRAD_TOL = 1e-2
MAX_ITER = 3000
N_JOBS = 4
BOOT_N = 1000
SEED = 0
TIME_BOX_S = 3 * 3600

# R3/R4 bands and the R6 relative band. Declared in the spec before any candidate was fitted.
BAND_BRIEF = 0.010
BAND_FOLD = 0.010
BAND_FA_REL = 0.10
BOOT_P_MIN = 0.90
R7_REAL_CAP = 0.25
SATURATED = 1.0 - 1e-12

LOG_COLUMNS = ("timestamp", "rung", "validation_score", "validation_score_clean_only",
               "csv_path", "notes", "benchmark_version")  # fmt: skip
RUNG = "V2-33-stage0"
BENCHMARK_VERSION = "none (pre-V2-01)"

# Every embedding set this lane may read. ASVspoof 2019 LA eval, In-the-Wild, the local MLAAD
# cohort, the NSA test set and the outer holdout are barred from every fit, mining and readout.
ALLOWED_SETS = frozenset(TRAIN_SETS)
BARRED_TOKENS = ("asv19_eval_probe", "ASVspoof2019_LA_eval", "itw_stress", "nsa_test",
                 "mlaad", "asv19_train_probe")  # fmt: skip


# --------------------------------------------------------------------------- cost readings
# The ONLY three readings. min_cost's defaults are c_fa=4, c_miss=1, so min_cost(y, s, 0.5) is a
# FOURTH reading (4*P_FA + P_miss) and is never used. Step 9's grep permits min_cost( only here.
def brief(y, s) -> float:
    """v1's reading: 9.33*P_FA + P_miss."""
    return float(min_cost(y, s))


def package05(y, s) -> float:
    """Primary (rule 5): P_FA + 4*P_miss. Body-identical to bakeoff_layers.averse."""
    return float(min_cost(y, s, 0.5, c_fa=1.0, c_miss=4.0))


def package03(y, s) -> float:
    """P_FA + 1.714*P_miss."""
    return float(min_cost(y, s, 0.3, c_fa=1.0, c_miss=4.0))


READINGS = {"package05": package05, "package03": package03, "brief": brief}


def thr_package05(y, s) -> float:
    """Exact package-0.5-optimal threshold. roc_curve's thresholds are incomplete under
    drop_intermediate=True, so re-sweep every unique score plus +inf. Ties resolve to the
    HIGHEST threshold (declared): the lowest would maximise false alarms among tied optima."""
    cand = np.r_[np.unique(np.asarray(s, dtype=np.float64)), np.inf]
    costs = np.array([cost_at(y, s, t, 0.5, c_fa=1.0, c_miss=4.0) for t in cand])
    return float(cand[int(np.flatnonzero(costs <= costs.min() + 1e-15)[-1])])


def degenerate_reason(oof: np.ndarray, readings: list[dict]) -> str | None:
    """R9, centralised (Codex audit round 1, #4): the refusals that make a cell INVALID rather than
    an ordinary FAIL, applied to the masked scores and to EVERY reading on every basis. Applied at
    fit time (the cell is written INVALID) and again at gate time from the checkpoint."""
    if not np.isfinite(oof).all():
        return "non-finite scores"
    if oof.size and float(np.ptp(oof)) == 0.0:
        return "constant scores"
    for r in readings:
        for k in ("auc", "package05", "package03", "brief"):
            if r.get(k) is None or not np.isfinite(float(r[k])):
                return f"non-finite reading {k}"    # clean() writes NaN as null
        if r["auc"] < 0.5:
            return f"inner AUC {r['auc']:.4f} < 0.5"
        for k in ("package05", "package03", "brief"):
            if saturated(r[k]):
                return f"{k} saturated at 1.0"
    return None


def saturated(v: float) -> bool:
    return bool(np.isfinite(v)) and v >= SATURATED


# --------------------------------------------------------------------------- loading
def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_layer(name: str, layer: int, root: Path | None = None):
    """One layer of one embedding set, in manifest row order.

    Keeps hearsay.probe.load_embeddings' exact-permutation guarantee (Codex #5: a plain
    `seen.all()` admits duplicates and negative indices, e.g. rows [0,1,2,3,4,-1]).
    """
    if name not in ALLOWED_SETS:
        raise ValueError(f"{name!r} is not in this lane's allow-list {sorted(ALLOWED_SETS)}")
    for tok in BARRED_TOKENS:
        if tok in name:
            raise ValueError(f"{name!r} matches barred token {tok!r}")
    d = (root or REPO / "outputs" / "embeddings") / MODEL / name
    m = pd.read_csv(d / "manifest.csv")
    shards = sorted(d.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {d}")
    X = np.zeros((len(m), 1024), np.float32)
    seen: list[np.ndarray] = []
    for p in shards:
        z = np.load(p)
        r = z["row"]
        if not np.issubdtype(r.dtype, np.integer):
            raise ValueError(f"{p.name}: row index dtype {r.dtype} is not integer")
        if r.size and (r.min() < 0 or r.max() >= len(m)):
            raise ValueError(f"{p.name}: row index out of bounds [0, {len(m)})")
        emb = z["emb"]
        if emb.ndim != 3 or emb.shape[2] != 1024:
            raise ValueError(f"{p.name}: emb shape {emb.shape} is not (n, n_layers, 1024)")
        if layer >= emb.shape[1]:
            raise ValueError(f"{p.name}: layer {layer} >= n_layers {emb.shape[1]}")
        X[r] = emb[:, layer, :].astype(np.float32)
        seen.append(r)
    rows = np.concatenate(seen)
    if not np.array_equal(np.sort(rows), np.arange(len(m))):
        raise ValueError(f"{name}: shard row indices are not an exact permutation of 0..{len(m) - 1}")
    if "flag" not in m.columns:
        raise ValueError(f"{name}: manifest has no flag column")
    if not m["flag"].fillna("").eq("").all():
        raise ValueError(f"{name}: decode_error rows present")
    if (X == 0).all(axis=1).any():
        raise ValueError(f"{name}: all-zero embedding rows present")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: non-finite embedding values")
    return X, m


def load_world(layer: int, root: Path | None = None):
    """Inner rows for one layer, in outputs/v2/bakeoff/rows.json order, plus both pool masks."""
    parts = [load_layer(n, layer, root) for n in TRAIN_SETS]
    X = np.concatenate([p[0] for p in parts])
    m = pd.concat([p[1] for p in parts], ignore_index=True)

    ff = pd.read_csv(FOLDS)
    if ff.path.duplicated().any():
        raise ValueError("fold file has duplicate paths")
    if m.path.duplicated().any():
        raise ValueError("concatenated manifest has duplicate paths")
    # Codex #4: the manifests already carry label/generator/speaker/source, so .join of those
    # raises "columns overlap". Take only fold+group and turn the overlap into a free check.
    idx = ff.set_index("path")
    m = m.join(idx[["fold", "group"]], on="path")
    if not m.fold.notna().all():
        raise ValueError(f"{int(m.fold.isna().sum())} manifest rows are not in the fold file")
    for col in ("label", "generator", "speaker", "source"):
        theirs = m.path.map(idx[col])
        if not theirs.eq(m[col]).all():
            raise ValueError(f"manifest/fold-file disagree on {col} "
                             f"({int((~theirs.eq(m[col])).sum())} rows)")

    inner = (m.fold != "holdout").to_numpy()  # outer holdout dropped here, unread
    X, m = X[inner], m[inner].reset_index(drop=True)

    rows = json.loads((BAKEOFF / "rows.json").read_text())
    ip = np.asarray(rows["inner_path"])
    if set(ip) != set(m.path):
        raise ValueError("inner path set disagrees with the bake-off row index")
    pos = pd.Series(np.arange(len(m)), index=m.path)
    order = pos.loc[ip].to_numpy()
    X, m = X[order], m.iloc[order].reset_index(drop=True)

    y = np.asarray(rows["inner_y"], dtype=int)
    if not (m.label == "spoof").to_numpy(int).__eq__(y).all():
        raise ValueError("label disagrees with the bake-off row index")
    fold = m.fold.astype(int).to_numpy()  # holdout already dropped, so the cast is safe
    if not np.array_equal(fold, np.asarray(rows["inner_fold"], dtype=int)):
        raise ValueError("fold disagrees with the bake-off row index")

    pools = {"pool_full": np.ones(len(y), bool),
             "pool_nsa": (m.source != "asvspoof2019").to_numpy()}  # fmt: skip
    return X, y, fold, m, pools


# --------------------------------------------------------------------------- arms
def balanced_weights(y_tr: np.ndarray) -> np.ndarray:
    """class_weight='balanced' on UNWEIGHTED counts. Never pass this to sklearn as
    class_weight='balanced' alongside sample_weight: sklearn recomputes 'balanced' on the
    sample-weighted counts, which cancels the class differential and leaves a global scale --
    exactly equivalent to scaling C (spec W1)."""
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    return np.where(y_tr == 0, cw[0], cw[1]).astype(np.float64)


def shaped_weights(y_tr: np.ndarray, picks: np.ndarray, mult: float = MULT) -> np.ndarray:
    """Balanced weights with `picks` up-weighted, renormalised so sum(w) == n. Holding sum(w)
    fixed is what makes every arm a comparison at MATCHED effective regularization."""
    w = balanced_weights(y_tr)
    w[picks] *= mult
    return w * (len(y_tr) / w.sum())


def fit_lr(Xs: np.ndarray, y_tr: np.ndarray, w: np.ndarray, c: float) -> LogisticRegression:
    """Bare LogisticRegression: sample_weight cannot pass through make_clf's Pipeline, and a
    pipeline here would re-scale already-scaled data."""
    return LogisticRegression(C=c, max_iter=MAX_ITER).fit(Xs, y_tr, sample_weight=w)


def focal_objective(Xs: np.ndarray, y_tr: np.ndarray, gamma: float, c: float):
    """-sum(a * (1-pt)^gamma * log pt) + 0.5*||w||^2, with the analytic gradient. Matches the
    pilot's objective; gradient verified against finite differences to 5e-7 relative and
    gamma=0 reproduces the class-balanced LR to 2e-7."""
    n = len(y_tr)
    a = np.where(y_tr == 1, n / (2 * y_tr.sum()), n / (2 * (n - y_tr.sum())))
    s = np.where(y_tr == 1, 1.0, -1.0)

    def f(theta):
        wv, b = theta[:-1], theta[-1]
        z = Xs @ wv + b
        pt = np.clip(1 / (1 + np.exp(-s * z)), 1e-12, 1 - 1e-12)
        loss = -(a * (1 - pt) ** gamma * np.log(pt)).sum()
        g = a * s * (gamma * (1 - pt) ** gamma * pt * np.log(pt) - (1 - pt) ** (gamma + 1))
        return c * loss + 0.5 * wv @ wv, np.r_[c * (Xs.T @ g) + wv, c * g.sum()]

    return f


def normalised_margin(d: np.ndarray, y_tr: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Signed margin scaled by the IQR of the reference margins (Codex #6). The reference is the
    inner model's OWN training rows, so the scale is fitted inside the training fold and never
    uses the rows being ranked. Pooling raw margins from several inner models would let model
    scale decide which fold supplies the 'hardest' examples."""
    marg = np.where(y_tr == 1, d, -d)
    r = marg if ref is None else ref
    q75, q25 = np.percentile(r, [75, 25])
    iqr = float(q75 - q25)
    return marg / (iqr if iqr > 0 else 1.0)


def inner_oof_margin(Xtr: np.ndarray, y_tr: np.ndarray, f_tr: np.ndarray, c: float) -> np.ndarray:
    """Nested inner-inner OOF normalised margins over the training folds. A fold never selects
    its own hard examples. In-sample margins are unusable: the probe is perfectly separable
    in-sample (0 errors in all 5 folds), which collapses hardest/easiest/wrong-at-threshold."""
    marg = np.full(len(y_tr), np.nan)
    for j in sorted(set(f_tr.tolist())):
        a, b = f_tr != j, f_tr == j
        sc = StandardScaler().fit(Xtr[a])
        clf = fit_lr(sc.transform(Xtr[a]), y_tr[a], balanced_weights(y_tr[a]), c)
        d_in = clf.decision_function(sc.transform(Xtr[a]))
        d_out = clf.decision_function(sc.transform(Xtr[b]))
        ref = np.where(y_tr[a] == 1, d_in, -d_in)
        marg[b] = normalised_margin(d_out, y_tr[b], ref=ref)
    if not np.isfinite(marg).all():
        raise ValueError("inner-inner margin coverage incomplete")
    return marg


class Arm:
    """One weighting arm. A plain class, not a dataclass: `dataclasses` resolves
    `sys.modules[cls.__module__]` at class-creation time, so a dataclass here would crash under
    the loader idiom that does not register the module (tests/test_fuse_sweep_v3.py:19-23),
    while working under the one that does (scripts/bakeoff_layers.py:53-58). This repo uses
    both, so Arm must be loadable either way."""

    __slots__ = ("layers", "loss", "name", "params", "shape", "side")

    def __init__(self, name: str, shape: str, side: str, loss: str = "lr",
                 layers: tuple[int, ...] = LAYERS, params: dict | None = None):
        self.name = name            # unique cell key
        self.shape = shape          # none | hardest | easiest | random | pilot_quantile
        self.side = side            # both | reals | fakes
        self.loss = loss            # lr | focal
        self.layers = layers
        self.params = dict(params or {})

    def __repr__(self) -> str:
        return f"Arm({self.name!r}, {self.shape!r}, {self.side!r}, loss={self.loss!r})"

    @property
    def four_layer(self) -> bool:
        return tuple(self.layers) == LAYERS


ARMS: tuple[Arm, ...] = (
    Arm("none", "none", "both"),
    Arm("hardest_both", "hardest", "both"),
    Arm("easiest_both", "easiest", "both"),
    Arm("random_both", "random", "both"),
    Arm("hardest_reals", "hardest", "reals", layers=(PRIMARY_LAYER,)),
    Arm("hardest_fakes", "hardest", "fakes", layers=(PRIMARY_LAYER,)),
    Arm("easiest_reals", "easiest", "reals", layers=(PRIMARY_LAYER,)),
    Arm("easiest_fakes", "easiest", "fakes", layers=(PRIMARY_LAYER,)),
    # focal does its own class balancing inside the objective, so it is NOT on the matched
    # sum(sample_weight) footing the weighting arms share. Descriptive, L7 only, never a
    # recommendation -- it answers "does a continuous hard-example emphasis help", not the
    # three-way question.
    Arm("focal_g2", "none", "both", loss="focal", layers=(PRIMARY_LAYER,),
        params={"gamma": FOCAL_GAMMA, "matched_sum_sw": False}),
    Arm("pilot_quantile", "pilot_quantile", "both", layers=(PRIMARY_LAYER,),
        params={"real_q": 0.98, "fake_frac": 0.05, "note": "pilot-comparable; not recommendable"}),
)
ARMS_BY_NAME = {a.name: a for a in ARMS}
PRIMARY_FAMILY = ("none", "hardest_both", "easiest_both", "random_both")


def select_picks(arm: Arm, marg: np.ndarray, y_tr: np.ndarray, rng: np.random.Generator):
    """Indices to up-weight. `marg` is the nested-OOF normalised margin: lower = harder."""
    if arm.shape == "none":
        return np.array([], dtype=int), {}
    picks, per_class = [], {}
    classes = {"both": (0, 1), "reals": (0,), "fakes": (1,)}[arm.side]
    for cls in classes:
        idx = np.flatnonzero(y_tr == cls)
        if arm.shape == "pilot_quantile":
            if cls == 0:
                # The pilot took reals ABOVE the 98th score percentile, i.e. the most
                # synthetic-looking reals, which in margin terms is the lowest 2%. Asymmetric
                # (2% reals / 5% fakes) unlike every other arm -- stated in the report.
                sel = idx[marg[idx] <= np.quantile(marg[idx], 1 - arm.params["real_q"])]
            else:
                k = max(1, int(len(idx) * arm.params["fake_frac"]))
                sel = idx[np.argsort(marg[idx], kind="stable")[:k]]
        else:
            k = max(1, int(len(idx) * FRAC))
            if arm.shape == "hardest":
                sel = idx[np.argsort(marg[idx], kind="stable")[:k]]
            elif arm.shape == "easiest":
                sel = idx[np.argsort(-marg[idx], kind="stable")[:k]]
            else:
                sel = rng.choice(idx, k, replace=False)
        per_class[int(cls)] = len(sel)
        picks.append(sel)
    return np.concatenate(picks) if picks else np.array([], dtype=int), per_class


def arm_oof(arm: Arm, X: np.ndarray, y: np.ndarray, fold: np.ndarray, c: float):
    """Out-of-fold scores for one arm. NaN-initialised: 0.0 is a legal decision value, so
    np.zeros would hide an unwritten fold (spec C9)."""
    oof = np.full(len(y), np.nan)
    rng = np.random.default_rng(SEED)
    mined: dict[int, dict] = {}
    picks_by_fold: dict[int, np.ndarray] = {}   # row indices into X, per outer fold (R7/H3/H4)
    grad_norms: list[float] = []
    for k in sorted(set(fold.tolist())):
        tr, te = fold != k, fold == k
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte, y_tr = sc.transform(X[tr]), sc.transform(X[te]), y[tr]
        if arm.loss == "focal":
            f = focal_objective(Xtr, y_tr, arm.params["gamma"], c)
            res = minimize(f, np.zeros(Xtr.shape[1] + 1), jac=True, method="L-BFGS-B",
                           options={"maxiter": MAX_ITER})
            gn = float(np.linalg.norm(f(res.x)[1]))
            grad_norms.append(gn)
            if gn > FOCAL_GRAD_TOL:
                raise ValueError(f"focal gradient norm {gn:.3g} > {FOCAL_GRAD_TOL} on fold {k}")
            oof[te] = Xte @ res.x[:-1] + res.x[-1]
            continue
        if arm.shape == "none":
            w = balanced_weights(y_tr)
            w = w * (len(y_tr) / w.sum())
        else:
            marg = inner_oof_margin(Xtr, y_tr, fold[tr], c)
            picks, per_class = select_picks(arm, marg, y_tr, rng)
            for cls, n in per_class.items():
                if n < MIN_MINED:
                    raise ValueError(f"mined set for class {cls} is {n} < {MIN_MINED} on fold {k}")
            mined[int(k)] = per_class
            picks_by_fold[int(k)] = np.flatnonzero(tr)[picks]   # training rows only, never te
            w = shaped_weights(y_tr, picks, MULT)
        oof[te] = fit_lr(Xtr, y_tr, w, c).decision_function(Xte)
    missing = int((~np.isfinite(oof)).sum())
    if missing:
        raise ValueError(f"score coverage incomplete: {missing} rows unwritten")
    return oof, {"mined_per_fold": mined, "picks_by_fold": picks_by_fold,
                 "focal_grad_norms": grad_norms}  # fmt: skip


# --------------------------------------------------------------------------- readings
def read_cell(y, s, fold, meta, mask) -> dict:
    yy, ss, ff = y[mask], s[mask], fold[mask]
    out: dict = {"n": int(mask.sum()), "n_real": int((yy == 0).sum()),
                 "n_fake": int((yy == 1).sum()), "eer": round(float(eer(yy, ss)), 6)}  # fmt: skip
    for nm, fn in READINGS.items():
        out[nm] = round(fn(yy, ss), 6)
        out[f"{nm}_by_fold"] = {int(k): round(fn(yy[ff == k], ss[ff == k]), 6)
                                for k in sorted(set(ff.tolist()))
                                if len(set(yy[ff == k].tolist())) == 2}
    thr = thr_package05(yy, ss)
    fa = (yy == 0) & (ss >= thr)
    out["thr_package05"] = round(thr, 6)
    out["n_fa_at_thr"] = int(fa.sum())
    out["n_miss_at_thr"] = int(((yy == 1) & (ss < thr)).sum())
    out["auc"] = round(float(roc_auc_score(yy, ss)), 6)
    # H5: a false alarm whose normalised real-side margin is in the worst decile.
    real = np.flatnonzero(yy == 0)
    nm_real = normalised_margin(ss[real], yy[real])
    cut = float(np.percentile(nm_real, 10))
    cw = fa[real] & (nm_real <= cut)
    out["confidently_wrong_fa"] = int(cw.sum())
    out["confidently_wrong_cut"] = round(cut, 6)
    # Population composition of the basis, for context only. R7 reads the MINED set (below).
    sub = meta[mask]
    for lbl, nm in ((0, "real"), (1, "fake")):
        g = sub.group[(yy == lbl)]
        out[f"pop_group_concentration_{nm}"] = round(float(g.value_counts(normalize=True).iloc[0]), 6)
        out[f"pop_top_group_{nm}"] = str(g.value_counts().index[0])
    return out


def _shares(series: pd.Series, top: int | None = None) -> dict:
    vc = series.value_counts(normalize=True)
    if top is not None:
        vc = vc.head(top)
    return {str(k): round(float(v), 4) for k, v in vc.items()}


def characterise(meta, y, per_fold_counts, picks_by_fold) -> dict:
    """R7/H4: what a mined set is actually made of, computed on the MINED rows -- pooled across
    the outer folds, each pick counted once per fold that selected it -- never on the population.

    The Oct 13 run computed group concentration on the whole real/fake population of the basis
    instead (every cell read 0.0316 / lj:LJ028), so R7 could not fail: the spec's E1 failure mode.
    Fixed and re-run Oct 17 with byte-identical OOF scores; see the results report."""
    if not picks_by_fold:
        return {}
    out: dict = {"per_fold_counts": per_fold_counts,
                 "duration": "not available: the embedding manifests carry no duration column"}
    for lbl, nm in ((0, "real"), (1, "fake")):
        per_fold = [v[y[v] == lbl] for v in picks_by_fold.values()]
        per_fold = [v for v in per_fold if v.size]
        if not per_fold:
            continue
        idx = np.concatenate(per_fold)
        sub = meta.iloc[idx]
        g, sp = sub.group.value_counts(normalize=True), sub.speaker.value_counts(normalize=True)
        out[f"n_picks_{nm}"] = int(idx.size)
        out[f"n_unique_rows_{nm}"] = int(np.unique(idx).size)
        out[f"group_concentration_{nm}"] = round(float(g.iloc[0]), 6)
        out[f"top_group_{nm}"] = str(g.index[0])
        out[f"per_fold_max_group_share_{nm}"] = round(max(
            float(meta.iloc[v].group.value_counts(normalize=True).iloc[0]) for v in per_fold), 6)
        out[f"speaker_concentration_{nm}"] = round(float(sp.iloc[0]), 6)
        out[f"top_speaker_{nm}"] = str(sp.index[0])
        out[f"source_shares_{nm}"] = _shares(sub.source)
        if lbl == 1:
            out["generator_shares_fake"] = _shares(sub.generator, top=10)
    return out


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """H3: overlap of two mined sets (unique rows). 1.0 = the arms collapsed onto the same clips."""
    sa, sb = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------- bootstrap
def group_bootstrap(y, s_base, s_arm, groups, n=BOOT_N, seed=SEED, cost=package05):
    """Paired cluster bootstrap over fold-file `group` units. delta = base - arm, so positive
    means the arm is better (convention asserted by a test; the repo holds three different ones).
    Single-class draws are skipped and counted. The same resample is reused for both scores."""
    g = pd.Series(groups)
    if g.isna().any() or g.astype(str).str.strip().eq("").any():
        raise ValueError("blank group values in the bootstrap unit column")
    codes, _ = pd.factorize(g)
    members = [np.flatnonzero(codes == i) for i in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    deltas, skipped = [], 0
    for _ in range(n):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        if len(set(y[idx].tolist())) < 2:
            skipped += 1
            continue
        deltas.append(cost(y[idx], s_base[idx]) - cost(y[idx], s_arm[idx]))
    d = np.asarray(deltas, dtype=float)
    p = float(np.mean(d > 0)) if d.size else float("nan")
    se = float(np.sqrt(max(p * (1 - p), 0.0) / d.size)) if d.size else float("nan")
    return {"p_better": round(p, 6), "p_better_se": round(se, 6),
            "p5": round(float(np.percentile(d, 5)), 6) if d.size else None,
            "median": round(float(np.median(d)), 6) if d.size else None,
            "n": int(d.size), "skipped_single_class": skipped,
            "n_groups": len(members)}  # fmt: skip


# --------------------------------------------------------------------------- gate
def gate_cell(arm: Arm, cell: dict, base: dict, boot: dict) -> dict:
    """R1-R10 as a flat dict of booleans plus the verdict, so the gate is auditable."""
    r: dict = {}
    pc, pb = cell["primary"], base["primary"]

    def num(x):  # clean() writes NaN as null; a null reading must not raise, it must fail R9
        return float("nan") if x is None else float(x)

    def d(key):  # 4-dp rounding before subtraction, EPS on the comparison
        return round(round(num(pc[key]), 4) - round(num(pb[key]), 4), 4)

    eps = 1e-12
    p, se = boot.get("p_better", float("nan")), boot.get("p_better_se", float("nan"))
    r["R1"] = bool(d("package05") < -eps and np.isfinite(p) and p >= BOOT_P_MIN - eps)
    r["R1_indeterminate"] = bool(np.isfinite(p) and np.isfinite(se)
                                 and abs(p - BOOT_P_MIN) <= 2 * se)
    r["R2"] = bool(np.sign(d("package03")) == np.sign(d("package05")) and d("package05") < -eps)
    r["R3"] = bool(d("brief") <= BAND_BRIEF + eps)
    signs = {np.sign(d(k)) for k in ("package05", "package03", "brief")} - {0.0}
    r["R3b"] = len(signs) <= 1
    # Cells come back from JSON with STRING fold keys ("0", "4"); in memory they are ints. The
    # Oct 13 gate looked up int keys on JSON-loaded cells, got None, and failed R4a/R4b in all
    # 107 non-base cells -- 13 of them on R4 alone, several with f0 and f4 actually better.
    bf_c = {int(k): v for k, v in pc["package05_by_fold"].items()}
    bf_b = {int(k): v for k, v in pb["package05_by_fold"].items()}
    for rule, k in (("R4a", 0), ("R4b", 4)):
        cv, bv = num(bf_c.get(k)), num(bf_b.get(k))
        r[rule] = bool(np.isfinite(cv) and np.isfinite(bv)
                       and round(round(cv, 4) - round(bv, 4), 4) <= BAND_FOLD + eps)
    r["R5_eligible"] = arm.four_layer
    r["R6"] = bool(num(pc["n_fa_at_thr"]) <= num(pb["n_fa_at_thr"]) * (1 + BAND_FA_REL) + eps)
    # R7 reads the MINED set's real-side group concentration, never the population's. A missing
    # characterisation readout is a failure of R7 ("readout present"), not a pass.
    mined = cell.get("mined") or {}
    conc = mined.get("group_concentration_real")
    if arm.shape == "none":
        r["R7"] = True
    elif arm.side == "fakes":
        r["R7"] = bool(mined)                       # no mined reals to cap; the readout must exist
    else:
        r["R7"] = bool(conc is not None and np.isfinite(conc) and conc <= R7_REAL_CAP + eps)
    vals = [num(pc.get(k)) for k in ("package05", "package03", "brief", "auc")]
    bad = (not np.isfinite(vals).all() or vals[3] < 0.5 or any(saturated(v) for v in vals[:3]))
    r["R9"] = not bad
    gated = [k for k in ("R1", "R2", "R3", "R3b", "R4a", "R4b", "R6", "R7", "R9") if not r[k]]
    r["failed"] = gated
    if [k for k in gated if k != "R1"]:
        r["verdict"] = "FAIL"
    elif r["R1_indeterminate"]:
        # FINAL DESIGN R1: within 2 SE of the 0.90 band is INDETERMINATE, never a pass (and never
        # counted by R5). Codex audit round 1, #2: the flag alone had let such a cell PASS.
        r["verdict"] = "INDETERMINATE"
    else:
        r["verdict"] = "PASS" if not gated else "FAIL"
    return r


# --------------------------------------------------------------------------- run plumbing
_DIGEST_CACHE: dict[str, str] = {}


def shard_digest(name: str, root: Path | None = None) -> str:
    """sha256 over a set's manifest AND every shard file. Codex audit round 1, #3: the manifest
    alone misses a re-extracted shard whose manifest did not change (the repair tool rewrites rows
    in place). ~2.8 GB hashed once per process and cached."""
    d = (root or REPO / "outputs" / "embeddings") / MODEL / name
    files = [d / "manifest.csv", *sorted(d.glob("shard_*.npz"))]
    # keyed on every file's size and mtime too, so a file rewritten in place is re-hashed
    key = json.dumps([str(d)] + [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in files])
    if key not in _DIGEST_CACHE:
        h = hashlib.sha256()
        for p in files:
            h.update(p.name.encode())
            h.update(_sha(p).encode())
        _DIGEST_CACHE[key] = h.hexdigest()
    return _DIGEST_CACHE[key]


# Every function on the score-producing path. Their SOURCE is hashed into the fingerprint, so a
# change to how a cell is fitted or read invalidates every checkpoint, while a change to the gate,
# the report, a docstring elsewhere or this comment does not (the Oct 17 whole-file hash forced a
# full refit for every gate-layer fix).
FINGERPRINT_SOURCES = ("load_layer", "load_world", "balanced_weights", "shaped_weights", "fit_lr",
                       "focal_objective", "normalised_margin", "inner_oof_margin", "Arm",
                       "select_picks", "arm_oof", "brief", "package05", "package03",
                       "thr_package05", "saturated", "read_cell", "_shares", "characterise",
                       "degenerate_reason")  # fmt: skip


def fitting_path_digest() -> str:
    """Source segments taken from this file's AST (not inspect.getsource, which cannot resolve a
    class under the unregistered loader idiom this repo also uses)."""
    src = Path(__file__).read_text()
    defs = {n.name: n for n in ast.parse(src).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    h = hashlib.sha256()
    for name in FINGERPRINT_SOURCES:
        h.update(name.encode())
        h.update(ast.get_source_segment(src, defs[name]).encode())
    h.update(json.dumps({"MAX_ITER": MAX_ITER, "FOCAL_GAMMA": FOCAL_GAMMA,
                         "FOCAL_GRAD_TOL": FOCAL_GRAD_TOL, "SATURATED": SATURATED,
                         "TRAIN_SETS": TRAIN_SETS, "MODEL": MODEL}, sort_keys=True).encode())
    return h.hexdigest()


def fingerprint(arm: Arm, layer: int, pool: str, c: float, pool_mask: np.ndarray,
                root: Path | None = None) -> str:
    """Codex #3: a checkpoint is reusable only if every score-changing input is unchanged: fold
    file, row index, the fitting-path source, the embedding sets (manifest + shard contents), the
    pool mask and every arm parameter."""
    h = hashlib.sha256()
    for p in (FOLDS, BAKEOFF / "rows.json"):
        h.update(_sha(p).encode())
    h.update(fitting_path_digest().encode())
    for n in TRAIN_SETS:
        h.update(shard_digest(n, root).encode())
    h.update(hashlib.sha256(pool_mask.tobytes()).hexdigest().encode())
    h.update(json.dumps({"arm": arm.name, "shape": arm.shape, "side": arm.side,
                         "loss": arm.loss, "params": arm.params, "layer": layer,
                         "pool": pool, "C": c, "frac": FRAC, "mult": MULT,
                         "min_mined": MIN_MINED, "seed": SEED},
                        sort_keys=True).encode())  # fmt: skip
    return h.hexdigest()


def reusable_checkpoint(path: Path, fp: str) -> bool:
    """Resume rule (Codex #3): reuse only a checkpoint that reads whole, carries THIS fingerprint
    and, if OK, still has its score file. Anything else is recomputed; nothing aborts the grid."""
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if rec.get("fingerprint") != fp:
        return False
    return not (rec.get("status") == "OK" and not path.with_suffix(".npz").exists())


def declared_grid() -> list[tuple[str, int, str, float]]:
    """The frozen requested grid. Completeness is derived from THIS, never from files on disk."""
    return [(a.name, layer, pool, c)
            for a in ARMS for layer in a.layers for pool in POOLS for c in C_GRID]


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(payload), indent=1, allow_nan=False))
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays) -> None:
    """Codex audit round 1, #3: a checkpoint is written whole or not at all."""
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def clean(x):
    """JSON-safe: numpy scalars to Python, non-finite floats to None (never a bare NaN)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(float(x)) else float(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


class Deadline:
    """One persisted run deadline honoured by every subcommand (Codex #8)."""

    def __init__(self, seconds: int = TIME_BOX_S):
        f = OUT / "deadline.json"
        if f.exists():
            self.at = float(json.loads(f.read_text())["at"])
        else:
            self.at = time.time() + seconds
            OUT.mkdir(parents=True, exist_ok=True)
            atomic_write_json(f, {"at": self.at, "seconds": seconds})

    @property
    def expired(self) -> bool:
        return time.time() >= self.at

    def left(self) -> float:
        return max(0.0, self.at - time.time())


# --------------------------------------------------------------------------- stages
def self_check(X7, y, fold) -> None:
    """Must RAISE, not print. The pilot's equivalent printed max|diff| 57.4 -- a raw difference on
    a Platt-rescaled reference -- and the run continued to produce the quoted table."""
    ref = np.load(BAKEOFF / "probes" / "L07.npz")["oof"]
    mine, _ = arm_oof(ARMS_BY_NAME["none"], X7, y, fold, 1.0)
    rho = float(spearmanr(mine, ref).statistic)
    gap = abs(package05(y, mine) - package05(y, ref))
    if rho != 1.0 or gap > 1e-12:
        raise SystemExit(f"self-check failed: base does not reproduce the shipped L7 probe "
                         f"(spearman {rho!r}, package05 gap {gap:.3e})")
    print(f"self-check OK: spearman {rho}, package05 gap {gap:.1e}", flush=True)


def stage_fit(layers: tuple[int, ...], jobs: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cells").mkdir(exist_ok=True)
    dl = Deadline()
    t0 = time.time()

    worlds = {}
    for layer in sorted(set(layers) | {PRIMARY_LAYER}):
        worlds[layer] = load_world(layer)
        print(f"loaded L{layer:02d} {worlds[layer][0].shape} {time.time() - t0:.1f}s", flush=True)

    X7, y, fold, meta, _pools = worlds[PRIMARY_LAYER]
    self_check(X7, y, fold)

    rows_payload = {"inner_path": meta.path.tolist(), "inner_y": y.tolist(),
                    "inner_fold": fold.tolist(), "inner_source": meta.source.tolist(),
                    "inner_speaker": meta.speaker.tolist(),
                    "inner_group": meta.group.tolist()}  # fmt: skip
    atomic_write_json(OUT / "rows.json", rows_payload)

    requested = [g for g in declared_grid() if g[1] in layers]
    done = skipped = failed = 0
    for arm_name, layer, pool, c in requested:
        if dl.expired:
            print(f"DEADLINE reached; {len(requested) - done - skipped} cells unrun", flush=True)
            break
        arm = ARMS_BY_NAME[arm_name]
        Xl, yl, foldl, metal, poolsl = worlds[layer]
        mask = poolsl[pool]
        fp = fingerprint(arm, layer, pool, c, mask)
        f = OUT / "cells" / f"L{layer:02d}_{pool}_C{c:g}_{arm_name}.json"
        if f.exists() and reusable_checkpoint(f, fp):
            skipped += 1
            continue
        rec = {"arm": arm_name, "layer": layer, "pool": pool, "C": c, "fingerprint": fp,
               "four_layer": arm.four_layer, "params": arm.params}  # fmt: skip
        try:
            oof, extra = arm_oof(arm, Xl[mask], yl[mask], foldl[mask], c)
            full = np.full(len(yl), np.nan)
            full[mask] = oof
            where = np.flatnonzero(mask)            # picks stored as GLOBAL row indices (H3)
            atomic_savez(OUT / "cells" / f.with_suffix(".npz").name, oof=full, mask=mask,
                         **{f"picks_f{k}": where[v] for k, v in extra["picks_by_fold"].items()})
            sub = metal[mask].reset_index(drop=True)
            nsa = (sub.source != "asvspoof2019").to_numpy()
            rec["primary"] = read_cell(yl[mask], oof, foldl[mask], sub, nsa)
            rec["secondary"] = (read_cell(yl[mask], oof, foldl[mask], sub,
                                          np.ones(int(mask.sum()), bool))
                                if pool == "pool_full" else "N/A - declared in advance")
            rec["mined"] = characterise(sub, yl[mask], extra["mined_per_fold"],
                                        extra["picks_by_fold"])
            readings = [rec["primary"]] + ([rec["secondary"]] if pool == "pool_full" else [])
            why = degenerate_reason(oof, readings)
            if why:
                raise ValueError(f"R9 refusal: {why}")
            rec["focal_grad_norms"] = extra["focal_grad_norms"]
            rec["status"] = "OK"
            done += 1
        except Exception as e:  # noqa: BLE001 - one bad fit must not kill the grid
            rec["status"] = "INVALID"
            rec["reason"] = f"{type(e).__name__}: {e}"
            failed += 1
        atomic_write_json(f, rec)
        print(f"  {f.stem} {rec['status']} {time.time() - t0:.0f}s", flush=True)
    print(f"fit done: {done} computed, {skipped} reused, {failed} INVALID, "
          f"{time.time() - t0:.0f}s, {dl.left() / 60:.0f} min left", flush=True)


def stage_csweep(layers: tuple[int, ...]) -> None:
    """Descriptive only: the C curve is a LEAD for the V2-01 benchmark lane, never a selection.
    Selecting C on inner OOF is the L7+WavLM trap."""
    OUT.mkdir(parents=True, exist_ok=True)
    dl = Deadline()
    out: dict = {"note": "descriptive; NOT a selection. Inner OOF cannot price wild false alarms.",
                 "rows": []}  # fmt: skip
    for layer in layers:
        X, y, fold, meta, pools = load_world(layer)
        for pool in POOLS:
            mask = pools[pool]
            nsa = (meta[mask].source != "asvspoof2019").to_numpy()
            for c in C_SWEEP:
                if dl.expired:
                    out["incomplete"] = True
                    break
                oof, _ = arm_oof(ARMS_BY_NAME["none"], X[mask], y[mask], fold[mask], c)
                r = read_cell(y[mask], oof, fold[mask], meta[mask].reset_index(drop=True), nsa)
                out["rows"].append({"layer": layer, "pool": pool, "C": c,
                                    "package05": r["package05"], "package03": r["package03"],
                                    "brief": r["brief"], "eer": r["eer"],
                                    "n_fa_at_thr": r["n_fa_at_thr"],
                                    "package05_by_fold": r["package05_by_fold"]})  # fmt: skip
                print(f"  csweep L{layer:02d} {pool} C={c:g} pkg05={r['package05']:.4f}", flush=True)
    atomic_write_json(OUT / "csweep.json", out)


def load_cells() -> dict:
    cells = {}
    for f in sorted((OUT / "cells").glob("*.json")):
        d = json.loads(f.read_text())
        cells[(d["arm"], d["layer"], d["pool"], d["C"])] = d
    return cells


def _npz_path(key) -> Path:
    arm_name, layer, pool, c = key
    return OUT / "cells" / f"L{layer:02d}_{pool}_C{c:g}_{arm_name}.npz"


def expected_fingerprints() -> dict:
    """What every declared cell's fingerprint must be under the inputs and script on disk NOW."""
    _X, _y, _fold, _meta, pools = load_world(PRIMARY_LAYER)
    return {(a.name, layer, pool, c): fingerprint(a, layer, pool, c, pools[pool])
            for a in ARMS for layer in a.layers for pool in POOLS for c in C_GRID}


def validate_cells(cells: dict, expected: dict | None) -> dict:
    """Codex audit round 1, #3 and #4: a checkpoint is usable only if its fingerprint matches what
    this script would compute now, its score file loads whole, and its scores are not degenerate.
    Anything else becomes INVALID with a reason -- never silently reused, never an ordinary FAIL,
    never a contrast stratum."""
    out: dict = {}
    for key, cell in cells.items():
        if expected is not None and key in expected and cell.get("fingerprint") != expected[key]:
            out[key] = {**cell, "status": "INVALID",
                        "reason": "stale checkpoint: fingerprint does not match the current inputs"}
            continue
        if cell["status"] == "OK":
            try:
                z = np.load(_npz_path(key))
                oof, mask = z["oof"], z["mask"]
                if oof.shape != mask.shape or not np.isfinite(oof[mask]).all():
                    raise ValueError("score coverage incomplete")
                readings = [cell["primary"]] + ([cell["secondary"]]
                                                if isinstance(cell.get("secondary"), dict) else [])
                why = degenerate_reason(oof[mask], readings)
            except Exception as e:  # noqa: BLE001 - a truncated or unreadable checkpoint
                out[key] = {**cell, "status": "INVALID",
                            "reason": f"unreadable checkpoint: {type(e).__name__}: {e}"}
                continue
            if why:
                out[key] = {**cell, "status": "INVALID", "reason": f"R9 refusal: {why}"}
                continue
        out[key] = cell
    return out


def aggregate_contrast(y, groups, pairs, n=BOOT_N, seed=SEED) -> dict:
    """Codex audit round 1, #1: one contrast at ONE C, averaged over its (layer, pool) strata inside
    each shared group-resample, so the aggregate carries its own uncertainty. `pairs` is a list of
    (s_base, s_arm) on identical rows. Same draw plan and seed as group_bootstrap, so the strata's
    own bootstraps and this aggregate see the same resamples. delta = base - arm, positive = arm
    better."""
    codes, _ = pd.factorize(pd.Series(groups))
    members = [np.flatnonzero(codes == i) for i in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    deltas, skipped = [], 0
    for _ in range(n):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        if len(set(y[idx].tolist())) < 2:
            skipped += 1
            continue
        deltas.append(float(np.mean([package05(y[idx], sb[idx]) - package05(y[idx], sa[idx])
                                     for sb, sa in pairs])))
    d = np.asarray(deltas, dtype=float)
    p = float(np.mean(d > 0)) if d.size else float("nan")
    return {"n_strata": len(pairs), "p_better": round(p, 6),
            "p_better_se": (round(float(np.sqrt(max(p * (1 - p), 0.0) / d.size)), 6)
                            if d.size else None),
            "median": round(float(np.median(d)), 6) if d.size else None,
            "p5": round(float(np.percentile(d, 5)), 6) if d.size else None,
            "p95": round(float(np.percentile(d, 95)), 6) if d.size else None,
            "n": int(d.size), "skipped_single_class": skipped}  # fmt: skip


CONTRASTS = (("hardest_both", "none"), ("easiest_both", "none"),
             ("hardest_both", "random_both"), ("easiest_both", "random_both"))


class _DeadlineExpired(Exception):
    pass


def stage_gate(expected: dict | None = None, dl: Deadline | None = None) -> dict:
    """Verdicts, R5, the four pre-declared contrasts (per C, with shared-resample aggregates), H3
    and the H5 saturation check. Refuses a partial grid; honours the persisted deadline (Codex #8:
    an expiry mid-way keeps what was computed and says so, and makes no completeness claim)."""
    dl = dl or Deadline()
    cells = validate_cells(load_cells(), expected_fingerprints() if expected is None else expected)
    rows = json.loads((OUT / "rows.json").read_text())
    y = np.asarray(rows["inner_y"], dtype=int)
    groups = np.asarray(rows["inner_group"])
    source = np.asarray(rows["inner_source"])

    requested = declared_grid()
    present = [k for k in requested if k in cells]
    rep: dict = {"spec": "docs/specs/2026-10-13_v2-33-hardneg-stage0.md",
                 "framing": "DIAGNOSTIC on the legacy split, not a validation",
                 "inference": "exploratory",
                 "n_declared_cells": len(requested), "n_present": len(present),
                 "n_invalid": sum(1 for k in present if cells[k]["status"] != "OK"),
                 "invalid_reasons": {str(k): cells[k].get("reason")
                                     for k in present if cells[k]["status"] != "OK"},
                 "complete": len(present) == len(requested),
                 "gate_complete": False,
                 "na_cells": "pool_nsa x full-basis does not exist; declared in advance",
                 "bands": {"brief": BAND_BRIEF, "fold": BAND_FOLD, "fa_relative": BAND_FA_REL,
                           "boot_p_min": BOOT_P_MIN, "r7_real_cap": R7_REAL_CAP},
                 "gate": {}, "contrasts": {}}  # fmt: skip
    if not rep["complete"]:
        rep["refused"] = (f"incomplete grid: {len(present)} of {len(requested)} declared cells "
                          f"present; no verdict is computed on a partial grid")
        rep["missing"] = [str(k) for k in requested if k not in cells]
        atomic_write_json(OUT / "report.json", rep)
        return rep

    def is_ok(key):
        return key in cells and cells[key]["status"] == "OK"

    bk = ("none", PRIMARY_LAYER, "pool_nsa", 1.0)
    if is_ok(bk):
        rep["base_reference"] = {"cell": str(bk),
                                 **{k: cells[bk]["primary"][k]
                                    for k in ("package05", "package03", "brief", "eer",
                                              "n_fa_at_thr")}}  # fmt: skip

    def oof_of(key):
        z = np.load(_npz_path(key))
        return z["oof"], z["mask"]

    def picks_of(key):
        z = np.load(_npz_path(key))
        parts = [z[k] for k in z.files if k.startswith("picks_f")]
        return np.concatenate(parts) if parts else np.array([], dtype=int)

    def tick():
        if dl.expired:
            raise _DeadlineExpired

    nsa = source != "asvspoof2019"
    try:
        for key in present:
            tick()
            arm_name, layer, pool, c = key
            cell = cells[key]
            if cell["status"] != "OK":
                rep["gate"][str(key)] = {"verdict": "INVALID", "reason": cell.get("reason")}
                continue
            bkey = ("none", layer, pool, c)
            if not is_ok(bkey):
                rep["gate"][str(key)] = {"verdict": "INVALID", "reason": "matched base missing"}
                continue
            if arm_name == "none":
                rep["gate"][str(key)] = {"verdict": "BASE"}
                continue
            s_arm, mask = oof_of(key)
            s_base, _ = oof_of(bkey)
            sel = mask & nsa
            boot = group_bootstrap(y[sel], s_base[sel], s_arm[sel], groups[sel])
            g = gate_cell(ARMS_BY_NAME[arm_name], cell, cells[bkey], boot)
            g["bootstrap"] = boot
            rep["gate"][str(key)] = g

        # R5: the four-layer family only; PASS only (INDETERMINATE and FAIL do not count).
        for arm in ARMS:
            if not arm.four_layer:
                rep["gate"].setdefault("R5", {})[arm.name] = "N/A - single layer by design"
                continue
            for pool in POOLS:
                for c in C_GRID:
                    ok = [L for L in LAYERS
                          if rep["gate"].get(str((arm.name, L, pool, c)), {}).get("verdict") == "PASS"]
                    rep["gate"].setdefault("R5", {})[f"{arm.name}|{pool}|C{c:g}"] = {
                        "layers_passing": ok, "n": len(ok), "includes_L20": 20 in ok,
                        "R5": bool(len(ok) >= 3 and 20 in ok)}  # fmt: skip

        # The primary answer: four paired contrasts. Inference is PER C (C is a stratum and is never
        # averaged over): each C's (layer, pool) strata are averaged inside shared group-resamples.
        for a, b in CONTRASTS:
            per, by_c = [], {}
            for c in C_GRID:
                pairs, sel_ref = [], None
                for layer in LAYERS:
                    for pool in POOLS:
                        tick()
                        ka, kb = (a, layer, pool, c), (b, layer, pool, c)
                        if not (is_ok(ka) and is_ok(kb)):
                            continue
                        sa, mask = oof_of(ka)
                        sb, _ = oof_of(kb)
                        sel = mask & nsa
                        if sel_ref is None:
                            sel_ref = sel
                        elif not np.array_equal(sel, sel_ref):
                            raise ValueError("contrast strata are not on identical rows")
                        bt = group_bootstrap(y[sel], sb[sel], sa[sel], groups[sel])
                        per.append({"layer": layer, "pool": pool, "C": c,
                                    "delta_package05": round(cells[ka]["primary"]["package05"]
                                                             - cells[kb]["primary"]["package05"], 6),
                                    "p_better": bt["p_better"], "p_better_se": bt["p_better_se"]})
                        pairs.append((sb[sel], sa[sel]))
                tick()
                by_c[f"C{c:g}"] = (aggregate_contrast(y[sel_ref], groups[sel_ref], pairs)
                                   if pairs else None)
            deltas = [p["delta_package05"] for p in per]
            rep["contrasts"][f"{a}_minus_{b}"] = {
                "by_C": by_c, "strata": per, "n_strata": len(per),
                "mean_delta_package05_all_strata": (round(float(np.mean(deltas)), 6)
                                                    if deltas else None),
                "note": "Inference is by_C: per-C aggregate over the (layer, pool) strata inside "
                        "shared group-resamples, with P(better), its SE, median and p5/p95. C is "
                        "never averaged over; mean_delta_package05_all_strata is DESCRIPTIVE only "
                        "and pools C. INVALID strata are excluded and counted."}  # fmt: skip

        # H3: do two arms silently mine the same clips? Jaccard over unique mined rows, per stratum.
        pairs_h3 = (("hardest_both", "easiest_both"), ("hardest_both", "random_both"),
                    ("easiest_both", "random_both"), ("hardest_both", "pilot_quantile"),
                    ("hardest_reals", "easiest_reals"), ("hardest_fakes", "easiest_fakes"))
        h3: dict = {"note": "Jaccard of unique mined rows, pooled over the five outer folds; "
                            "1.0 means the arms collapsed onto the same clips"}
        for a, b in pairs_h3:
            for layer in LAYERS:
                for pool in POOLS:
                    for c in C_GRID:
                        ka, kb = (a, layer, pool, c), (b, layer, pool, c)
                        if is_ok(ka) and is_ok(kb):
                            h3[f"{a}|{b}|L{layer:02d}|{pool}|C{c:g}"] = round(
                                jaccard(picks_of(ka), picks_of(kb)), 4)
        rep["h3_jaccard"] = h3

        # H5 as frozen saturates: the count is the real-side worst decile whenever the false alarms
        # at the package-0.5 threshold already cover that decile. Reported, not redefined.
        okk = [k for k in present if is_ok(k)]
        cw = sorted({cells[k]["primary"]["confidently_wrong_fa"] for k in okk})
        fa_min = min(cells[k]["primary"]["n_fa_at_thr"] for k in okk) if okk else None
        n_real = cells[bk]["primary"]["n_real"] if is_ok(bk) else None
        rep["h5_saturation"] = {"distinct_confidently_wrong_fa_primary": cw,
                                "min_n_fa_at_thr_primary": fa_min, "n_real_primary": n_real,
                                "decile_of_reals": (n_real // 10) if n_real else None,
                                "saturated": bool(len(cw) == 1 and n_real and fa_min is not None
                                                  and fa_min >= n_real // 10)}  # fmt: skip
        rep["gate_complete"] = True
    except _DeadlineExpired:
        rep["incomplete"] = True
        rep["incomplete_note"] = ("the persisted run deadline expired during the gate; verdicts and "
                                  "contrasts computed before it are kept, nothing after it exists, "
                                  "and no completeness is claimed (Codex #8)")
    atomic_write_json(OUT / "report.json", rep)
    return rep


def log_row(rep: dict) -> dict:
    """The one experiments/log.csv row (spec F4): basis and all three readings in `notes`.
    `validation_score` is the matched reference cell -- the `none` arm at L7 / pool_nsa / C=1
    read on the NSA-only basis -- because no candidate passed and nothing is adopted."""
    n_pass = sum(1 for g in rep.get("gate", {}).values()
                 if isinstance(g, dict) and g.get("verdict") == "PASS")
    base = rep.get("base_reference", {})
    con = rep.get("contrasts", {})

    def md(k):
        v = con.get(k, {}).get("mean_delta_package05_all_strata")
        return "n/a" if v is None else f"{v:+.4f}"

    def pb(k):
        byc = con.get(k, {}).get("by_C") or {}
        return " ".join(f"{c}={v['p_better']:.3f}" for c, v in byc.items() if v)

    notes = (f"V2-33 stage 0 DIAGNOSTIC on the legacy split (41/113 speakers span >1 fold); "
             f"basis = NSA-only 16142 inner rows; reference cell = none arm L7/pool_nsa/C=1: "
             f"package05={base.get('package05')} package03={base.get('package03')} "
             f"brief={base.get('brief')} eer={base.get('eer')}; "
             f"{rep['n_declared_cells']} declared cells / {rep['n_present']} present / "
             f"{rep['n_invalid']} INVALID; gate PASS cells = {n_pass}; "
             f"mean delta package05 vs none (descriptive, C pooled): hardest "
             f"{md('hardest_both_minus_none')} easiest {md('easiest_both_minus_none')} "
             f"(negative = better); per-C aggregate P(better) hardest-none "
             f"[{pb('hardest_both_minus_none')}] easiest-none [{pb('easiest_both_minus_none')}]; "
             f"inference exploratory; nothing adopted")
    return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()), "rung": RUNG,
            "validation_score": base.get("package05", ""),
            "validation_score_clean_only": base.get("package05", ""),
            "csv_path": "outputs/v2/hardneg/report.json", "notes": notes,
            "benchmark_version": BENCHMARK_VERSION}  # fmt: skip


def append_log(rep: dict, path: Path | None = None) -> dict:
    """Lane-local appender. hearsay.submission.append_log writes 6 columns to
    submissions/log.csv and is v1 code this lane may not edit."""
    f = path or REPO / "experiments" / "log.csv"
    row = log_row(rep)
    header = f.read_text().splitlines()[0] if f.exists() else ",".join(LOG_COLUMNS)
    if header != ",".join(LOG_COLUMNS):
        raise ValueError(f"{f} header is {header!r}")
    with f.open("a") as fh:
        fh.write(",".join(f'"{row[c]}"' if "," in str(row[c]) else str(row[c])
                          for c in LOG_COLUMNS) + "\n")
    return row


def stage_report() -> None:
    rep = json.loads((OUT / "report.json").read_text()) if (OUT / "report.json").exists() \
        else stage_gate()
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("gate", "contrasts", "h3_jaccard")}, indent=1))
    for k, v in rep.get("contrasts", {}).items():
        agg = {c: (a["p_better"], a["median"]) for c, a in (v.get("by_C") or {}).items() if a}
        print(f"  {k}: by C (P(better), median delta) {agg}; descriptive all-strata mean "
              f"{v['mean_delta_package05_all_strata']} over {v['n_strata']} strata")
    h3 = {k: v for k, v in rep.get("h3_jaccard", {}).items() if k != "note"}
    if h3:
        print(f"  H3 Jaccard: min {min(h3.values()):.3f}, max {max(h3.values()):.3f} "
              f"over {len(h3)} arm pairs")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("stage", choices=("fit", "csweep", "gate", "report", "log"))
    p.add_argument("--jobs", type=int, default=N_JOBS,
                   help="BLAS thread cap. The grid runs serially (one LR fit is ~0.05 s), so the "
                        "parallelism that matters here is inside the matrix ops, not across "
                        "processes; nothing in this repo sets OMP_NUM_THREADS for CPU probes and "
                        "the donor scripts hard-code n_jobs=6 on an 18 GB Mac shared by 3 lanes.")
    p.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
    a = p.parse_args(argv)
    layers = tuple(int(x) for x in a.layers.split(","))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(max(1, min(a.jobs, N_JOBS)))
    if a.stage == "fit":
        stage_fit(layers, a.jobs)
    elif a.stage == "csweep":
        stage_csweep(layers)
    elif a.stage == "gate":
        rep = stage_gate()
        if rep.get("refused") or rep.get("incomplete"):
            print(rep.get("refused") or rep.get("incomplete_note"), file=sys.stderr)
            return 2
    elif a.stage == "log":
        row = append_log(json.loads((OUT / "report.json").read_text()))
        print(json.dumps(row, indent=1))
    else:
        stage_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
