"""V2-60: the pre-declared fusion test's learned candidate (C2) and verdict.

Run spec: docs/specs/2026-10-11_v2-benchmark-fusion.md; pre-declaration:
benchmarks/v2_60_candidates.json (K1-K13). Pure: numpy, scipy and
hearsay.benchmark only.

C2 = non-negative logistic regression on train-fold ranks of the five option-B
columns, cost-weighted at prior 0.8 (package-0.5), weights normalized to sum 1
and shrunk toward C0's weights with alpha chosen by inner-inner leave-one-fold-
out on the pre-E base; then the frozen E step. No Platt (K1, Nathan 2026-10-12).
Ported from outputs/v2/research/inner-screening/common.py:124-156 and
run_screen.py:89-116, with the K2-K6 changes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from hearsay import benchmark as B

ALPHA_GRID = (0.0, 0.25, 0.5, 0.75)
PRIOR_FAKE = 0.8
L2 = 1e-6
TIE_EPS = 1e-12


# ---------------------------------------------------------------------------
# Fit pieces
# ---------------------------------------------------------------------------


def refuse_fit_feature(cols) -> None:
    if B.E_COLUMN in cols:
        raise B.BenchmarkError(f"{B.E_COLUMN} may never be a fit feature (frozen E step only)")


def sample_weights(y: np.ndarray, prior_fake: float = PRIOR_FAKE) -> np.ndarray:
    """Per-row weights: real rows share (1 - prior), fake rows share prior
    (common.py:124-134; package-0.5 = prior 0.8)."""
    y = np.asarray(y)
    nr, nf = int((y == 0).sum()), int((y == 1).sum())
    if nr == 0 or nf == 0:
        raise B.BenchmarkError("sample_weights: both classes are required")
    return np.where(y == 0, (1 - prior_fake) / nr, prior_fake / nf)


def fit_nnlr(X: np.ndarray, y: np.ndarray, sw: np.ndarray, l2: float = L2) -> tuple[np.ndarray, float, dict]:
    """Weighted logistic regression with w >= 0 (L-BFGS-B), no penalty on b
    (common.py:143-156). Returns (w, b, {success, nit})."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    k = X.shape[1]

    def fg(p):
        w, b = p[:k], p[k]
        f = X @ w + b
        loss = np.sum(sw * (np.logaddexp(0, f) - y * f)) + l2 * np.sum(w ** 2)
        g = sw * (1.0 / (1.0 + np.exp(-f)) - y)
        return loss, np.r_[X.T @ g + 2 * l2 * w, g.sum()]

    res = minimize(fg, np.r_[np.ones(k), 0.0], jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] * k + [(None, None)], options={"maxiter": 2000})
    return res.x[:k].copy(), float(res.x[k]), {"success": bool(res.success), "nit": int(res.nit)}


def shrink(w_fit: np.ndarray, alpha: float, target: np.ndarray) -> np.ndarray:
    """K4: normalize first, then shrink: alpha*target + (1-alpha)*w_fit/sum(w_fit).
    A zero-sum fit raises (no silent fallback)."""
    w_fit = np.asarray(w_fit, dtype=float)
    s = w_fit.sum()
    if not np.isfinite(s) or s <= 0:
        raise B.BenchmarkError("C2: the non-negative fit returned all-zero weights")
    return alpha * np.asarray(target, dtype=float) + (1 - alpha) * (w_fit / s)


def rank_matrix(L: np.ndarray, ref: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Ranks of L[rows] against the sorted L[ref], per column."""
    return np.column_stack([B.rank_vs(np.sort(L[ref, j]), L[rows, j]) for j in range(L.shape[1])])


def blend_cols(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted rank sum accumulated in column order (as benchmark.blend)."""
    out = None
    for j in range(X.shape[1]):
        if w[j] == 0:
            continue
        term = w[j] * X[:, j]
        out = term if out is None else out + term
    return out


def choose_alpha(costs: dict) -> float:
    """Lowest cost; ties within 1e-12 go to the larger alpha (more shrinkage, K3)."""
    best = None
    for a in sorted(costs):
        c = costs[a]
        if best is None or c < costs[best] - TIE_EPS or abs(c - costs[best]) <= TIE_EPS:
            best = a
    return best


def inner_inner_blocks(L: np.ndarray, y: np.ndarray, fold: np.ndarray, tr: np.ndarray, prior: float = PRIOR_FAKE):
    """For each inner fold g of the training rows tr: references = tr rows not in
    g (so neither the outer fold nor g, K2); fit on those rows' ranks; rank g's
    rows against the same references. Yields (g, X_g, y_g, w_fit_g)."""
    for g in sorted(set(fold[tr])):
        ref = tr & (fold != g)
        te = tr & (fold == g)
        X_ref = rank_matrix(L, ref, ref)
        w, _, _ = fit_nnlr(X_ref, y[ref], sample_weights(y[ref], prior))
        yield g, rank_matrix(L, ref, te), y[te], w


def select_alpha(L, y, fold, tr, target, *, grid=ALPHA_GRID, prior=PRIOR_FAKE) -> tuple[float, dict]:
    """K3: alpha by pooled package-0.5 over the four g blocks, on the pre-E base."""
    blocks = list(inner_inner_blocks(L, y, fold, tr, prior))
    costs = {}
    for a in grid:
        s = np.concatenate([blend_cols(X, shrink(w, a, target)) for _, X, _, w in blocks])
        yy = np.concatenate([yb for _, _, yb, _ in blocks])
        costs[a] = B.reading(yy, s, PRIMARY)
    return choose_alpha(costs), costs


def effective_params(alpha: float, w_fit: np.ndarray) -> float:
    """K6: 1 (alpha) + (1 - alpha) * (nnz(w_fit) - 1)."""
    return 1.0 + (1 - alpha) * (int(np.count_nonzero(np.asarray(w_fit) > 0)) - 1)


def fit_c2(L, y, fold, tr, target, *, grid=ALPHA_GRID, prior=PRIOR_FAKE) -> dict:
    """Fit C2 on the rows tr (references = tr). Returns weights and diagnostics."""
    alpha, costs = select_alpha(L, y, fold, tr, target, grid=grid, prior=prior)
    X_tr = rank_matrix(L, tr, tr)
    w_fit, b, info = fit_nnlr(X_tr, y[tr], sample_weights(y[tr], prior))
    w = shrink(w_fit, alpha, target)
    return {"alpha": alpha, "alpha_costs": costs, "alpha_at_grid_max": alpha == max(grid),
            "w_fit_raw": w_fit, "w_fit_l1": float(np.abs(w_fit).sum()), "intercept": b,
            "w_fit_normalized": w_fit / w_fit.sum(), "weights": w, "effective_params": effective_params(alpha, w_fit),
            **info}


def _diff(a: dict, b: dict) -> dict:
    return {k: (None if a[k] is None or b[k] is None else a[k] - b[k]) for k in B.READINGS}


def run_c2_nested(L, y, fold, spectra, cols, target, *, population=None, grid=ALPHA_GRID,
                  prior=PRIOR_FAKE) -> dict:
    """Nested OOF C2 on inner rows: per outer fold f, fit on the other folds
    (references = those folds), score fold f, pool, then apply E (K1: no Platt).

    Overfitting (owner's P2 risk) is measured relative to C0 on the SAME rows:
    advantage_in = cost(C2) - cost(C0) on the training folds (in-sample fit,
    train-fold refs); advantage_oof = the same difference on fold f. optimism =
    advantage_oof - advantage_in > 0 means C2's edge shrinks out of fold."""
    if population is not None:
        B.require_selectable(population)
    refuse_fit_feature(cols)
    L = np.asarray(L, dtype=float)
    y, fold = np.asarray(y), np.asarray(fold)
    spectra = np.asarray(spectra, dtype=float)
    target = np.asarray(target, dtype=float)
    base = np.empty(len(y))
    folds = {}
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        fit = fit_c2(L, y, fold, tr, target, grid=grid, prior=prior)
        X_tr, X_te = rank_matrix(L, tr, tr), rank_matrix(L, tr, te)
        base[te] = blend_cols(X_te, fit["weights"])
        c2_in = B.apply_e(blend_cols(X_tr, fit["weights"]), spectra[tr])[0]
        c0_in = B.apply_e(blend_cols(X_tr, target), spectra[tr])[0]
        c2_te = B.apply_e(base[te], spectra[te])[0]
        c0_te = B.apply_e(blend_cols(X_te, target), spectra[te])[0]
        fit["in_sample"] = {k: B.reading(y[tr], c2_in, k) for k in B.READINGS}
        fit["oof_fold"] = {k: B.reading(y[te], c2_te, k) for k in B.READINGS}
        fit["gap"] = _diff(fit["oof_fold"], fit["in_sample"])
        fit["advantage_in"] = _diff(fit["in_sample"], {k: B.reading(y[tr], c0_in, k) for k in B.READINGS})
        fit["advantage_oof"] = _diff(fit["oof_fold"], {k: B.reading(y[te], c0_te, k) for k in B.READINGS})
        fit["optimism"] = _diff(fit["advantage_oof"], fit["advantage_in"])
        folds[f] = fit
    fused, mask = B.apply_e(base, spectra)
    alphas = [folds[f]["alpha"] for f in folds]
    optimism_mean = {k: (None if any(folds[f]["optimism"][k] is None for f in folds)
                         else float(np.mean([folds[f]["optimism"][k] for f in folds]))) for k in B.READINGS}
    return {"cols": list(cols), "base": base, "fused": fused, "e_mask": mask, "folds": folds,
            "alpha_spread": float(max(alphas) - min(alphas)),
            "alpha_at_grid_max_all": all(folds[f]["alpha_at_grid_max"] for f in folds),
            "optimism_mean": optimism_mean,
            "mean_weights": dict(zip(cols, np.mean([folds[f]["weights"] for f in folds], axis=0))),
            "platt": None}


def fit_c2_offinner(L_inner, y, fold, cols, target, *, grid=ALPHA_GRID, prior=PRIOR_FAKE) -> dict:
    """K5: the model applied off-inner. Refit on all inner rows with all-inner
    references; alpha by the same procedure one level up (5-fold LOFO on inner).
    It sees inner rows only."""
    refuse_fit_feature(cols)
    L = np.asarray(L_inner, dtype=float)
    tr = np.ones(len(y), dtype=bool)
    fit = fit_c2(L, np.asarray(y), np.asarray(fold), tr, target, grid=grid, prior=prior)
    fit["weights_by_col"] = dict(zip(cols, fit["weights"]))
    return fit


# ---------------------------------------------------------------------------
# Verdict (K11)
# ---------------------------------------------------------------------------

BASIS = "inner_nested_oof"
# The V2-60 pre-declaration's primary reading (benchmarks/v2_60_candidates.json "primary_reading").
# Fixed here so the closed verdict reproduces even though the benchmark default moved to package03 on Oct 20.
PRIMARY = "package05"


def _tt(x: float) -> int:
    """Integer ten-thousandths of a cost rounded to 4 decimals."""
    return round(round(float(x), 4) * 10000)


def verdict(control: dict, cand: dict, bands: dict, f4_guard: float, *, label_provisional: bool = False) -> dict:
    """WIN / TIE / LOSE from pooled nested-inner readings with E.

    control / cand: {"basis": "inner_nested_oof", "pooled": {reading: cost}, "f4_package05": cost}.
    """
    for name, d in (("control", control), ("candidate", cand)):
        if d.get("basis") != BASIS:
            raise B.BenchmarkError(f"verdict reads nested inner OOF only; {name} basis is {d.get('basis')!r}")
        missing = [k for k in bands if d["pooled"].get(k) is None] + (["f4_package05"] if d.get("f4_package05") is None
                                                                       else [])
        if missing:
            raise B.BenchmarkError(f"verdict: {name} has no value for {missing}")
    reasons = []
    worse = {k: _tt(cand["pooled"][k]) - _tt(control["pooled"][k]) for k in bands}
    band = {k: _tt(bands[k]) for k in bands}
    lose = [k for k in bands if worse[k] > band[k]]
    if lose:
        return {"verdict": "LOSE", "reasons": [f"{k} worse by {worse[k] / 1e4:.4f} > band {bands[k]}" for k in lose]}
    better = -worse[PRIMARY] > band[PRIMARY]
    f4_ok = _tt(cand["f4_package05"]) <= _tt(control["f4_package05"]) + _tt(f4_guard)
    if not better:
        reasons.append(f"package05 improvement {-worse[PRIMARY] / 1e4:.4f} not > band {bands[PRIMARY]}")
    if not f4_ok:
        reasons.append(f"f4 guard: {cand['f4_package05']:.4f} > control {control['f4_package05']:.4f} + {f4_guard}")
    if better and f4_ok:
        v = "WIN (provisional)" if label_provisional else "WIN"
        return {"verdict": v, "reasons": ["beats control beyond the package05 band; no reading regresses; f4 guard holds"]}
    return {"verdict": "TIE", "reasons": reasons}


def followon_allowed(c2_verdict: str) -> bool:
    return c2_verdict.startswith("WIN")


# ---------------------------------------------------------------------------
# Pre-declaration checks
# ---------------------------------------------------------------------------


def frozen_region(spec_text: str, start: str, end: str) -> str:
    a, b = spec_text.index(start), spec_text.index(end)
    return "\n".join(line.rstrip() for line in spec_text[a:b].replace("\r\n", "\n").split("\n"))


# The pre-declaration as committed in f8e4ca6, before any code read data. Any change to the
# candidates file (weights, bands, membership, E, alpha grid, prior) fails this pin; an authorized
# amendment needs a new constant here, visible in review, plus a dated spec amendment.
DECLARED_CANDIDATES_COMMIT = "f8e4ca6"
DECLARED_CANDIDATES_SHA256 = "aede43b3ffb90f15a81ed54bf73b505a1c6899b09320cd348a3b007b8f17fcdb"


def check_candidates(cand_path, spec_path) -> dict:
    """The candidates file must be byte-identical to the pre-declaration, and its
    recorded frozen-spec-region hash must still match the spec."""
    if hashlib.sha256(Path(cand_path).read_bytes()).hexdigest() != DECLARED_CANDIDATES_SHA256:
        raise B.BenchmarkError(f"candidates file differs from the pre-declaration ({DECLARED_CANDIDATES_COMMIT})")
    cand = json.loads(Path(cand_path).read_text())
    r = cand["spec_frozen_region"]
    region = frozen_region(Path(spec_path).read_text(), r["start"], r["end"])
    sha = hashlib.sha256(region.encode()).hexdigest()
    if sha != r["sha256"]:
        raise B.BenchmarkError("frozen spec region changed since the candidates were declared")
    if tuple(cand["candidates"]["C2"]["alpha_grid"]) != ALPHA_GRID or cand["candidates"]["C2"]["prior_fake"] != PRIOR_FAKE:
        raise B.BenchmarkError("candidates JSON C2 settings differ from the module constants")
    if cand["candidates"]["C2"].get("platt") is not False:
        raise B.BenchmarkError("C2 must carry platt: false (K1)")
    return cand


def committed_blob_sha(path, repo: Path = B.REPO) -> str:
    """The file's blob sha at HEAD; raises if untracked or modified."""
    path = Path(path)
    rel = path.resolve().relative_to(Path(repo).resolve()).as_posix()
    if subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=repo, capture_output=True, check=False).returncode != 0:
        raise B.BenchmarkError(f"{rel} is not committed")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=repo, check=False).returncode != 0:
        raise B.BenchmarkError(f"{rel} has uncommitted changes")
    return subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def check_readout_fresh(readout: dict, *, candidates_blob: str, registry_sha: str) -> None:
    """A stored readout is usable only if its candidates and registry hashes match now."""
    if readout.get("candidates_blob_sha") != candidates_blob:
        raise B.BenchmarkError("readout was produced under a different candidates file")
    if readout.get("registry_sha256") != registry_sha:
        raise B.BenchmarkError("readout was produced under a different registry")
