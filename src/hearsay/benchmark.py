"""V2-01-lite: the shared v2 benchmark harness.

One scorer for any per-clip score file or fused rule, on every population we
hold today, under the three metric readings plus EER, with nested rank
references and a class-stratified cluster bootstrap.

Run spec: docs/specs/2026-10-11_v2-benchmark-fusion.md (Interface v1).
Plan: docs/reports/2026-10-11_v2-benchmark-fusion-plan.md (K1-K13).

Pure module: numpy, pandas, scipy, sklearn and hearsay.metrics only. It imports
nothing from scripts/, and never torch or lightgbm.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from hearsay.metrics import eer, min_cost

BENCHMARK_VERSION = "v1"
REPO = Path(__file__).resolve().parents[2]

SPLITS = ("inner_oof", "holdout", "test", "itw")
SPLIT_ALIASES = {"stress": "itw"}
POP_SPLIT = {"inner": "inner_oof", "holdout": "holdout", "test": "test", "itw": "itw"}
ROLES = ("selectable", "read_once", "report_only", "in_sample", "reference")
COHORT_KINDS = ("none", "mp3", "noise20", "speed", "shift1")

# name -> (pi_synth, c_fa, c_miss), passed explicitly to min_cost every time.
READINGS = {
    "brief": (0.3, 4.0, 1.0),
    "package05": (0.5, 1.0, 4.0),
    "package03": (0.3, 1.0, 4.0),
}
PRIMARY = "package05"

E_COLUMN = "spectra_aasist"
E_RULE = {"spectra_logit_below": -3.0, "base_above": 0.5, "multiply_by": 0.5}
SHIP_WEIGHTS = {"m1b_v3": 0.6, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.2}

ASV19_EVAL_MARKERS = ("ASVspoof2019_LA_eval", "LA_E_", "asv19_eval_probe")

N_INNER, N_HOLDOUT, N_TEST, N_ITW = 16142, 3858, 1671, 3000


class BenchmarkError(ValueError):
    """Any contract violation. Raised, never silently repaired."""


# ---------------------------------------------------------------------------
# Roots and canonical keys
# ---------------------------------------------------------------------------


def main_root(repo: Path = REPO) -> Path:
    """The main checkout's root (the git common dir's parent); REPO if not in git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return Path(out).parent
    except (OSError, subprocess.CalledProcessError):
        return repo


_MAIN = None


def _main() -> Path:
    global _MAIN
    if _MAIN is None:
        _MAIN = main_root()
    return _MAIN


def canonical_key(path, *, main: Path | None = None, repo: Path | None = None) -> str:
    """Repo-relative POSIX key for a clip path. Lexical only: never resolves symlinks.

    Accepts an absolute path under the main checkout or any of its
    `.claude/worktrees/<name>/` worktrees (or `repo`), or a repo-relative path.
    Refuses paths outside the repo and ASVspoof 2019 LA eval paths.
    """
    main = Path(main) if main is not None else _main()
    repo = Path(repo) if repo is not None else REPO
    p = str(path).replace("\\", "/")
    if not p:
        raise BenchmarkError("empty path")
    if p.startswith("/"):
        p = posixpath.normpath(p)
        key = None
        wt_prefix = posixpath.join(str(main), ".claude/worktrees") + "/"
        if p.startswith(wt_prefix):
            rest = p[len(wt_prefix):]
            if "/" not in rest:
                raise BenchmarkError(f"path is a worktree root, not a clip: {path}")
            key = rest.split("/", 1)[1]
        else:
            for root in (str(main), str(repo)):
                root = root.rstrip("/") + "/"
                if p.startswith(root):
                    key = p[len(root):]
                    break
        if key is None:
            raise BenchmarkError(f"path outside the repo: {path}")
    else:
        key = posixpath.normpath(p)
        if key.startswith("../") or key == "..":
            raise BenchmarkError(f"path escapes the repo: {path}")
    if any(m in key for m in ASV19_EVAL_MARKERS):
        raise BenchmarkError(f"ASVspoof 2019 LA eval is barred from every path: {path}")
    return key


def code_sha(repo: Path = REPO) -> str:
    """HEAD plus a hash of any uncommitted tracked changes ('<head>+dirty:<hash>')."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head if not diff else f"{head}+dirty:{hashlib.sha256(diff).hexdigest()[:16]}"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path) -> str:
    """Hash of a directory: sorted relative file names with each file's sha256."""
    path = Path(path)
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(b"\0")
        h.update(sha256_file(f).encode())
        h.update(b"\n")
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Score files (Interface v1)
# ---------------------------------------------------------------------------


def _fold_str(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else str(v)
    s = str(v)
    try:
        f = float(s)
        if f.is_integer() and s.replace(".", "", 1).replace("-", "", 1).isdigit():
            return str(int(f))
    except ValueError:
        pass
    return s


@dataclass(frozen=True)
class ScoreFile:
    """A validated score file: one logit per canonical key, per split."""

    name: str
    df: pd.DataFrame  # columns: key, split, fold, logit
    shas: dict
    itw_source: str | None = None
    extra_columns: tuple = ()

    def split(self, split: str) -> pd.Series | None:
        d = self.df[self.df["split"] == split]
        if d.empty:
            return None
        return d.set_index("key")["logit"]

    def folds(self, split: str) -> pd.Series | None:
        d = self.df[self.df["split"] == split]
        if d.empty:
            return None
        return d.set_index("key")["fold"]


def read_itw_override(path, *, main=None, repo=None) -> tuple[pd.DataFrame, tuple]:
    """ITW scores kept in a separate file: path + logit required; split optional
    and in {itw, stress}; any other columns are ignored and listed."""
    df = pd.read_csv(path)
    for c in ("path", "logit"):
        if c not in df.columns:
            raise BenchmarkError(f"{path}: ITW override missing column {c!r}")
    if "split" in df.columns:
        bad = set(df["split"].astype(str)) - {"itw", "stress"}
        if bad:
            raise BenchmarkError(f"{path}: ITW override has split values {sorted(bad)}")
    extra = tuple(sorted(set(df.columns) - {"path", "logit", "split"}))
    logit = pd.to_numeric(df["logit"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(logit).all():
        raise BenchmarkError(f"{path}: non-finite logit in ITW override")
    keys = [canonical_key(p, main=main, repo=repo) for p in df["path"]]
    out = pd.DataFrame({"key": keys, "split": "itw", "fold": "", "logit": logit})
    if out["key"].duplicated().any():
        raise BenchmarkError(f"{path}: duplicate paths in ITW override")
    return out, extra


def read_score_file(path, *, name: str | None = None, itw=None, main=None, repo=None) -> ScoreFile:
    """Read and validate a score file per Interface v1. Refuses, never repairs."""
    path = Path(path)
    df = pd.read_csv(path)  # pandas default float parser (K9)
    missing = [c for c in ("path", "split", "logit") if c not in df.columns]
    if missing:
        raise BenchmarkError(f"{path}: missing required column(s) {missing}")
    split = df["split"].astype(str).replace(SPLIT_ALIASES)
    bad = sorted(set(split) - set(SPLITS))
    if bad:
        raise BenchmarkError(f"{path}: unknown split value(s) {bad}")
    logit = pd.to_numeric(df["logit"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(logit).all():
        raise BenchmarkError(f"{path}: non-finite logit on {int((~np.isfinite(logit)).sum())} row(s)")
    if "score" in df.columns:
        score = pd.to_numeric(df["score"], errors="coerce").to_numpy(dtype=float)
        dev = np.abs(score - expit(logit))
        if not np.all(dev <= 1e-6):
            raise BenchmarkError(f"{path}: score != sigmoid(logit) (max dev {np.nanmax(dev):.3g})")
    fold = df["fold"].map(_fold_str) if "fold" in df.columns else pd.Series([""] * len(df))
    keys = [canonical_key(p, main=main, repo=repo) for p in df["path"]]
    out = pd.DataFrame({"key": keys, "split": split.to_numpy(), "fold": fold.to_numpy(), "logit": logit})
    if out["key"].duplicated().any():
        raise BenchmarkError(f"{path}: {int(out['key'].duplicated().sum())} duplicate path(s)")
    shas = {path.as_posix(): sha256_file(path)}
    extra = ()
    itw_source = None
    if itw is not None:
        if (out["split"] == "itw").any():
            raise BenchmarkError(f"{path}: has its own itw rows and an ITW override was also given")
        o, extra = read_itw_override(itw, main=main, repo=repo)
        out = pd.concat([out, o], ignore_index=True)
        if out["key"].duplicated().any():
            raise BenchmarkError(f"{itw}: ITW override paths collide with the score file")
        shas[Path(itw).as_posix()] = sha256_file(itw)
        itw_source = Path(itw).as_posix()
    return ScoreFile(name or path.stem, out, shas, itw_source, extra)


def read_cohort_file(path, *, column: str | None = None, main=None, repo=None) -> pd.DataFrame:
    """A cohort score file keyed by (path, kind). Uses `logit`, or the named
    column of a legacy wide file. Precomputed fused/fused_base/e_applied columns
    (transductive SHIP) are never read."""
    df = pd.read_csv(path)
    col = "logit" if column is None else column
    for c in ("path", "kind", col):
        if c not in df.columns:
            raise BenchmarkError(f"{path}: cohort file missing column {c!r}")
    if "error" in df.columns:
        err = df["error"].notna() & (df["error"].astype(str).str.len() > 0)
        if err.any():
            raise BenchmarkError(f"{path}: {int(err.sum())} row(s) carry an error")
    kinds = set(df["kind"].astype(str))
    bad = kinds - set(COHORT_KINDS)
    if bad:
        raise BenchmarkError(f"{path}: unknown cohort kind(s) {sorted(bad)}")
    logit = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(logit).all():
        raise BenchmarkError(f"{path}: non-finite cohort logit")
    out = pd.DataFrame({
        "key": [canonical_key(p, main=main, repo=repo) for p in df["path"]],
        "kind": df["kind"].astype(str).to_numpy(),
        "logit": logit,
    })
    if out.duplicated(["key", "kind"]).any():
        raise BenchmarkError(f"{path}: duplicate (path, kind) rows")
    counts = out.groupby("kind").size()
    if set(counts.index) != set(COHORT_KINDS) or counts.nunique() != 1:
        raise BenchmarkError(f"{path}: partial cohort (rows per kind {counts.to_dict()})")
    return out


# ---------------------------------------------------------------------------
# Labels and populations
# ---------------------------------------------------------------------------


@dataclass
class Population:
    name: str
    role: str
    keys: np.ndarray
    y: np.ndarray | None
    fold: np.ndarray | None = None
    cluster: np.ndarray | None = None
    stratum: np.ndarray | None = None

    def __post_init__(self):
        if self.role not in ROLES:
            raise BenchmarkError(f"population {self.name}: unknown role {self.role!r}")

    @property
    def n(self) -> int:
        return len(self.keys)


def _nsa_cluster(row) -> tuple[str, str]:
    if row["label"] == "spoof":
        return f"gen:{row['generator']}", "spoof"
    if row["source"] == "ljspeech":
        return "lj", "lj"
    return f"spk:{row['speaker']}", "libri"


EXPECTED_ROWS = {"inner": N_INNER, "holdout": N_HOLDOUT, "itw": N_ITW, "test": N_TEST}


def load_populations(root: Path = REPO, *, main=None, expect: dict | None = EXPECTED_ROWS) -> dict[str, Population]:
    """Inner (16,142), holdout (3,858), ITW (3,000) and test (1,671) with keys,
    labels, folds, bootstrap clusters and strata."""
    root = Path(root)
    folds = pd.read_csv(root / "splits/nsa_folds.csv", dtype=str)
    folds["key"] = [canonical_key(p, main=main, repo=root) for p in folds["path"]]
    if folds["key"].duplicated().any():
        raise BenchmarkError("duplicate keys in splits/nsa_folds.csv")
    cl = folds.apply(_nsa_cluster, axis=1, result_type="expand")
    folds["cluster"], folds["stratum"] = cl[0], cl[1]
    y = (folds["label"] == "spoof").astype(int).to_numpy()
    inner = (folds["fold"] != "holdout").to_numpy()
    pops = {
        "inner": Population("inner", "selectable", folds["key"].to_numpy()[inner], y[inner],
                            folds["fold"].to_numpy()[inner], folds["cluster"].to_numpy()[inner],
                            folds["stratum"].to_numpy()[inner]),
        "holdout": Population("holdout", "read_once", folds["key"].to_numpy()[~inner], y[~inner],
                              folds["fold"].to_numpy()[~inner], folds["cluster"].to_numpy()[~inner],
                              folds["stratum"].to_numpy()[~inner]),
    }
    itw = pd.read_csv(root / "outputs/manifests/itw_stress.csv", dtype=str)
    iy = (itw["label"] == "spoof").astype(int).to_numpy()
    pops["itw"] = Population(
        "itw", "report_only",
        np.array([canonical_key(p, main=main, repo=root) for p in itw["path"]]), iy,
        None, np.array([f"{lab}:{spk}" for lab, spk in zip(itw["label"], itw["speaker"])]),
        np.where(iy == 1, "itw-spoof", "itw-real"))
    test = pd.read_csv(root / "outputs/manifests/nsa_test.csv", dtype=str)
    pops["test"] = Population(
        "test", "report_only", np.array([canonical_key(p, main=main, repo=root) for p in test["path"]]), None)
    for k, pop in pops.items():
        if expect is not None and pop.n != expect[k]:
            raise BenchmarkError(f"population {k} has {pop.n} rows, expected {expect[k]}")
        if len(set(pop.keys)) != pop.n:
            raise BenchmarkError(f"population {k} has duplicate keys")
    return pops


def align(sf: ScoreFile, pop: Population) -> np.ndarray | None:
    """Exact-coverage join of a score file onto a population. None if the split
    is absent (reported as 'not scored'); raises on any missing or extra row."""
    s = sf.split(POP_SPLIT[pop.name])
    if s is None:
        return None
    have, want = set(s.index), set(pop.keys)
    missing, extra = want - have, have - want
    if missing or extra:
        raise BenchmarkError(
            f"{sf.name} on {pop.name}: coverage mismatch ({len(missing)} missing, {len(extra)} extra; "
            f"exactly {pop.n} rows required)")
    return s.reindex(pop.keys).to_numpy(dtype=float)


def check_inner_folds(sf: ScoreFile, pop: Population) -> None:
    """Inner OOF must be out-of-fold by splits/nsa_folds.csv folds (16,142 basis)."""
    f = sf.folds("inner_oof")
    if f is None:
        return
    f = f.reindex(pop.keys)
    have = f.to_numpy()
    if (have == "").all():
        return
    if not np.array_equal(have, pop.fold):
        n = int((have != pop.fold).sum())
        raise BenchmarkError(f"{sf.name}: inner fold ids disagree with splits/nsa_folds.csv on {n} row(s)")


def direction_guard(y: np.ndarray, s: np.ndarray, name: str = "column") -> float:
    auc = float(roc_auc_score(y, s))
    if auc < 0.5:
        raise BenchmarkError(f"{name}: pooled inner AUC {auc:.4f} < 0.5 (inverted direction?)")
    return auc


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def reading(y: np.ndarray, s: np.ndarray, name: str) -> float | None:
    """One named minDCF reading. None for a single-class subset."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    pi, cfa, cmiss = READINGS[name]
    return float(min_cost(y, np.asarray(s, dtype=float), pi, c_fa=cfa, c_miss=cmiss))


def readings(y: np.ndarray, s: np.ndarray) -> dict:
    out = {k: reading(y, s, k) for k in READINGS}
    out["eer"] = float(eer(np.asarray(y), np.asarray(s, dtype=float))) if len(np.unique(y)) > 1 else None
    return out


def counts_at(y: np.ndarray, s: np.ndarray, name: str) -> dict:
    """FA / miss counts at the minimum-cost threshold of one reading. Always
    carries the reading and the threshold (+inf = call everything real)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return {"reading": name, "threshold": None, "n_fa": None, "n_miss": None}
    pi, cfa, cmiss = READINGS[name]
    thr = np.r_[np.unique(s), np.inf]
    real, spoof = np.sort(s[y == 0]), np.sort(s[y == 1])
    n_fa = len(real) - np.searchsorted(real, thr, side="left")
    n_miss = np.searchsorted(spoof, thr, side="left")
    cost = (cfa * (n_fa / len(real)) * (1 - pi) + cmiss * (n_miss / len(spoof)) * pi) / min(cfa * (1 - pi), cmiss * pi)
    i = int(np.argmin(cost))
    t = float(thr[i])
    return {"reading": name, "threshold": "inf" if not np.isfinite(t) else t,
            "n_fa": int(n_fa[i]), "n_miss": int(n_miss[i]),
            "n_real": len(real), "n_spoof": len(spoof), "cost": float(min(cost[i], 1.0))}


def fold_table(y: np.ndarray, s: np.ndarray, fold: np.ndarray) -> dict:
    """Pooled, per fold, and macro (each fold rounded to 4 decimals, then mean,
    then rounded; as outputs/v2/research/inner-screening/extra.py:74-78)."""
    y, s, fold = np.asarray(y), np.asarray(s, dtype=float), np.asarray(fold)
    out = {"pooled": readings(y, s), "per_fold": {}, "macro": {}}
    for f in sorted(set(fold)):
        m = fold == f
        out["per_fold"][f] = readings(y[m], s[m])
    for k in list(READINGS) + ["eer"]:
        vals = [round(v[k], 4) for v in out["per_fold"].values() if v[k] is not None]
        out["macro"][k] = round(float(np.mean(vals)), 4) if vals else None
    return out


# ---------------------------------------------------------------------------
# Ranks and fusion primitives
# ---------------------------------------------------------------------------


def rank_vs(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rank of x against a sorted reference: searchsorted side 'left' / len(ref)
    (scripts/fuse_sweep_v3.py:105-106; src/hearsay/pipeline.py:285-288)."""
    return np.searchsorted(ref_sorted, x, side="left") / len(ref_sorted)


def nested_ranks(logit: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Each fold's rows ranked against the other folds' rows only (rule 6)."""
    logit, fold = np.asarray(logit, dtype=float), np.asarray(fold)
    out = np.empty(len(logit))
    for f in sorted(set(fold)):
        te = fold == f
        tr = ~te
        if not tr.any():
            raise BenchmarkError("nested ranks need at least two folds")
        out[te] = rank_vs(np.sort(logit[tr]), logit[te])
    return out


def full_ranks(ref_logit: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rank against all inner rows. Correct for holdout, ITW, test and cohorts;
    transductive (leaky) if applied to inner rows themselves."""
    return rank_vs(np.sort(np.asarray(ref_logit, dtype=float)), np.asarray(x, dtype=float))


def validate_weights(weights: dict, known: set | None = None) -> None:
    if not weights:
        raise BenchmarkError("empty weights")
    if E_COLUMN in weights:
        raise BenchmarkError(f"{E_COLUMN} may never carry a weight (it is the frozen E step only)")
    w = np.array(list(weights.values()), dtype=float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise BenchmarkError(f"weights must be finite and non-negative: {weights}")
    if abs(w.sum() - 1.0) > 1e-9:
        raise BenchmarkError(f"weights must sum to 1 (got {w.sum():.12f})")
    if known is not None and set(weights) - set(known):
        raise BenchmarkError(f"unknown column(s) {sorted(set(weights) - set(known))}")


def blend(ranks: dict, weights: dict) -> np.ndarray:
    """Weighted rank sum, accumulated in the weights' order (float parity with
    the shipped pipeline). Zero-weight columns are skipped."""
    out = None
    for col, w in weights.items():
        if w == 0:
            continue
        term = w * ranks[col]
        out = term if out is None else out + term
    if out is None:
        raise BenchmarkError("all weights are zero")
    return out


def apply_e(base: np.ndarray, spectra_logit: np.ndarray, *, below: float = -3.0,
            base_above: float = 0.5, mult: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The frozen E step: base * 0.5 where Spectra logit < -3 and base > 0.5
    (both strict). Raises on non-finite input (NaN never passes silently)."""
    base = np.asarray(base, dtype=float)
    m3 = np.asarray(spectra_logit, dtype=float)
    if base.shape != m3.shape:
        raise BenchmarkError("E: base and Spectra arrays differ in length")
    if not (np.isfinite(base).all() and np.isfinite(m3).all()):
        raise BenchmarkError("E: non-finite input")
    mask = (m3 < below) & (base > base_above)
    out = base.copy()
    out[mask] = out[mask] * mult
    return out, mask


def shipped_probability(fused: np.ndarray, platt: dict) -> np.ndarray:
    """v1's monotone output map: 0.001 + 0.999 * sigmoid(a*fused + b + prior_shift)."""
    z = platt["a"] * np.asarray(fused, dtype=float) + platt["b"] + platt["prior_shift"]
    return 0.001 + 0.999 * expit(z)


@dataclass(frozen=True)
class Rule:
    """A fixed rank-blend rule over score columns, optionally followed by E."""

    name: str
    weights: dict
    e: bool = True

    def definition(self) -> dict:
        return {"kind": "fixed", "weights": dict(self.weights), "e": self.e}


# ---------------------------------------------------------------------------
# The benchmark context
# ---------------------------------------------------------------------------


@dataclass
class Bench:
    """Populations plus aligned per-population logits for every loaded column."""

    pops: dict
    cols: dict = field(default_factory=dict)  # name -> {pop: array or None}
    files: dict = field(default_factory=dict)  # name -> ScoreFile

    def add(self, sf: ScoreFile, *, check_direction: bool = True) -> None:
        aligned = {p: align(sf, pop) for p, pop in self.pops.items()}
        if aligned["inner"] is None:
            raise BenchmarkError(f"{sf.name}: no inner_oof rows; every column needs the inner population")
        check_inner_folds(sf, self.pops["inner"])
        if check_direction:
            direction_guard(self.pops["inner"].y, aligned["inner"], sf.name)
        self.cols[sf.name] = aligned
        self.files[sf.name] = sf

    def ranks(self, col: str, pop: str, refs: str = "nested") -> np.ndarray | None:
        x = self.cols[col][pop]
        if x is None:
            return None
        inner = self.cols[col]["inner"]
        if pop == "inner":
            if refs == "nested":
                return nested_ranks(inner, self.pops["inner"].fold)
            if refs == "transductive":
                return full_ranks(inner, inner)
            raise BenchmarkError(f"unknown refs {refs!r}")
        return full_ranks(inner, x)

    def score_rule(self, rule: Rule, pop: str, refs: str = "nested") -> dict:
        """Base and with-E scores for one population, or a 'not scored' reason."""
        validate_weights(rule.weights)
        need = [c for c, w in rule.weights.items() if w > 0]
        missing = [c for c in need if c not in self.cols or self.cols[c][pop] is None]
        if rule.e and (E_COLUMN not in self.cols or self.cols[E_COLUMN][pop] is None):
            missing.append(E_COLUMN)
        if missing:
            return {"scored": False, "reason": f"no scores for {missing} on {pop}"}
        r = {c: self.ranks(c, pop, refs) for c in need}
        base = blend(r, {c: rule.weights[c] for c in rule.weights})
        if rule.e:
            fused, mask = apply_e(base, self.cols[E_COLUMN][pop])
        else:
            fused, mask = base, np.zeros(len(base), dtype=bool)
        return {"scored": True, "base": base, "fused": fused, "e_mask": mask}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_strata(cluster, stratum, lj_mode: str = "fixed") -> tuple[np.ndarray, dict]:
    """(indices always included once, {stratum: [cluster index arrays]})."""
    cluster = np.asarray(cluster).astype(str)
    st = np.asarray(stratum).astype(str).copy()
    if lj_mode not in ("fixed", "cluster"):
        raise BenchmarkError(f"unknown lj_mode {lj_mode!r}")
    if lj_mode == "cluster":
        st[st == "lj"] = "libri"
    fixed_idx = np.flatnonzero(st == "lj") if lj_mode == "fixed" else np.array([], dtype=int)
    groups = {}
    for s_name in sorted(set(st) - ({"lj"} if lj_mode == "fixed" else set())):
        m = st == s_name
        groups[s_name] = [np.flatnonzero(m & (cluster == c)) for c in sorted(set(cluster[m]))]
    return fixed_idx, groups


def _draws(fixed_idx: np.ndarray, groups: dict, n: int, seed: int):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        parts = [fixed_idx]
        for s_name in sorted(groups):
            g = groups[s_name]
            pick = rng.integers(0, len(g), len(g))
            parts.extend(g[i] for i in pick)
        yield np.concatenate(parts)


def bootstrap_draws(cluster, stratum, *, n: int, seed: int = 0, lj_mode: str = "fixed"):
    """The index draws cluster_bootstrap uses (exposed for tests)."""
    fixed_idx, groups = bootstrap_strata(cluster, stratum, lj_mode)
    return list(_draws(fixed_idx, groups, n, seed))


def cluster_bootstrap(y, s_control, s_cand, cluster, stratum, *, n: int = 2000, seed: int = 0,
                      lj_mode: str = "fixed", readings_: tuple = tuple(READINGS)) -> dict:
    """Paired, class-stratified cluster bootstrap of cost(cand) - cost(control).

    Strata are resampled independently (clusters with replacement). lj_mode
    'fixed': the LJ stratum is present once in every draw (CIs conditional on
    LJ); 'cluster': LJ is one cluster among the real clusters (sensitivity).
    A stratum with <= 2 clusters makes the bootstrap degenerate (no CI).
    """
    y = np.asarray(y)
    a = np.asarray(s_control, dtype=float)
    b = np.asarray(s_cand, dtype=float)
    cluster = np.asarray(cluster).astype(str)
    stratum = np.asarray(stratum).astype(str)
    if not (len(y) == len(a) == len(b) == len(cluster) == len(stratum)):
        raise BenchmarkError("bootstrap: arrays differ in length")
    fixed_idx, groups = bootstrap_strata(cluster, stratum, lj_mode)
    small = {str(k): len(v) for k, v in groups.items() if len(v) <= 2}
    point = {k: (reading(y, b, k), reading(y, a, k)) for k in readings_}
    point = {k: (None if None in v else v[0] - v[1]) for k, v in point.items()}
    if small:
        return {"degenerate": True, "reason": f"stratum cluster counts {small} (<= 2)", "point": point,
                "lj_mode": lj_mode}
    diffs = {k: [] for k in readings_}
    skipped = 0
    lj_rows = stratum == "lj"
    missing_lj = 0
    for ix in _draws(fixed_idx, groups, n, seed):
        yy = y[ix]
        if lj_rows.any() and not lj_rows[ix].any():
            missing_lj += 1
        if len(np.unique(yy)) < 2:
            skipped += 1
            continue
        for k in readings_:
            diffs[k].append(reading(yy, b[ix], k) - reading(yy, a[ix], k))
    if skipped == n:
        raise BenchmarkError("bootstrap: every draw was single-class")
    out = {"degenerate": False, "n": n, "seed": seed, "lj_mode": lj_mode, "skipped_single_class": skipped,
           "draws_missing_lj": missing_lj, "point": point}
    for k in readings_:
        d = np.asarray(diffs[k])
        out[k] = {"p5": float(np.percentile(d, 5)), "p50": float(np.percentile(d, 50)),
                  "p95": float(np.percentile(d, 95)), "p_better": float(np.mean(d < 0))}
    return out


def bootstrap_ci(y, s, cluster, stratum, *, n: int = 2000, seed: int = 0, lj_mode: str = "fixed",
                 readings_: tuple = tuple(READINGS)) -> dict:
    """Marginal class-stratified cluster bootstrap interval (p5, p50, p95) of each
    reading for one score vector. Degenerate (no CI) when a resampled stratum has
    <= 2 clusters (holdout: 2 spoof generators; inner f3/f4: 1)."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    cluster = np.asarray(cluster).astype(str)
    stratum = np.asarray(stratum).astype(str)
    if not (len(y) == len(s) == len(cluster) == len(stratum)):
        raise BenchmarkError("bootstrap: arrays differ in length")
    fixed_idx, groups = bootstrap_strata(cluster, stratum, lj_mode)
    small = {str(k): len(v) for k, v in groups.items() if len(v) <= 2}
    if small:
        return {"degenerate": True, "reason": f"stratum cluster counts {small} (<= 2)", "lj_mode": lj_mode}
    vals = {k: [] for k in readings_}
    skipped = 0
    lj_rows = stratum == "lj"
    missing_lj = 0
    for ix in _draws(fixed_idx, groups, n, seed):
        if lj_rows.any() and not lj_rows[ix].any():
            missing_lj += 1
        yy = y[ix]
        if len(np.unique(yy)) < 2:
            skipped += 1
            continue
        for k in readings_:
            vals[k].append(reading(yy, s[ix], k))
    if skipped == n:
        raise BenchmarkError("bootstrap: every draw was single-class")
    out = {"degenerate": False, "n": n, "seed": seed, "lj_mode": lj_mode, "skipped_single_class": skipped,
           "draws_missing_lj": missing_lj}
    for k in readings_:
        v = np.asarray(vals[k])
        out[k] = {"p5": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
                  "p95": float(np.percentile(v, 95))}
    return out


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------


def dominance(ranks_real: dict, fused_real: np.ndarray, weights: dict) -> dict:
    """Weight share and Spearman of each column with the fused score on real
    rows. Flag: Spearman >= 0.95 while the weight share is < 0.5 (K12)."""
    out = {}
    for col, r in ranks_real.items():
        rho = float(spearmanr(r, fused_real).statistic)
        share = float(weights.get(col, 0.0))
        out[col] = {"weight_share": share, "spearman_real": rho, "flag": bool(rho >= 0.95 and share < 0.5)}
    return out


def shipped_tsv_path() -> Path:
    return _main() / "submissions/CrossExam_predictions.tsv"


def agreement_with_shipped(test_keys: np.ndarray, fused: np.ndarray, threshold, control_fused=None,
                   control_threshold=None, tsv: Path | None = None) -> dict:
    """Unlabelled test set: share above the inner-chosen threshold, Spearman with
    the shipped TSV, and decision flips vs the control rule."""
    tsv = Path(tsv) if tsv is not None else shipped_tsv_path()
    t = pd.read_csv(tsv, sep="\t")
    names = [Path(k).name for k in test_keys]
    shipped = t.set_index("filename")["cm-score"].reindex(names).to_numpy(dtype=float)
    if not np.isfinite(shipped).all():
        raise BenchmarkError("shipped TSV does not cover the test population")
    thr = np.inf if threshold == "inf" else float(threshold)
    above = fused >= thr
    out = {"share_above_threshold": float(above.mean()), "threshold": threshold,
           "spearman_vs_shipped_tsv": float(spearmanr(fused, shipped).statistic)}
    if control_fused is not None:
        cthr = np.inf if control_threshold == "inf" else float(control_threshold)
        out["flips_vs_control"] = int((above != (control_fused >= cthr)).sum())
    return out


def clean(obj):
    """JSON-safe: numpy scalars/arrays to Python, NaN/inf to strings."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else str(f)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def dumps(obj) -> str:
    return json.dumps(clean(obj), sort_keys=True, indent=1)


# ---------------------------------------------------------------------------
# Registry and roles
# ---------------------------------------------------------------------------


def load_registry(path) -> dict:
    """benchmarks/v1.yaml: JSON content (valid YAML 1.2), parsed with the stdlib."""
    reg = json.loads(Path(path).read_text())
    for name, p in reg.get("populations", {}).items():
        if p.get("role") not in ROLES:
            raise BenchmarkError(f"registry population {name}: unknown role {p.get('role')!r}")
    return reg


def check_registry(reg: dict, root: Path = REPO, *, trees: bool = True) -> list[str]:
    """Problems found comparing the registry to disk (empty list = clean)."""
    root = Path(root)
    problems = []
    for rel, meta in sorted(reg.get("inputs", {}).items()):
        p = root / rel
        if not p.exists():
            problems.append(f"missing input {rel}")
            continue
        if meta.get("kind") == "tree":
            if trees and sha256_tree(p) != meta["sha256"]:
                problems.append(f"sha256 mismatch (tree) {rel}")
        elif sha256_file(p) != meta["sha256"]:
            problems.append(f"sha256 mismatch {rel}")
    return problems


def require_selectable(pop: Population | str, role: str | None = None, *, rule_columns=()) -> None:
    """Selection may only read the selectable population. MLAAD is in-sample for
    any rule that contains m5_xlsr_ft."""
    name = pop.name if isinstance(pop, Population) else pop
    role = pop.role if isinstance(pop, Population) else role
    if name == "mlaad" and "m5_xlsr_ft" in rule_columns:
        raise BenchmarkError("MLAAD is in-sample for M5: readout only, never selection")
    if role != "selectable":
        raise BenchmarkError(f"population {name} has role {role!r}: report-only, never used for selection")


# ---------------------------------------------------------------------------
# Holdout ledger (K8)
# ---------------------------------------------------------------------------


class HoldoutAlreadyRead(BenchmarkError):
    pass


REFERENCE_COLUMN_FILES = {
    "m1b_v3": "outputs/detector_scores/m1b_v3.csv",
    "handcrafted_v5": "outputs/detector_scores/handcrafted_v5.csv",
    "m5_xlsr_ft": "outputs/detector_scores/m5_xlsr_ft.csv",
    "spectra_aasist": "outputs/detector_scores/spectra_aasist.csv",
}


def column_def(name: str, sha: str) -> dict:
    """Ledger definition of a single column: its name AND its file's sha256."""
    return {"kind": "column", "name": name, "sha256": sha}


def ship_def(input_shas: dict) -> dict:
    return {"kind": "fixed", "weights": dict(SHIP_WEIGHTS), "e": True, "refs": "full", "inputs": dict(input_shas)}


def reference_defs(pinned: dict) -> list:
    """The two published holdout references, bound to the registry's file pins
    (`pinned`: registry inputs, rel path -> {"sha256": ...})."""
    sha = {c: pinned[rel]["sha256"] for c, rel in REFERENCE_COLUMN_FILES.items()}
    return [column_def("m1b_v3", sha["m1b_v3"]), ship_def(sha)]


def ledger_key(rule_defs: list, holdout_sha: str) -> str:
    """Key = sorted rule definitions + the holdout population's label/fold sha.
    The benchmark version is deliberately excluded (a registry bump cannot mint
    a new holdout read)."""
    defs = sorted(json.dumps(d, sort_keys=True) for d in rule_defs)
    return sha256_json({"rules": defs, "holdout": holdout_sha})


def is_reference(rule_defs: list, pinned: dict | None) -> bool:
    """Exempt only if every definition is one of the two references with the
    registry-pinned file hashes. Without pins nothing is exempt."""
    if not pinned or not rule_defs:
        return False
    ref = {json.dumps(d, sort_keys=True) for d in reference_defs(pinned)}
    return all(json.dumps(d, sort_keys=True) in ref for d in rule_defs)


class HoldoutLedger:
    """Read-once accounting for the outer holdout.

    The authority is one file per key, created with O_CREAT|O_EXCL in a
    directory every checkout shares (outputs/v2/benchmark/holdout_ledger/).
    The tracked JSONL (benchmarks/holdout_ledger.jsonl) and the mirror JSONL are
    audit copies appended on every event and rebuildable from the authority.
    """

    def __init__(self, authority_dir, jsonl_paths=()):
        self.dir = Path(authority_dir)
        self.jsonl = [Path(p) for p in jsonl_paths]

    @classmethod
    def default(cls, root: Path = REPO) -> HoldoutLedger:
        env = os.environ.get("HEARSAY_HOLDOUT_LEDGER")
        if env and os.environ.get("PYTEST_CURRENT_TEST"):
            base = Path(env)
            return cls(base / "authority", [base / "tracked.jsonl", base / "mirror.jsonl"])
        root = Path(root)
        return cls(root / "outputs/v2/benchmark/holdout_ledger",
                   [root / "benchmarks/holdout_ledger.jsonl", root / "outputs/v2/benchmark/holdout_ledger.jsonl"])

    def _append(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True) + "\n"
        for p in self.jsonl:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(line)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def entries(self) -> list[dict]:
        out = []
        for f in sorted(self.dir.glob("*.json")) if self.dir.exists() else []:
            try:
                out.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                out.append({"key": f.stem, "status": "unreadable"})
        return out

    def count(self) -> int:
        """Distinct holdout reads (aliases of an earlier read are not counted)."""
        return sum(1 for e in self.entries() if not e.get("alias_of"))

    def audit_keys(self) -> set:
        keys = set()
        for p in self.jsonl:
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        keys.add(json.loads(line).get("key"))
        return keys - {None}

    def rebuild(self) -> int:
        """Append any authority event missing from an audit copy. Returns events added."""
        added = 0
        for p in self.jsonl:
            have = set()
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.strip():
                        e = json.loads(line)
                        have.add((e.get("event"), e.get("key")))
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                for e in self.entries():
                    k = e.get("key")
                    events = ["alias"] if e.get("alias_of") else ["reserve"] + (
                        ["complete"] if e.get("status") == "complete" else [])
                    for ev in events:
                        if (ev, k) not in have:
                            body = {"event": ev, "key": k, "rebuilt": True}
                            if ev in ("complete", "alias"):
                                body["digest"] = e.get("digest")
                            if ev == "alias":
                                body["alias_of"] = e.get("alias_of")
                            f.write(json.dumps(body, sort_keys=True) + "\n")
                            added += 1
        return added

    def alias(self, new_key: str, old_key: str, meta: dict) -> dict:
        """Record that `new_key` is the same holdout read as the completed
        `old_key` (e.g. after a ledger key format change). Not a new read."""
        old = self.persisted(old_key)
        if old is None:
            raise BenchmarkError(f"cannot alias: {old_key[:12]} is not a completed read")
        self.dir.mkdir(parents=True, exist_ok=True)
        entry = {"key": new_key, "status": "complete", "alias_of": old_key, "meta": meta,
                 "readout": old["readout"], "digest": old["digest"]}
        fd = os.open(self._path(new_key), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f, sort_keys=True)
        self._append({"event": "alias", "key": new_key, "alias_of": old_key, "digest": old["digest"]})
        return entry

    def persisted(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return d if d.get("status") == "complete" else None

    def reserve(self, key: str, meta: dict) -> dict:
        """Atomically reserve a holdout read. Returns {'status': 'new'|'resumed'|'complete', ...}.
        A crashed reservation resumes only with identical code and input shas."""
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self._path(key)
        entry = {"key": key, "status": "reserved", "meta": meta}
        if not p.exists() and key in self.audit_keys():
            raise HoldoutAlreadyRead(f"holdout key {key[:12]} is in the audit log but its authority file is missing; "
                                     "restore it before reading again")
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            cur = None
            for _ in range(100):  # a racing writer may not have flushed yet
                try:
                    cur = json.loads(p.read_text())
                    break
                except json.JSONDecodeError:
                    time.sleep(0.01)
            if cur is None:
                raise HoldoutAlreadyRead(f"holdout key {key[:12]} is being reserved by another process") from None
            if cur.get("status") == "complete":
                return {"status": "complete", "entry": cur}
            same = (cur.get("meta", {}).get("code_sha") == meta.get("code_sha")
                    and cur.get("meta", {}).get("input_shas") == meta.get("input_shas"))
            if not same:
                raise HoldoutAlreadyRead(f"holdout key {key[:12]} is reserved by a different code/input state")
            return {"status": "resumed", "entry": cur}
        with os.fdopen(fd, "w") as f:
            json.dump(entry, f, sort_keys=True)
        self._append({"event": "reserve", **entry})
        return {"status": "new", "entry": entry}

    def complete(self, key: str, readout: dict) -> None:
        p = self._path(key)
        cur = json.loads(p.read_text())
        body = clean(readout)
        cur.update({"status": "complete", "readout": body,
                    "digest": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()})
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, sort_keys=True))
        os.replace(tmp, p)
        self._append({"event": "complete", "key": key, "digest": cur["digest"]})


def read_holdout(rule_defs: list, compute, *, ledger: HoldoutLedger, holdout_sha: str,
                 meta: dict | None = None, pinned: dict | None = None) -> dict:
    """Run `compute()` on the holdout at most once per rule set. Reference rules
    (the two published holdout numbers, bound to the registry pins) are exempt.
    A completed key returns its persisted readout instead of reading again."""
    if is_reference(rule_defs, pinned):
        return {"ledger": "reference (exempt)", "readout": compute()}
    key = ledger_key(rule_defs, holdout_sha)
    res = ledger.reserve(key, meta or {})
    if res["status"] == "complete":
        return {"ledger": f"persisted {key[:12]}", "readout": res["entry"]["readout"],
                "ledger_count": ledger.count()}
    readout = compute()
    ledger.complete(key, readout)
    return {"ledger": f"read {key[:12]}", "readout": readout, "ledger_count": ledger.count()}
