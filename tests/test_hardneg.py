"""V2-33 stage 0 (docs/specs/2026-10-13_v2-33-hardneg-stage0.md): the pure functions of
scripts/v2_hardneg_bakeoff.py, its gate, its loader guards, and the integrity asserts.

Hermetic: synthetic arrays and a synthetic embedding world only. The four `needs_data` tests read
the saved embeddings and the shipped bake-off probe; everything else runs without any gitignored
asset. Nothing here writes a tracked file -- an autouse fixture asserts that.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hearsay.metrics import min_cost, sponsor_min_dcf

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _load(name: str):
    """Load a sibling script by path. Deliberately does NOT register in sys.modules: the script
    must work under both repo loader idioms (bakeoff_layers._mod registers, test_fuse_sweep_v3
    does not), and a dataclass in the script would break the unregistered one."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hn = _load("v2_hardneg_bakeoff")

HAVE_EMB = (REPO / "outputs/embeddings/wav2vec2-xls-r-300m/nsa_train_sample_v3/manifest.csv").exists()
HAVE_BAKEOFF = (REPO / "outputs/v2/bakeoff/probes/L07.npz").exists()
HAVE_DATA = HAVE_EMB and HAVE_BAKEOFF

# Tracked files this suite must never modify. outputs/ is gitignored but outputs/v2/research/ is
# the un-regenerable pilot (spec F1), and outputs/v2/bakeoff/ is another lane's input.
GUARDED = [REPO / "experiments" / "log.csv",
           REPO / "splits" / "nsa_folds.csv",
           REPO / "splits" / "nsa_folds_plus_asv19.csv",
           REPO / "outputs" / "v2" / "bakeoff" / "rows.json",
           REPO / "outputs" / "v2" / "bakeoff" / "probes" / "L07.npz"]  # fmt: skip
GUARDED_DIRS = [REPO / "outputs" / "v2" / "research"]


def _dir_state(d: Path) -> list[tuple[str, int, int]]:
    return sorted((f.name, f.stat().st_mtime_ns, f.stat().st_size)
                  for f in d.rglob("*") if f.is_file())


@pytest.fixture(autouse=True)
def _tracked_files_never_change():
    """The real incident this guards: mocked launches once appended six fabricated rows to the
    tracked expense ledger (Codex round 11 on the M5 lane)."""
    before = {p: p.read_bytes() for p in GUARDED if p.exists()}
    dirs_before = {d: _dir_state(d) for d in GUARDED_DIRS if d.exists()}
    yield
    for p, b in before.items():
        assert p.read_bytes() == b, f"test modified tracked file {p}"
    for d, b in dirs_before.items():
        assert _dir_state(d) == b, f"test modified {d}"


# --------------------------------------------------------------------------- synthetic world
def _world(tmp_path, n_per=60, d=12, seed=0):
    """A complete synthetic embedding world: two sets, 5 folds, both pools, groups and speakers.

    Returns (root, X, y, fold, meta) where root is an outputs/embeddings tree.
    """
    rng = np.random.default_rng(seed)
    recs, embs = [], []
    for si, (setname, source, n) in enumerate(
            [("nsa_train_sample_v3", "diffssd", n_per), ("asv19_addon_v3", "asvspoof2019", n_per)]):
        for i in range(n):
            spoof = i % 2 == 0
            fold = str(i % 5)
            recs.append({"set": setname, "path": f"/x/{setname}/{i}.wav",
                         "label": "spoof" if spoof else "bonafide",
                         "generator": f"gen{i % 3}" if spoof else "bonafide",
                         "speaker": f"spk{i % 7}", "utt": f"u{i}", "source": source,
                         "n_windows": 1, "flag": "",
                         "group": (f"gen:gen{i % 3}" if spoof else f"spk:spk{i % 7}"),
                         "fold": fold})  # fmt: skip
            embs.append(rng.normal(1.1 if spoof else 0.0, 1.0, d))
    m = pd.DataFrame(recs)
    X = np.asarray(embs, dtype=np.float32)
    root = tmp_path / "outputs" / "embeddings"
    for setname in ("nsa_train_sample_v3", "asv19_addon_v3"):
        sel = (m.set == setname).to_numpy()
        sd = root / hn.MODEL / setname
        sd.mkdir(parents=True, exist_ok=True)
        sub = m[sel].drop(columns=["set", "group", "fold"]).reset_index(drop=True)
        sub.to_csv(sd / "manifest.csv", index=False)
        e = np.zeros((int(sel.sum()), 25, 1024), np.float16)
        e[:, 7, :d] = X[sel]
        np.savez(sd / "shard_00000.npz", emb=e, row=np.arange(int(sel.sum())))
    y = (m.label == "spoof").to_numpy(int)
    fold = m.fold.astype(int).to_numpy()
    return root, X, y, fold, m


# --------------------------------------------------------------------------- A. load / align
def test_a1_cost_helpers_are_pairwise_distinct():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(80, int), np.ones(60, int)]
    s = np.r_[rng.normal(-1, 1, 80), rng.normal(1, 1, 60)]
    vals = {k: f(y, s) for k, f in hn.READINGS.items()}
    assert len(vals) == 3
    assert len({round(v, 9) for v in vals.values()}) == 3, vals


def test_a1_package05_matches_the_sponsor_implementation():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(70, int), np.ones(70, int)]
    s = np.r_[rng.normal(-1, 1, 70), rng.normal(1, 1, 70)]
    assert abs(hn.package05(y, s) - sponsor_min_dcf(y, s, True)) < 1e-12


def test_a1_package05_is_not_min_cost_with_default_kwargs():
    """min_cost(y, s, 0.5) with defaults is a FOURTH reading (4*P_FA + P_miss)."""
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(60, int), np.ones(40, int)]
    s = np.r_[rng.normal(-1, 1, 60), rng.normal(1, 1, 40)]
    assert abs(hn.package05(y, s) - float(min_cost(y, s, 0.5))) > 1e-6


def test_a4_decode_error_row_is_refused(tmp_path):
    root, *_ = _world(tmp_path)
    p = root / hn.MODEL / "nsa_train_sample_v3" / "manifest.csv"
    m = pd.read_csv(p)
    m["flag"] = m["flag"].astype(object)      # an all-empty flag column reads back as float64
    m.loc[0, "flag"] = "decode_error"
    m.to_csv(p, index=False)
    with pytest.raises(ValueError, match="decode_error"):
        hn.load_layer("nsa_train_sample_v3", 7, root=root)


def test_a4_all_zero_row_is_refused_even_without_a_flag(tmp_path):
    root, *_ = _world(tmp_path)
    sd = root / hn.MODEL / "nsa_train_sample_v3"
    z = np.load(sd / "shard_00000.npz")
    emb, rows = z["emb"].copy(), z["row"]
    emb[3, 7, :] = 0.0
    np.savez(sd / "shard_00000.npz", emb=emb, row=rows)
    with pytest.raises(ValueError, match="all-zero"):
        hn.load_layer("nsa_train_sample_v3", 7, root=root)


def test_a5_missing_flag_column_is_refused(tmp_path):
    """bakeoff_layers' assert uses `"flag" not in mm`, so a manifest with no flag column passes
    it silently. Ours must not."""
    root, *_ = _world(tmp_path)
    p = root / hn.MODEL / "nsa_train_sample_v3" / "manifest.csv"
    pd.read_csv(p).drop(columns=["flag"]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="no flag column"):
        hn.load_layer("nsa_train_sample_v3", 7, root=root)


@pytest.mark.parametrize("flagval", ["", np.nan])
def test_a5_empty_and_nan_flag_cells_both_pass(tmp_path, flagval):
    root, *_ = _world(tmp_path)
    p = root / hn.MODEL / "nsa_train_sample_v3" / "manifest.csv"
    m = pd.read_csv(p)
    m["flag"] = flagval
    m.to_csv(p, index=False)
    X, _ = hn.load_layer("nsa_train_sample_v3", 7, root=root)
    assert np.isfinite(X).all()


def test_a6_layer_beyond_the_stack_is_refused(tmp_path):
    root, *_ = _world(tmp_path)
    with pytest.raises(ValueError, match="layer 99"):
        hn.load_layer("nsa_train_sample_v3", 99, root=root)


def test_a2_duplicate_shard_row_index_is_refused(tmp_path):
    """Codex #5: `seen.all()` admits duplicates and silently overwritten rows."""
    root, *_ = _world(tmp_path)
    sd = root / hn.MODEL / "nsa_train_sample_v3"
    z = np.load(sd / "shard_00000.npz")
    rows = z["row"].copy()
    rows[1] = rows[0]
    np.savez(sd / "shard_00000.npz", emb=z["emb"], row=rows)
    with pytest.raises(ValueError, match="exact permutation"):
        hn.load_layer("nsa_train_sample_v3", 7, root=root)


def test_a2_negative_shard_row_index_is_refused(tmp_path):
    """[0,1,2,3,4,-1] passes seen.all() but is not a permutation."""
    root, *_ = _world(tmp_path)
    sd = root / hn.MODEL / "nsa_train_sample_v3"
    z = np.load(sd / "shard_00000.npz")
    rows = z["row"].copy()
    rows[-1] = -1
    np.savez(sd / "shard_00000.npz", emb=z["emb"], row=rows)
    with pytest.raises(ValueError, match="out of bounds|exact permutation"):
        hn.load_layer("nsa_train_sample_v3", 7, root=root)


def test_a3_non_finite_embedding_is_refused(tmp_path):
    root, *_ = _world(tmp_path)
    sd = root / hn.MODEL / "nsa_train_sample_v3"
    z = np.load(sd / "shard_00000.npz")
    emb = z["emb"].copy()
    emb[2, 7, 0] = np.inf
    np.savez(sd / "shard_00000.npz", emb=emb, row=z["row"])
    with pytest.raises(ValueError, match="non-finite|all-zero"):
        hn.load_layer("nsa_train_sample_v3", 7, root=root)


# --------------------------------------------------------------------------- B. barred sets
@pytest.mark.parametrize("name", ["asv19_eval_probe", "itw_stress_v3", "nsa_test_v3",
                                  "asv19_train_probe", "mlaad_cohort"])
def test_b5_barred_sets_are_refused_by_name(name):
    with pytest.raises(ValueError, match="allow-list|barred"):
        hn.load_layer(name, 7)


def test_b5_allow_list_is_exactly_the_two_training_sets():
    assert hn.ALLOWED_SETS == frozenset(("nsa_train_sample_v3", "asv19_addon_v3"))


def test_b6_predefined_split_sentinel_would_be_invisible_with_zeros():
    """A row whose fold id is -1 joins EVERY training split and keeps the array's initial value.
    With np.zeros that is invisible because 0.0 is a legal decision value; with NaN it is not."""
    from sklearn.model_selection import PredefinedSplit
    folds = np.array([0, 0, 1, 1, -1, -1])
    ps = PredefinedSplit(folds)
    assert ps.get_n_splits() == 2
    trains = [tr.tolist() for tr, _ in ps.split()]
    assert all(4 in tr and 5 in tr for tr in trains), "sentinel rows must appear in every train set"
    zeros, nans = np.zeros(6), np.full(6, np.nan)
    for _, te in ps.split():
        zeros[te] = 9.0
        nans[te] = 9.0
    assert np.isfinite(zeros).all(), "zeros hide the unwritten rows"
    assert not np.isfinite(nans).all(), "NaN exposes them"


def test_b6_string_fold_ids_must_be_masked_before_casting():
    from sklearn.model_selection import PredefinedSplit
    with pytest.raises(ValueError):
        PredefinedSplit(np.array(["0", "holdout"]))


@pytest.mark.needs_data
def test_b1_groups_and_generators_are_fold_disjoint_but_speakers_are_not():
    """CLAUDE.md requires validation by generator AND speaker. This fold file satisfies the
    generator half and not the speaker half; stage 0 is therefore a diagnostic, not a
    validation. Asserted as the measured state so a change is caught."""
    ff = pd.read_csv(REPO / "splits" / "nsa_folds_plus_asv19.csv")
    inner = ff[ff.fold != "holdout"]
    assert (inner.groupby("group").fold.nunique() > 1).sum() == 0
    multi = inner.groupby("speaker").fold.nunique()
    crossing = multi[multi > 1]
    rows = int(inner.speaker.isin(crossing.index).sum())
    assert len(crossing) == 41, len(crossing)
    assert rows == 19699, rows
    assert len(inner) == 23790


@pytest.mark.needs_data
def test_b7_cv_groups_is_not_the_fold_files_group_column():
    """A fold-integrity test written against cv_groups() would not be testing the fold file."""
    from hearsay.probe import cv_groups
    ff = pd.read_csv(REPO / "splits" / "nsa_folds_plus_asv19.csv")
    inner = ff[ff.fold != "holdout"].reset_index(drop=True)
    assert inner.group.nunique() == 158
    assert pd.Series(cv_groups(inner)).nunique() == 117
    a, b = pd.factorize(inner.group)[0], pd.factorize(cv_groups(inner))[0]
    split = pd.crosstab(a, b).gt(0).sum(axis=1).gt(1).sum()
    assert split == 0, "cv_groups is a strict coarsening, not a different partition"


# --------------------------------------------------------------------------- C. arms / loop
def test_c1_w1_regression_balanced_plus_sample_weight_is_just_a_c_change():
    """The bug the whole design was rebuilt around (spec W1). sklearn recomputes 'balanced' on
    the sample-weighted counts, cancelling the class differential and leaving a global scale."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    n_r, n_f, d = 300, 200, 20
    X = np.r_[rng.normal(0, 1, (n_r, d)), rng.normal(0.6, 1, (n_f, d))]
    y = np.r_[np.zeros(n_r, int), np.ones(n_f, int)]
    sw = np.where(y == 0, 4.0, 1.0)
    alpha = (4 * n_r + n_f) / (n_r + n_f)
    a = LogisticRegression(C=1.0, class_weight="balanced", max_iter=20000,
                           tol=1e-12).fit(X, y, sample_weight=sw)
    b = LogisticRegression(C=alpha, class_weight="balanced", max_iter=20000, tol=1e-12).fit(X, y)
    assert np.max(np.abs(a.coef_ - b.coef_)) < 1e-8, "the confound must still reproduce"


def test_c1_lane_weights_hold_sum_fixed_so_arms_are_at_matched_regularization():
    y = np.r_[np.zeros(300, int), np.ones(200, int)]
    base = hn.balanced_weights(y)
    assert abs(base.sum() - len(y)) < 1e-9, "balanced weights already sum to n"
    picks = np.arange(0, 25)
    w = hn.shaped_weights(y, picks, mult=5.0)
    assert abs(w.sum() - len(y)) < 1e-9, "a shaped arm must still sum to n"
    ratio = w / base
    assert len(np.unique(np.round(ratio, 9))) > 1, "the shape must not be a global scale"


def test_c1_identity_setting_reproduces_base_exactly():
    y = np.r_[np.zeros(200, int), np.ones(150, int)]
    base = hn.balanced_weights(y)
    base = base * (len(y) / base.sum())
    identity = hn.shaped_weights(y, np.array([], dtype=int), mult=5.0)
    assert np.allclose(identity, base, atol=0, rtol=1e-12)


def test_c1_multiplier_one_reproduces_base_exactly():
    y = np.r_[np.zeros(200, int), np.ones(150, int)]
    base = hn.balanced_weights(y)
    base = base * (len(y) / base.sum())
    assert np.allclose(hn.shaped_weights(y, np.arange(20), mult=1.0), base, atol=0, rtol=1e-12)


def test_c3_focal_gradient_matches_finite_differences():
    from scipy.optimize import approx_fprime
    rng = np.random.default_rng(0)
    n, d = 200, 8
    X = rng.normal(size=(n, d))
    y = (rng.normal(size=n) > 0).astype(int)
    for gamma in (0.0, 1.0, 2.0):
        f = hn.focal_objective(X, y, gamma, 1.0)
        th = rng.normal(scale=0.3, size=d + 1)
        ga = f(th)[1]
        gn = approx_fprime(th, lambda t, _f=f: _f(t)[0], 1e-6)   # bind, don't close over
        rel = np.max(np.abs(ga - gn)) / max(1.0, np.max(np.abs(gn)))
        assert rel < 1e-5, (gamma, rel)


def test_c4_focal_gamma_zero_reduces_to_class_balanced_logistic():
    from scipy.optimize import minimize
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    n, d = 300, 10
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    f = hn.focal_objective(X, y, 0.0, 1.0)
    th = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                  options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12}).x
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=20000, tol=1e-12).fit(X, y)
    got = X @ th[:-1] + th[-1]
    assert np.max(np.abs(got - clf.decision_function(X))) < 1e-4


def test_c9_guard_fires_when_a_folds_scores_are_not_finite(tmp_path, monkeypatch):
    """The closest test to the owner's stated risk. NaN-initialised scores plus this guard are
    what turn 'a fold produced nothing usable' into a loud failure instead of a plausible
    number -- np.zeros would have hidden it, because 0.0 is a legal decision value."""
    _root, X, y, fold, _m = _world(tmp_path)
    real_fit = hn.fit_lr
    calls = {"n": 0}

    def flaky(Xs, y_tr, w, c):
        calls["n"] += 1
        clf = real_fit(Xs, y_tr, w, c)
        if calls["n"] == 2:                      # one fold comes back unusable
            clf.coef_ = clf.coef_ * np.nan
        return clf

    monkeypatch.setattr(hn, "fit_lr", flaky)
    with pytest.raises(ValueError, match="coverage incomplete"):
        hn.arm_oof(hn.ARMS_BY_NAME["none"], X, y, fold, 1.0)


def test_c9_this_implementation_has_no_sentinel_fold_hole(tmp_path):
    """Why the PredefinedSplit hazard (B6) cannot bite here: arm_oof iterates the fold labels it
    is given, so a -1 label is just another fold and its rows ARE scored. The hazard is a
    property of PredefinedSplit, which this lane does not use."""
    _root, X, y, fold, _m = _world(tmp_path)
    bad = fold.copy()
    bad[fold == 4] = -1
    oof, _ = hn.arm_oof(hn.ARMS_BY_NAME["none"], X, y, bad, 1.0)
    assert np.isfinite(oof).all(), "every row must still be scored"
    assert len(set(bad.tolist())) == 5


def test_c9_normal_run_writes_every_row(tmp_path):
    _root, X, y, fold, _m = _world(tmp_path)
    oof, _ = hn.arm_oof(hn.ARMS_BY_NAME["none"], X, y, fold, 1.0)
    assert np.isfinite(oof).all()
    assert len(oof) == len(y)


def test_c_mined_set_below_minimum_is_refused(tmp_path, monkeypatch):
    _root, X, y, fold, _m = _world(tmp_path)
    monkeypatch.setattr(hn, "MIN_MINED", 10_000)
    with pytest.raises(ValueError, match="mined set"):
        hn.arm_oof(hn.ARMS_BY_NAME["hardest_both"], X, y, fold, 1.0)


def _gaussian_world(n_per=300, d=12, seed=0, sep=1.1):
    """Two overlapping Gaussian classes, five folds: big enough that a one-class up-weighting
    moves that class's error rate visibly and deterministically (checked over seeds 0-4)."""
    rng = np.random.default_rng(seed)
    y = (np.arange(2 * n_per) % 2).astype(int)
    X = rng.normal(0, 1, (2 * n_per, d)).astype(np.float32)
    X[y == 1] = rng.normal(sep, 1, (n_per, d))
    fold = (np.arange(2 * n_per) % 5).astype(int)
    return X, y, fold


def _rates(y, oof, thr=0.0):
    fa = float(((y == 0) & (oof >= thr)).sum() / (y == 0).sum())
    miss = float(((y == 1) & (oof < thr)).sum() / (y == 1).sum())
    return fa, miss


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_c2_a_shape_applied_to_one_class_moves_that_classs_error_rate(seed, monkeypatch):
    """The break-check test (spec, 'Test-design changes'): up-weighting the hardest REALS must
    cut false alarms (and trade misses for it); up-weighting the hardest FAKES must cut misses.
    The realistic bug this catches is the class mapping in `select_picks` pointing the wrong way,
    which still produces a plausible table -- the owner's stated P1 risk."""
    monkeypatch.setattr(hn, "MIN_MINED", 1)
    X, y, fold = _gaussian_world(seed=seed)
    base, _ = hn.arm_oof(hn.ARMS_BY_NAME["none"], X, y, fold, 1.0)
    reals, _ = hn.arm_oof(hn.ARMS_BY_NAME["hardest_reals"], X, y, fold, 1.0)
    fakes, _ = hn.arm_oof(hn.ARMS_BY_NAME["hardest_fakes"], X, y, fold, 1.0)
    fa_b, miss_b = _rates(y, base)
    fa_r, miss_r = _rates(y, reals)
    fa_f, miss_f = _rates(y, fakes)
    assert fa_r < fa_b, f"hardest_reals must cut false alarms: {fa_r} vs base {fa_b}"
    assert miss_r >= miss_b, "the trade: fewer false alarms, not fewer misses"
    assert miss_f < miss_b, f"hardest_fakes must cut misses: {miss_f} vs base {miss_b}"
    assert fa_f >= fa_b, "the trade: fewer misses, not fewer false alarms"


def test_c7_inner_margin_is_out_of_fold(tmp_path):
    """Nested: for each inner fold the margin comes from a model that did not see those rows."""
    rng = np.random.default_rng(0)
    n, d = 240, 10
    X = rng.normal(size=(n, d))
    y = (np.arange(n) % 2).astype(int)
    f = (np.arange(n) % 4).astype(int)
    marg = hn.inner_oof_margin(X, y, f, 1.0)
    assert np.isfinite(marg).all()
    assert len(marg) == n


# --------------------------------------------------------------------------- H. hard vs easy
def test_h2_semantic_hardest_and_easiest_select_the_expected_examples():
    """Codex #9: a fixture with independently specified margins and expected index sets, so a
    sort substitution is caught semantically rather than by mirroring the mutated line."""
    y = np.array([0] * 10 + [1] * 10)
    marg = np.array([5., 4., 3., 2., 1., -1., -2., -3., -4., -5.,
                     9., 8., 7., 6., 5., -5., -6., -7., -8., -9.])  # fmt: skip
    rng = np.random.default_rng(0)
    hardest, per_h = hn.select_picks(hn.ARMS_BY_NAME["hardest_both"], marg, y, rng)
    easiest, per_e = hn.select_picks(hn.ARMS_BY_NAME["easiest_both"], marg, y, rng)
    # 5% of 10 -> max(1, 0) == 1 per class
    assert per_h == {0: 1, 1: 1} and per_e == {0: 1, 1: 1}
    assert set(hardest) == {9, 19}, "hardest = lowest margin in each class"
    assert set(easiest) == {0, 10}, "easiest = highest margin in each class"
    assert set(hardest).isdisjoint(set(easiest))


def test_h2_side_selects_only_that_class():
    y = np.array([0] * 40 + [1] * 40)
    marg = np.r_[np.linspace(1, -1, 40), np.linspace(2, -2, 40)]
    rng = np.random.default_rng(0)
    _, per_r = hn.select_picks(hn.ARMS_BY_NAME["hardest_reals"], marg, y, rng)
    _, per_f = hn.select_picks(hn.ARMS_BY_NAME["hardest_fakes"], marg, y, rng)
    assert set(per_r) == {0} and set(per_f) == {1}


def test_h2_ties_are_resolved_deterministically():
    y = np.array([0] * 20 + [1] * 20)
    marg = np.zeros(40)
    rng = np.random.default_rng(0)
    a, _ = hn.select_picks(hn.ARMS_BY_NAME["hardest_both"], marg, y, np.random.default_rng(0))
    b, _ = hn.select_picks(hn.ARMS_BY_NAME["hardest_both"], marg, y, rng)
    assert set(a) == set(b), "a stable sort must give the same set on tied margins"


def test_h2_none_arm_selects_nothing():
    y = np.array([0] * 20 + [1] * 20)
    picks, per = hn.select_picks(hn.ARMS_BY_NAME["none"], np.zeros(40), y,
                                 np.random.default_rng(0))
    assert len(picks) == 0 and per == {}


def test_h7_margin_normalisation_is_scale_invariant():
    """Codex #6: pooling raw margins from several inner models would let model scale decide which
    fold supplies the hardest examples."""
    rng = np.random.default_rng(0)
    y = (np.arange(200) % 2).astype(int)
    d = rng.normal(size=200)
    ref = np.where(y == 1, d, -d)
    a = hn.normalised_margin(d, y, ref=ref)
    b = hn.normalised_margin(d * 37.0, y, ref=ref * 37.0)
    assert np.allclose(a, b, rtol=1e-9)


def test_h7_normalisation_survives_a_degenerate_iqr():
    y = (np.arange(20) % 2).astype(int)
    out = hn.normalised_margin(np.zeros(20), y, ref=np.zeros(20))
    assert np.isfinite(out).all()


def test_h5_confidently_wrong_is_a_subset_of_false_alarms():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    s = np.r_[rng.normal(-1, 1, 200), rng.normal(1, 1, 200)]
    fold = np.tile(np.arange(5), 80)
    meta = pd.DataFrame({"group": ["g" + str(i % 6) for i in range(400)]})
    r = hn.read_cell(y, s, fold, meta, np.ones(400, bool))
    assert 0 <= r["confidently_wrong_fa"] <= r["n_fa_at_thr"]


# --------------------------------------------------------------------------- D. readings
def test_d3_reading_weights_recover_their_cost_ratios():
    """package05 -> 1:4, package03 -> 1:1.714, brief -> 9.33:1."""
    y = np.r_[np.zeros(100, int), np.ones(100, int)]
    s = np.r_[np.zeros(100), np.ones(100)]      # perfectly separable
    for fn in hn.READINGS.values():
        assert fn(y, s) == pytest.approx(0.0, abs=1e-12)
    s_bad = np.r_[np.ones(100), np.zeros(100)]  # perfectly inverted
    for fn in hn.READINGS.values():
        assert fn(y, s_bad) == pytest.approx(1.0, abs=1e-9)


def test_d1_threshold_tie_resolves_to_the_highest_optimum():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.0, 1.0, 1.0, 2.0])
    t = hn.thr_package05(y, s)
    costs = {tt: float(hn.cost_at(y, s, tt, 0.5, c_fa=1.0, c_miss=4.0))
             for tt in np.r_[np.unique(s), np.inf]}
    best = min(costs.values())
    tied = [tt for tt, c in costs.items() if abs(c - best) < 1e-12]
    assert t == max(tied)


def test_d6_nan_reading_does_not_silently_pass_a_band():
    """cost_at returns NaN on an empty class slice and `not (nan > band)` is True."""
    y = np.ones(10, int)
    v = float(hn.cost_at(y, np.linspace(0, 1, 10), 0.5, 0.5, 1.0, 4.0))
    assert np.isnan(v)
    assert not (v > 0.01), "this is the silent pass the gate must not rely on"
    assert not hn.saturated(v), "a NaN must not count as saturated either"


def test_d7_saturation_flag():
    assert hn.saturated(1.0)
    assert hn.saturated(1.0 - 1e-15)
    assert not hn.saturated(0.9)
    assert not hn.saturated(float("nan"))


def test_d_read_cell_reports_every_reading_and_its_threshold():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(150, int), np.ones(150, int)]
    s = np.r_[rng.normal(-1, 1, 150), rng.normal(1, 1, 150)]
    fold = np.tile(np.arange(5), 60)
    meta = pd.DataFrame({"group": ["g" + str(i % 4) for i in range(300)]})
    r = hn.read_cell(y, s, fold, meta, np.ones(300, bool))
    for k in ("package05", "package03", "brief", "eer", "thr_package05", "n_fa_at_thr",
              "n_miss_at_thr", "auc", "confidently_wrong_fa", "pop_group_concentration_real",
              "pop_group_concentration_fake", "pop_top_group_real", "pop_top_group_fake"):
        assert k in r, k
    for k in ("package05", "package03", "brief"):
        assert set(r[f"{k}_by_fold"]) <= set(range(5))
    # The population readout is named as such so it can never be mistaken for the mined set's.
    assert "group_concentration_real" not in r


def test_h5_as_frozen_saturates_when_false_alarms_cover_the_real_decile():
    """The documented state of the frozen H5 definition (Codex #6): 'a false alarm in the worst
    decile of the real-side normalised margin'. Whenever the false alarms at the package-0.5
    threshold already cover that decile -- true in every cell of the Oct 13 run (min 996 FA
    against a decile of 814) -- the count is the decile size, whatever the arm did. Recorded so
    the limitation is a measured fact, not an assumption; a redefinition belongs to stage 0.5."""
    rng = np.random.default_rng(1)
    n_real = 500
    y = np.r_[np.zeros(n_real, int), np.ones(500, int)]
    s = np.r_[rng.normal(-0.5, 1, n_real), rng.normal(0.5, 1, 500)]  # heavy overlap: many FA
    fold = np.tile(np.arange(5), 200)
    meta = pd.DataFrame({"group": ["g" + str(i % 6) for i in range(1000)]})
    r = hn.read_cell(y, s, fold, meta, np.ones(1000, bool))
    assert r["n_fa_at_thr"] >= n_real // 10
    assert r["confidently_wrong_fa"] == pytest.approx(n_real // 10, abs=1)


def test_d4_clean_replaces_non_finite_with_none():
    out = hn.clean({"a": float("nan"), "b": float("inf"), "c": np.float64(1.5),
                    "d": np.int64(3), "e": np.True_, "f": np.array([1.0, np.nan])})
    assert out["a"] is None and out["b"] is None
    assert out["c"] == 1.5 and out["d"] == 3 and out["e"] is True
    assert out["f"] == [1.0, None]
    json.dumps(out, allow_nan=False)      # must not raise


# --------------------------------------------------------------------------- bootstrap
def _boot_fixture(seed=0, n_groups=12, per=20):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"g{i}" for i in range(n_groups)], per)
    y = np.tile(np.r_[np.zeros(per // 2, int), np.ones(per // 2, int)], n_groups)
    base = y + rng.normal(scale=1.0, size=len(y))
    better = y + rng.normal(scale=0.4, size=len(y))
    return y, base, better, groups


def test_boot_sign_convention_positive_means_arm_is_better():
    y, base, better, groups = _boot_fixture()
    r = hn.group_bootstrap(y, base, better, groups, n=200)
    assert r["p_better"] > 0.5, "delta = base - arm, so positive must mean the arm is better"
    rev = hn.group_bootstrap(y, better, base, groups, n=200)
    assert rev["p_better"] < 0.5


def test_boot_is_deterministic_under_a_fixed_seed():
    y, base, better, groups = _boot_fixture()
    a = hn.group_bootstrap(y, base, better, groups, n=150, seed=7)
    b = hn.group_bootstrap(y, base, better, groups, n=150, seed=7)
    assert a == b


def test_boot_reports_its_monte_carlo_error_and_group_count():
    y, base, better, groups = _boot_fixture()
    r = hn.group_bootstrap(y, base, better, groups, n=200)
    assert r["n_groups"] == 12
    assert 0.0 <= r["p_better_se"] < 0.2
    assert r["n"] + r["skipped_single_class"] == 200


def test_boot_refuses_blank_group_values():
    y, base, better, groups = _boot_fixture()
    g = groups.astype(object).copy()
    g[0] = "  "
    with pytest.raises(ValueError, match="blank group"):
        hn.group_bootstrap(y, base, better, g, n=20)


def test_boot_resamples_whole_groups():
    """A one-group world can only ever draw that group, so the delta has zero variance."""
    y, base, better, groups = _boot_fixture(n_groups=1, per=40)
    r = hn.group_bootstrap(y, base, better, groups, n=50)
    assert r["p5"] == r["median"]


# --------------------------------------------------------------------------- E. the gate
def _cell(package05=0.30, package03=0.20, brief=0.26, fa=100, auc=0.9,
          f0=0.45, f4=0.34, conc=0.10, pop_conc=0.03):
    """`conc` is the MINED set's real-side group concentration (what R7 gates); `pop_conc` is
    the population's, carried only so a test can prove R7 ignores it."""
    return {"primary": {"package05": package05, "package03": package03, "brief": brief,
                        "n_fa_at_thr": fa, "auc": auc,
                        "pop_group_concentration_real": pop_conc,
                        "package05_by_fold": {0: f0, 4: f4}},
            "mined": {"group_concentration_real": conc, "top_group_real": "spk:x",
                      "n_picks_real": 400}}  # fmt: skip


def _mined(cell, **kw):
    return {**cell, "mined": {**cell["mined"], **kw}}


def _pass_args():
    """An all-rules-pass configuration: the arm beats base on both package readings, holds brief,
    holds both folds, cuts false alarms, is unconcentrated and is not degenerate."""
    arm = hn.ARMS_BY_NAME["hardest_both"]
    cell = _cell(package05=0.27, package03=0.18, brief=0.25, fa=90, f0=0.44, f4=0.33)
    base = _cell()
    boot = {"p_better": 0.99, "p_better_se": 0.005}
    return arm, cell, base, boot


def test_e1_all_rules_pass_on_the_reference_configuration():
    g = hn.gate_cell(*_pass_args())
    assert g["verdict"] == "PASS", g["failed"]
    assert g["failed"] == []


@pytest.mark.parametrize("rule,mutate", [
    ("R1", lambda a, c, b, t: (a, c, b, {**t, "p_better": 0.50, "p_better_se": 0.005})),
    ("R2", lambda a, c, b, t: (a, {"primary": {**c["primary"], "package03": 0.25}}, b, t)),
    ("R3", lambda a, c, b, t: (a, {"primary": {**c["primary"], "brief": 0.40}}, b, t)),
    ("R4a", lambda a, c, b, t: (a, {"primary": {**c["primary"],
                                                "package05_by_fold": {0: 0.60, 4: 0.33}}}, b, t)),
    ("R4b", lambda a, c, b, t: (a, {"primary": {**c["primary"],
                                                "package05_by_fold": {0: 0.44, 4: 0.60}}}, b, t)),
    ("R6", lambda a, c, b, t: (a, {"primary": {**c["primary"], "n_fa_at_thr": 400}}, b, t)),
    ("R7", lambda a, c, b, t: (a, _mined(c, group_concentration_real=0.90), b, t)),
    ("R9", lambda a, c, b, t: (a, {**c, "primary": {**c["primary"], "auc": 0.40}}, b, t)),
])
def test_e1_each_gate_rule_can_fail_alone(rule, mutate):
    """A rule nothing can fail is not a rule. Each mutation must break exactly its own rule."""
    g = hn.gate_cell(*mutate(*_pass_args()))
    assert rule in g["failed"], f"{rule} did not fail: {g}"
    assert g["verdict"] == "FAIL"


def test_e_r7_reads_the_mined_set_not_the_population():
    """The Oct 13 run's defect: R7 was fed the population's group concentration (0.0316 in every
    cell) instead of the mined set's, so it could not fail. A concentrated mined set in an
    unconcentrated population must FAIL; a diffuse mined set in a concentrated population passes."""
    arm, cell, base, boot = _pass_args()
    g = hn.gate_cell(arm, _mined({**cell, "primary": {**cell["primary"],
                                                      "pop_group_concentration_real": 0.03}},
                                 group_concentration_real=0.60), base, boot)
    assert "R7" in g["failed"]
    g = hn.gate_cell(arm, _mined({**cell, "primary": {**cell["primary"],
                                                      "pop_group_concentration_real": 0.90}},
                                 group_concentration_real=0.05), base, boot)
    assert "R7" not in g["failed"]


def test_e_r7_missing_readout_is_a_failure_for_a_mined_arm():
    arm, cell, base, boot = _pass_args()
    g = hn.gate_cell(arm, {**cell, "mined": {}}, base, boot)
    assert "R7" in g["failed"], "'characterisation readout present' is part of R7"
    g = hn.gate_cell(arm, {"primary": cell["primary"]}, base, boot)
    assert "R7" in g["failed"]


def test_e_r7_fakes_only_arm_needs_a_readout_but_has_no_real_side_to_cap():
    _arm, cell, base, boot = _pass_args()
    fakes = hn.ARMS_BY_NAME["hardest_fakes"]
    g = hn.gate_cell(fakes, {**cell, "mined": {"group_concentration_fake": 0.70,
                                                "top_group_fake": "gen:grad_tts"}}, base, boot)
    assert g["R7"] is True, "the fake side is reported, not gated (spec W3)"
    g = hn.gate_cell(fakes, {**cell, "mined": {}}, base, boot)
    assert g["R7"] is False


def test_e_r4_survives_the_json_round_trip():
    """The Oct 13 gate's defect: cells are written with `clean()` and read back with json, which
    turns the per-fold keys 0/4 into "0"/"4". An int lookup then returns None and R4a/R4b fail
    for every candidate, however good its folds. The gate must read a round-tripped cell exactly
    as it reads an in-memory one."""
    arm, cell, base, boot = _pass_args()
    rt = json.loads(json.dumps(hn.clean(cell)))
    rb = json.loads(json.dumps(hn.clean(base)))
    assert set(rt["primary"]["package05_by_fold"]) == {"0", "4"}, "the round trip stringifies"
    g = hn.gate_cell(arm, rt, rb, boot)
    assert g["R4a"] is True and g["R4b"] is True and g["verdict"] == "PASS", g
    worse = json.loads(json.dumps(hn.clean(
        {**cell, "primary": {**cell["primary"], "package05_by_fold": {0: 0.60, 4: 0.33}}})))
    assert "R4a" in hn.gate_cell(arm, worse, rb, boot)["failed"]


def test_e_r3b_sign_disagreement_across_readings_fails():
    arm, cell, base, boot = _pass_args()
    # better on both package readings, worse on brief -> signs disagree
    cell = {"primary": {**cell["primary"], "brief": 0.265}}
    g = hn.gate_cell(arm, cell, base, boot)
    assert "R3b" in g["failed"]


def test_e_r1_indeterminate_near_the_band():
    arm, cell, base, _boot = _pass_args()
    g = hn.gate_cell(arm, cell, base, {"p_better": 0.905, "p_better_se": 0.01})
    assert g["R1_indeterminate"] is True


def test_e_r1_not_indeterminate_when_far_from_the_band():
    g = hn.gate_cell(*_pass_args())
    assert g["R1_indeterminate"] is False


def test_e_r5_eligibility_is_false_for_single_layer_arms():
    for name in ("hardest_reals", "focal_g2", "pilot_quantile"):
        g = hn.gate_cell(hn.ARMS_BY_NAME[name], *_pass_args()[1:])
        assert g["R5_eligible"] is False, name
    for name in hn.PRIMARY_FAMILY:
        g = hn.gate_cell(hn.ARMS_BY_NAME[name], *_pass_args()[1:])
        assert g["R5_eligible"] is True, name


def test_e_r7_is_not_applied_to_the_none_arm():
    arm = hn.ARMS_BY_NAME["none"]
    _, cell, base, boot = _pass_args()
    g = hn.gate_cell(arm, {"primary": cell["primary"], "mined": {}}, base, boot)
    assert g["R7"] is True, "R7 gates mined sets only"


def test_e_r9_refuses_a_saturated_reading():
    arm, cell, base, boot = _pass_args()
    g = hn.gate_cell(arm, {"primary": {**cell["primary"], "package05": 1.0}}, base, boot)
    assert "R9" in g["failed"]


def test_e_r9_refuses_non_finite():
    arm, cell, base, boot = _pass_args()
    g = hn.gate_cell(arm, {"primary": {**cell["primary"], "package05": float("nan")}}, base, boot)
    assert "R9" in g["failed"]


def test_e2_boundary_exactly_at_the_brief_band_passes():
    arm, cell, base, boot = _pass_args()
    cell = {"primary": {**cell["primary"], "brief": round(0.26 + hn.BAND_BRIEF, 4)}}
    g = hn.gate_cell(arm, cell, base, boot)
    assert "R3" not in g["failed"], "the band is inclusive, declared"


def test_e3_deltas_are_rounded_before_comparison():
    arm, cell, base, boot = _pass_args()
    # a difference below 4-dp resolution must not tip R3
    cell = {"primary": {**cell["primary"], "brief": 0.26 + hn.BAND_BRIEF + 1e-6}}
    g = hn.gate_cell(arm, cell, base, boot)
    assert "R3" not in g["failed"]


def test_e7_whole_gate_rejects_a_synthetic_overfit_candidate():
    """If the gate passes a candidate that is excellent on two folds and chance elsewhere, with
    a worse brief reading and more false alarms, the gate is decorative."""
    arm = hn.ARMS_BY_NAME["hardest_both"]
    overfit = _cell(package05=0.28, package03=0.26, brief=0.33, fa=500, auc=0.55,
                    f0=0.20, f4=0.60, conc=0.80)
    g = hn.gate_cell(arm, overfit, _cell(), {"p_better": 0.55, "p_better_se": 0.02})
    assert g["verdict"] == "FAIL"
    for r in ("R1", "R2", "R3", "R4b", "R6", "R7"):
        assert r in g["failed"], f"{r} should have caught the overfit candidate: {g['failed']}"


# --------------------------------------------------------------------------- H3 / H4 / F4
def test_h4_characterise_measures_the_mined_rows_not_the_population(tmp_path):
    """Picks concentrated in one group inside a diffuse population must read as concentrated."""
    _root, _X, y, _fold, m = _world(tmp_path)
    reals = np.flatnonzero(y == 0)
    one_group = reals[m.group.to_numpy()[reals] == "spk:spk1"]
    assert one_group.size >= 3
    picks = {0: one_group[:3], 1: one_group[:3]}
    out = hn.characterise(m, y, {0: {0: 3}, 1: {0: 3}}, picks)
    assert out["group_concentration_real"] == 1.0
    assert out["top_group_real"] == "spk:spk1"
    assert out["n_picks_real"] == 6 and out["n_unique_rows_real"] == 3
    assert "group_concentration_fake" not in out, "no fakes were mined"
    assert out["source_shares_real"]
    # the population is nowhere near that concentrated
    assert m.group[y == 0].value_counts(normalize=True).iloc[0] < 0.5
    assert hn.characterise(m, y, {}, {}) == {}


def test_c7_mined_picks_are_training_rows_of_their_fold(tmp_path, monkeypatch):
    """A fold may never mine the rows it is about to score (spec B2/C7)."""
    _root, X, y, fold, _m = _world(tmp_path)
    monkeypatch.setattr(hn, "MIN_MINED", 1)        # 5% of the synthetic world is 2 per class
    _oof, extra = hn.arm_oof(hn.ARMS_BY_NAME["hardest_both"], X, y, fold, 1.0)
    assert set(extra["picks_by_fold"]) == set(range(5))
    for k, idx in extra["picks_by_fold"].items():
        assert idx.size > 0
        assert not (fold[idx] == k).any(), f"fold {k} mined its own held-out rows"
        assert extra["mined_per_fold"][k] == {0: int((y[idx] == 0).sum()),
                                              1: int((y[idx] == 1).sum())}


def test_h3_jaccard_detects_collapsed_arms():
    a, b = np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6])
    assert hn.jaccard(a, a) == 1.0
    assert hn.jaccard(a, b) == pytest.approx(2 / 6)
    assert hn.jaccard(a, np.array([7, 8])) == 0.0
    assert np.isnan(hn.jaccard(np.array([], int), np.array([], int)))


def test_f4_log_row_names_the_basis_and_every_reading(tmp_path):
    rep = {"n_declared_cells": 132, "n_present": 132, "n_invalid": 1,
           "base_reference": {"package05": 0.2405, "package03": 0.1766, "brief": 0.2567,
                              "eer": 0.0686},
           "gate": {"x": {"verdict": "FAIL"}},
           "contrasts": {"hardest_both_minus_none": {"mean_delta_package05": 0.004},
                         "easiest_both_minus_none": {"mean_delta_package05": -0.0261}}}  # fmt: skip
    f = tmp_path / "log.csv"
    f.write_text(",".join(hn.LOG_COLUMNS) + "\n")
    row = hn.append_log(rep, path=f)
    lines = f.read_text().splitlines()
    assert len(lines) == 2
    assert row["rung"] == hn.RUNG and row["benchmark_version"] == hn.BENCHMARK_VERSION
    assert row["validation_score"] == 0.2405
    for token in ("NSA-only", "package05=0.2405", "package03=0.1766", "brief=0.2567",
                  "eer=0.0686", "PASS cells = 0", "exploratory"):
        assert token in row["notes"], token
    parsed = pd.read_csv(f)
    assert list(parsed.columns) == list(hn.LOG_COLUMNS) and len(parsed) == 1
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,rung\n")
    with pytest.raises(ValueError, match="header"):
        hn.append_log(rep, path=bad)


# --------------------------------------------------------------------------- F. provenance
def test_f_declared_grid_is_the_frozen_requested_set():
    g = hn.declared_grid()
    assert len(g) == 132
    four = sum(1 for k in g if k[0] in hn.PRIMARY_FAMILY)
    assert four == 4 * len(hn.LAYERS) * len(hn.POOLS) * len(hn.C_GRID) == 96
    assert len(g) - four == 36
    assert len(set(g)) == len(g), "no duplicate cells"


def test_f_every_arm_name_is_unique_and_resolvable():
    assert len(hn.ARMS_BY_NAME) == len(hn.ARMS)
    for name, arm in hn.ARMS_BY_NAME.items():
        assert arm.name == name
        assert arm.shape in ("none", "hardest", "easiest", "random", "pilot_quantile")
        assert arm.side in ("both", "reals", "fakes")
        assert arm.loss in ("lr", "focal")


def test_f_primary_family_is_exactly_the_four_layer_arms():
    assert set(hn.PRIMARY_FAMILY) == {a.name for a in hn.ARMS if a.four_layer}


@pytest.mark.needs_data
def test_f2_fingerprint_changes_when_a_score_changing_input_changes():
    """Codex #3: skip-if-exists must not accept a cell computed under different inputs."""
    arm = hn.ARMS_BY_NAME["hardest_both"]
    mask = np.ones(10, bool)
    a = hn.fingerprint(arm, 7, "pool_full", 1.0, mask)
    assert a == hn.fingerprint(arm, 7, "pool_full", 1.0, mask)
    assert a != hn.fingerprint(arm, 7, "pool_full", 8.0, mask)
    assert a != hn.fingerprint(arm, 8, "pool_full", 1.0, mask)
    assert a != hn.fingerprint(arm, 7, "pool_nsa", 1.0, mask)
    assert a != hn.fingerprint(hn.ARMS_BY_NAME["easiest_both"], 7, "pool_full", 1.0, mask)
    other = mask.copy()
    other[0] = False
    assert a != hn.fingerprint(arm, 7, "pool_full", 1.0, other)


def test_f_log_columns_match_the_tracked_header():
    f = REPO / "experiments" / "log.csv"
    assert f.read_text().splitlines()[0] == ",".join(hn.LOG_COLUMNS)


def test_f_atomic_write_leaves_no_tmp_file(tmp_path):
    p = tmp_path / "x.json"
    hn.atomic_write_json(p, {"a": 1, "b": float("nan")})
    assert json.loads(p.read_text()) == {"a": 1, "b": None}
    assert not list(tmp_path.glob("*.tmp"))


def test_f_script_never_imports_torch_or_lightgbm():
    """The only mechanical enforcement of 'CPU only, no torch'."""
    code = ("import importlib.util,sys;"
            "spec=importlib.util.spec_from_file_location('m','scripts/v2_hardneg_bakeoff.py');"
            "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
            "print('lightgbm' in sys.modules, 'torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                       text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False False", r.stdout


def test_f_output_root_is_inside_the_lane():
    assert hn.OUT == REPO / "outputs" / "v2" / "hardneg"
    assert "research" not in str(hn.OUT)
    assert "bakeoff" not in str(hn.OUT)


# --------------------------------------------------------------------------- needs_data
@pytest.fixture(scope="module")
def real_l7():
    if not HAVE_DATA:
        pytest.skip("needs the saved embeddings and the shipped bake-off probe")
    return hn.load_world(7)


@pytest.mark.needs_data
def test_a1_base_reproduces_the_shipped_probe_monotonically(real_l7):
    """The self-check, as a test. A raw max|diff| reads 57.4 on an identical ranking because the
    shipped OOF is Platt-rescaled -- the pilot printed exactly that and continued."""
    from scipy.stats import spearmanr
    X, y, fold, _meta, _pools = real_l7
    mine, _ = hn.arm_oof(hn.ARMS_BY_NAME["none"], X, y, fold, 1.0)
    ref = np.load(REPO / "outputs/v2/bakeoff/probes/L07.npz")["oof"]
    assert float(spearmanr(mine, ref).statistic) == 1.0
    assert abs(hn.package05(y, mine) - hn.package05(y, ref)) < 1e-12
    assert np.max(np.abs(mine - ref)) > 1.0, "the raw difference is large; that is the trap"


@pytest.mark.needs_data
def test_pools_have_the_declared_sizes(real_l7):
    _X, _y, _fold, _meta, pools = real_l7
    assert pools["pool_full"].sum() == 23790
    assert pools["pool_nsa"].sum() == 16142


@pytest.mark.needs_data
def test_base_reference_numbers_are_unchanged(real_l7):
    """The spec's frozen reference values (spec:344-345)."""
    _X, y, _fold, _meta, pools = real_l7
    ref = np.load(REPO / "outputs/v2/bakeoff/probes/L07.npz")["oof"]
    nsa = pools["pool_nsa"]
    assert hn.package05(y, ref) == pytest.approx(0.3104, abs=5e-5)
    assert hn.package05(y[nsa], ref[nsa]) == pytest.approx(0.3307, abs=5e-5)
    thr = hn.thr_package05(y, ref)
    assert int(((y == 0) & (ref >= thr)).sum()) == 2348
    thr_n = hn.thr_package05(y[nsa], ref[nsa])
    assert int(((y[nsa] == 0) & (ref[nsa] >= thr_n)).sum()) == 1642


@pytest.mark.needs_data
def test_in_sample_separability_is_why_nested_margins_are_used(real_l7):
    """Spec W2: the probe has zero in-sample errors, so an in-sample 'hardest' is undefined."""
    from sklearn.preprocessing import StandardScaler
    X, y, fold, _meta, _pools = real_l7
    tr = fold != 0
    sc = StandardScaler().fit(X[tr])
    Xtr = sc.transform(X[tr])
    clf = hn.fit_lr(Xtr, y[tr], hn.balanced_weights(y[tr]), 1.0)
    d = clf.decision_function(Xtr)
    marg = np.where(y[tr] == 1, d, -d)
    assert int((marg <= 0).sum()) == 0, "in-sample errors would make the in-sample family usable"
