"""Channel-robustness rung (docs/specs/2026-09-26_channel-robustness.md): the pure functions of
scripts/channel_lambda.py (A), scripts/m3_probes.py (E) and scripts/channel_codec.py (B).
Hermetic: synthetic signals and arrays only; no data, weights or ffmpeg-dependent assertions
beyond what the codec helpers themselves need."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SR = 16000


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lam = _load("channel_lambda")


def _speechlike(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Syllable-rate amplitude-modulated harmonic signal with short pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 12))
    env = (np.sin(2 * np.pi * 4 * t) > -0.2).astype(float) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    env[(t % 1.0) > 0.8] = 0.0  # a pause every second
    return (0.3 * x * env + 1e-4 * rng.standard_normal(t.size)).astype(np.float32)


# ------------------------------------------------------------------ A: per-clip statistics


def test_channel_stats_finite_on_degenerate_inputs():
    for x in (np.zeros(SR * 2, np.float32), _speechlike(0.3),
              np.sign(np.sin(2 * np.pi * 200 * np.arange(SR * 2) / SR)).astype(np.float32)):
        f = lam.channel_stats(x)
        assert f and all(np.isfinite(v) for v in f.values()), f


def test_channel_stats_deterministic_and_level_invariant():
    x = _speechlike()
    a, b, c = lam.channel_stats(x), lam.channel_stats(x.copy()), lam.channel_stats(0.1 * x)
    assert a == b
    for k in a:  # 1%: the 1e-12 floor inside the dB conversions makes near-silent bins level-dependent
        assert abs(a[k] - c[k]) <= 1e-2 * max(1.0, abs(a[k])), (k, a[k], c[k])


def test_noise_raises_the_floor_level():
    x = _speechlike()
    noisy = x + 0.01 * np.random.default_rng(1).standard_normal(x.size).astype(np.float32)
    assert lam.channel_stats(noisy)["floor_level_db"] > lam.channel_stats(x)["floor_level_db"] + 3


def test_reverb_tail_slows_the_decay():
    x = _speechlike()
    ir = np.exp(-np.arange(int(0.4 * SR)) / (0.12 * SR)).astype(np.float32)
    ir *= np.random.default_rng(2).standard_normal(ir.size).astype(np.float32)
    ir[0] = 1.0
    wet = np.convolve(x, ir)[: x.size].astype(np.float32)
    dry_s = lam.extra_stats(x)["decay_slope_db_per_s"]
    wet_s = lam.extra_stats(wet / np.abs(wet).max() * 0.5)["decay_slope_db_per_s"]
    assert wet_s > dry_s  # less negative dB/s = slower decay


def test_stationarity_orders_steady_vs_gated_floor():
    rng = np.random.default_rng(3)
    steady = 0.01 * rng.standard_normal(SR * 4)
    gated = steady * np.repeat(np.tile([1.0, 0.02], 4), SR // 2)
    s_steady = lam.extra_stats(steady.astype(np.float32))["floor_stationarity_db"]
    s_gated = lam.extra_stats(gated.astype(np.float32))["floor_stationarity_db"]
    assert s_gated > s_steady + 5


# ------------------------------------------------------------------ A: folds and estimators


def test_grouped_folds_never_split_a_group_and_the_leak_check_can_fail():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 40, 800).astype(str)
    fold = lam.grouped_folds(groups, 5)
    assert set(fold) == set(range(5)) and not lam.folds_leak(groups, fold)
    naive = rng.integers(0, 5, 800)  # a clip-level split: groups spread over folds
    assert lam.folds_leak(groups, naive)


def test_grouped_oof_fills_every_row():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (200, 3)), rng.normal(2, 1, (200, 3))]
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    g = np.r_[rng.integers(0, 20, 200), 100 + rng.integers(0, 20, 200)].astype(str)
    p = lam.grouped_oof(X, y, g)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


@pytest.mark.parametrize("true_mix", [0.0, 0.3, 0.7, 1.0])
@pytest.mark.parametrize("sep", [1.5, 3.0])
def test_adjusted_estimate_recovers_known_mixtures_including_overlap(true_mix, sep):
    rng = np.random.default_rng(int(true_mix * 10 + sep * 100))
    Xc, Xw = rng.normal(0, 1, (1500, 2)), rng.normal(sep / np.sqrt(2), 1, (1500, 2))
    X, y = np.r_[Xc, Xw], np.r_[np.zeros(1500), np.ones(1500)].astype(int)
    g = np.arange(3000).astype(str)
    oof = lam.grouped_oof(X, y, g)
    n_w = round(true_mix * 1000)
    Xt = np.r_[rng.normal(0, 1, (1000 - n_w, 2)), rng.normal(sep / np.sqrt(2), 1, (n_w, 2))]
    pt = lam.make_clf().fit(X, y).predict_proba(Xt)[:, 1]
    est = lam.acc_stat(pt, oof[y == 0], oof[y == 1])
    assert abs(est - true_mix) <= 0.08, (true_mix, sep, est)


def test_mean_posterior_is_biased_under_overlap_which_is_why_it_is_descriptive_only():
    rng = np.random.default_rng(5)
    X = np.r_[rng.normal(0, 1, (1500, 1)), rng.normal(1.0, 1, (1500, 1))]
    y = np.r_[np.zeros(1500), np.ones(1500)].astype(int)
    Xt = rng.normal(0, 1, (1000, 1))  # true mix 0
    pt = lam.make_clf().fit(X, y).predict_proba(Xt)[:, 1]
    assert lam.lambda_mean_posterior(pt) > 0.2  # shrinks toward 0.5, far from the truth 0


def test_adjusted_estimate_refuses_a_weak_classifier_and_clips():
    v, why = lam.lambda_acc(0.5, 0.55, 0.45)
    assert np.isnan(v) and "too weak" in why
    assert lam.lambda_acc(0.01, 0.9, 0.05) == (0.0, "")
    assert lam.lambda_acc(0.99, 0.9, 0.05) == (1.0, "")


def test_feature_position_signs_and_degenerate_case():
    assert lam.feature_position(5, 0, 10) == 0.5
    assert lam.feature_position(-5, 0, 10) == -0.5
    assert lam.feature_position(15, 0, 10) == 1.5
    assert lam.feature_position(12, 10, 0) == pytest.approx(-0.2)
    assert np.isnan(lam.feature_position(3, 1, 1))


def test_cluster_index_resamples_whole_groups():
    g = np.repeat(np.array(["a", "b", "c"]), [5, 1, 3])
    idx = lam.cluster_index(g, np.random.default_rng(0))
    picked = g[idx]
    for u in np.unique(picked):
        assert (picked == u).sum() % (g == u).sum() == 0


def test_identifiability_rule_each_guard_fails_alone():
    good = {"ctl_vctk_clean_read": {"share_above_0.5": 0.1}, "ctl_diffssd_spoof": {"share_above_0.5": 0.0}}
    assert lam.identifiable(0.95, good, 0.5, 0.45) == (True, [])
    assert not lam.identifiable(0.7, good, 0.5, 0.45)[0]
    assert not lam.identifiable(0.95, {**good, "ctl_vctk_clean_read": {"share_above_0.5": 0.4}}, 0.5, 0.45)[0]
    assert not lam.identifiable(0.95, {**good, "ctl_diffssd_spoof": {"share_above_0.5": 0.3}}, 0.5, 0.45)[0]
    assert not lam.identifiable(0.95, good, 0.5, 0.2)[0]
    assert not lam.identifiable(0.95, {}, 0.5, 0.45)[0]  # missing controls never pass
    assert not lam.identifiable(0.95, good, float("nan"), 0.45)[0]  # NaN never passes


def test_novelty_share_near_one_percent_in_distribution_and_high_out_of_it():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (3000, 4))
    assert lam.novelty_share(X, rng.normal(0, 1, (3000, 4))) < 0.03
    assert lam.novelty_share(X, rng.normal(6, 1, (300, 4))) > 0.9


def test_novelty_share_survives_a_constant_column():
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(0, 1, (500, 3)), np.ones(500)]
    assert np.isfinite(lam.novelty_share(X, X[:50]))


pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
