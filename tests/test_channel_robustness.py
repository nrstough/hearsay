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


# ------------------------------------------------------------------ E: perturbations and verdict

m3p = _load("m3_probes")


def test_each_perturbation_changes_the_signal_and_none_does_not():
    x = _speechlike(2.0)
    assert np.array_equal(m3p.perturb(x, "none", 0), x)
    kinds = ("noise20", "speed", "shift1", "mp3") if HAVE_FFMPEG else ("noise20", "speed", "shift1")
    for k in kinds:
        y = m3p.perturb(x, k, 0)
        assert y.shape != x.shape or not np.array_equal(y, x), k


def test_noise_hits_twenty_db_snr():
    x = _speechlike(3.0)
    assert m3p.snr_db(x, m3p.perturb(x, "noise20", 7)) == pytest.approx(20.0, abs=0.5)


def test_shift_is_exactly_one_sample():
    x = _speechlike(1.0)
    y = m3p.perturb(x, "shift1", 0)
    assert np.array_equal(y[1:], x[:-1]) and y[0] == x[-1]


@pytest.mark.parametrize(("seed", "ratio"), [(0, 50 / 51), (1, 51 / 50)])
def test_speed_changes_length_by_two_percent(seed, ratio):
    x = _speechlike(3.0)
    assert abs(m3p.perturb(x, "speed", seed).size - x.size * ratio) <= 1


def test_unknown_perturbation_is_refused():
    with pytest.raises(ValueError):
        m3p.perturb(_speechlike(1.0), "reverb", 0)


@pytest.mark.parametrize("lag", [0, 37, 1105, -250])
def test_align_to_undoes_a_codec_style_delay_and_restores_length(lag):
    x = _speechlike(2.0)
    y = np.r_[np.zeros(lag, np.float32), x, np.zeros(500, np.float32)] if lag >= 0 else x[-lag:]
    z = m3p.align_to(y.astype(np.float32), x)
    assert z.size == x.size
    k = x.size - abs(lag) - 10
    assert np.allclose(z[abs(lag) : k] if lag < 0 else z[:k], x[abs(lag) : k] if lag < 0 else x[:k], atol=1e-6)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
@pytest.mark.parametrize("dur", [0.4, 3.0])
def test_mp3_perturbation_keeps_length_and_alignment(dur):
    x = _speechlike(dur)
    y = m3p.perturb(x, "mp3", 0)
    assert y.size == x.size and y.dtype == np.float32
    assert np.corrcoef(x, y)[0, 1] > 0.8  # aligned (an unaligned round-trip correlates near 0)


def test_delta_auc_sign_and_pairing():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    clean = y + 0.01 * rng.standard_normal(400)
    randomised = rng.standard_normal(400)
    assert m3p.delta_auc(y, clean, randomised) == pytest.approx(-0.5, abs=0.1)
    pert = clean.copy()
    pert[:10] = np.nan  # unpaired rows are dropped, not scored
    assert m3p.delta_auc(y, clean, pert) == pytest.approx(0.0, abs=1e-9)


def test_m3_verdict_each_guard_alone_and_nan_is_inconclusive():
    assert m3p.m3_verdict(0.01, 0.01, 0.01, 0.0) == ("kept", [])
    assert m3p.m3_verdict(0.05, 0.01, 0.01, 0.0)[0] == "at risk"  # (a)
    assert m3p.m3_verdict(0.015, 0.005, 0.01, 0.0)[0] == "kept"  # (a) needs > 0.02 absolute too
    assert m3p.m3_verdict(0.01, 0.01, 0.06, 0.0)[0] == "at risk"  # (c)
    assert m3p.m3_verdict(0.01, 0.01, 0.01, 0.02)[0] == "at risk"  # (d)
    assert m3p.m3_verdict(float("nan"), 0.01, 0.01, 0.0)[0] == "inconclusive"
    assert m3p.m3_verdict(0.01, 0.01, None, 0.0)[0] == "inconclusive"


def test_spearman_gap_zero_for_identical_rankings():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(300)
    r = m3p.spearman_gap(a, a, a[:200], a[:200], n=200)
    assert r["gap"] == 0 and r["gap_ci95"] == [0.0, 0.0]


# ------------------------------------------------------------------ B: codec statistics and match rule

cc = _load("channel_codec")


def _lowpassed(cut_hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    x = np.random.default_rng(0).standard_normal(SR * 3) * 0.1
    return sosfiltfilt(butter(12, cut_hz, fs=SR, output="sos"), x).astype(np.float32)


def test_highband_stats_see_where_the_wall_is():
    low, high = cc.highband_stats(_lowpassed(7000)), cc.highband_stats(_lowpassed(7750))
    assert low["drop_7500_vs_6500"] < high["drop_7500_vs_6500"] - 10
    assert all(np.isfinite(v) for v in low.values())


def test_match_distance_units_and_degenerate_column():
    import pandas as pd

    rng = np.random.default_rng(0)
    t = pd.DataFrame({s: rng.normal(0, 1, 2000) for s in cc.STATS})
    t["floor_p2_db"] = 3.0  # zero IQR: excluded, never a division by zero
    same, excl = cc.match_distance(t, t)
    assert same == 0 and excl == ["floor_p2_db"]
    iqr = t.drop(columns="floor_p2_db").quantile(0.75) - t.drop(columns="floor_p2_db").quantile(0.25)
    shifted = t.copy()
    for s in iqr.index:
        shifted[s] = t[s] + iqr[s]
    assert cc.match_distance(shifted, t)[0] == pytest.approx(1.0, abs=1e-9)


def test_codec_match_rule_needs_distance_and_both_hole_statistics():
    import pandas as pd

    def table(dist, gd, gl):
        return pd.DataFrame([
            {"variant": "raw", "distance": 5.0, "gap_deep_hole_frac": 1, "gap_local_hole_frac": 1},
            {"variant": "kaiser", "distance": 1.0, "gap_deep_hole_frac": 0.1, "gap_local_hole_frac": 0.1},
            {"variant": "mp3-32k@16000", "distance": dist, "gap_deep_hole_frac": gd, "gap_local_hole_frac": gl},
        ])  # fmt: skip

    assert cc.codec_match(table(0.75, 0.05, 0.05))["match"] is True
    assert cc.codec_match(table(0.75, 0.05, 0.2))["match"] is False  # worse on one hole statistic
    assert cc.codec_match(table(0.85, 0.05, 0.05))["match"] is False  # only 15% better
    r = cc.codec_match(table(0.85, 0.05, 0.05))
    assert r["closest_variant"] == "mp3-32k@16000" and r["matched_variant"] is None


def test_variant_grid_has_every_codec_alone_and_with_the_kaiser_pass():
    g = cc.variant_grid()
    codecs = [v for v in g if v not in ("raw", "kaiser") and not v.endswith("+kaiser")]
    assert g[:2] == ["raw", "kaiser"] and len(codecs) == 16
    assert all(f"{c}+kaiser" in g for c in codecs)


def test_perturb_readout_survives_a_one_class_slice(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(m3p, "THRESHOLDS_FROM_EXPORTS", False)

    rows = []
    for i in range(6):
        for k in m3p.PERTURBATIONS:
            rows.append({"key": f"p{i}|{k}|0", "path": f"p{i}", "kind": k, "label": "bonafide",
                         **{c: float(i) for c in (*m3p.MODELS, "fused_base", "fused")}, "e_applied": 0.0})
    r = m3p.perturb_readout(pd.DataFrame(rows))
    assert r["n_spoof"] == 0 and np.isnan(r["m1b_v3"]["none"]["eer"])



# ------------------------------------------------------------------ critique round 1 (13:30): the untested paths


def test_speaker_groups_put_lj_in_one_group_even_when_the_fold_file_splits_it_by_chapter():
    import pandas as pd

    f = pd.DataFrame({"speaker": ["lj", "lj", "lj", "libri_1", "libri_2"],
                      "group": ["lj_ch1", "lj_ch2", "lj_ch3", "g1", "g2"]})  # fmt: skip
    g = lam.speaker_groups(f)
    assert g.iloc[0] == g.iloc[1] == g.iloc[2] and g.nunique() == 3


def test_crossover_reading_needs_the_whole_interval_on_one_side():
    assert lam.crossover_reading(0.5, (0.4, 0.6), 0.32).startswith("above")
    assert lam.crossover_reading(0.2, (0.1, 0.3), 0.32).startswith("below")
    assert lam.crossover_reading(0.51, (0.12, 0.64), 0.32).startswith("indeterminate")
    assert lam.crossover_reading(float("nan"), (0.1, 0.6), 0.32).startswith("undefined")


def test_acc_ci_has_no_interval_when_most_resamples_are_refused():
    rng = np.random.default_rng(0)
    p_clean, p_wild = rng.uniform(0.4, 0.6, 200), rng.uniform(0.45, 0.65, 200)  # TPR - FPR ~ 0.2 or less
    g = np.arange(200).astype(str)
    lo, hi = lam.acc_ci(rng.uniform(0, 1, 300), p_clean, g, p_wild, g, n=100)
    assert np.isnan(lo) and np.isnan(hi)


def test_cache_identity_refuses_a_mismatch_and_an_orphan_cache(tmp_path):
    cache, meta = tmp_path / "c.csv", tmp_path / "c.meta.json"
    lam.check_cache_identity(meta, cache, {"v": 1})  # no cache yet: writes the identity
    cache.write_text("key\n")
    lam.check_cache_identity(meta, cache, {"v": 1})  # same identity: fine
    with pytest.raises(SystemExit):
        lam.check_cache_identity(meta, cache, {"v": 2})
    meta.unlink()
    with pytest.raises(SystemExit):
        lam.check_cache_identity(meta, cache, {"v": 1})


def test_json_safe_writes_strict_json():
    import json

    out = lam.json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2}})
    assert json.dumps(out, allow_nan=False) == '{"a": null, "b": [1.0, null], "c": {"d": 2}}'


def test_parse_key_and_crop_to():
    assert lam.parse_key("/a.wav") == ("/a.wav", "", None)
    assert lam.parse_key("/a.wav|mp3-64k@16000") == ("/a.wav", "mp3-64k@16000", None)
    assert lam.parse_key("/a.wav||3.41") == ("/a.wav", "", 3.41)
    x = _speechlike(6.0)
    y = lam.crop_to(x, 2.0, 1)
    assert y.size == 2 * SR and np.array_equal(y, lam.crop_to(x, 2.0, 1))
    assert lam.crop_to(x, None, 1) is x


def _p(effects, n=500):
    return {"mean_abs_d_auc": {"spectra_aasist": 0.001, "m1b_v3": 0.004}, "n_paired": n,
            "e_step_effect_min_dcf": dict(zip(m3p.PERTURBATIONS, effects))}  # fmt: skip


def test_verdict_from_is_inconclusive_on_any_nan_step_effect_even_when_max_would_skip_it():
    m = {"e_applied_share": 0.01, "n": 572}
    assert m3p.verdict_from(_p([0.0, -0.01, 0.0, -0.02, -0.02]), m)["verdict"] == "kept"
    assert m3p.verdict_from(_p([-0.008, float("nan"), 0.0, -0.02, -0.02]), m)["verdict"] == "inconclusive"
    assert m3p.verdict_from(_p([0.0, 0.0, 0.0, 0.03, 0.0]), m)["verdict"] == "at risk"
    assert m3p.verdict_from(_p([0.0] * 4), m)["verdict"] == "inconclusive"  # a perturbation missing


def test_verdict_from_is_inconclusive_when_a_probe_is_short_of_its_cohort():
    good = _p([0.0] * 5)
    assert m3p.verdict_from(_p([0.0] * 5, n=300), {"e_applied_share": 0.01, "n": 572})["verdict"] == "inconclusive"
    assert m3p.verdict_from(good, {"e_applied_share": 0.01, "n": 100})["verdict"] == "inconclusive"
    assert m3p.verdict_from(good, {"n": 572})["verdict"] == "inconclusive"  # damped share missing


def test_select_requested_drops_stale_rows_from_other_cohorts():
    import pandas as pd

    rows = pd.DataFrame({"path": ["a", "b"], "seed": [0, 1]})
    cache = pd.DataFrame({"key": ["a|none|0", "b|none|1", "a|none|7", "b|none|1"], "v": [1, 2, 3, 4]})
    got = m3p.select_requested(cache, rows, ("none",))
    assert sorted(got.key) == ["a|none|0", "b|none|1"] and got.set_index("key").loc["b|none|1", "v"] == 4


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
@pytest.mark.parametrize("spec", [("aac", 64, 44100), ("aac", 32, 16000), ("mp3", 64, 44100)])
def test_codec_round_trips_align_at_both_encode_rates(spec):
    from hearsay.compression import launder

    x = _speechlike(2.0)
    y = m3p.align_to(launder(x, *spec), x)
    assert y.size == x.size and np.corrcoef(x, y)[0, 1] > 0.8


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_a_failed_codec_round_trip_raises_instead_of_returning_audio():
    from hearsay.audio import DecodeError

    with pytest.raises((DecodeError, ValueError)):
        m3p.perturb(np.zeros(0, np.float32), "mp3", 0)


def test_highband_stats_ignore_leading_and_trailing_silence():
    x = _lowpassed(7000)
    padded = np.r_[np.zeros(SR, np.float32), x, np.zeros(SR, np.float32)]
    a, b = cc.highband_stats(x), cc.highband_stats(padded)
    for k in ("drop_7500_vs_6500", "lvl_6500", "hb_flatness_6_7k"):
        assert abs(a[k] - b[k]) < 1.0, (k, a[k], b[k])
