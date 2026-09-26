"""Stopband probe (docs/specs/2026-09-26_channel-stopband.md): pure functions of
scripts/channel_stopband.py. Hermetic: synthetic signals and frames only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SR = 16000
_spec = importlib.util.spec_from_file_location("channel_stopband", REPO / "scripts" / "channel_stopband.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)


def _band_db(x: np.ndarray, lo: float, hi: float) -> float:
    X = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    f = np.fft.rfftfreq(x.size, 1 / SR)
    return float(10 * np.log10(X[(f >= lo) & (f < hi)].mean() + 1e-20))


def test_the_second_pass_leaves_the_passband_and_deepens_the_stopband():
    from hearsay.handcrafted import band_limit

    once = band_limit(np.random.default_rng(0).standard_normal(SR * 3).astype(np.float32) * 0.1)
    twice = sb.stopband(once, "twice")
    for lo, hi in ((300, 1000), (1000, 3000), (3000, 6500)):
        assert abs(_band_db(twice, lo, hi) - _band_db(once, lo, hi)) < 0.5, (lo, hi)
    assert _band_db(twice, 7500, 8000) < _band_db(once, 7500, 8000) - 20


def test_once_is_the_untouched_segment_and_unknown_conditions_are_refused():
    x = np.random.default_rng(1).standard_normal(SR).astype(np.float32)
    assert np.array_equal(sb.stopband(x, "once"), x)
    with pytest.raises(ValueError):
        sb.stopband(x, "thrice")


def _frame(n_real=3, n_spoof=3, shift=0.0):
    rows = []
    for i in range(n_real + n_spoof):
        lab = "bonafide" if i < n_real else "spoof"
        for k in sb.KINDS:
            v = float(i >= n_real) + 0.01 * i + (shift if k == "twice" else 0.0)
            rows.append({"key": f"p{i}|{k}|0", "path": f"p{i}", "kind": k, "label": lab,
                         **{c: v for c in (*sb.DETECTORS, *sb.FUSED)}, "e_applied": 0.0})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("kind", sb.KINDS)
def test_a_non_finite_value_under_either_condition_drops_the_clip(bad, kind):
    d = _frame()
    d.loc[(d.path == "p4") & (d.kind == kind), "spectra_aasist"] = bad
    once, twice, _y = sb.paired(d)
    assert len(once) == 5 and "p4" not in once.index and list(once.index) == list(twice.index)


def test_a_clip_scored_under_one_condition_only_is_dropped():
    d = _frame()
    d = d[~((d.path == "p1") & (d.kind == "twice"))]
    assert len(sb.paired(d)[0]) == 5


def test_shift_stats_and_flips():
    once = np.array([0.1, 0.4, 0.6, 0.9])
    twice = once + np.array([0.0, 0.2, -0.2, 0.0])
    y = np.array([0, 0, 1, 1])
    s = sb.shift_stats(once, twice, y)
    assert s["mean"] == 0 and s["max_abs"] == pytest.approx(0.2) and s["mean_real"] == pytest.approx(0.1)
    assert sb.flips(once, twice, 0.5) == 2


def test_reading_rule_each_guard():
    assert sb.responds(0.2, 1.0) == "responds"
    assert sb.responds(0.05, 1.0) == "no response"
    assert sb.responds(0.2, 0.0) == "inconclusive"
    assert sb.responds(float("nan"), 1.0) == "inconclusive"
    assert sb.responds(True, 1.0) == "inconclusive"
    assert sb.fused_responds(0.012, 0.06) == "responds"
    assert sb.fused_responds(0.012, 0.03) == "no response"
    assert sb.fused_responds(0.012, float("inf")) == "inconclusive"


def test_readout_short_cohort_is_inconclusive_and_a_shift_is_detected(monkeypatch):
    r = sb.readout(_frame(), iqr_of=lambda c: 1.0)
    assert r["n_paired"] == 6 and r["ledger"] == "inconclusive" and "short_cohort" in r
    monkeypatch.setattr(sb, "N_EXPECTED", 6)
    assert sb.readout(_frame(), iqr_of=lambda c: 1.0)["ledger"] == "measured, no response"
    shifted = sb.readout(_frame(shift=0.5), iqr_of=lambda c: 1.0)
    assert shifted["ledger"].startswith("responds") and shifted["verdicts"]["m1b_v3"] == "responds"


def test_error_only_frame_is_inconclusive():
    d = pd.DataFrame({"key": ["p0|once|0", "p0|twice|0"], "path": ["p0", "p0"], "kind": list(sb.KINDS),
                      "label": "spoof", "error": "DecodeError('x')"})  # fmt: skip
    r = sb.readout(d, iqr_of=lambda c: 1.0)
    assert r["n_paired"] == 0 and r["ledger"] == "inconclusive"
