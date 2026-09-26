"""Splice detector: synthetic seams and the contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hearsay import SR
from hearsay.detectors import base
from hearsay.detectors.base import ClipContext, safe_run
from hearsay.detectors.splice import SCORE_NONE, SCORE_SEAM, SpliceDetector, analyze


def _speechlike(n: int, seed: int, level: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.convolve(rng.standard_normal(n), np.ones(8) / 8, mode="same")
    x = x - np.convolve(x, np.ones(SR // 50) / (SR // 50), mode="same")  # no sub-50 Hz, as audio
    env = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 2.5 * np.arange(n) / SR))
    return (level * x * env).astype(np.float32)


def clip(kind: str, dur: float = 4.0, seed: int = 0) -> ClipContext:
    n = int(dur * SR)
    x = _speechlike(n, seed)
    if kind == "click":
        x[n // 2] += 0.6  # one-sample step
    elif kind == "dc":
        x[n // 2 :] += 0.05  # second half rides on a DC offset
    elif kind == "plosive":  # a 10 ms burst: many large steps, no single outlier
        x[n // 2 : n // 2 + SR // 100] += 0.5 * np.random.default_rng(9).standard_normal(SR // 100)
    return ClipContext.from_array(x)


@pytest.fixture
def det():
    return SpliceDetector()


def test_continuous_signal_is_neutral(det):
    r = safe_run(det, clip("none"))
    assert r.status == "ok" and r.score == SCORE_NONE and r.features["n_seams"] == 0.0
    assert "no clicks" in r.evidence


def test_click_is_a_seam(det):
    r = safe_run(det, clip("click"))
    assert r.score == SCORE_SEAM and r.features["n_clicks"] == 1.0
    assert abs(r.features["first_seam_s"] - 2.0) < 0.01 and "click" in r.evidence


def test_dc_jump_is_a_seam(det):
    r = safe_run(det, clip("dc"))
    assert r.score == SCORE_SEAM and r.features["n_dc_jumps"] >= 1.0
    assert r.features["max_dc_jump"] > 0.02 and "DC-offset" in r.evidence


def test_plosive_burst_is_not_a_click(det):
    r = safe_run(det, clip("plosive"))
    assert r.features["n_clicks"] == 0.0


def test_short_and_silent_clips_score(det):
    assert safe_run(det, ClipContext.from_array(np.zeros(100, np.float32))).status == "ok"
    assert safe_run(det, clip("none", dur=0.3)).status == "ok"


def test_contract_and_determinism(det):
    c = clip("click", seed=2)
    r1, r2 = safe_run(det, c), safe_run(det, c)
    assert r1 == r2 and r1.name == "splice" and "splice.analysis" in c.cache
    assert all(type(v) is float and math.isfinite(v) for v in r1.features.values())


def test_analyze_keys():
    assert set(analyze(clip("none").audio)) == {"n_clicks", "max_click_ratio", "n_dc_jumps",
                                                "max_dc_jump", "floor_range_db", "n_seams",
                                                "first_seam_s"}  # fmt: skip


def test_registered_under_its_name():
    assert "splice" in base.REGISTRY.names()
