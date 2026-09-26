"""ENF (mains hum) detector: synthetic hum cases and the contract."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hearsay import SR
from hearsay.detectors import base
from hearsay.detectors.base import ClipContext, safe_run
from hearsay.detectors.enf import (
    SCORE_JUMPY,
    SCORE_NONE,
    SCORE_STABLE,
    EnfDetector,
    analyze,
)


def _speechlike(n: int, seed: int) -> np.ndarray:
    """Band-limited noise with slow amplitude modulation (stands in for speech)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = np.convolve(x, np.ones(8) / 8, mode="same")  # crude low-pass
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 3 * np.arange(n) / SR))
    return (0.1 * x * env).astype(np.float32)


def _hum(n: int, f0: float, level: float, drift_hz: float = 0.0, jump_at: float | None = None,
         jump_hz: float = 0.0) -> np.ndarray:  # fmt: skip
    t = np.arange(n) / SR
    f = f0 + drift_hz * np.sin(2 * np.pi * 0.1 * t)
    if jump_at is not None:
        f = f + np.where(t >= jump_at, jump_hz, 0.0)
    phase = 2 * np.pi * np.cumsum(f) / SR
    return (level * np.sin(phase)).astype(np.float32)


def clip(kind: str, dur: float = 6.0, seed: int = 0) -> ClipContext:
    n = int(dur * SR)
    x = _speechlike(n, seed)
    if kind == "stable":
        x = x + _hum(n, 60.0, 0.02, drift_hz=0.03)
    elif kind == "stable50":
        x = x + _hum(n, 50.0, 0.02, drift_hz=0.03)
    elif kind == "jumpy":
        x = x + _hum(n, 60.0, 0.02, jump_at=dur / 2, jump_hz=1.0)
    return ClipContext.from_array(x)


@pytest.fixture
def det():
    return EnfDetector()


def test_stable_hum_is_found_and_scores_mildly_real(det):
    r = safe_run(det, clip("stable"))
    assert r.status == "ok" and r.score == SCORE_STABLE
    assert r.features["enf_present"] == 1.0 and r.features["enf_stable"] == 1.0
    assert r.features["enf_candidate_hz"] == 60.0 and abs(r.features["enf_freq_hz"] - 60.0) < 0.2
    assert "stable" in r.evidence and "60.0" in r.evidence


def test_fifty_hz_candidate(det):
    r = safe_run(det, clip("stable50"))
    assert r.score == SCORE_STABLE and r.features["enf_candidate_hz"] == 50.0


def test_no_hum_is_neutral(det):
    r = safe_run(det, clip("none"))
    assert r.status == "ok" and r.score == SCORE_NONE and r.features["enf_present"] == 0.0
    assert "no mains hum" in r.evidence


def test_jumping_hum_scores_mildly_synthetic(det):
    r = safe_run(det, clip("jumpy"))
    assert r.score == SCORE_JUMPY and r.features["enf_present"] == 1.0
    assert r.features["enf_stable"] == 0.0 and r.features["enf_freq_range_hz"] > 0.5
    assert "discontinuous" in r.evidence


def test_short_clip_still_scores(det):
    r = safe_run(det, clip("stable", dur=1.5))
    assert r.status == "ok" and r.features["enf_n_frames"] >= 1.0


def test_contract_and_determinism(det):
    c = clip("stable", seed=3)
    r1, r2 = safe_run(det, c), safe_run(det, c)
    assert r1 == r2 and r1.name == "enf" and "enf.analysis" in c.cache
    assert all(type(v) is float and math.isfinite(v) for v in r1.features.values())
    assert 0.0 <= r1.score <= 1.0 and r1.evidence.strip()


def test_analyze_keys():
    m = analyze(clip("none").audio)
    assert {"enf_candidate_hz", "enf_snr_db", "enf_frac_present", "enf_freq_hz",
            "enf_freq_std_hz", "enf_freq_range_hz", "enf_n_frames", "enf_present",
            "enf_stable"} == set(m)  # fmt: skip


def test_registered_under_its_name():
    assert "enf" in base.REGISTRY.names()
