"""Non-speech gate: the preflight cases (silence, chord) plus tone, noise, and speech-like
material; the default-answer policy; the contract."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hearsay import SR
from hearsay.detectors import base
from hearsay.detectors.base import ClipContext, safe_run
from hearsay.detectors.speech_gate import (
    DEFAULT_ANSWER,
    SpeechGateDetector,
    analyze,
    apply_default_answer,
)

REPO = Path(__file__).resolve().parents[1]


def _t(dur=4.0):
    return np.arange(int(dur * SR)) / SR


def silence():
    return np.zeros(4 * SR, np.float32)


def chord():  # the submission preflight's "music"
    t = _t()
    return sum(0.1 * np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0)).astype(np.float32)


def tone():
    return (0.1 * np.sin(2 * np.pi * 220.0 * _t())).astype(np.float32)


def white_noise(seed=0):
    return (0.05 * np.random.default_rng(seed).standard_normal(4 * SR)).astype(np.float32)


def speechlike(seed=0):
    """Harmonic source with a moving pitch, syllable-rate loudness modulation, formant-like
    filtering and short pauses: what the gate must let through."""
    rng = np.random.default_rng(seed)
    t = _t()
    f0 = 130 + 25 * np.sin(2 * np.pi * 0.9 * t) + 10 * np.sin(2 * np.pi * 3.1 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    src = sum(np.sin(k * ph) / k for k in range(1, 25))
    env = np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None) ** 0.5 * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    env[int(1.8 * SR) : int(2.2 * SR)] = 0.0  # a pause
    x = src * env + 0.01 * rng.standard_normal(t.size)
    b = np.exp(-np.arange(64) / 12.0); x = np.convolve(x, b / b.sum(), mode="same")  # spectral tilt
    return (0.1 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


@pytest.fixture
def det():
    return SpeechGateDetector()


@pytest.mark.parametrize(
    ("make", "reason"),
    [(silence, "silence"), (chord, "tone"), (tone, "tone"), (white_noise, "noise")],
)
def test_non_speech_is_gated_with_the_right_reason(det, make, reason):
    r = safe_run(det, ClipContext.from_array(make()))
    assert r.status == "ok" and r.score == 0.5 and r.features["is_speech"] == 0.0
    assert r.features[f"reason_{reason}"] == 1.0, {k: v for k, v in r.features.items() if k.startswith("reason")}
    assert "default answer" in r.evidence


def test_speechlike_passes(det):
    r = safe_run(det, ClipContext.from_array(speechlike()))
    assert r.features["is_speech"] == 1.0, r.features
    assert r.features["voiced_frac"] > 0.3 and r.features["energy_db_std"] > 3.0
    assert "speech present" in r.evidence


def test_quiet_but_real_level_speech_is_not_silence(det):
    r = safe_run(det, ClipContext.from_array(speechlike() * 0.03))  # about -50 dBFS peak
    assert r.features["reason_silence"] == 0.0 and r.features["is_speech"] == 1.0


def test_contract_and_determinism(det):
    c = ClipContext.from_array(speechlike(3))
    r1, r2 = safe_run(det, c), safe_run(det, c)
    assert r1 == r2 and r1.name == "speech_gate" and "speech_gate.analysis" in c.cache
    assert all(type(v) is float and math.isfinite(v) for v in r1.features.values())
    assert safe_run(det, ClipContext.from_array(np.zeros(100, np.float32))).status == "ok"


def test_default_answer_policy_orders_gated_below_speech():
    fused = np.array([0.9, 0.1, 0.6, 0.02, 0.5])
    is_speech = np.array([1, 1, 0, 0, 1])
    out = apply_default_answer(fused, is_speech)
    assert np.array_equal(out[[0, 1, 4]], fused[[0, 1, 4]])
    assert (out[[2, 3]] < fused[[0, 1, 4]].min()).all()
    assert out[2] > out[3]  # deterministic order among gated files follows the fused score
    assert abs(out[3] - DEFAULT_ANSWER) < 1e-3 and DEFAULT_ANSWER < 0.5


def test_registered():
    assert "speech_gate" in base.REGISTRY.names()


@pytest.mark.needs_data
@pytest.mark.slow
def test_real_test_clips_pass_the_gate():
    wavs = sorted((REPO / "data" / "nsa" / "HackGTHearsayTesting").glob("*.wav"))[:60]
    if not wavs:
        pytest.skip("NSA test set not present")
    passed = [analyze(ClipContext(p).audio)["is_speech"] for p in wavs]
    assert np.mean(passed) >= 0.98, np.mean(passed)
