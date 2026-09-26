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
    BLOCK_TOP,
    FAILURE_TOP,
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


def test_default_answer_policy_pins_gated_files_strictly_below_every_scored_file():
    fused = np.array([0.9, 0.0009, 0.0, 0.6, 0.02, 0.5])  # the runner scores some real files at 0.0009 and 0.0
    is_speech = np.array([1, 1, 1, 0, 0, 1])
    out = apply_default_answer(fused, is_speech)
    det, gated = out[is_speech == 1], out[is_speech == 0]
    assert (det >= BLOCK_TOP).all() and det.max() <= 1.0
    assert (gated < BLOCK_TOP).all() and (gated >= FAILURE_TOP).all()
    assert gated.max() < det.min()  # strictly below, even below the 0.0 file
    assert np.array_equal(np.argsort(det), np.argsort(fused[is_speech == 1]))  # ranking kept
    assert out[2] == BLOCK_TOP and out[0] == BLOCK_TOP + (1 - BLOCK_TOP) * 0.9
    assert out[3] > out[4]  # within the block, the weak signal (fused here) orders files


def test_block_uses_order_by_keys_and_puts_failures_at_the_bottom():
    fused = np.array([0.3, 0.3, 0.3, np.nan, 0.3])
    is_speech = np.zeros(5, dtype=bool)
    weak = np.array([0.2, 0.8, 0.2, 0.9, 0.2])
    keys = ["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"]
    out = apply_default_answer(fused, is_speech, order_by=weak, keys=keys, failed=[0, 0, 0, 0, 1])
    assert out[1] > out[0] and out[1] > out[2] and out[1] > out[4]  # highest weak signal on top
    assert len(set(out.tolist())) == 5  # jitter: no two files share a value
    assert out[3] < FAILURE_TOP and out[4] < FAILURE_TOP  # NaN signal and flagged failure: bottom
    assert (out[[0, 1, 2]] >= FAILURE_TOP).all() and (out < BLOCK_TOP).all()
    again = apply_default_answer(fused, is_speech, order_by=weak, keys=keys, failed=[0, 0, 0, 0, 1])
    assert np.array_equal(out, again)  # deterministic
    with pytest.raises(ValueError):
        apply_default_answer(fused, is_speech, keys=keys[:3])


def test_only_the_gate_flag_gates_a_file():
    """Disagreement or low confidence is a fusion matter: a speech file always stays determinate."""
    out = apply_default_answer(np.array([0.5, 0.0, 1.0]), np.array([1, 1, 1]))
    assert (out >= BLOCK_TOP).all()


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
