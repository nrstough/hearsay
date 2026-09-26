"""Speaker-drift detector: the cosine rule with a stub encoder (no weights needed), the window
grid, and, when the ECAPA weights are present, the real encoder on a two-speaker splice."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hearsay import SR
from hearsay.detectors import base
from hearsay.detectors import speaker_drift as sd
from hearsay.detectors.base import ClipContext, safe_run


class StubEncoder:
    """Deterministic embeddings: one direction per 'speaker', chosen by window energy."""

    def encode_batch(self, w):
        import torch

        w = w.numpy() if hasattr(w, "numpy") else np.asarray(w)
        out = np.zeros((w.shape[0], 1, 192), np.float32)
        for i, win in enumerate(w):
            spk = 0 if win.std() < 0.2 else 1  # loud windows are the second voice
            out[i, 0, spk] = 1.0
            out[i, 0, 2] = 0.1  # a little shared component
        return torch.from_numpy(out)


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(sd, "_ENCODER", StubEncoder())
    yield
    monkeypatch.setattr(sd, "_ENCODER", None)


def test_windows_cover_the_clip_flush_at_the_end():
    x = (0.1 * np.random.default_rng(0).standard_normal(int(3.3 * SR))).astype(np.float32)
    w = sd.windows(x)
    assert w.shape[1] == SR and w.shape[0] >= 5
    assert sd.windows(np.zeros(100, np.float32)).shape == (1, SR)  # short clip: one padded window


def test_one_voice_is_consistent_and_two_voices_drift(stub):
    rng = np.random.default_rng(1)
    one = (0.1 * rng.standard_normal(4 * SR)).astype(np.float32)
    two = np.r_[0.1 * rng.standard_normal(2 * SR), 0.5 * rng.standard_normal(2 * SR)].astype(np.float32)
    a = safe_run(sd.DETECTOR, ClipContext.from_array(one))
    b = safe_run(sd.DETECTOR, ClipContext.from_array(two))
    assert a.status == "ok" and a.score == sd.SCORE_NONE and a.features["drift"] == 0.0
    assert a.features["cos_min"] > 0.99 and "one consistent voice" in a.evidence
    assert b.status == "ok" and b.score == sd.SCORE_DRIFT and b.features["drift"] == 1.0
    assert b.features["cos_min"] < sd.DRIFT_COS and "drifts" in b.evidence
    assert all(type(v) is float and math.isfinite(v) for v in b.features.values())


def test_single_window_clip_scores_neutral(stub):
    r = safe_run(sd.DETECTOR, ClipContext.from_array(np.zeros(SR // 2, np.float32)))
    assert r.status == "ok" and r.features["n_windows"] == 1.0 and r.score == sd.SCORE_NONE


def test_registered():
    assert "speaker_drift" in base.REGISTRY.names()


@pytest.mark.needs_weights
@pytest.mark.slow
def test_encoder_loads_offline_from_the_weights_dir_with_an_empty_hf_cache(tmp_path):
    """Docker has no HF cache: the encoder must load from weights/ alone (no Hub, no cache)."""
    if not (sd.WEIGHTS / "embedding_model.ckpt").exists():
        pytest.skip("ECAPA weights not present")
    import os
    import subprocess
    import sys

    code = (
        "import numpy as np; from hearsay import SR; from hearsay.detectors.base import ClipContext, safe_run; "
        "from hearsay.detectors.speaker_drift import DETECTOR, WEIGHTS; "
        "r = safe_run(DETECTOR, ClipContext.from_array((0.1*np.random.default_rng(0).standard_normal(3*SR)).astype('float32'))); "
        "assert r.status == 'ok', r.error; assert not (WEIGHTS / 'label_encoder.ckpt').is_symlink(); print('offline ok')"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HOME": str(tmp_path / "empty_hf_home")}  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         check=False)  # fmt: skip
    assert out.returncode == 0 and "offline ok" in out.stdout, out.stderr[-800:]


@pytest.mark.needs_weights
@pytest.mark.slow
def test_real_encoder_separates_a_two_speaker_splice():
    if not (sd.WEIGHTS / "embedding_model.ckpt").exists():
        pytest.skip("ECAPA weights not present")
    sd._ENCODER = None
    rng = np.random.default_rng(2)
    t = np.arange(2 * SR) / SR
    # two synthetic 'voices': different pitch, formant-like filters and breathiness
    def voice(f0, tilt, noise):
        ph = 2 * np.pi * np.cumsum(f0 + 8 * np.sin(2 * np.pi * 3 * t)) / SR
        src = sum(np.sin(k * ph) / k for k in range(1, 30))
        b = np.exp(-np.arange(64) / tilt)
        x = np.convolve(src, b / b.sum(), mode="same") + noise * rng.standard_normal(t.size)
        return (0.1 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)
    a, b = voice(110.0, 8.0, 0.02), voice(240.0, 30.0, 0.08)
    same = sd.analyze(np.r_[a, a])
    mixed = sd.analyze(np.r_[a, b])
    assert same["cos_min"] > mixed["cos_min"]
    sd._ENCODER = None
