"""M5 augmentation (spec appendix D1-D9)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from hearsay import SR
from hearsay.augment import (
    OPS,
    Augmenter,
    add_noise,
    band_limit,
    gain_clip,
    holdout_slice,
    rawboost_conv,
    reverb,
    trim_jitter,
)


def _speech_like(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 560, 1100, 2200), 1))
    x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(t.size)
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


@pytest.mark.parametrize("op", [o for o in OPS if o != "codec"])
def test_d1_every_op_keeps_length_and_is_finite(op):
    x = _speech_like()
    rng = np.random.default_rng(1)
    y = {"noise": lambda: add_noise(x, rng, 15.0), "band": lambda: band_limit(x, rng, 4000.0),
         "reverb": lambda: reverb(x, rng, 0.5), "gain_clip": lambda: gain_clip(x, rng, 6.0),
         "rawboost_conv": lambda: rawboost_conv(x, rng)}[op]()  # fmt: skip
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    assert not np.array_equal(y, x)


def test_d2_noise_snr_is_accurate():
    x = _speech_like()
    for snr in (5.0, 15.0, 30.0):
        y = add_noise(x, np.random.default_rng(0), snr)
        n = y - x
        got = 10 * np.log10(np.mean(x**2) / np.mean(n**2))
        assert abs(got - snr) < 0.5


def test_d3_band_limit_attenuates_above_cutoff():
    rng = np.random.default_rng(0)
    white = rng.standard_normal(4 * SR).astype(np.float32)
    for cut, stop_from, min_db in ((3400.0, 4250.0, 40), (4000.0, 5000.0, 40), (7000.0, 7800.0, 8)):
        y = band_limit(white, rng, cut)
        f, p = signal.welch(y, fs=SR, nperseg=2048)
        inband = p[(f > 200) & (f < cut * 0.8)].mean()
        stop = p[f >= stop_from].mean()  # 7 kHz leaves only 7.8-8 kHz below Nyquist
        assert 10 * np.log10(inband / stop) > min_db


def test_d4_reverb_keeps_onset_and_is_nonzero():
    x = np.zeros(2 * SR, np.float32)
    x[SR // 2] = 1.0  # an impulse: the output's first peak is the direct path
    y = reverb(x, np.random.default_rng(0), 0.5)
    assert y.any() and np.isfinite(y).all()
    lag = int(np.argmax(np.abs(y))) - SR // 2
    assert abs(lag) < SR // 200  # < 5 ms


def test_d5_pure_gain_is_a_no_op_after_normalization_but_clipping_is_not():
    from hearsay.embed import normalize_windows

    x = _speech_like()
    g = (x * 10 ** (6 / 20)).astype(np.float32)  # pure gain, no clip
    assert np.allclose(normalize_windows(x[None]), normalize_windows(g[None]), atol=1e-4)
    c = gain_clip(x, np.random.default_rng(0), 12.0)  # peak 0.5 -> 2.0, clipped at 1.0
    assert np.abs(c).max() <= 1.0
    assert not np.allclose(normalize_windows(x[None]), normalize_windows(c[None]), atol=1e-2)


def test_d6_augmentation_rate_is_class_blind_with_counterfactual():
    aug = Augmenter(p_aug=0.65, p_rawboost=0.25)
    labels = np.array([0, 1] * 2000)
    rates = {0: [], 1: []}
    for i, y in enumerate(labels):
        plan = aug.plan(np.random.default_rng([7, i]), has_codec=bool(i % 3))
        rates[int(y)].append(bool(plan))
    r0, r1 = np.mean(rates[0]), np.mean(rates[1])
    assert abs(r0 - r1) < 0.03 and 0.6 < r0 < 0.7

    # counterfactual: a label-aware augmenter would fail this check
    def biased_plan(rng, label):
        return aug.plan(rng, False) if label == 1 else []

    b0 = np.mean([bool(biased_plan(np.random.default_rng([7, i]), 0)) for i in range(2000)])
    b1 = np.mean([bool(biased_plan(np.random.default_rng([7, i]), 1)) for i in range(2000)])
    assert abs(b0 - b1) > 0.5


def test_d7_silent_input_stays_finite():
    z = np.zeros(2 * SR, np.float32)
    aug = Augmenter(p_aug=1.0, p_rawboost=1.0)
    for i in range(20):
        y = aug(z, np.random.default_rng(i))
        assert y.shape == z.shape and np.isfinite(y).all()


def test_d8_clean_is_identity_and_d9_holdout_slice_is_fixed():
    x = _speech_like()
    assert Augmenter.clean(x) is x
    a, b = holdout_slice(50, seed=1234), holdout_slice(50, seed=1234)
    assert a == b
    ops = [p[0][0] for p in a]
    assert set(ops) == {"noise", "band", "reverb", "gain_clip", "rawboost_conv"}
    assert holdout_slice(50, seed=99) != a


def test_augmenter_is_deterministic_given_rng():
    x = _speech_like()
    aug = Augmenter()
    y1 = aug(x, np.random.default_rng([3, 4]), has_codec=True)
    y2 = aug(x, np.random.default_rng([3, 4]), has_codec=True)
    assert np.array_equal(y1, y2)


def test_trim_jitter_returns_a_slice_of_the_input():
    x = np.concatenate([np.zeros(SR, np.float32), _speech_like(2.0), np.zeros(SR // 2, np.float32)])
    for i in range(10):
        y = trim_jitter(x, np.random.default_rng(i))
        assert 2 * SR - SR // 10 <= y.size <= x.size and np.isfinite(y).all()
