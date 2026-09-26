"""M5 crops, batching and the train/test transform (spec appendix C1-C7)."""

from __future__ import annotations

import numpy as np
import pytest

from hearsay import SR
from hearsay.embed import TEST_DURATIONS, normalize_windows
from hearsay.m5_data import LengthMixer, collate, crop, deploy_transform, epoch_rng


def _clip(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.3 * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def test_c1_crops_reproducible_per_seed_epoch_and_change_per_epoch():
    x = _clip(10)
    a = crop(x, 3.4, epoch_rng(0, 1, 7))
    b = crop(x, 3.4, epoch_rng(0, 1, 7))
    c = crop(x, 3.4, epoch_rng(0, 2, 7))
    assert a.size == int(3.4 * SR) and np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_c2_no_tiling_short_clip_is_whole_plus_zeros_and_mask():
    short = _clip(2.0)
    c = crop(short, 5.0, epoch_rng(0, 0, 0))
    assert c.size == short.size and np.array_equal(c, short)
    xs, mask = collate([c, _clip(5.0, 1)])
    assert xs.shape == (2, 5 * SR) and mask.shape == xs.shape
    assert mask[0].sum() == short.size and mask[1].sum() == 5 * SR
    assert not xs[0, short.size:].any()  # zeros, not repeated audio
    # autocorrelation at lag = clip length is ~0 (a tiled clip would peak there)
    v = xs[0]
    lag = short.size
    r = float(np.dot(v[:-lag], v[lag:]) / (np.dot(v, v) + 1e-9))
    assert abs(r) < 0.05


def test_c3_length_mix_shares_and_range():
    if not TEST_DURATIONS.exists():
        pytest.skip("test durations not present")
    mixer = LengthMixer(seed=0, p_test=0.7)
    draws = np.array([mixer.draw() for _ in range(10_000)])
    assert 3.0 <= draws.min() and draws.max() <= 14.0
    import pandas as pd

    test_set = set(np.round(pd.read_csv(TEST_DURATIONS).duration_s.to_numpy(), 6))
    share = np.mean([round(d, 6) in test_set for d in draws])
    assert 0.67 <= share <= 0.73


def test_c5_normalization_over_valid_samples_only():
    a, b = _clip(2.0, 3), _clip(4.0, 4)
    xs, _ = collate([a, b])
    assert np.allclose(xs[0, : a.size], normalize_windows(a[None])[0], atol=1e-6)
    assert np.allclose(xs[1], normalize_windows(b[None])[0], atol=1e-6)
    assert abs(xs[0, : a.size].mean()) < 1e-4 and abs(xs[0, : a.size].std() - 1) < 1e-3


def test_c6_bundle_path_matches_deploy_path_and_caps():
    """Bundle clips are trim_silence'd at build time and band-limited at train time; the raw
    deploy path band-limits then trims (M1's prepare_segment). The two may differ by a trim
    frame at the edges only."""
    from hearsay.audio import trim_silence
    from hearsay.m5_data import bundle_transform

    x = np.concatenate([np.zeros(SR // 2, np.float32), _clip(6.0, 5), np.zeros(SR // 4, np.float32)])
    dep = deploy_transform(x)                      # raw path
    bun = bundle_transform(trim_silence(x))        # bundle path
    assert abs(dep.size - bun.size) <= SR // 50    # at most one 20 ms trim frame
    n = min(dep.size, bun.size) - SR // 50
    assert np.abs(dep[:n] - bun[:n]).max() < 0.05  # same audio, same filter, same normalization
    long = _clip(16.0, 6)
    assert deploy_transform(long).size == 8 * SR   # fusion cap = prepare_segment's default
    assert deploy_transform(long, max_s=14.0).size == 14 * SR
    assert bundle_transform(long, max_s=14.0).size == 14 * SR


def test_c7_one_length_per_batch_and_padding_fraction():
    clips = [crop(_clip(9.0, i), 4.0, epoch_rng(0, 0, i)) for i in range(4)]
    assert len({c.size for c in clips}) == 1
    _, mask = collate(clips)
    assert mask.all()  # equal-length crops: no padding at all
