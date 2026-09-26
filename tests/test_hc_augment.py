"""Training-side augmentation for the handcrafted features (v5): tilt + low-pass, codec
laundering, reproducible draws, and that test-time paths never see it."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from hearsay import SR
from hearsay.handcrafted import apply_augment, draw_augment, features, tilt_lowpass


def _noise(seed: int = 0, dur: float = 3.0) -> np.ndarray:
    return (0.05 * np.random.default_rng(seed).standard_normal(int(dur * SR))).astype(np.float32)


def has_encoder(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                         check=False)  # fmt: skip
    return name in out.stdout


def test_tilt_lowpass_darkens_and_band_limits():
    x = _noise()
    y = tilt_lowpass(x, -4.0, 5000.0)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    fx, fy = (features(v, crop_mode="first4s") for v in (x, y))
    assert fy["centroid_mean"] < fx["centroid_mean"]
    assert fy["rolloff95_mean"] < fx["rolloff95_mean"]
    assert fy["hf_ratio_6k_mean"] < 0.1 * fx["hf_ratio_6k_mean"] + 1e-9


def test_positive_tilt_brightens():
    x = _noise(1)
    y = tilt_lowpass(x, 3.0, 7000.0)
    assert features(y, crop_mode="first4s")["centroid_mean"] > features(x, crop_mode="first4s")["centroid_mean"]


def test_draws_are_reproducible_and_respect_fractions():
    a = [draw_augment(np.random.default_rng(r), 0.35, 0.35) for r in range(200)]
    b = [draw_augment(np.random.default_rng(r), 0.35, 0.35) for r in range(200)]
    assert a == b
    n_l = sum("launder:" in s for s in a)
    n_t = sum("tilt:" in s for s in a)
    assert 40 <= n_l <= 100 and 40 <= n_t <= 100  # ~70 each of 200
    assert all(draw_augment(np.random.default_rng(r), 0.0, 0.0) == "" for r in range(20))
    assert any("+" in s for s in a)  # both can apply


def test_apply_augment_clean_and_tilt_and_unknown():
    x = _noise(2)
    assert apply_augment(x, "") is x
    y = apply_augment(x, "tilt:-2.50:6000")
    assert y.shape == x.shape and not np.allclose(x, y)
    with pytest.raises(ValueError):
        apply_augment(x, "reverb:1")


@pytest.mark.skipif(not has_encoder("libmp3lame"), reason="ffmpeg lacks libmp3lame")
def test_apply_augment_launder_then_tilt():
    x = _noise(3)
    y = apply_augment(x, "launder:mp3-48k@22050+tilt:1.00:6500")
    assert abs(y.size - x.size) < SR // 4 and np.isfinite(y).all()


def test_features_for_path_signature_has_augment_default_off():
    import inspect

    from hearsay.handcrafted import features_for_path

    assert inspect.signature(features_for_path).parameters["augment"].default == ""
