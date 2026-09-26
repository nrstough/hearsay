"""v4 feature families (hearsay.hc_v4): key names, finiteness, no collision with the v3 columns,
and offset invariance (a shifted copy of the same clip gives the same statistics), per the
brief's rule that crop edges must not become a feature."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hearsay import SR
from hearsay.handcrafted import features
from hearsay.hc_v4 import FAMILIES, N_LFCC


def _harmonic(seed: int = 0, dur: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    f0 = 140 + 5 * np.sin(2 * np.pi * 0.7 * t)  # slowly varying pitch
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 20))
    env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 2.0 * t))
    return (0.05 * x * env + 0.002 * rng.standard_normal(t.size)).astype(np.float32)


def _noise(seed: int = 0, dur: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.05 * rng.standard_normal(int(dur * SR))).astype(np.float32)


V3 = set(features(_harmonic(), crop_mode="first4s"))
EXPECTED = {
    "lfcc": {f"lfcc{i}_{s}" for i in range(N_LFCC) for s in ("mean", "std", "dstd")},
    "phase": {
        "gd_std_mean", "gd_std_std", "gd_std_p10", "gd_std_p90", "gd_iqr_mean", "gd_iqr_std",
        "gd_iqr_p10", "gd_iqr_p90", "pc_err_median", "pc_err_mean", "pc_err_p90",
        "pc_frac_incoherent",
    },
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_family_keys_finite_and_disjoint_from_v3(family):
    for x in (_harmonic(), _noise()):
        f = features(x, crop_mode="first4s", families=(family,))
        new = {k: v for k, v in f.items() if k not in V3}
        assert set(new) == EXPECTED[family]
        assert all(type(v) is float and math.isfinite(v) for v in new.values())
        assert not (set(new) & V3)
    assert len(features(x, crop_mode="first4s")) == len(V3)  # no families -> v3 unchanged


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_offset_invariance(family):
    x = _harmonic(seed=3, dur=4.0)
    a = features(x, crop_mode="first4s", families=(family,))
    b = features(x[137:], crop_mode="first4s", families=(family,))
    for k in EXPECTED[family]:
        assert math.isclose(a[k], b[k], rel_tol=0.25, abs_tol=0.05), (k, a[k], b[k])


def test_phase_family_separates_coherent_from_incoherent_phase():
    """A harmonic signal has structured phase; white noise does not."""
    h = features(_harmonic(), crop_mode="first4s", families=("phase",))
    n = features(_noise(), crop_mode="first4s", families=("phase",))
    assert h["pc_err_median"] < n["pc_err_median"]
    assert h["pc_frac_incoherent"] < n["pc_frac_incoherent"]


def test_lfcc_first_coefficient_tracks_spectral_tilt():
    """Noise is flat across 0-7 kHz; the harmonic series falls off, so its LFCC1 differs."""
    h = features(_harmonic(), crop_mode="first4s", families=("lfcc",))
    n = features(_noise(), crop_mode="first4s", families=("lfcc",))
    assert abs(h["lfcc1_mean"] - n["lfcc1_mean"]) > 0.5


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        features(_noise(), crop_mode="first4s", families=("nope",))
