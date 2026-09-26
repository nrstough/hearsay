"""Handcrafted spectral + prosody detector (D-track): the contract wrapper around
`hearsay.handcrafted.features` and the model saved by scripts/train_handcrafted.py.

Score = P(synthetic) from the saved fold-validated classifier; evidence names the top
contributing features (see `hearsay.detectors._learned`). Features are the raw feature values
plus `hc_logit`. Bundle: models/hc_<kind>_<stamp>/model.joblib, newest stamp by default.
"""

from __future__ import annotations

from pathlib import Path

from hearsay.detectors._learned import MODELS, FeatureModelDetector, contributions
from hearsay.detectors._learned import latest_model_dir as _latest
from hearsay.detectors.base import REGISTRY, register
from hearsay.handcrafted import features

NAME = "handcrafted"

# Plain-English names for the evidence string; unnamed features fall back to their key.
LABELS = {
    "hf_ratio_4k_mean": "energy share above 4 kHz",
    "hf_ratio_6k_mean": "energy share above 6 kHz",
    "hf_ratio_7k_mean": "energy share above 7 kHz",
    "hf_ratio_4k_std": "variability of energy above 4 kHz",
    "hf_ratio_6k_std": "variability of energy above 6 kHz",
    "hf_ratio_7k_std": "variability of energy above 7 kHz",
    "rolloff85_mean": "85% spectral roll-off",
    "rolloff95_mean": "95% spectral roll-off",
    "rolloff85_std": "variability of the 85% roll-off",
    "rolloff95_std": "variability of the 95% roll-off",
    "centroid_mean": "spectral centroid",
    "centroid_std": "spectral centroid variability",
    "bandwidth_mean": "spectral bandwidth",
    "bandwidth_std": "spectral bandwidth variability",
    "flatness_mean": "spectral flatness (noisiness)",
    "flatness_std": "spectral flatness variability",
    "flux_mean": "spectral flux (frame-to-frame change)",
    "flux_std": "spectral flux variability",
    "voiced_frac": "voiced fraction",
    "f0_median": "median pitch",
    "f0_iqr_log": "pitch variability (IQR of log F0)",
    "f0_std_log": "pitch spread (std of log F0)",
    "f0_slope_std": "pitch-contour jitter (std of F0 slope)",
    "energy_db_std": "loudness dynamics",
    "energy_db_p10": "quiet-frame level",
    "pause_frac": "pause fraction",
    "pause_count": "pause count",
    "zcr_mean": "zero-crossing rate",
    "zcr_std": "zero-crossing variability",
}
LABELS |= {f"contrast{i}_mean": f"spectral contrast, band {i}" for i in range(6)}
LABELS |= {f"mfcc{i}_mean": f"MFCC {i} mean" for i in range(20)}
LABELS |= {f"mfcc{i}_std": f"MFCC {i} variability" for i in range(20)}
# v4 families (hearsay.hc_v4)
LABELS |= {f"lfcc{i}_mean": f"LFCC {i} mean" for i in range(20)}
LABELS |= {f"lfcc{i}_std": f"LFCC {i} variability" for i in range(20)}
LABELS |= {f"lfcc{i}_dstd": f"LFCC {i} frame-to-frame change" for i in range(20)}
LABELS |= {
    "gd_std_mean": "group-delay spread across harmonics",
    "gd_std_std": "variability of the group-delay spread",
    "gd_std_p10": "group-delay spread, quietest frames",
    "gd_std_p90": "group-delay spread, most spread frames",
    "gd_iqr_mean": "group-delay interquartile spread",
    "gd_iqr_std": "variability of the group-delay interquartile spread",
    "gd_iqr_p10": "group-delay interquartile spread (10th pct)",
    "gd_iqr_p90": "group-delay interquartile spread (90th pct)",
    "pc_err_median": "phase-vs-magnitude disagreement at harmonics (median, Hz)",
    "pc_err_mean": "phase-vs-magnitude disagreement at harmonics (mean, Hz)",
    "pc_err_p90": "phase-vs-magnitude disagreement at harmonics (90th pct, Hz)",
    "pc_frac_incoherent": "share of harmonics with incoherent phase",
}

__all__ = ["LABELS", "NAME", "HandcraftedDetector", "contributions", "latest_model_dir"]


def latest_model_dir(models: Path = MODELS) -> Path:
    return _latest(models, "hc")


class HandcraftedDetector(FeatureModelDetector):
    """Spectral + prosody features -> fold-validated classifier -> P(synthetic) + evidence."""

    name = NAME
    prefix = "hc"
    kind_label = "spectral/prosody"
    memo_key = "handcrafted.features"
    labels = LABELS
    feature_fn = staticmethod(features)


DETECTOR = HandcraftedDetector()
if NAME not in REGISTRY.names():
    register(DETECTOR)
