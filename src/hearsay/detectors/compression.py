"""Compression forensics detector (brief rubric, technique 5): the contract wrapper around
`hearsay.compression.features` and the model saved by scripts/train_handcrafted.py with
--features-dir outputs/compression --model-prefix cmp.

The features read codec history out of the signal (spectral holes and their flicker, floor
depth, effective bandwidth, high-band tilt), because every NSA test file is PCM WAV and the
container says nothing. The classifier is trained with a random half of both classes
laundered through MP3/AAC, so it cannot key on "lossy = one particular generator"; what it
learns is what survives that equalization. Score = P(synthetic); features are the raw values
plus `cmp_logit`; evidence names the top contributing traces. Bundle:
models/cmp_<kind>_<stamp>/model.joblib, newest stamp by default.
"""

from __future__ import annotations

from pathlib import Path

from hearsay.compression import features
from hearsay.detectors._learned import MODELS, FeatureModelDetector, contributions
from hearsay.detectors._learned import latest_model_dir as _latest
from hearsay.detectors.base import REGISTRY, register

NAME = "compression"
LABELS = {
    "bw_hz": "effective bandwidth",
    "band_4_5k_db": "4-5 kHz level vs 1-3 kHz",
    "band_5_6k_db": "5-6 kHz level vs 1-3 kHz",
    "band_6_7k_db": "6-7 kHz level vs 1-3 kHz",
    "band_7_8k_db": "7-8 kHz level vs 1-3 kHz",
    "hf_slope_db_per_khz": "high-band spectral tilt",
    "deep_hole_frac": "deep spectral holes (3-7 kHz)",
    "deep_hole_flicker": "deep-hole flicker frame to frame",
    "deep_hole_persistence": "deep-hole persistence frame to frame",
    "deep_hole_runs": "deep-hole runs per frame",
    "local_hole_frac": "local spectral holes (3-7 kHz)",
    "local_hole_flicker": "local-hole flicker frame to frame",
    "local_hole_persistence": "local-hole persistence frame to frame",
    "local_hole_runs": "local-hole runs per frame",
    "floor_p2_db": "3-7 kHz floor depth (2nd percentile)",
    "floor_p10_db": "3-7 kHz floor depth (10th percentile)",
    "valley_depth_db": "3-7 kHz valley depth",
    "floor_p2_lowband_db": "0.3-3 kHz floor depth",
    "clip_floor_db": "whole-clip floor depth",
}

__all__ = ["LABELS", "NAME", "CompressionDetector", "contributions", "latest_model_dir"]


def latest_model_dir(models: Path = MODELS) -> Path:
    return _latest(models, "cmp")


class CompressionDetector(FeatureModelDetector):
    """Codec-history traces -> fold-validated classifier -> P(synthetic) + evidence."""

    name = NAME
    prefix = "cmp"
    kind_label = "compression-trace"
    memo_key = "compression.features"
    labels = LABELS
    feature_fn = staticmethod(features)


DETECTOR = CompressionDetector()
if NAME not in REGISTRY.names():
    register(DETECTOR)
