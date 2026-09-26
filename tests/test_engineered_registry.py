"""All five engineered detectors register under their names and satisfy the Detector protocol."""

from __future__ import annotations

import numpy as np

from hearsay import SR
from hearsay.detectors import Detector, all_detectors, base
from hearsay.detectors.base import ClipContext, safe_run
from hearsay.detectors.engineered import LEARNED, NAMES, RULE_BASED


def test_all_five_registered_and_sorted():
    assert NAMES == ("compression", "container", "enf", "handcrafted", "splice")
    assert set(NAMES) <= set(base.REGISTRY.names())
    assert [d.name for d in all_detectors() if d.name in NAMES] == list(NAMES)
    assert all(isinstance(base.get(n), Detector) for n in NAMES)


def test_rule_based_detectors_run_without_any_model():
    x = (0.1 * np.random.default_rng(0).standard_normal(2 * SR)).astype(np.float32)
    for name in RULE_BASED:
        if name == "container":
            continue  # needs a real file (ffprobe); covered in tests/test_container_detector.py
        r = safe_run(base.get(name), ClipContext.from_array(x))
        assert r.status == "ok" and r.name == name and 0.0 <= r.score <= 1.0


def test_learned_detectors_without_a_bundle_become_error_results(tmp_path):
    """No models/ directory in CI: the contract still holds (error result, row kept)."""
    x = (0.1 * np.random.default_rng(0).standard_normal(2 * SR)).astype(np.float32)
    for name in LEARNED:
        det = type(base.get(name))(model_dir=tmp_path / "missing")
        r = safe_run(det, ClipContext.from_array(x))
        assert r.status == "error" and r.score == 0.5 and r.name == name
