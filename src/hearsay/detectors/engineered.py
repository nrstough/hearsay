"""Import this module to register every engineered (D-track) detector in the shared registry:

    import hearsay.detectors.engineered  # noqa: F401
    from hearsay.detectors import all_detectors
    all_detectors()  # -> [compression, container, enf, handcrafted, splice], sorted by name

Learned (need a models/ bundle): handcrafted (spectral + prosody), compression (codec-history
traces). Rule-based (no bundle): container (header facts + tag rules), enf (mains hum), splice
(clicks and DC jumps). Each is CPU-only and returns a `DetectorResult` per the contract in
`hearsay.detectors.base`; run them through `safe_run`. Score exports for fusion live in
outputs/detector_scores/<name>.csv (see scripts/train_handcrafted.py, scripts/score_detector.py).
"""

from __future__ import annotations

from hearsay.detectors import compression, container, enf, handcrafted, splice

LEARNED = (handcrafted.NAME, compression.NAME)
RULE_BASED = (container.NAME, enf.NAME, splice.NAME)
NAMES = tuple(sorted(LEARNED + RULE_BASED))

__all__ = ["LEARNED", "NAMES", "RULE_BASED", "compression", "container", "enf", "handcrafted",
           "splice"]  # fmt: skip
