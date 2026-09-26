"""Forensic detectors and the contract they share (see `hearsay.detectors.base`)."""

from hearsay.detectors.base import (
    REGISTRY,
    ClipContext,
    Detector,
    DetectorResult,
    Registry,
    all_detectors,
    get,
    register,
    safe_run,
)

__all__ = [
    "REGISTRY",
    "ClipContext",
    "Detector",
    "DetectorResult",
    "Registry",
    "all_detectors",
    "get",
    "register",
    "safe_run",
]
