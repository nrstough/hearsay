"""Detector contract: what every forensic detector returns (NSA brief, docs/nsa-challenge.md).

The brief scores "each distinct technique your system actually leverages" and asks for
explainable results, so every detector (deep SSL, metadata, spectral, prosody, ...) returns a
`DetectorResult`: a score, a one-sentence reason, and optional named numbers.

Rules (validated in `DetectorResult.__post_init__`; violations raise ValueError, never clip):
- `score` is a finite real in [0, 1] and **increases with synthetic likelihood**.
- `evidence` is a non-empty, human-readable sentence.
- `features` is a flat {str: finite float} dict, for explanations and logs.
- `status` is "ok", "skipped" (detector does not apply), or "error"; `error` is set only for
  "error".

Fusion input convention (docs/plan.md, Architecture): fusion v1 uses one column per detector,
`<name>.logit = logit(clip(score, 1e-4, 1 - 1e-4))` when status == "ok"; otherwise the value
is missing (imputed with the fold-local train mean) plus a `<name>.missing` indicator. The 0.5
placeholder on skipped/error results is never fed to fusion as a score.

Run detectors through `safe_run`, which never raises `Exception`: a crashing or misbehaving
detector becomes an "error" result so no file ever loses its row. Audio is 16 kHz mono
float32 from `hearsay.audio.load_audio`, decoded at most once per clip and read-only.
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from hearsay import audio

Status = Literal["ok", "skipped", "error"]
STATUSES = ("ok", "skipped", "error")
NEUTRAL_SCORE = 0.5


@dataclass
class ClipContext:
    """One clip as detectors see it: path, lazily decoded audio, probe metadata, shared cache."""

    path: Path
    cache: dict[str, Any] = field(default_factory=dict)
    _audio: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _audio_err: audio.DecodeError | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _probe: dict | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_array(
        cls, x: np.ndarray, probe: dict | None = None, path: str | Path = "<array>"
    ) -> ClipContext:
        """Build a clip from 1-D 16 kHz mono float samples without ffmpeg (tests, synthetic
        inputs). Integer PCM is not rescaled: pass floats in [-1, 1]."""
        ctx = cls(Path(path))
        arr = np.array(x, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"expected 1-D mono samples, got shape {arr.shape}")
        arr.flags.writeable = False
        ctx._audio = arr
        ctx._probe = dict(probe) if probe is not None else {}
        return ctx

    @property
    def audio(self) -> np.ndarray:
        """16 kHz mono float32, decoded once; read-only. A decode failure is cached and
        re-raised (no second ffmpeg call). Not locked: decode before fanning out to threads."""
        if self._audio is None and self._audio_err is None:
            try:
                x = audio.load_audio(self.path)
            except audio.DecodeError as e:
                self._audio_err = e
            else:
                x.flags.writeable = False
                self._audio = x
        if self._audio_err is not None:
            raise self._audio_err
        return self._audio

    @property
    def probe(self) -> dict:
        """`hearsay.audio.probe_audio` facts, computed once (never raises)."""
        if self._probe is None:
            self._probe = audio.probe_audio(self.path)
        return self._probe

    def memo(self, key: str, fn: Callable[[], Any]) -> Any:
        """Per-clip shared cache for expensive intermediates (e.g. SSL windows)."""
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]


def _check_real(value: Any, what: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{what} must be a real number, got {type(value).__name__}")  # noqa: TRY004 - contract (D3) mandates ValueError
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{what} must be finite, got {v}")
    return v


@dataclass(frozen=True)
class DetectorResult:
    """One detector's verdict on one clip. See the module docstring for the rules."""

    name: str
    score: float
    evidence: str
    features: dict[str, float] = field(default_factory=dict)
    status: Status = "ok"
    error: str | None = None

    __hash__ = None  # type: ignore[assignment]  # results hold a dict; deliberately unhashable

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty str, got {self.name!r}")
        score = _check_real(self.score, "score")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")
        if not isinstance(self.evidence, str) or not self.evidence.strip():
            raise ValueError("evidence must be a non-empty str")
        if not isinstance(self.features, Mapping):
            raise ValueError(f"features must be a mapping, got {type(self.features).__name__}")  # noqa: TRY004 - contract (D3) mandates ValueError
        feats: dict[str, float] = {}
        for k, v in self.features.items():
            if not isinstance(k, str) or not k:
                raise ValueError(f"feature keys must be non-empty str, got {k!r}")
            feats[k] = _check_real(v, f"feature {k!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.status == "error":
            if not isinstance(self.error, str) or not self.error:
                raise ValueError("status 'error' requires a non-empty error message")
        elif self.error is not None:
            raise ValueError(f"status {self.status!r} must have error=None")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "features", feats)


@runtime_checkable
class Detector(Protocol):
    """A forensic technique. Load heavy models lazily inside `run`, not at import."""

    name: str

    def applies(self, ctx: ClipContext) -> bool: ...

    def run(self, ctx: ClipContext) -> DetectorResult: ...


def _error(name: str, msg: str) -> DetectorResult:
    return DetectorResult(name, NEUTRAL_SCORE, f"detector failed: {msg}", status="error", error=msg)


def safe_run(det: Any, ctx: ClipContext) -> DetectorResult:
    """Run one detector; never raises `Exception` (KeyboardInterrupt/SystemExit propagate).

    Skipped if `applies` is false (`run` is not called). Any exception, a non-DetectorResult
    return, a name mismatch, a non-"ok" status from `run`, or a result that no longer validates
    (e.g. features mutated after construction) becomes status "error" with score 0.5.
    """
    name = "<" + type(det).__name__ + ">"
    try:
        det_name = det.name
        if isinstance(det_name, str) and det_name.strip():
            name = det_name
        else:
            raise ValueError(f"detector name must be a non-empty str, got {det_name!r}")
        if not det.applies(ctx):
            return DetectorResult(name, NEUTRAL_SCORE, f"not applicable: {name}", status="skipped")
        res = det.run(ctx)
        if not isinstance(res, DetectorResult):
            raise TypeError(f"run() returned {type(res).__name__}, not DetectorResult")
        if res.name != name:
            raise ValueError(f"run() returned name {res.name!r}, expected {name!r}")
        if res.status != "ok":
            raise ValueError(f"run returned status={res.status}")
        return dataclasses.replace(res)  # re-validates, copies features
    except Exception as e:  # noqa: BLE001 - contract boundary: one bad detector must not lose a row
        try:
            msg = f"{type(e).__name__}: {e}"
        except Exception:  # noqa: BLE001 - an exception whose __str__ raises must not escape
            msg = type(e).__name__
        return _error(name, msg)


class Registry:
    """Name -> detector. Names are unique; iteration is sorted by name (stable fusion columns)."""

    def __init__(self) -> None:
        self._by_name: dict[str, Detector] = {}

    def register(self, det: Detector) -> Detector:
        name = getattr(det, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"detector name must be a non-empty str, got {name!r}")
        if name in self._by_name:
            raise ValueError(f"detector {name!r} is already registered")
        self._by_name[name] = det
        return det

    def get(self, name: str) -> Detector:
        return self._by_name[name]

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def all_detectors(self) -> list[Detector]:
        return [self._by_name[n] for n in self.names()]


REGISTRY = Registry()
register = REGISTRY.register
get = REGISTRY.get
all_detectors = REGISTRY.all_detectors
