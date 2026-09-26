"""Shared wrapper for the learned engineered detectors (handcrafted spectral + prosody,
compression forensics): a feature function, a fold-validated classifier bundle saved by
scripts/train_handcrafted.py, P(synthetic) as the score, and evidence naming the top
contributing features in plain English with their distance from real speech, e.g. "pitch
variability (IQR of log F0) 2.1 SD below real speech (toward synthetic)".

Bundle (models/<prefix>_<kind>_<stamp>/model.joblib, newest stamp by default): features
(column order), kind ("logreg" | "lgbm"), feature_stats ({name: {real_mean, real_std, auc}} on
the inner bona fide rows), crop_mode, band_match, and either `model` (a scikit-learn pipeline,
logreg) or `trees_dump` (a LightGBM booster dump run by `hearsay.trees`, so lightgbm is never
imported at inference; see that module for why). Loaded lazily on first run. Test-time audio
is prepared the way the training features were (never a random crop at test time).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import numpy as np

from hearsay.detectors.base import ClipContext, DetectorResult
from hearsay.trees import Trees

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "models"
TOP_K = 3


def latest_model_dir(models: Path = MODELS, prefix: str = "hc") -> Path:
    """`models/<prefix>_selected` (a symlink or directory holding model.joblib) wins when it
    exists: the chosen bundle, not merely the newest. Otherwise the newest stamp."""
    selected = models / f"{prefix}_selected"
    if (selected / "model.joblib").exists():
        return selected
    dirs = sorted((p for p in models.glob(f"{prefix}_*") if (p / "model.joblib").exists()
                   and p.name != selected.name),
                  key=lambda p: (p.name.rsplit("_", 1)[-1], p.name))  # by <stamp>, then name
    if not dirs:
        raise FileNotFoundError(
            f"no {prefix}_* model under {models} (run scripts/train_handcrafted.py)"
        )
    return dirs[-1]


def predictor(bundle: dict):
    """The object with `predict_proba` for this bundle (built once, cached on the bundle)."""
    if "_predictor" not in bundle:
        if bundle["kind"] == "lgbm":
            bundle["_predictor"] = Trees(bundle["trees_dump"])
        else:
            bundle["_predictor"] = bundle["model"]
    return bundle["_predictor"]


def contributions(bundle: dict, x: np.ndarray) -> np.ndarray:
    """Per-feature contribution to the decision logit for one raw feature row."""
    if bundle["kind"] == "logreg":
        model = bundle["model"]
        scaler = model.named_steps["standardscaler"]
        clf = model.named_steps["logisticregression"]
        return scaler.transform(x[None, :])[0] * clf.coef_[0]
    return predictor(bundle).contrib(x)[0]  # path attribution over the dumped trees


class FeatureModelDetector:
    """Subclasses set name, prefix, kind_label, memo_key, labels and feature_fn."""

    name: str
    prefix: str
    kind_label: str
    memo_key: str
    labels: ClassVar[dict[str, str]] = {}
    feature_fn: Callable[..., dict[str, float]]

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self._bundle: dict | None = None

    @property
    def bundle(self) -> dict:
        if self._bundle is None:
            import joblib

            d = self.model_dir or latest_model_dir(MODELS, self.prefix)
            self._bundle = joblib.load(d / "model.joblib")
            self.model_dir = d
        return self._bundle

    def applies(self, ctx: ClipContext) -> bool:
        return True  # any clip; a decode failure surfaces as an error result via safe_run

    def run(self, ctx: ClipContext) -> DetectorResult:
        b = self.bundle
        fn = type(self).feature_fn
        kw: dict = {"crop_mode": b["crop_mode"], "band_match": b.get("band_match", False)}
        if b.get("families"):
            kw["families"] = tuple(b["families"])  # v4 families (handcrafted only)
        feats = ctx.memo(self.memo_key, lambda: fn(ctx.audio, **kw))
        x = np.array([feats[c] for c in b["features"]], dtype=np.float32)
        p = float(np.clip(predictor(b).predict_proba(x[None, :])[0, 1], 1e-6, 1 - 1e-6))
        logit = float(np.log(p / (1 - p)))
        out = {c: float(feats[c]) for c in b["features"]}
        out[f"{self.prefix}_logit"] = logit
        return DetectorResult(self.name, p, self._evidence(b, feats, contributions(b, x), p), out)

    def _evidence(self, b: dict, feats: dict, contrib: np.ndarray, p: float) -> str:
        parts = []
        for i in np.argsort(-np.abs(contrib))[:TOP_K]:
            c = b["features"][i]
            st = b["feature_stats"][c]
            z = (feats[c] - st["real_mean"]) / (st["real_std"] + 1e-12)
            parts.append(
                f"{self.labels.get(c, c)} {abs(z):.1f} SD {'above' if z >= 0 else 'below'} real "
                f"speech ({'toward synthetic' if contrib[i] > 0 else 'toward real'})"
            )
        verdict = "synthetic-like" if p >= 0.5 else "real-like"
        return f"{self.kind_label} features {verdict} (P={p:.2f}): " + "; ".join(parts)
