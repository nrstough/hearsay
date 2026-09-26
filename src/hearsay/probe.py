"""M1 probe: standardized logistic regression on one layer of frozen SSL embeddings.

Layer choice and calibration use generator-grouped CV on the training set only; the held-out
validation split is scored once, at the end, and never fit on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hearsay.metrics import decision_logit, sigmoid

REPO = Path(__file__).resolve().parents[2]


def load_embeddings(model: str, name: str) -> tuple[np.ndarray, pd.DataFrame]:
    """(N, n_layers, dim) float32 in manifest row order, plus the manifest."""
    d = REPO / "outputs" / "embeddings" / model / name
    m = pd.read_csv(d / "manifest.csv")
    shards = sorted(d.glob("shard_*.npz"))
    zs = [np.load(p) for p in shards]
    rows = np.concatenate([z["row"] for z in zs])
    emb = np.concatenate([z["emb"] for z in zs]).astype(np.float32)
    assert len(rows) == len(m) and np.array_equal(np.sort(rows), np.arange(len(m)))
    return emb[np.argsort(rows)], m


def cv_groups(m: pd.DataFrame) -> np.ndarray:
    """Spoof grouped by generator, bona fide by speaker: folds hold out whole generators."""
    return np.where(m.label == "spoof", "gen_" + m.generator.astype(str), "spk_" + m.speaker)


def make_clf(c: float = 1.0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight="balanced", max_iter=3000),
    )


def oof_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float, k: int = 5):
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=0).split(X, y, groups):
        oof[te] = make_clf(c).fit(X[tr], y[tr]).decision_function(X[te])
    return oof


@dataclass
class Probe:
    backbone: str
    layer: int
    clf: object
    platt_a: float
    platt_b: float
    win_s: float = 4.0
    max_windows: int = 4

    def llr(self, emb: np.ndarray) -> np.ndarray:
        """emb: (N, n_layers, dim) or (n_layers, dim) -> Platt-calibrated, prior-neutral LLR
        (class-balanced fit), increasing with synthetic likelihood."""
        emb = emb[None] if emb.ndim == 2 else emb
        z = self.clf.decision_function(emb[:, self.layer, :])
        return self.platt_a * z + self.platt_b

    def p_synthetic(self, emb: np.ndarray) -> np.ndarray:
        """Prior-neutral calibrated P(synthetic)."""
        return sigmoid(self.llr(emb))

    def p_decision(self, emb: np.ndarray, pi_synth: float) -> np.ndarray:
        """Cost-shifted output: > 0.5 exactly when the 4:1 Bayes decision says synthetic."""
        return sigmoid(decision_logit(self.llr(emb), pi_synth))

    def save(self, path: Path, meta: dict) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "probe.joblib")
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def load(path: Path) -> Probe:
        return joblib.load(Path(path) / "probe.joblib")
