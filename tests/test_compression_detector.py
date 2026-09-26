"""Compression forensics: feature sanity, laundering, and the detector against the contract
(tiny bundle built here, so no gitignored models/ or data/ are needed)."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hearsay import SR
from hearsay.compression import (
    draw_laundering,
    features,
    launder,
    parse_laundering,
)
from hearsay.detectors import base
from hearsay.detectors.base import ClipContext, safe_run
from hearsay.detectors.compression import LABELS, CompressionDetector, latest_model_dir


def _signal(kind: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(3 * SR) / SR
    if kind == "noise":  # broadband, synthetic-marked
        return (rng.standard_normal(t.size) * (0.05 + 0.1 * rng.random())).astype(np.float32)
    f0 = 120 + 120 * rng.random()  # harmonic tone, real-marked
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return (0.05 * x + 0.002 * rng.standard_normal(t.size)).astype(np.float32)


def _clip(kind: str, seed: int = 0) -> ClipContext:
    return ClipContext.from_array(_signal(kind, seed))


def has_encoder(name: str) -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True,
                         check=False)  # fmt: skip
    return name in out.stdout


# --- features ------------------------------------------------------------------------------


def test_features_are_finite_floats_and_deterministic():
    f1, f2 = features(_signal("noise")), features(_signal("noise"))
    assert f1 == f2 and len(f1) == len(LABELS) and set(f1) == set(LABELS)
    assert all(type(v) is float and math.isfinite(v) for v in f1.values())
    assert 3000.0 <= f1["bw_hz"] <= 8000.0 and 0.0 <= f1["local_hole_frac"] <= 1.0


def test_band_match_moves_effective_bandwidth_below_8k():
    x = _signal("noise")
    assert features(x, band_match=False)["bw_hz"] == 8000.0
    assert features(x, band_match=True)["bw_hz"] <= 7500.0


@pytest.mark.skipif(not has_encoder("libmp3lame"), reason="ffmpeg lacks libmp3lame")
def test_laundering_leaves_codec_traces():
    x = _signal("tone", 1)
    y = launder(x, "mp3", 32, 44100)
    assert abs(y.size - x.size) < SR // 4  # encoder delay/padding only
    clean, dirty = features(x, band_match=False), features(y, band_match=False)
    assert dirty["local_hole_frac"] > clean["local_hole_frac"]
    assert dirty["floor_p2_db"] < clean["floor_p2_db"]


def test_launder_rejects_unknown_codec():
    with pytest.raises(ValueError):
        launder(_signal("tone"), "flac", 64, 16000)


def test_laundering_draws_are_reproducible_and_parse():
    draws = [draw_laundering(np.random.default_rng(r), 0.5) for r in range(20)]
    assert draws == [draw_laundering(np.random.default_rng(r), 0.5) for r in range(20)]
    assert any(draws) and not all(draws)
    for d in draws:
        spec = parse_laundering(d)
        assert (spec is None) == (d == "")
        if spec:
            codec, kbps, sr = spec
            assert codec in ("mp3", "aac") and kbps > 0 and sr in (16000, 22050, 44100)
    assert all(draw_laundering(np.random.default_rng(r), 0.0) == "" for r in range(5))


# --- detector ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def det(tmp_path_factory):
    rows, y = [], []
    for i in range(12):
        rows.append(features(_signal("noise", i))); y.append(1)
        rows.append(features(_signal("tone", i))); y.append(0)
    cols = list(rows[0])
    X, y = np.array([[r[c] for c in cols] for r in rows], np.float32), np.array(y)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    stats = {c: {"real_mean": float(X[y == 0, i].mean()), "real_std": float(X[y == 0, i].std()),
                 "auc": 0.5} for i, c in enumerate(cols)}  # fmt: skip
    d = tmp_path_factory.mktemp("models") / "cmp_logreg_toy"
    d.mkdir()
    joblib.dump({"model": model, "features": cols, "kind": "logreg", "feature_stats": stats,
                 "crop_mode": "segment", "band_match": True, "train": "toy", "folds": "toy"},
                d / "model.joblib")  # fmt: skip
    return CompressionDetector(model_dir=d)


def test_contract_result_is_valid(det):
    r = safe_run(det, _clip("noise"))
    assert r.status == "ok" and r.error is None and r.name == "compression"
    assert 0.0 <= r.score <= 1.0 and "cmp_logit" in r.features
    assert all(type(v) is float and math.isfinite(v) for v in r.features.values())


def test_direction_follows_the_model(det):
    assert safe_run(det, _clip("noise", 5)).score > safe_run(det, _clip("tone", 5)).score


def test_evidence_names_traces(det):
    r = safe_run(det, _clip("noise"))
    assert "compression-trace" in r.evidence and "SD" in r.evidence
    assert any(lbl in r.evidence for lbl in LABELS.values())


def test_deterministic_and_memoized(det):
    c = _clip("tone", 2)
    assert safe_run(det, c) == safe_run(det, c) and "compression.features" in c.cache


def test_missing_model_is_an_error_result(tmp_path):
    (tmp_path / "cmp_none").mkdir()
    r = safe_run(CompressionDetector(model_dir=tmp_path / "cmp_none"), _clip("noise"))
    assert r.status == "error" and r.score == 0.5
    with pytest.raises(FileNotFoundError):
        latest_model_dir(tmp_path)
    (tmp_path / "hc_logreg_20260926-0100").mkdir()  # another prefix is not ours
    (tmp_path / "hc_logreg_20260926-0100" / "model.joblib").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        latest_model_dir(tmp_path)


def test_registered_under_its_name():
    assert "compression" in base.REGISTRY.names()


def test_scripts_exist():
    repo = Path(__file__).resolve().parents[1]
    assert (repo / "scripts" / "extract_compression.py").exists()
