"""Handcrafted detector against the contract (tests/test_detector_contract.py B1-B9), using
tiny model bundles: a logistic regression fit here, and a LightGBM toy read from
tests/fixtures/lgbm_toy.json (dumped once by a lightgbm-only process; this suite must never
load lightgbm next to torch, see hearsay.trees). No gitignored models/ or data/ needed."""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hearsay import SR
from hearsay.detectors import base
from hearsay.detectors.base import ClipContext, safe_run
from hearsay.detectors.handcrafted import LABELS, HandcraftedDetector, latest_model_dir
from hearsay.handcrafted import features

REPO = Path(__file__).resolve().parents[1]


def _clip(kind: str, seed: int = 0) -> ClipContext:
    rng = np.random.default_rng(seed)
    t = np.arange(2 * SR) / SR
    if kind == "noise":  # broadband, synthetic-marked: high energy share above 4 kHz
        x = rng.standard_normal(t.size) * (0.05 + 0.1 * rng.random())
    else:  # low tone, real-marked: almost no energy above 4 kHz
        f0 = 120 + 120 * rng.random()
        x = 0.1 * np.sin(2 * np.pi * f0 * t) + 0.002 * rng.standard_normal(t.size)
    return ClipContext.from_array(x.astype(np.float32))


_TOY: dict = {}


def _toy_data():
    """Features of 16 noise (synthetic-marked) and 16 tone (real-marked) clips, computed once."""
    if not _TOY:
        rows, y = [], []
        for i in range(16):
            rows.append(features(_clip("noise", seed=i).audio)); y.append(1)
            rows.append(features(_clip("tone", seed=i).audio)); y.append(0)
        cols = list(rows[0])
        _TOY.update(cols=cols, X=np.array([[r[c] for c in cols] for r in rows], np.float32),
                    y=np.array(y))
    return _TOY["cols"], _TOY["X"], _TOY["y"]


def _make_bundle(out: Path, kind: str) -> Path:
    d = out / f"hc_{kind}"
    d.mkdir()
    common = {"kind": kind, "crop_mode": "segment", "band_match": True, "train": "toy",
              "folds": "toy"}  # fmt: skip
    if kind == "logreg":
        cols, X, y = _toy_data()
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        model.fit(X, y)
        stats = {c: {"real_mean": float(X[y == 0, i].mean()),
                     "real_std": float(X[y == 0, i].std()), "auc": 0.5}
                 for i, c in enumerate(cols)}  # fmt: skip
        bundle = {"model": model, "features": cols, "feature_stats": stats, **common}
    else:
        fx = json.loads((REPO / "tests" / "fixtures" / "lgbm_toy.json").read_text())
        bundle = {"trees_dump": fx["dump"], "features": fx["features"],
                  "feature_stats": fx["feature_stats"], **common}  # fmt: skip
    joblib.dump(bundle, d / "model.joblib")
    return d


@pytest.fixture(scope="module", params=["logreg", "lgbm"])
def det(request, tmp_path_factory):
    d = _make_bundle(tmp_path_factory.mktemp("models"), request.param)
    return HandcraftedDetector(model_dir=d)


def test_contract_result_is_valid(det):
    r = safe_run(det, _clip("noise"))
    assert r.status == "ok" and r.error is None and r.name == "handcrafted"
    assert 0.0 <= r.score <= 1.0
    assert len(r.features) == len(det.bundle["features"]) + 1 and "hc_logit" in r.features
    assert all(type(v) is float and math.isfinite(v) for v in r.features.values())
    assert math.isclose(r.features["hc_logit"], math.log(r.score / (1 - r.score)), rel_tol=1e-6)


def test_direction_follows_the_model(det):
    """B2: the toy models were fit with noise = synthetic, so noise must outscore a tone."""
    assert safe_run(det, _clip("noise")).score > safe_run(det, _clip("tone")).score


def test_evidence_names_features_and_direction(det):
    r = safe_run(det, _clip("noise"))
    assert "SD" in r.evidence and ("above" in r.evidence or "below" in r.evidence)
    assert any(lbl in r.evidence for lbl in LABELS.values())
    assert "toward synthetic" in r.evidence or "toward real" in r.evidence


def test_deterministic_and_memoized(det):
    c = _clip("noise", seed=3)
    r1, r2 = safe_run(det, c), safe_run(det, c)
    assert r1 == r2 and "handcrafted.features" in c.cache


def test_missing_model_is_an_error_result_not_a_crash(tmp_path):
    empty = tmp_path / "hc_missing"
    empty.mkdir()
    r = safe_run(HandcraftedDetector(model_dir=empty), _clip("noise"))
    assert r.status == "error" and r.score == 0.5 and "model.joblib" in r.error
    with pytest.raises(FileNotFoundError):
        latest_model_dir(tmp_path)


def test_latest_model_dir_picks_the_newest_stamp(tmp_path):
    for name in ("hc_logreg_20260926-0112", "hc_lgbm_20260926-0300", "hc_logreg_20260926-0118"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "model.joblib").write_bytes(b"")
    (tmp_path / "hc_logreg_20260927-0000").mkdir()  # no model.joblib: ignored
    assert latest_model_dir(tmp_path).name == "hc_lgbm_20260926-0300"


def test_registered_under_its_name():
    assert "handcrafted" in base.REGISTRY.names()
    assert base.get("handcrafted").name == "handcrafted"


@pytest.mark.needs_data
@pytest.mark.slow
def test_real_model_on_a_test_file():
    wavs = sorted((REPO / "data" / "nsa" / "HackGTHearsayTesting").glob("*.wav"))
    if not wavs or not (REPO / "models").exists() or not list((REPO / "models").glob("hc_*")):
        pytest.skip("needs the NSA test set and a trained models/hc_* bundle")
    r = safe_run(HandcraftedDetector(), ClipContext(wavs[0]))
    assert r.status == "ok" and 0.0 <= r.score <= 1.0 and r.evidence


def test_bundle_families_are_recomputed_at_inference(tmp_path):
    """A v4 bundle names its families; the detector must compute them for new audio."""
    fams = ("modulation", "breath")  # the cheap ones
    rows, y = [], []
    for i in range(6):
        rows.append(features(_clip("noise", i).audio, families=fams)); y.append(1)
        rows.append(features(_clip("tone", i).audio, families=fams)); y.append(0)
    cols = list(rows[0])
    assert any(c.startswith("mod_") for c in cols) and any(c.startswith("breath_") for c in cols)
    X = np.array([[r[c] for c in cols] for r in rows], np.float32)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)
    stats = {c: {"real_mean": 0.0, "real_std": 1.0, "auc": 0.5} for c in cols}
    d = tmp_path / "hc_logreg_v4toy"
    d.mkdir()
    joblib.dump({"model": model, "features": cols, "kind": "logreg", "feature_stats": stats,
                 "crop_mode": "segment", "band_match": True, "families": list(fams)},
                d / "model.joblib")  # fmt: skip
    r = safe_run(HandcraftedDetector(model_dir=d), _clip("noise", 9))
    assert r.status == "ok", r.error
    assert "mod_peak_hz" in r.features and "breath_frac" in r.features
    assert len(r.features) == len(cols) + 1  # + hc_logit
