"""hearsay.trees: the numpy LightGBM evaluator reproduces lightgbm exactly (fixture dumped by a
lightgbm-only process) and its path attribution sums to the raw score."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hearsay.trees import Trees

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "lgbm_toy.json"


@pytest.fixture(scope="module")
def fx():
    return json.loads(FIXTURE.read_text())


def test_matches_lightgbm_probabilities_and_raw_scores(fx):
    t = Trees(fx["dump"])
    X = np.array(fx["X"], dtype=np.float32)
    assert t.n_features == len(fx["features"]) == X.shape[1]
    assert np.abs(t.predict_proba(X)[:, 1] - np.array(fx["expected_proba"])).max() < 1e-9
    assert np.abs(t.raw(X) - np.array(fx["expected_raw"])).max() < 1e-9
    p = t.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0) and p.shape == (len(X), 2)


def test_contributions_sum_to_the_raw_score(fx):
    t = Trees(fx["dump"])
    X = np.array(fx["X"])
    raw = t.raw(X)
    for i in range(len(X)):
        c, bias = t.contrib(X[i])
        assert c.shape == (t.n_features,) and abs(c.sum() + bias - raw[i]) < 1e-9


def test_contributions_agree_with_lightgbm_shap_on_the_top_features(fx):
    t = Trees(fx["dump"])
    for i, shap in enumerate(fx["expected_shap_first3"]):
        c, _ = t.contrib(np.array(fx["X"][i]))
        assert set(np.argsort(-np.abs(c))[:2]) & set(np.argsort(-np.abs(np.array(shap[:-1])))[:3])


def test_nan_follows_default_direction_and_shape_is_checked(fx):
    t = Trees(fx["dump"])
    x = np.array(fx["X"][0], dtype=float)
    x[:] = np.nan
    assert np.isfinite(t.raw(x[None, :])).all()
    c, bias = t.contrib(x)
    assert np.isfinite(c).all() and np.isfinite(bias)
    with pytest.raises(ValueError):
        t.raw(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        t.raw(np.zeros(t.n_features))


def test_categorical_splits_are_rejected():
    dump = {"max_feature_idx": 0, "tree_info": [{"tree_structure": {
        "split_feature": 0, "threshold": "1||2", "decision_type": "==", "internal_value": 0.0,
        "left_child": {"leaf_value": 1.0}, "right_child": {"leaf_value": -1.0}}}]}  # fmt: skip
    with pytest.raises(ValueError):
        Trees(dump)


def test_no_test_or_source_module_imports_lightgbm():
    repo = Path(__file__).resolve().parents[1]
    offenders = [p for p in list((repo / "src").rglob("*.py")) + list((repo / "tests").rglob("*.py"))
                 if "lightgbm" in p.read_text() and "import" in p.read_text()
                 and any(ln.strip().startswith(("import lightgbm", "from lightgbm"))
                         for ln in p.read_text().splitlines())]  # fmt: skip
    assert offenders == [], [str(p.relative_to(repo)) for p in offenders]
