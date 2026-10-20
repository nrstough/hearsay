"""scripts/v2_backbone_to_benchmark.py: V2-53 exports -> one Interface v1 score file. Hermetic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from hearsay import benchmark as B

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v2_backbone_to_benchmark", REPO / "scripts/v2_backbone_to_benchmark.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def _write(lane: Path, model: str, pop: str, n: int, split: str, *, fold="0", flag="", dup=False):
    df = pd.DataFrame({"path": [f"data/x/{pop}_{i}.wav" for i in range(n)], "fold": fold, "split": split,
                       "score": 0.5, "logit": 0.0, "flag": flag})
    if dup:
        df.loc[1, "path"] = df.loc[0, "path"]
    df.to_csv(lane / f"{model}__{pop}.csv", index=False)


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(A.B, "N_INNER", 10)
    monkeypatch.setattr(A.B, "N_HOLDOUT", 4)
    monkeypatch.setattr(A.B, "N_TEST", 3)
    monkeypatch.setattr(A.B, "N_ITW", 5)
    monkeypatch.setattr(A, "EXPECT", {"inner": (10, "inner_oof"), "holdout": (4, "holdout"),
                                      "test": (3, "test"), "itw": (5, "itw")})


def _lane(tmp_path, **kw):
    lane = tmp_path / "lane"; lane.mkdir()
    _write(lane, "m", "inner", 10, "inner_oof", **kw.get("inner", {}))
    _write(lane, "m", "holdout", 4, "holdout", fold="holdout", **kw.get("holdout", {}))
    _write(lane, "m", "test", 3, "test", fold="", **kw.get("test", {}))
    _write(lane, "m", "itw", 5, "itw", fold="", **kw.get("itw", {}))
    return lane


def test_assembles_main_and_itw_with_asserted_counts(tmp_path, small):
    lane = _lane(tmp_path)
    f_main, f_itw = A.assemble("m", lane=lane, out=tmp_path / "out")
    main = pd.read_csv(f_main, dtype=str)
    assert list(main.columns) == ["path", "fold", "split", "score", "logit"]
    assert main["split"].value_counts().to_dict() == {"inner_oof": 10, "holdout": 4, "test": 3}
    itw = pd.read_csv(f_itw)
    assert list(itw.columns) == ["path", "logit", "split"] and len(itw) == 5
    # the output is readable by the benchmark reader (no repo canonicalisation needed here: relative paths)
    B.read_itw_override(f_itw, main=tmp_path, repo=tmp_path)


@pytest.mark.parametrize("bad, match", [
    ({"inner": {"flag": "decode_error"}}, "flagged rows"),
    ({"holdout": {"dup": True}}, "duplicate"),
    ({"inner": {"fold": "holdout"}}, "inner folds"),
])
def test_refuses_flagged_duplicate_or_misfolded(tmp_path, small, bad, match):
    lane = _lane(tmp_path, **bad)
    with pytest.raises(B.BenchmarkError, match=match):
        A.assemble("m", lane=lane, out=tmp_path / "out")


def test_refuses_wrong_count_and_missing_part(tmp_path, small):
    lane = _lane(tmp_path)
    _write(lane, "m", "test", 2, "test", fold="")
    with pytest.raises(B.BenchmarkError, match="2 rows, expected 3"):
        A.assemble("m", lane=lane, out=tmp_path / "out")
    (lane / "m__itw.csv").unlink()
    _write(lane, "m", "test", 3, "test", fold="")
    with pytest.raises(B.BenchmarkError, match="missing"):
        A.assemble("m", lane=lane, out=tmp_path / "out")
