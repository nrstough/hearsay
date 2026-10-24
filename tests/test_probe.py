"""hearsay.probe.load_embeddings: rows the extractor flagged `decode_error` (zeros in place of
the clip) are refused by default and load only on request. Hermetic: a temp embedding set laid
out as scripts/extract_embeddings.py writes it (fp16 shards out of row order plus manifest.csv);
no audio, weights or torch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hearsay import probe

MODEL, NAME = "toy-backbone", "toy_set"


def _make(root, flagged=(), n=7, shard=3, flag_col=True):
    out = root / "outputs" / "embeddings" / MODEL / NAME
    out.mkdir(parents=True)
    emb = np.arange(n * 2 * 3, dtype=np.float16).reshape(n, 2, 3)
    flag = ["decode_error" if r in flagged else "" for r in range(n)]
    for k, s0 in enumerate(reversed(range(0, n, shard))):  # shard files out of row order
        rows = np.arange(s0, min(s0 + shard, n))
        np.savez(out / f"shard_{k:05d}.npz", emb=emb[rows], row=rows,
                 n_windows=np.ones(len(rows), int), flag=np.array([flag[r] for r in rows]))  # fmt: skip
    man = pd.DataFrame({"path": [f"clip_{i}.wav" for i in range(n)], "label": "bonafide"})
    man = man.assign(n_windows=1, flag=flag) if flag_col else man
    man.to_csv(out / "manifest.csv", index=False)
    return emb.astype(np.float32)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "REPO", tmp_path)
    return tmp_path


def test_clean_set_loads_in_manifest_row_order(root):
    emb = _make(root)
    X, m = probe.load_embeddings(MODEL, NAME)
    assert X.dtype == np.float32 and np.array_equal(X, emb)
    assert list(m.path) == [f"clip_{i}.wav" for i in range(7)]


def test_manifest_without_flag_column_loads(root):
    emb = _make(root, flag_col=False)
    assert np.array_equal(probe.load_embeddings(MODEL, NAME)[0], emb)


def test_flagged_rows_are_refused_and_named(root):
    _make(root, flagged=(1, 5))
    with pytest.raises(ValueError) as e:
        probe.load_embeddings(MODEL, NAME)
    msg = str(e.value)
    assert "2 row(s) flagged decode_error" in msg
    assert "1 (clip_1.wav)" in msg and "5 (clip_5.wav)" in msg and "clip_0.wav" not in msg
    assert "repair_decode_errors.py" in msg and "allow_decode_errors=True" in msg


def test_long_flag_lists_are_truncated(root):
    _make(root, flagged=tuple(range(12)), n=13)
    with pytest.raises(ValueError, match=r"12 row\(s\).* and 2 more"):
        probe.load_embeddings(MODEL, NAME)


def test_allow_decode_errors_loads_the_flagged_set_unchanged(root):
    emb = _make(root, flagged=(1, 5))
    X, m = probe.load_embeddings(MODEL, NAME, allow_decode_errors=True)
    assert np.array_equal(X, emb)
    assert list(m.flag.fillna("")) == ["", "decode_error", "", "", "", "decode_error", ""]
