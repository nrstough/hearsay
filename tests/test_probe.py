"""hearsay.probe.load_embeddings: rows the extractor flagged `decode_error` (zeros in place of
the clip) are refused by default and load only on request, and the CLIs that call it
(train_probe, export_probe_scores, tsv_from_embeddings) pass that request on only with
--allow-decode-errors. Hermetic: a temp embedding set laid out as scripts/extract_embeddings.py
writes it (fp16 shards out of row order plus manifest.csv); no audio, weights or torch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hearsay import probe

MODEL, NAME = "toy-backbone", "toy_set"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


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


# --- the CLIs: --allow-decode-errors reaches the loader, and only on request ---------------

CLIS = {  # script -> argv whose first embedding load reads the toy set
    "train_probe": ["--model", MODEL, "--train", NAME, "--val", NAME],
    "export_probe_scores": ["--probe", "p", "--name", "n", "--train", NAME, "--folds", "f.csv",
                            "--test", NAME],
    "tsv_from_embeddings": ["--probe", "p", "--embeddings", NAME],
}


class _Loaded(Exception):
    """Raised after the first load: the rest of main() needs a real probe and real data."""


def _cli(name: str):
    spec = importlib.util.spec_from_file_location(f"cli_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_first_load(name, argv, monkeypatch):
    """main() up to its first embedding load, through the real loader on the toy set; returns
    the keyword arguments the script passed it."""
    mod = _cli(name)
    calls = []

    def first_load(*args, **kw):
        calls.append(kw)
        probe.load_embeddings(*args, **kw)
        raise _Loaded

    monkeypatch.setattr(mod, "load_embeddings", first_load)
    if hasattr(mod, "Probe"):  # a saved probe is read before any set; only its backbone is used
        stub = SimpleNamespace(load=lambda path: SimpleNamespace(backbone=MODEL))
        monkeypatch.setattr(mod, "Probe", stub)
    monkeypatch.setattr(sys, "argv", [name, *argv])
    with pytest.raises(_Loaded):
        mod.main()
    return calls[0]


@pytest.mark.parametrize("name", CLIS)
def test_cli_refuses_a_flagged_set_by_default(root, monkeypatch, name):
    _make(root, flagged=(1, 5))
    with pytest.raises(ValueError, match=r"2 row\(s\) flagged decode_error"):
        _run_first_load(name, CLIS[name], monkeypatch)


@pytest.mark.parametrize("name", CLIS)
def test_cli_allow_decode_errors_loads_the_flagged_set(root, monkeypatch, name):
    _make(root, flagged=(1, 5))
    kw = _run_first_load(name, [*CLIS[name], "--allow-decode-errors"], monkeypatch)
    assert kw == {"allow_decode_errors": True}


@pytest.mark.parametrize("name", CLIS)
def test_cli_default_passes_the_refusing_default(root, monkeypatch, name):
    _make(root)
    assert _run_first_load(name, CLIS[name], monkeypatch) == {"allow_decode_errors": False}


@pytest.mark.parametrize("name", CLIS)
def test_cli_every_load_carries_the_flag(name):
    """Every set a script reads (train, val, test, stress) goes through the one loader bound to
    the flag; a direct load_embeddings(...) call would bypass it."""
    src = (SCRIPTS / f"{name}.py").read_text()
    assert "partial(load_embeddings, allow_decode_errors=args.allow_decode_errors)" in src
    assert "load_embeddings(" not in src
