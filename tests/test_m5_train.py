"""scripts/m5_train.py end to end on a synthetic bundle with the tiny CPU model (spec appendix
E4, E7, E9, E10, G4; the plan's smoke-from-tarball test)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from hearsay import SR
from hearsay.m5_bundle import tree_sha
from hearsay.m5_data import shortcut_aucs, training_rows

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "scripts" / "m5_train.py"


def _load_trainer():
    spec = importlib.util.spec_from_file_location("m5_train", TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clip(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = np.sin(2 * np.pi * (120 + 40 * (seed % 5)) * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 2 * t))
    return (0.4 * x + 0.02 * rng.standard_normal(t.size)).astype(np.float32)


def make_bundle(root: Path, n_per_cell: int = 3) -> Path:
    """core rows over folds 0..4 + holdout, both classes, three sources; a few extras; a test set."""
    (root / "core").mkdir(parents=True)
    (root / "extra").mkdir()
    (root / "test").mkdir()
    rows, i = [], 0
    for fold in ["0", "1", "2", "3", "4", "holdout"]:
        for label, src, gen in (("bonafide", "ljspeech", "bonafide"), ("bonafide", "librispeech", "bonafide"),
                                ("spoof", "diffssd", f"gen{fold}")):
            for _ in range(n_per_cell):
                f = f"c{i}.flac"
                sf.write(root / "core" / f, _clip(1.0 + 0.1 * (i % 7), i), SR, subtype="PCM_16", format="FLAC")
                rows.append({"id": f"c{i}", "path": f"/d/{f}", "label": label, "generator": gen,
                             "speaker": f"s{i}", "source": src, "group": f"g{i}", "fold": fold,
                             "train_scope": "core", "model_name": gen, "file": f,
                             "duration": 1.0 + 0.1 * (i % 7), "peak": 0.4, "lead_s": 0.0})
                i += 1
    for k in range(4):
        f = f"x{k}.flac"
        sf.write(root / "extra" / f, _clip(1.2, 100 + k), SR, subtype="PCM_16", format="FLAC")
        rows.append({"id": f"x{k}", "path": f"/d/{f}", "label": "spoof", "generator": "MLAAD-M",
                     "speaker": "m", "source": "mlaad", "group": "mlaad:M", "fold": "extra",
                     "train_scope": "extra_mlaad", "model_name": "MLAAD-M", "file": f,
                     "duration": 1.2, "peak": 0.4, "lead_s": 0.0})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    trows = []
    for k in range(5):
        f = f"t{k}.flac"
        sf.write(root / "test" / f, _clip(1.1, 200 + k), SR, subtype="PCM_16", format="FLAC")
        trows.append({"filename": f"HGT{k}.wav", "path": f"data/nsa/HackGTHearsayTesting/HGT{k}.wav",
                      "id": f"t{k}", "file": f, "duration": 1.1})
    pd.DataFrame(trows).to_csv(root / "test" / "manifest.csv", index=False)
    (root / "config_sha.txt").write_text("x")
    sha = tree_sha(root)
    (root / "TREE_SHA").write_text(sha)
    (root / "bundle_meta.json").write_text(json.dumps({"tree_sha": sha, "n_rows": len(rows),
                                                       "git_sha": "test"}))
    return root


def _run(bundle: Path, out: Path, *extra: str, fold: str = "0", steps: int = 3):
    cmd = [sys.executable, str(TRAIN), "--bundle", str(bundle), "--fold", fold, "--steps", str(steps),
           "--eval-every", "2", "--steps-per-epoch", "2", "--eval-n", "6", "--workers", "0",
           "--cpu-smoke", "--out", str(out), "--config", '{"batch_size": 4, "keep_layers": 2}',
           *extra]  # fmt: skip
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=False)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp("bundle") / "v1")


@pytest.mark.slow
def test_smoke_fold_model_end_to_end(bundle, tmp_path):
    r = _run(bundle, tmp_path / "f0")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "DONE" in r.stdout
    s = pd.read_csv(tmp_path / "f0" / "scores_0.csv")
    assert (s.split == "inner_oof").all() and (s.fold.astype(str) == "0").all()
    assert np.isfinite(s.logit).all() and s.score.between(0, 1).all()
    meta = json.loads((tmp_path / "f0" / "run_meta.json").read_text())
    assert meta["steps"] == 3 and (tmp_path / "f0" / "model" / "hashes.json").exists()
    events = [json.loads(line)["event"] for line in open(tmp_path / "f0" / "train_log.jsonl")]
    assert "shortcut_gate" in events and "aug_rates" in events and "export_oof" in events


@pytest.mark.slow
def test_smoke_full_model_exports_holdout_test_and_diagnostics(bundle, tmp_path):
    r = _run(bundle, tmp_path / "full", fold="full")
    assert r.returncode == 0, r.stderr[-3000:]
    for f in ("scores_holdout.csv", "scores_test.csv", "diag_holdout_testlen.csv",
              "diag_holdout_aug.csv", "diag_holdout_14s.csv", "diag_test_14s.csv"):
        assert (tmp_path / "full" / f).exists(), f
    t = pd.read_csv(tmp_path / "full" / "scores_test.csv")
    assert t.path.tolist() == pd.read_csv(bundle / "test" / "manifest.csv").path.tolist()
    h = pd.read_csv(tmp_path / "full" / "scores_holdout.csv")
    assert (h.split == "holdout").all() and len(h) == 9


@pytest.mark.slow
def test_g4_resume_continues_from_checkpoint(bundle, tmp_path):
    out = tmp_path / "res"
    r = _run(bundle, out, steps=2)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "ckpt" / "last.pt").exists()
    r = _run(bundle, out, "--resume", steps=4)
    assert r.returncode == 0, r.stderr[-2000:]
    events = [json.loads(line) for line in open(out / "train_log.jsonl")]
    resumed = [e for e in events if e["event"] == "resumed"]
    assert resumed and resumed[-1]["step"] == 2
    assert json.loads((out / "run_meta.json").read_text())["steps"] == 4


def test_e10_refuses_without_cuda_unless_smoke(bundle, tmp_path):
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    cmd = [sys.executable, str(TRAIN), "--bundle", str(bundle), "--fold", "0", "--steps", "1",
           "--out", str(tmp_path / "x"), "--skip-tree-check"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=False)
    assert r.returncode != 0 and "CUDA only" in (r.stderr + r.stdout)


def test_tree_sha_mismatch_refused(bundle, tmp_path):
    bad = tmp_path / "bad"
    import shutil

    shutil.copytree(bundle, bad)
    (bad / "manifest.csv").write_text((bad / "manifest.csv").read_text() + "\n")
    r = _run(bad, tmp_path / "o", steps=1)
    assert r.returncode != 0 and "tree sha" in (r.stderr + r.stdout)


def test_e4_sampler_quotas_and_label_balance(bundle):
    m5 = _load_trainer()
    m = pd.read_csv(bundle / "manifest.csv")
    m["fold"] = m.fold.astype(str)
    tr = training_rows(m, "0").reset_index(drop=True)
    mix = {"real": {"ljspeech": 0.25, "librispeech": 0.75}, "spoof": {"diffssd": 0.8, "mlaad": 0.2}}

    class Mixer:
        def draw(self):
            return 1.0

    s = m5.MixBatchSampler(tr, 16, mix, Mixer(), seed=0, steps_per_epoch=10)
    labels, srcs = [], []
    for batch in s.batches(0, 200):
        idx = [b[0] for b in batch]
        labels += tr.label.iloc[idx].tolist()
        srcs += tr.source.iloc[idx].tolist()
    labels, srcs = np.array(labels), np.array(srcs)
    assert abs((labels == "spoof").mean() - 0.5) < 0.02
    real = srcs[labels == "bonafide"]
    assert abs((real == "librispeech").mean() - 0.75) < 0.05
    spoof = srcs[labels == "spoof"]
    assert abs((spoof == "mlaad").mean() - 0.2) < 0.05
    assert not set(tr.iloc[[b[0] for b in next(s.batches(0, 1))]].fold) & {"0", "holdout"}


def test_e9_gate_ignores_holdout_rows(bundle):
    m = pd.read_csv(bundle / "manifest.csv")
    m["fold"] = m.fold.astype(str)
    before = shortcut_aucs(training_rows(m, "1"))
    m2 = m.copy()
    hold = m2.fold == "holdout"
    m2.loc[hold, "duration"] = 99.0
    m2.loc[hold, "label"] = np.where(m2.loc[hold, "label"] == "spoof", "bonafide", "spoof")
    assert shortcut_aucs(training_rows(m2, "1")) == before


def test_e7_nan_loss_aborts():
    """bce_with_smoothing on non-finite logits is non-finite, which the loop turns into a hard
    exit (m5_train.py checks torch.isfinite(loss) every step)."""
    import torch

    from hearsay.m5_model import bce_with_smoothing

    loss = bce_with_smoothing(torch.tensor([float("nan"), 1.0]), torch.tensor([1, 0]), 0.05)
    assert not torch.isfinite(loss)
    src = TRAIN.read_text()
    assert "if not torch.isfinite(loss):" in src and 'sys.exit("FATAL: non-finite loss")' in src
