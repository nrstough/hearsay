"""M5 score export and readouts (spec appendix F1-F5, F7) on synthetic run directories."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hearsay import SR
from hearsay.m5_data import FOLDS, INNER_FOLDS, collate, deploy_transform
from hearsay.m5_model import M5Config, build_model, save_m5, score_batch
from hearsay.metrics import sigmoid

REPO = Path(__file__).resolve().parents[1]


def _fake_runs(tmp_path: Path, folds: pd.DataFrame, test: pd.DataFrame, arm="nsa_extra",
               drop_fold: str | None = None) -> Path:  # fmt: skip
    """Run dirs shaped like m5_train.py's output, with scores that separate the classes."""
    rng = np.random.default_rng(0)
    runs = tmp_path / "runs" / "job"
    net = build_model(M5Config(keep_layers=1, spec_augment=False), tiny=True).eval()

    def scores(rows):
        y = (rows.label == "spoof").to_numpy(float)
        return 2.5 * (2 * y - 1) + rng.standard_normal(len(rows))

    for k in INNER_FOLDS:
        if k == drop_fold:
            continue
        d = runs / f"fold{k}_{arm}"
        d.mkdir(parents=True)
        va = folds[folds.fold == k]
        lg = scores(va)
        pd.DataFrame({"path": va.path, "fold": k, "split": "inner_oof", "score": sigmoid(lg),
                      "logit": lg}).to_csv(d / f"scores_{k}.csv", index=False)
        save_m5(net, d / "model")
        (d / "run_meta.json").write_text(json.dumps({"fold": k, "arm": arm, "steps": 2500,
                                                     "config": {}, "config_hash": "x",
                                                     "bundle_tree": "t"}))
    d = runs / f"foldfull_{arm}"
    d.mkdir(parents=True)
    hr = folds[folds.fold == "holdout"]
    lg = scores(hr)
    pd.DataFrame({"path": hr.path, "fold": "holdout", "split": "holdout", "score": sigmoid(lg),
                  "logit": lg}).to_csv(d / "scores_holdout.csv", index=False)
    tl = rng.standard_normal(len(test))
    pd.DataFrame({"path": test.path, "fold": "test", "split": "test", "score": sigmoid(tl),
                  "logit": tl}).to_csv(d / "scores_test.csv", index=False)
    pd.DataFrame({"path": hr.path, "crop_s": 3.4, "logit": lg + 0.1}).to_csv(
        d / "diag_holdout_testlen.csv", index=False)
    ops = ["noise", "band", "reverb", "gain_clip", "rawboost_conv"]
    pd.DataFrame({"path": hr.path, "op": [ops[i % 5] for i in range(len(hr))], "logit": lg - 0.2}
                 ).to_csv(d / "diag_holdout_aug.csv", index=False)
    save_m5(net, d / "model")
    (d / "run_meta.json").write_text(json.dumps({
        "fold": "full", "arm": arm, "steps": 3000, "config": {"keep_layers": 1},
        "config_hash": "x", "bundle_tree": "t", "gpu": "test", "aug_rates": {},
        "shortcut_gate": {}, "hashes": {}}))
    return runs.parent


@pytest.fixture
def real_folds():
    if not FOLDS.exists():
        pytest.skip("fold file not present")
    f = pd.read_csv(FOLDS)
    f["fold"] = f.fold.astype(str)
    return f


@pytest.mark.slow
def test_f1_f5_assemble_counts_schema_meta(tmp_path, real_folds):
    test = pd.read_csv(REPO / "outputs/manifests/nsa_test.csv")
    runs = _fake_runs(tmp_path, real_folds, test)
    bm = tmp_path / "bundle_manifest.csv"
    real_folds.assign(duration=np.random.default_rng(1).uniform(2, 9, len(real_folds))).to_csv(
        bm, index=False)
    cmd = [sys.executable, str(REPO / "scripts/m5_assemble.py"), "--runs", str(runs),
           "--bundle-manifest", str(bm), "--out-name", "m5_test_assemble", "--arm", "nsa_extra"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    out = REPO / "outputs/detector_scores/m5_test_assemble.csv"
    try:
        s = pd.read_csv(out)
        assert list(s.columns) == ["path", "fold", "split", "score", "logit"]
        assert s.split.value_counts().to_dict() == {"inner_oof": 16142, "holdout": 3858,
                                                    "test": 1671}
        assert s.path.is_unique and np.isfinite(s.logit).all() and s.score.between(0, 1).all()
        # F2: each inner row scored by its own fold's model
        j = s[s.split == "inner_oof"].set_index("path").join(real_folds.set_index("path").fold,
                                                             rsuffix="_true")
        assert (j.fold.astype(str) == j.fold_true.astype(str)).all()
        # F3: test rows in nsa_test order
        assert s[s.split == "test"].path.tolist() == test.path.tolist()
        mdirs = sorted((REPO / "models").glob("m5_test_assemble_*"))
        meta = json.loads((mdirs[-1] / "meta.json").read_text())
        for key in ("val_min_dcf", "val_eer", "val_sponsor_code_asis", "val_sponsor_code_flipped",
                    "val_by_generator", "val_by_bonafide_source", "val_by_length_bucket",
                    "val_diagnostics", "oof_pooled", "cv", "gate", "m1_bar", "test", "stackable",
                    "score_duration_spearman_bonafide"):
            assert key in meta, key
        assert meta["stackable"] is True and set(meta["cv"]) == set(INNER_FOLDS)
        assert set(meta["val_by_generator"]) == {"playht", "wavegrad2"}
        assert "diag_holdout_aug_by_op" in meta["val_diagnostics"]
        assert meta["gate"]["rule"].startswith("replace M1 only if")
    finally:
        out.unlink(missing_ok=True)
        for d in (REPO / "models").glob("m5_test_assemble_*"):
            import shutil

            shutil.rmtree(d)


@pytest.mark.slow
def test_f1_fallback_no_partial_inner_oof(tmp_path, real_folds):
    test = pd.read_csv(REPO / "outputs/manifests/nsa_test.csv")
    runs = _fake_runs(tmp_path, real_folds, test, drop_fold="3")
    bm = tmp_path / "bundle_manifest.csv"
    real_folds.assign(duration=5.0).to_csv(bm, index=False)
    cmd = [sys.executable, str(REPO / "scripts/m5_assemble.py"), "--runs", str(runs),
           "--bundle-manifest", str(bm), "--out-name", "m5_test_fallback"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=False)
    assert r.returncode == 0, r.stderr[-3000:]
    out = REPO / "outputs/detector_scores/m5_test_fallback.csv"
    try:
        s = pd.read_csv(out)
        assert s.split.value_counts().to_dict() == {"holdout": 3858, "test": 1671}
        mdirs = sorted((REPO / "models").glob("m5_test_fallback_*"))
        assert json.loads((mdirs[-1] / "meta.json").read_text())["stackable"] is False
    finally:
        out.unlink(missing_ok=True)
        for d in (REPO / "models").glob("m5_test_fallback_*"):
            import shutil

            shutil.rmtree(d)


def test_f4_scores_are_sigmoid_of_logits_and_direction():
    lg = np.array([-3.0, 0.0, 4.0])
    s = sigmoid(lg)
    assert (np.diff(s) > 0).all() and 0 < s[0] < 0.5 < s[2] < 1


def test_f7_long_clip_scored_whole_at_14s_and_capped_at_8s_by_default():
    net = build_model(M5Config(keep_layers=1, spec_augment=False), tiny=True).eval()
    rng = np.random.default_rng(0)
    x = (0.3 * rng.standard_normal(int(13.6 * SR))).astype(np.float32)
    x8, x14 = deploy_transform(x), deploy_transform(x, max_s=14.0)
    assert x8.size == 8 * SR and x14.size == int(13.6 * SR)
    xs, m = collate([x14])
    frames = net.backbone._get_feat_extract_output_lengths(int(m.sum()))
    assert int(frames) == int(net.backbone._get_feat_extract_output_lengths(x14.size))
    assert np.isfinite(score_batch(net, xs, m).numpy()).all()


def test_newer_short_run_never_replaces_a_final_model(tmp_path):
    """A resume probe / pilot written after the finals must not be selected (Codex round 4)."""
    import importlib.util
    import time

    spec = importlib.util.spec_from_file_location("m5_assemble", REPO / "scripts" / "m5_assemble.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path / "runs"
    for name, steps, age in (("fold4_frozen", 2500, 2), ("resume_probe", 600, 1), ("fold4_ft", 2500, 0)):
        d = root / name
        d.mkdir(parents=True)
        (d / "scores_4.csv").write_text("path,fold,split,score,logit\n")
        cfg = {"train_top": 12 if name == "fold4_ft" else 0}
        (d / "run_meta.json").write_text(json.dumps({"fold": "4", "arm": "nsa_extra", "steps": steps, "config": cfg}))
        t = time.time() - age
        import os

        os.utime(d / "run_meta.json", (t, t))
    picked = mod.find_runs(root, "nsa_extra", train_top=0)
    assert picked["4"].name == "fold4_frozen"  # not the newer 600-step probe
    assert mod.find_runs(root, "nsa_extra", train_top=0, min_steps=100)["4"].name == "resume_probe"
    assert mod.find_runs(root, "nsa_extra", train_top=12)["4"].name == "fold4_ft"
