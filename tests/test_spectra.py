"""M3 Spectra-AASIST scoring: input path, short-clip modes, export, gates, readouts.

Hermetic: a fake torch module stands in for Spectra, tiny WAVs are written with soundfile and
decoded through the real loader, and every script run points --repo-root at a temp directory
(run spec docs/specs/2026-09-26_m3-spectra-aasist.md, test IDs A1-A14, B1-B8, C1-C16, D1-D11, W1).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import torchaudio

import hearsay.embed as embed_mod
from hearsay import SR
from hearsay.audio import load_audio, windows
from hearsay.embed import prepare_segment
from hearsay.embed import test_duration_sampler as duration_sampler
from hearsay.metrics import by_group, min_cost, sigmoid
from hearsay.spectra import (
    HOP,
    MAX_INPUT_SAMPLES,
    MIN_WHOLE_SAMPLES,
    PAD_MODES,
    WIN,
    crop_plan,
    direction_check,
    forward_windows,
    fusion_frame,
    masked_report,
    pad_samples,
    pad_windows,
    peak_normalize,
    platt_diagnostic,
    prepare_input,
    score_clip,
    spectra_logits,
    stress_readout,
    sweep_thresholds,
    synth_logit,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "score_spectra.py"
BASE_HEADER = "path,fold,split,score,logit"


# --- helpers --------------------------------------------------------------------------------


def _clip(seconds: float, seed: int = 0, amp: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(seconds * SR))).astype(np.float32)


def _tone(seconds: float, hz: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _wav(path: Path, x: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, SR, subtype="FLOAT")
    return path


class FakeSpectra(torch.nn.Module):
    """Louder input = more spoof (margin negative for quiet clips). Two logits that are not an exact difference of each other
    (so a recomputed synth_logit on resume is detectable). `fail_calls` maps the k-th forward
    (1-based) to "raise", "nan" or "shape"."""

    def __init__(self, gain: float = 1.0, fail_calls: dict | None = None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.gain, self.fail_calls, self.calls = gain, fail_calls or {}, []
        self.eval()

    def forward(self, x):
        assert x.ndim == 2 and x.dtype == torch.float32, (x.ndim, x.dtype)
        assert x.device == self.w.device and not torch.is_grad_enabled() and not self.training
        self.calls.append(x.detach().cpu().numpy().copy())
        k = len(self.calls)
        m = x.abs().mean(dim=1)
        if self.fail_calls.get(k) == "raise":
            raise RuntimeError("fake model failure")
        if self.fail_calls.get(k) == "nan":
            return torch.stack([m * float("nan"), 0.1 * m], dim=1)
        if self.fail_calls.get(k) == "shape":
            return torch.stack([m, m, m], dim=1)
        return torch.stack([self.gain * 10 * (m - 0.15), 0.1 * m], dim=1)


@torch.inference_mode()
def _old_spectra_logits(model, x, max_windows=16):
    """spectra_logits exactly as it was before M3 (B5 reference)."""
    device = next(model.parameters()).device
    w = torch.from_numpy(windows(x, WIN, HOP)[:max_windows])
    w = torchaudio.functional.preemphasis(w)
    return model(w.to(device)).float().cpu().numpy().mean(axis=0)


def _load_script():
    spec = importlib.util.spec_from_file_location("score_spectra", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_repo(root: Path, inner=(3, 3), hold=(2, 2), n_test=4, sec=2.0, bad=(),
              n_durations=20) -> dict:  # fmt: skip
    """A fake repo: fold file (absolute paths), test manifest (relative paths), training manifest
    in fold order, durations file. `inner`/`hold` = (n_spoof, n_bonafide). `bad` = row indices
    (fold rows first, then test rows) whose file is not audio."""
    rows, i = [], 0
    for fold, (ns, nb) in (("0", inner), ("holdout", hold)):
        for k in range(ns + nb):
            spoof = k < ns
            rows.append({"path": str(root / "data" / f"c{i}.wav"),
                         "label": "spoof" if spoof else "bonafide",
                         "generator": ("g1" if k % 2 else "g2") if spoof else "bonafide",
                         "speaker": f"s{i}", "source": "diffssd" if spoof else ("src_a" if k % 2 else "src_b"),
                         "group": f"grp{i}", "fold": fold if fold != "0" or k % 2 else "1",
                         "_amp": 0.5 if spoof else 0.05})  # fmt: skip
            i += 1
    test = [{"filename": f"t{j}.wav", "path": f"data/test/t{j}.wav", "_amp": 0.5 if j % 2 else 0.05}
            for j in range(n_test)]  # fmt: skip
    all_rows = rows + test
    for j, r in enumerate(all_rows):
        p = root / r["path"] if not Path(r["path"]).is_absolute() else Path(r["path"])
        if j in bad:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"not audio at all")
        else:
            _wav(p, _clip(sec, seed=j, amp=r["_amp"]))
    f = pd.DataFrame(rows).drop(columns="_amp")
    (root / "splits").mkdir(exist_ok=True)
    (root / "outputs" / "manifests").mkdir(parents=True, exist_ok=True)
    f.to_csv(root / "splits" / "nsa_folds.csv", index=False)
    f[["path", "label", "generator", "speaker", "source"]].to_csv(
        root / "outputs" / "manifests" / "nsa_train_sample.csv", index=False)
    tdf = pd.DataFrame(test, columns=["filename", "path", "_amp"])
    tdf.drop(columns="_amp").to_csv(root / "outputs" / "manifests" / "nsa_test.csv", index=False)
    durs = np.round(np.linspace(1.2, 3.6, n_durations), 6)
    pd.DataFrame({"filename": [f"d{k}.wav" for k in range(n_durations)], "duration_s": durs}).to_csv(
        root / "splits" / "nsa_test_durations.csv", index=False)
    return {"folds": f, "test": tdf, "n_fold": len(rows), "n_test": n_test}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    info = make_repo(tmp_path)
    monkeypatch.setattr(embed_mod, "TEST_DURATIONS", tmp_path / "splits" / "nsa_test_durations.csv")
    return tmp_path, info


def _run(mod, root: Path, *args, loader=None, **kw) -> int:
    loader = loader or (lambda device: FakeSpectra())
    argv = ["--repo-root", str(root), "--device", "cpu", "--chunk", "2", *args]
    try:
        return mod.run(argv, loader=loader)
    except SystemExit as e:
        return e.code


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


# --- A: input path --------------------------------------------------------------------------


@pytest.mark.parametrize("crop_s,band", [(None, True), (1.5, True), (None, False), (1.5, False)])
def test_a1_prepare_input_is_prepare_segment(crop_s, band):
    x = _clip(4.0, seed=1)
    a = prepare_input(x, crop_s, 7, band_match=band)
    b = prepare_segment(x, crop_s, 7, band_match=band)
    assert a.dtype == np.float32 and np.array_equal(a, b)


def test_a2_band_match_counterfactual_and_fir_warm():
    import hearsay.handcrafted as hc

    x = _tone(2.0, 7800)
    y = prepare_input(x)
    drop_db = 20 * np.log10(np.sqrt(np.mean(y**2)) / np.sqrt(np.mean(x**2)))
    assert drop_db < -40, drop_db
    y_off = prepare_input(x, band_match=False)
    off_db = 20 * np.log10(np.sqrt(np.mean(y_off**2)) / np.sqrt(np.mean(x**2)))
    assert abs(off_db) < 0.5, off_db  # the tone is untouched without the band match
    assert hc._FIR is not None and hc._FIR.dtype == np.float32


def test_a3_test_rows_no_crop_and_cap():
    x = _clip(10.0)
    y = prepare_input(x)
    assert y.size == int(8.0 * SR)  # white noise is not trimmed; capped, never cropped
    assert prepare_input(_clip(3.0)).size == 3 * SR


def test_a4_crop_plan_reproduces_shared_recipe(repo):
    root, _ = repo
    man = root / "outputs" / "manifests" / "nsa_train_sample.csv"
    plan = crop_plan(man, 0)
    paths = pd.read_csv(man).path.tolist()
    draw = duration_sampler(0)
    for i, p in enumerate(paths):
        assert plan[p] == (draw(), i)
    other = crop_plan(man, 200)
    assert any(other[p][0] != plan[p][0] for p in paths) and other[paths[0]][1] == 200
    dup = root / "dup.csv"
    pd.DataFrame({"path": [paths[0], paths[0]]}).to_csv(dup, index=False)
    with pytest.raises(ValueError):
        crop_plan(dup, 0)


def test_a4_real_seed0_draws_pinned(monkeypatch):
    real = REPO / "splits" / "nsa_test_durations.csv"
    if not real.exists():
        pytest.skip("test durations not present")
    monkeypatch.setattr(embed_mod, "TEST_DURATIONS", real)
    draw = duration_sampler(0)
    got = [round(draw(), 6) for _ in range(5)]
    assert got == [3.552688, 3.25075, 3.041875, 3.1115, 3.3205], got


def _autocorr_at(v: np.ndarray, lag: int) -> float:
    a, b = v[:-lag].astype(np.float64), v[lag:].astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0


def test_a5_a6_a7_short_clip_modes():
    x = _clip(2.0, seed=3)
    rep, zero, whole = (pad_windows(x, m) for m in PAD_MODES)
    assert rep.shape == (1, WIN) and _autocorr_at(rep[0], x.size) > 0.9
    assert zero.shape == (1, WIN) and not zero[0, x.size :].any() and _autocorr_at(zero[0], x.size) < 0.05
    assert whole.shape == (1, x.size) and np.array_equal(whole[0], x)
    tiny = _clip(0.3, seed=4)
    assert pad_windows(tiny, "whole").shape == (1, MIN_WHOLE_SAMPLES)
    assert [pad_samples(x.size, m) for m in PAD_MODES] == [WIN - x.size, WIN - x.size, 0]
    assert pad_samples(tiny.size, "whole") == MIN_WHOLE_SAMPLES - tiny.size
    assert pad_samples(WIN, "zero") == 0
    for m in PAD_MODES:
        assert pad_windows(x, m).dtype == np.float32 and pad_windows(x, m).flags.c_contiguous  # A12


def test_a7_forward_guard():
    f = FakeSpectra()
    assert forward_windows(f, np.zeros((1, MAX_INPUT_SAMPLES), np.float32)).shape == (1, 2)
    with pytest.raises(ValueError):
        forward_windows(f, np.zeros((1, 128_400), np.float32))


@pytest.mark.parametrize("seconds,n_win", [(4.5, 2), (6.0, 2), (8.0, 3)])
def test_a8_a9_long_clips_identical_across_modes(seconds, n_win):
    x = _clip(seconds, seed=5)
    ws = [pad_windows(x, m) for m in PAD_MODES]
    assert all(np.array_equal(ws[0], w) for w in ws[1:])
    assert ws[0].shape == (n_win, WIN) and np.array_equal(ws[0][-1], x[-WIN:])
    assert pad_windows(x, "zero", max_windows=1).shape == (1, WIN)


def test_a10_preemphasis_applied():
    f = FakeSpectra()
    x = np.r_[np.zeros(SR, np.float32), np.ones(SR, np.float32)]  # a DC step
    spectra_logits(f, x, pad_mode="zero")
    expect = torchaudio.functional.preemphasis(torch.from_numpy(pad_windows(x, "zero"))).numpy()
    assert np.array_equal(f.calls[0], expect)
    assert not np.array_equal(f.calls[0], pad_windows(x, "zero"))


def test_a11_degenerate_inputs_finite(tmp_path):
    f = FakeSpectra()
    nan_wav = np.r_[_clip(1.5), np.array([np.nan, np.inf, -np.inf], np.float32), _clip(1.5, seed=9)]
    x_nan = load_audio(_wav(tmp_path / "nan.wav", nan_wav))
    for x in (np.zeros(3 * SR, np.float32), _clip(0.1), x_nan):
        for m in PAD_MODES:
            out = score_clip(f, prepare_input(x), (m,))
            assert np.all(np.isfinite(out[m])) and np.isfinite(synth_logit(out[m]))


def test_a14_peak_normalize(repo):
    x = _clip(2.0, amp=0.05)
    y = peak_normalize(x)
    assert y.dtype == np.float32 and np.abs(y).max() == pytest.approx(0.95, abs=1e-6)
    assert np.allclose(y / 0.95, x / np.abs(x).max(), atol=1e-6)
    z = np.zeros(SR, np.float32)
    assert np.array_equal(peak_normalize(z), z)
    assert np.abs(peak_normalize(_clip(1.0, amp=3.0))).max() == pytest.approx(0.95, abs=1e-6)
    root, _ = repo
    mod = _load_script()
    assert _run(mod, root, "--pad-mode", "zero", "--peak-norm", "--out-name", "pn") == 0
    assert _run(mod, root, "--pad-mode", "zero", "--out-name", "raw") == 0
    meta = {m["config"]["peak_norm"]: m for m in
            (json.loads(p.read_text()) for p in (root / "models").glob("m3_spectra_*/meta.json"))}
    assert set(meta) == {True, False} and meta[True]["config_hash"] != meta[False]["config_hash"]
    a = pd.read_csv(root / "outputs" / "detector_scores" / "pn.csv")
    b = pd.read_csv(root / "outputs" / "detector_scores" / "raw.csv")
    assert not np.allclose(a.logit, b.logit)  # the fake is level-driven, so the arm must change scores
    raw = pd.read_csv(root / "outputs" / "spectra" / "pn_raw.csv")
    assert raw.peak.max() > 0.95 * 1.2 or raw.peak.min() < 0.3  # `peak` is the pre-normalization level


def test_a13_bad_mode_raises():
    with pytest.raises(ValueError):
        pad_windows(_clip(1.0), "tile")


# --- B: scoring -----------------------------------------------------------------------------


def test_b1_direction_contract():
    assert synth_logit(np.array([3.0, -2.0])) == 5.0
    assert synth_logit(np.array([4.0, -2.0])) > synth_logit(np.array([3.0, -2.0]))
    assert synth_logit(np.array([3.0, -1.0])) < synth_logit(np.array([3.0, -2.0]))


def test_b2_window_mean_and_b4_device_no_grad():
    f = FakeSpectra()
    x = np.r_[_clip(3.0, amp=0.1), _clip(3.0, seed=2, amp=0.9)]  # two windows, different levels
    with torch.enable_grad():  # inference_mode inside must still win
        lg = spectra_logits(f, x)
    per_window = forward_windows(f, pad_windows(x))
    assert per_window.shape == (2, 2) and abs(per_window[0, 0] - per_window[1, 0]) > 0.1
    assert np.allclose(lg, per_window.mean(axis=0))


def test_b3_max_windows_cap_reaches_model():
    f = FakeSpectra()
    spectra_logits(f, _clip(40.0, seed=6))
    assert f.calls[-1].shape[0] == 16
    spectra_logits(f, _clip(12.0, seed=6), max_windows=2)
    assert f.calls[-1].shape[0] == 2


def test_b5_default_path_unchanged():
    f = FakeSpectra()
    for x in (_clip(2.0, seed=7), _clip(6.0, seed=8)):
        assert np.array_equal(spectra_logits(f, x), _old_spectra_logits(f, x))
        assert np.array_equal(spectra_logits(f, x, 4), _old_spectra_logits(f, x, 4))
        assert np.array_equal(spectra_logits(f, x), spectra_logits(f, x, pad_mode="repeat"))


def test_b8_non_finite_or_bad_shape_is_a_failure():
    x = prepare_input(_clip(2.0))
    with pytest.raises(RuntimeError):
        score_clip(FakeSpectra(fail_calls={1: "nan"}), x, ("zero",))
    with pytest.raises(RuntimeError):
        score_clip(FakeSpectra(fail_calls={1: "shape"}), x, ("zero",))


# --- C: script, export, gates ---------------------------------------------------------------


def test_c1_to_c7_c12_c13_full_run(repo):
    root, info = repo
    mod = _load_script()
    assert _run(mod, root, "--pad-mode", "zero", "--out-name", "spectra_test") == 0
    pub = root / "outputs" / "detector_scores" / "spectra_test.csv"
    assert pub.read_text().splitlines()[0] == BASE_HEADER  # C4
    df = pd.read_csv(pub)
    assert len(df) == info["n_fold"] + info["n_test"] and df.path.is_unique  # C1
    assert df.path.tolist() == info["folds"].path.tolist() + info["test"].path.tolist()  # C2, C3
    assert (df.split.value_counts().to_dict() == {"inner_oof": 6, "holdout": 4, "test": 4})
    assert (df.loc[df.split == "test", "fold"] == "test").all()
    ok = df.logit.notna()
    assert ok.all() and np.allclose(df.score[ok], sigmoid(df.logit[ok]))  # C5, no calibration
    raw = pd.read_csv(root / "outputs" / "spectra" / "spectra_test_raw.csv")
    assert raw.path.tolist() == df.path.tolist() and np.allclose(raw.synth_logit, df.logit)  # C13
    assert list(raw.columns[:5]) == BASE_HEADER.split(",")
    summary = json.loads((root / "outputs" / "detector_scores" / "spectra_test_summary.json").read_text())
    run_dir = root / summary["run_dir"]
    assert (run_dir / "meta.json").exists() and (run_dir / "fusion.csv").exists()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["platt_diagnostic"]["in_sample_diagnostic"] and abs(meta["platt_diagnostic"]["a"] - 1) > 1e-6
    assert not (root / "outputs" / "spectra" / "spectra_test_partial.csv").exists()
    assert meta["config"]["pad_modes"] == ["zero"] and meta["config"]["band_match"] is True  # D7
    for k in ("holdout", "inner", "test", "direction", "thresholds", "level_shortcut", "failures"):
        assert k in meta, k
    assert meta["holdout"]["by_generator"].keys() == {"g1", "g2"}
    assert meta["test"]["frac_above_half"] == pytest.approx(0.5)  # 2 loud, 2 quiet (D6)


def test_c6_duplicate_paths_refused(repo):
    root, info = repo
    t = pd.read_csv(root / "outputs" / "manifests" / "nsa_test.csv")
    t.loc[0, "path"] = info["folds"].path.iloc[0]
    t.to_csv(root / "outputs" / "manifests" / "nsa_test.csv", index=False)
    assert _run(_load_script(), root) == 2
    with pytest.raises(ValueError):
        fusion_frame(["a", "a"], ["0", "0"], ["inner_oof", "inner_oof"], [0.1, 0.2])


def test_c8_failure_gate_per_split(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setattr(embed_mod, "TEST_DURATIONS", tmp_path / "a" / "splits" / "nsa_test_durations.csv")
    make_repo(tmp_path / "a", inner=(8, 8), hold=(0, 0), n_test=4, bad=(0, 1))  # 2 of 16 inner = 12.5%
    assert _run(mod, tmp_path / "a") == 3
    assert not (tmp_path / "a" / "outputs" / "detector_scores").exists()
    monkeypatch.setattr(embed_mod, "TEST_DURATIONS", tmp_path / "b" / "splits" / "nsa_test_durations.csv")
    make_repo(tmp_path / "b", inner=(10, 10), hold=(0, 0), n_test=0, bad=(0,))  # 1 of 20 = 5%
    assert _run(mod, tmp_path / "b") == 0
    monkeypatch.setattr(embed_mod, "TEST_DURATIONS", tmp_path / "c" / "splits" / "nsa_test_durations.csv")
    make_repo(tmp_path / "c", inner=(18, 18), hold=(0, 0), n_test=4, bad=(36, 37))  # 2/40 overall, 2/4 test
    assert _run(mod, tmp_path / "c") == 3


def test_c9_direction_gate_blocks_export(repo):
    root, _ = repo
    mod = _load_script()
    assert _run(mod, root, loader=lambda d: FakeSpectra(gain=-1.0)) == 4
    assert not (root / "outputs" / "detector_scores").exists()
    runs = list((root / "models").glob("m3_spectra_*"))
    assert len(runs) == 1 and (runs[0] / "raw.csv").exists() and not (runs[0] / "fusion.csv").exists()
    assert json.loads((runs[0] / "meta.json").read_text())["gate"]["direction"] == "failed"


def test_c10_pilot_writes_only_under_pilot_dir(repo):
    root, _ = repo
    before = _tree(root)
    assert _run(_load_script(), root, "--limit", "4", "--test-limit", "2",
                "--pad-modes", "repeat,zero,whole", "--name", "p") == 0  # fmt: skip
    created = _tree(root) - before
    assert created and all(p.startswith("outputs/spectra/pilot/") for p in created), created
    assert not (root / "outputs" / "detector_scores").exists() and not (root / "models").exists()
    assert not (root / "outputs" / "spectra" / "spectra_aasist_partial.csv").exists()
    pdir = next((root / "outputs" / "spectra" / "pilot").iterdir())
    meta = json.loads((pdir / "meta.json").read_text())
    assert set(meta["pilot"]) == set(PAD_MODES) and meta["selection"]["pick"] in PAD_MODES
    for m in PAD_MODES:  # direction recorded per mode, holdout rows skipped (A4/--limit parity)
        assert "direction" in meta["pilot"][m] and "holdout" not in meta["pilot"][m]
        sc = pd.read_csv(pdir / f"scores_{m}.csv")
        assert set(sc.split) == {"inner_oof", "test"}
    full_ok = _run(_load_script(), root, "--pad-mode", "zero") == 0
    assert full_ok
    raw = pd.read_csv(root / "outputs" / "spectra" / "spectra_aasist_raw.csv").set_index("path")
    sc = pd.read_csv(pdir / "scores_zero.csv").set_index("path")
    assert np.allclose(sc.crop_s, raw.loc[sc.index, "crop_s"], equal_nan=True)  # A4: --limit changes no crop
    assert sc.crop_s.notna().sum() == 4  # the inner rows were cropped, the test rows were not
    assert _run(_load_script(), root, "--limit", "4", "--test-limit", "0", "--name", "q") == 0
    qdir = max((root / "outputs" / "spectra" / "pilot").iterdir())
    assert set(pd.read_csv(qdir / "scores_zero.csv").split) == {"inner_oof"}  # --test-limit 0 means none


def test_c11_manifest_mode(repo):
    root, info = repo
    mod = _load_script()
    m = info["folds"][["path", "label", "generator", "source"]].copy()
    m.loc[m.label == "bonafide", "label"] = "bona-fide"
    m.to_csv(root / "stress.csv", index=False)
    assert _run(mod, root, "--manifest", "stress.csv", "--name", "st", "--pad-mode", "zero") == 0
    sc = pd.read_csv(root / "outputs" / "spectra" / "st" / "scores.csv")
    assert set(sc.split) == {"stress"} and set(sc.label) == {"spoof", "bonafide"}
    meta = json.loads((root / "outputs" / "spectra" / "st" / "meta.json").read_text())
    assert meta["stress"]["status"] == "ok" and meta["config"]["stress_seed"] == 200
    assert not (root / "outputs" / "detector_scores").exists()
    m.drop(columns="label").to_csv(root / "nolabel.csv", index=False)
    assert _run(mod, root, "--manifest", "nolabel.csv", "--name", "x") == 2


def test_c14_c16_resume(repo):
    root, _ = repo
    mod = _load_script()
    assert _run(mod, root, "--pad-mode", "zero", "--out-name", "a") == 0
    ref = (root / "outputs" / "detector_scores" / "a.csv").read_bytes()
    partial = root / "outputs" / "spectra" / "b_partial.csv"
    assert _run(mod, root, "--pad-mode", "zero", "--out-name", "b", "--max-chunks", "1") == 5
    assert partial.exists() and len(pd.read_csv(partial)) == 2
    assert not (root / "outputs" / "detector_scores" / "b.csv").exists()
    assert _run(mod, root, "--pad-mode", "zero", "--out-name", "b", "--resume") == 0
    assert (root / "outputs" / "detector_scores" / "b.csv").read_bytes() == ref
    assert not partial.exists()
    assert _run(mod, root, "--pad-mode", "zero", "--out-name", "b", "--resume") == 2  # nothing to resume
    assert _run(mod, root, "--pad-mode", "zero", "--out-name", "c", "--max-chunks", "1") == 5
    assert _run(mod, root, "--pad-mode", "repeat", "--out-name", "c", "--resume") == 2  # config changed
    man = root / "outputs" / "manifests" / "nsa_train_sample.csv"
    t = pd.read_csv(man)
    t.iloc[[0, 1]] = t.iloc[[1, 0]].to_numpy()
    t.to_csv(man, index=False)
    assert _run(mod, root, "--pad-mode", "zero", "--out-name", "c", "--resume") == 2  # C16
    assert _run(mod, root, "--pad-mode", "zero", "--out-name", "c") == 2  # partial exists, no --resume


def test_c15_no_lightgbm():
    _load_script()
    assert "lightgbm" not in sys.modules
    for p in (SCRIPT, Path(__file__), REPO / "src" / "hearsay" / "spectra.py"):
        for ln in p.read_text().splitlines():
            assert not ln.strip().startswith(("import lightgbm", "from lightgbm")), p


def test_b6_b7_row_failures_kept_and_counted(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setattr(embed_mod, "TEST_DURATIONS", tmp_path / "splits" / "nsa_test_durations.csv")
    info = make_repo(tmp_path, inner=(32, 32), hold=(4, 4), n_test=8, bad=(3,))  # 3 of 64 inner = 4.7%
    fails = {2: "raise", 5: "nan"}  # forward calls 2 and 5 (bad row 3 never reaches the model)
    assert _run(mod, tmp_path, loader=lambda d: FakeSpectra(fail_calls=fails)) == 0
    raw = pd.read_csv(tmp_path / "outputs" / "spectra" / "spectra_aasist_raw.csv")
    raw["flag"] = raw["flag"].fillna("")
    assert len(raw) == info["n_fold"] + info["n_test"]
    assert raw.flag.value_counts().to_dict() == {"": len(raw) - 3, "decode_error": 1, "model_error": 2}
    assert raw.loc[raw.flag != "", ["synth_logit", "logit"]].isna().all().all()
    pub = pd.read_csv(tmp_path / "outputs" / "detector_scores" / "spectra_aasist.csv")
    assert pub.logit.isna().sum() == 3 and pub.score.isna().sum() == 3  # C7: rows kept, NaN
    meta = json.loads(next((tmp_path / "models").glob("m3_spectra_*/meta.json")).read_text())
    assert meta["failures"]["decode_error"] == 1 and meta["failures"]["model_error"] == 2
    assert meta["inner"]["n_dropped"] + meta["holdout"]["n_dropped"] + meta["test"]["n"] - meta["test"]["n_used"] == 3


# --- D: metrics and readouts ----------------------------------------------------------------


def test_d1_d2_d3_by_group():
    dv = pd.DataFrame({"generator": ["bonafide", "bonafide", "g1", "g1", "g2"],
                       "source": ["src_a", "src_b", "diffssd", "diffssd", "diffssd"]})  # fmt: skip
    y, s = np.array([0, 0, 1, 1, 1]), np.array([0.2, 0.8, 0.9, 0.1, 0.7])
    per_gen, per_src = by_group(dv, y, s, 0.3)
    assert set(per_gen) == {"g1", "g2"} and set(per_src) == {"src_a", "src_b"}
    assert per_gen["g2"]["min_dcf"] == round(min_cost(np.array([0, 0, 1]), np.array([0.2, 0.8, 0.7]), 0.3), 4)
    assert per_src["src_a"]["min_dcf"] == round(min_cost(np.array([0, 1, 1, 1]), np.array([0.2, 0.9, 0.1, 0.7]), 0.3), 4)


def test_d4_d5_holdout_and_platt_isolated_from_other_rows():
    mod = _load_script()
    rng = np.random.default_rng(0)
    n = 40
    sc = pd.DataFrame({"split": ["inner_oof"] * n + ["holdout"] * n + ["test"] * 10,
                       "label": (["spoof", "bonafide"] * n) + [np.nan] * 10,
                       "generator": (["g1", "bonafide"] * n) + [np.nan] * 10,
                       "source": (["diffssd", "src_a"] * n) + [np.nan] * 10})  # fmt: skip
    sc["synth_logit"] = np.where(sc.label == "spoof", 3, -3) + rng.normal(size=len(sc))
    sc["logit_bonafide"] = -sc.synth_logit / 2
    sc["score"] = sigmoid(sc.synth_logit)
    sc["peak"] = rng.uniform(0.2, 1.0, len(sc))
    a = mod.split_readouts(sc)
    sc2 = sc.copy()
    sc2.loc[sc2.split != "holdout", "synth_logit"] += rng.normal(size=(sc2.split != "holdout").sum())
    b = mod.split_readouts(sc2)
    assert a["holdout"] == b["holdout"] and a["platt_diagnostic"] != b["platt_diagnostic"]
    sc3 = sc.copy()
    sc3.loc[sc3.split == "holdout", "synth_logit"] *= -1
    c = mod.split_readouts(sc3)
    assert c["platt_diagnostic"] == a["platt_diagnostic"] and c["holdout"] != a["holdout"]


def test_d8_one_class_and_empty_are_skipped():
    for fn in (masked_report, direction_check, sweep_thresholds, platt_diagnostic):
        assert fn([1, 1, 1], [0.1, 0.2, 0.3])["status"] == "skipped_one_class"
        assert fn([0, 1], [np.nan, np.nan])["status"] == "empty"
    assert direction_check([1, 1], [0.1, 0.2])["ok"] is None


def test_d9_readout_mode_and_threshold_sweep(repo):
    root, info = repo
    mod = _load_script()
    assert _run(mod, root, "--limit", "6", "--test-limit", "2", "--pad-modes", "zero", "--name", "p") == 0
    pdir = next((root / "outputs" / "spectra" / "pilot").iterdir())
    st = info["folds"][["path", "label", "generator", "source"]]
    st.to_csv(root / "stress.csv", index=False)
    assert _run(mod, root, "--manifest", "stress.csv", "--name", "st", "--pad-mode", "zero") == 0
    out = root / "ro.json"
    assert _run(mod, root, "--readout", f"inner={pdir / 'scores_zero.csv'}",
                f"stress=itw={root / 'outputs' / 'spectra' / 'st' / 'scores.csv'}", "--out", str(out)) == 0  # fmt: skip
    ro = json.loads(out.read_text())
    inner = pd.read_csv(pdir / "scores_zero.csv")
    inner = inner[inner.split == "inner_oof"]
    thr = sweep_thresholds((inner.label == "spoof").astype(int), inner.synth_logit)
    r = ro["stress"]["itw"]
    assert r["provisional"] and r["inner_thresholds"]["thr_dcf"] == thr["thr_dcf"]
    stf = pd.read_csv(root / "outputs" / "spectra" / "st" / "scores.csv")
    y, s = (stf.label == "spoof").to_numpy(), stf.synth_logit.to_numpy()
    t = thr["thr_dcf"]
    assert r["at_inner_thresholds"]["thr_dcf"]["p_fa"] == round(float(np.mean(s[~y] >= t)), 4)
    assert r["at_inner_thresholds"]["thr_dcf"]["p_miss"] == round(float(np.mean(s[y] < t)), 4)
    # a sweep that never beats the constant decision: None plus a flag, and no Infinity in JSON
    # M1 disagreement on the test rows plus merge into a meta.json (D9b)
    assert _run(mod, root, "--pad-mode", "zero") == 0
    meta_path = next((root / "models").glob("m3_spectra_*/meta.json"))
    raw = root / "outputs" / "spectra" / "spectra_aasist_raw.csv"
    tsv = root / "m1.tsv"
    test = pd.read_csv(raw); test = test[test.split == "test"]
    pd.DataFrame({"filename": [Path(p).name for p in test.path],
                  "cm-score": [0.9, 0.1, 0.9, 0.9]}).to_csv(tsv, sep="\t", index=False)  # fmt: skip
    assert _run(mod, root, "--readout", f"inner={raw}", f"test={raw}", f"m1_tsv={tsv}",
                "--merge-into", str(meta_path)) == 0  # fmt: skip
    merged = json.loads(meta_path.read_text())
    d = merged["readout"]["m1_disagreement"]
    assert d["n_joined"] == 4 and d["both_synthetic"] + d["both_real"] + d["m3_only_synthetic"] + d["m1_only_synthetic"] == 4
    assert merged["caveats"]["inner_rows_possibly_in_sample"] is True
    deg = sweep_thresholds([0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2])
    assert deg["thr_dcf"] is None and deg["thr_dcf_flag"] and "Infinity" not in json.dumps(deg)
    assert stress_readout([0, 1, 0, 1], [0.9, 0.1, 0.8, 0.2], y, s)["at_inner_thresholds"]["thr_dcf"] is None


def test_d11_failures_in_every_split_keep_every_readout(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setattr(embed_mod, "TEST_DURATIONS", tmp_path / "splits" / "nsa_test_durations.csv")
    # 64 inner (rows 0-63), 20 holdout (64-83), 20 test (84-103); one bad file in each split (<= 5%)
    info = make_repo(tmp_path, inner=(32, 32), hold=(10, 10), n_test=20, bad=(5, 70, 90))
    assert _run(mod, tmp_path, "--pad-mode", "zero") == 0
    meta = json.loads(next((tmp_path / "models").glob("m3_spectra_*/meta.json")).read_text())
    assert meta["failures"]["by_split"] == {
        "inner_oof": {"n": 64, "not_ok": 1, "rate": round(1 / 64, 4)},
        "holdout": {"n": 20, "not_ok": 1, "rate": 0.05},
        "test": {"n": 20, "not_ok": 1, "rate": 0.05}}
    assert meta["inner"]["n_dropped"] == 1 and meta["inner"]["n_used"] == 63 and meta["inner"]["status"] == "ok"
    assert meta["holdout"]["n_dropped"] == 1 and meta["holdout"]["n_used"] == 19 and "min_dcf" in meta["holdout"]
    assert meta["test"]["n"] == 20 and meta["test"]["n_used"] == 19
    for k in ("direction", "thresholds", "platt_diagnostic"):
        assert meta[k]["n_dropped"] == 1, k
    assert meta["failures"]["fold_rows_not_ok_by_generator"]  # per-generator coverage recorded
    # stress set with two bad files out of 40 (5%): manifest mode and the readout both keep going
    st = info["folds"][["path", "label", "generator", "source"]].iloc[10:50].copy()  # no bad row inside
    for j in (7, 8):
        Path(st.path.iloc[j]).write_bytes(b"garbage")
    st.to_csv(tmp_path / "stress.csv", index=False)
    assert _run(mod, tmp_path, "--manifest", "stress.csv", "--name", "st", "--pad-mode", "zero") == 0
    sm = json.loads((tmp_path / "outputs" / "spectra" / "st" / "meta.json").read_text())
    assert sm["stress"]["n_dropped"] == 2 and sm["stress"]["status"] == "ok"
    raw = tmp_path / "outputs" / "spectra" / "spectra_aasist_raw.csv"
    assert _run(mod, tmp_path, "--readout", f"inner={raw}",
                f"stress=itw={tmp_path / 'outputs' / 'spectra' / 'st' / 'scores.csv'}", "--final") == 0  # fmt: skip


def test_c14b_failure_late_in_the_run_publishes_nothing(repo):
    """Publication is the last step: a run that trips the gate on its final chunk leaves the run
    directory (raw + gate meta) and no fusion file, no sidecar, no raw companion."""
    root, info = repo
    mod = _load_script()
    n = info["n_fold"] + info["n_test"]  # 14 rows, chunk 2: the last chunk is the last two test rows
    fails = {n - 1: "raise", n: "raise"}
    assert _run(mod, root, "--pad-mode", "zero", loader=lambda d: FakeSpectra(fail_calls=fails)) == 3
    assert not (root / "outputs" / "detector_scores").exists()
    assert not (root / "outputs" / "spectra" / "spectra_aasist_raw.csv").exists()
    run_dir = next((root / "models").glob("m3_spectra_*"))
    assert (run_dir / "raw.csv").exists() and (run_dir / "meta.json").exists()
    assert not (run_dir / "fusion.csv").exists()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["gate"]["failure_rate"] == "failed" and "test" in meta["gate"]["splits_over_5pct"]
    assert list(pd.read_csv(run_dir / "raw.csv").columns[:5]) == BASE_HEADER.split(",")  # mode frame, not wide


def test_d10_level_measurement_recorded(repo):
    root, _ = repo
    mod = _load_script()
    assert _run(mod, root, "--pad-mode", "zero") == 0
    meta = json.loads(next((root / "models").glob("m3_spectra_*/meta.json")).read_text())
    lv = meta["level_shortcut"]["within_class_spearman_peak_vs_synth_logit"]
    assert set(lv) == {"bonafide", "spoof"} and all(v is None or "spearman" in v for v in lv.values())
    assert meta["failures"]["by_split"]["test"]["rate"] == 0.0


# --- W: real weights (CPU) ------------------------------------------------------------------


@pytest.mark.needs_weights
@pytest.mark.slow
def test_w1_real_model_all_modes_finite():
    from hearsay.spectra import WEIGHTS, load_spectra

    if not (WEIGHTS / "model.safetensors").exists():
        pytest.skip("weights not present")
    model = load_spectra("cpu")
    short, long = _tone(2.0, 440, amp=0.3), _tone(6.0, 440, amp=0.3)
    outs = {m: spectra_logits(model, short, pad_mode=m) for m in PAD_MODES}
    assert all(o.shape == (2,) and np.all(np.isfinite(o)) for o in outs.values())
    assert not (np.allclose(outs["repeat"], outs["zero"]) and np.allclose(outs["zero"], outs["whole"]))
    assert spectra_logits(model, long, pad_mode="whole").shape == (2,)
