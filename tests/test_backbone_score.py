"""V2-53: the scorer's input path, forward, export contract and metrics.

Stages C (I1-I8), D (F1-F11), E (E1-E6) and F (M1-M11) of
`docs/specs/2026-10-11_v2-backbone.md`.

Hermetic throughout: the scorer takes its model loader as an argument
(`run(argv, loader=...)`, the `scripts/score_spectra.py:477` pattern) and every path through
`--repo-root`, so a fake detector and a fake repo under `tmp_path` exercise the whole pipeline
with no weights, no network and no real audio.

The trap these tests exist for is in `E1`. Measured on the real exports: a scorer that
re-spells its paths loses **20,000 of 24,671 rows (81.1%)** on the join, because
`data/librispeech`, `data/ljspeech` and `data/nsa/DiffSSD` are symlinks and the repo's
normaliser resolves relative paths while passing absolute ones through. In-the-Wild survives
3,000/3,000, so every ITW readout looks healthy while the whole in-sample arm is gone.
"""

from __future__ import annotations

import importlib.util
import json
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from hearsay.antideepfake import receptive_field
from hearsay.antideepfake_eval import (
    brief,
    cluster_bootstrap,
    package03,
    package05,
    per_group,
    readings,
    refuse_degenerate,
)
from hearsay.metrics import sponsor_min_dcf

REPO = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "v2_backbone_score", REPO / "scripts" / "v2_backbone_score.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _script()


# --- a fake detector and a fake repo --------------------------------------------------------


class FakeCfg:
    conv_kernel = (10, 3, 3, 3, 3, 2, 2)
    conv_stride = (5, 2, 2, 2, 2, 2, 2)


class FakeW2V(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeCfg()


class FakeDetector(nn.Module):
    """Returns `[fake, real]` logits driven by the input's mean, so direction is controllable.

    `gain=-1` inverts the relationship, which is how the direction-gate test makes the gate
    fail without touching the gate.
    """

    def __init__(self, gain: float = 1.0, bad_shape: bool = False, non_finite: bool = False):
        super().__init__()
        self.wav2vec2 = FakeW2V()
        self.gain, self.bad_shape, self.non_finite = gain, bad_shape, non_finite

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bad_shape:
            return torch.zeros(1, 3)
        if self.non_finite:
            return torch.full((1, 2), float("nan"))
        # Keyed on kurtosis, which is **scale-invariant**. The scorer standardises every clip to
        # zero mean and unit variance before the forward, so a marker in amplitude (a DC offset,
        # a gain change) is erased before the model sees it -- the fake must respond to waveform
        # shape instead. A tone has kurtosis ~1.5, gaussian noise ~3.
        t = torch.as_tensor(x, dtype=torch.float64).flatten()
        t = (t - t.mean()) / (t.std() + 1e-12)
        kurt = float((t**4).mean())
        v = (3.0 - kurt) * self.gain
        return torch.tensor([[v, -v]], dtype=torch.float32)


def write_wav(path: Path, seconds: float = 1.0, spoof: bool = False, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sr)
    rng = np.random.default_rng(int(path.stem) if path.stem.isdigit() else 0)
    if spoof:
        # a tone: low kurtosis, and the shape survives per-clip standardisation
        f = 300.0 + 40.0 * (int(path.stem) if path.stem.isdigit() else 0)
        x = 0.4 * np.sin(2 * np.pi * f * np.arange(n) / sr)
    else:
        x = rng.normal(0, 0.1, n)
    pcm = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())


@pytest.fixture
def repo(tmp_path):
    """A fake repo with an ITW-style manifest: relative paths, both classes, 54 speakers."""
    root = tmp_path / "repo"
    (root / "outputs" / "manifests").mkdir(parents=True)
    (root / "outputs" / "v2" / "backbone").mkdir(parents=True)
    rows = []
    for i in range(24):
        spoof = i % 3 == 0
        rel = f"data/in_the_wild/release_in_the_wild/{i}.wav"
        write_wav(root / rel, seconds=1.0 + (i % 4) * 0.5, spoof=spoof)
        rows.append({
            "path": rel, "label": "spoof" if spoof else "bonafide",
            "generator": "itw_spoof" if spoof else "bonafide",
            "speaker": f"spk{i % 6}", "source": "in_the_wild",
        })  # fmt: skip
    pd.DataFrame(rows).to_csv(root / "outputs" / "manifests" / "itw_stress.csv", index=False)
    return root


def run_scorer(mod, repo, loader, extra=()):
    argv = ["--model", str(repo / "fakeweights"), "--population", "itw",
            "--repo-root", str(repo), "--device", "cpu", "--no-lock", *extra]  # fmt: skip
    (repo / "fakeweights").mkdir(exist_ok=True)
    (repo / "fakeweights" / "model.safetensors").write_bytes(b"not a real checkpoint")
    (repo / "fakeweights" / "config.json").write_text("{}")
    mod.OUT = repo / "outputs" / "v2" / "backbone"
    return mod.run(argv, loader=lambda *a, **k: loader)


# === Stage C: input path ====================================================================


def test_i1_i2_our_view_is_prepare_segment_and_band_match_is_load_bearing(mod):
    """I1/I2: the primary view is the shared band-matched path, and the band match matters."""
    from hearsay.embed import prepare_segment

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, 16000).astype(np.float32)
    views = mod.prepare_views(x, ("ours", "native"))
    assert np.array_equal(views["ours"], prepare_segment(x, band_match=True))
    assert not np.array_equal(prepare_segment(x, band_match=True), prepare_segment(x, band_match=False))


def test_i3_the_operation_order_is_pinned_byte_identically(mod):
    """I3: M5's band-limit-before-trim bug showed up as a 0.279 logit error, which a
    probability-level check would have missed. The order is pinned, not described."""
    from hearsay.embed import prepare_segment, trim_silence
    from hearsay.handcrafted import band_limit

    rng = np.random.default_rng(1)
    x = np.concatenate([np.zeros(4000), rng.normal(0, 0.1, 16000), np.zeros(4000)]).astype(np.float32)
    right = prepare_segment(x, band_match=True)
    wrong = trim_silence(band_limit(x))  # same ops, and this IS the shipped order
    assert np.allclose(right, wrong[: right.size], atol=0)
    inverted = band_limit(trim_silence(x))
    assert not np.array_equal(right, inverted[: right.size]), "the two orders must differ"


def test_i8_the_two_views_are_recorded_separately_and_differ(mod, repo):
    """I8: native and ours are both recorded per clip, in their own columns."""
    rc = run_scorer(mod, repo, FakeDetector())
    assert rc == 0
    raw = pd.read_csv(repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.raw.csv")
    assert {"logit_ours", "logit_native"} <= set(raw.columns)
    assert not np.allclose(raw.logit_ours, raw.logit_native), "a band match must change the score"


def test_i6_clips_below_the_receptive_field_are_flagged_not_crashed(mod, repo):
    """I6: a very short clip is a legitimate question with a named answer."""
    need = receptive_field(FakeCfg.conv_kernel, FakeCfg.conv_stride)
    short = np.random.default_rng(0).normal(0, 0.1, need - 1).astype(np.float32)
    out = mod.score_one(FakeDetector(), {"ours": short})
    assert out["flag"] == f"below_receptive_field_{need}"


# === Stage D: forward and scoring ===========================================================


def test_f1_a_wrong_output_shape_raises(mod):
    """F1: the exact M3 Codex round-2 finding -- two finite rows for one window were averaged
    silently. Anything but (1, 2) must raise."""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.1, 16000).astype(np.float32)
    with pytest.raises(RuntimeError, match=r"expected \(1, 2\)"):
        mod.score_one(FakeDetector(bad_shape=True), {"ours": x})


def test_f2_non_finite_logits_are_flagged_and_the_row_is_kept(mod):
    """F2: a bad row is flagged and counted, never dropped."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.1, 16000).astype(np.float32)
    assert mod.score_one(FakeDetector(non_finite=True), {"ours": x})["flag"] == "non_finite_logits"


def test_f3_f4_all_zero_audio_is_flagged_not_scored(mod, repo):
    """F3/F4: this repo's documented failure mode is a transient decode error becoming 1 s of
    zeros that then trains and scores silently. Zeros are a flag, not a score."""
    assert mod.score_one(FakeDetector(), {"ours": np.zeros(16000, np.float32)})["flag"] == "silent_or_zeros"
    rc = run_scorer(mod, repo, FakeDetector())
    assert rc == 0
    meta = json.loads((repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.meta.json").read_text())
    assert set(meta["flags"]) == {""}, f"unexpected flags: {meta['flags']}"


def test_f9_the_direction_gate_blocks_the_export(mod, repo):
    """F9: polarity is verified on labelled rows, never assumed, and nothing is flipped
    silently -- the M3 precedent. Exit 4, and no export written."""
    rc = run_scorer(mod, repo, FakeDetector(gain=-1.0))
    assert rc == 4
    assert not (repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.csv").exists()
    meta = json.loads((repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.meta.json").read_text())
    assert meta["direction"]["ok"] is False


def test_f9b_a_single_class_population_also_blocks(mod, repo):
    """F9: `direction_ok` returns `ok: None` on one class, and the caller uses `is not True`,
    so `None` must block too. That subtlety is why M3's gate reads the way it does."""
    man = repo / "outputs" / "manifests" / "itw_stress.csv"
    d = pd.read_csv(man)
    d["label"] = "bonafide"
    d.to_csv(man, index=False)
    assert run_scorer(mod, repo, FakeDetector()) == 4


def test_f10_scoring_is_deterministic(mod, repo):
    """F10: the same input twice must give the same number."""
    assert run_scorer(mod, repo, FakeDetector()) == 0
    first = pd.read_csv(repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.csv")
    (repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.csv").unlink()
    assert run_scorer(mod, repo, FakeDetector()) == 0
    second = pd.read_csv(repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.csv")
    pd.testing.assert_frame_equal(first, second)


# === Stage E: export and join ===============================================================


def test_e1_paths_are_emitted_verbatim_so_the_join_survives(mod, repo):
    """E1, the finding that would have cost the lane its in-sample arm.

    Measured on the real exports: a scorer that re-spells its paths loses 20,000 of 24,671 rows
    (81.1%) silently, because three `data/` directories are symlinks and the repo's normaliser
    resolves relative paths while passing absolute ones through. ITW survives intact, so the
    damage is invisible in exactly the readout one would check.
    """
    assert run_scorer(mod, repo, FakeDetector()) == 0
    exported = pd.read_csv(repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.csv")
    manifest = pd.read_csv(repo / "outputs" / "manifests" / "itw_stress.csv")
    assert list(exported.path) == list(manifest.path), "paths must be the manifest's own spelling"
    assert set(exported.path) == set(manifest.path)
    assert len(exported) == len(manifest), "full coverage, no silent drops"


def test_e2_duplicate_paths_are_refused(mod, repo):
    """E2: the `fusion_frame` precedent."""
    man = repo / "outputs" / "manifests" / "itw_stress.csv"
    d = pd.read_csv(man)
    pd.concat([d, d.head(1)]).to_csv(man, index=False)
    with pytest.raises(SystemExit) as e:
        run_scorer(mod, repo, FakeDetector())
    assert e.value.code == 2


def test_e3_split_labels_are_from_the_declared_contract(mod, repo):
    """E3: `split` must be one of the four the fusion contract allows."""
    assert run_scorer(mod, repo, FakeDetector()) == 0
    exported = pd.read_csv(repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.csv")
    assert set(exported.split) <= {"inner_oof", "holdout", "test", "itw"}
    assert set(exported.split) == {"itw"}
    for pop, (_, split, _) in mod.POPULATIONS.items():
        assert split in {"inner_oof", "holdout", "test", "itw"}, pop


def test_e4_e6_logit_rises_with_synthetic_and_score_is_its_sigmoid(mod, repo):
    """E4/E6: the contract's direction, and `D10` -- no calibration on the export."""
    assert run_scorer(mod, repo, FakeDetector()) == 0
    exported = pd.read_csv(repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.csv")
    man = pd.read_csv(repo / "outputs" / "manifests" / "itw_stress.csv")
    j = exported.merge(man[["path", "label"]], on="path")
    assert j[j.label == "spoof"].logit.median() > j[j.label == "bonafide"].logit.median()
    assert np.allclose(exported.score, 1 / (1 + np.exp(-exported.logit)), atol=1e-9)


def test_e5_the_scorer_writes_only_under_its_own_lane_dir(mod, repo):
    """E5: every artifact must land under `outputs/v2/backbone/`."""
    assert run_scorer(mod, repo, FakeDetector()) == 0
    lane = repo / "outputs" / "v2" / "backbone"
    produced = {p.name for p in lane.iterdir()}
    assert any(n.endswith("__itw.csv") for n in produced), f"export not in the lane dir: {produced}"
    assert not (repo / "outputs" / "detector_scores").exists()

    # the escape this test previously missed: `--model` as a path made `out_name` path-like,
    # and `OUT / "/abs/name"` resolves to `/abs/name`, so every artifact landed outside the
    # lane dir while the run reported success.
    with pytest.raises(ValueError, match="outside the lane dir"):
        mod.lane_path("/tmp/escaped.csv")
    with pytest.raises(ValueError, match="outside the lane dir"):
        mod.lane_path("../escaped.csv")
    assert mod.lane_path("fine.csv").parent == mod.OUT.resolve()


def test_e_pilot_publishes_nothing(mod, repo):
    """A pilot writes only pilot-named artifacts, even when a gate fails."""
    assert run_scorer(mod, repo, FakeDetector(), extra=["--limit", "8"]) == 0
    lane = repo / "outputs" / "v2" / "backbone"
    assert not (lane / "fakeweights__itw.csv").exists()
    assert (lane / "fakeweights__itw.pilot.raw.csv").exists()


# === Stage F: metrics =======================================================================


def test_m2_package05_is_exactly_the_sponsors_flipped_code():
    """M2: the selection metric, pinned against the sponsor's own implementation."""
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(700, int), np.ones(300, int)]
    s = np.r_[rng.normal(-1, 1, 700), rng.normal(1, 1, 300)]
    assert abs(package05(y, s) - sponsor_min_dcf(y, s, flip=True)) < 2e-16


def test_m2b_the_three_readings_are_distinct_and_package03_is_the_winners_formula():
    """M2: package-0.3 has no helper in this repo, so its arithmetic is pinned here.

    Verified against the winning team's stated formula, FPR + 1.714 * FNR.
    """
    from sklearn.metrics import roc_curve

    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(600, int), np.ones(400, int)]
    s = np.r_[rng.normal(-0.8, 1, 600), rng.normal(0.8, 1, 400)]
    fpr, tpr, _ = roc_curve(y, s)
    theirs = float(np.min(fpr + (4 * 0.3 / 0.7) * (1 - tpr)))
    assert abs(package03(y, s) - theirs) < 1e-12
    assert len({round(package05(y, s), 6), round(package03(y, s), 6), round(brief(y, s), 6)}) == 3


def test_m11_every_reading_carries_eer():
    """Rule 5: EER is reported beside every reading. Nothing asserted this before."""
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(300, int), np.ones(200, int)]
    s = np.r_[rng.normal(-1, 1, 300), rng.normal(1, 1, 200)]
    r = readings(y, s)
    assert {"package05", "package03", "brief", "eer"} <= set(r)
    assert 0.0 <= r["eer"] <= 1.0


def test_m_min_cost_clamps_at_one_and_we_do_not_read_it_as_break_even():
    """`min_cost` returns exactly 1.0 for an inverted or random detector. That means "no better
    than the constant decision", not "break-even", and `refuse_degenerate` is what acts on it."""
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(300, int), np.ones(200, int)]
    s = np.r_[rng.normal(-1, 1, 300), rng.normal(1, 1, 200)]
    assert package05(y, -s * 50) == 1.0
    with pytest.raises(ValueError, match="AUC"):
        refuse_degenerate(y, -s * 50)


def test_m_degenerate_candidates_are_refused(mod):
    """H10: a constant or worse-than-chance candidate is refused, not reported with its
    flattering in-sample number."""
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    with pytest.raises(ValueError, match="constant score"):
        refuse_degenerate(y, np.full(100, 0.5))
    with pytest.raises(ValueError, match="non-finite"):
        refuse_degenerate(y, np.full(100, np.nan))


def test_m7_the_asvspoof_eval_partition_is_barred_from_every_population(mod):
    """M7: LA eval (A07-A19) is the only partition carrying unseen attacks, so reading it spends
    it. `D9`'s control is train/dev only, asserted here against the shipped manifest."""
    for name, (rel, _, _) in mod.POPULATIONS.items():
        assert "asv19_eval" not in rel, f"{name} points at the barred eval partition"
    ctrl = REPO / "outputs" / "v2" / "backbone" / "asv19_control.csv"
    if not ctrl.exists():
        pytest.skip("control manifest not built yet")
    d = pd.read_csv(ctrl)
    assert not d.path.str.contains("ASVspoof2019_LA_eval").any()
    assert set(d.partition) == {"train", "dev"}
    assert set(d.generator) <= {"bonafide", "A01", "A02", "A03", "A04", "A05", "A06"}


def test_m3_the_in_sample_arm_names_its_basis():
    """M3: every inner number states which population it is on. The pre-declared sample records
    its basis, its strata and its seed, so the number can be read later."""
    decl = REPO / "outputs" / "v2" / "backbone" / "sample_predeclared.json"
    if not decl.exists():
        pytest.skip("sample not declared yet")
    d = json.loads(decl.read_text())
    assert "16,142" in d["population"]
    assert d["n"] == len(d["paths"]) == 3000
    assert d["strata"] and d["seed"] == 0
    assert len(set(d["paths"])) == 3000, "the declared sample must be duplicate-free"


def test_m10_an_interval_accompanies_an_absolute_number():
    """M10: AC10a asks for an interval on every headline number, and the gain bootstrap does
    not provide one -- it bootstraps a difference."""
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(300, int), np.ones(200, int)]
    s = np.r_[rng.normal(-1, 1, 300), rng.normal(1, 1, 200)]
    groups = np.r_[np.arange(300) % 20, np.arange(200) % 20]
    b = cluster_bootstrap(y, s, groups, cost=package05, n=100, seed=0)
    assert b["status"] == "ok" and b["paired"] is False
    assert b["p5"] <= b["median"] <= b["p95"] and b["width90"] >= 0


def test_h3_the_bootstrap_guards_the_existing_one_lacks():
    """H3: `boot_gain` has the right shape but no NaN-group guard and no single-class skip, and
    `roc_curve` returns NaN on a single-class draw rather than raising."""
    y = np.r_[np.zeros(20, int), np.ones(20, int)]
    s = np.r_[np.zeros(20), np.ones(20)]
    with pytest.raises(ValueError, match="missing speaker/group"):
        cluster_bootstrap(y, s, [None] * 40, n=5)
    with pytest.raises(ValueError, match="missing speaker/group"):
        cluster_bootstrap(y, s, [""] * 40, n=5)
    # one cluster per class: most draws are single-class and must be counted, not averaged in
    out = cluster_bootstrap(y, s, ["a"] * 20 + ["b"] * 20, cost=package05, n=50, seed=0)
    assert out.get("skipped_single_class", 0) > 0


def test_h4_paired_draws_are_identical_for_both_candidates():
    """H4: candidate and reference must see the same resamples, or the difference is noise."""
    rng = np.random.default_rng(5)
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    a = np.r_[rng.normal(-1, 1, 200), rng.normal(1, 1, 200)]
    groups = np.r_[np.arange(200) % 25, np.arange(200) % 25]
    same = cluster_bootstrap(y, a, groups, cost=package05, n=80, seed=0, s_other=a)
    assert abs(same["median"]) < 1e-12, "a candidate against itself must give a zero difference"
    assert abs(same["p5"]) < 1e-12 and abs(same["p95"]) < 1e-12


def test_h6_per_group_works_at_package05_unlike_by_group():
    """H6: `hearsay.metrics.by_group` hardcodes the brief costs, so a package-0.5 per-generator
    table needs its own implementation. Thin groups are reported but excluded from the worst."""
    rng = np.random.default_rng(6)
    n = 120
    frame = pd.DataFrame({"generator": ["g1"] * 50 + ["g2"] * 50 + ["tiny"] * 20})
    y = np.r_[np.zeros(25, int), np.ones(25, int)] * 1
    y = np.r_[y, y, np.r_[np.zeros(10, int), np.ones(10, int)]]
    s = rng.normal(0, 1, n) + y
    out = per_group(frame, y, s, "generator", cost=package05, min_rows=30)
    assert set(out["per_group"]) == {"g1", "g2", "tiny"}
    assert out["excluded_below_min_rows"] == ["tiny"]
    assert out["worst_group"] in {"g1", "g2"}
    assert "worst_cost" in out and "mean_of_groups" in out


def test_f_input_matches_the_models_dtype_not_only_its_device(mod):
    """The input must follow the model's dtype as well as its device.

    Moving it to the right device while leaving it float32 against an fp16 model fails with
    `Input type (float) and bias type (c10::Half) should be the same` -- which is what happened
    to **all 3,000 rows** of the 2B run after it was switched to fp16 to stop the machine
    swapping. The flag recorded only `error:RuntimeError`, so the cause was invisible until it
    was reproduced by hand; flags now carry the message too.
    """
    x = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    assert mod.standardize(x).dtype == torch.float32
    assert mod.standardize(x, dtype=torch.float16).dtype == torch.float16

    class HalfDetector(FakeDetector):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1, dtype=torch.float16))

        def forward(self, t):
            if t.dtype != self.w.dtype:
                raise RuntimeError(
                    f"Input type ({t.dtype}) and bias type ({self.w.dtype}) should be the same"
                )
            return super().forward(t)

    m = HalfDetector()
    _, dt = mod.model_place(m)
    assert dt == torch.float16, "the lookup must report the dtype, not only the device"
    out = mod.score_one(m, {"ours": x})
    assert out["flag"] == "", f"an fp16 model must score an fp32 clip: {out}"


def test_f_a_failed_row_records_why_not_just_the_exception_type(mod, repo):
    """`error:RuntimeError` across every row is not a diagnosis."""
    class Boom(FakeDetector):
        def forward(self, t):
            raise RuntimeError("a very specific cause")

    rc = run_scorer(mod, repo, Boom())
    assert rc == 3  # the failure gate refuses to publish
    raw = pd.read_csv(repo / "outputs" / "v2" / "backbone" / "fakeweights__itw.raw.csv")
    assert raw.flag.str.contains("a very specific cause").all(), raw.flag.unique()[:3]
