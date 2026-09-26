"""hearsay.pipeline and scripts/run_pipeline.py, hermetically: toy fusion constants, fake
scorers and fake detectors (no weights, no data). The exact-parity checks against the exported
score files and the logged TSVs run only when those gitignored files are present."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hearsay import SR
from hearsay.detectors.base import ClipContext, DetectorResult
from hearsay.detectors.speech_gate import BLOCK_TOP
from hearsay.metrics import sigmoid
from hearsay.pipeline import (
    DEEP,
    DETECTOR_ORDER,
    RULES,
    FusionConstants,
    analyze_clip,
    detector_entry,
    refuse,
    routing_log,
    verdict_for,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_pipeline.py"
CONSTANTS = REPO / "models" / "fusion_v0" / "constants.json"
EXPORTS = {n: REPO / "outputs" / "detector_scores" / f"{n}.csv" for n in DETECTOR_ORDER}
TSV = {
    "zmean": REPO / "submissions" / "20260926-0602_M4_fusion_zmean_m1bv3_hcv5_m3.tsv",
    "stack_nonlj": REPO / "submissions" / "20260926-0602_M4_fusion_stack_nonlj_m1bv3_hcv5_m3.tsv",
}


def _runner():
    spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- toy constants in scripts/fuse.py's file layout ----------------------------------------

TOY = {
    "detectors": ["m1b_v3", "handcrafted_v5", "spectra_aasist"],
    "pi_synth": 0.3,
    "standardize": {
        "m1b_v3": {"mean": 1.0, "std": 2.0, "inner_oof_sorted": [0.0, 1.0, 2.0]},
        "handcrafted_v5": {"mean": -1.5, "std": 5.0, "inner_oof_sorted": [-3.0, -1.5, 0.0]},
        "spectra_aasist": {"mean": 2.0, "std": 8.0, "inner_oof_sorted": [-6.0, 2.0, 10.0]},
    },
    "rules": {
        "stack_nonlj": {"weights": [2.0, 0.5, 7.0], "intercept": -0.5,
                        "platt": {"a": 1.75, "b": -1.9, "prior_shift": math.log(0.3 / 0.7)}},
        "zmean": {"platt": {"a": 11.7, "b": 2.8, "prior_shift": math.log(0.3 / 0.7)}},
        "alone:m1b_v3": {"platt": {"a": 6.8, "b": 1.9, "prior_shift": math.log(0.3 / 0.7)}},
    },
    "how": {"zmean": "mean of z over detectors"},
}  # fmt: skip
LOGITS = {"m1b_v3": -6.3, "handcrafted_v5": -1.33, "spectra_aasist": -7.07}


@pytest.fixture
def consts():
    return FusionConstants.from_dict(TOY, source="toy")


def _z(logits):
    return np.array([(logits[d] - TOY["standardize"][d]["mean"]) / TOY["standardize"][d]["std"]
                     for d in TOY["detectors"]])  # fmt: skip


def test_constants_parse_and_rules(consts):
    assert consts.detectors == tuple(DETECTOR_ORDER)
    assert consts.rules() == RULES
    assert consts.stack_weights == {"m1b_v3": 2.0, "handcrafted_v5": 0.5, "spectra_aasist": 7.0}
    assert consts.weights("zmean") == {d: pytest.approx(1 / 3) for d in DETECTOR_ORDER}
    with pytest.raises(ValueError):
        consts.weights("rankmean")


def test_constants_load_from_file(tmp_path):
    p = tmp_path / "constants.json"
    p.write_text(json.dumps(TOY))
    c = FusionConstants.load(p)
    assert c.source == str(p) and c.pi_synth == 0.3
    assert c.platt["zmean"]["a"] == 11.7


def test_constants_reject_mismatched_weights():
    bad = json.loads(json.dumps(TOY))
    bad["rules"]["stack_nonlj"]["weights"] = [1.0, 2.0]
    with pytest.raises(ValueError):
        FusionConstants.from_dict(bad)
    bad = json.loads(json.dumps(TOY))
    del bad["standardize"]["spectra_aasist"]
    with pytest.raises(ValueError):
        FusionConstants.from_dict(bad)


def test_zmean_is_fuse_py_arithmetic(consts):
    fo = consts.fuse(LOGITS, "zmean")
    z = _z(LOGITS)
    fused = float(np.mean(z))
    pl = TOY["rules"]["zmean"]["platt"]
    assert fo.fused == pytest.approx(fused, abs=1e-15)
    assert fo.p == pytest.approx(float(sigmoid(pl["a"] * fused + pl["b"] + pl["prior_shift"])), abs=1e-15)
    assert fo.imputed == () and fo.rule == "zmean"
    assert list(fo.inputs) == list(DETECTOR_ORDER)
    assert fo.z == {d: pytest.approx(v) for d, v in zip(DETECTOR_ORDER, z, strict=True)}


def test_stack_nonlj_is_fuse_py_arithmetic(consts):
    fo = consts.fuse(LOGITS, "stack_nonlj")
    z = _z(LOGITS)
    fused = float(z @ np.array([2.0, 0.5, 7.0]) - 0.5)  # LogisticRegression.decision_function
    pl = TOY["rules"]["stack_nonlj"]["platt"]
    assert fo.fused == pytest.approx(fused, abs=1e-15)
    assert fo.p == pytest.approx(float(sigmoid(pl["a"] * fused + pl["b"] + pl["prior_shift"])), abs=1e-15)
    assert fo.weights == consts.stack_weights


def test_missing_logit_is_imputed_at_the_inner_mean(consts):
    fo = consts.fuse({**LOGITS, "spectra_aasist": None}, "stack_nonlj")
    assert fo.imputed == ("spectra_aasist",) and fo.z["spectra_aasist"] == 0.0
    assert fo.inputs["spectra_aasist"] is None
    fo2 = consts.fuse({**LOGITS, "spectra_aasist": float("nan")}, "zmean")
    assert fo2.imputed == ("spectra_aasist",)
    with pytest.raises(ValueError):
        consts.fuse(LOGITS, "rankmean")  # no Platt map for it in the constants


# --- fake models and detectors -------------------------------------------------------------


class FakeDet:
    def __init__(self, name, score=0.5, features=None, evidence=None, fail=False):
        self.name, self.score, self.features, self.fail = name, score, features or {}, fail
        self.evidence = evidence or f"{name} says so"

    def applies(self, ctx):
        return True

    def run(self, ctx):
        if self.fail:
            raise RuntimeError("boom")
        return DetectorResult(self.name, self.score, self.evidence, self.features)


def fake_dets(is_speech=1.0, hc_fail=False, gate_fail=False):
    return [
        FakeDet("compression", 0.4, {"bw_hz": 7250.0}),
        FakeDet("container", 0.5, {"lossy": 0.0, "is_pcm_wav": 1.0, "ffmpeg_written": 1.0}),
        FakeDet("enf", 0.5, {"enf_present": 0.0, "enf_stable": 0.0}),
        FakeDet("handcrafted", 0.21, {"hc_logit": LOGITS["handcrafted_v5"], "voiced_frac": 0.8}, fail=hc_fail),
        FakeDet("speaker_drift", 0.5, {"drift": 0.0, "cos_min": 0.31}),
        FakeDet("speech_gate", 0.5, {"is_speech": is_speech, "voiced_frac": 0.81}, fail=gate_fail),
        FakeDet("splice", 0.5, {"n_seams": 0.0}),
    ]


class FakeModels:
    def __init__(self, m1=LOGITS["m1b_v3"], spectra=LOGITS["spectra_aasist"], dets=None,
                 m1_fail=False):  # fmt: skip
        self._m1, self._sp, self._dets, self.m1_fail = m1, spectra, dets, m1_fail
        self.calls = 0

    def m1_logit(self, x):
        self.calls += 1
        if self.m1_fail:
            raise RuntimeError("no backbone")
        return self._m1

    def spectra_logit(self, x):
        return self._sp

    def engineered(self):
        return self._dets if self._dets is not None else fake_dets()

    def version(self):
        return {"m1": "m1_toy", "handcrafted": "hc_toy", "spectra": "spectra_toy"}


def _ctx(seconds=3.0, seed=0):
    x = (0.1 * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)
    return ClipContext.from_array(x, path="clip.wav")


REQUIRED = {"filename", "duration_s", "probability_synthetic", "verdict", "is_speech",
            "default_answer_applied", "fusion", "detectors", "routing_log", "version"}  # fmt: skip


def test_analyze_response_shape_and_fusion(consts):
    doc = analyze_clip(_ctx(), FakeModels(), consts, rule="stack_nonlj")
    assert REQUIRED <= set(doc)
    assert doc["filename"] == "clip.wav" and doc["duration_s"] == 3.0
    assert doc["is_speech"] is True and doc["default_answer_applied"] is False
    fo = consts.fuse(LOGITS, "stack_nonlj")
    assert doc["probability_synthetic"] == pytest.approx(BLOCK_TOP + (1 - BLOCK_TOP) * fo.p)  # determinate map
    assert doc["verdict"] == "real"
    assert doc["fusion"]["rule"] == "stack_nonlj" and doc["fusion"]["inputs"] == fo.inputs
    assert doc["fusion"]["weights"] == consts.stack_weights and doc["fusion"]["imputed"] == []
    names = [d["name"] for d in doc["detectors"]]
    assert names == [d.name for d in fake_dets()] + list(DEEP)
    roles = {d["name"]: d["role"] for d in doc["detectors"]}
    assert roles["handcrafted"] == roles["m1b_v3"] == roles["spectra_aasist"] == "fused"
    assert roles["container"] == "routing" and roles["speech_gate"] == "gate"
    assert roles["enf"] == roles["splice"] == roles["speaker_drift"] == roles["compression"] == "evidence"
    for d in doc["detectors"]:
        assert d["status"] == "ok" and 0.0 <= d["score"] <= 1.0 and d["evidence"] and d["error"] is None
        assert d["seconds"] >= 0 and all(math.isfinite(v) for v in d["features"].values())
    deep = {d["name"]: d for d in doc["detectors"] if d["name"] in DEEP}
    assert deep["m1b_v3"]["features"]["logit"] == LOGITS["m1b_v3"]
    assert deep["m1b_v3"]["score"] == pytest.approx(float(sigmoid(LOGITS["m1b_v3"])))
    assert doc["version"]["models"]["m1"] == "m1_toy" and doc["version"]["rule"] == "stack_nonlj"
    json.dumps(doc)  # serializable as written


def test_routing_log_reads_the_detector_features(consts):
    doc = analyze_clip(_ctx(), FakeModels(), consts, rule="zmean")
    log = doc["routing_log"]
    heads = [line.split(":")[0] for line in log]
    assert heads == ["container", "compression", "enf", "splice", "speaker_drift", "speech_gate", "fusion"]
    assert "PCM WAV, not lossy, FFmpeg-written" in log[0] and "for evidence, not score" in log[0]
    assert "7250 Hz" in log[1] and "no mains hum" in log[2] and "no editing seams" in log[3]
    assert "one consistent voice" in log[4] and "never fused" in log[4]
    assert "is_speech=true" in log[5] and "not applied" in log[5]
    assert log[6] == "fusion: zmean over m1b_v3, handcrafted_v5, spectra_aasist"


def test_routing_log_lossy_seams_hum_and_drift():
    dets = [
        FakeDet("container", 0.5, {"lossy": 1.0, "is_pcm_wav": 0.0, "ffmpeg_written": 0.0}),
        FakeDet("enf", 0.6, {"enf_present": 1.0, "enf_stable": 0.0}),
        FakeDet("splice", 0.6, {"n_seams": 2.0}),
        FakeDet("speaker_drift", 0.6, {"drift": 1.0, "cos_min": -0.1}),
    ]
    items = [detector_entry(d.run(None), 0.0) for d in dets]
    fo = FusionConstants.from_dict(TOY).fuse(LOGITS, "zmean")
    log = routing_log(items, True, fo, decoded=True)
    assert "lossy codec" in log[0] and "compression forensics apply" in log[0]
    assert "discontinuous" in log[1] and "2 editing seam(s)" in log[2] and "voice drifts" in log[3]
    assert log[-2].startswith("speech_gate: missing")


def test_default_answer_applied_when_no_speech(consts):
    m = FakeModels(dets=fake_dets(is_speech=0.0))
    doc = analyze_clip(_ctx(), m, consts, rule="zmean")
    fo = consts.fuse(LOGITS, "zmean")
    assert doc["is_speech"] is False and doc["default_answer_applied"] is True
    assert 0.0 <= doc["probability_synthetic"] < BLOCK_TOP  # pinned block, below every scored file
    assert doc["fusion"]["p_fused"] == fo.p  # the fused value is still reported
    assert doc["verdict"] == "undetermined"
    assert any("is_speech=false" in line and "policy applied" in line for line in doc["routing_log"])
    ungated = analyze_clip(_ctx(), m, consts, rule="zmean", apply_gate=False)
    assert ungated["probability_synthetic"] == fo.p  # apply_gate=False: no policy, raw fused p
    assert ungated["default_answer_applied"] is False
    assert ungated["version"]["policy"] == "none"


def test_gate_error_does_not_gate(consts):
    doc = analyze_clip(_ctx(), FakeModels(dets=fake_dets(gate_fail=True)), consts)
    assert doc["is_speech"] is True and doc["default_answer_applied"] is False
    gate = next(d for d in doc["detectors"] if d["name"] == "speech_gate")
    assert gate["status"] == "error" and gate["score"] == 0.5 and "boom" in gate["error"]
    assert any(line.startswith("speech_gate: error") for line in doc["routing_log"])


def test_failed_fused_detector_is_imputed_and_logged(consts):
    doc = analyze_clip(_ctx(), FakeModels(dets=fake_dets(hc_fail=True), m1_fail=True), consts, "stack_nonlj")
    assert set(doc["fusion"]["imputed"]) == {"m1b_v3", "handcrafted_v5"}
    assert doc["fusion"]["inputs"]["m1b_v3"] is None and doc["fusion"]["z"]["m1b_v3"] == 0.0
    m1 = next(d for d in doc["detectors"] if d["name"] == "m1b_v3")
    assert m1["status"] == "error" and m1["score"] == 0.5 and "no backbone" in m1["error"]
    assert "imputed at the inner-fold mean: m1b_v3, handcrafted_v5" in doc["routing_log"][-1]
    assert 0.0 <= doc["probability_synthetic"] <= 1.0


def test_decode_failure_is_undetermined_with_the_default_answer(consts, tmp_path):
    bad = tmp_path / "not_audio.wav"
    bad.write_bytes(b"this is not a wav file")
    m = FakeModels()
    doc = analyze_clip(ClipContext(bad), m, consts, rule="zmean")
    assert doc["duration_s"] is None and doc["verdict"] == "undetermined"
    assert doc["is_speech"] is False and doc["default_answer_applied"] is True
    assert m.calls == 0  # no scorer ran on a clip that never decoded
    assert all(d["status"] == "error" for d in doc["detectors"])
    assert set(doc["fusion"]["imputed"]) == set(DETECTOR_ORDER)
    assert doc["routing_log"][0].startswith("decode:")
    assert 0.0 <= doc["probability_synthetic"] < BLOCK_TOP


def test_refuse_switches_rule_and_policy_without_models(consts):
    a = analyze_clip(_ctx(), FakeModels(), consts, rule="stack_nonlj")
    b = analyze_clip(_ctx(), FakeModels(), consts, rule="zmean")
    r = refuse(a, consts, "zmean")
    assert r["probability_synthetic"] == b["probability_synthetic"]
    assert r["fusion"] == b["fusion"] and r["routing_log"] == b["routing_log"]
    assert r["version"]["rule"] == "zmean" and r["detectors"] == a["detectors"]
    g = analyze_clip(_ctx(), FakeModels(dets=fake_dets(is_speech=0.0)), consts, rule="zmean")
    u = refuse(g, consts, "zmean", apply_gate=False)
    assert u["probability_synthetic"] == g["fusion"]["p_fused"] and u["default_answer_applied"] is False


def test_verdict_rule():
    assert verdict_for(0.5, True, True) == "synthetic" and verdict_for(0.49, True, True) == "real"
    assert verdict_for(0.9, False, True) == "undetermined" and verdict_for(0.1, True, False) == "undetermined"


# --- the runner's helpers --------------------------------------------------------------------


def test_runner_reads_a_tab_template_in_order(tmp_path):
    rp = _runner()
    t = tmp_path / "key.tsv"
    t.write_text("filename\tcm-score\nb.wav\t0.5\na.wav\t0.5\n")
    assert rp.read_template_ids(t) == ["b.wav", "a.wav"]
    for n in ("a.wav", "b.wav"):
        (tmp_path / n).write_bytes(b"x")
    items = rp.list_inputs(tmp_path, t)
    assert [i for i, _ in items] == ["b.wav", "a.wav"] and items[0][1] == tmp_path / "b.wav"
    c = tmp_path / "key.csv"
    c.write_text("filename,cm-score\nz.wav,0.5\n")
    with pytest.raises(ValueError, match="tab-separated"):  # NSA's template is a TSV; no comma fallback
        rp.read_template_ids(c)
    dup = tmp_path / "dup.tsv"
    dup.write_text("filename\tcm-score\na.wav\t0.5\na.wav\t0.5\n")
    with pytest.raises(ValueError):
        rp.read_template_ids(dup)
    (tmp_path / "nohead.tsv").write_text("path\tscore\na.wav\t0.5\n")
    with pytest.raises(ValueError):
        rp.read_template_ids(tmp_path / "nohead.tsv")


def test_runner_resolves_template_names_by_basename_and_flags_missing(tmp_path):
    rp = _runner()
    data = tmp_path / "data"
    for n in ("top.wav", "a/nested.wav", "a/dup.wav", "b/dup.wav"):
        p = data / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    t = tmp_path / "key.tsv"
    t.write_text("filename\tcm-score\nnested.wav\t0.5\nmissing.wav\t0.5\ntop.wav\t0.5\n")
    items = rp.list_inputs(data, t)
    assert [i for i, _ in items] == ["nested.wav", "missing.wav", "top.wav"]  # template order kept
    assert items[0][1] == data / "a" / "nested.wav" and items[1][1] is None and items[2][1] == data / "top.wav"
    amb = tmp_path / "amb.tsv"
    amb.write_text("filename\tcm-score\ndup.wav\t0.5\n")
    with pytest.raises(ValueError, match="ambiguous"):
        rp.list_inputs(data, amb)
    direct = tmp_path / "direct.tsv"
    direct.write_text("filename\tcm-score\na/dup.wav\t0.5\n")  # an exact relative path is not ambiguous
    assert rp.list_inputs(data, direct)[0][1] == data / "a" / "dup.wav"


def test_runner_missing_doc_gets_the_default_answer(consts):
    rp = _runner()
    d = rp.missing_doc("gone.wav", consts, "zmean", True, 0.3, {"git_sha": "x"})
    assert d["flag"] == "missing_file" and d["verdict"] == "undetermined" and d["detectors"] == []
    assert 0.0 <= d["probability_synthetic"] < BLOCK_TOP
    m1 = rp.missing_doc("gone.wav", None, "m1b_only", False, 0.3, {})
    assert m1["probability_synthetic"] == pytest.approx(0.3)  # M1 only, ungated: the prior-only posterior


def test_runner_resolve_scorers(tmp_path):
    rp = _runner()
    assert rp.resolve_scorers("m1b", None, tmp_path) == (("m1b_v3",), None)
    sc, fu = rp.resolve_scorers("m1b,spectra,handcrafted", None, tmp_path)
    assert sc == tuple(DETECTOR_ORDER) and fu == tmp_path / "models" / "fusion_v0" / "constants.json"
    sc, fu = rp.resolve_scorers("m1b", tmp_path / "c.json", tmp_path)
    assert sc == tuple(DETECTOR_ORDER) and fu == tmp_path / "c.json"  # a bundle needs every column
    with pytest.raises(ValueError):
        rp.resolve_scorers("m1b,wavlm", None, tmp_path)


def test_m1_only_is_make_probe_csv_arithmetic():
    from hearsay.pipeline import m1_only

    fo = m1_only(1.5, 0.3)
    assert fo.rule == "m1b_only" and fo.fused == pytest.approx(1.5 + math.log(0.3 / 0.7))
    assert fo.p == pytest.approx(float(sigmoid(1.5 + math.log(0.3 / 0.7)))) and fo.imputed == ()
    fb = m1_only(None, 0.3)
    assert fb.p == pytest.approx(0.3) and fb.imputed == ("m1b_v3",)
    doc = analyze_clip(_ctx(), FakeModels(m1=1.5), None, scorers=("m1b_v3",))
    assert doc["fusion"]["rule"] == "m1b_only"
    assert doc["probability_synthetic"] == pytest.approx(BLOCK_TOP + (1 - BLOCK_TOP) * fo.p)
    assert [d["name"] for d in doc["detectors"] if d["name"] in DEEP] == ["m1b_v3"]
    assert "handcrafted" not in {d["name"] for d in doc["detectors"]}
    assert doc["routing_log"][-1].startswith("fusion: none (m1b_only")


def test_runner_lists_sorted_audio_without_a_template(tmp_path):
    rp = _runner()
    for n in ("b.wav", "a.flac", "notes.txt", "sub/c.mp3"):
        p = tmp_path / n
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x")
    assert [i for i, _ in rp.list_inputs(tmp_path, None)] == ["a.flac", "b.wav", "sub/c.mp3"]


def test_runner_cache_tolerates_a_torn_line(tmp_path):
    rp = _runner()
    p = tmp_path / "results.jsonl"
    p.write_text(json.dumps({"_header": "h", "probe": "p"}) + "\n" + json.dumps({"filename": "a.wav", "x": 1}) + "\n"
                 + json.dumps({"filename": "a.wav", "x": 2}) + "\n" + '{"filename": "b.wav", "x"')
    header, cache = rp.load_cache(p)
    assert header == {"_header": "h", "probe": "p"}
    assert cache == {"a.wav": {"filename": "a.wav", "x": 2}}
    assert rp.load_cache(tmp_path / "missing.jsonl") == (None, {})


def test_runner_cache_identity_gates_the_resume(tmp_path):
    rp = _runner()
    p = tmp_path / "results.jsonl"
    ident = {"_header": "hearsay", "git_sha": "abc", "probe": "m1_x", "hc": "hc_y",
             "constants_sha": "c1", "scorers": ["m1b_v3"], "m1_mode": "segment", "m1_fp16": True,
             "truncate": True}  # fmt: skip
    assert rp.open_cache(p, ident, fresh=False) == {}  # new file gets a header
    header, _ = rp.load_cache(p)
    assert header["probe"] == "m1_x" and "created" in header
    with p.open("a") as f:
        f.write(json.dumps({"filename": "a.wav", "x": 1}) + "\n")
    assert rp.open_cache(p, ident, fresh=False) == {"a.wav": {"filename": "a.wav", "x": 1}}  # same identity: reused
    assert rp.open_cache(p, {**ident, "probe": "m1_other"}, fresh=False) == {}  # other probe: moved aside
    stale = list(tmp_path.glob("results.jsonl.stale-*"))
    assert len(stale) == 1 and rp.load_cache(stale[0])[1] == {"a.wav": {"filename": "a.wav", "x": 1}}
    with p.open("a") as f:
        f.write(json.dumps({"filename": "b.wav", "x": 2}) + "\n")
    assert rp.open_cache(p, ident, fresh=True) == {}  # --fresh: moved aside too
    assert len(list(tmp_path.glob("results.jsonl.stale-*"))) == 2
    headerless = tmp_path / "old.jsonl"
    headerless.write_text(json.dumps({"filename": "a.wav"}) + "\n")
    assert rp.open_cache(headerless, ident, fresh=False) == {}  # a pre-header cache is not trusted


def test_runner_compare_tsv(tmp_path):
    rp = _runner()
    ref = tmp_path / "ref.tsv"
    ref.write_text("filename\tcm-score\na.wav\t0.1\nb.wav\t0.2\nc.wav\t0.9\nd.wav\t0.5\n")
    r = rp.compare_tsv(["a.wav", "b.wav", "c.wav", "zz.wav"], [0.1, 0.25, 0.9, 0.3], ref)
    assert r["n"] == 3 and r["spearman"] == 1.0 and r["max_abs_diff"] == pytest.approx(0.05)
    assert r["n_diff_gt_0.01"] == 1 and r["n_diff_gt_0.05"] == 0


def test_runner_preflight_tolerance():
    """Multithreaded x86 BLAS is not bit-reproducible run to run: agree within 1e-6, not ==."""
    rp = _runner()
    assert rp.preflight_consistent(0.020098520111101158, 0.020098519315586292)  # the amd64 image's 8e-10
    assert rp.preflight_consistent(0.5, 0.5 + 1e-9) and rp.preflight_consistent(0.5, 0.5)
    assert not rp.preflight_consistent(0.5, 0.5 + 1e-3)
    assert not rp.preflight_consistent(float("nan"), float("nan"))
    assert not rp.preflight_consistent(float("inf"), float("inf")) and not rp.preflight_consistent(0.5, float("nan"))


def test_runner_require_offline_refuses_without_the_env(tmp_path, monkeypatch):
    rp = _runner()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(SystemExit) as e:
        rp.main(["--in", str(tmp_path), "--out", str(tmp_path / "out"), "--require-offline"])
    assert "HF_HUB_OFFLINE" in str(e.value)


def test_truncate_backbone_on_a_tiny_model_keeps_the_probe_layer():
    """hidden_states[k] is the input to encoder layer k in both Wav2Vec2 encoder variants, so
    layers[:k+1] leaves it untouched. The real-weights check is the slow test below."""
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    from hearsay.pipeline import truncate_backbone

    for stable in (True, False):
        cfg = Wav2Vec2Config(hidden_size=16, num_hidden_layers=6, num_attention_heads=2, intermediate_size=32,
                             conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=8,
                             num_conv_pos_embedding_groups=2, do_stable_layer_norm=stable, vocab_size=10)  # fmt: skip
        torch.manual_seed(0)
        m = Wav2Vec2Model(cfg).eval()
        x = torch.randn(1, 4000)
        with torch.inference_mode():
            full = m(x, output_hidden_states=True).hidden_states
            kept = truncate_backbone(m, 3)
            trunc = m(x, output_hidden_states=True).hidden_states
        assert kept == 3 and len(m.encoder.layers) == 4 and len(trunc) == 5
        for k in range(4):
            assert torch.equal(full[k], trunc[k]), (stable, k)


@pytest.mark.slow
@pytest.mark.needs_weights
@pytest.mark.skipif(not (REPO / "weights" / "wav2vec2-xls-r-300m" / "config.json").exists(),
                    reason="XLS-R weights not on this machine")  # fmt: skip
def test_truncated_backbone_keeps_layer_7_identical():
    from hearsay.embed import embed_segment, load_backbone, prepare_segment
    from hearsay.pipeline import truncate_backbone

    x = prepare_segment((0.1 * np.random.default_rng(0).standard_normal(3 * SR)).astype(np.float32))
    m = load_backbone("wav2vec2-xls-r-300m", device="cpu")
    full = embed_segment(m, x)
    assert truncate_backbone(m, 7) == 7 and len(m.encoder.layers) == 8
    trunc = embed_segment(m, x)
    assert full.shape == (25, 1024) and trunc.shape == (9, 1024)
    assert np.array_equal(full[:8], trunc[:8])


# --- the real constants and exports, when present --------------------------------------------


@pytest.mark.needs_data
@pytest.mark.skipif(not CONSTANTS.exists(), reason="models/fusion_v0/constants.json not on this machine")
def test_real_constants_match_the_pipeline_order():
    c = FusionConstants.load(CONSTANTS)
    assert c.detectors == tuple(DETECTOR_ORDER) and set(RULES) <= set(c.rules())
    assert all(c.std[d] > 0 for d in c.detectors) and c.pi_synth == 0.3


@pytest.mark.needs_data
@pytest.mark.parametrize("rule", RULES)
@pytest.mark.skipif(not (CONSTANTS.exists() and all(p.exists() for p in EXPORTS.values())
                         and all(p.exists() for p in TSV.values())),
                    reason="exports, constants or logged TSVs not on this machine")  # fmt: skip
def test_fusion_from_exported_logits_reproduces_the_logged_tsv(rule):
    """The fusion step alone (exported logits -> constants -> p) equals the logged TSV to 1e-6."""
    import pandas as pd

    c = FusionConstants.load(CONSTANTS)
    ex = {}
    for n, p in EXPORTS.items():
        e = pd.read_csv(p)
        e = e[e.split == "test"]
        ex[n] = dict(zip((Path(q).name for q in e.path), e.logit, strict=True))
    ref = pd.read_csv(TSV[rule], sep="\t")
    got = np.array([c.fuse({n: ex[n][f] for n in ex}, rule).p for f in ref.filename])
    assert np.max(np.abs(got - ref["cm-score"].to_numpy())) < 1e-6
