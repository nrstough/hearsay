"""hearsay.pipeline and scripts/run_pipeline.py, hermetically: toy fusion constants in both
file layouts (fusion_v0 z rules, fusion_v1 rank rule), fake scorers and fake detectors (no
weights, no data). The exact-parity checks against the exported score files and the logged TSVs
run only when those gitignored files are present."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hearsay import SR
from hearsay.detectors.base import ClipContext, DetectorResult
from hearsay.detectors.speech_gate import BLOCK_TOP, FAILURE_TOP
from hearsay.metrics import sigmoid
from hearsay.pipeline import (
    DEEP,
    DETECTOR_ORDER,
    RULES,
    Z_RULES,
    FusionConstants,
    analyze_clip,
    detector_entry,
    final_score,
    refuse,
    routing_log,
    verdict_for,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_pipeline.py"
CONSTANTS_V0 = REPO / "models" / "fusion_v0" / "constants.json"
CONSTANTS_V1 = REPO / "models" / "fusion_v1" / "constants.json"
EXPORTS = {n: REPO / "outputs" / "detector_scores" / f"{n}.csv" for n in DETECTOR_ORDER}
TSV_V0 = {
    "zmean": REPO / "submissions" / "20260926-0602_M4_fusion_zmean_m1bv3_hcv5_m3.tsv",
    "stack_nonlj": REPO / "submissions" / "20260926-0602_M4_fusion_stack_nonlj_m1bv3_hcv5_m3.tsv",
}
TSV_V1 = {
    False: REPO / "submissions" / "20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv",
    True: REPO / "submissions" / "20260926-0813_M4_sweep_E_on_A_alpha0.2_FLIPPED_only_if_NSA_scores_inverted.tsv",
}


def _runner():
    spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def det_map(p: float) -> float:
    """The policy's determinate map, applied exactly once on the Platt probability."""
    return BLOCK_TOP + (1 - BLOCK_TOP) * p


# --- toy constants in scripts/fuse.py's file layout (fusion_v0) ------------------------------

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

# --- toy constants in scripts/fuse_sweep.py's file layout (fusion_v1) ------------------------

TOY_V1 = {
    "final": "E_on_A_alpha0.2",
    "pi_synth": 0.3,
    "rank_ref_inner_oof_sorted": {"m1b_v3": [-10.0, -5.0, 0.0, 5.0, 10.0],
                                  "handcrafted_v5": [-4.0, -2.0, 0.0, 2.0, 4.0]},
    "alpha_handcrafted": 0.2,
    "e_rule": {"applied": True, "m3_logit_below": -3.0, "base_rank_above": 0.5, "multiply_by": 0.5},
    "platt": {"a": 17.8, "b": -8.3, "prior_shift": math.log(0.3 / 0.7)},
    "determinate_map": "0.001 + 0.999 * sigmoid(a*fused + b + prior_shift); gated files below 0.001",
    "how": "rank_d = searchsorted(rank_ref[d], logit_d)/len; base = (1-alpha)*rank_m1b + alpha*rank_hc; "
           "if m3_logit < -3 and base > 0.5: base *= 0.5; p = determinate_map(base)",
}  # fmt: skip
LOGITS_V1 = {"m1b_v3": 1.0, "handcrafted_v5": 3.0, "spectra_aasist": -7.0}  # ranks 0.6, 0.8; M3 suppresses


@pytest.fixture
def consts():
    return FusionConstants.from_dict(TOY, source="toy")


@pytest.fixture
def consts_v1():
    return FusionConstants.from_dict(TOY_V1, source="toy_v1")


def _z(logits):
    return np.array([(logits[d] - TOY["standardize"][d]["mean"]) / TOY["standardize"][d]["std"]
                     for d in TOY["detectors"]])  # fmt: skip


def _platt(pl, fused):
    return float(sigmoid(pl["a"] * fused + pl["b"] + pl["prior_shift"]))


def test_constants_parse_and_rules(consts):
    assert consts.detectors == tuple(DETECTOR_ORDER)
    assert consts.rules() == Z_RULES  # a fusion_v0 file has no e_on_a
    assert consts.stack_weights == {"m1b_v3": 2.0, "handcrafted_v5": 0.5, "spectra_aasist": 7.0}
    assert consts.weights("zmean") == {d: pytest.approx(1 / 3) for d in DETECTOR_ORDER}
    with pytest.raises(ValueError, match="not defined"):
        consts.weights("rankmean")
    with pytest.raises(ValueError, match="not defined"):
        consts.fuse(LOGITS, "e_on_a")  # the default rule is refused on a v0 file


def test_constants_load_from_file(tmp_path):
    p = tmp_path / "constants.json"
    p.write_text(json.dumps(TOY))
    c = FusionConstants.load(p)
    assert c.source == str(p) and c.pi_synth == 0.3
    assert c.platt["zmean"]["a"] == 11.7
    p1 = tmp_path / "v1.json"
    p1.write_text(json.dumps(TOY_V1))
    c1 = FusionConstants.load(p1)
    assert c1.rules() == ("e_on_a",) and c1.final == "E_on_A_alpha0.2" and c1.alpha == 0.2


def test_constants_reject_mismatched_weights():
    bad = json.loads(json.dumps(TOY))
    bad["rules"]["stack_nonlj"]["weights"] = [1.0, 2.0]
    with pytest.raises(ValueError):
        FusionConstants.from_dict(bad)
    bad = json.loads(json.dumps(TOY))
    del bad["standardize"]["spectra_aasist"]
    with pytest.raises(ValueError):
        FusionConstants.from_dict(bad)
    bad = json.loads(json.dumps(TOY_V1))
    bad["rank_ref_inner_oof_sorted"]["m1b_v3"] = [1.0, 0.0]  # not sorted
    with pytest.raises(ValueError, match="sorted"):
        FusionConstants.from_dict(bad)
    bad = json.loads(json.dumps(TOY_V1))
    bad["alpha_handcrafted"] = 1.5
    with pytest.raises(ValueError, match="alpha"):
        FusionConstants.from_dict(bad)


def test_zmean_is_fuse_py_arithmetic(consts):
    fo = consts.fuse(LOGITS, "zmean")
    z = _z(LOGITS)
    fused = float(np.mean(z))
    assert fo.fused == pytest.approx(fused, abs=1e-15)
    assert fo.p == pytest.approx(_platt(TOY["rules"]["zmean"]["platt"], fused), abs=1e-15)
    assert fo.imputed == () and fo.rule == "zmean"
    assert list(fo.inputs) == list(DETECTOR_ORDER)
    assert fo.terms == {d: pytest.approx(v) for d, v in zip(DETECTOR_ORDER, z, strict=True)}


def test_stack_nonlj_is_fuse_py_arithmetic(consts):
    fo = consts.fuse(LOGITS, "stack_nonlj")
    z = _z(LOGITS)
    fused = float(z @ np.array([2.0, 0.5, 7.0]) - 0.5)  # LogisticRegression.decision_function
    assert fo.fused == pytest.approx(fused, abs=1e-15)
    assert fo.p == pytest.approx(_platt(TOY["rules"]["stack_nonlj"]["platt"], fused), abs=1e-15)
    assert fo.weights == consts.stack_weights


def test_missing_logit_is_imputed_at_the_inner_mean(consts):
    fo = consts.fuse({**LOGITS, "spectra_aasist": None}, "stack_nonlj")
    assert fo.imputed == ("spectra_aasist",) and fo.terms["spectra_aasist"] == 0.0
    assert fo.inputs["spectra_aasist"] is None
    fo2 = consts.fuse({**LOGITS, "spectra_aasist": float("nan")}, "zmean")
    assert fo2.imputed == ("spectra_aasist",)


def test_e_on_a_is_fuse_sweep_arithmetic(consts_v1):
    """Steps 1-4 of the shipped rule; step 5 (the determinate map) belongs to the policy."""
    c = consts_v1
    assert c.rules() == ("e_on_a",) and c.detectors == tuple(DETECTOR_ORDER)
    assert c.weights("e_on_a") == {"m1b_v3": pytest.approx(0.8), "handcrafted_v5": pytest.approx(0.2),
                                   "spectra_aasist": 0.0}  # fmt: skip
    fo = c.fuse(LOGITS_V1, "e_on_a")
    assert fo.terms == {"m1b_v3": 0.6, "handcrafted_v5": 0.8}  # searchsorted / 5
    base = 0.8 * 0.6 + 0.2 * 0.8
    assert fo.detail["base"] == pytest.approx(base) and fo.detail["e_applied"] is True
    assert fo.fused == pytest.approx(base * 0.5)  # M3 margin -7 < -3 and base 0.64 > 0.5
    assert fo.p == pytest.approx(_platt(TOY_V1["platt"], base * 0.5), abs=1e-15)
    assert fo.imputed == () and fo.detail["final"] == "E_on_A_alpha0.2"
    assert fo.p < 0.5  # p is the Platt probability, not yet mapped to [0.001, 1]
    # M3 never promotes: a strongly synthetic M3 margin leaves the base alone
    fo2 = c.fuse({**LOGITS_V1, "spectra_aasist": 9.0}, "e_on_a")
    assert fo2.fused == pytest.approx(base) and fo2.detail["e_applied"] is False
    # suppression needs base > 0.5: a real-looking file is not pushed further down
    fo3 = c.fuse({"m1b_v3": -6.0, "handcrafted_v5": -3.0, "spectra_aasist": -7.0}, "e_on_a")
    assert fo3.terms == {"m1b_v3": 0.2, "handcrafted_v5": 0.2} and fo3.detail["e_applied"] is False
    assert fo3.fused == pytest.approx(0.2)
    # boundary: searchsorted is left-sided, so a logit equal to a reference value ranks below it
    assert c.rank("m1b_v3", -10.0) == 0.0 and c.rank("m1b_v3", 10.0) == 0.8 and c.rank("m1b_v3", 11.0) == 1.0


def test_e_on_a_imputation(consts_v1):
    fo = consts_v1.fuse({**LOGITS_V1, "m1b_v3": None}, "e_on_a")
    assert fo.imputed == ("m1b_v3",) and fo.terms["m1b_v3"] == 0.5  # the inner-fold median
    assert fo.detail["base"] == pytest.approx(0.8 * 0.5 + 0.2 * 0.8)
    fo2 = consts_v1.fuse({**LOGITS_V1, "spectra_aasist": None}, "e_on_a")
    assert fo2.imputed == ("spectra_aasist",) and fo2.detail["e_applied"] is False  # no evidence, no suppression
    assert fo2.fused == pytest.approx(fo2.detail["base"])
    with pytest.raises(ValueError, match="not defined"):
        consts_v1.fuse(LOGITS_V1, "zmean")  # a v1 file has no z rules


def test_e_rule_off_in_the_file():
    c = FusionConstants.from_dict({**TOY_V1, "e_rule": {"applied": False}})
    fo = c.fuse(LOGITS_V1, "e_on_a")
    assert fo.detail["e_applied"] is False and fo.fused == pytest.approx(fo.detail["base"])


# --- the policy: determinate map once, pinned block, flip ------------------------------------


def test_final_score_maps_once_and_pins_the_block():
    p = 0.37
    assert final_score(p, True) == pytest.approx(det_map(p))
    assert final_score(1.0, True) == 1.0 and final_score(0.0, True) == pytest.approx(BLOCK_TOP)
    # policy off: every file is determinate, still mapped (the map is step 5 of the rule)
    assert final_score(p, False, apply_gate=False) == pytest.approx(det_map(p))
    # gated: pinned block, ordered by the weak signal, jittered by the key, below every scored file
    g = final_score(p, False, key="a.wav", order_by=0.9)
    assert FAILURE_TOP <= g < BLOCK_TOP
    assert g != final_score(p, False, key="b.wav", order_by=0.9)  # hash jitter keyed by filename
    assert final_score(p, False, key="a.wav", order_by=0.9) == g  # deterministic
    assert final_score(p, False, key="a.wav", order_by=0.1) < g  # ordered by the signal
    # failures (decode errors, missing files) sit below FAILURE_TOP
    f = final_score(p, False, key="a.wav", order_by=0.9, failed=True)
    assert 0.0 <= f < FAILURE_TOP
    # flip: 1 - p through the same map; the block stays at the minimum
    assert final_score(p, True, flip=True) == pytest.approx(det_map(1 - p))
    assert final_score(p, False, key="a.wav", order_by=0.9, flip=True) < BLOCK_TOP


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


def fake_dets(is_speech=1.0, hc_fail=False, gate_fail=False, hc_logit=LOGITS["handcrafted_v5"]):
    return [
        FakeDet("compression", 0.4, {"bw_hz": 7250.0}),
        FakeDet("container", 0.5, {"lossy": 0.0, "is_pcm_wav": 1.0, "ffmpeg_written": 1.0}),
        FakeDet("enf", 0.5, {"enf_present": 0.0, "enf_stable": 0.0}),
        FakeDet("handcrafted", 0.21, {"hc_logit": hc_logit, "voiced_frac": 0.8}, fail=hc_fail),
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


def fake_models_v1(**kw):
    """Fake scorers producing LOGITS_V1 (ranks 0.6 / 0.8, M3 suppresses)."""
    return FakeModels(m1=LOGITS_V1["m1b_v3"], spectra=LOGITS_V1["spectra_aasist"],
                      dets=fake_dets(hc_logit=LOGITS_V1["handcrafted_v5"], **kw))  # fmt: skip


def _ctx(seconds=3.0, seed=0, name="clip.wav"):
    x = (0.1 * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)
    return ClipContext.from_array(x, path=name)


REQUIRED = {"filename", "duration_s", "probability_synthetic", "verdict", "is_speech",
            "default_answer_applied", "fusion", "detectors", "routing_log", "version"}  # fmt: skip


def test_analyze_response_shape_and_fusion(consts):
    doc = analyze_clip(_ctx(), FakeModels(), consts, rule="stack_nonlj")
    assert REQUIRED <= set(doc)
    assert doc["filename"] == "clip.wav" and doc["duration_s"] == 3.0
    assert doc["is_speech"] is True and doc["default_answer_applied"] is False
    fo = consts.fuse(LOGITS, "stack_nonlj")
    assert doc["probability_synthetic"] == pytest.approx(det_map(fo.p))  # the determinate map, once
    assert doc["fusion"]["p_fused"] == fo.p and doc["verdict"] == "real"
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
    assert doc["version"]["polarity"] == "our_direction"
    json.dumps(doc)  # serializable as written


def test_analyze_with_the_shipped_rule_applies_the_map_exactly_once(consts_v1):
    """Steps 1-4 in fusion, step 5 in the policy: probability_synthetic == 0.001 + 0.999 * p."""
    doc = analyze_clip(_ctx(), fake_models_v1(), consts_v1)  # default rule: e_on_a
    fo = consts_v1.fuse(LOGITS_V1, "e_on_a")
    assert doc["fusion"]["rule"] == "e_on_a" and doc["fusion"]["p_fused"] == fo.p
    assert doc["probability_synthetic"] == pytest.approx(det_map(fo.p), abs=1e-15)
    assert doc["probability_synthetic"] != pytest.approx(det_map(det_map(fo.p)), abs=1e-6)  # not twice
    assert doc["fusion"]["detail"]["e_applied"] is True and doc["fusion"]["terms"] == fo.terms
    assert doc["fusion"]["weights"]["spectra_aasist"] == 0.0
    line = doc["routing_log"][-1]
    assert line.startswith("fusion: e_on_a (E_on_A_alpha0.2): 0.8 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5)")
    assert "M3 false-alarm suppression applied" in line and "x0.5" in line
    doc2 = analyze_clip(_ctx(), FakeModels(m1=1.0, spectra=9.0, dets=fake_dets(hc_logit=3.0)), consts_v1)
    assert "no suppression (M3 never promotes)" in doc2["routing_log"][-1]
    flipped = analyze_clip(_ctx(), fake_models_v1(), consts_v1, flip=True)
    assert flipped["probability_synthetic"] == pytest.approx(det_map(1 - fo.p), abs=1e-15)
    assert flipped["version"]["polarity"] == "flipped" and flipped["verdict"] == doc["verdict"]


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
    assert FAILURE_TOP <= doc["probability_synthetic"] < BLOCK_TOP  # pinned block, below every scored file
    assert doc["fusion"]["p_fused"] == fo.p  # the fused value is still reported
    assert doc["verdict"] == "undetermined"
    assert any("is_speech=false" in line and "policy applied" in line for line in doc["routing_log"])
    other = analyze_clip(_ctx(name="other.wav"), m, consts, rule="zmean")
    assert other["probability_synthetic"] != doc["probability_synthetic"]  # jitter keyed by filename
    ungated = analyze_clip(_ctx(), m, consts, rule="zmean", apply_gate=False)
    assert ungated["probability_synthetic"] == pytest.approx(det_map(fo.p))
    assert ungated["default_answer_applied"] is False
    assert ungated["version"]["policy"] == "none"
    assert any("policy off" in line for line in ungated["routing_log"])


def test_gate_error_does_not_gate(consts):
    doc = analyze_clip(_ctx(), FakeModels(dets=fake_dets(gate_fail=True)), consts, "zmean")
    assert doc["is_speech"] is True and doc["default_answer_applied"] is False
    gate = next(d for d in doc["detectors"] if d["name"] == "speech_gate")
    assert gate["status"] == "error" and gate["score"] == 0.5 and "boom" in gate["error"]
    assert any(line.startswith("speech_gate: error") for line in doc["routing_log"])


def test_failed_fused_detector_is_imputed_and_logged(consts):
    doc = analyze_clip(_ctx(), FakeModels(dets=fake_dets(hc_fail=True), m1_fail=True), consts, "stack_nonlj")
    assert set(doc["fusion"]["imputed"]) == {"m1b_v3", "handcrafted_v5"}
    assert doc["fusion"]["inputs"]["m1b_v3"] is None and doc["fusion"]["terms"]["m1b_v3"] == 0.0
    m1 = next(d for d in doc["detectors"] if d["name"] == "m1b_v3")
    assert m1["status"] == "error" and m1["score"] == 0.5 and "no backbone" in m1["error"]
    assert "imputed at the inner-fold centre: m1b_v3, handcrafted_v5" in doc["routing_log"][-1]
    assert BLOCK_TOP <= doc["probability_synthetic"] <= 1.0


def test_decode_failure_is_undetermined_below_the_block(consts, tmp_path):
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
    assert 0.0 <= doc["probability_synthetic"] < FAILURE_TOP and doc["flag"] == "decode_error"


def test_refuse_switches_rule_policy_and_polarity_without_models(consts, consts_v1):
    a = analyze_clip(_ctx(), FakeModels(), consts, rule="stack_nonlj")
    b = analyze_clip(_ctx(), FakeModels(), consts, rule="zmean")
    r = refuse(a, consts, "zmean")
    assert r["probability_synthetic"] == b["probability_synthetic"]
    assert r["fusion"] == b["fusion"] and r["routing_log"] == b["routing_log"]
    assert r["version"]["rule"] == "zmean" and r["detectors"] == a["detectors"]
    g = analyze_clip(_ctx(), FakeModels(dets=fake_dets(is_speech=0.0)), consts, rule="zmean")
    u = refuse(g, consts, "zmean", apply_gate=False)
    assert u["probability_synthetic"] == pytest.approx(det_map(g["fusion"]["p_fused"]))
    assert u["default_answer_applied"] is False
    v = analyze_clip(_ctx(), fake_models_v1(), consts_v1)
    f = refuse(v, consts_v1, "e_on_a", flip=True)
    assert f["probability_synthetic"] == pytest.approx(det_map(1 - v["fusion"]["p_fused"]))
    assert f["version"]["polarity"] == "flipped" and f["routing_log"] == v["routing_log"]
    same = refuse(v, consts_v1, "e_on_a")
    assert same["probability_synthetic"] == v["probability_synthetic"] and same["fusion"] == v["fusion"]


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


def test_runner_missing_doc_sits_below_the_block(consts, consts_v1):
    rp = _runner()
    d = rp.missing_doc("gone.wav", consts, "zmean", True, 0.3, {"git_sha": "x"})
    assert d["flag"] == "missing_file" and d["verdict"] == "undetermined" and d["detectors"] == []
    assert 0.0 <= d["probability_synthetic"] < FAILURE_TOP
    m1 = rp.missing_doc("gone.wav", None, "m1b_only", False, 0.3, {})
    assert m1["probability_synthetic"] == pytest.approx(det_map(0.3))  # M1 only, ungated: the prior-only posterior
    fl = rp.missing_doc("gone.wav", consts_v1, "e_on_a", True, 0.3, {}, flip=True)
    assert 0.0 <= fl["probability_synthetic"] < FAILURE_TOP  # the block stays at the minimum when flipped


def test_runner_resolve_scorers(tmp_path):
    rp = _runner()
    assert rp.resolve_scorers("m1b", None, tmp_path) == (("m1b_v3",), None)
    sc, fu = rp.resolve_scorers("m1b,spectra,handcrafted", None, tmp_path)
    assert sc == tuple(DETECTOR_ORDER) and fu == tmp_path / "models" / "fusion_v1" / "constants.json"
    sc, fu = rp.resolve_scorers("m1b", tmp_path / "c.json", tmp_path)
    assert sc == tuple(DETECTOR_ORDER) and fu == tmp_path / "c.json"  # a bundle needs every column
    with pytest.raises(ValueError):
        rp.resolve_scorers("m1b,wavlm", None, tmp_path)


def test_runner_refuses_a_rule_the_file_does_not_define(tmp_path):
    rp = _runner()
    (tmp_path / "v0.json").write_text(json.dumps(TOY))
    (tmp_path / "x.wav").write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        rp.main(["--in", str(tmp_path), "--out", str(tmp_path / "out"), "--fusion", str(tmp_path / "v0.json"),
                 "--rule", "e_on_a", "--no-preflight"])  # fmt: skip
    assert "not defined" in str(e.value) and "zmean" in str(e.value)


def test_m1_only_is_make_probe_csv_arithmetic():
    from hearsay.pipeline import m1_only

    fo = m1_only(1.5, 0.3)
    assert fo.rule == "m1b_only" and fo.fused == pytest.approx(1.5 + math.log(0.3 / 0.7))
    assert fo.p == pytest.approx(float(sigmoid(1.5 + math.log(0.3 / 0.7)))) and fo.imputed == ()
    fb = m1_only(None, 0.3)
    assert fb.p == pytest.approx(0.3) and fb.imputed == ("m1b_v3",)
    doc = analyze_clip(_ctx(), FakeModels(m1=1.5), None, scorers=("m1b_v3",))
    assert doc["fusion"]["rule"] == "m1b_only"
    assert doc["probability_synthetic"] == pytest.approx(det_map(fo.p))
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
    assert rp.open_cache(p, {**ident, "constants_sha": "c2"}, fresh=False) == {}  # other constants: moved aside
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


def _exports():
    import pandas as pd

    ex = {}
    for n, p in EXPORTS.items():
        e = pd.read_csv(p)
        e = e[e.split == "test"]
        ex[n] = dict(zip((Path(q).name for q in e.path), e.logit, strict=True))
    return ex


@pytest.mark.needs_data
@pytest.mark.skipif(not (CONSTANTS_V0.exists() and CONSTANTS_V1.exists()),
                    reason="models/fusion_v*/constants.json not on this machine")  # fmt: skip
def test_real_constants_match_the_pipeline_order():
    c0 = FusionConstants.load(CONSTANTS_V0)
    assert c0.detectors == tuple(DETECTOR_ORDER) and c0.rules() == Z_RULES and c0.pi_synth == 0.3
    assert all(c0.std[d] > 0 for d in c0.detectors)
    c1 = FusionConstants.load(CONSTANTS_V1)
    assert c1.rules() == ("e_on_a",) and c1.alpha == 0.2 and c1.e_rule["applied"] is True
    assert c1.final == "E_on_A_alpha0.2" and all(len(c1.rank_ref[d]) == 16142 for d in c1.rank_ref)
    assert set(RULES) == set(c0.rules()) | set(c1.rules())


@pytest.mark.needs_data
@pytest.mark.parametrize("rule", Z_RULES)
@pytest.mark.skipif(not (CONSTANTS_V0.exists() and all(p.exists() for p in EXPORTS.values())
                         and all(p.exists() for p in TSV_V0.values())),
                    reason="exports, constants or logged TSVs not on this machine")  # fmt: skip
def test_fusion_v0_from_exported_logits_reproduces_the_logged_tsv(rule):
    """The fusion step alone (exported logits -> constants -> p) equals the 06:02 TSVs to 1e-6.
    Those TSVs predate the determinate map, so the Platt p itself is compared."""
    import pandas as pd

    c = FusionConstants.load(CONSTANTS_V0)
    ex = _exports()
    ref = pd.read_csv(TSV_V0[rule], sep="\t")
    got = np.array([c.fuse({n: ex[n][f] for n in ex}, rule).p for f in ref.filename])
    assert np.max(np.abs(got - ref["cm-score"].to_numpy())) < 1e-6


@pytest.mark.needs_data
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.skipif(not (CONSTANTS_V1.exists() and all(p.exists() for p in EXPORTS.values())
                         and all(p.exists() for p in TSV_V1.values())),
                    reason="exports, constants or the 0813 TSVs not on this machine")  # fmt: skip
def test_e_on_a_from_exported_logits_reproduces_the_0813_tsv(flip):
    """Exported logits -> fusion_v1 constants -> policy (map once, no gating on the test set)
    equals the 0813 TSV, our direction and pre-flipped, to 1e-6."""
    import pandas as pd

    c = FusionConstants.load(CONSTANTS_V1)
    ex = _exports()
    ref = pd.read_csv(TSV_V1[flip], sep="\t")
    got = np.array([final_score(c.fuse({n: ex[n][f] for n in ex}, "e_on_a").p, True, flip=flip)
                    for f in ref.filename])  # fmt: skip
    assert np.max(np.abs(got - ref["cm-score"].to_numpy())) < 1e-6
    assert got.min() >= BLOCK_TOP and got.max() <= 1.0
