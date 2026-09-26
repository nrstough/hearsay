"""hearsay.api, hermetically: fake models behind /analyze, a temp results directory behind
/results, no weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from hearsay import SR
from hearsay import api as api_mod
from hearsay.detectors.speech_gate import BLOCK_TOP
from hearsay.pipeline import DEFAULT_CONSTANTS_PATH, DETECTOR_ORDER, FusionConstants
from test_pipeline import (
    LOGITS,
    LOGITS_V1,
    TOY,
    TOY_V1,
    TOY_V2,
    V1_DETECTORS,
    FakeModels,
    fake_dets,
    fake_models_v1,
    fake_models_v2,
)

FRESH_STATE = {"models": None, "consts": None, "load_seconds": None, "scorers": None, "error": None}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("HEARSAY_RULE", raising=False)
    monkeypatch.delenv("HEARSAY_POLICY", raising=False)
    monkeypatch.delenv("HEARSAY_DETECTORS", raising=False)
    monkeypatch.delenv("HEARSAY_FUSION", raising=False)
    monkeypatch.delenv("HEARSAY_M5", raising=False)
    monkeypatch.setenv("HEARSAY_RESULTS", str(tmp_path / "out"))
    (tmp_path / "out" / "results").mkdir(parents=True)
    (tmp_path / "out" / "results" / "HGT1.wav.json").write_text(json.dumps({"filename": "HGT1.wav", "probability_synthetic": 0.12}))
    (tmp_path / "out" / "results" / "bad.wav.json").write_text("{not json")
    monkeypatch.setattr(api_mod, "_state", dict(FRESH_STATE))
    api_mod._scorers_cache.clear()
    return TestClient(api_mod.app)


def _write(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def _wav(path: Path, seconds=2.0):
    x = (0.1 * np.random.default_rng(0).standard_normal(int(seconds * SR))).astype(np.float32)
    sf.write(path, x, SR)
    return path


def test_health_without_loading_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models_loaded"] is False
    assert body["rule"] == "e_on_a" and body["policy"] == "speech_gate" and body["n_results"] == 2
    if DEFAULT_CONSTANTS_PATH.exists():  # the real (gitignored) v2 file names all four columns
        assert body["scorers"] == list(DETECTOR_ORDER) and body["scorers_from"] == "constants"
    else:
        assert body["scorers"] == list(DETECTOR_ORDER) and body["scorers_from"] == "env (fusion file missing)"
    assert body["version"]["fusion"] == "fusion_v2/constants.json"


def test_health_reports_m1_only_mode(client, monkeypatch):
    monkeypatch.setenv("HEARSAY_DETECTORS", "m1b")
    body = client.get("/health").json()
    assert body["rule"] == "m1b_only" and body["scorers"] == ["m1b_v3"] and body["version"]["fusion"] == "none"
    monkeypatch.setenv("HEARSAY_RULE", "rankmean")
    assert client.get("/health").status_code == 500


def test_results_lookup(client):
    r = client.get("/results/HGT1.wav")
    assert r.status_code == 200 and r.json() == {"filename": "HGT1.wav", "probability_synthetic": 0.12}
    assert client.get("/results/nope.wav").status_code == 404
    assert client.get("/results/bad.wav").status_code == 500
    assert client.get("/results/..%2Fx.json").status_code in (400, 404)
    listing = client.get("/results").json()
    assert listing["n"] == 2 and listing["filenames"] == ["HGT1.wav", "bad.wav"]


def test_results_need_the_env(client, monkeypatch):
    monkeypatch.delenv("HEARSAY_RESULTS")
    assert client.get("/results/HGT1.wav").status_code == 503
    assert client.get("/results").status_code == 503
    assert client.get("/health").json()["results_dir"] is None


def test_analyze_uses_the_pipeline(client, monkeypatch, tmp_path):
    monkeypatch.setenv("HEARSAY_RULE", "zmean")  # the toy file is fusion_v0-shaped
    consts = FusionConstants.from_dict(TOY, source="toy")
    fake = FakeModels(dets=fake_dets())
    monkeypatch.setattr(api_mod, "get_models", lambda: (fake, consts))
    wav = _wav(tmp_path / "clip one.wav")
    with wav.open("rb") as f:
        r = client.post("/analyze", files={"file": ("clip one.wav", f, "audio/wav")})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["filename"] == "clip one.wav" and doc["duration_s"] == 2.0
    assert doc["probability_synthetic"] == pytest.approx(
        BLOCK_TOP + (1 - BLOCK_TOP) * consts.fuse(LOGITS, "zmean").p
    )  # the default-answer policy maps scored files into [0.001, 1]
    assert doc["fusion"]["rule"] == "zmean" and doc["verdict"] == "real" and doc["api_seconds"] >= 0
    assert {d["name"] for d in doc["detectors"]} >= {"speech_gate", "m1b_v3", "spectra_aasist"}
    assert fake.calls == 1
    assert not list(Path(tmp_path).glob("hearsay_api_*"))  # the upload's temp dir is gone
    assert client.get("/health").json()["models_loaded"] is False  # the fake never touched _state


def test_analyze_respects_rule_and_policy_env(client, monkeypatch, tmp_path):
    consts = FusionConstants.from_dict(TOY, source="toy")
    monkeypatch.setattr(api_mod, "get_models", lambda: (FakeModels(dets=fake_dets(is_speech=0.0)), consts))
    monkeypatch.setenv("HEARSAY_RULE", "stack_nonlj")
    monkeypatch.setenv("HEARSAY_POLICY", "none")
    wav = _wav(tmp_path / "x.wav")
    with wav.open("rb") as f:
        doc = client.post("/analyze", files={"file": ("x.wav", f, "audio/wav")}).json()
    assert doc["fusion"]["rule"] == "stack_nonlj" and doc["is_speech"] is False
    assert doc["default_answer_applied"] is False  # policy off: no block, but the determinate map still applies
    assert doc["probability_synthetic"] == pytest.approx(BLOCK_TOP + (1 - BLOCK_TOP) * doc["fusion"]["p_fused"])


def test_analyze_default_rule_is_e_on_a_mapped_once(client, monkeypatch, tmp_path):
    consts = FusionConstants.from_dict(TOY_V1, source="toy_v1")
    monkeypatch.setattr(api_mod, "get_models", lambda: (fake_models_v1(), consts))
    wav = _wav(tmp_path / "v1.wav")
    with wav.open("rb") as f:
        doc = client.post("/analyze", files={"file": ("v1.wav", f, "audio/wav")}).json()
    fo = consts.fuse(LOGITS_V1, "e_on_a")
    assert doc["fusion"]["rule"] == "e_on_a" and doc["fusion"]["p_fused"] == fo.p
    assert doc["probability_synthetic"] == pytest.approx(BLOCK_TOP + (1 - BLOCK_TOP) * fo.p, abs=1e-15)
    assert doc["fusion"]["detail"]["e_applied"] is True and doc["version"]["polarity"] == "our_direction"
    assert client.get("/health").json()["rule"] == "e_on_a"
    monkeypatch.setenv("HEARSAY_RULE", "zmean")  # a rule the file does not define is refused
    with wav.open("rb") as f:
        r = client.post("/analyze", files={"file": ("v1.wav", f, "audio/wav")})
    assert r.status_code == 500 and "not defined" in r.json()["detail"]


def test_analyze_undecodable_upload_is_undetermined(client, monkeypatch, tmp_path):
    monkeypatch.setenv("HEARSAY_RULE", "zmean")
    consts = FusionConstants.from_dict(TOY, source="toy")
    monkeypatch.setattr(api_mod, "get_models", lambda: (FakeModels(), consts))
    r = client.post("/analyze", files={"file": ("junk.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 200
    doc = r.json()
    assert doc["verdict"] == "undetermined" and doc["duration_s"] is None and doc["flag"] == "decode_error"
    assert client.post("/analyze", files={"file": ("empty.wav", b"", "audio/wav")}).status_code == 400


def test_health_reports_the_file_s_scorers(client, monkeypatch, tmp_path):
    v1 = _write(tmp_path / "fusion_v1" / "constants.json", TOY_V1)
    v2 = _write(tmp_path / "fusion_v2" / "constants.json", TOY_V2)
    monkeypatch.setenv("HEARSAY_FUSION", str(v1))
    b = client.get("/health").json()
    assert b["scorers"] == list(V1_DETECTORS) and b["scorers_from"] == "constants" and b["models_loaded"] is False
    assert b["version"]["fusion"] == "fusion_v1/constants.json"
    monkeypatch.setenv("HEARSAY_FUSION", str(v2))
    b = client.get("/health").json()
    assert b["scorers"] == list(DETECTOR_ORDER) and b["version"]["fusion"] == "fusion_v2/constants.json"
    monkeypatch.setenv("HEARSAY_FUSION", str(tmp_path / "missing.json"))
    b = client.get("/health").json()
    assert b["scorers"] == list(DETECTOR_ORDER) and b["scorers_from"] == "env (fusion file missing)"
    _write(tmp_path / "broken.json", {**TOY_V2, "weights": {"m1b_v3": 0.5}})
    monkeypatch.setenv("HEARSAY_FUSION", str(tmp_path / "broken.json"))
    assert client.get("/health").status_code == 500  # a malformed file is a loud 500, not a silent list


def test_api_loads_m5_only_when_the_constants_need_it(client, monkeypatch, tmp_path):
    seen = []

    class Recorder:  # stands in for hearsay.pipeline.Models: records the kwargs, serves a fake
        def __init__(self, **kw):
            seen.append(kw)
            self._fake = fake_models_v2() if kw.get("load_m5") else fake_models_v1()
            self.m5_hashes = self._fake.m5_hashes if kw.get("load_m5") else None

        def __getattr__(self, name):
            return getattr(self._fake, name)

    monkeypatch.setattr(api_mod, "Models", Recorder)
    v2 = _write(tmp_path / "fusion_v2" / "constants.json", TOY_V2)
    v1 = _write(tmp_path / "fusion_v1" / "constants.json", TOY_V1)
    m5 = _write(tmp_path / "m5" / "hashes.json", TOY_V2["m5_checkpoint"]).parent
    monkeypatch.setenv("HEARSAY_FUSION", str(v2))
    monkeypatch.setenv("HEARSAY_M5", str(m5))
    _, consts = api_mod.get_models()
    assert seen[-1]["load_m5"] is True and seen[-1]["m5_dir"] == m5 and consts.detectors == tuple(DETECTOR_ORDER)
    assert api_mod._state["scorers"] == tuple(DETECTOR_ORDER)
    b = client.get("/health").json()
    assert b["models_loaded"] is True and b["scorers"] == list(DETECTOR_ORDER) and b["scorers_from"] == "loaded"
    wav = _wav(tmp_path / "v2.wav")
    with wav.open("rb") as f:
        doc = client.post("/analyze", files={"file": ("v2.wav", f, "audio/wav")}).json()
    assert {d["name"] for d in doc["detectors"]} >= {"m1b_v3", "m5_xlsr_ft", "spectra_aasist"}
    assert doc["fusion"]["weights"]["m5_xlsr_ft"] == 0.2 and doc["fusion"]["imputed"] == []

    monkeypatch.setattr(api_mod, "_state", dict(FRESH_STATE))  # a v1 file: M5 is not loaded
    monkeypatch.setenv("HEARSAY_FUSION", str(v1))
    api_mod.get_models()
    assert seen[-1]["load_m5"] is False and api_mod._state["scorers"] == V1_DETECTORS

    monkeypatch.setattr(api_mod, "_state", dict(FRESH_STATE))  # another checkpoint: refused before any load
    _write(m5 / "hashes.json", {**TOY_V2["m5_checkpoint"], "head_sha256": "x" * 64})
    monkeypatch.setenv("HEARSAY_FUSION", str(v2))
    n = len(seen)
    with pytest.raises(HTTPException) as e:
        api_mod.get_models()
    assert "head_sha256" in str(e.value.detail) and len(seen) == n  # Models was never constructed
    assert api_mod._state["error"] and client.get("/health").json()["models_loaded"] is False
    with pytest.raises(HTTPException):  # remembered: the next request fails fast, no reload
        api_mod.get_models()
    assert len(seen) == n
