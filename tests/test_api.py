"""hearsay.api, hermetically: fake models behind /analyze, a temp results directory behind
/results, no weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from hearsay import SR
from hearsay import api as api_mod
from hearsay.pipeline import DETECTOR_ORDER, FusionConstants
from test_pipeline import LOGITS, TOY, FakeModels, fake_dets


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("HEARSAY_RULE", raising=False)
    monkeypatch.delenv("HEARSAY_POLICY", raising=False)
    monkeypatch.delenv("HEARSAY_DETECTORS", raising=False)
    monkeypatch.delenv("HEARSAY_FUSION", raising=False)
    monkeypatch.setenv("HEARSAY_RESULTS", str(tmp_path / "out"))
    (tmp_path / "out" / "results").mkdir(parents=True)
    (tmp_path / "out" / "results" / "HGT1.wav.json").write_text(json.dumps({"filename": "HGT1.wav", "probability_synthetic": 0.12}))
    (tmp_path / "out" / "results" / "bad.wav.json").write_text("{not json")
    monkeypatch.setattr(api_mod, "_state", {"models": None, "consts": None, "load_seconds": None})
    return TestClient(api_mod.app)


def _wav(path: Path, seconds=2.0):
    x = (0.1 * np.random.default_rng(0).standard_normal(int(seconds * SR))).astype(np.float32)
    sf.write(path, x, SR)
    return path


def test_health_without_loading_models(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models_loaded"] is False
    assert body["rule"] == "zmean" and body["policy"] == "speech_gate"
    assert body["scorers"] == list(DETECTOR_ORDER) and body["n_results"] == 2
    assert body["version"]["fusion"] == "constants.json"


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
    consts = FusionConstants.from_dict(TOY, source="toy")
    fake = FakeModels(dets=fake_dets())
    monkeypatch.setattr(api_mod, "get_models", lambda: (fake, consts))
    wav = _wav(tmp_path / "clip one.wav")
    with wav.open("rb") as f:
        r = client.post("/analyze", files={"file": ("clip one.wav", f, "audio/wav")})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["filename"] == "clip one.wav" and doc["duration_s"] == 2.0
    assert doc["probability_synthetic"] == consts.fuse(LOGITS, "zmean").p
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
    assert doc["default_answer_applied"] is False and doc["probability_synthetic"] == doc["fusion"]["p_fused"]


def test_analyze_undecodable_upload_is_undetermined(client, monkeypatch, tmp_path):
    consts = FusionConstants.from_dict(TOY, source="toy")
    monkeypatch.setattr(api_mod, "get_models", lambda: (FakeModels(), consts))
    r = client.post("/analyze", files={"file": ("junk.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 200
    doc = r.json()
    assert doc["verdict"] == "undetermined" and doc["duration_s"] is None and doc["flag"] == "decode_error"
    assert client.post("/analyze", files={"file": ("empty.wav", b"", "audio/wav")}).status_code == 400
