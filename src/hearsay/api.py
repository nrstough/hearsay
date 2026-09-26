"""Thin HTTP API over `hearsay.pipeline` for the demo UI (frontend contract v0,
docs/handoffs/2026-09-26_frontend-contract.md). Not on the scoring path: the Docker image runs
scripts/run_pipeline.py; this wrapper imports the same `analyze_clip`, holds the models in
memory and serves the precomputed responses of a runner --out directory.

  uv run uvicorn hearsay.api:app --port 8000

  POST /analyze            multipart `file` (any audio) -> AnalyzeResponse (2-4 s cold-cached CPU)
  GET  /results/{filename} the precomputed AnalyzeResponse from $HEARSAY_RESULTS (a runner --out)
  GET  /results            the filenames available there
  GET  /health             status, whether the models are loaded, versions

Environment: HEARSAY_RESULTS (runner --out dir with results/<filename>.json), HEARSAY_RULE
(zmean | stack_nonlj, default zmean), HEARSAY_POLICY (speech_gate | none), HEARSAY_FUSION
(constants.json; unset = models/fusion_v0/constants.json), HEARSAY_DETECTORS (m1b for the
probe-only mode, default m1b,spectra,handcrafted), HEARSAY_PROBE, HEARSAY_HC, OMP_NUM_THREADS.
Models load lazily on the first /analyze (about 5 s) and one clip scores at a time.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from hearsay.detectors.base import ClipContext
from hearsay.pipeline import (
    CONSTANTS_PATH,
    DEFAULT_RULE,
    DETECTOR_ORDER,
    HC_DIR,
    PROBE_DIR,
    RULES,
    SCORER_FLAGS,
    FusionConstants,
    Models,
    analyze_clip,
    git_sha,
)

app = FastAPI(title="HEARSAY", version="0.1", description=__doc__)
_lock = threading.Lock()
_state: dict[str, Any] = {"models": None, "consts": None, "load_seconds": None}


def settings() -> dict[str, Any]:
    """The environment, read on every call so tests can monkeypatch it."""
    rule = os.environ.get("HEARSAY_RULE", DEFAULT_RULE)
    if rule not in RULES:
        raise HTTPException(500, f"HEARSAY_RULE must be one of {RULES}, got {rule!r}")
    flags = [f for f in os.environ.get("HEARSAY_DETECTORS", "m1b,spectra,handcrafted").split(",") if f]
    bad = [f for f in flags if f not in SCORER_FLAGS]
    if bad:
        raise HTTPException(500, f"HEARSAY_DETECTORS: unknown {bad}; use {sorted(SCORER_FLAGS)}")
    scorers = tuple(d for d in DETECTOR_ORDER if d in {SCORER_FLAGS[f] for f in flags})
    fusion = os.environ.get("HEARSAY_FUSION")
    m1_only = scorers == ("m1b_v3",) and not fusion
    results = os.environ.get("HEARSAY_RESULTS")
    return {
        "rule": rule, "policy": os.environ.get("HEARSAY_POLICY", "speech_gate"),
        "scorers": scorers if m1_only else DETECTOR_ORDER, "m1_only": m1_only,
        "fusion": None if m1_only else Path(fusion) if fusion else CONSTANTS_PATH,
        "probe": Path(os.environ.get("HEARSAY_PROBE", PROBE_DIR)),
        "hc": Path(os.environ.get("HEARSAY_HC", HC_DIR)),
        "results": Path(results) if results else None,
    }  # fmt: skip


def get_models() -> tuple[Models, FusionConstants | None]:
    """Load once, under a lock; later calls return the loaded pair."""
    with _lock:
        if _state["models"] is None:
            s = settings()
            t = time.time()
            consts = FusionConstants.load(s["fusion"]) if s["fusion"] is not None else None
            threads = int(os.environ["OMP_NUM_THREADS"]) if os.environ.get("OMP_NUM_THREADS") else None
            models = Models(device="cpu", probe_dir=s["probe"], hc_dir=s["hc"], threads=threads,
                            load_spectra="spectra_aasist" in s["scorers"])  # fmt: skip
            _state.update(models=models, consts=consts, load_seconds=round(time.time() - t, 2))
        return _state["models"], _state["consts"]


def _results_file(filename: str) -> Path | None:
    s = settings()
    if s["results"] is None:
        raise HTTPException(503, "HEARSAY_RESULTS is not set: no precomputed results to serve")
    name = Path(filename).name
    if not name or name != filename:
        raise HTTPException(400, "filename must be a bare file name")
    for cand in (s["results"] / "results" / f"{name}.json", s["results"] / f"{name}.json"):
        if cand.is_file():
            return cand
    return None


@app.get("/health")
def health() -> dict[str, Any]:
    s = settings()
    n = None
    if s["results"] is not None:
        d = s["results"] / "results" if (s["results"] / "results").is_dir() else s["results"]
        n = len(list(d.glob("*.json"))) if d.is_dir() else 0
    models = _state["models"]
    return {
        "status": "ok",
        "models_loaded": models is not None,
        "model_load_seconds": _state["load_seconds"],
        "rule": "m1b_only" if s["m1_only"] else s["rule"], "policy": s["policy"],
        "scorers": list(s["scorers"]),
        "results_dir": str(s["results"]) if s["results"] else None, "n_results": n,
        "version": {"git_sha": git_sha(), "models": models.version() if models else None,
                    "fusion": s["fusion"].name if s["fusion"] else "none"},
    }  # fmt: skip


@app.get("/results")
def list_results(limit: int = 2000) -> dict[str, Any]:
    s = settings()
    if s["results"] is None:
        raise HTTPException(503, "HEARSAY_RESULTS is not set: no precomputed results to serve")
    d = s["results"] / "results" if (s["results"] / "results").is_dir() else s["results"]
    names = sorted(p.name[: -len(".json")] for p in d.glob("*.json")) if d.is_dir() else []
    return {"n": len(names), "filenames": names[:limit]}


@app.get("/results/{filename}")
def get_result(filename: str) -> dict[str, Any]:
    p = _results_file(filename)
    if p is None:
        raise HTTPException(404, f"no precomputed result for {filename!r}")
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"{p.name}: not valid JSON") from e


def _analyze_path(path: Path) -> dict[str, Any]:
    models, consts = get_models()
    s = settings()
    with _lock:  # one clip at a time: the models and torch threads are shared
        return analyze_clip(ClipContext(path), models, consts, s["rule"],
                            apply_gate=s["policy"] == "speech_gate", scorers=s["scorers"])  # fmt: skip


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008 - FastAPI idiom
    """One uploaded audio file -> AnalyzeResponse. The file is decoded by the same loader as
    the runner; an undecodable upload gets the `undetermined` response, not an error."""
    name = Path(file.filename or "").name or "upload.wav"
    tmp = Path(tempfile.mkdtemp(prefix="hearsay_api_"))
    try:
        dest = tmp / name
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        if dest.stat().st_size == 0:
            raise HTTPException(400, "empty upload")
        t0 = time.time()
        doc = await run_in_threadpool(_analyze_path, dest)
        doc["api_seconds"] = round(time.time() - t0, 3)
        return doc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
