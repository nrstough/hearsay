# Frontend ↔ backend contract (Sat Sep 26, 2026, ~06:30)

For Hrushi. Short version: the per-file analysis you want to display exists today as Python functions and produces real evidence sentences, but there is no HTTP server and no single "analyze one file" function yet. Build the UI against the JSON contract below and a static dump of the 1,671 test files; the live endpoint is a thin wrapper that lands with the pipeline runner the Docker lane is building. Nothing here is a graded deliverable on its own, but per-file explanations and the routing log feed the 20% explainability and orchestration rubric, so the UI should show exactly those.

## What exists right now

| Piece | Callable today | Notes |
|---|---|---|
| Every engineered detector on one file, with evidence | yes, CPU, about 5 s cold for the first file, well under 1 s after | `import hearsay.detectors.engineered`, then `safe_run(det, ClipContext(path))` for each of `all_detectors()`. Returns name, status, score, evidence sentence, features. Sample output below is from a real test file. |
| Non-speech gate + default-answer policy | yes | `hearsay.detectors.speech_gate`; `apply_default_answer(fused_score, is_speech)` |
| M1 / M1b deep score on one file | yes, as functions | `hearsay.embed.load_backbone(device="cpu")`, `prepare_segment`, `embed_segment`, then `Probe.load(dir).llr(e)`; see the `score()` closure in `scripts/make_probe_csv.py` lines 60–67. About 1–3 s per file on CPU after a 15 s model load. |
| M3 Spectra-AASIST on one file | yes, as functions | `hearsay.spectra.load_spectra("cpu")`, then the scoring path in `scripts/score_spectra.py` (zero-padded windows on the prepared clip). About 0.5 s per file on CPU. |
| Fusion rule | batch only | `scripts/fuse.py` fuses the exported score files for all 21,671 rows. It computes each detector's standardization (inner-fold mean and std) at run time and prints the stacker weights; it does not save them. A live endpoint needs those constants persisted. |
| HTTP API | no | No web framework is installed. `uv add fastapi uvicorn` is the obvious choice; keep it out of the Docker scoring path. |
| Orchestrator / explanation report | no | Planned M4. The routing log is derived from detector features (container `lossy`, compression `bw_hz`, ENF presence, splice seams, `speech_gate.is_speech`). |

## The contract: `AnalyzeResponse` v0

One object per file. Field names are stable; fields marked *pending* are empty until the runner exists.

```json
{
  "filename": "HGT1013455.wav",
  "duration_s": 4.74,
  "probability_synthetic": 0.12,
  "verdict": "real",
  "is_speech": true,
  "default_answer_applied": false,
  "fusion": {
    "rule": "stack_nonlj",
    "inputs": {"m1b_v3": -1.84, "spectra_aasist": -2.31, "handcrafted_v5": 0.42},
    "weights": {"m1b_v3": 0.9, "spectra_aasist": 1.6, "handcrafted_v5": 0.3}
  },
  "detectors": [
    {
      "name": "handcrafted",
      "role": "fused",
      "status": "ok",
      "score": 0.2095,
      "evidence": "spectral/prosody features real-like (P=0.21): spectral contrast, band 0 1.1 SD below real speech (toward synthetic); CQCC 1 frame-to-frame change 2.6 SD above real speech (toward synthetic); MFCC 13 variability 1.7 SD above real speech (toward synthetic)",
      "features": {"hf_ratio_4k_mean": 0.0707, "voiced_frac": 0.81},
      "seconds": 0.21,
      "error": null
    },
    {
      "name": "container",
      "role": "routing",
      "status": "ok",
      "score": 0.5,
      "evidence": "wav/pcm_s16le 16000 Hz mono 16-bit, 1 tag(s) [encoder=Lavf58.29.100]: no class evidence in the container; written by FFmpeg (libavformat), so the file was re-muxed or transcoded at least once",
      "features": {"lossy": 0.0, "is_pcm_wav": 1.0, "sample_rate": 16000.0},
      "seconds": 0.1,
      "error": null
    },
    {
      "name": "enf",
      "role": "evidence",
      "status": "ok",
      "score": 0.5,
      "evidence": "no mains hum at 50 or 60 Hz (best 60 Hz candidate: median SNR 2.8 dB, above 12 dB in 0% of frames): no environment evidence either way",
      "features": {"enf_present": 0.0, "enf_snr_db": 2.81},
      "seconds": 0.0,
      "error": null
    }
  ],
  "routing_log": [
    "container: PCM WAV, not lossy, FFmpeg-written; compression forensics run for evidence, not score",
    "speech_gate: is_speech=true (voiced 81% of frames); default-answer policy not applied",
    "fusion: stack_nonlj over m1b_v3, spectra_aasist, handcrafted_v5"
  ],
  "version": {"git_sha": "271c24b", "models": {"m1": "m1_wav2vec2-xls-r-300m_L7_20260926-0521", "handcrafted": "hc_lgbm_20260926-055451", "spectra": "lab260/Spectra-AASIST"}}
}
```

Rules the UI can rely on:
- `score` is always in [0, 1] and higher means more synthetic; `evidence` is always a non-empty sentence; `status` is `ok`, `skipped` or `error`, and an errored detector has score 0.5 and an `error` string. This is enforced by the detector contract (`src/hearsay/detectors/base.py`).
- `role` says how the detector is used: `fused` (enters the score), `evidence` (shown, never fused: enf, splice, speaker_drift, compression), `routing` (container), `gate` (speech_gate).
- `verdict` is `synthetic` when `probability_synthetic` ≥ 0.5, `real` below, `undetermined` when `is_speech` is false or the file failed to decode.
- `probability_synthetic` for the test set equals the logged TSV's value for that file.

## Build against a static dump today

**Update 07:30: the dump already exists.** The runner produced `outputs/runner/full_20260926/results/<filename>.json` for all 1,671 test files (AnalyzeResponse v0, rule zmean, gate on; the run also verified the TSV against the logged submission). Copy that directory into the frontend's fixtures instead of running the script below. The live API is `uv run uvicorn hearsay.api:app --port 8000` with `HEARSAY_RESULTS=outputs/runner/full_20260926` for `GET /results/{filename}`; `POST /analyze` takes a multipart file and answers in about 1.5 s after the first request.

Every field except `fusion.weights` and a full `routing_log` can be produced now for all 1,671 test files, with the engineered detectors run live on CPU and the deep scores taken from the exports. This gives a real dataset for the UI without a server. About 15 minutes on 6 cores.

```bash
cd ~/Projects/hearsay && uv run python - <<'EOF'
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import hearsay.detectors.engineered  # registers every engineered detector
from hearsay.detectors import all_detectors
from hearsay.detectors.base import ClipContext, safe_run
from hearsay.detectors.speech_gate import apply_default_answer

ROLE = {"handcrafted": "fused", "compression": "evidence", "container": "routing", "enf": "evidence",
        "splice": "evidence", "speaker_drift": "evidence", "speech_gate": "gate"}
test = pd.read_csv("outputs/manifests/nsa_test.csv")
scores = {n: pd.read_csv(f"outputs/detector_scores/{n}.csv").query("split == 'test'").set_index("path")
          for n in ["m1b_v3", "spectra_aasist", "handcrafted_v5"]}
tsv = pd.read_csv(sorted(Path("submissions").glob("*_M4_fusion_stack_nonlj_*.tsv"))[-1], sep="\t").set_index("filename")
out = Path("outputs/frontend"); out.mkdir(exist_ok=True)
dets = all_detectors()
for _, row in test.iterrows():
    ctx = ClipContext(Path(row.path)); items = []
    for d in dets:
        t0 = time.time(); r = safe_run(d, ctx)
        items.append({"name": r.name, "role": ROLE.get(r.name, "evidence"), "status": r.status, "score": r.score,
                      "evidence": r.evidence, "features": r.features, "seconds": round(time.time() - t0, 3), "error": r.error})
    gate = next(i for i in items if i["name"] == "speech_gate")
    is_speech = bool(gate["features"].get("is_speech", 1.0))
    p = float(tsv.loc[row.filename, "cm-score"])
    p = float(apply_default_answer(np.array([p]), np.array([is_speech]))[0])  # array in, array out
    doc = {"filename": row.filename, "duration_s": round(ctx.audio.size / 16000, 2), "probability_synthetic": p,
           "verdict": "undetermined" if not is_speech else ("synthetic" if p >= 0.5 else "real"),
           "is_speech": is_speech, "default_answer_applied": not is_speech,
           "fusion": {"rule": "stack_nonlj", "inputs": {n: float(s.loc[row.path, "logit"]) for n, s in scores.items()}, "weights": {}},
           "detectors": items, "routing_log": [], "version": {"git_sha": "271c24b", "models": {}}}
    (out / f"{row.filename}.json").write_text(json.dumps(doc))
print("wrote", len(test), "files to outputs/frontend/")
EOF
```

`outputs/` is gitignored, so commit the frontend's copy of the dump wherever the UI keeps fixtures, not under `outputs/`. `apply_default_answer(fused, is_speech)` takes and returns arrays (`src/hearsay/detectors/speech_gate.py:113`); the snippet wraps the scalars accordingly.

## The live endpoint, when the runner exists

`POST /analyze` with a file → `AnalyzeResponse`; `GET /results/<filename>` → the precomputed object. The handler is: decode once into a `ClipContext`, run the deep scorers and the engineered detectors, standardize each fused logit with the persisted constants, apply the fusion rule, apply the gate policy, assemble the routing log. That is the same function the Docker image needs (`scripts/run_pipeline.py` in `docs/handoffs/2026-09-26_docker-handoff.md`), so the endpoint should import it, not reimplement it. Expect 2–4 s per file on CPU with the models held in memory. Owner: whoever builds the runner; the wrapper itself is under an hour.

## Boundaries

- **No LLM or hosted service on the scoring path.** A UI may narrate or summarize the evidence sentences with a language model, but `probability_synthetic` and every `score` must come from the pipeline alone, and the Docker image runs offline.
- **Files:** keep the frontend in its own directory (e.g. `web/`) and the API wrapper in one new file; do not edit `src/hearsay/` or `scripts/` owned by other lanes (list in `docs/code-map.md`). Stage only your own files; never `git add -A`; don't push without asking Nathan.
- **Time:** the TSV, the Docker image and the README are the deliverables; the UI must not pull Nathan or the Docker owner off them before Sunday 08:00.

## Rewire note (Sat 09:50), after commit 19925a3

The committed frontend (`src/app`, `src/components`) calls `python -m hearsay.analyzer`, a separate re-implementation under `src/hearsay/analyzer.py`. That module does not run the shipped detectors: its "deep SSL probe" is a zero-crossing-rate rule that returns a fixed 0.84 or 0.16, its verdict is a hand-weighted sum of toy analyses, and `src/app/api/forensic/chat/route.ts` answers with canned keyword-matched text (one reply describes a "frequency cliff at 16.0 kHz", which cannot exist in 16 kHz audio). Shown to a judge as how HEARSAY decides, it contradicts the README and the submitted TSV. Required changes, in order:

1. **Score from the real pipeline.** Replace the `execFile` call in `src/app/api/forensic/analyze/route.ts` with a `POST` of the uploaded file to the running API (`uv run uvicorn hearsay.api:app --port 8000`, endpoint `/analyze`), or with `uv run python scripts/run_pipeline.py --in <tmpdir> --out <tmpdir>/out` and read `out/results/<file>.json`. Both return `AnalyzeResponse` v0 (this document): `probability_synthetic`, `verdict`, `detectors[]` with real evidence sentences, `fusion`, `routing_log`, `version`. Map the components to those fields; the eight-modality cards correspond to `detectors[].name` (handcrafted covers spectral and prosody; container, compression, enf, splice, speaker_drift, speech_gate are one each; the deep score is `fusion.inputs`).
2. **The copilot must not present canned text as analysis.** Either remove it, or label it "guide" and have it quote only fields from the real response (evidence sentences, routing log, the fusion inputs), or call a language model over that JSON, which is allowed because it is off the scoring path and not in the Docker image.
3. **Move `src/hearsay/analyzer.py` out of the package** (for example into `web/tools/`) or delete it once the route is rewired; while it stays, it fails `uv run ruff check .` (two unused imports) and every lane's lint gate.
4. Keep the batch and export-tsv routes only if they call the same pipeline; a TSV produced by the toy analyzer must never be exported.

The static dump for all 1,671 test files is the fastest way to build and demo without the server. **Use `outputs/runner/v2_full/results/*.json`** (Sat 12:15 onward: the shipped rule A3 w0.2 + E / `models/fusion_v2/constants.json`, our direction, 0 rows differing from the submitted TSV). The older `outputs/runner/full_20260926/` dump is the rejected zmean rule and must not be shown. Under the shipped rule `fusion.weights` has four keys (`m1b_v3` 0.6, `handcrafted_v5` 0.2, `m5_xlsr_ft` 0.2, `spectra_aasist` 0.0) and `fusion.detail.final` reads `A3_w0.2_E`; render the weights from the JSON, never as fixed text.


## Rewire done (Sat Sep 26, ~22:10, consult chat on Nathan's instruction)

Items 1–4 above are done without changing the UI's layout or components: `src/app/api/forensic/analyze` forwards uploads to `hearsay.api` (`HEARSAY_API_URL`, default port 8000) and adapts the response through `src/lib/hearsay.ts`; the batch route reads the submitted TSV and the runner's per-file JSON (`HEARSAY_RESULTS`); the export route serves `submissions/CrossExam_predictions.tsv` byte for byte; the chat route is a rule-based guide that quotes only the response's fields; the keyword and microphone fallbacks that invented verdicts are gone (a failed request shows the error); `src/hearsay/analyzer.py` is removed; the fabricated presets are replaced by `demo/` clips scored by the shipped pipeline (`/api/forensic/presets`), and the fixed labels that stated numbers we never measured (minDCF 0.124, a 16 kHz cutoff) now show the draft-review number and the pipeline's own fields. Verified in the browser and with curl: presets, batch (1,671 rows, 457 above 0.5), export hash fb783076, upload of a demo clip through the app route.
