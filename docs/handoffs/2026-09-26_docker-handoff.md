# Handoff: K, the Docker image (Sat Sep 26, 2026, ~05:45)

**Purpose of this chat:** build the offline CPU inference image that NSA will run, and prove it produces the same TSV the logged submissions do. This is a required deliverable with **no owner and no file yet**. Gates from `docs/plan.md`: **first amd64 image with the M1 payload by Sat 14:00**; rebuild with fusion Sun 00:00 with a 3-file smoke test under `--network none` and a CPU-vs-Mac parity check on 50 files; final TSV Sun 05:00; DM before 08:00 Sunday. Time box: about 3 hours for the first image.

Read `CLAUDE.md`, `docs/STATUS.md`, `docs/architecture.md` (sections 2, 4 and 10.7) and `docs/code-map.md` first. This file tells you what to build and what already exists.

## Interface (from `README.md`)

```bash
docker run --network none -v <test_dir>:/data:ro -v <out_dir>:/out hearsay
```

In: a directory of audio files (NSA's test set is 1,671 WAVs, 16 kHz mono PCM, 3.0–13.6 s). Out: `/out/<teamName>_predictions.tsv`, header `filename<TAB>cm-score`, one row per file, probabilities with **1.0 = synthetic**, all rows in the template's order. Row order comes from NSA's template `HearsayScoreKey4TeamX.tsv` (copied at `data/nsa/HearsayScoreKey4TeamX.tsv`), so accept an optional mounted template and fall back to sorted filenames. The team name is still an open question (STATUS.md); make it an environment variable with a placeholder default.

## What already exists

| Piece | Where | Notes |
|---|---|---|
| The one audio path | `hearsay.audio.load_audio` | FFmpeg subprocess, so the image needs the `ffmpeg` apt package |
| M1/M1b scorer, audio to TSV | `scripts/make_probe_csv.py --probe <dir> --test-dir <dir> [--manifest <tsv> --id-col filename]` | loads XLS-R via `hearsay.embed.load_backbone` (device defaults to CPU when MPS is absent), runs `prepare_segment` (band-limit, trim, 8 s cap) and the probe; validates and logs the TSV; runs a silence-and-chord preflight first |
| Best probe today | `models/m1_wav2vec2-xls-r-300m_L7_20260926-0521/probe.joblib` (M1b v3: holdout minDCF 0.072, In-the-Wild 0.342) and `…-0518` (M1 v3) | `meta.json` beside each; the main chat decides which ships |
| M3 scorer | `scripts/score_spectra.py --pad-mode zero --device cpu` | Spectra-AASIST; `hearsay.spectra.load_spectra` patches the checkpoint's hub id to the local `weights/wav2vec2-xls-r-300m`; about 0.1 s per clip on MPS, expect 3–5× that on CPU |
| Handcrafted detector | `hearsay.detectors.handcrafted` with the newest `models/hc_lgbm_<stamp>/model.joblib` | numpy tree evaluator; never imports lightgbm; recomputes v4 families at inference |
| Other engineered detectors | `import hearsay.detectors.engineered`; `scripts/score_detector.py` | rule-based; evidence and routing only |
| Non-speech gate + default-answer policy | `hearsay.detectors.speech_gate` (`apply_default_answer`) | rule-based; gated files rank below all speech files; the runner must apply it after fusion |
| Speaker drift | `hearsay.detectors.speaker_drift` | needs the `speechbrain` dependency (now in `pyproject.toml`/`uv.lock`) and `weights/spkrec-ecapa-voxceleb` (89 MB) copied into the image; evidence only, never fused |
| M5 scorer | `scripts/m5_score.py --model <dir> --files … --out <csv>` | CPU path with a sha check; only if M5 clears its gate at 12:00; its checkpoint (truncated backbone + head, safetensors, about 850 MB) will be on the HF Hub or a GitHub release, never in git |
| Fusion v0 | `scripts/fuse.py --detectors m1b_v3 handcrafted_v4 spectra_aasist …` | fuses **exported score files** under `outputs/detector_scores/`; it does not run detectors on audio |
| Submission writer | `hearsay.submission.write_submission`, `append_log`, `preflight`, `list_test_files` | refuses to overwrite; validates header, uniqueness, range, order |
| Weights | `weights/wav2vec2-xls-r-300m` (1.2 GB), `weights/Spectra-AASIST` (1.3 GB); exclude `weights/wavlm-*` (1.5 GB, unused) | all local; the image must never download |

**The missing piece is an end-to-end runner:** audio directory → each shipped detector → fusion rule → TSV, inside the container. Today only M1 goes from audio to TSV in one script. Coordinate the runner's contract with the main chat (it owns fusion and the TSV): the cleanest shape is `scripts/run_pipeline.py` that (1) lists files, (2) runs each shipped scorer to a per-detector score file in the same `path, score, logit` shape the exports use, (3) applies the fusion rule the main chat names, (4) calls `write_submission`. Until fusion is frozen (Sat 22:00), the M1b probe alone is the payload and step 3 is identity.

## Design (from `docs/plan.md`, "Docker")

- **Base:** `python:3.12-slim-bookworm` plus `apt-get install ffmpeg`. Python is pinned to 3.12 in `.python-version`.
- **Dependencies:** the project is uv-managed (`pyproject.toml`, `uv.lock`). The lockfile resolves torch 2.14.0 and torchaudio 2.11.0; install the **CPU** wheels from the PyTorch CPU index and skip the lockfile's CUDA wheels (`uv sync --no-install-package torch --no-install-package torchaudio`, then install the CPU pair explicitly), or build a requirements list from the lock. transformers is 5.17.0. Keep the editable install of `src/hearsay`.
- **Offline:** `ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. `hearsay.spectra` already sets the first. Test every run with `--network none`.
- **Weights in the image:** copy `weights/wav2vec2-xls-r-300m`, `weights/Spectra-AASIST`, `weights/spkrec-ecapa-voxceleb`, the shipped `models/m1_*/probe.joblib` and `models/hc_lgbm_*/model.joblib`. `.dockerignore` must exclude `weights/wavlm-*`, `.venv`, `data/`, `outputs/`, `submissions/`, `.cache`, `.git`. Expect a 4–5 GB image.
- **Runtime:** one heavy model loaded at a time (XLS-R, then Spectra, then the CPU detectors), a resumable per-file cache (one JSON line per scored file, so a crash resumes), a new output file per run, never an overwrite. Thread count from `OMP_NUM_THREADS`/`torch.set_num_threads`, default to all cores.
- **Platform:** this Mac is arm64; NSA's box is almost certainly amd64. Build with `docker buildx build --platform linux/amd64` (emulated, slow: budget an hour for the first build) or build natively on a Linux box. Verify with `docker run --platform linux/amd64 hearsay python -c "import platform; print(platform.machine())"`.

## Checks before calling it done

1. **Smoke:** 3 test files in, a valid 3-row TSV out, with `--network none`, on amd64.
2. **Parity:** score the same 50 test files in the image and on the Mac (the logged TSV `submissions/20260926-0522_M1_*.tsv` has M1b v3's scores; `scripts/tsv_from_embeddings.py` regenerates from embeddings without a GPU). Require Spearman > 0.99 and max absolute score difference < 0.05. Differences come from CPU fp32 vs MPS and from the FFmpeg version; both are expected to be tiny.
3. **Preflight:** silence and a synthetic chord through the image scorer, finite and reproducible (`hearsay.submission.preflight` does this; `make_probe_csv.py` calls it).
4. **Order:** the TSV row order equals the template's order when a template is mounted.
5. **Timing:** log wall time for the full 1,671 files on CPU; M1 alone should be well under an hour, Spectra adds roughly 10–15 minutes.
6. **Full suite still green:** `uv run pytest -q` (301 passing at 05:38), `uv run ruff check .`.

## Do not touch

`src/hearsay/embed.py`, `probe.py`, `submission.py`, `spectra.py`, `m5_*.py`; `scripts/make_probe_csv.py`, `train_probe.py`, `extract_embeddings.py`, `fuse.py`, `score_spectra.py`, `m5_*.py`; `submissions/`; `splits/`; the detector modules. Your files: `Dockerfile`, `.dockerignore`, `docker/` (entrypoint, README section), `scripts/run_pipeline.py` if the main chat agrees to that name, `tests/test_docker_*.py` (hermetic; anything needing the image or weights is marked `slow` and `needs_weights`). Stage only your own files with `git add <paths>`; never `-A`; don't push without asking Nathan. Never `import lightgbm` anywhere on the inference path.

## Environment

```bash
cd ~/Projects/hearsay && uv sync
uv run pytest -q          # 301 passed at 05:38
docker --version          # Docker Desktop must be running; buildx for amd64
ls weights/ models/       # weights present; models/ has the probe and hc bundles
```

`models/` and `outputs/` are gitignored and not archived; the probe and hc bundles regenerate in about 25 minutes from the commands in `docs/reports/2026-09-26_cpu-detectors.md` and `scripts/train_probe.py`. Weights exist only on this Mac and its external drive.

## Analytical notes

- The test set is band-limited at 7.2 kHz and `prepare_segment` reproduces that on every clip, so the image must use the repo's own loader and preparation, never a re-implementation.
- The sponsor's scoring code treats a higher score as bona fide, the opposite of the instructions. We submit 1.0 = synthetic and never flip; the draft review at 14:00 settles it. The image's output must keep that convention.
- M1 reads only layer 7 of the 24-layer backbone; dropping layers 8–24 at load time is a safe 70% compute saving with bit-identical output, if wall time matters. Not needed for correctness.
- M5, if it ships, needs its checkpoint fetched at build time from the Hub or a release (not at run time) and the same sha check `m5_score.py` performs.

## Pointers

`CLAUDE.md` → `docs/plan.md` (Docker, Timeline, Cut order) → `docs/architecture.md` §2, §4, §10.7 → `docs/code-map.md` ("M1 deep path", "M3 … submission machinery", recipe d) → `scripts/make_probe_csv.py` → `src/hearsay/submission.py` → `scripts/score_spectra.py` → `scripts/fuse.py` docstring.
