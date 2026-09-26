# HEARSAY

## What HEARSAY is

HEARSAY is a software-only audio authentication system, built at HackGT 13 (Sep 25–27, 2026) for the NSA HEARSAY challenge.
- **Input and output.** Any audio file in; out comes a probability that the voice is synthetic (0.0 real, 1.0 synthetic), plus a plain-English explanation of why.
- **How it decides.** A rule-based orchestrator picks forensic detectors per file, and a logistic stacker fuses their scores.
- **Deliverables.** `teamName_predictions.tsv` on NSA's 1,671-file test set, a Docker image that runs inference offline, and a README.
- **Validation.** Every rung is measured on a generator- and speaker-held-out split, and there is always a valid TSV on hand.

The earlier device-and-demo design is dropped (cut); see `docs/plan.md`.

## Team and ownership

Four-person team.

| Area | Owner |
|------|-------|
| Deep detectors (XLS-R probe, Spectra-AASIST, fine-tune), fold file, fusion, orchestrator, TSV | Nathan (@nrstough) |
| Engineered detectors: container/metadata and compression forensics | Teammate A (_fill in_) |
| Engineered detectors: spectral and prosody | Teammate B (_fill in_) |
| Engineered detectors: speaker-embedding drift, splice, ENF; Docker image from Sat morning | Teammate C (_fill in_) |

## Detector contract

Every detector builds against `src/hearsay/detectors/base.py`.
- **Input.** A `ClipContext`: path, 16 kHz mono audio decoded once and read-only, probe metadata, and a shared per-clip cache.
- **Output.** A `DetectorResult` with these fields:
  - `name`
  - `score`: finite, in [0, 1], and **increasing with synthetic likelihood**
  - `evidence`: a non-empty plain-English reason
  - `features`: a flat dict of finite floats
  - `status`: `ok`, `skipped` or `error`
  - `error`
- **Validation.** Violations raise `ValueError`; values are never clipped.
- **Running.** Always through `safe_run`, which never loses a row: a crash becomes `status="error"` with score 0.5.
- **Fusion.** Fusion v1 uses one column per detector, `logit(clip(score, 1e-4, 1 − 1e-4))`, when the status is `ok`; otherwise it adds a missing indicator.
- **Rules.** A change to `DetectorResult` fields must name the affected detector owners. Detectors never use filesystem timestamps or filenames.

## Model ladder and model rules

**Ladder** (full table and gates in `docs/plan.md`):
- M0: loader, TSV writer, constant rollback, fold file
- M1: frozen XLS-R probe, first learned TSV
- D-track: teammates' engineered detectors
- M3: Spectra-AASIST
- M5: fine-tuned XLS-R on the H100
- M4: fusion, orchestrator and explanation report
- K: Docker
- DOC: README and experiment log

**Model rules**
- Validate by generator and speaker, not by clip, using nested folds from the fold file.
- Log every TSV with its validation score.
- Cap any rung at a few hours.
- Keep a clean-only validation score beside any augmented one.

## Conventions

- **NSA metric and format.** Sources: `docs/nsa-challenge.md`, `data/nsa/HEARSAY_HackGT2026_Instructions.pdf`, and the kickoff notes in `docs/reports/2026-09-25_sponsor-questions.md`.
  - Judged 60% on MinDCF, with false alarms (real called synthetic) costing 4× a miss; about 70% of test files are real.
  - `hearsay.metrics` uses `C_FA = 4`, `C_miss = 1`, `π_synth = 0.3`, i.e. normalized minDCF = 9.33·P_FA + P_miss. Only score ranking matters.
  - Select everything on normalized minDCF and report EER alongside.
  - The sponsor's scoring code treats a higher score as bona fide. That's an open question with NSA; see `docs/plan.md`, "Score direction".
  - Submit `teamName_predictions.tsv`: header `filename<TAB>cm-score`, probabilities (1.0 = synthetic), all rows in the template's order. `hearsay.submission.write_submission` enforces the header, unique filenames, finite scores in [0, 1] and no-overwrite; the template's row order comes from the manifest passed to the submission scripts. The final file is due 8 AM Sunday.
- **16 kHz mono everywhere.** Every loader resamples and downmixes at the boundary; nothing downstream sees another format.
- **Every experiment appends a row to `submissions/log.csv`** with timestamp, rung, validation score and submission path.
  - Columns: `timestamp,rung,validation_score,validation_score_clean_only,csv_path,notes`. `csv_path` is the historical column name for the submission path.
  - The clean-only column exists because of the model rule above; leave it equal to `validation_score` when no augmentation is used.
  - `validation_score` is normalized minDCF on the outer holdout; put EER in `notes`.
- **Offline scoring path.** No hosted API or LLM on the scoring path; the Docker image runs offline.
- **Secrets in `.env` only, never committed.** `.env` is gitignored.
- **Environment.** Run `uv sync` once, then `uv run <cmd>` from the repo root (Python 3.12, pinned in `.python-version`). Data, weights, model outputs, submission TSVs and audio files are gitignored; only `submissions/log.csv` is tracked.
- **Tests and lint.**
  - Tests: `uv run pytest` (tests in `tests/`). Markers `needs_data`, `needs_weights` and `slow` are for anything that needs gitignored assets.
  - Lint: `uv run ruff check .`
  - `tests/test_docs_consistency.py` pins these docs to the code: metric constants, no dropped-hardware wording, and the plan's technique coverage.
- Doc layout: run specs in `docs/specs/`, plan files and reports in `docs/reports/`, handoffs in `docs/handoffs/`, consult records in `docs/consults/`. No feature-spec layer.
- Codex review: `.claude/review-plan.sh <plan.md>` and `.claude/review-audit.sh <run-spec.md>`, with HEARSAY rubrics in `.claude/codex-*-prompt.md`. Requires the Codex CLI; without it, the plan-review Claude critique is the gate.
- Known setup gotcha: the installed torchaudio (2.11) cannot `torchaudio.load()` files without the separate `torchcodec` package, which is not installed. `soundfile` and `librosa` load audio fine.
- Known setup gotcha: LightGBM and torch each ship their own `libomp.dylib`; importing both in one process on macOS segfaults or hangs (either order). Only `scripts/train_handcrafted.py` imports lightgbm; bundles store the dumped trees and `hearsay.trees` runs them in numpy, so detectors, tests and the orchestrator never import it. A test pins this.

## Working rules for Claude Code

- Use `/plan-review` for any rung expected to take more than an hour.
- Never overwrite a submitted TSV. Every submission gets a new file and a new log row.
- Never train on the validation split (including scalers, calibration, and stacking folds).
- Ask before renting a cloud GPU.
- When a time box runs out, stop and report rather than push through.
- Freeze is Sunday morning (Sep 27): after it, only fixes that unbreak the TSV or the Docker image.

## Sources of truth

- `docs/nsa-challenge.md` and `data/nsa/HEARSAY_HackGT2026_Instructions.pdf`: the NSA brief and instructions. Authoritative on rules and scoring.
- `docs/plan.md`: the working plan (software-only, rewritten Sep 26).
- `docs/scoping.md` and `docs/master-doc.md`: historical, pre-rescope background.

Where these and this file disagree, the NSA documents win on rules, `docs/plan.md` wins on plan and scope, and `submissions/log.csv` wins on numbers.

## Pre-event work

Done before the coding window opened at 8:00 pm Friday, Sep 25, 2026. Copy into Devpost.

- Created the repository scaffold: `pyproject.toml`, `.python-version`, `.gitignore`, and empty directories (`data/`, `weights/`, `docs/`, `submissions/`, `scripts/`, `src/hearsay/`).
- Installed the Python dependencies (torch, torchaudio, transformers, huggingface_hub, librosa, soundfile, numpy, scipy, scikit-learn, lightgbm, pandas, pydub, ffmpeg-python) into a uv-managed virtualenv, and confirmed ffmpeg is installed. Added pytest and ruff as dev dependencies.
- Downloaded pretrained weights and public datasets into the gitignored `weights/` and `data/` directories (event rules allow pre-event downloads): the models and datasets listed under AI use disclosure. Nothing was trained or run on them before the event.
- Copied general-purpose Claude Code skills (plan-review, p3, handoff, consult, diagnose, validate-training-data, test-runner, doc-sync, zoom-out) into `.claude/skills/` and added HEARSAY-specific notes to them.
- Copied the Codex plan-review and audit scripts into `.claude/` and added HEARSAY-specific checks to their rubrics.
- Added project boilerplate: `README.md`, `.env.example`, `.claude/settings.json` (Claude Code permissions), pytest and ruff config, and empty `tests/` and `docs/` subdirectories.
- Wrote this `CLAUDE.md`, added the planning docs (`docs/scoping.md`, the external scoping memo; `docs/plan.md`, the working plan), a header-only `submissions/log.csv`, and `scripts/first-commit.sh` (repo creation and first commit, run at 8:00 pm).
- No application code (loader, pipeline, model, scoring) was written before the event.

## AI use disclosure

Keep this current through the weekend; copy into Devpost. The rules require crediting frameworks and stating what AI was used for vs. what we built.

**AI tools**
- Claude Code (Anthropic): used for environment setup, writing project docs and scripts, and as a coding assistant during the event. So far:
  - the loader, the TSV writer and the metrics
  - the M1 probe pipeline
  - data pulls
  - the software-only rescope: plan, detector contract, doc-consistency tests
  - reading the NSA brief, instructions and scoring code
  - the D-track engineered detectors (Sat Sep 26, CPU-only chat): handcrafted spectral + prosody, compression forensics, container/metadata, ENF and splice detectors, their fold-validated training and fusion score exports (`outputs/detector_scores/*.csv`), the container and high-band inventories, and the test-set band-match finding (`docs/reports/2026-09-26_cpu-detectors.md`)
  - M3 (Sat Sep 26, Spectra-AASIST chat): the Spectra-AASIST score stream through the shared band-matched input path with test-length crops (`scripts/score_spectra.py`, `hearsay.spectra`), its short-clip pad-mode pilot, direction and failure gates, the hermetic tests (`tests/test_spectra.py`) and the report (`docs/reports/2026-09-26_m3-spectra.md`)

  _Add specifics as they happen._
- `/plan-review` workflow (a Claude Code skill): used to plan and review rungs taking over an hour. Its plan review and post-commit audit call OpenAI Codex (Codex CLI, via `.claude/review-*.sh`) when the CLI is installed. Used for the software-only rescope (Sep 25–26): Codex plan review, Claude critique and exploration agents.

**Pretrained models / checkpoints** (name, source, license, how used)
Downloaded before the event to `weights/`; fill in "how used" as each is actually used.
- XLS-R 300M, `facebook/wav2vec2-xls-r-300m` (Hugging Face, Meta), Apache-2.0: core SSL front end (planned M1/M5).
- WavLM Large, `microsoft/wavlm-large` (Hugging Face, Microsoft): bake-off challenger. No license on the Hugging Face card; WavLM was released through Microsoft's unilm repo (MIT). _Verify before submission._
- WavLM Base, `microsoft/wavlm-base` (Hugging Face, Microsoft): bake-off / speed spare. No license on the Hugging Face card; WavLM was released through Microsoft's unilm repo (MIT). _Verify before submission._
- Spectra-AASIST, `lab260/Spectra-AASIST` (Hugging Face): M3 score stream (`outputs/detector_scores/spectra_aasist.csv`), run off the shelf with no training through the shared band-matched input path, as a second deep score for fusion beside the XLS-R probe. License unclear: repo header says Apache-2.0, model card text says MIT. Output index 0 = spoof, 1 = bonafide.

**Public datasets** (name, source, license, how used)
Downloaded before the event to `data/`; fill in "how used" as each is actually used.
- ASVspoof 2019 LA (University of Edinburgh DataShare, doi handle 10283/3336), Open Data Commons Attribution License: training/baseline data.
- In-the-Wild (Müller et al. 2022; `mueller91/In-The-Wild` on Hugging Face), license listed as CC-BY-SA-4.0 on Hugging Face and Apache-2.0 on deepfake-total.com: held-out stress test only, never trained on.
- MLAAD (`mueller91/MLAAD` on Hugging Face, gated, non-commercial notice): 6,390 English synthetic clips pulled during the event (30 per model across 143 models, plus 300 per model for the ElevenLabs, Gemini and Qwen3 families named on the kickoff slide). Planned training data.
- NSA HEARSAY data (provided by the sponsor during the event): 1,671-file test set, a resampled LJ Speech subset (real), DiffSSD (synthetic, Purdue), and the ASVspoof5 evaluation code for MinDCF.
- LibriSpeech dev-clean/dev-other/test-clean/test-other (openslr.org/12, CC BY 4.0): 5,323 multi-speaker bona fide clips; 4,000 of them in the NSA training sample.

**Planned, not yet downloaded or used** (move each item up to the list above once it is actually used):
- ECAPA-TDNN speaker embeddings (SpeechBrain), for the speaker-drift detector.
- VCTK as a second multi-speaker bona fide source (LibriSpeech dev+test is in use; see above).
- Full LJ Speech (keithito.com).
- ReplayDF (optional).
- ExifTool, for metadata forensics. _Not used: ffprobe (FFmpeg) covered the embedded fields, and the test set carries none beyond one encoder tag._

**Frameworks and libraries**
- PyTorch, torchaudio, Hugging Face transformers and huggingface_hub, librosa, soundfile, NumPy, SciPy, scikit-learn, LightGBM, pandas, pydub, ffmpeg-python, FFmpeg.

**What we built vs. what AI did**
- _Fill in per component (each detector, fusion, orchestrator, Docker image) before submission._

## When a teammate opens this repo

The skills in `.claude/skills/` and the permissions in `.claude/settings.json` load automatically when you run Claude Code in this repo; nothing to install. Setup: `uv sync`, then `cp .env.example .env`. For Codex review in `/plan-review`, install the Codex CLI (`npm install -g @openai/codex`) and run `codex login`; it's optional. Personal overrides go in `.claude/settings.local.json` (gitignored). The skills:

| Skill | Use it for |
|-------|-----------|
| `/plan-review` | Any rung or change expected to take over an hour: problem → failure modes and tests → plan → execute. Includes this project's standing failure modes. |
| `/p3` | Executing a plan that has already been approved. |
| `/diagnose` | A bug you can reproduce, e.g. wrong scores, loader output, or a detector that errors. |
| `/validate-training-data` | Before training on a new dataset or split: generator leakage, imbalance, format leaks. |
| `/test-runner` | Running the tests and diagnosing failures. |
| `/doc-sync` | Checking this file and `docs/` against the code, especially the disclosure and log. |
| `/handoff` | Ending a session: writes a prompt so the next chat (or teammate) picks up cleanly. |
| `/consult` | Writing a self-contained question for an outside LLM or expert, and saving the answer. |
| `/zoom-out` | Getting a map of an unfamiliar part of the code (e.g. someone else's detector). |
