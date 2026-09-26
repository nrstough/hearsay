# HEARSAY

## What HEARSAY is

HEARSAY is a software-only audio authentication system, built at HackGT 13 (Sep 25–27, 2026) for the NSA HEARSAY challenge.
- **Input and output.** Any audio file in; out comes a probability that the voice is synthetic (0.0 real, 1.0 synthetic), plus a plain-English explanation of why.
- **How it decides.** A rule-based orchestrator picks forensic detectors per file, and a logistic stacker fuses their scores.
- **Deliverables.** `CrossExam_predictions.tsv` (team Cross Exam) on NSA's 1,671-file test set and a README. A Docker image that runs inference offline was listed in the brief but is not required (NSA, Sat Sep 26 ~12:00); ours is built and kept as reproducibility evidence.
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
- Known gotcha: `scripts/extract_embeddings.py` records a transient decode failure as 1 s of zeros with `flag = decode_error` in the set's `manifest.csv`, silently (about 0.5% of rows under memory pressure, seen with three WavLM processes on the 18 GB Mac, Sat Sep 26 evening). Check the flag column after any extraction under load; `scripts/repair_decode_errors.py` re-embeds only the flagged rows. The shipped M1b v3 sets (`nsa_train_sample_v3`, `asv19_addon_v3`, `nsa_test_v3`, `itw_stress_v3`) were checked and carry no such rows.
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
- Claude Code (Anthropic): used for environment setup, writing project docs and scripts, and as the coding assistant throughout the event, in parallel sessions that each started from a written handoff under `docs/handoffs/` and reported back to an oversight session. What it wrote, by lane:
  - the loader, the TSV writer and the metrics, including the re-implementation of the sponsor's ASVspoof5 scoring code (`hearsay.metrics.sponsor_min_dcf`)
  - the fold file and the nested validation scheme (`scripts/make_folds.py`, `splits/nsa_folds.csv`), the manifests and the data pulls (LJ Speech, LibriSpeech, ASVspoof 2019, In-the-Wild, MLAAD)
  - the software-only rescope (Sep 25–26): plan, detector contract, doc-consistency tests
  - reading the NSA brief, instructions and scoring code; the sponsor-question notes were taken by the team at the kickoff and transcribed
  - M0 and M1 (main chat, Fri night to Sat morning): the constant rollback TSV, the XLS-R embedding and probe pipeline (`hearsay.embed`, `hearsay.probe`, `scripts/extract_embeddings.py`, `scripts/train_probe.py`), the shortcut inventories (level, leading silence, tiling), the segment-mode fix, the M1 v2 and v3 TSVs, the band match of the deep path (`prepare_segment`), and **M1b** with the ASVspoof 2019 real speakers as inner-fold-only extra rows (`docs/reports/2026-09-26_m1b-asv19-bonafide.md`); the In-the-Wild stress readouts for every deep detector
  - the D-track engineered detectors (Sat Sep 26, CPU-only chat): handcrafted spectral + prosody, compression forensics, container/metadata, ENF and splice detectors, their fold-validated training and fusion score exports (`outputs/detector_scores/*.csv`), the container and high-band inventories, and the test-set band-match finding (`docs/reports/2026-09-26_cpu-detectors.md`); then the six added feature families with their per-column gate (`hearsay.hc_v4`, `docs/reports/2026-09-26_handcrafted-v4.md`), the test-domain-shift diagnosis and the augmented v5 training (`--extra-rows-from`), the non-speech gate and the pinned-block default-answer policy, the speaker-drift detector on ECAPA embeddings (`docs/reports/2026-09-26_gate-and-drift.md`), the numpy tree evaluator that keeps LightGBM out of inference, the orchestration ablation (`scripts/orchestration_ablation.py`), the worked examples (`docs/reports/2026-09-26_worked-examples.md`) and the README rewrite (Sat Sep 26, README chat)
  - M3 (Sat Sep 26, Spectra-AASIST chat): the Spectra-AASIST score stream through the shared band-matched input path with test-length crops (`scripts/score_spectra.py`, `hearsay.spectra`), its short-clip pad-mode pilot, direction and failure gates, the hermetic tests (`tests/test_spectra.py`) and the report (`docs/reports/2026-09-26_m3-spectra.md`)
  - the M5 rung (Sat Sep 26, vast.ai chat): the consult question on extra data and fine-tuning, the data bundle, fold-disciplined manifest, augmentation, trainer/scorer/assembler, the vast.ai launch/reaper scripts, the ablation and the report (`docs/reports/2026-09-26_m5-xlsr-finetune.md`); rented A100 boxes on vast.ai ($5.42 of credit, `docs/reports/cloud-expense-ledger.md`) with Cloudflare R2 as the transfer store
  - fusion (main chat, Sat Sep 26): the first fusion script and the 06:02 TSVs (`scripts/fuse.py`), the fusion consult question, the pre-declared sweep and the frozen rule (`scripts/fuse_sweep.py`, `models/fusion_v1/constants.json`, `docs/reports/2026-09-26_fusion-sweep-predeclared.md`), the 08:13 draft-review payload with its pre-flipped twin, and the decoding table for NSA's returned number
  - the end-to-end runner and the API (main chat, Sat Sep 26): `scripts/run_pipeline.py`, `hearsay.pipeline`, `hearsay.api`, the per-file explanation JSON and routing log, the parity checks against the logged TSVs (`docs/reports/2026-09-26_runner-docker.md`), and the frontend contract (`docs/handoffs/2026-09-26_frontend-contract.md`)
  - K (Sat Sep 26, Docker chat): the offline CPU linux/amd64 inference image (`Dockerfile`, `.dockerignore`, `docker/`: build, entrypoint, smoke, shipped-asset manifest, PCM parity), the Colima + Rosetta build setup on this Mac, the hermetic build-file tests (`tests/test_docker_image.py`), and the smoke and 50-file parity checks of the image against the logged fusion TSV (`docs/specs/2026-09-26_k-docker-image.md`)
  - the channel-robustness rung (Sat Sep 26, 12:25–13:30): the λ̂ estimate of the test set's wild-domain weight (`scripts/channel_lambda.py`), the codec diagnosis of the 7.2 kHz wall (`scripts/channel_codec.py`), the M3 leakage probes through the shipped fusion rule (`scripts/m3_probes.py`), their hermetic tests (`tests/test_channel_robustness.py`) and the report (`docs/reports/2026-09-26_channel-robustness.md`); reviewed by Codex (plan) and a Claude critique
  - the post-draft gate lane (Sat Sep 26, 17:20–): the pre-declared candidate manifest and seven-rule gate with its pre-data clarifications and Nathan's rulings (`docs/reports/2026-09-26_post-draft-manifest.md`), the final-name copy of the shipped TSV (sleep equals ship), `scripts/fuse_sweep_v3.py` with the W4 fresh-evidence bake-off (perturbation cohort plus speaker-cluster bootstrap) and its tests (`tests/test_fuse_sweep_v3.py`), the run spec and the locked table (`docs/specs/2026-09-26_post-draft-gate-sweep.md`, `docs/reports/2026-09-26_post-draft-sweep.md`); reviewed by Codex (plan) and a Claude critique
  - the documents: `docs/plan.md`, `docs/architecture.md`, `docs/code-map.md`, `docs/STATUS.md`, every report, run spec, handoff and consult record under `docs/`, this disclosure, and `README.md`
  - an oversight session (Sat Sep 26) that relayed the team's decisions between lanes, reviewed commits and kept `docs/STATUS.md` current
- `/plan-review` workflow (a Claude Code skill): used to plan and review rungs taking over an hour. Its plan review and post-commit audit call OpenAI Codex (Codex CLI, via `.claude/review-*.sh`) when the CLI is installed. Used for the software-only rescope (Sep 25–26), M3, M5 and the Docker image: Codex plan review and post-commit audit, Claude critique and exploration agents. The plan reviews and audits are under `docs/reports/*-plan-review.md` and `docs/specs/*-audit.md`.
- `/consult` workflow (a Claude Code skill) with VeriLM's "Deep Think" memo service (two experts, Claude and Gemini, with a synthesis): two written consults, each drafted by Claude Code from the repository's own numbers, sent by the team, and recorded verbatim with the adopted items marked: extra data and fine-tuning for M5 (`docs/consults/2026-09-26_m5-extra-data-finetune_*.md`) and fusion strategy, false alarms and the default answer (`docs/consults/2026-09-26_fusion-strategy_*.md`). The fusion memo's recommendations (M1b-dominant rank blend, a pre-declared sweep, M3 as a capped or suppression-only input, the pinned block strictly below every determinate score, the draft-review decoding table) are what the 08:13 rule implements.

**Pretrained models / checkpoints** (name, source, license, how used)
Downloaded before the event to `weights/`; fill in "how used" as each is actually used.
- XLS-R 300M, `facebook/wav2vec2-xls-r-300m` (Hugging Face, Meta), Apache-2.0: core SSL front end. M1: frozen, layer-7 probe. M5: the first 12 transformer layers kept frozen with a learned layer-weighted sum, attentive statistics pooling and a linear head trained on vast.ai (A100), with laundering augmentation; checkpoint on the HF Hub (private `nrs124554433/hearsay-m5-xlsr`). See `docs/reports/2026-09-26_m5-xlsr-finetune.md`.
- WavLM Large, `microsoft/wavlm-large` (Hugging Face, Microsoft): downloaded as a bake-off challenger; **not used** in any run or shipped artifact. No license on the Hugging Face card; WavLM was released through Microsoft's unilm repo (MIT).
- WavLM Base, `microsoft/wavlm-base` (Hugging Face, Microsoft): downloaded as a bake-off / speed spare; **not used**. No license on the Hugging Face card; WavLM was released through Microsoft's unilm repo (MIT).
- Spectra-AASIST, `lab260/Spectra-AASIST` (Hugging Face): M3 score stream (`outputs/detector_scores/spectra_aasist.csv`), run off the shelf with no training through the shared band-matched input path, as a second deep score for fusion beside the XLS-R probe. License unclear: repo header says Apache-2.0, model card text says MIT. Output index 0 = spoof, 1 = bonafide.
- ECAPA-TDNN speaker embeddings, `speechbrain/spkrec-ecapa-voxceleb` (Hugging Face, SpeechBrain), Apache-2.0, fetched Sat Sep 26 to `weights/spkrec-ecapa-voxceleb` (89 MB): the speaker-drift detector (`hearsay.detectors.speaker_drift`, rubric technique 6), loaded from the local directory. Docker must include it.

**Public datasets** (name, source, license, how used)
Downloaded before the event to `data/`; fill in "how used" as each is actually used.
- ASVspoof 2019 LA (University of Edinburgh DataShare, doi handle 10283/3336), Open Data Commons Attribution License: public shakedown of the M1 pipeline; its train+dev bona fide (40 VCTK speakers) plus a 2.5k A01–A06 anchor are training-only extra rows for M1b and M5 (`splits/nsa_folds_plus_asv19.csv`), never scored or held out.
- In-the-Wild (Müller et al. 2022; `mueller91/In-The-Wild` on Hugging Face), license listed as CC-BY-SA-4.0 on Hugging Face and Apache-2.0 on deepfake-total.com: held-out stress test only, never trained on.
- MLAAD (`mueller91/MLAAD` on Hugging Face, gated, non-commercial notice): 6,180 unique English synthetic clips pulled during the event (30 per model across 143 models, plus 300 per model for the ElevenLabs, Gemini and Qwen3 families named on the kickoff slide; the two pulls overlap by 210). Used by M5 as training-only spoof data, capped at 12% of the spoof side per batch and ≤ 120 clips per model, with the OpenVoice / XTTS / ElevenLabs families excluded from the fold models that validate on those generators.
- NSA HEARSAY data (provided by the sponsor during the event): 1,671-file test set, a resampled LJ Speech subset (real), DiffSSD (synthetic, Purdue), and the ASVspoof5 evaluation code for MinDCF.
- Full LJ Speech 1.1 (keithito.com, public domain): 13,100 clips of one speaker; 6,000 in the NSA training sample, the rest training-only extra rows for M5 (the sponsor's resampled subset is under `data/nsa/LJRealResampled`).
- LibriSpeech dev-clean/test-clean (openslr.org/12, CC BY 4.0): 5,323 multi-speaker bona fide clips; 4,000 of them in the NSA training sample, the rest training-only extra rows for M5.

**Planned, not yet downloaded or used** (move each item up to the list above once it is actually used):
- VCTK as a second multi-speaker bona fide source (LibriSpeech dev+test is in use; see above).
- ReplayDF (optional).
- ExifTool, for metadata forensics. _Not used: ffprobe (FFmpeg) covered the embedded fields, and the test set carries none beyond one encoder tag._

**Frameworks and libraries**
- PyTorch, torchaudio, Hugging Face transformers and huggingface_hub, SpeechBrain (ECAPA speaker embeddings), librosa, soundfile, NumPy, SciPy, scikit-learn, LightGBM, pandas, pydub, ffmpeg-python, FFmpeg.

**What we built vs. what AI did**

The short version: the team decided what to build and why, and Claude Code built it. Every design decision below is recorded with its reason in `docs/plan.md`, `docs/architecture.md` or a report; every line of application code, every test and every document was written by Claude Code sessions under the team's direction and reviewed by the team and by Codex. Nothing on the scoring path calls a hosted model.

| Component | The team (Nathan unless noted) | Claude Code | Outside review |
|---|---|---|---|
| Problem framing and rules | The software-only scope; 1.0 = synthetic and never flipped; validate by generator and speaker; never train on the holdout or In-the-Wild; the model ladder and the cut order; the pre-event repository, `CLAUDE.md` and the working plan's first version | The rescope plan and its detector contract; the doc-consistency tests | Codex plan review of the rescope |
| Data and folds | Chose the corpora; obtained the sponsor data; asked the sponsor's questions at the kickoff and took the notes | Manifests, the fold file, the leakage and shortcut inventories, the data pulls | |
| Loader, TSV writer, metrics | The conventions (16 kHz mono everywhere, never overwrite a TSV, log every run) | `hearsay.audio`, `hearsay.submission`, `hearsay.metrics` and their tests | |
| M1 / M1b (frozen XLS-R probe) | Which layer-selection metric to use; approving M1b's extra data as inner-fold-only rows; sending each TSV | Embedding and probe pipeline, segment mode, band match of the deep path, M1b, the In-the-Wild readouts, the reports | |
| M3 (Spectra-AASIST) | The decision to admit it only as a false-alarm suppressor because its training data is undisclosed | The score stream, pad-mode pilot, gates, tests, report | Codex plan review and audit |
| M5 (XLS-R fine-tune) | Approving the GPU rental and the budget; the gate criteria set before the run; accepting the gate's verdict | The consult question, data bundle, trainer, cloud scripts, ablation, report | VeriLM consult on extra data; Codex plan review and audit |
| Handcrafted spectral + prosody (v3, v4 families, v5 augmentation) | The rubric mapping; approving the feature-family brief; choosing v5 for fusion on the false-alarm evidence | Features, the per-column gate, the trainer, the numpy tree evaluator, the domain-shift diagnosis, augmentation, reports | |
| Compression, container, ENF, splice | The rule that these are evidence and routing, not score, once the mechanism was shown | Detectors, laundering-equalized training, inventories, tests | |
| Speech gate and default-answer policy | The direction-robust placement (from the fusion consult's item 5), adopted as a decision | Gate cues set from the test set's extremes, `apply_default_answer`, tests | VeriLM consult |
| Speaker drift | Adopting it as evidence only after the "fakes are the most consistent voices" finding | ECAPA windows, calibration on synthetic splices, tests | |
| Fusion | Rejecting equal-weight fusion; the pre-declared selection rule; freezing the 08:13 rule; ratifying the pre-declared M5 candidate (A3 w0.2 + E) as the shipped rule at ~12:05 Sat once the Docker image was no longer required; the draft-review payload and its decoding table | The fusion scripts, the sweep, the constants, the consult question, the reports | VeriLM fusion consult |
| Orchestrator, runner, API, explanation JSON | The routing policy (which facts route, which detectors are trusted) | `hearsay.pipeline`, `scripts/run_pipeline.py`, `hearsay.api`, the routing log, parity checks | |
| Frontend | Hrushi built the UI against the JSON contract | The contract (`docs/handoffs/2026-09-26_frontend-contract.md`) and the static dump it reads | |
| Docker image | Colima + Rosetta on this Mac; shipping the M1b v3 probe; offline enforcement | Dockerfile, build/entrypoint/smoke/parity scripts, asset manifest, tests, spec | Codex plan review and audit |
| Orchestration ablation and worked examples | Asked for proof that routing changed decisions | `scripts/orchestration_ablation.py`, `docs/reports/2026-09-26_worked-examples.md` | |
| Documentation and README | Set the rule that every number is cited to a file; reviewed | All of it, including this disclosure | |

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
