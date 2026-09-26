# HEARSAY

## What HEARSAY is

HEARSAY is an audio deepfake detector built at HackGT 13 (Sep 25–27, 2026) for the NSA sponsor challenge: given any audio clip, it outputs a 0–100% likelihood that the voice is AI-generated, and we submit a CSV of predictions on a hidden test set. The same model drives a live demo: a pan-tilt "listening head" with a 4-mic array that turns toward whichever of two phones is talking and turns red on the cloned voice. The detector is climbed as a model ladder (M0–M6), with every rung measured on a generator-held-out validation split and a valid CSV always on hand after M1.

## Team and ownership

Four-person team.

| Area | Owner |
|------|-------|
| Detector, validation split, submission CSV, streaming scorer | Nathan (@nrstough) |
| Mic capture and direction finding | _teammate — fill in_ |
| Pan-tilt firmware and aiming | _teammate — fill in_ |
| Dashboard | _teammate — fill in_ |

## Data contract

Data contract everyone builds against:
- Audio front end emits {t_start, t_end, angle_deg, audio_window} while someone is speaking
- Detector takes a 16 kHz mono window and returns {p_synthetic, per_detector_scores}
- Tracker emits {source_id, angle_deg, score_smoothed, verdict} to the head and the dashboard
- Head driver takes {pan_deg, tilt_deg, laser, color} and returns {ok, error}

## Model ladder and model rules

Model ladder (each rung measured on a held-out split, never without a valid CSV after M1): M0 loader and generator-held-out split; M1 frozen SSL embeddings plus logistic regression, first CSV; M2 hand-crafted features; M3 off-the-shelf detector scores as features; M4 stacked gradient-boosted ensemble with calibrated output; M5 fine-tuned SSL front end (GPU only); M6 manipulation-type head if labeled. Model rules: validate by generator, not by clip; log every CSV with its validation score; cap any rung at a few hours; keep a clean-only validation score alongside any replay-augmented one.

## Conventions

- **16 kHz mono everywhere.** Every loader resamples and downmixes at the boundary; nothing downstream sees another format.
- **Every experiment appends a row to `submissions/log.csv`** with timestamp, rung, validation score, and CSV path. Columns: `timestamp,rung,validation_score,validation_score_clean_only,csv_path,notes` (the clean-only column exists because of the model rule above; leave it equal to `validation_score` when no replay augmentation is used).
- **No hosted APIs on the live path.** The demo runs locally end to end.
- **Secrets in `.env` only, never committed.** `.env` is gitignored.
- Environment: `uv sync` once, then `uv run <cmd>` from the repo root (Python 3.12, pinned in `.python-version`). Data, weights, model outputs, submission CSVs, and audio files are gitignored; only `submissions/log.csv` is tracked.
- Tests: `uv run pytest` (tests in `tests/`; markers `needs_data`, `needs_weights`, `slow` for anything that needs gitignored assets). Lint: `uv run ruff check .`. The suite is empty until the event starts.
- Doc layout: run specs in `docs/specs/`, plan files and reports in `docs/reports/`, handoffs in `docs/handoffs/`, consult records in `docs/consults/`. No feature-spec layer.
- Codex review: `.claude/review-plan.sh <plan.md>` and `.claude/review-audit.sh <run-spec.md>`, with HEARSAY rubrics in `.claude/codex-*-prompt.md`. Requires the Codex CLI; without it, the plan-review Claude critique is the gate.
- Known setup gotcha: the installed torchaudio (2.11) cannot `torchaudio.load()` files without the separate `torchcodec` package, which is not installed. `soundfile` and `librosa` load audio fine.

## Working rules for Claude Code

- Use `/plan-review` for any rung expected to take more than an hour.
- Never overwrite a submitted CSV. Every submission gets a new file and a new log row.
- Never train on the validation split (including scalers, calibration, and stacking folds).
- Ask before renting a cloud GPU.
- When a time box runs out, stop and report rather than push through.
- Freeze is Sunday morning (Sep 27): after it, only fixes that unbreak the CSV or the demo.

## Sources of truth

- `docs/scoping.md`: scoping document from the external review.
- `docs/master-doc.md`: master plan, exported from Claude Docs.
- `docs/plan.md`: the condensed working plan the team builds from (written Fri Sep 25 from the scoping memo plus the master doc).

Where these and this file disagree, the docs win on plan and scope, and `submissions/log.csv` wins on numbers.

## Pre-event work

Done before the coding window opened at 8:00 pm Friday, Sep 25, 2026. Copy into Devpost.

- Created the repository scaffold: `pyproject.toml`, `.python-version`, `.gitignore`, and empty directories (`data/`, `weights/`, `docs/`, `submissions/`, `scripts/`, `src/hearsay/`).
- Installed the Python dependencies (torch, torchaudio, transformers, huggingface_hub, librosa, soundfile, numpy, scipy, scikit-learn, lightgbm, pandas, pydub, ffmpeg-python) into a uv-managed virtualenv, and confirmed ffmpeg is installed. Added pytest and ruff as dev dependencies.
- Downloaded pretrained weights and public datasets into the gitignored `weights/` and `data/` directories (event rules allow pre-event downloads): the models and datasets listed under AI use disclosure. Nothing was trained or run on them before the event.
- Copied general-purpose Claude Code skills (plan-review, p3, handoff, consult, diagnose, validate-training-data, test-runner, doc-sync, zoom-out) into `.claude/skills/` and added HEARSAY-specific notes to them.
- Copied the Codex plan-review and audit scripts into `.claude/` and added HEARSAY-specific checks to their rubrics.
- Added project boilerplate: `README.md`, `.env.example`, `.claude/settings.json` (Claude Code permissions), pytest and ruff config, and empty `tests/` and `docs/` subdirectories.
- Wrote this `CLAUDE.md`, added the planning docs (`docs/scoping.md`, the external scoping memo; `docs/plan.md`, the working plan), a header-only `submissions/log.csv`, and `scripts/first-commit.sh` (repo creation and first commit, run at 8:00 pm).
- No application code: no loader, pipeline, model, scoring, or firmware code was written before the event.

## AI use disclosure

Keep this current through the weekend; copy into Devpost. The rules require crediting frameworks and stating what AI was used for vs. what we built.

**AI tools**
- Claude Code (Anthropic): used for environment setup, writing project docs and scripts, and as a coding assistant during the event. _Add specifics as they happen._
- `/plan-review` workflow (a Claude Code skill): used to plan and review rungs taking over an hour. Its plan review and post-commit audit call OpenAI Codex (Codex CLI, via `.claude/review-*.sh`) when the CLI is installed. _Record whether it was actually used._

**Pretrained models / checkpoints** (name, source, license, how used)
Downloaded before the event to `weights/`; fill in "how used" as each is actually used.
- XLS-R 300M, `facebook/wav2vec2-xls-r-300m` (Hugging Face, Meta), Apache-2.0: core SSL front end (planned M1/M5).
- WavLM Large, `microsoft/wavlm-large` (Hugging Face, Microsoft): bake-off challenger. No license on the Hugging Face card; WavLM was released through Microsoft's unilm repo (MIT). _Verify before submission._
- WavLM Base, `microsoft/wavlm-base` (Hugging Face, Microsoft): live-path speed spare. No license on the Hugging Face card; WavLM was released through Microsoft's unilm repo (MIT). _Verify before submission._
- Spectra-AASIST, `lab260/Spectra-AASIST` (Hugging Face): off-the-shelf detector score stream (planned M3). License unclear: repo header says Apache-2.0, model card text says MIT. Output index 0 = spoof, 1 = bonafide.

**Public datasets** (name, source, license, how used)
Downloaded before the event to `data/`; fill in "how used" as each is actually used.
- ASVspoof 2019 LA (University of Edinburgh DataShare, doi handle 10283/3336), Open Data Commons Attribution License: training/baseline data.
- In-the-Wild (Müller et al. 2022; `mueller91/In-The-Wild` on Hugging Face), license listed as CC-BY-SA-4.0 on Hugging Face and Apache-2.0 on deepfake-total.com: held-out stress test only, never trained on.

**Frameworks and libraries**
- PyTorch, torchaudio, Hugging Face transformers and huggingface_hub, librosa, soundfile, NumPy, SciPy, scikit-learn, LightGBM, pandas, pydub, ffmpeg-python, FFmpeg.

**What we built vs. what AI did**
- _Fill in per component (detector, direction finding, firmware, dashboard) before submission._

## When a teammate opens this repo

The skills in `.claude/skills/` and the permissions in `.claude/settings.json` load automatically when you run Claude Code in this repo; nothing to install. Setup: `uv sync`, then `cp .env.example .env`. For Codex review in `/plan-review`, install the Codex CLI (`npm install -g @openai/codex`) and run `codex login`; it's optional. Personal overrides go in `.claude/settings.local.json` (gitignored). The skills:

| Skill | Use it for |
|-------|-----------|
| `/plan-review` | Any rung or change expected to take over an hour: problem → failure modes and tests → plan → execute. Includes this project's standing failure modes. |
| `/p3` | Executing a plan that has already been approved. |
| `/diagnose` | A bug you can reproduce, e.g. wrong scores, loader output, or window timing. |
| `/validate-training-data` | Before training on a new dataset or split: generator leakage, imbalance, format leaks. |
| `/test-runner` | Running the tests and diagnosing failures (no suite exists yet). |
| `/doc-sync` | Checking this file and `docs/` against the code, especially the disclosure and log. |
| `/handoff` | Ending a session: writes a prompt so the next chat (or teammate) picks up cleanly. |
| `/consult` | Writing a self-contained question for an outside LLM or expert, and saving the answer. |
| `/zoom-out` | Getting a map of an unfamiliar part of the code (e.g. someone else's stage). |
