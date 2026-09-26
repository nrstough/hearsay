# HEARSAY: The Plan (software only)

Rewritten Fri Sep 25, ~23:30 → Sat Sep 26, 00:30, after the NSA brief (`docs/nsa-challenge.md`), the NSA instructions (`data/nsa/HEARSAY_HackGT2026_Instructions.pdf`), the scoring code (`data/nsa/HackGTMinDCF/`) and the kickoff notes (`docs/reports/2026-09-25_sponsor-questions.md`). Run spec: `docs/specs/2026-09-25_software-only-rescope.md`. The earlier device-and-demo plan is historical (cut).

## The one-paragraph version

HEARSAY is a software-only, multi-detector audio authentication system.
- For every file, a rule-based orchestrator chooses which forensic detectors to run.
- Each detector returns a score and a plain-English reason.
- A logistic stacker fuses the scores into one probability that the clip is synthetic.
- The output is `teamName_predictions.tsv` plus a per-file explanation report.
- Everything ships as a Docker image that runs offline on the test set, with a README written in our own words.

A frozen and then fine-tuned XLS-R carries the detection score. The engineered detectors carry the diversity and explainability points. We always have a valid TSV.

## Scoring and deliverables

**Judging**
- **60% MinDCF.**
  - Points = (1 − ours)/(1 − best) × 60.
  - False alarms (real called synthetic) cost 4× a miss. About 70% of test files are real.
  - Our metric (`hearsay.metrics`): `C_FA = 4`, `C_miss = 1`, `π_synth = 0.3`, i.e. normalized minDCF = 9.33·P_FA + P_miss.
- **20% creativity, quality, innovation and depth.** The brief itemizes forensic diversity; see the techniques section below.
- **20% GitHub documentation.** Approach, architecture, how to run, and what worked and what didn't, "in your own words".

**Deliverables**
- `teamName_predictions.tsv`.
  - Header `filename<TAB>cm-score`; filename with extension; probability 0.0–1.0, 1.0 = synthetic.
  - All 1,671 rows, in the template's order (`data/nsa/HearsayScoreKey4TeamX.tsv`).
  - Sent by Discord DM.
- A Docker image that runs inference on the test set.
- The README.
- The optional one-time draft review, which we use on purpose.
- **Final DM by 8 AM Sunday.**

**Score direction is an open blocker.** The sponsor's evaluation code (ASVspoof5 `compute_det_curve`, unmodified) treats a *higher* cm-score as **bona fide**, the opposite of the instructions. Fed as-is, a good detector with 1.0 = synthetic scores minDCF 1.0, the worst possible; the same scores flipped score about 0.32. The code also uses `Pspoof = 0.5, Cmiss = 1, Cfa = 4` with Cfa on *accepted spoofs*.
- Until the sponsor answers: submit 1.0 = synthetic as instructed.
- `write_submission` gets a one-flag flip in the next change.
- The draft review is the tripwire: a strong model scoring near 1.0 means the direction is flipped.

**Metric readings.** Every candidate is logged under four readings, plus EER:
- π_synth 0.3 (primary)
- π = 0.5
- the slide-literal form
- the sponsor code as shipped (π = 0.5; Cfa on accepted spoofs, i.e. P_FA + 4·P_miss after a flip)

A choice that flips between readings is flagged.

## Architecture

1. **Ingest.** `hearsay.audio.probe_audio` (container facts) and `load_audio` (16 kHz mono float32, one FFmpeg path; never denoised or loudness-normalized).
2. **Orchestrator.** Rule-based routing on container, codec, sample rate, duration and early findings. It records why each detector ran or was skipped. This is where the orchestration bonus comes from; no LLM sits on the scoring path.
3. **Detectors.** Each follows the contract in `src/hearsay/detectors/base.py`: a `DetectorResult` with a score in [0, 1] that increases with synthetic likelihood, non-empty evidence, flat numeric features, and a status. `safe_run` never loses a row.
4. **Fusion.**
   - L2 logistic stacking, with one column per detector: `logit(clip(score, 1e-4, 1 − 1e-4))` when `status == "ok"`, else missing, imputed with the fold-local train mean, plus a `<name>.missing` indicator.
   - Columns are sorted by detector name. The 0.5 placeholder on skipped or errored results is never used as a score.
   - If every detector is missing for a clip, it gets the constant prior-decision score and a flag.
   - Per-source minDCF is reported, so a detector that only works on one corpus is visible.
5. **Outputs.** The TSV, plus an explanation report per file: top contributing detectors (coefficient × standardized value) and their evidence strings.

## Forensic techniques

One entry per category in the brief's rubric.

**Test set facts, from the inventory.** All 1,671 files are WAV, 16-bit PCM, 16 kHz mono, 3.0–13.6 s (median 3.4 s), with almost no leading silence and peaks near full scale. The container carries no class signal, so file-level detectors must look inside the audio.

| # | Technique | Method | Owner | Time box |
|---|---|---|---|---|
| 1 | Container / metadata forensics | ffprobe/ExifTool header fields, encoder tags, codec chain, WAV header quirks. Embedded tags only; never filesystem MAC times or filenames, which git, Docker and unzip rewrite. Low expected signal on this test set; documented either way. | Teammate A | 1.5 h |
| 2 | Spectral / frequency-domain analysis | High-frequency roll-off and band-limiting, spectral flatness, vocoder harmonic regularity, LFCC statistics | Teammate B | 2 h |
| 3 | Prosody & phonetics | Pause and breath statistics, pitch-contour variability and monotonicity, speaking-rate regularity | Teammate B | 2 h |
| 4 | Acoustic environment (ENF) | Mains-hum trace at 50/60 Hz harmonics, and its consistency | Teammate C | 1 h (first cut) |
| 5 | Compression forensics | Double-encoding and transcoding traces (quantization and spectral-hole patterns) | Teammate A | 1.5 h |
| 6 | Speaker-embedding consistency | ECAPA speaker embeddings on sliding windows; drift across the clip | Teammate C | 1.5 h |
| 7 | Deep-learning anti-spoofing | Frozen XLS-R probe (M1), Spectra-AASIST (M3), fine-tuned XLS-R (M5) | Nathan | see ladder |
| 8 | Splice / discontinuity | Phase breaks, DC-offset jumps, background seams | Teammate C | 1 h |
| + | Agentic orchestration bonus | Rule-based router, and a routing log in the explanation report | Nathan | 1 h |

Audio stays 16 kHz mono everywhere. A native-rate loader API for forensic detectors is a separate follow-up change, with its own review and conversion tests, and it matters little here because the test set is already 16 kHz.

## Ladder

| Rung | Deliverable | Owner | Time box | Gate |
|---|---|---|---|---|
| M0 | Loader, TSV writer, constant rollback TSV (all "real"), inventory, fold file | Nathan | 1 h from data | Never cut |
| M1 | Frozen XLS-R probe on NSA training data → first learned TSV | Nathan | 2 h | Before the draft review |
| D-track | Engineered detectors 1–6 and 8 against the contract, with out-of-fold scores on the fold file | Teammates A/B/C | per the techniques table | Sat 19:00 |
| M3 | Spectra-AASIST as a second deep detector | Nathan | 45 min | Drop on friction |
| M5 | XLS-R fine-tune on the H100, with laundering augmentation | Nathan | 1–3 runs | Must beat M1 on the outer holdout |
| M4 | Fusion, orchestrator and explanation report | Nathan | 3 h | Fusion frozen Sat 22:00 |
| K | Docker image (CPU inference on the test set) | Teammate C from Sat morning | 3 h plus rebuild | First amd64 image Sat 14:00 |
| DOC | README in our own words, experiment log, ablation table | All | ongoing, final Sun 05:00–07:30 | Never cut |

## Data

**Training data**
- **NSA training data:** LJ Speech (real; a single speaker, Linda Johnson; 242 resampled clips in `data/nsa/LJRealResampled`, plus the full set on the HexLabs environment) and DiffSSD (synthetic; ElevenLabs and diffusion TTS; downloading).
- **Public data:**
  - MLAAD English, 6,390 clips.
  - A multi-speaker bona fide source (LibriSpeech/VCTK), not yet downloaded. Without it, "LJ's voice = real" is a shortcut the test set will punish.
  - ASVspoof 2019 as a baseline only.
  - In-the-Wild as a stress test only; never trained on.

**Shortcuts found in the inventory.**
- LJ clips run 4–10 s (median 7.1 s) with peaks around 0.54; test clips run 3–13.6 s (median 3.4 s) with peaks near 1.0.
- Training clips are cropped to 3–5 s and level-matched to the test peak distribution, for both classes alike.
- Duration, peak and silence are checked per class every time data is added.

**Fold file** (`splits/nsa_folds.csv`), published within 1 h of the training data being usable:
- The group key is the connected components linking clips that share a generator, a speaker or a near-duplicate hash. Folds therefore hold out whole generators and whole speakers at once.
- The builder states which of those links actually exist in the labels.
- An untouched outer holdout (about 20% of groups), plus 5 inner folds.
- At least 30 clips per class in every fold, and class-balanced training.
- Every learned piece is fitted fold-locally: detectors, layer choice, calibration, imputation, scaler, fusion. Stacking features are inner-fold out-of-fold predictions. Fusion is scored once on the outer holdout, and that number is the logged `validation_score`.

**Augmentation**, applied to both classes: codec and telephony round-trips, noise, RIR and replay simulation, band-limiting, random ffmpeg transcodes.

**Leakage.** Hash-dedupe NSA data against every public set (`scripts/hash_audio.py`).

**Weights still needed.** ECAPA (speaker drift, and inside Docker).

## Timeline

**Moving gates** shift with the data:
- T0 (training data usable) + 1 h: inventory, fold file, constant TSV.
- T0 + 3 h: M1 NSA TSV.

**Hard cutoffs never move:**

| When | What |
|---|---|
| Sat 02:00–06:00 | Nathan sleeps |
| Sat 09:00 | H100 run 1, pending Nathan's approval (spend cap about $75) |
| Sat 12:00 | If there's still no NSA training data: cut ENF and splice; the draft review uses the public-data M1 TSV |
| Sat 14:00 | Draft TSV sent for the one-time review; first amd64 Docker image (M1 payload) |
| Sat 19:00 | Teammate detectors pass the contract tests and deliver out-of-fold scores on the fold file |
| Sat 22:00 | Fusion frozen; leave-one-detector-out ablation on the stored out-of-fold columns; no new detectors |
| Sun 00:00 | Docker rebuild; amd64 smoke test on 3 files; CPU vs MPS parity check on 50 files |
| Sun 01:00–04:30 | Nathan sleeps |
| Sun 05:00 | Final TSV, after the silence/tone/music/noise preflight and the score-direction check |
| Sun 05:00–07:30 | README, ablation table, AI disclosure |
| **Before 8 AM Sun** | **Final TSV DM'd by a named person** |

## Cut order

Cut in this order: ENF → splice → speaker drift → M5 reduced to one run → orchestrator reduced to static routing → fusion reduced to a mean.

Never cut: a valid TSV at all times; generator- and speaker-grouped validation; the deep SSL detector; the Docker image; the README.

## Docker

- **Base and install.** `python:3.12-slim-bookworm` plus apt ffmpeg.
  - CPU torch 2.14.0 and torchaudio 2.11.0 come from the PyTorch CPU index; the lockfile's CUDA wheels are skipped with `--no-install-package`.
  - Keep the editable install.
- **Offline.** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- **Weights.**
  - XLS-R and Spectra-AASIST (both are needed), plus the probe's joblib.
  - The fine-tuned checkpoint goes on the HF Hub or a GitHub release, not git LFS.
  - `.dockerignore` excludes the WavLM weights, `.venv`, `data/`, `outputs/` and `.cache`.
- **Runtime.**
  - One model loaded at a time.
  - A resumable per-file cache, and a new output file per run.
  - Tested with `--network none`.

## Risks

**High**
- **Score direction and the scoring code.** Covered in Scoring and deliverables above: flip flag plus the draft-review tripwire.
- **Nathan is the critical path.** Docker goes to a teammate; the teammates' detectors enter only through the contract and the fold file.
- **Stacking leakage.** Nested folds; out-of-fold predictions only.
- **Shortcuts.** LJ single speaker, duration, peak level, silence, and container fields (checked for AUC > 0.85). Report minDCF with and without each suspect detector.
- **Docker on judges' machines.** Image size, amd64, CPU runtime, memory.

**Medium**
- **Routing shift between train and test.** Compute every cheap detector on training data; gates must be rank-neutral.
- **Missing grouping keys.** Fall back to embedding-based speaker clustering.
- **Proxy public data is confounded by corpus.**
- **Physical replay is not covered.** No hardware; only simulated RIR, plus an optional ReplayDF subset.

## Rules that never move

- Always have a valid TSV. Never overwrite a submitted one. Log every TSV in `submissions/log.csv` with its outer-holdout minDCF (EER in notes).
- Validate by generator and speaker, never by clip. Never train on In-the-Wild or on the validation split, including scalers, calibration and stacking folds.
- Keep a clean-only validation score beside any augmented one.
- Never denoise or loudness-normalize test audio. Level-match *training* data to the test distribution instead.
- Submit calibrated probabilities, never saturated 0/1. Never be confidently wrong on a whole subpopulation (silence, a noise type, a speaker).
- Before every TSV: silence/tone/music/noise preflight, score-direction check on known clips, schema, row order against the template, no duplicate filenames.
- Fine-tune on the H100 only. Ask before renting; cap spend at about $75.
- Time-box every rung. When the box runs out, stop and report. Freeze Sunday morning.
- The scoring path runs offline: no hosted API and no LLM. Secrets live in `.env` only.
- Credit every model and dataset. The README is written in our own words, with AI use disclosed.
