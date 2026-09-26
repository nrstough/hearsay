# HEARSAY: current state (Sat Sep 26, ~01:15)

A snapshot for teammates joining now. The plan, with its rules and timeline, is [docs/plan.md](plan.md). The NSA brief is [docs/nsa-challenge.md](nsa-challenge.md). Setup and repo rules are in [CLAUDE.md](../CLAUDE.md).

## What we're building

HEARSAY is software only; the hardware idea was dropped Friday night. It's an **audio authentication system** that takes any audio file and outputs the probability that it's synthetic (1.0 = synthetic), plus a plain-English reason.

The design:
- A rule-based orchestrator decides which forensic detectors to run on each file.
- Each detector returns a score and evidence.
- A logistic stacker fuses the scores.

**How we're graded** (NSA instructions, `data/nsa/HEARSAY_HackGT2026_Instructions.pdf`):

| Weight | What |
|---|---|
| 60% | **MinDCF** on NSA's 1,671-file test set. False alarms (real called synthetic) cost 4× a miss, and about 70% of files are real. Points = (1 − ours)/(1 − best) × 60. |
| 20% | **Diversity and depth**: points for each distinct technique actually used. Metadata/container, spectral, prosody, ENF, compression, speaker-embedding consistency, deep anti-spoofing, splice, plus a bonus for choosing which analyses to run per file. |
| 20% | **Documentation** in this GitHub repo, **in our own words**: approach, architecture, what worked, what didn't. |

**Deliverables** (final DM **before 8 AM Sunday**):
- `teamName_predictions.tsv`: header `filename<TAB>cm-score`, all 1,671 rows in the template's order.
- A Docker image that runs inference offline.
- This repo's README.
- One optional early "draft" submission for feedback. We'll use it Saturday afternoon.

## Where things stand

| Area | State |
|---|---|
| Loader and submission writer | Done and tested. One FFmpeg path to 16 kHz mono; the writer enforces the TSV format, never overwrites, and logs to `submissions/log.csv`. |
| Rollback submission | Done: `submissions/20260926-0036_M0_constant.tsv`, all 0.0 ("always real"). Valid, but minDCF 1.0 by definition. |
| Detector contract | Done: `src/hearsay/detectors/base.py`. **Build against this** (see below). |
| Metric | Done: `hearsay.metrics`, plus an exact re-implementation of NSA's own scoring code (`sponsor_min_dcf`). |
| Validation split | Done: `splits/nsa_folds.csv`. Whole generators and speakers are held out; the outer holdout is the `playht` and `wavegrad2` generators plus 26 real-speech groups. |
| Deep detector (XLS-R probe, M1) | **Running now.** Length-matched embeddings, then training. First learned TSV expected tonight. |
| Handcrafted spectral and prosody features (75, interpretable) | Features are extracted for all 21,671 clips; training next. |
| Spectra-AASIST (second deep detector), fine-tuning, fusion, orchestrator, Docker | Not started. See the ladder in `docs/plan.md`. |
| Tests | 152 passing (`uv run pytest -q`), ruff clean. |

## Data

| Data | What | Where |
|---|---|---|
| NSA test set | 1,671 WAVs, 16 kHz mono PCM, 3.0–13.6 s (median 3.4 s), ~70% real | NSA Challenge Google Drive folder → `HackGTHearsayTesting.zip`, unpacked to `data/nsa/HackGTHearsayTesting/` |
| Answer template | Row order for the TSV | Drive → `HearsayScoreKey4TeamX.tsv`, copied to `data/nsa/` |
| DiffSSD (synthetic training) | 70,000 clips from 10 TTS generators, including ElevenLabs and PlayHT; 21.5 GB | Drive → `DiffSSD.zip`, or the HexLabs environment |
| LJ Speech (real training) | 13,100 clips, one speaker | keithito.com (see the NSA instructions), or the HexLabs environment |
| LibriSpeech dev+test (real training) | 5,323 clips, about 80 speakers | openslr.org/12 |
| NSA scoring code | ASVspoof5 evaluation package | Drive → `HackGTMinDCF.zip`, unpacked to `data/nsa/HackGTMinDCF/` |

Data is gitignored. Put it under `data/` with the same paths as above (`data/nsa/...`, `data/ljspeech`, `data/librispeech`), or ask Nathan for a copy from his external drive.

## Findings so far (the "what worked / what didn't" log starts here)

- **The NSA scoring code runs "backwards" from their instructions.** It's ASVspoof5 code that treats a higher score as *real*; the instructions say 1.0 = synthetic.
  - Our decision: follow the instructions (1.0 = synthetic) and never flip.
  - We log how our scores would come out under their code both ways. The draft submission will show which one they actually use.
- **Shortcuts we found and removed:**
  - **Loudness.** Test clips peak near full scale, while training real speech peaks around 0.5, so loudness alone predicted the label (AUC 0.66). Fixed by normalizing each input.
  - **Leading silence.** LibriSpeech real clips start with about 0.37 s of silence; test clips have about 0.06 s. Fixed with the same silence trim on train and test.
  - **Clip length and tiling.** Training clips are 5–9 s and test clips about 3.4 s. The old 4 s window repeat-padded short test clips, which put a splice seam only in test data. Fixed by embedding each clip at its real length, and training on crops drawn from the test length distribution.
- **Container format is a trap.** Some training fakes are MP3 or 22 kHz, while every test file is 16 kHz WAV. Any metadata or compression detector must be trained on clips converted to 16 kHz WAV first, or it will learn "MP3 = fake".
- **Leakage check is clean.** No test file is an exact copy of anything in LJ Speech, the NSA LJ subset, or In-the-Wild.
- **Public-data sanity check** (not an NSA number): frozen XLS-R plus logistic regression gets minDCF 0.023 on ASVspoof 2019's unseen attacks. The same model scored pure silence as 99% synthetic, which is why the silence handling above matters.

## How to help right now

**1. Setup**

```bash
git clone https://github.com/nrstough/hearsay && cd hearsay
uv sync
cp .env.example .env
uv run pytest -q          # expect all passing; tests needing data/weights skip cleanly
```

Requires `uv` and `ffmpeg` (`brew install uv ffmpeg`).

**2. Build a detector** against `src/hearsay/detectors/base.py`.
- Put it in `src/hearsay/detectors/<your_detector>.py`.
- `run(ctx)` returns a `DetectorResult` with:
  - `score` in [0, 1], **higher = more synthetic**
  - `evidence`: one plain-English sentence explaining the score
  - `features`: a few named numbers
- `ctx.audio` is 16 kHz mono float32 and read-only.
- **Never use file dates or filenames.** Git, zip and Docker rewrite them.

Proposed split (see `CLAUDE.md`):

| Owner | Detectors |
|---|---|
| A | Container/metadata and compression forensics |
| B | Spectral and prosody (the handcrafted features in `src/hearsay/handcrafted.py` are a starting point) |
| C | Speaker drift within a clip, and splice. C also owns the Docker image from Saturday morning. |

**3. Validate on the shared folds.**
- Use `outputs/manifests/nsa_train_sample.csv` (paths and labels) and `splits/nsa_folds.csv`.
- Fit only on inner folds `0`–`4` and report the outer holdout. Never train on holdout rows.
- Score with `hearsay.metrics.report(y, scores)`.
- Export scores as `outputs/detector_scores/<name>.csv` with columns `path, fold, split (inner_oof | holdout | test), score, logit`, so fusion can use them.

**4. Deadlines** (from `docs/plan.md`):
- Detector scores due **Sat 19:00**.
- Fusion freezes **Sat 22:00**.
- Final TSV **Sun 05:00**.
- DM sent **before 08:00**.

**5. Working in the repo**
- Small commits.
- `git add` only your own files.
- Keep `uv run pytest -q` green.
- Write down what didn't work too; it's graded.

## Open questions (ask in the NSA Discord HEARSAY channel)

- Does the HexLabs environment have GPUs, and where are the data paths? That would decide whether we rent a cloud GPU for fine-tuning.
- Which prior and cost model will they actually score with? Their code uses Pspoof 0.5 with the 4× cost on accepted spoofs, which isn't the brief's "false alarm ×4 at 70% real".
- Team name for the TSV filename.
