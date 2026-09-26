# HEARSAY: current state (Sat Sep 26, ~04:40)

A snapshot for teammates joining now. The plan, with its rules and timeline, is [docs/plan.md](plan.md). The architecture, with diagrams, is [docs/architecture.md](architecture.md). The NSA brief is [docs/nsa-challenge.md](nsa-challenge.md). Setup and repo rules are in [CLAUDE.md](../CLAUDE.md).

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
| Deep detector (XLS-R probe, M1) | Done: first learned TSV logged (`submissions/20260926-0308_M1_*.tsv`, holdout minDCF 0.146, EER 2.5%). M1b (+ASVspoof 2019 real speakers, inner folds only) reaches holdout 0.079 but raises In-the-Wild false alarms (report: `docs/reports/2026-09-26_m1b-asv19-bonafide.md`). The deep path is band-matched since commit `133e536`; the v3 re-extraction is running and its numbers replace these. |
| Handcrafted spectral and prosody detector (75 interpretable features) | Done: `hearsay.detectors.handcrafted`. Holdout minDCF 0.25, EER 4.3% (v3: test-length crops + band match, LightGBM). Blind to grad_tts, pro_diff and ElevenLabs. Export `outputs/detector_scores/handcrafted.csv`. **v4 feature families** (LFCC, phase, CQCC, modulation, breath, jitter/shimmer; all implemented and gated, see `docs/reports/2026-09-26_handcrafted-v4.md`) are opt-in via `--families` in `hearsay.hc_v4`; the teammate brief is `docs/handoffs/2026-09-26_handcrafted-v4-brief.md`. **v4 result:** inner OOF minDCF 0.434 (v3 0.614), pro_diff 1.00 → 0.32, LibriSpeech real 0.77 → 0.54, holdout 0.170; grad_tts still 1.00; In-the-Wild P_FA 0.65% at the inner threshold but minDCF 1.0 (does not transfer). The first fusion run found a test-domain shift (v4 alone calls 56% of the test set synthetic under a 0.3-prior Platt map; v3 49%; the envelope of the test recordings is darker, codec-like, across nearly every column). **v5b** (same columns + augmented fit-only training rows: codec round-trips, random tilt + low-pass) is the recommended fusion column: clean holdout 0.137, calibrated test share 51%, holdout-real false alarms 6.3% vs 9.1%. Exports `handcrafted_v4.csv` and `handcrafted_v5.csv`; bundle `models/hc_selected` = v5b; fusion decides. |
| Compression, container, ENF, splice detectors | Done against the contract (`hearsay.detectors.{compression,container,enf,splice}`; `import hearsay.detectors.engineered` registers all five). Container is rule-based (constant on this test set, by design); ENF and splice are mild rule-based scores plus evidence. Exports in `outputs/detector_scores/`. Report: `docs/reports/2026-09-26_cpu-detectors.md`. |
| Spectra-AASIST (M3, second deep detector) | **Done (Sat 06:00).** Off-the-shelf Spectra-AASIST scored through the shared band-matched input path with the same test-length crops as M1 and the handcrafted detector; zero-pad for short clips (chosen on inner rows). Outer holdout minDCF **0.012**, EER 0.16% (playht 0.010, wavegrad2 0.012, LibriSpeech real 0.011, LJ real 0.008); In-the-Wild minDCF 0.065, EER 3.2%, real P_FA 0.0% at its own threshold (possibly optimistic: training data undisclosed); 29.3% of test files above a zero margin (prior ~30%); disagrees with M1 v3 on 154 test files (Spearman 0.57). Export `outputs/detector_scores/spectra_aasist.csv` (`logit` = raw margin, read it directly). Report: `docs/reports/2026-09-26_m3-spectra.md`. |
| Fine-tuned XLS-R (M5) | In flight on rented A100s: 12-layer truncated backbone, attentive pooling, class-blind augmentation; gate is Sat 12:00 (beat M1 on inner folds and holdout, not worse on In-the-Wild false alarms). Spec: `docs/specs/2026-09-26_m5-xlsr-finetune.md`. |
| Fusion, orchestrator, explanation report (M4) | Planned; starts when detector scores land (Sat 19:00), frozen Sat 22:00. Design and open decisions in `docs/architecture.md`. The fusion consult is drafted and not yet sent: `docs/consults/2026-09-26_fusion-strategy_CONSULTATION.md`. |
| Docker | Not started and unowned. First amd64 image (M1 payload) is due Sat 14:00. |
| Tests | 351 passing over the tracked test files at ed31cb9 (`uv run pytest -q`; M3's `tests/test_spectra.py` has 40; other chats' untracked tests add more), ruff clean. |
| Orchestration credit (ablation, worked examples) | **Done (Sat, CPU chat).** `scripts/orchestration_ablation.py` re-fuses the exported scores with each routing rule on and off (`outputs/fusion/orchestration_ablation.{json,md}`): M3 suppression halves the holdout cost (0.030 → 0.014) and takes In-the-Wild from 0.322 to 0.258 while touching only real files (123 holdout, 34 In-the-Wild); the gate touches 31 holdout rows, 9 In-the-Wild rows and 0 test files; fusing everything equally is better on the holdout (0.0085) and worse on In-the-Wild under the brief's cost (0.276). Eight real test files with their routing logs and every evidence sentence, reproduced live with the frozen rule to four decimals of the shipped TSV: `docs/reports/2026-09-26_worked-examples.md`. `CLAUDE.md` disclosure extended to every lane and "what we built vs. what AI did" filled per component. |

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
- **The container *is* the label on the training data** (measured): LibriSpeech = FLAC, LJ = 22 kHz WAV, DiffSSD = WAV or MP3, and PlayHT's MP3s carry the same FFmpeg encoder tag (`Lavf58.29.100`) as all 1,671 test files. So the container detector is never learned; it returns routing facts and a constant score here.
- **The test set is low-passed at about 7.2 kHz** and no training corpus is (not even the sponsor's resampled LJ clips). Everything above 7 kHz was a train-vs-test fingerprint; the handcrafted detector called 90% of the test set synthetic until `hearsay.handcrafted.band_limit` (a 71-tap Kaiser low-pass matching the roll-off) was applied to every clip, train and test. Now 42% of test files score above 0.5. The deep path got the same band match on Sat ~04:20 (commit `133e536`: `prepare_segment` calls `band_limit` first), and M1's embeddings are being re-extracted; details and numbers in `docs/reports/2026-09-26_cpu-detectors.md`.
- **Our holdout is optimistic about real-world audio.** On the eval-only In-the-Wild set, both M1 and M1b score minDCF 0.37–0.40 against 0.08–0.15 on the NSA holdout. Adding 40 VCTK read speakers (M1b) improved every holdout slice but raised In-the-Wild false alarms at the inner threshold from 1.2% to 5.7%. Channel robustness, not more read speech or more spoof data, is the lever; every deep detector now reports In-the-Wild false alarms as a gate.
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
