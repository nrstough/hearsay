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
- A Docker image that runs inference offline. **Not required** (NSA to Nathan in person, Sat Sep 26 ~12:00: the submission is the TSV alone); ours is built and verified anyway (`hearsay:20260926-0916`) and stays in the README as reproducibility evidence.
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
| Fine-tuned XLS-R (M5) | Done; **gate not passed**, and the shipped arm is head-only: the runs that actually fine-tuned the 12 kept layers (`train_top=12`, six runs) were worse or equal on the hardest folds (ElevenLabs fold: 0.38–0.41 vs 0.31 frozen; grad_tts fold: 0.52 vs 0.52), so the frozen-backbone arm (learned layer weights + attentive pooling + linear head, 3,000 steps, `arm=nsa_extra`) was shipped. Holdout 0.363 (playht 0.05, wavegrad2 0.58; LibriSpeech real 0.42; clips ≤ 4 s 0.60) vs M1 v3 0.159; pooled inner OOF 0.302 vs 0.257; In-the-Wild false alarms 0.45% vs 0.7% (the one axis it wins). Ships only as a stacker column (`outputs/detector_scores/m5_xlsr_ft.csv`); since 8316e50 (11:01) the runner can also score it live under `models/fusion_v2/constants.json` (`--m5`, loaded only when the constants file weights it), still not the shipped rule. Spend $5.09 over seven rentals plus a ~$0.25 checkpoint-restore probe; all boxes destroyed. Spec `docs/specs/2026-09-26_m5-xlsr-finetune.md`; hand-off report to follow. |
| Fusion (M4) | **Rule frozen by pre-declared sweep at 08:13** (`docs/reports/2026-09-26_fusion-sweep-predeclared.md`, constants `models/fusion_v1/constants.json`): rank blend 0.8·M1b v3 + 0.2·handcrafted v5, M3 (Spectra) used only to suppress false alarms, Platt at the 0.3 prior, determinate scores in [0.001, 1], gated block below. Readout: inner 0.140, holdout 0.014, In-the-Wild 0.260 (0.267 under the sponsor code's cost), all better than M1b alone (0.301 / 0.072 / 0.343 / 0.296). Draft-review payload: `submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv`, with a pre-flipped twin ready if NSA's returned number says their code scores inverted. Consult record: `docs/consults/2026-09-26_fusion-strategy_RESPONSE.md`. The runner and the Docker image are being re-pointed to this rule. **Candidate, not shipped (Nathan, 09:25: rule stays frozen; revisit only after NSA's draft-review number, if time allows):** the pre-declared M5 sweep (addendum in the same report) found that A3 w0.2 + E (rank 0.6 M1b v3 + 0.2 handcrafted v5 + 0.2 M5, same M3 suppression) qualifies on crop-fair In-the-Wild scores: ITW 0.228 / 0.239 (both costs) vs 0.260 / 0.267, holdout 0.0065, inner 0.135; Spearman 0.974 against the current TSV, 3 files cross 0.5. Artifacts: `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_*.tsv`, sweep constants under `outputs/fusion/fusion_v2_candidate/`, and the runner's file `models/fusion_v2/constants.json` (written by `scripts/fuse_sweep_m5.py --write`). The pipeline half of shipping it is done (8316e50, "build now, switch later", approved ~10:20; the frozen path reproduces the draft file to 1.1e-16 on the new code); the image is no longer a deliverable (NSA, Sat ~12:00), so a switch is a file copy plus a README line. **Ratified by Nathan at ~12:05: A3 w0.2 + E is the shipped rule.** The draft went to NSA as `CrossExam_predictions.tsv` = `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv` (sha256 `fb783076…`, log row 12:03); the runner's default is `models/fusion_v2/constants.json` since cc6670c (12:35; `scripts/run_pipeline.py --team CrossExam` with no flags reproduces the submitted file: 50 files Spearman 1.0, max 2.6e-4; suite 510); `e_on_a` / `fusion_v1` is the documented `--fusion` fallback and what the evidence Docker image reproduces. Handoff `docs/handoffs/2026-09-26_m4-fusion-handoff.md`; run spec `docs/specs/2026-09-26_m4-fusion-v2-m5-scorer.md`. |
| End-to-end runner + API | Built (frozen path at 9614c18; 8316e50 added the opt-in fusion_v2 path with `--m5` and `--fusion models/fusion_v2/constants.json`, default unchanged; suite 504): `scripts/run_pipeline.py` takes an audio directory and writes the TSV plus one explanation JSON per file (all detectors, M1b, Spectra, the frozen fusion rule `e_on_a` from `models/fusion_v1/constants.json`, the pinned-block policy applied exactly once), resumable, 0.78 s/file on the Mac; `--flip` writes the pre-flipped twin. Reproduces the 08:13 draft TSV from exported logits to 4e-16 on all 1,671 rows and live from audio on 50 files to 2.7e-5. `hearsay.api` serves POST /analyze and GET /results for the frontend. Report: `docs/reports/2026-09-26_runner-docker.md`. |
| Docker | **Shipped-rule image built and verified (09:41)**: `hearsay:20260926-0916` = `hearsay:latest` (linux/amd64, 7.5 GB in the image store, ~4.5 GB unpacked; torch 2.14.0+cpu), built from source `37c26b8`, default rule `e_on_a` from `models/fusion_v1/constants.json` (v0 and the unused v2 candidate staged beside it); entrypoint = the runner with `--require-offline` and a six-thread cap; ships the M1b v3 probe (0521), the selected handcrafted bundle, the compression bundle, fusion constants and all three weight sets with a 72-file sha manifest verified at start. In-image checks with `--network none`: 50-file parity vs `submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv` Spearman 1.0, max diff 2.02e-4, mean 9.1e-6, 0 gated; decoded audio identical to the Mac on 50/50; smoke exit 0 (WAV/MP3/FLAC, reversed template, two runs byte-identical, offline refusal on real input, `--flip` twin with determinate rows reversed and gating identical); preflight silence and music land in the pinned block as `undetermined`. 4.0 s/file on an idle VM, 7.0 when the Mac is busy. The full in-image run stopped at 664/1,671 on the old rule and stands as a partial timing record only (~2 h extrapolated). K commits through 00a717c; spec `docs/specs/2026-09-26_k-docker-image.md`. Rebuild needed only if the rule changes or the frozen path moves; 8316e50 is backward compatible and the Docker chat agreed no rebuild is needed until a switch. |
| README | Draft committed (ec4c65b, README chat): how it decides with a real per-file trace, results at a glance, architecture, validation, what worked and what did not with mechanisms, numbers tables with a source file per row, reproduce, AI use and credits. Pending placeholders: orchestration ablation and worked examples, the draft-review number, the M5 re-sweep outcome, the Docker rebuild parity. Polish Sun 05:00–07:30. |
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
- **The test set reads as codec-processed, not clean (Sat ~12:45, channel-robustness lane).** λ̂, the label-free estimate of the test set's wild-domain weight (a speaker-grouped domain classifier on channel statistics ≤ 7 kHz, holdout real vs In-the-Wild real, OOF AUC 0.994, adjusted classify-and-count), is **0.947 (95% CI 0.93–0.98)**, far above the 0.32 crossover the fusion consult named; on the files the shipped rule calls real it is 0.99. Controls never trained on: clean VCTK reads 1% wild, inner real 3%, the same inner real clips after one MP3 64 kbps round-trip 91%. So "wild" here means "has been through a lossy codec", which fits the 7.2 kHz wall. Consequences: the clean holdout overstates test performance; selecting fusion on In-the-Wild numbers was right; expect NSA's draft number nearer our In-the-Wild figures (0.23) than the holdout (0.0065); the codec match (B) and a symmetric codec refit of M1b (C) matter more than assumed. Artifacts `outputs/channel/lambda.json`, `scripts/channel_lambda.py`; run spec `docs/specs/2026-09-26_channel-robustness.md`.
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
