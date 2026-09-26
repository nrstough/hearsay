# Run spec: M5, XLS-R 300M fine-tune on vast.ai (Sat Sep 26, 2026, ~02:50)

Status: P1 and P2 frozen (deep mode). Plan: `docs/reports/2026-09-26_m5-xlsr-finetune-plan.md`.
- **Worktree:** branch `main`, `~/Projects/hearsay`, shared with the main chat and the CPU-detector chat. Stage only M5 files.
- **Handoff:** `docs/handoffs/2026-09-26_m5-finetune-handoff.md`.
- **Consult:** `docs/consults/2026-09-26_m5-extra-data-finetune_{CONSULTATION,RESPONSE}.md` (non-blocking).
- **Rung:** M5.
- **Time box:**
  - about 3 h of build (to roughly 05:30)
  - about 3 h on the box: pilot, ablation, 5 fold models plus 1 full model
- **Hard stop:** Sat 12:00 EDT. If M5 hasn't beaten M1's outer-holdout minDCF (π = 0.3) by then, M1 stays the primary deep detector (the `docs/plan.md` gate).

## Problem

M1 is a frozen XLS-R 300M plus a logistic probe on one layer. The frozen representation was never adapted to modern diffusion or commercial TTS, to NSA's length distribution (3.0–13.6 s, median 3.41 s), or to laundering.
- **Why M5:** fine-tuning the upper transformer layers usually gives the largest single gain in SSL anti-spoofing.
- **Where:** it runs on a rented H100, because the local MPS GPU belongs to the main chat.
- **What M5 must do:**
  - keep the level, silence, tiling and container shortcuts closed
  - never see the outer holdout
  - produce out-of-fold scores the fusion stacker can use
  - ship a single checkpoint the Docker image runs on CPU

## Solution

### D1. Data: core plus extra (extra is training-only)
- **Core:** the 20,000-row NSA sample (`outputs/manifests/nsa_train_sample.csv`), with folds from `splits/nsa_folds.csv`. The fold file is read-only and its sha is pinned by a test.
- **Extra:**
  - **Rest of the NSA manifest** (`nsa_train_full.csv`). Rows inherit their fold from the fold file's group → fold map. Real rows from groups that aren't in the fold file get fold `extra` (train-only for every model). Any `playht` or `wavegrad2` row is dropped.
  - **ASVspoof 2019 LA, train + dev:** all bona fide, and spoof subsampled with a fixed seed to about 1:1 with bona fide. _Amended 03:20:_ trimmed ASV19 bona fide (median 1.9 s) is shorter than its spoof (2.1 s, p90 4.6 s), so both classes are restricted to trimmed duration ≥ 2.0 s and spoof is histogram-matched to bona fide by trimmed duration (0.25 s bins).
  - **MLAAD English:** spoof only. Its matching bona fide (M-AILABS) is unavailable: both hosts gave no response at 02:36 EDT.
- **Every extra row is train-only.** No extra row ever gets a score in `m5_xlsr_ft.csv` or appears in any readout except the per-source training diagnostics.
- **In-the-Wild** is never trained on. _Amended 03:25 (consult):_ it is read out, bona fide only, as the MLAAD drop-rule probe (ΔP_FA > +1 point at the inner-fold threshold → drop MLAAD).
- _Amended 03:25 (consult):_ extra DiffSSD spoof capped at ~2k per inner generator (not all 50k); ASV19 = all 40-speaker bona fide plus a ~2.5k spoof channel anchor; MLAAD at ~28% of the spoof side, ≤ 120 clips per model, gated by a CPU corpus probe (MLAAD vs DiffSSD spoof on handcrafted features). Batch mix constant: real 25/35/40 LJ/LibriSpeech/ASV19, spoof 62/28/10 DiffSSD/MLAAD/ASV19.
- **Family exclusion.** The model for fold k never trains on an extra generator from the same family as fold k's validation generators.
  - fold 1: `openvoicev2`, excluding MLAAD OpenVoiceV2
  - fold 2: `xtts_v2`, excluding MLAAD xtts_v1.1, xtts_v2 and vixTTS
  - fold 4: `elevenlabs`, excluding MLAAD ElevenLabs Turbo v2.5, v2 Multilingual and v3
  - An extra generator matching a **holdout** family (`playht`, `wavegrad`) makes the build refuse.

### D2. Bundle
- **Per clip:** `load_audio` (16 kHz mono), then `trim_silence`, then cap at 15 s, then 16-bit FLAC. A clip that peaks above 0.999 is scaled to 0.999 first; that's harmless, because training normalizes each input.
- **Too-short clips:** under 0.5 s after trimming, a clip is dropped and counted.
- **Decode failures:** listed; the build fails if they exceed 0.5%.
- **Codec round-trips:** MP3, Opus, AAC, AMR-NB (8 kHz) and μ-law (8 kHz). Each is decoded back through `load_audio` and re-trimmed. _Amended 03:20 (plan section 0):_ the copies are made **on the box** at setup (32 cores, apt ffmpeg), not on the Mac, because the home disk has 14 GiB free and five variants would multiply a 6.7 GB upload. The same script runs locally in tests with the local ffmpeg. Encoders a box lacks are recorded and skipped, not faked.
- **Manifest columns:** id, original path, label, generator, source, group, fold, `train_scope`, duration, codec-copy ids.
- **Test set:** 1,671 WAVs in `nsa_test.csv` order.
- **Identity:** the bundle's tree-sha is written into its manifest.
- **Weights:** shipped with a sha manifest.
- **Location:** `r2:pa-source/hearsay/bundle/` **only**.

### D3. Lengths
- **Training crop length:** drawn per batch, 70% from `test_duration_sampler` and 30% uniform over 3–14 s, capped at the longest length both classes can reach (C4).
- **Crops:** a fresh random offset each epoch. A clip shorter than the drawn length is used whole, with zero padding plus `attention_mask`; there's **no tiling or padding by repetition**.
- **Normalization:** zero-mean, unit-variance over **valid samples only**, identical to `normalize_windows` on the unpadded clip.
- **Test time:** `prepare_segment(x, max_s=14)`, i.e. the whole trimmed clip, then normalization, identical to training with no crop and no augmentation.
- **Holdout readouts:**
  - test-length crops, fixed seed
  - whole clips up to 14 s
  - length buckets: ≤ 4 s, 4–6 s, > 6 s

### D4. Augmentation
Applied to both classes with the same probability; a test fails if the per-class augmentation rate diverges.
- **Precomputed codec copies:** MP3, Opus, AAC, AMR-NB, μ-law.
- **On the fly:**
  - additive noise at 5–30 dB SNR
  - band-limiting at 3.4, 4 and 7 kHz
  - synthetic reverb: exponential-decay RIR with the direct path aligned
  - gain plus hard clipping. Pure gain is a no-op after normalization, so it isn't used alone.
  - _Amended 03:25 (consult):_ RawBoost's convolutive (linear + nonlinear) algorithm at p = 0.25; training-only trim jitter (30–40 dB, 0–200 ms edges); overall p_aug = 0.65 with band-limit upweighted. Realized augmentation rates are logged per source × class and must match across classes.
- **Scoring:** the clean path applies nothing. The augmented holdout slice is fixed by seed and is the same for every model.
- **Physical replay isn't covered** (simulated reverb only). The report says so.

### D5. Model
- **Backbone:** XLS-R 300M from the local weights (sha-pinned).
  - The CNN feature encoder is frozen.
  - _Amended 03:20:_ only the first `keep_layers` transformer layers are kept. _Amended 03:25 (consult):_ default `keep_layers = 12`, all retained layers trainable (fallback 16 if the learned layer weights concentrate at the top); `layerdrop` 0; SpecAugment on.
  - _Amended 03:25 (consult):_ pooling = learned softmax-weighted layer sum → masked attentive statistics pooling (mean + std); masked mean is the fallback.
- **Head:** a linear head outputting a logit, where **high = synthetic**; score = sigmoid(logit).
- **Optimization:**
  - bf16 on CUDA
  - _Amended 03:25 (consult):_ AdamW, backbone 1e-5 with layer-wise decay 0.85, head 5e-4, weight decay 0.01, 8% warmup + cosine, BCE with label smoothing 0.05
  - label-balanced batches with the constant source mix in D1
- **Consult:** its answer may change N, the LRs, the augmentation probabilities, the ratios and the pooling, as config only.
- **Device rules:** training refuses any device but CUDA. `--cpu-smoke` is for tests and uses a tiny random-init config.

### D6. Fold protocol and selection
- **Pilot:** trains on folds 0–3 and validates on fold 4. It measures throughput and loss, checks the fold-4 minDCF readout and the AUC > 0.5 direction tripwire, and fixes the epoch count E.
- **Ablation:** NSA-only vs NSA+extra, on inner folds only, at most about 2 H100-hours. The winner is chosen on inner-fold minDCF at π = 0.3. The readings at π = 0.5 and both sponsor-code directions are reported, with a flag if the decision flips.
- **Final models:**
  - 5 fold models, each for E epochs, produce `inner_oof` scores for all 16,142 inner rows
  - 1 full model trained on folds 0–4 for E epochs scores the holdout (3,858) and the test set (1,671)
- **The holdout is never used for selection.**

### D7. Outputs
- **`outputs/detector_scores/m5_xlsr_ft.csv`:**
  - columns `path, fold, split, score, logit`
  - `split` is one of `inner_oof`, `holdout`, `test`, matching `handcrafted_v1.csv`. The handoff said `stack`; the main chat will be told.
- **`models/m5_xlsr_ft_<stamp>/meta.json`:**
  - `hearsay.metrics.report` for clean and augmented holdout, per generator, per bona fide source, per length bucket and per augmentation type
  - both sponsor-code readings
  - the M1 bar and the margin
  - config hash, bundle tree-sha, weights sha, git sha
  - spend
- **Checkpoint:** a private HF Hub repo, or a GitHub release if there's no HF token. It is downloaded again and sha-verified.
- **CPU parity:** on 50 test clips, CPU fp32 vs the box's bf16 must have Spearman above 0.99 and max absolute logit difference below 0.05.
- **Report:** `docs/reports/2026-09-26_m5-xlsr-finetune.md`.
- **No TSVs, and nothing in `submissions/`.** The main chat turns the scores into submissions.

### D8. Cloud
- **GPU:** _Amended 03:20:_ A100 SXM4 boxes (about $0.67–0.80/h, 32 cores, CUDA ≥ 12.8) instead of an H100 ($2.20/h, 2 offers, one with a driver too old for current torch). Several boxes run at once so the 6 final models land before the gate.
- **Scripts:** the Powerlifting-Analyzer vast scripts, adapted into `scripts/cloud/`.
- **Path guard:** all R2 paths go through a guard that refuses anything outside `r2:pa-source/hearsay/`.
- **Budget guard:** refuse a launch if credit is below the estimated cost plus $1.50.
- **Pilot before fan-out.**
- **Checkpointing:** a checkpoint goes to R2 every epoch, and a run can resume from it.
- **Self-destroy:** `trap` destroy on EXIT/ERR plus destroy at DONE. After every run the box is verified gone with `vastai show instances`.
- **Ledger:** one row per rental in `docs/reports/cloud-expense-ledger.md`.
- **Secrets:** the API key is never printed.
- **Rentals need Nathan's go-ahead.**

## Acceptance criteria

1. **Tests:**
   - `uv run pytest -q`: the 215 existing tests (the handoff's 152 was stale; `docs/STATUS.md` says 215) plus about 70 new ones all pass.
   - `uv run ruff check .`: clean.
   - Every item in the P2 failure-mode tables (A1–A10, B1–B8, C1–C7, D1–D9, E1–E10, F1–F8, G1–G6) maps to a test, or to a named runtime check recorded in the report.
2. **Pilot:** throughput measured, loss falling, fold-4 minDCF finite, AUC > 0.5, and one resume from an R2 checkpoint.
3. **Outputs:**
   - F1–F6 pass on the real files.
   - `meta.json` is complete.
   - The gate result (M5 vs M1 v2 = 0.1458 on the outer holdout at π = 0.3, **and** pooled 5-fold OOF minDCF, per the consult: a holdout gap under 0.15 is within noise) is recorded honestly. A failed gate is a valid outcome.
4. **Spend:** within the credit, 0 instances at the end, and a ledger row for every rental.
5. **Checkpoint:** downloadable and sha-verified.
6. **No diff** in:
   - `src/hearsay/embed.py`
   - `scripts/{extract_embeddings,train_probe,make_probe_csv}.py`
   - `splits/nsa_folds.csv`
   - `submissions/`
   - the CPU chat's files

   No MPS use.

## Test commands

```bash
cd ~/Projects/hearsay
uv run pytest -q tests/test_m5_manifest.py tests/test_m5_bundle.py tests/test_m5_crops.py tests/test_augment.py tests/test_m5_model.py tests/test_m5_scores.py tests/test_cloud_scripts.py
uv run pytest -q
uv run ruff check .
git diff --stat -- src/hearsay/embed.py scripts/extract_embeddings.py scripts/train_probe.py scripts/make_probe_csv.py splits/ submissions/
```

## Standing failure modes (project plan-review skill)

1. **Laundering, telephony and replay:**
   - clean and augmented holdout slices, plus per-augmentation-type readouts (D4, F5)
   - physical replay is waived; there's no hardware
2. **Overfitting to known generators or speakers:**
   - fold file, family exclusion (A3), out-of-fold scoring (F2), per-generator readouts
   - the report states the gap between inner-fold and holdout scores
3. **Class imbalance and prior:**
   - per-fold class counts (A9) and label-balanced sampling (E4)
   - decisions reported under all four metric readings, flagging any flip
4. **Shortcuts:**
   - per-source AUC of post-trim duration, peak and leading silence on the training stream; each must be below 0.85
   - the MLAAD-vs-LibriSpeech LightGBM probe on handcrafted features
   - clean vs transcoded readouts
5. **Loader format conversion:** the B1 format fixtures (including MP4/AAC) and the B3 clipping check.

## Docs

- **Create:**
  - this run spec
  - the plan and its Codex review
  - the results report
  - the cloud expense ledger
  - the consult records (already written)
- **Update:** the "how used" lines in `CLAUDE.md`'s AI-use disclosure for XLS-R (M5), ASVspoof 2019, MLAAD and LibriSpeech, plus vast.ai.
- **Conditional, confirmed at execution:**
  - an M5 row in `docs/STATUS.md`, which the main chat owns
  - `docs/plan.md`, only if the gate outcome changes the ladder or the cut order

## Results

### Execution notes (dated; deviations from the plan)

- **03:40** ASV19 rows come from the main chat's `splits/nsa_folds_plus_asv19.csv` (commit c7a3275: 5,128 bona fide + 2,520 A01–A06 anchor, speaker-grouped into inner folds) instead of a second selection here; they keep their fold (never trained on by that fold's model) and are `extra_asv19`, never exported. The bundle applies the ≥ 2.0 s cut and the 0.5× duration matching on top (2,496 bona fide + 1,092 spoof kept).
- **03:58** Bundle `m5_bundle/v1`: 44,850 training rows (3,724 short extras dropped, 0 decode errors, 0 cross-scope PCM collisions, 0 core clips identical to a test clip), 1,671 test rows, 5.3 GB; trivial-feature AUCs per scope all ≤ 0.64 (`diagnostics.json`). Extra DiffSSD spoof capped at 1,000 per generator.
- **04:00** Main chat commit 133e536 made `prepare_segment` band-limit at 7.25 kHz by default. Adopted: `hearsay.handcrafted.band_limit` runs once per clip, after augmentation, on every clip the model sees; the bundle stays raw (trimmed only). Fusion exports use the 8 s default cap (`DEPLOY_MAX_S`); 14 s whole clips are `diag_*_14s.csv` only. The 7 kHz augmentation op was dropped (3.4 and 4 kHz telephony only).
- **04:05** Gate extended (main chat + oversight): beat M1 on the outer holdout **and** the pooled 5-fold OOF **and** In-the-Wild bona fide P_FA at the OOF minDCF threshold ≤ M1's (read from M1's `meta.json` stress block). The bar is the NSA-only M1 selected by its `train` field, not by timestamp; M1b is reported beside it.
- **04:30** MLAAD probe A (`outputs/m5_probe_a.json`): MLAAD-vs-DiffSSD spoof AUC 0.90 on augmented handcrafted features; LibriSpeech 66% and VCTK 90% land on the MLAAD side → MLAAD capped at **12%** of the spoof side (config), pending the ITW read-out.
- **04:29–04:51** Pilot #1 (A100, instance 52716242, $0.61/h): setup 9 min, codec pass 16,015 variants in 273 s (no AMR-NB encoder on the image; MP3/Opus/AAC/μ-law), then FAIL: the trainer's tree check ran stale code (code.tgz built before the `codecs`-dir exclusion patch). Reaper destroyed the box within 60 s. Fix: `scripts/cloud/push_code.sh` rebuilds code.tgz + TREE_SHA + meta and pushes them in one step. Also found and fixed: the box's rclone config needed `no_check_bucket = true` (R2 writes were 403 while reads worked); the reaper exited before any job existed and used bash-4 arrays on macOS bash 3.2.
- **04:54** Pilot #2 launched (instance 52718897) with the codec variants pulled from R2.


## Appendix: P2 failure-mode tables (frozen 03:05; ids referenced by the plan and tests)

**A. Manifest and fold discipline**

| # | Failure mode | Test |
|---|---|---|
| A1 | A holdout row (fold `holdout`, generator `playht`/`wavegrad2`, or a holdout bona fide group, including extra rows from `nsa_train_full`) reaches any training set | absent from all 6 training sets |
| A2 | Fold k's validation rows appear in fold k's training set | per-fold disjointness |
| A3 | An extra generator from fold k's validation family trains fold k's model | exclusion test + counterfactual (empty exclusion map must fail) |
| A4 | A holdout-family alias sneaks in through extra data | build refuses loudly |
| A5 | Extra NSA rows get the wrong fold | inherit from the group → fold map; synthetic manifest |
| A6 | Extra rows duplicate sample rows | dedupe by path |
| A7 | In-the-Wild paths end up in the manifest | refused |
| A8 | `splits/nsa_folds.csv` modified | sha256 tripwire |
| A9 | A fold has only one class | both classes in all 6 training and 5 validation sets |
| A10 | ASV19 subsampling not reproducible | fixed seed → identical rows |

**B. Bundle**

| # | Failure mode | Test |
|---|---|---|
| B1 | Format conversion errors | fixtures WAV int16/float32, 22 kHz stereo, MP3, FLAC, MP4/AAC, Opus → 16 kHz mono |
| B2 | Trim differs between bundle and test time | bundled clip == `trim_silence(load_audio(x))` up to FLAC quantization |
| B3 | 16-bit FLAC clips peaks above 1.0 | 1.5-peak input → 0.999, unchanged after normalization |
| B4 | Length cap / very short clips | cap 15 s; extras < 0.5 s dropped with a count; core/test never dropped |
| B5 | Codec copies shift time, add lead silence, or change label | length ±5%, lag < 50 ms, lead < 30 ms after re-trim, label unchanged; each codec |
| B6 | Decode failures drop rows silently | extras: listed, abort > 0.5%; core/test: abort |
| B7 | Stale or mismatched bundle | tree-sha over every file; trainer refuses on mismatch |
| B8 | Test bundle order/count drift | 1,671 rows in `nsa_test.csv` order |

**C. Lengths and crops**

| # | Failure mode | Test |
|---|---|---|
| C1 | Crops not reproducible | same (seed, epoch) → same crop; new epoch → new offset |
| C2 | Tiling or repeat-padding returns | 2 s clip under 5 s crop stays 2 s + zeros, mask 0, no repeated samples |
| C3 | Length mix wrong | 70% ± 3% test-duration draws over 10k; all in [3, 14] |
| C4 | Length becomes a label shortcut | per-scope and pooled `max(AUC,1−AUC)` of duration ≤ 0.85 on the model's training rows; gate in the trainer |
| C5 | Normalization includes padding | valid region == `normalize_windows(clip)` |
| C6 | Train/test preprocessing diverge | `test_transform` == training transform with no crop/augment, bit for bit |
| C7 | Mixed crop lengths within a batch | one length per batch |

**D. Augmentation**

| # | Failure mode | Test |
|---|---|---|
| D1 | Changes length or label | parametrized over every op |
| D2 | SNR inaccurate | measured within ±0.5 dB |
| D3 | Band-limit leaks | > 40 dB down above cutoff |
| D4 | Reverb shifts onset or zeroes output | lag < 5 ms; finite, non-zero |
| D5 | Gain a no-op after normalization | gain + hard clip changes the output; pure gain documented invariant |
| D6 | Augmentation rate depends on class | per-class rate within 3 points; counterfactual |
| D7 | NaN/inf on silent input | all-zero clip stays finite |
| D8 | "Clean" evaluation not clean | identity |
| D9 | Augmented holdout slice differs between models | fixed-seed determinism |

**E. Model and training**

| # | Failure mode | Test |
|---|---|---|
| E1 | Wrong layers train | CNN + lower layers unchanged after a step; top N change; counterfactual N = 0 |
| E2 | Attention mask ignored | padded vs unpadded logit within 1e-4 |
| E3 | Score direction flips | spoof = 1; runtime abort if fold AUC ≤ 0.5 |
| E4 | Sampler breaks quotas | balance ± 5%, NSA ≥ 50%, MLAAD ≤ cap, excluded families absent |
| E5 | Non-deterministic | seeded first-step loss identical |
| E6 | Config/artifact drift | hashes in `meta.json`; scorer refuses on mismatch |
| E7 | NaN loss continues | abort, nonzero exit |
| E8 | Checkpoint doesn't round-trip | save/load identical logits |
| E9 | Selection touches the holdout | training/validation row sets disjoint from holdout; gate values unchanged when holdout rows are perturbed |
| E10 | Training on the local MPS GPU | trainer refuses non-CUDA except `--cpu-smoke` |

**F. Scores and readouts**

| # | Failure mode | Test |
|---|---|---|
| F1 | Schema/count mismatch | columns; `split` ∈ {inner_oof, holdout, test}; 16,142 / 3,858 / 1,671 (or zero `inner_oof` with `stackable:false`) |
| F2 | OOF row scored by a model that saw its fold | `fold` == scoring model's fold; each row once |
| F3 | Paths don't join | exact match with `nsa_folds.csv` and `nsa_test.csv` |
| F4 | Non-finite, out of range, wrong direction | finite, [0, 1], 1 = synthetic |
| F5 | `meta.json` incomplete | required keys |
| F6 | CPU differs from box; bundled differs from raw | fp32 box vs Mac max Δlogit < 1e-2, Spearman > 0.999; WAV vs FLAC max Δlogit < 0.05 |
| F7 | 13.6 s clip truncated | scored whole |
| F8 | Code writes into another lane | static: no `submissions/`, no `.tsv` writing, no write-mode `nsa_folds.csv` |

**G. Cloud and spend**

| # | Failure mode | Test |
|---|---|---|
| G1 | A script touches powerlifting data in R2 | every rclone destination in `scripts/cloud/*.sh` and `scripts/m5_*.py` starts with `r2:pa-source/hearsay/`; guard refuses otherwise |
| G2 | A box outlives a crash or DONE | launcher trap; reaper on DONE/FAIL/stall/11:45; 0 instances verified |
| G3 | Overspend | guard: credit − running commitments ≥ est + $1.50; ledger row per launch |
| G4 | Box dies mid-run | checkpoint to R2 every eval; resume exercised in the pilot |
| G5 | vast API key leaks | no `set -x`, never echoed, never on the box |
| G6 | Wrong weights on the box | HF revision pinned; config sha checked; tree-sha checked |
