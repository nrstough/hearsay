# M5: XLS-R 300M fine-tune on vast.ai (Sat Sep 26, 2026)

_Status: in progress. Numbers marked TBD are filled in as runs land._

Run spec: `docs/specs/2026-09-26_m5-xlsr-finetune.md`. Plan: `docs/reports/2026-09-26_m5-xlsr-finetune-plan.md`. Consult: `docs/consults/2026-09-26_m5-extra-data-finetune_RESPONSE.md`.

## What M5 is

A truncated XLS-R 300M (first 12 of 24 transformer layers, all trainable; CNN feature encoder frozen) with a learned softmax-weighted sum over the retained layers, masked attentive statistics pooling (mean + std) and a linear head, trained with BCE (label smoothing 0.05), AdamW with layer-wise LR decay (backbone 1e-5, γ 0.85, head 5e-4), 8% warmup + cosine, bf16 on an A100, gradient clip 1.0. Output: a logit that increases with synthetic likelihood; score = sigmoid(logit).

**Input path (identical to M1 v3 and every detector):** `load_audio` (16 kHz mono) → `hearsay.handcrafted.band_limit` (7.25 kHz Kaiser low-pass matching the NSA test set's wall) → `trim_silence` → cap at 8 s → per-input zero-mean/unit-variance normalization. No tiling, no repeat padding. Bundle clips are trimmed at build time and band-limited once at train time; the raw/Docker scorer (`scripts/m5_score.py`) uses `prepare_segment` unchanged.

**Training data** (`outputs/manifests/m5_manifest.csv`, bundle `m5_bundle/v1`, 44,850 rows after the drop rules; 5.3 GB FLAC):

| Scope | Rows | Notes |
|---|---|---|
| core (fold file) | 20,000 | 10 DiffSSD generators, LJ Speech, LibriSpeech; holdout = playht + wavegrad2 + 26 bona fide groups |
| extra_nsa | 15,083 | rest of `nsa_train_full`, inheriting the fold of their group; extra spoof capped at 1,000 per generator; holdout-group rows dropped |
| extra_asv19 | 2,496 bona fide + 1,092 spoof | from the main chat's `splits/nsa_folds_plus_asv19.csv` (40 VCTK speakers, A01–A06 anchor); ≥ 2.0 s after trim, spoof duration-matched to bona fide |
| extra_mlaad | 6,180 spoof | 143 TTS models; ≤ 120 clips per model per training set; excluded per fold family (OpenVoice → fold 1, XTTS → fold 2, ElevenLabs → fold 4) |

Extras are training-only: never scored, never in the holdout. Batch mix (constant): label-balanced; real 25/35/40 LJ/LibriSpeech/ASV19; spoof 62/28/10 DiffSSD/MLAAD/ASV19.

**Crops:** one crop length per batch, 70% drawn from the NSA test durations and 30% uniform 3–14 s, re-cropped every epoch; shorter clips used whole with zero padding + attention mask.

**Augmentation** (both classes, class-blind by construction, p = 0.65, realized rates logged per source × label): additive white/pink noise 5–30 dB SNR, telephony band-limits (3.4, 4 kHz), synthetic reverb (RT60 0.2–0.8 s), gain + hard clipping, RawBoost's convolutive algorithm (p 0.25), trim-threshold jitter, and precomputed codec round-trips (MP3, Opus, AAC, AMR-NB, μ-law; made on the box for a seeded 30% of training clips and every holdout clip). Physical replay is **not** covered (no hardware; simulated reverb only).

**Fold discipline:** five fold models produce out-of-fold scores for the 16,142 inner rows; one full model (all inner folds) scores the 3,858 holdout rows and the 1,671 test files. A fold model never trains on its validation fold, on any holdout row, or on extra generators from its validation families. Selection is wall-clock fixed (steps set by the pilot's throughput); no checkpoint is picked on a validation fold.

## Gate

Rule (main chat + oversight + consult): M5 replaces M1 as the primary deep detector only if it beats M1 on the outer holdout **and** on the pooled 5-fold OOF, **and** its In-the-Wild real-speech false-alarm rate at the OOF minDCF threshold is not worse than M1's. A holdout gap under ~0.15 minDCF is within noise (σ ≈ 0.07–0.10 with two held-out generators). Either way M5's scores go to the fusion stacker as their own column.

| | M1 (bar) | M5 |
|---|---|---|
| outer-holdout minDCF (π = 0.3) | TBD (newest `models/m1_*/meta.json`; v2 was 0.1458) | TBD |
| pooled 5-fold OOF minDCF | TBD | TBD |
| In-the-Wild bona fide P_FA at the inner threshold | 1.2% (M1 v2) | TBD |

## Results

### Pilot and ablation (inner folds, out-of-fold, 8 s deployment transform, π = 0.3)

M1 per-fold OOF recomputed from its saved embeddings (`outputs/m5_runs/m1_v{2,3}_oof_by_fold.json`); M5 rows from the run logs under `outputs/m5_runs/`.

| Fold (held-out generators) | M1 v3 (bar) | M5 NSA-only, 2,500 steps | M5 NSA+extra, 2,500 steps |
|---|---|---|---|
| 0: grad_tts + unit_speech | 0.524 | 0.528 (EER 20.0%) | TBD |
| 4: elevenlabs | 0.415 | 0.762 (EER 19.1%) | **0.410 (EER 8.8%)** |
| pooled (5 folds) | 0.257 | | |

- **Pilot** (fold 4, 600 steps, LR decayed to zero): 0.81; throughput ~190 clips/s at 3.4 s crops on an A100 40 GB, 2,500 steps ≈ 16–20 min including evals and checkpoint pushes.
- **The extra pools are the difference on the unseen commercial generator**: ASV19 bona fide (40 VCTK speakers) + the 2.5k A01–A06 anchor + MLAAD at 12% of the spoof side take fold 4 from 0.76 to 0.41. NSA-only fine-tuning fits the seen generators (training loss near the label-smoothing floor) and does not transfer.
- Realized augmentation rates per source × label: 0.646–0.655 in every cell (class-blind confirmed).
- Conservative recipes on fold 4 (frozen backbone + learned pooling head; top-3 layers at LR 5e-6): TBD.

_TBD: pilot throughput; ablation (NSA-only vs NSA+extra, folds 0 and 4); holdout clean / augmented-slice / test-length-crop / 14 s readouts; per generator, per bona fide source, per length bucket, per augmentation op; both sponsor-code readings; MLAAD probe verdict; CPU parity; spend._

## What worked, what didn't

_TBD._

## Artifacts

- Scores: `outputs/detector_scores/m5_xlsr_ft.csv` (path, fold, split ∈ {inner_oof, holdout, test}, score, logit).
- Model + readouts: `models/m5_xlsr_ft_<stamp>/{meta.json, model/}`; checkpoint on the HF Hub (private): TBD.
- Bundle: `r2:pa-source/hearsay/bundle/v1` (tree sha in `bundle_meta.json`); runs: `r2:pa-source/hearsay/runs/`.
- Spend: `docs/reports/cloud-expense-ledger.md`.
