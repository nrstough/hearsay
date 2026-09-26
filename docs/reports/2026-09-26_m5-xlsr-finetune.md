# M5: XLS-R 300M fine-tune on vast.ai (Sat Sep 26, 2026)

_Status: done (Sat Sep 26, 07:45). Gate: **not passed** (holdout and pooled OOF worse than M1; In-the-Wild real-speech false alarms better). M5 ships as a fusion column, not as the primary deep detector._

Run spec: `docs/specs/2026-09-26_m5-xlsr-finetune.md`. Plan: `docs/reports/2026-09-26_m5-xlsr-finetune-plan.md`. Consult: `docs/consults/2026-09-26_m5-extra-data-finetune_RESPONSE.md`.

## What M5 is

The shipped model keeps the first 12 of XLS-R 300M's 24 transformer layers **frozen** (CNN feature encoder frozen too) and trains only a small head on top: a learned softmax-weighted sum over the 13 hidden states (weights peak at layers 5–6), masked attentive statistics pooling (mean + std, 128-d bottleneck) and a linear output; 136k trainable parameters. **Caveat (Codex audit):** in the shipped checkpoint the backbone's final encoder LayerNorm (2,048 params) and `masked_spec_embed` (1,024) were also trainable, a gap in the freezing code fixed afterwards (`train_top = 0` now freezes every backbone parameter, with a test); the effect is a per-channel affine on the last layer's output, not a change to the representation. Trained with BCE (label smoothing 0.05), AdamW at 5e-4, 8% warmup + cosine, bf16 on an A100, 3,000 steps × 32 crops (fold models 2,500). Output: a logit that increases with synthetic likelihood; score = sigmoid(logit). The full fine-tune (all 12 kept layers at LR 1e-5 with layer-wise decay 0.85) was the plan's default; the ablation below replaced it.

**Input path:** `load_audio` (16 kHz mono) → `trim_silence` → `hearsay.handcrafted.band_limit` (the 7.25 kHz Kaiser low-pass matching the NSA test set's wall) → cap at 8 s → per-input zero-mean/unit-variance normalization; no tiling, no repeat padding. Bundle clips are trimmed at build time and band-limited once at train time, and the raw/Docker scorer (`scripts/m5_score.py`, `hearsay.m5_data.deploy_transform`) applies the same order, so it reproduces the exported scores (parity below). M1's `prepare_segment` band-limits before trimming; the two differ by at most one 20 ms trim frame at the clip edges (pinned by a test).

**Training data** (`outputs/manifests/m5_manifest.csv`, bundle `m5_bundle/v1`, 44,850 rows after the drop rules; 5.3 GB FLAC):

| Scope | Rows | Notes |
|---|---|---|
| core (fold file) | 20,000 | 10 DiffSSD generators, LJ Speech, LibriSpeech; holdout = playht + wavegrad2 + 26 bona fide groups |
| extra_nsa | 15,083 | rest of `nsa_train_full`, inheriting the fold of their group; extra spoof capped at 1,000 per generator; holdout-group rows dropped |
| extra_asv19 | 2,496 bona fide + 1,092 spoof | from the main chat's `splits/nsa_folds_plus_asv19.csv` (40 VCTK speakers, A01–A06 anchor); ≥ 2.0 s after trim, spoof duration-matched to bona fide |
| extra_mlaad | 6,180 spoof | 143 TTS models; ≤ 120 clips per model per training set; excluded per fold family (OpenVoice → fold 1, XTTS → fold 2, ElevenLabs → fold 4) |

Extras are training-only: never scored, never in the holdout. Batch mix (constant): label-balanced; real 25/35/40 LJ/LibriSpeech/ASV19; spoof **78/12/10** DiffSSD/MLAAD/ASV19 (the consult's 62/28/10 was the starting point; MLAAD was cut to 12% after the corpus probe, `outputs/m5_probe_a.json`).

**Crops:** one crop length per batch, 70% drawn from the NSA test durations and 30% uniform 3–14 s, re-cropped every epoch; shorter clips used whole with zero padding + attention mask.

**Augmentation** (both classes, class-blind by construction, p = 0.65, realized rates logged per source × label): additive white/pink noise 5–30 dB SNR, telephony band-limits (3.4, 4 kHz), synthetic reverb (RT60 0.2–0.8 s), gain + hard clipping, RawBoost's convolutive algorithm (p 0.25), trim-threshold jitter, and precomputed codec round-trips (MP3, Opus, AAC, AMR-NB, μ-law; made on the box for a seeded 30% of training clips and every holdout clip). Physical replay is **not** covered (no hardware; simulated reverb only).

**Fold discipline:** five fold models produce out-of-fold scores for the 16,142 inner rows; one full model (all inner folds) scores the 3,858 holdout rows and the 1,671 test files. A fold model never trains on its validation fold, on any holdout row, or on extra generators from its validation families. Selection is wall-clock fixed (steps set by the pilot's throughput); no checkpoint is picked on a validation fold. **Two caveats:** the fold-0 and fold-4 models in the export are the same runs whose numbers chose the frozen recipe over the fine-tune (06:40), so those 6,267 `inner_oof` rows and the pooled-OOF gate reading carry recipe-selection bias in M5's favour; and the fold models ran 2,500 steps while the full model ran 3,000 (more data), a deviation from the spec's "same E".

## Gate

Rule (main chat + oversight + consult): M5 replaces M1 as the primary deep detector only if it beats M1 on the outer holdout **and** on the pooled 5-fold OOF, **and** its In-the-Wild real-speech false-alarm rate at the OOF minDCF threshold is not worse than M1's. A holdout gap under ~0.15 minDCF is within noise (σ ≈ 0.07–0.10 with two held-out generators). Either way M5's scores go to the fusion stacker as their own column.

| | M1 v3 (bar, `models/m1_…_0518`) | M5 (frozen, `models/m5_xlsr_ft_20260926-0741`) | clause |
|---|---|---|---|
| outer-holdout minDCF (π = 0.3) | **0.159** (EER 2.9%) | 0.363 (EER 6.0%) | fail (−0.20, beyond the ±0.10 noise) |
| pooled 5-fold OOF minDCF | **0.257** (EER 6.9%) | 0.302 (EER 11.0%) | fail |
| In-the-Wild bona fide P_FA at the OOF minDCF threshold | 0.7% | **0.45%** (2,000 clips) | pass |

M1b v3 (NSA + ASV19 bona fide, `…_0521`) for the fusion decision: holdout 0.072, ITW P_FA 0.8%. **Verdict: keep M1 as the primary deep detector; M5 goes into the stacker as its own column** (`outputs/detector_scores/m5_xlsr_ft.csv`, 16,142 `inner_oof` + 3,858 `holdout` + 1,671 `test` rows, all finite, 1.0 = synthetic).

## Results

### Pilot and ablation (inner folds, out-of-fold, 8 s deployment transform, π = 0.3)

M1 per-fold OOF recomputed from its saved embeddings (`outputs/m5_runs/m1_v{2,3}_oof_by_fold.json`); M5 rows from the run logs under `outputs/m5_runs/`.

| Fold (held-out generators) | M1 v3 (bar) | fine-tune, NSA-only | fine-tune, NSA+extra | **frozen backbone, NSA+extra** |
|---|---|---|---|---|
| 0: grad_tts + unit_speech | 0.524 | 0.528 (EER 20.0%) | 0.515 (EER 21.0%) | **0.520 (EER 17.9%)** |
| 4: elevenlabs | 0.415 | 0.762 (EER 19.1%) | 0.410 / 0.377 (two seeds; EER 8.8 / 8.2%) | **0.308 (EER 5.9%)** |

All M5 rows: 2,500 steps × 32 crops, augmentation p = 0.65, MLAAD 12% of the spoof side. **Final fold models (frozen recipe, 2,500 steps), out-of-fold, vs M1 v3:**

| Fold | held-out generators | M1 v3 | M5 frozen |
|---|---|---|---|
| 0 | grad_tts, unit_speech | 0.524 | 0.520 |
| 1 | diffgan_tts, openvoicev2 | 0.090 | 0.263 |
| 2 | pro_diff, xtts_v2 | 0.016 | 0.026 |
| 3 | your_tts | 0.000 | 0.000 |
| 4 | elevenlabs | 0.415 | 0.308 |
| **pooled (16,142 rows)** | | **0.257** (EER 6.9%) | **0.302** (EER 11.0%) |

M5 wins the hardest fold (ElevenLabs) and ties three, but loses fold 1 (diffgan_tts + openvoicev2) by 0.17, and the pooled 5-fold OOF, which is one of the three gate clauses, comes out worse than M1 (0.302 vs 0.257). Pooling also penalizes M5 slightly because its five fold models are separately trained heads with their own logit scales, whereas the metric is computed on the concatenation; the per-fold numbers are the fairer comparison.

"Fine-tune" = all 12 kept layers trainable at LR 1e-5 (LLRD 0.85); "frozen" = the same 12-layer XLS-R with only the learned layer-weighted sum, attentive statistics pooling and head trained. Seed-to-seed noise on a fold ≈ ±0.03.

**Decision (06:40): the frozen-backbone recipe for all six final models.** It wins fold 4 by ~0.08 (≈3× the seed noise) over the fine-tune and by 0.11 over M1, ties on fold 0, runs twice as fast (12 min per fold model), and its learned layer weights peak at layers 5–6, consistent with M1's layer-7 optimum. The full fine-tune's failure mode is the classic one: training loss near the label-smoothing floor on the seen generators, poor transfer to the unseen commercial one.

- **Pilot** (fold 4, 600 steps, LR decayed to zero): 0.81; throughput ~190 clips/s at 3.4 s crops on an A100 40 GB, 2,500 steps ≈ 16–20 min including evals and checkpoint pushes.
- **The extra pools are the difference on the unseen commercial generator**: ASV19 bona fide (40 VCTK speakers) + the 2.5k A01–A06 anchor + MLAAD at 12% of the spoof side take fold 4 from 0.76 to 0.41. NSA-only fine-tuning fits the seen generators (training loss near the label-smoothing floor) and does not transfer.
- Realized augmentation rates per source × label: 0.646–0.655 in every cell (class-blind confirmed). Trivial-feature AUCs on the realized training stream (1,920 items after sampling, crop, augmentation and band-limit; `outputs/m5_runs/stream_shortcut.json`): pooled duration 0.6881, peak 0.5585, RMS 0.5429, leading silence 0.5083; worst per source 0.7333 (DiffSSD duration: long spoofs are cropped to the batch length while short real clips are used whole, so realized length carries a mild cue; the attention mask exposes it). All under the 0.85 gate.
- The top-3-layers recipe (LR 5e-6) was cancelled once the frozen result came in; not measured.

### Full model on the outer holdout (3,858 rows; playht + wavegrad2 + 26 bona fide groups)

| Readout | minDCF | EER |
|---|---|---|
| clean, whole clip ≤ 8 s (the fusion export) | **0.363** | 6.0% |
| whole clip ≤ 14 s (diagnostic) | 0.359 | |
| test-length crops, 3.0–13.6 s drawn from the test distribution (diagnostic) | 0.516 | 8.9% |
| augmented slice, one op per clip (diagnostic) | 0.465 | 8.3% |
| per op: gain + clip / band-limit / RawBoost-conv / reverb / noise | 0.30 / 0.37 / 0.42 / 0.48 / 0.57 | |
| **transcoded** (one codec per clip, all 3,858; `outputs/m5_runs/codec_readout.json`) | **0.328** vs 0.363 clean on the same rows | 6.3% |
| per codec, transcoded vs clean on its rows: μ-law 8 kHz / Opus 20 kbps / MP3 48 kbps / AAC 40 kbps | 0.22 vs 0.29 / 0.27 vs 0.32 / 0.36 vs 0.41 / 0.37 vs 0.37 | |
| per generator: playht / wavegrad2 (each vs all bona fide) | 0.051 / 0.578 | 1.1% / 7.9% |
| per bona fide source: LJ Speech / LibriSpeech (each vs all spoof) | 0.167 / 0.425 | 3.8% / 8.8% |
| per length bucket: ≤ 4 s / 4–6 s / > 6 s | 0.595 / 0.342 / 0.286 | 9.0 / 6.2 / 5.7% |
| minDCF at π = 0.5 | 0.267 | |
| sponsor code as-is / flipped | 1.000 / 0.179 | |

- **Where M5 loses to M1:** wavegrad2 (0.58 vs M1's 0.21 per-generator) and LibriSpeech real speech (0.43 vs 0.16). playht is easy for both.
- **Short clips are the weak spot:** ≤ 4 s clips score 0.60 vs 0.29 above 6 s; the test set's median is 3.4 s. The frozen head was trained on test-length crops, but the attentive pooling has less to work with on short inputs.
- **Additive noise is the hardest laundering** (0.57 on the noise slice); gain/clipping and band-limiting barely move the score, and **transcoding does not hurt** (0.33 vs 0.36 clean): the codec variants were in training, and MP3/Opus/AAC/μ-law strip the same high band the test set lacks. AMR-NB was not available on the box's ffmpeg, so narrowband telephony is represented by μ-law at 8 kHz only.
- Score–duration coupling on bona fide: Spearman(score, log duration) = −0.12 (mild; longer real clips look slightly more real).
- Test set: mean score 0.29, 25.8% above 0.5 (M1 v3: 28.9%), consistent with the ~30% synthetic prior; the same direction as M1 (`sponsor_code_asis` = 1.0 for every detector because the sponsor's code treats high as bona fide).

### Runs, throughput, spend

- A100 SXM4/PCIe on vast.ai at $0.61–0.81/h, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`; box setup 8–9 min (apt, pip, 5.3 GB bundle pull at 6 Gbps, XLS-R from the HF Hub, tree-sha check); codec pass 16,015 variants in 273 s on 16 cores.
- Fine-tune: ~190 clips/s at 3.4 s crops; 2,500 steps ≈ 16–20 min with evals and checkpoint pushes. Frozen recipe: 2,500 steps ≈ 12 min; the full model 3,000 steps ≈ 13 min plus 7 min of holdout/test scoring.
- 12 instances over the day (six never booted and were destroyed by the launcher within 10 min each; two chains failed on my own config-string bugs; one resume-probe attempt failed on a tree-sha mismatch, see below); **total spend $5.42** ($38.37 → $32.95 credit; $5.09 through the final models, $0.33 for the two resume probes) (`docs/reports/cloud-expense-ledger.md`).
- CPU inference (Docker path, `scripts/m5_score.py`): 0.19–0.25 s per clip on the M3 Pro; 1,671 test clips ≈ 6 min. Parity over 50 test clips (`outputs/m5_runs/parity_f6.json`): same input (bundled FLAC), Mac CPU vs A100: max |Δlogit| 0.00445, Spearman 1.0; raw WAV vs bundled FLAC, both on CPU: max 0.01117, median 0.0; combined: max 0.012, Spearman 0.9999. The first parity run showed a 0.28 outlier because the CPU scorer band-limited before trimming (M1's order) while the exports trimmed first; the scorer now uses the exported order (commit after the 07:50 critique).
- Checkpoint: private HF Hub repo `nrs124554433/hearsay-m5-xlsr` (backbone 657 MB safetensors + head + `hashes.json`), re-downloaded and sha-verified. Provenance: the base weights were fetched from the Hub by the box without a revision pin, but the Hub repo's newest commit is 2022-08-10 (`outputs/m5_runs/weights_provenance.json`), so the download was necessarily revision `1a640f32`, whose `pytorch_model.bin` LFS sha256 `d5e49057…` equals the Mac's local copy; the pin and per-file sha check now run on every box.

## What worked, what didn't

**Worked**
- The extra real speech. ASV19's 40 VCTK speakers (plus the 2.5k anchor) and the MLAAD slice took the ElevenLabs fold from 0.76 to 0.41 for the fine-tune; the consult's "real-side diversity beats spoof volume" call held.
- Freezing the backbone. With only the layer-weighted sum, attentive-stats pooling and head trained, fold 4 went to 0.31 (M1: 0.41) and In-the-Wild false alarms fell below M1's (0.45% vs 0.7%); the learned layer weights peaked at layers 5–6, matching M1's layer-7 optimum. The full fine-tune fit the seen generators (loss at the label-smoothing floor) and transferred worse.
- Class-blind augmentation by construction: realized rates 0.645–0.657 in every source × label cell, logged per run.
- The guards: sha-pinned fold file, tree-sha'd bundle, hash-gated checkpoint, fold-family exclusions, shortcut gate on each model's own training rows (worst trivial-feature AUC 0.62), a Mac-side reaper so the vast API key never left the Mac, and a budget guard. Every failure was caught by one of them.

**Didn't**
- M5 does not beat M1 on the outer holdout (0.36 vs 0.16) or the pooled OOF (0.30 vs 0.26). It loses on wavegrad2 and on LibriSpeech real speech and on clips under 4 s; it wins the unseen commercial generator (ElevenLabs) and In-the-Wild real speech. That profile is a useful second column for the stacker, not a replacement.
- fold 1 (diffgan_tts + openvoicev2): 0.26 vs M1's 0.09. Not diagnosed within the time box; the MLAAD OpenVoiceV2 family is excluded from that fold's training, so it is not a leak in either direction.
- Operational: six vast hosts never booted (each destroyed by the launcher within 10 min), one run failed on a stale `code.tgz` (fixed with `push_code.sh`), and two full-model jobs failed because zsh's `$VAR:e` modifier emptied a base64 config string (fixed with `${VAR}`; documented in `swap_chain.sh`). The resume-from-R2 path was exercised on a box after the audit (300 steps → checkpoint to R2 → local checkpoint deleted → resumed at 300 from R2 → 600; `outputs/m5_runs/resume_probe/`). The chain, reaper and NEXT-jobs mechanism otherwise ran three boxes unattended.
- Not covered: physical replay (simulated reverb only), AMR-NB (the box's ffmpeg had no encoder; MP3/Opus/AAC/μ-law were used), the top-3-layers recipe (cancelled), and the seed-noise study beyond the one fold-4 replicate (0.41 vs 0.38).

**If there were another hour:** train the frozen head longer on short crops only (the ≤ 4 s bucket), or stack M5 with M1 on the fold file and read the holdout of the pair, which the main chat's fusion will do anyway.

## Artifacts

- Scores: `outputs/detector_scores/m5_xlsr_ft.csv` (path, fold, split ∈ {inner_oof, holdout, test}, score, logit).
- Model + readouts: `models/m5_xlsr_ft_20260926-0741/{meta.json, model/, scores.csv}`; checkpoint on the HF Hub: private `nrs124554433/hearsay-m5-xlsr` (commit 77dbf281). MLAAD probe: `outputs/m5_probe_a.json`; parity: `outputs/m5_runs/parity_f6.json`; M1 per-fold OOF: `outputs/m5_runs/m1_v{2,3}_oof_by_fold.json`; ITW scores: `outputs/m5_runs/itw_bonafide_m5.csv`.
- Bundle: `r2:pa-source/hearsay/bundle/v1` (tree sha in `bundle_meta.json`); runs: `r2:pa-source/hearsay/runs/`.
- Spend: `docs/reports/cloud-expense-ledger.md`.
