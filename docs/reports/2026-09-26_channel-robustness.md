# Channel robustness on the frozen deep path (Sat Sep 26, 2026)

**Summary.**
- **λ̂** (how wild the test set is) = 0.51 (95% CI 0.12–0.64): partly wild. The fusion_v2 choice is robust to it.
- **The 7.2 kHz wall** is a smooth low-pass, not an MP3 or AAC round-trip, so the codec refit (C) did not run.
- **M3's false-alarm suppression is kept:** it is perturbation-stable, it has the best miss rate on unseen generators, and its step never raised the fused minDCF.
- **New:** 20 dB additive noise breaks handcrafted v5 and costs the shipped rule 0.27 holdout minDCF.
- The first λ̂ (0.947) was wrong and is kept below as a documented mistake.

Run spec: `docs/specs/2026-09-26_channel-robustness.md`. Plan: `docs/reports/2026-09-26_channel-robustness-plan.md` (Codex plan review: `docs/reports/2026-09-26_channel-robustness-plan-review.md`). Handoff: `docs/handoffs/2026-09-26_channel-robustness-handoff.md`. Diagnostic only: this rung produces no TSV and changes no shipped rule.

## A. How wild is the test set? (λ̂)

**Answer: partly. λ̂ = 0.51 (95% CI 0.12–0.64), an interval that contains the fusion consult's 0.32 crossover.** The test set sits between our clean holdout and In-the-Wild on most channel statistics. That is not strong enough to say which validation set represents it. For the shipped rule this does not matter: fusion_v2 beats fusion_v1 on both the holdout (0.0065 vs 0.014) and In-the-Wild (0.228 vs 0.260), so the choice is robust to λ. Source: `outputs/channel/lambda.json`, `scripts/channel_lambda.py`.

**Method.**
- Every clip is prepared the same way (band match → silence trim → 8 s cap → RMS normalize), then described by label-free channel statistics:
  - band levels up to 7 kHz
  - the compression detector's spectral-hole and noise-floor statistics (3–7 kHz)
  - noise-floor level and stationarity
  - a decay (reverberation) proxy
  - flatness, loudness spread and pause fraction
- Columns that read above 7 kHz are dropped, because the test files pass two low-passes and everything else one.
- A logistic classifier separates holdout real speech (1,858 clips: LJ and LibriSpeech) from In-the-Wild real speech (2,000 clips, 54 speakers). Out-of-fold scores come from 5-fold CV grouped by speaker, with LJ as a single speaker.
- λ̂ is adjusted classify-and-count, (q − FPR)/(TPR − FPR). Its interval resamples test files and reference speakers. The mean posterior is descriptive only: it is biased toward 0.5 when the domains overlap (pinned by a test).

**Result.**

| Readout | Value |
|---|---|
| Domain classifier, grouped out-of-fold AUC | 0.84 (TPR 0.84, FPR 0.28) |
| **λ̂** | **0.51** (95% CI 0.12–0.64) |
| Mean posterior P(wild) over the 1,671 test files (descriptive) | 0.56 |
| λ̂ on the 1,170 files the shipped rule calls real | 0.58 |
| λ̂ with LJ removed from the clean side | 0.54 |
| λ̂ without all floor and hole features | 0.80, but the classifier is weak (AUC 0.65) |
| Without the top remaining feature (`local_hole_runs`) | classifier collapses (AUC 0.47); λ̂ refused |
| Test files outside both domains (Mahalanobis p99) | 1.0% (1% expected) |

**Controls, never trained on (share read as "wild"):**

| Control | Share |
|---|---|
| Inner-fold real clips | 10% |
| Inner real after one MP3 64 kbps round-trip | 9% |
| DiffSSD spoof | 19% |
| VCTK clean read speech | 24.7% |

The pre-declared identifiability rule passes, but only just: VCTK sits at 24.7% against a 25% limit.

**Per-feature positions** (0 = the clean median, 1 = the wild median):
- The test set sits between the two domains on the band levels at 5–7 kHz (0.69–0.70), valley depth (0.57) and local-hole statistics (0.35–0.65).
- It sits at the wild end on noise-floor stationarity (1.08).
- It sits at the clean end on deep-hole flicker and persistence (−0.2).

**What went wrong first, and why it matters for the README.** The first full run reported λ̂ = 0.947 with AUC 0.99 and read "wild" as "codec-processed", because one MP3 round-trip moved our clean clips to 91% "wild". That run was driven by `clip_floor_db`, a 1st percentile over the whole spectrogram, which lands in the 7–8 kHz stopband. Action B measured the test files at −118 dB as delivered and −157 dB after the pipeline's second low-pass. So the feature measured the double filtering (and In-the-Wild's own band limits), not the recording channel. Dropping it under the rule already declared for >7 kHz columns gave the numbers above, and the codec reading disappeared (MP3 control 91% → 9%). This is the same class of shortcut as the 7.2 kHz wall in `architecture.md` §11: a band no training clip has, read through a feature that did not look like a band feature.

**Limits.**
- In-the-Wild stands in for "wild"; telephony and physical replay are not represented.
- The classifier rests on a handful of hole statistics (dropping the top one collapses it), so λ̂ is a weak, single-method estimate.

## E. Is M3 memorising, and what does its suppression step cost?

**Verdict under the pre-declared rule (run spec D9, revised): kept.** None of the three triggers fires, and every probe points away from memorisation. Source: `outputs/channel/m3_perturb.json`, `m3_mlaad.json`, `m3_agreement.json`, `m3_verdict.json`; `scripts/m3_probes.py`.

| Trigger | Measured | Limit |
|---|---|---|
| (a) M3 mean \|ΔAUC\| over 4 perturbations vs M1b | 0.0008 vs 0.0041 | > 2× M1b and > 0.02 |
| (c) unseen MLAAD spoof damped by the shipped rule | 1.2% (7 of 572) | > 5% |
| (d) largest rise in fused holdout minDCF from the M3 step, any perturbation | 0.000 (it lowers minDCF in 4 of 5) | > 0.01 |

**1. Perturbation probe.**
- **Cohort:** 500 outer-holdout clips (250 real: LJ and LibriSpeech; 250 spoof: playht and wavegrad2). These are out of sample for M1b, M5 and handcrafted v5.
- **Scoring:** each clip is cropped once to a test length and scored through the runner's own paths (`hearsay.pipeline.Models`, CPU). It is fused by `FusionConstants.fuse` with the shipped constants (`models/fusion_v2/constants.json`).
- **Perturbations:** MP3 64 kbps at 16 kHz (aligned back to the input), white noise at 20 dB SNR, ±2% speed, and a one-sample shift.
- **Scale:** with 250 real clips, one false alarm is worth 0.037 minDCF, so differences below that are single files.

| Detector | Clean | MP3 | Noise 20 dB | Speed ±2% | Shift 1 | Mean \|ΔAUC\| |
|---|---|---|---|---|---|---|
| M1b v3 | 0.056 | 0.085 | **0.271** | 0.052 | 0.048 | 0.0041 |
| Handcrafted v5 | 0.048 | 0.088 | **0.996** (AUC 0.64) | 0.089 | 0.048 | 0.0899 |
| M5 (fine-tune) | 0.433 | 0.432 | 0.487 | 0.425 | 0.428 | 0.0037 |
| M3 (Spectra-AASIST) | 0.000 | 0.008 | 0.116 | 0.000 | 0.000 | 0.0008 |
| Fused base (fusion_v2 without the M3 step) | 0.020 | 0.048 | 0.269 | 0.020 | 0.028 | 0.0029 |
| **Fused, shipped rule** | **0.012** | **0.037** | **0.269** | **0.000** | **0.008** | 0.0029 |

(Cells are holdout minDCF under the brief cost.)

The M3 step fires on 4.0–5.2% of real clips and on no spoof clip under any perturbation. Under noise it fires on nothing, because noise pushes M3 out of its "strongly bona fide" region (logit < −3). A memorising model is fragile to label-preserving perturbations; M3 is the least fragile of the four.

**2. Unseen-generator probe.**
- **Cohort:** 572 MLAAD English spoof clips, 4 per model across 143 models, each cropped to a test length.
- **Readout:** miss rate at each detector's own inner-OOF brief-cost threshold.

| Detector | Miss rate on unseen MLAAD spoof |
|---|---|
| **M3** | **14.9%** |
| M1b v3 | 22.7% |
| M5 (trained on MLAAD, in-sample) | 28.7% |
| Handcrafted v5 | 56.1% |

- M3 falls below −3 on 1.75% of these clips.
- The shipped rule's actual conjunction (M3 < −3 **and** base rank > 0.5) damps 1.2% (7 files).
- Those 7 files come from seven models, one clip each: MiniMax-Speech-2.8-Turbo, GPA-v1.0, dots.tts (two variants), DeepGram and MOSS-TTS (two sizes).

**3. Agreement (descriptive only, per the Codex plan review).** Spearman(M1b v3, M3) is 0.796 on holdout rows and 0.620 on test rows (gap 0.175, 95% CI 0.141–0.214). The two sets differ in class mix and generators, so the gap is not evidence of memorisation by itself. It says the two models disagree more on the test audio.

**What remains unknown.** M3's training data is still undisclosed; ten minutes on the model card did not settle it (see below). A perfect clean holdout score (AUC 1.000, minDCF 0.000) is consistent with M3 having seen DiffSSD-like data, but perturbation stability and the best unseen-generator miss rate argue for generalisation, not a lookup.

**Model card check** (`weights/Spectra-AASIST/README.md`, the local copy of `lab260/Spectra-AASIST`):
- The card reports evaluation EERs on ASVspoof 2019 LA (0.159%), ASVspoof 2021 LA and DF, ASVspoof5, ADD2022, **In-the-Wild** and several ADD tracks.
- It names **no training set**.
- Its In-the-Wild entry means our earlier "0 of 2,000 In-the-Wild false alarms" for M3 comes from a set its authors evaluated on and may have tuned against. That is another reason the probes above, not M3's In-the-Wild number, carry the verdict.

**New finding for the fusion lane: additive noise is the shipped rule's weak spot.**
- At 20 dB SNR white noise, handcrafted v5 loses almost all discrimination (AUC 0.998 → 0.640).
- M1b's minDCF rises from 0.056 to 0.271, and the shipped rule's from 0.012 to 0.269.
- M3 stays near-perfect (0.116), but its step never fires under noise, so it cannot help there.
- Nothing here changes the shipped rule. It is a README "what did not work" entry and a pointer for any later robustness work. Noise-augmented training of the handcrafted features is the obvious next step, symmetric across classes.

## B. Is the 7.2 kHz wall a codec?

**Answer: no, not an MP3 or AAC round-trip at any tested setting. The pipeline's own Kaiser low-pass (`hearsay.handcrafted.band_limit`) remains the closest reproduction.** Source: `outputs/channel/codec_match.json`, `outputs/channel/codec_grid.csv`, `scripts/channel_codec.py`.

**Grid.**
- 100 inner-fold real clips (50 LJ, 50 LibriSpeech).
- Variants:
  - raw
  - Kaiser alone
  - MP3 at 24/32/48/64/96 kbps and AAC at 32/48/64 kbps, encoded at 16 and 44.1 kHz, each alone and followed by Kaiser (34 variants)
- Statistics are compared with all 1,671 test files as delivered:
  - band levels at 6.5–7.75 kHz relative to 1–3 kHz, and the 7.5-vs-6.5 kHz drop
  - roll-off slope over 6.5–8 kHz and 6–7 kHz flatness
  - hole and floor statistics
- Distance = mean |median difference| in test-IQR units.

| Variant | Distance | 7.5 vs 6.5 kHz drop (dB) | Deep-hole frac | Local-hole frac |
|---|---|---|---|---|
| **Test files (target)** | — | −44.4 | 0.400 | 0.003 |
| AAC 64k @16k + Kaiser (closest codec) | 1.045 | −47.5 | 0.331 | 0.011 |
| **Kaiser alone** | **1.081** | −47.5 | 0.348 | 0.012 |
| AAC 48k @16k + Kaiser | 1.085 | −47.8 | 0.319 | 0.013 |
| MP3 96k @16k | 1.947 | −66.1 | 0.345 | 0.011 |
| MP3 32k @16k | 4.142 | −65.7 | 0.397 | 0.094 |
| Raw | 2.639 | −1.8 | 0.348 | 0.012 |

**Pre-declared rule (D5):** a codec matches if its distance is at least 20% below Kaiser's and it is closer on both hole statistics.
- The best codec is 3% better on distance and worse on deep holes, so there is no match.
- MP3 encoded at 16 kHz makes the wall far too steep (LAME's own low-pass).
- Encodes at 44.1 kHz leave no wall until our Kaiser adds one.
- No variant reaches the test files' near-zero local-hole rate.

**Reading.** The test set's wall is a smooth low-pass, like a resampler's anti-alias filter, not the signature of these codecs. The band match already in the pipeline is the right treatment, and a symmetric codec round-trip of the training data (C) has no measured target. B also exposed the stopband artifact in A's first run (see A).

## C and D

- **C (symmetric codec refit of M1b): not run.** B found no codec match, and the frozen P1 runs C only on a match. No refit M1b column comes from this rung. The laundering and re-extraction path was prepared and smoke-tested (aligned, length-preserving round-trips mapped back to the original fold paths) and then removed unused.
- **D (a wild bona fide corpus): not attempted.** Dropped for the time box at the 12:16 P1 revision: a download, extraction and refit would not fit before 16:00.
