# Channel robustness on the frozen deep path (Sat Sep 26, 2026)

**Summary.**
- **λ̂** (how wild the test set is) = 0.51 (95% CI 0.12–0.64), and 0.36–0.57 across consistent variants. It is **indeterminate** against the 0.32 crossover. The fusion_v2 choice does not depend on it.
- **The 7.2 kHz wall** is a smooth low-pass, not an MP3 or AAC round-trip, so the codec refit (C) did not run.
- **M3's false-alarm suppression is kept** under the pre-declared rule:
  - its AUC and minDCF barely move under perturbation (but its rank order moves the most of the deep models)
  - it has the lowest miss rate on MLAAD
  - its step never raised the fused minDCF
- **New:** 20 dB additive noise breaks handcrafted v5 and turns fakes into misses. The shipped rule's holdout minDCF rises from 0.012 to 0.269.
- The first λ̂ (0.947) was wrong and is kept below as a documented mistake.

Run spec: `docs/specs/2026-09-26_channel-robustness.md`. Plan: `docs/reports/2026-09-26_channel-robustness-plan.md` (Codex plan review: `docs/reports/2026-09-26_channel-robustness-plan-review.md`). Handoff: `docs/handoffs/2026-09-26_channel-robustness-handoff.md`. Diagnostic only: this rung produces no TSV and changes no shipped rule.

## A. How wild is the test set? (λ̂)

**Answer: indeterminate. λ̂ = 0.51 (95% CI 0.12–0.64), an interval that contains the fusion consult's 0.32 crossover. Consistent variants range from 0.36 (duration-matched) to 0.57.** The test set sits between our clean holdout and In-the-Wild on about half the channel statistics and outside both on the rest. That is not strong enough to say which validation set represents it. For the shipped rule this does not matter: fusion_v2 beats fusion_v1 on both the holdout (0.0065 vs 0.014) and In-the-Wild (0.228 vs 0.260), so the choice is robust to λ. Source: `outputs/channel/lambda.json`, `scripts/channel_lambda.py`.

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
| Domain classifier, grouped out-of-fold AUC | 0.84 pooled; per fold 0.68 / 0.82 / 0.70 / 0.92 (the fifth fold holds the single LJ speaker, one class, no AUC) |
| **λ̂** (adjusted classify-and-count, TPR 0.84 / FPR 0.28) | **0.51** (95% CI 0.12–0.64) |
| Rate-consistent variants: q from the fold models / TPR and FPR in-sample | 0.48 / 0.57 |
| **Duration-matched** (reference clips cropped to test lengths) | **0.36** (AUC 0.81) |
| LJ removed from the clean side | 0.54 |
| λ̂ on the 1,170 files the shipped rule calls real | 0.58 |
| Mean posterior P(wild) (descriptive; biased toward 0.5) | 0.56 |
| Without all floor and hole features | 0.80, but the classifier is weak (per-fold AUC 0.32–0.74) |
| Without the top feature (`local_hole_runs`) | refused: pooled TPR − FPR is 0.13, but per-fold AUC stays 0.63–0.89. The refusal comes from the LJ fold (see below), not a collapse |
| Test files outside both domains (Mahalanobis p99) | 1.0% (1% expected) |

**Why the pooled rates are pessimistic.**
- LJ is one speaker, so grouped CV puts all 885 LJ clips in one fold. That fold's model has never seen LJ and calls much of it "wild", which inflates the pooled FPR (0.28) that the adjusted estimate divides by.
- Meanwhile the test posteriors come from the full fit, which has seen LJ.
- The two rate-consistent variants (0.48 and 0.57) bracket the point estimate, so it holds, but the pooled AUC and FPR understate the classifier.

**Controls, never trained on (share read as "wild"):**

| Control | Share |
|---|---|
| Inner-fold real clips | 10% |
| Inner real after one MP3 64 kbps round-trip | 9% |
| DiffSSD spoof | 19% |
| VCTK clean read speech | 24.7% |

The pre-declared identifiability rule passes, but only just: VCTK sits at 24.7% against a 25% limit.

**Per-feature positions** (0 = the clean median, 1 = the wild median; `lambda.json` `per_feature`):
- **Between the two domains (11 of 21):** band levels at 4–7 kHz (0.54–0.70), local-hole statistics (0.35–0.69), valley depth (0.57), pause fraction (0.65), noise-floor level (0.59), deep-hole runs (0.14).
- **Beyond the wild end:** noise-floor stationarity (1.08) and flatness (1.74).
- **Beyond the clean end:** loudness spread (−3.5), deep-hole fraction (−0.74), floor p10 (−0.70), floor p2 (−0.51), decay slope (−0.45), deep-hole flicker and persistence (−0.2), low-band floor (−0.14).
- So the test set is not a blend of the two reference domains on every axis. It is its own recording population, and that is why λ̂ carries a wide interval.

**What went wrong first, and why it matters for the README.** The first full run reported λ̂ = 0.947 with AUC 0.99 and read "wild" as "codec-processed", because one MP3 round-trip moved our clean clips to 91% "wild". That run was driven by `clip_floor_db`, a 1st percentile over the whole spectrogram, which lands in the 7–8 kHz stopband. Action B measured the test files at −118 dB as delivered and −157 dB after the pipeline's second low-pass. So the feature measured the double filtering (and In-the-Wild's own band limits), not the recording channel. Dropping it under the rule already declared for >7 kHz columns gave the numbers above, and the codec reading disappeared (MP3 control 91% → 9%). The 0.947 figure came from the run at ~12:36 and its JSON was overwritten by the corrected run; the retained `outputs/channel/lambda_run.log` holds an earlier, pre-review run of the same feature set (AUC 0.999, λ̂ 0.945). This is the same class of shortcut as the 7.2 kHz wall in `architecture.md` §11: a band no training clip has, read through a feature that did not look like a band feature.

**Limits.**
- In-the-Wild stands in for "wild"; telephony and physical replay are not represented.
- LJ is one speaker, which leaves one validation fold with a single class (see above).
- λ̂ is a weak, single-method estimate: the interval is wide, and the duration-matched variant (0.36) sits near the crossover.

## E. Is M3 memorising, and what does its suppression step cost?

**Verdict under the pre-declared rule (run spec D9, revised, computed by `verdict_from`): kept.** None of the three triggers fires, and both probes are complete (500 and 572 clips). What the verdict does and does not show is spelled out below. Source: `outputs/channel/m3_perturb.json`, `m3_mlaad.json`, `m3_agreement.json`, `m3_verdict.json`; `scripts/m3_probes.py`.

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

The M3 step fires on 4.0–5.2% of real clips and on no spoof clip under any perturbation. Under noise it fires on nothing, because noise pushes M3 out of its "strongly bona fide" region (logit < −3).

**Two stability readings, and they disagree.**
- **Margin:** M3's AUC and minDCF barely move (mean |ΔAUC| 0.0008). But its clean AUC is 1.000, so its ΔAUC has almost no room to fall, and trigger (a) can barely fire for M3.
- **Rank order:** Spearman between each detector's clean and perturbed scores. M3 has the lowest rank stability of the three deep models under every perturbation:

  | Perturbation | M3 | M1b | M5 |
  |---|---|---|---|
  | MP3 | 0.987 | 0.997 | 0.999 |
  | Noise | 0.875 | 0.913 | 0.939 |
  | Speed | 0.962 | 0.989 | 0.992 |
  | Shift | 0.992 | 0.999 | 0.999 |

- So M3 keeps its decisions under perturbation, but its fine ordering is the least stable. That is not the collapse a lookup would show, and it is not strong evidence of generalisation either.

**At each detector's fixed inner-OOF threshold** (`at_inner_threshold` in the JSON):
- Noise shows up as misses, not false alarms: M1b P_miss 0.07 → 0.88 and handcrafted 0.05 → 0.99, both at P_FA ≤ 0.4%. M3 P_miss 0.008 → 0.32.
- MP3, speed and shift move P_FA by at most 0.4 points for any detector.

**2. MLAAD probe** (unseen by M1b and handcrafted v5; seen by M5; unknown for M3, whose training data is undisclosed).
- **Cohort:** 572 MLAAD English spoof clips, 4 per model across 143 models, each cropped to a test length.
- **Readout:** miss rate at each detector's own inner-OOF brief-cost threshold.

| Detector | Miss rate on unseen MLAAD spoof |
|---|---|
| **M3** (MLAAD exposure unknown) | **14.9%** |
| M1b v3 | 22.7% |
| M5 (trained on MLAAD, in-sample) | 28.7% |
| Handcrafted v5 | 56.1% |

- M3 falls below −3 on 1.75% of these clips.
- The shipped rule's actual conjunction (M3 < −3 **and** base rank > 0.5) damps 1.2% (7 files).
- Those 7 files come from seven models, one clip each: MiniMax-Speech-2.8-Turbo, GPA-v1.0, dots.tts (two variants), DeepGram and MOSS-TTS (two sizes).

**3. Agreement (descriptive only, per the Codex plan review).** Spearman(M1b v3, M3) is 0.796 on holdout rows and 0.620 on test rows (gap 0.175, 95% CI 0.141–0.214). The two sets differ in class mix and generators, so the gap is not evidence of memorisation by itself. It says the two models disagree more on the test audio.

**What remains unknown.** M3's training data is still undisclosed; the model card does not settle it (see below). Its perfect clean holdout score (AUC 1.000, minDCF 0.000) is consistent with M3 having seen DiffSSD-like data, and its best MLAAD miss rate is consistent with having seen MLAAD. The probes show that the *suppression step* is safe on these cohorts: it fires only on real clips and never raised the fused minDCF. They do not show that M3 generalises.

**Model card check** (`weights/Spectra-AASIST/README.md`, the local copy of `lab260/Spectra-AASIST`):
- The card reports evaluation EERs on ASVspoof 2019 LA (0.159%), ASVspoof 2021 LA and DF, ASVspoof5, ADD2022, **In-the-Wild** and several ADD tracks.
- It names **no training set**.
- Its In-the-Wild entry means our earlier "0 of 2,000 In-the-Wild false alarms" for M3 comes from a set its authors evaluated on and may have tuned against. That is another reason the probes above, not M3's In-the-Wild number, carry the verdict.

**New finding for the fusion lane: additive noise is the shipped rule's weak spot.**
- At 20 dB SNR white noise, handcrafted v5 loses almost all discrimination (AUC 0.998 → 0.640), and noise turns fakes into misses: at the inner threshold, M1b misses 88% of spoof and handcrafted 99%, with no rise in false alarms.
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

**Sensitivity.** Without `clip_floor_db` (a whole-spectrogram percentile that reads the stopband, the column A had to drop), Kaiser scores 0.660 and the best codec (AAC 64k @16k + Kaiser) 0.647. That is 2%, still no match (`codec_match.json` `without_clip_floor_db`).

**The full grid** (34 variants, sorted by distance; test medians: drop −44.4 dB, roll-off −44.7 dB/kHz, deep-hole 0.400, local-hole 0.003, flatness 0.485):

| Variant | Distance | 7.5 vs 6.5 kHz drop (dB) | Roll-off (dB/kHz) | Deep-hole frac | Local-hole frac | 6–7 kHz flatness |
|---|---|---|---|---|---|---|
| aac-64k@16000+kaiser | 1.045 | -47.5 | -47.9 | 0.331 | 0.011 | 0.448 |
| kaiser | 1.081 | -47.5 | -47.9 | 0.348 | 0.012 | 0.444 |
| aac-48k@16000+kaiser | 1.085 | -47.8 | -47.9 | 0.319 | 0.013 | 0.442 |
| aac-32k@16000+kaiser | 1.123 | -48.0 | -48.0 | 0.301 | 0.015 | 0.444 |
| aac-48k@44100+kaiser | 1.309 | -53.4 | -55.3 | 0.334 | 0.009 | 0.480 |
| mp3-96k@44100+kaiser | 1.371 | -53.0 | -53.9 | 0.346 | 0.013 | 0.443 |
| aac-64k@44100+kaiser | 1.378 | -53.3 | -56.2 | 0.330 | 0.010 | 0.458 |
| aac-32k@44100+kaiser | 1.395 | -53.4 | -54.6 | 0.344 | 0.013 | 0.444 |
| mp3-64k@44100+kaiser | 1.576 | -53.0 | -54.0 | 0.349 | 0.021 | 0.425 |
| mp3-96k@16000 | 1.947 | -66.1 | -65.1 | 0.345 | 0.011 | 0.444 |
| aac-48k@44100 | 2.084 | -8.8 | -12.8 | 0.334 | 0.009 | 0.480 |
| aac-64k@44100 | 2.102 | -8.3 | -14.1 | 0.330 | 0.010 | 0.458 |
| aac-32k@44100 | 2.140 | -9.1 | -12.3 | 0.344 | 0.013 | 0.443 |
| mp3-96k@16000+kaiser | 2.147 | -67.6 | -67.2 | 0.345 | 0.011 | 0.444 |
| mp3-48k@44100+kaiser | 2.159 | -53.3 | -54.4 | 0.360 | 0.037 | 0.375 |
| mp3-64k@16000 | 2.182 | -66.2 | -65.4 | 0.340 | 0.020 | 0.412 |
| mp3-96k@44100 | 2.302 | -7.7 | -10.5 | 0.346 | 0.013 | 0.443 |
| mp3-64k@16000+kaiser | 2.386 | -67.9 | -67.5 | 0.340 | 0.020 | 0.412 |
| mp3-64k@44100 | 2.414 | -7.6 | -10.7 | 0.349 | 0.021 | 0.425 |
| aac-64k@16000 | 2.630 | -1.8 | -2.8 | 0.331 | 0.011 | 0.448 |
| mp3-48k@16000 | 2.633 | -64.4 | -64.2 | 0.344 | 0.037 | 0.365 |
| raw | 2.639 | -1.8 | -2.8 | 0.348 | 0.012 | 0.444 |
| aac-48k@16000 | 2.658 | -1.8 | -2.8 | 0.319 | 0.013 | 0.442 |
| aac-32k@16000 | 2.661 | -2.1 | -2.8 | 0.301 | 0.015 | 0.444 |
| mp3-48k@44100 | 2.752 | -7.7 | -10.8 | 0.360 | 0.037 | 0.375 |
| mp3-48k@16000+kaiser | 2.833 | -66.2 | -65.7 | 0.344 | 0.037 | 0.365 |
| mp3-32k@16000 | 4.142 | -65.7 | -64.6 | 0.397 | 0.094 | 0.219 |
| mp3-32k@16000+kaiser | 4.306 | -66.5 | -65.8 | 0.397 | 0.094 | 0.219 |
| mp3-32k@44100+kaiser | 4.688 | -52.4 | -53.9 | 0.471 | 0.135 | 0.180 |
| mp3-32k@44100 | 5.901 | -7.4 | -12.1 | 0.471 | 0.135 | 0.180 |
| mp3-24k@44100+kaiser | 9.162 | -10.9 | -7.0 | 0.504 | 0.215 | 0.062 |
| mp3-24k@44100 | 9.171 | -10.9 | -7.0 | 0.504 | 0.215 | 0.062 |
| mp3-24k@16000 | 10.621 | -0.6 | -0.5 | 0.567 | 0.293 | 0.401 |
| mp3-24k@16000+kaiser | 10.622 | -0.8 | -0.7 | 0.567 | 0.293 | 0.400 |

**Reading.** The test set's wall is a smooth low-pass, like a resampler's anti-alias filter, not the signature of these codecs. The band match already in the pipeline is the right treatment, and a symmetric codec round-trip of the training data (C) has no measured target. B also exposed the stopband artifact in A's first run (see A).

## C and D

- **C (symmetric codec refit of M1b): not run.** B found no codec match, and the frozen P1 runs C only on a match. No refit M1b column comes from this rung. The laundering and re-extraction path was prepared and smoke-tested (aligned, length-preserving round-trips mapped back to the original fold paths) and then removed unused.
- **D (a wild bona fide corpus): not attempted.** Dropped for the time box at the 12:16 P1 revision: a download, extraction and refit would not fit before 16:00.
