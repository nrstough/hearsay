# Channel robustness on the frozen deep path (Sat Sep 26, 2026)

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

_In progress: holdout perturbation probe and MLAAD probe running (`scripts/m3_probes.py`)._

**Agreement (descriptive only, per the Codex plan review):** Spearman(M1b v3, M3) is 0.796 on holdout rows and 0.620 on test rows (gap 0.175, 95% CI 0.141–0.214). The two sets differ in class mix and generators, so this gap is not by itself evidence of memorisation. It says the two models disagree more on the test audio than on the holdout.

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
