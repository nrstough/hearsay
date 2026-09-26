# Channel robustness on the frozen deep path (Sat Sep 26, 2026)

Run spec: `docs/specs/2026-09-26_channel-robustness.md`. Plan: `docs/reports/2026-09-26_channel-robustness-plan.md` (Codex plan review: `docs/reports/2026-09-26_channel-robustness-plan-review.md`). Handoff: `docs/handoffs/2026-09-26_channel-robustness-handoff.md`. Diagnostic only: this rung produces no TSV and changes no shipped rule.

## A. How wild is the test set? (λ̂)

**Answer: wild-like. λ̂ lies between 0.5 and 0.95 depending on the feature set, above the fusion consult's 0.32 crossover in every variant.** "Wild" here mostly means "has been through a lossy codec": one MP3 round-trip moves our own clean real clips from 3% to 91% "wild". Source: `outputs/channel/lambda.json`, `scripts/channel_lambda.py`.

**Method.**
- Every clip is prepared the same way (band match → silence trim → 8 s cap → RMS normalize), then described by label-free channel statistics:
  - band levels up to 7 kHz
  - spectral-hole and noise-floor statistics from `hearsay.compression.features`
  - noise-floor level and stationarity
  - a decay (reverberation) proxy
  - flatness, loudness spread and pause fraction
- Columns that read above 7 kHz are dropped: the test files pass two low-passes, everything else one.
- A logistic classifier separates holdout real speech (1,858 clips: LJ and LibriSpeech) from In-the-Wild real speech (2,000 clips, 54 speakers). Out-of-fold scores come from 5-fold CV grouped by speaker, with LJ as a single speaker.
- λ̂ is adjusted classify-and-count, (q − FPR)/(TPR − FPR). Its interval resamples test files and reference speakers. The mean posterior is reported too, but only as a description: it is biased toward 0.5 when the domains overlap (pinned by a test).

**Result.**

| Readout | Value |
|---|---|
| Domain classifier, grouped out-of-fold AUC | 0.994 (TPR 0.988, FPR 0.109) |
| **λ̂, full feature set** | **0.947** (95% CI 0.927–0.978) |
| Mean posterior P(wild) over the 1,671 test files (descriptive) | 0.894 |
| λ̂ on the 1,170 files the shipped rule calls real (lowest 70%) | 0.99 |
| λ̂ with LJ removed from the clean side | 0.96 |
| λ̂ without the dominant feature (`clip_floor_db`) | 0.51 (AUC 0.84) |
| λ̂ without all 15 floor and hole features | 0.80 (AUC 0.65) |
| Test files outside both domains (Mahalanobis p99) | 1.0% (1% expected) |

**Controls, never trained on (share read as "wild"):**

| Control | Share read as "wild" |
|---|---|
| Inner-fold real clips | 3% |
| VCTK clean read speech (unseen corpus) | 1% |
| DiffSSD spoof | 22% |
| **The same inner real clips after one MP3 64 kbps round-trip at 16 kHz** | **91%** |

The pre-declared identifiability rule passed:
- AUC ≥ 0.8
- VCTK and DiffSSD each read wild under 25%
- dropping LJ moves λ̂ by at most 0.15

**Reading.**
- The full classifier leans mostly on `clip_floor_db`, the spectrogram's 1st-percentile level. Lossy codecs push that floor to near-zero bins. Medians: clean −138 dB, In-the-Wild −155 dB, test −157 dB.
- On the other columns the test set sits between the two domains (per-feature positions −0.2 to 0.7).
- So the magnitude of λ̂ depends on how much weight goes to the codec floor. The direction does not.

What follows from this:
1. The clean NSA holdout overstates how the models behave on the test set, which supports choosing rules on In-the-Wild (fusion_v2 was chosen that way).
2. The test audio carries a codec history, which fits the 7.2 kHz wall and makes B and C more relevant than the consult expected.

**Limits.**
- In-the-Wild stands in for "wild"; telephony and physical replay are not represented.
- The test set is 30% synthetic, and DiffSSD spoof reads 22% wild, so spoof files pull λ̂ slightly toward "wild". The likely-real subset (0.99) says the conclusion does not come from the spoof share.

## E. Is M3 memorising, and what does its suppression step cost?

_In progress: holdout perturbation probe and MLAAD probe running (`scripts/m3_probes.py`)._

**Agreement (descriptive only, per the Codex plan review):** Spearman(M1b v3, M3) is 0.796 on holdout rows and 0.620 on test rows (gap 0.175, 95% CI 0.141–0.214). The two sets differ in class mix and generators, so this gap is not by itself evidence of memorisation. It says the two models disagree more on the test audio than on the holdout.

## B. Is the 7.2 kHz wall a codec?

_Pending._

## C and D

_C: pending B's result and Nathan's approval. D (a wild bona fide corpus): not attempted, time box (frozen P1, 12:25)._
