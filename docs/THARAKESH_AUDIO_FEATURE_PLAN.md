# Modern synthetic speech: acoustic feature investigation

**Owner:** Tharakesh Suresh  
**Purpose:** A working guide for the hackathon acoustic-feature task.  
**Status:** Repository-grounded task guide, reviewed against `main` on Sep 26, 2026. This document records a proposed experiment plan; it does not claim new feature results.

## The assignment in plain language

The existing 75-feature spectral/prosody detector measures MFCCs, spectral contrast/flux/flatness, F0 variability, energy, pauses, and zero-crossing rate. Its v3 model misses `grad_tts`, `pro_diff`, and much of `elevenlabs`. My job is to find **additional measurable acoustic features** that help distinguish these generators from real speech after the project's normal preprocessing. A promising feature must also avoid flagging genuine recordings from many different speakers.

This is a short feature-discovery exercise, not a replacement for the team's entire detector. Each experiment should have a hypothesis, an implementation, held-out results broken down by generator, a real-audio false-positive check, and a keep/drop decision. Budget roughly 30 minutes for each family.

## Read the project before changing code

1. [`docs/STATUS.md`](STATUS.md): team snapshot, fold convention, data, scoring, and deadlines. It was written around 01:15 and can lag later commits.
2. [`docs/reports/2026-09-26_cpu-detectors.md`](reports/2026-09-26_cpu-detectors.md): actual detector measurements, data traps, per-generator numbers, and score export.
3. [`src/hearsay/handcrafted.py`](../src/hearsay/handcrafted.py): current 75 features and the `band_limit` → `prepare_segment` → RMS-normalize preprocessing path. Its `features(x, crop_s, seed, crop_mode="segment", band_match=True)` returns named finite floats.
4. [`CLAUDE.md`](../CLAUDE.md), [`src/hearsay/detectors/base.py`](../src/hearsay/detectors/base.py), [`scripts/extract_handcrafted.py`](../scripts/extract_handcrafted.py), [`scripts/train_handcrafted.py`](../scripts/train_handcrafted.py), and [`src/hearsay/metrics.py`](../src/hearsay/metrics.py): integration contract, extraction, grouped validation, and cost metric. Check current code again before implementing; teammates are actively changing this repo.

There was no `AGENTS.md` in the `main` snapshot reviewed for this guide.

## What the existing results actually say

The v3 handcrafted model was chosen by **inner-fold CV** and scored **0.254 normalized minDCF, 4.3% EER, and 0.993 AUC on the outer holdout**. That holdout contains `playht`, `wavegrad2`, and held-out bona fide groups. Its strong aggregate holdout result does **not** mean it catches the three target generators: per-generator inner out-of-fold minDCF was `grad_tts` **1.00**, `pro_diff` **1.00**, and `elevenlabs` **0.80**. These figures compare each generator to the relevant bona fide pool, as described in the report. LJ and LibriSpeech bona fide holdout source minDCF were 0.10 and 0.32 respectively. Those are source-level comparisons, not directly measured per-clip false-positive rates.

The report found that pause count/fraction, voiced fraction, loudness dynamics, and zero-crossing rate were near chance as individual features. F0 variability, spectral contrast/flux, some MFCCs, and flatness carried more signal. Avoid repackaging an existing feature as a new technique; rhythm and breathing need genuinely different measurements and evidence.

## Non-negotiable constraints

| Constraint | What to do |
| --- | --- |
| Test bandwidth has a ~7.2 kHz wall | `hearsay.handcrafted.band_limit` uses a 71-tap Kaiser low-pass (7,250 Hz cutoff) **before** trimming/cropping. Use it on train and test, then `prepare_segment` and RMS normalization as appropriate. The loader supplies 16 kHz mono. Avoid relying on the 7–8 kHz band. |
| Fixed train/held-out split | `splits/nsa_folds.csv`: inner folds `0`–`4` for out-of-fold training/selection; outer `holdout` = `playht`, `wavegrad2`, and 26 real-speech groups. Fit scalers, imputation, selection, and calibration inside training folds only. Never fit on the holdout. |
| Modern generators | Report `grad_tts`, `pro_diff`, and `elevenlabs` separately using their **labeled inner-fold rows**. The NSA test labels are unknown; do not claim per-generator blind-test results. |
| Real speaker diversity | Report LJ Speech and LibriSpeech bona fide groups separately, including false alarms at a threshold selected without the outer holdout. LibriSpeech contains many speakers **across clips**; it is not a substitute for a labeled clip with multiple speakers within it. Test those separately if available. |
| False alarms cost four times misses | `hearsay.metrics` uses `C_FA=4`, `C_miss=1`, `π_synth=0.3`: normalized minDCF = `9.33 × P_FA + P_miss` (rounded coefficient). MinDCF chooses an operating point; report EER too. The sponsor's bundled code has the opposite score-direction convention, an unresolved team issue. |
| Score polarity and schema | `outputs/detector_scores/<name>.csv` has `path,fold,split,score,logit`; `split` is `inner_oof`, `holdout`, or `test`. `score` rises toward synthetic and is in [0,1]; `logit` is its log-odds after clipping. Final NSA TSV is a separate artifact with `filename<TAB>cm-score`, 1,671 rows in template order. |
| Ownership | Commit only my files; inspect `git status` and stage explicit paths. Do not touch the mains-hum, splice, container, or compression detectors. |
| Timing | Detector scores due **Sat Sep 26, 19:00 EDT**; fusion freezes **22:00 EDT**. Plan says final TSV by Sunday 05:00 and DM before 08:00. Confirm any changes with the team. |

## Experiments, in the requested order

The hypotheses below are possibilities, not established fingerprints. Modern generators may reproduce any of these properties, and recording conditions can also change them.

### 1. Phase and group delay

For a short-time Fourier transform, `X(t, f) = |X(t, f)| exp(j φ(t, f))`. Phase encodes timing relationships. Group delay is approximately `−∂φ/∂ω`; phase wrapping, weak bins, time alignment, and window boundaries make naive averages unreliable. Try robust statistics of phase derivatives or modified group delay over informative speech frames and frequencies **below the usable bandwidth**. Mask low-energy bins, aggregate by clip, and sanity-check robustness to small time shifts and preprocessing. Compare against the baseline on each generator and real multi-speaker audio. A magnitude-matched synthetic signal is not automatically phase-defective; this is an empirical test.

### 2. LFCC and CQCC

MFCCs summarize a mel-spaced log spectrum. LFCCs use a linearly spaced filterbank; CQCCs derive cepstral coefficients from a constant-Q representation. These provide different spectral resolutions and may reveal generator-specific structure absent from the existing MFCC summary. **LFCCs do not inherently examine low harmonics more finely than MFCCs**; linear spacing redistributes resolution. CQCC has finer frequency resolution at low frequencies, subject to its parameters.

Start with modest coefficient counts and clip-level summaries (for example, per-coefficient mean and standard deviation, optionally temporal deltas). Restrict analysis to the shared usable band. Fit any normalization only on training folds. Compare LFCC and CQCC separately against the existing 20 MFCC mean/std pairs and check whether either adds useful information, rather than treating a high-dimensional feature set as proof of improvement.

### 3. Cycle-to-cycle jitter and shimmer

Jitter summarizes neighboring **glottal period** differences, such as `mean(|T[i+1]−T[i]|) / mean(T)`. Shimmer similarly summarizes cycle amplitude differences, `mean(|A[i+1]−A[i]|) / mean(A)`. Calculate them only over reliably voiced, sufficiently clean stretches. Reject unvoiced frames and tracking failures; a framewise `1/F0` contour is not, by itself, an accurate measurement of individual glottal cycles. Pitch doubling/halving, overlapping speakers, compression, and noise can create false irregularity. If reliable cycles cannot be found in a three-second clip, mark features missing through the team's normal convention instead of inventing a value.

### 4. Rhythm / amplitude modulation

Compute a short-time energy or amplitude envelope after standard preprocessing, then examine its low-frequency modulation spectrum. Try energy in broad bands around phrase and syllable rates (for example, roughly 1–2, 2–4, and 4–8 Hz), variability, or modulation entropy. With clips near three seconds, frequency resolution is coarse and the content of the spoken phrase strongly affects these values. Test whether the feature survives differences in clip duration, loudness normalization, and speaker count.

### 5. Breathing and low-energy intervals

Look for plausible breath-like noisy intervals near phrase boundaries, distinguishing them from silence, room noise, fricatives, and recording artifacts. In three-second excerpts a real person may never inhale, so **absence of a breath is weak evidence**. A simple low-energy interval statistic is a quick exploratory feature, but it should be dropped if it mostly measures editing or the recording environment, or raises real-audio false positives.

### Optional: within-clip speaker consistency

If there is time, extract pretrained speaker embeddings from several speech-rich short windows and compare cosine similarities. This addresses the judges' “variety of techniques” item, but the median NSA clip is about 3.4 seconds and provides little speech per window. Phonetic content, silence, noise, and genuine speaker changes can all lower similarity. Treat it as an auxiliary analysis, not a primary fake score. `CLAUDE.md` assigns speaker drift to teammate C while the CPU report says it had not started as of 03:00; coordinate ownership before coding it. Check model availability and license before adding a dependency.

## Evaluation loop for each family

1. Record the hypothesis and exact feature definition, including preprocessing, frequency limits, window size, voiced-frame criteria, and missing-value handling.
2. Extract features with the project's existing pipeline. Check finite values, runtime, and a few manually inspected clips. Cache only if the existing workflow supports it.
3. Use `outputs/manifests/nsa_train_sample.csv` and `splits/nsa_folds.csv`. Follow the repo's out-of-fold workflow on inner folds 0–4. Fit transforms only on training data. Select models and ideas by inner CV, then read the outer holdout as a final check. `scripts/train_handcrafted.py` already implements logistic regression versus LightGBM selection and exports fusion scores for a compatible feature directory.
4. Compare to the current baseline **and** to a baseline-plus-feature model under the same split. A single feature's separability may not imply an ensemble gain.
5. Record sample counts and metrics for LJ and LibriSpeech bona fide, `grad_tts`, `pro_diff`, `elevenlabs`, and outer-holdout `playht`/`wavegrad2`. Use `hearsay.metrics.report(y, scores)` and the report's per-generator convention. Include `P_FA` at any training-selected threshold, minDCF, EER, and AUC where available. Note uncertainty for small groups.
6. Keep a feature only if its useful gain survives the real-audio cost check. If it fails, document that outcome and move on. Do not keep tuning on the held-out fold until it improves.
7. Export `outputs/detector_scores/<name>.csv` for all inner, holdout, and 1,671 test rows with **higher-is-fake** polarity. Preserve `path`, `fold`, and `split`. Inner scores must be out-of-fold; holdout and test use the final inner-trained model. Check missing/error handling against the existing exporter. The fusion export is distinct from the final NSA TSV.

### Compact results template

| Experiment | grad_tts | pro_diff | elevenlabs | LJ P_FA | LibriSpeech P_FA | Overall minDCF vs baseline | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline v3 | 1.00 | 1.00 | 0.80 | TBD | TBD | 0.254 (outer holdout) | — |
| Phase | TBD | TBD | TBD | TBD | TBD | TBD | Keep/drop |
| LFCC | TBD | TBD | TBD | TBD | TBD | TBD | Keep/drop |
| CQCC | TBD | TBD | TBD | TBD | TBD | TBD | Keep/drop |
| Jitter/shimmer | TBD | TBD | TBD | TBD | TBD | TBD | Keep/drop |
| Rhythm | TBD | TBD | TBD | TBD | TBD | TBD | Keep/drop |
| Breathing | TBD | TBD | TBD | TBD | TBD | TBD | Keep/drop |

Define the generator columns' metric (for example, per-generator minDCF against the shared bona fide pool or recall at a fixed training-selected threshold) when filling in the table. Do not compare percentages with different denominators as if they were the same metric. The baseline report supplies 1.00, 1.00, and 0.80 inner-OOF minDCF respectively; these are not new results from this guide.

## Delivery checklist

- [x] Read the three required repository sources and update this guide with exact code paths, preprocessing, split, baseline, and score schema.
- [ ] Finish time-boxed experiments; retain actual per-generator and real-audio evidence.
- [ ] Save correctly oriented scores in the team's shared format before the score deadline.
- [ ] Inspect `git status`; stage and commit **only** my guide, feature code, results, and score files that the team expects.
- [ ] Tell the team what was tested, which features helped, where they failed, and the location of the score output before the combined-model lock.

## Core mental model

The useful question is whether an acoustic statistic differs between real and modern synthetic speech **after matched preprocessing**, while keeping the costly false alarms on genuine speech low. No phase, cepstral, timing, or breath feature is inherently a reliable proof of AI generation. The held-out results and the team's weighted objective decide whether it belongs in the combined detector.
