# Handcrafted v4: feature discovery for the three blind generators

Sat Sep 26, 2026, 04:00–. Brief: `docs/handoffs/2026-09-26_handcrafted-v4-brief.md`. Baseline: v3 (`docs/reports/2026-09-26_cpu-detectors.md`), inner OOF minDCF 0.614, holdout 0.254, and three generators it cannot see at all: `grad_tts` 1.00, `pro_diff` 1.00, `elevenlabs` 0.80. Metric: normalized minDCF, `9.33 · P_FA + P_miss`. All gating on inner rows only; the holdout is read once per trained variant.

## Summary

- **v4 wins on inner out-of-fold, the selection rule:** minDCF 0.614 → **0.434**, LibriSpeech real (the 4×-cost side) 0.77 → **0.54**, pro_diff 1.00 → **0.32**; holdout 0.254 → 0.170 (read once; inside the noise band). Export: `outputs/detector_scores/handcrafted_v4.csv`.
- **But the first fusion run found a test-domain shift:** under a Platt map at the 0.3 prior, v4 alone calls 56% of the test set synthetic (v3: 49%; deep detectors: 27%). The shift is the spectral envelope of the test recordings (darker, codec-like), spread over nearly every column; no family and no column subset removes it. **v5b**, the same columns trained with augmented fit-only twins (codec round-trips, random tilt + low-pass), is the recommended fusion column: clean holdout 0.137, calibrated test share 51%, holdout-real false alarms 6.3% (v4: 9.1%), In-the-Wild real 0.40%, at an inner cost of 0.418 → 0.469. Export `handcrafted_v5.csv`; bundle pinned at `models/hc_selected`.
- **Still blind:** grad_tts (1.00 in every variant; its phase cue is shown by no other generator, so it cannot be learned under generator-held-out folds) and ElevenLabs (0.80 → 0.92).
- **In-the-Wild:** P_FA **0.65%** at the inner-OOF threshold on 2,000 real clips (safe on the expensive side), but P_miss 97% and minDCF 1.0 (does not catch that domain's fakes); AUC 0.64 → 0.76 vs v3.
- **Families:** all six implemented and gated; cqcc, lfcc and phase carry the gain; jitter, modulation and breath are marginal. No passing column is a codec cue. Dropping corpus-cue columns hurt every variant.

## Method

Six feature families were added to `hearsay.hc_v4` behind `features(..., families=(...))`, each computed from what `features()` already has (complex and power STFT at 512/160, the voiced mask, the loudness envelope, the YIN track) on the shared preprocessing (band match at 7.25 kHz, trim, test-length crops, RMS normalization). Column names carry a family prefix; the trainer records the families in the bundle and the detector recomputes them at inference. Extraction with all six costs about 0.05 s per clip on top of the v3 block (0.14 → 0.20 worker-seconds per clip; the gate sample of 2,000 clips ran in 80–90 s on 4 workers).

**Gate.** 200 inner-fold clips per generator and per real source (2,000 clips, `outputs/manifests/hc_gate_sample.csv`; never the holdout). Per new column: AUC of each blind generator against all real speech, AUC of each real source against all synthetic speech, and the corpus check (LJ real vs LibriSpeech real, want 0.5). Keep rule per column: |AUC − 0.5| ≥ 0.15 on grad_tts, pro_diff or elevenlabs, corpus deviation ≤ 0.15, and the two real sources within 0.10 of each other. A family survives if any column does. Full tables: `outputs/handcrafted/hc_gate_all_columns.csv`.

**Codec check.** For every column that passed on elevenlabs (an MP3-sourced generator), 80–100 real clips were laundered through MP3 128 kbps at 44.1 kHz (`hearsay.compression.launder`) and the feature recomputed. No passing column of any family moved by more than 0.06 clean standard deviations (threshold 0.5), so nothing below is a codec cue.

## Gate results per family

| Family | Columns | Pass | Best blind separations (AUC; < 0.5 means lower on synthetic) | Verdict |
|---|---|---|---|---|
| **phase** (group delay + peak phase-vs-magnitude coherence) | 12 | 7 | `pc_frac_incoherent` / `pc_err_median`: grad_tts **0.80**, corpus 0.57, real LJ 0.42 vs Libri 0.49. `gd_std_p10`: pro_diff **0.21**, grad_tts 0.66, corpus 0.45. `gd_std_mean`: grad_tts 0.72, elevenlabs 0.28. | **Keep.** The first family that sees grad_tts on a corpus-neutral cue. |
| **cqcc** (constant-Q cepstra, 65 Hz–7 kHz) | 72 | 16 | `cqcc2_dstd`: pro_diff **0.94**, elevenlabs **0.82**, grad_tts 0.72, corpus 0.61. `cqcc1_dstd`: pro_diff 0.86, elevenlabs 0.87, grad_tts 0.67, corpus 0.53. `cqcc9_mean`: pro_diff 0.87. | **Keep.** Strongest family; frame-to-frame change of the low-order coefficients carries it. Fifteen of its columns are corpus cues (see blocklist). |
| **lfcc** (linear-frequency cepstra, 0–7 kHz) | 60 | 16 | `lfcc17_std` / `lfcc18_std`: pro_diff **0.13**, elevenlabs 0.75, grad_tts 0.33, corpus 0.50–0.52. `lfcc0_dstd`: elevenlabs 0.85, pro_diff 0.71, grad_tts 0.63. | **Keep.** Upper-coefficient variability, the same cue as the v3 upper-MFCC stds, sharper at linear spacing. Thirty-two of its columns are corpus cues. |
| **jitter** (cycle-level jitter and shimmer) | 5 | 2 | `jit_rap` / `jit_local`: grad_tts 0.63, pro_diff 0.34, corpus 0.50. | Marginal keep (0.16). Cheap. |
| **modulation** (loudness-envelope modulation spectrum) | 6 | 3 | `mod_8_16`: elevenlabs 0.66; `mod_1_2`: pro_diff 0.35; `mod_peak_hz`: pro_diff 0.65. | Marginal keep (0.15–0.16), as the brief predicted for 3 s clips. Cheap. |
| **breath** (between-speech frames) | 4 | 1 | `breath_flatness`: pro_diff 0.28. | Marginal keep (0.22 on one generator). Cheap. |

Nothing was killed outright by the rule; the two weakest families (modulation, breath) stay only because they cost nothing and their one passing column each is corpus-neutral.

**A note on the phase family's first version.** The first instantaneous-frequency measure (per-bin temporal std of the phase advance) read the same 1.4–1.5 rad for real and synthetic speech alike, the level of random phase, because a fixed frequency bin sees every pitch movement as phase noise. Replacing it with a peak-tracking measure (the frequency implied by the phase advance against the frequency implied by the interpolated magnitude peak, per harmonic per frame) removed the pitch-movement dependence and produced the grad_tts separation above. Group-delay spread likewise had to be restricted to bins within 30 dB of each frame's peak, or the empty bins between harmonics dominated it.

## Corpus cues (what the model must not lean on)

Across all 234 columns (159 new, 75 v3), 83 separate LJ real from LibriSpeech real by more than 0.25 AUC or make the two real sources disagree by more than 0.25 on how synthetic they look: 49 new columns and 34 v3 columns. The worst v3 offenders are the ones that make LibriSpeech real look synthetic and therefore drive the 4×-cost false alarms: `contrast0_mean` (LJ vs Libri 0.95; LibriSpeech real vs all synthetic 0.64), `mfcc2_std` (0.94; 0.76), `hf_ratio_7k_std` (0.89; 0.68), and the MFCC means 9, 11, 13, 14 (0.07–0.11; LJ real looks synthetic instead). Lists: `outputs/handcrafted/hc_gate_all_columns.csv`. The trainer's `--drop-columns` takes them out at training time, so variants can be compared without re-extracting.

## Training variants

Full run: all six families extracted for the 20,000-clip training sample (`--crop test --seed 0`, the deep detector's crops) and the 1,671 test files, 234 columns, 0 failures, 993 s at 4 workers. Four variants trained with `scripts/train_handcrafted.py --drop-columns`, each selecting logistic vs LightGBM by inner OOF (LightGBM won every time), holdout read once per variant. **Selection is by inner OOF only**: overall minDCF, the blind generators, and LibriSpeech real. v3 is the reference row.

| Variant | Columns | Inner OOF minDCF | EER | grad_tts | pro_diff | elevenlabs | LibriSpeech real (inner) | Holdout minDCF | Holdout EER | Holdout LibriSpeech | Test share > 0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3 (reference) | 75 | 0.614 | 17.6% | 1.00 | 1.00 | 0.80 | 0.77 | 0.254 | 4.3% | 0.32 | 42% |
| **v4a: all columns** | 234 | **0.434** | 14.1% | 1.00 | **0.32** | 0.92 | **0.54** | **0.170** | 3.6% | 0.22 | 41% |
| v4b: minus the 49 new corpus-cue columns | 185 | 0.467 | | 1.00 | 0.45 | 0.86 | 0.59 | 0.186 | 3.2% | 0.26 | 41% |
| v4c: minus all 83 corpus-cue columns | 151 | 0.528 | | 1.00 | 0.67 | 0.95 | 0.61 | 0.275 | 5.2% | 0.34 | 51% |
| v3c: v3 minus its 34 corpus-cue columns | 41 | 0.848 | | 1.00 | 1.00 | 0.88 | 0.93 | 0.640 | 13.5% | 0.71 | 60% |

**Winner: v4a, all columns** (`models/hc_lgbm_20260926-0438`, pinned as `models/hc_selected`; export `outputs/detector_scores/handcrafted_v4.csv`). Per generator, inner OOF: diffgan_tts 0.02, unit_speech 0.06, your_tts 0.09, pro_diff 0.32, openvoicev2 0.41, xtts_v2 0.46, elevenlabs 0.92, grad_tts 1.00; holdout playht 0.08, wavegrad2 0.24.

**Reading.**
- The inner OOF gain is real and broad: overall 0.614 → 0.434, and LibriSpeech real, the 4×-cost side, 0.77 → 0.54 without any column pruning. The holdout moves 0.254 → 0.170, inside the ±0.15 band the consult calls noise, so it is consistent with the inner gain but not evidence on its own.
- **pro_diff is now visible** (1.00 → 0.32), the CQCC frame-to-frame cues from the gate translating into out-of-fold detection when pro_diff is held out.
- **grad_tts is still invisible** (1.00 in every variant). The phase family separates it from real speech at AUC 0.80 at the single-feature level, but when grad_tts is the held-out generator no other generator teaches the model that cue, so the fold-0 model never uses it. That is the honest limit of "held out by generator": a cue only one generator shows cannot be learned for that generator.
- **elevenlabs got worse** (0.80 → 0.92), and it is the one commercial generator in the inner set, so fusion should know the handcrafted column is now weaker there. Fold 4 holds out ElevenLabs alone, so its OOF model is trained on the other seven generators only; the likely cause is that the new columns fit those seven more tightly and the LFCC upper-coefficient cue that separated ElevenLabs at the gate did not carry through. The variant data does not cleanly confirm the "more columns" part (185 columns gave 0.86, 151 gave 0.95), so treat the mechanism as partly unexplained; the regression itself is not in doubt.
- **Dropping corpus-cue columns hurt every time**, including on LibriSpeech real (v4b 0.59, v4c 0.61 vs v4a 0.54), and stripping v3's own cue columns destroys v3 (0.614 → 0.848). Columns that separate LJ from LibriSpeech also separate real from fake in ways that generalize across held-out generators; the corpus check is a warning to read per-source readouts, not a pruning rule. The blocklists stay in `outputs/handcrafted/hc_gate_all_columns.csv` for the record.
- Test-set share above 0.5 is unchanged at 41% (v3 42%), against a 30% prior.

**Cost, measured** (worker-seconds per clip from the extraction sidecars): v3's 75 features 0.14 s (20,000 clips, 450 s on 6 workers); all 234 with the six families 0.20 s (993 s on 4 workers); so the families add about 0.06 s per clip, the CQT and the second STFT for group delay being most of it. The 0.63 s figure in the earlier handoff was a single-process measurement that included the numba warm-up. Inference through the contract detector recomputes exactly these families (`bundle["families"]`).

## In-the-Wild readout (eval only)

3,000 clips (2,000 bona fide, 1,000 spoof) from `outputs/manifests/itw_stress.csv`, features extracted with `--crop test --seed 200` (the main chat's `itw_stress_v3` crops), scored with the pinned bundle by `scripts/eval_bundle.py`; the threshold is the one that minimizes the bundle's own inner-OOF DCF (logit 1.54 for v4a). Never fitted on. v3 scored from the same feature file for reference.

| | minDCF | EER | AUC | P_FA at the inner threshold (2,000 real) | P_miss at the inner threshold | Mean score, real / fake |
|---|---|---|---|---|---|---|
| **v4a** | 1.00 | 30.4% | 0.762 | **0.65%** | 97.2% | 0.035 / 0.106 |
| v3 | 1.00 | 41.5% | 0.639 | 0.55% | 98.1% | |

**Reading.** The handcrafted detector does not transfer to In-the-Wild fakes at its NSA-trained threshold: it calls 97% of them real, and even the best threshold gives minDCF 1.0, no better than "always real" under the 9.33× false-alarm weight. What fails there is the threshold, not the ranking: with AUC 0.76 (v3 0.64; EER 41% → 30%) the In-the-Wild fakes sit just below the inner-fold cut (mean score 0.11 against 0.04 for real), so the logit column still orders them. Fusion refits the scale of that column, so it can use the ordering even though the fixed threshold cannot. What it does not do is false-alarm: 0.65% of the 2,000 real clips cross the threshold, against 1.2–5.7% the main chat reports for the deep detectors. For fusion that is the useful fact: the column is safe on the 4×-cost side in a domain it was never trained for, and it contributes nothing to catching In-the-Wild-style fakes. Its value stays in the NSA domain (holdout 0.17) and on generators like pro_diff. The NSA-holdout-to-ITW gap (0.17 → 1.0) is the same domain-shift headline the main chat found for M1 (0.08–0.15 → 0.37–0.40), larger here because 234 handcrafted numbers are a narrower description than XLS-R's.

## Test-domain shift (found by the first fusion run)

The main chat's fusion run calibrates each column with a Platt map fit on inner OOF logits, shifted to the 0.3 prior, and thresholds at 0.5. Under that map the handcrafted v4 column alone calls **56%** of the NSA test set synthetic (deep detectors: 26–27%; prior 30%). Reproducing the map here gives 56.1% for v4 and **48.8% for v3**, so the inflation predates v4. Inner and holdout numbers show nothing of it: at the same map, 9% of holdout real clips and 99% of holdout fakes score above 0.5.

**Where the shift is** (`outputs/handcrafted/v4_domain_shift.csv`: per column, KS statistic of the 1,671 test files against inner train-real and against inner train-fake, the shift of the test median in train-real SD, and whether the test median lies beyond both classes). Flag = KS > 0.3 against *both* classes with the test median beyond both, i.e. a domain shift rather than a class mix.

| Family | Columns | Flagged | Train-vs-test AUC, family alone |
|---|---|---|---|
| v3 (75 original) | 75 | 37 | **0.969** |
| lfcc | 60 | 14 | 0.954 |
| cqcc | 72 | 9 | 0.947 |
| breath | 4 | 4 | 0.837 |
| phase | 12 | 0 | 0.796 |
| jitter | 5 | 0 | 0.622 |
| modulation | 6 | 0 | 0.619 |
| all 234 | 234 | 64 | 0.991 |

What moves is the **spectral envelope**, not the dynamics: cepstral means (`mfcc4/5/6/10/11/13/14/16_mean`, `cqcc1/2/3_mean`, `lfcc12/13_mean`; 1.4–1.9 SD), and `centroid_mean`, `rolloff85/95_mean`, `zcr_mean` moving *down* (the test set is darker below 7 kHz, the codec-like signature the compression lens already read on 72% of it). **84% of the flagged columns move in the fake direction**, so the shift produces false alarms, the 4×-cost error. The variability columns that carry most of the class signal (stds, delta-stds, flux, contrast, pitch spread) are far less shifted, and the phase, jitter and modulation families are close to domain-neutral.

**Leave-one-family-out** (same calibrated share, computed identically for every export):

| Variant | Inner OOF | LibriSpeech real | pro_diff | elevenlabs | Holdout | Calibrated test share | Raw share > 0.5 |
|---|---|---|---|---|---|---|---|
| v3 | 0.614 | 0.77 | 1.00 | 0.80 | 0.254 | 48.8% | 42% |
| v4a, all | 0.434 | 0.54 | 0.32 | 0.92 | 0.170 | 56.1% | 41% |
| v4 − lfcc | 0.561 | 0.72 | 0.77 | 0.68 | 0.343 | 56.4% | 51% |
| v4 − phase | 0.435 | 0.53 | 0.30 | 0.91 | 0.175 | 55.3% | 39% |
| v4 − cqcc | 0.468 | 0.55 | 0.41 | 0.94 | 0.176 | 53.8% | 34% |
| v4 − modulation | 0.428 | 0.52 | 0.29 | 0.91 | 0.173 | 57.5% | 41% |
| v4 − breath | 0.435 | 0.55 | 0.33 | 0.93 | 0.170 | 57.3% | 42% |
| v4 − jitter | 0.436 | 0.54 | 0.32 | 0.92 | 0.163 | 55.6% | 40% |

No single family is responsible; the shift is spread over every envelope-carrying column, v3's included. Two remedies were tried next, results below: **envelope-blind variants** (drop all cepstral, centroid, roll-off and ZCR means; drop the 64 flagged columns) and **v5, training-side augmentation** (a random subset of training rows re-encoded through MP3/AAC and/or given a random spectral tilt of ±4 dB/kHz plus a 4.5–7 kHz low-pass before feature extraction, test rows untouched), so that envelope means stop being trustworthy class cues and the model leans on dynamics.

**Envelope-blind variants: worse on every count.**

| Variant | Columns | Inner OOF | LibriSpeech real | pro_diff | Holdout | Calibrated test share | Train-vs-test AUC of the kept columns |
|---|---|---|---|---|---|---|---|
| v4a, all | 234 | 0.434 | 0.54 | 0.32 | 0.170 | 56.1% | 0.991 |
| minus all cepstral / centroid / roll-off / ZCR means (78 columns) | 156 | 0.481 | 0.56 | 0.41 | 0.325 | **67.7%** | 0.984 |
| minus the 64 KS-flagged columns | 170 | 0.480 | 0.56 | 0.29 | 0.257 | 59.5% | 0.982 |
| minus both | 131 | 0.508 | 0.60 | 0.40 | 0.353 | 60.5% | 0.976 |

Removing the shifted columns does not remove the shift: the remaining columns still separate train from test at AUC 0.98, and the calibrated share goes *up*, so the variability columns are shifted in the fake direction too, just less visibly per column. Column pruning is not the remedy. The domain difference is a property of the whole feature description of the test recordings (darker, codec-processed), not of a few columns.

**v5: training-side augmentation.** `scripts/extract_handcrafted.py --launder-frac 0.35 --tilt-frac 0.35` re-extracted the training sample with 11,550 of 20,000 rows augmented (MP3/AAC round-trip at 24–128 kbps and/or a random ±4 dB/kHz tilt plus a 4.5–7 kHz low-pass, drawn per row from `default_rng(seed + row)`, recorded in `hc_augment`); test and In-the-Wild rows untouched. Two ways to use those rows:
- **v5a, replace:** train on the augmented sample as is (58% of rows altered). Its inner and holdout readouts mix clean and altered rows, so the table below re-reads its inner OOF on the 6,842 clean inner rows only, and reports the trainer's clean-only holdout.
- **v5b, extra rows** (`--extra-rows-from`): train on the clean v4 rows plus the 9,300 augmented twins of the inner rows as fit-only extras (each takes its clean twin's fold, so no fold leakage); every readout and the export stay on clean rows, directly comparable with v4.

| | Inner OOF (clean rows) | LibriSpeech real (inner) | pro_diff | elevenlabs | Holdout (clean) | Holdout LibriSpeech | Holdout real above 0.5, calibrated | Calibrated test share | ITW P_FA / P_miss at the inner threshold | ITW AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 0.614 | 0.77 | 1.00 | 0.80 | 0.254 | 0.32 | 9.5% | 48.8% | 0.55% / 98% | 0.64 |
| v4a | 0.418 | 0.51 | 0.30 | 0.88 | 0.170 | 0.22 | 9.1% | 56.1% | 0.65% / 97% | 0.76 |
| v5a, replace | 0.469 | 0.56 | 0.73 | 0.63 | 0.153 | 0.41* | 6.5% | 47.9% | 0.25% / 97% | 0.76 |
| **v5b, extra rows** | 0.469 | 0.59 | 0.63 | 0.67 | **0.137** | **0.16** | **6.3%** | 50.6% | **0.40% / 96%** | 0.77 |

*v5a's per-source holdout is on mixed rows. v4a's inner numbers here are on the same 6,842 clean inner rows as v5a's (its all-rows figure is 0.434).

**Reading.** Augmentation buys robustness on the false-alarm side at a modest inner cost. v5b keeps most of v4's inner gain (0.469 vs 0.418 on the same rows), improves the clean holdout (0.170 → 0.137, LibriSpeech 0.22 → 0.16), and lowers every false-alarm reading we have: holdout real at the calibrated threshold 9.1% → 6.3%, In-the-Wild real 0.65% → 0.40%, calibrated test share 56% → 51%. It trades pro_diff (0.30 → 0.63) for ElevenLabs (0.88 → 0.67). Neither v5 brings the test share to the 30% prior; the inflation is largely inherent to describing these test recordings with envelope statistics (v3 already sat at 49%), and augmentation removes only the part the model can learn to ignore. **Recommendation for fusion: v5b** (`outputs/detector_scores/handcrafted_v5.csv`, bundle `models/hc_lgbm_20260926-055451`, pinned as `models/hc_selected`), because a false alarm costs 9.3 misses in this metric and v5b is better on that side everywhere it can be measured; `handcrafted_v4.csv` stays available for the leave-one-out comparison, and a fusion-side re-centering of the logit column (an additive shift; the ranking is unaffected) remains the fallback for the residual offset.

## What worked / what had no effect (v4)

**Worked**
- CQCC frame-to-frame change (the `_dstd` columns) and, behind it, LFCC upper-coefficient variability: pro_diff went from invisible to 0.32 out of fold, and the overall inner OOF minDCF from 0.614 to 0.434.
- The corpus-neutral phase cues (peak phase-vs-magnitude coherence, group-delay spread over energetic bins) as *evidence*: they separate grad_tts from real speech at AUC 0.80 without any corpus confound, and they name a vocoder property a judge can read.
- Gating on a 2,000-clip inner sample with a per-column corpus check and a codec check: every family was measured in under 3 minutes, and the corpus check exposed 83 columns that would otherwise have been read as class cues.
- The In-the-Wild readout beside the holdout: it is what tells fusion the column is safe on real-world false alarms.

**No effect or hurt**
- Dropping corpus-cue columns: worse on every readout, including LibriSpeech real. The check is a warning to read per-source results, not a pruning rule.
- Modulation, breath and jitter: marginal at the gate (0.15–0.22 on one generator each) and not distinguishable inside the full model; kept because they cost nothing.
- Anything for grad_tts out of fold: a cue shown by one generator only cannot be learned for that generator under generator-held-out validation. It stays at 1.00.
- The first phase-jitter measure (fixed-bin instantaneous frequency): pitch movement, not synthesis.
- ElevenLabs: slightly worse (0.80 → 0.92); nothing in v4 targets it.

## State for the next chat

- **Tracked:** `src/hearsay/hc_v4.py`, the `families` plumbing in `handcrafted.py`, `extract_handcrafted.py`, `train_handcrafted.py` (`--drop-columns`, second-resolution stamps), `_learned.py` (`<prefix>_selected`), `scripts/eval_bundle.py`, `tests/test_hc_v4.py`, this report.
- **Not tracked (gitignored, regenerable):** `outputs/handcrafted/{nsa_test_v4,nsa_train_sample_v4,itw_stress_v4,hc_gate_*}.csv` (about 20 minutes at 4 workers), `models/hc_lgbm_20260926-0438` (the winner, pinned by `models/hc_selected`), `outputs/detector_scores/handcrafted_v4.csv` and the per-variant exports.
- **Fusion:** `handcrafted.csv` is still v3; the main chat decides among `handcrafted_v4.csv` (max inner gain, 56% calibrated test share) and `handcrafted_v5.csv` (v5b, recommended: fewer false alarms, 51%); `handcrafted_v5a_replace.csv` is the replace-style run for the record. All have the same 21,671 rows and format.
- **Open:** grad_tts and ElevenLabs remain the blind spots; speaker-embedding drift (rubric technique 6) is not started; the three losing variants' bundles were re-trained with second-resolution stamps only for their `meta.json` records.
