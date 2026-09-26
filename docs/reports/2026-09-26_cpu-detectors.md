# CPU detector track (D-track): engineered detectors and two data traps

Sat Sep 26, 2026, 01:10–03:00. CPU only, in parallel with the main chat's GPU work (M1 v2, TSVs). Handoff: `docs/handoffs/2026-09-26_cpu-detectors-handoff.md`. All numbers below are normalized minDCF with `C_FA = 4`, `π_synth = 0.3` (`hearsay.metrics`), on the shared fold file `splits/nsa_folds.csv` (outer holdout = generators `playht` + `wavegrad2` plus 26 bona fide groups; inner folds 0–4). Score direction: 1.0 = synthetic, never flipped.

## Summary

- **Five engineered detectors** build against the contract (`src/hearsay/detectors/`): `handcrafted` (spectral + prosody, learned), `compression` (codec-history traces, learned with laundering-equalized training), `container` (header facts + tag rules), `enf` (mains hum), `splice` (clicks, DC jumps). `import hearsay.detectors.engineered` registers all five. 57 new hermetic tests (toy model bundles, ffmpeg-generated files, synthetic hum and seams); the suite is at 209 passed, ruff clean.
- **Fusion exports** in the M4 format (`path, fold, split, score, logit`) are in `outputs/detector_scores/`: `handcrafted.csv` (the best version; `handcrafted_v1/v2/v3.csv` kept for the ablation), `compression.csv`, `container.csv`, `enf.csv`, `splice.csv`. Every row of the 20,000-clip training sample and the 1,671 test files is present; inner rows are out-of-fold, the scaler included. **Fuse `handcrafted.csv`.** `compression.csv` is exported but suspect (its section explains why); `container`, `enf` and `splice` are constant or two-valued on the test set and carry evidence and routing, not score.
- **Two traps in the data**, both measured, both with a fix:
  1. **The container is the label on the training data.** LibriSpeech is FLAC 16 kHz, LJ is 22.05 kHz WAV, DiffSSD is 22.05/24/44.1 kHz WAV or MP3, and PlayHT's MP3s carry the exact FFmpeg tag (`encoder=Lavf58.29.100`) that every one of the 1,671 test files carries. A learned metadata model would identify PlayHT by its tag and then call the whole test set PlayHT. The container detector is therefore rule-based and never learned; on this test set it returns a constant 0.5 and routing facts.
  2. **The NSA test set is low-passed at about 7.2 kHz.** 1,602 of 1,671 test files drop by a median 44 dB between the 6.5 kHz and 7.5 kHz bands. No training corpus does, not even the sponsor's own resampled LJ clips (`data/nsa/LJRealResampled`). Every feature above 7 kHz was a corpus fingerprint, and the v1/v2 handcrafted detector called about 90% of the test set synthetic. A 71-tap Kaiser low-pass (`hearsay.handcrafted.band_limit`) reproduces the roll-off within 3 dB and is now applied to every clip, train and test. **The deep detector sees the same mismatch; see Recommendations.**
- **Handcrafted detector, holdout minDCF:** v1 (first 4 s) 0.709 → v2 (test-length crops, same as the deep detector's `prepare_segment`) 0.661 → **v3 (v2 + band match) 0.254**, EER 4.3%, AUC 0.993; the share of test files scored above 0.5 went from 90% to 42% (the prior is 30%). Two effects are entangled and separated below: LightGBM generalizes to the held-out generators far better than logistic regression, and the band match is what made LightGBM win the inner CV and what moved the test-score distribution. LibriSpeech bona fide stays the weak spot; the LJ-voice diffusion generators `grad_tts` and `pro_diff` are invisible to these features at any version.

## What was delivered

| Piece | Where |
|---|---|
| Feature modules | `src/hearsay/handcrafted.py` (spectral + prosody, v3 prep), `src/hearsay/compression.py` (codec traces + `launder`) |
| Contract detectors | `src/hearsay/detectors/{handcrafted,compression,container,enf,splice}.py`, shared learned wrapper `_learned.py`, registrar `engineered.py` |
| LightGBM without lightgbm | `src/hearsay/trees.py`: numpy evaluator + path attribution over the dumped booster (exact to 1e-9 against lightgbm), because lightgbm and torch each bundle a `libomp.dylib` and loading both in one process segfaults or hangs on macOS. Bundles store `trees_dump`; nothing outside the trainer imports lightgbm (`tests/test_trees.py` pins it). |
| Extraction | `scripts/extract_handcrafted.py`, `scripts/extract_compression.py` (crop lengths drawn exactly as `extract_embeddings.py --segment --crop test --seed 0`; verified against the main chat's shard 0) |
| Training + export | `scripts/train_handcrafted.py` (any feature dir; writes the M4 export and `models/<prefix>_<kind>_<stamp>/`) |
| Rule-based export | `scripts/score_detector.py` (runs a registered detector over the fold file + test manifest) |
| Inventory | `scripts/inventory_container.py` → `outputs/inventory/container_{nsa_test,nsa_train_sample}.csv`; `outputs/inventory/nsa_test_highband.csv` (per-file 6.5–8 kHz band levels) |
| Tests | `tests/test_{handcrafted,container,compression,enf,splice}_detector.py`, `tests/test_engineered_registry.py` (all hermetic: toy bundles, ffmpeg-generated files) |
| Models (gitignored, not archived) | `models/hc_logreg_*`, `models/cmp_*` with `meta.json` and `scores.csv` |

## Trap 1: the container is the label

`scripts/inventory_container.py` on all 1,671 test files and 300 clips per generator of the training sample (ffprobe; embedded fields only, never filesystem times or filenames).

**Test set: one format.** wav / pcm_s16le / 16,000 Hz / mono / 16-bit / one tag, `encoder=Lavf58.29.100`, for every file. The only per-file variation is the byte-rate ffprobe derives from size and duration.

**Training sample: format = source.**

| Source / generator | Container | Rate | Tag |
|---|---|---|---|
| LibriSpeech (real) | FLAC | 16 kHz | none |
| LJ Speech (real) | WAV PCM16 | 22.05 kHz | none |
| diffgan_tts, grad_tts, openvoicev2, pro_diff, unit_speech, wavegrad2 | WAV PCM16 | 22.05 kHz | none |
| your_tts | WAV PCM16 | 16 kHz | none |
| xtts_v2 | WAV PCM16 | 24 kHz | none |
| elevenlabs | MP3 128 kbps | 44.1 kHz | none |
| **playht** | MP3 ~160 kbps | 24 kHz | **`encoder=Lavf58.29.100`** |

So "16 kHz PCM with a Lavf tag" is 100% of the test set and 0% of the training set, and "Lavf tag" alone is PlayHT, the holdout generator. Any learned container model gets a perfect holdout score for the wrong reason and then collapses on the test set. The sponsor's resampled LJ clips also carry `Lavf58.29.100`, so the tag marks the sponsor's pipeline, not a class.

**What the container detector does instead** (`hearsay.detectors.container`): always runs (cheap); returns the header facts as features for routing (`lossy`, `is_pcm_wav`, `sample_rate`, `bits`, `n_tags`, `ffmpeg_written`) and a rule score that is 0.5 unless an embedded tag *value* names a speech-synthesis tool (0.9) or claims synthesis (0.75). No NSA or DiffSSD file trips either rule. Export: constant 0.5 on all 21,671 rows (minDCF 1.0, as it must be). This is the technique "leveraged" honestly: it routes (compression forensics only makes sense when the header or the signal says lossy) and it documents that the test set's container carries no class signal.

## Trap 2: the test set is low-passed at ~7.2 kHz

Long-term band level relative to the 1–3 kHz level, medians (dB). Computed on the silence-trimmed, level-normalized first 8 s; per-file values for the test set in `outputs/inventory/nsa_test_highband.csv`.

| Corpus | n | 6.5 k | 7.0 k | 7.25 k | 7.5 k | 7.75 k |
|---|---|---|---|---|---|---|
| **NSA test set** | 1,671 | −11.8 | −15.5 | **−29.5** | **−55.8** | **−62.1** |
| LJRealResampled (sponsor's real, 16 kHz) | 242 | −0.5 | 1.5 | 0.5 | −2.0 | −5.5 |
| LJ Speech (ours, 22.05 → 16 kHz) | 40 | −2.1 | 2.2 | 1.0 | −2.1 | −5.6 |
| LibriSpeech (native 16 kHz) | 40 | −13.3 | −17.8 | −19.3 | −19.6 | −18.8 |
| DiffSSD elevenlabs / playht / your_tts / xtts_v2 | 25 each | −13 … −8 | −18 … −9 | −20 … −10 | −23 … −12 | −25 … −15 |
| DiffSSD grad_tts / wavegrad2 | 25 each | −2 … −4 | 2 … 0 | 3 … 0 | −2 | −5 … −4 |

Reading: between 6.5 and 7.5 kHz the test set falls 44 dB (5th–95th percentile of the per-file drop: −52 to −25 dB). Nothing on the training side falls more than about 10 dB over the same span. 69 test files (4%) do not show the wall; see below. The roll-off starts near 7.0 kHz and reaches −50 dB by 7.75 kHz, which is what a resampler or a wideband speech-codec front end with a ~7.2 kHz cutoff leaves behind (a plain FFmpeg resample, as in LJRealResampled, cuts at ~7.8 kHz, and that is not it).

**Why it matters.** `hf_ratio_7k_*`, `rolloff95_*`, `bandwidth`, `centroid`, `flatness`, the upper MFCCs and the 7–8 kHz spectral-contrast band all differ between every training clip and every test clip for a reason that has nothing to do with synthesis. A model trained on our data sees test clips as "unlike any real clip I saw" and extrapolates. The v2 handcrafted detector put 90% of the test set above 0.5; with a 70% real test set that is a false-alarm rate of at least 86% at that threshold.

**Fix: match the training audio to the test set** (the plan's rule for level and silence, applied to bandwidth). A grid over Kaiser low-pass designs on 60 training clips against the test-set medians (error measured relative to the 6.5 kHz band):

| Design | 7.0 k | 7.25 k | 7.5 k | 7.75 k | rms error |
|---|---|---|---|---|---|
| target (test set) | −3.7 | −17.7 | −44.0 | −50.3 | |
| **Kaiser, cutoff 7250 Hz, 600 Hz transition, 45 dB (71 taps)** | −0.2 | −13.2 | −46.4 | −52.2 | **3.3 dB** |
| Kaiser 7200 / 700 / 45 dB | −1.7 | −15.5 | −49.5 | −54.6 | 3.8 |
| brick-wall FIR (1025 taps, 7200 / 500) | −0.4 | −54.6 | −67.8 | −68.0 | 23.8 |
| soxr HQ 16 → 22.05 → 16 kHz | 0 | 1.3 | −10.5 | −60.0 | 20.0 |
| LAME MP3 48 kbps at 16 kHz | 2.1 | −39.7 | −67.1 | −67.8 | 18.4 |

`hearsay.handcrafted.band_limit` is the first row, applied zero-delay (`fftconvolve`, `mode="same"`) to every clip before trimming and cropping, for training and test alike. On a test clip a second pass changes the waveform by 0.4% rms. `BAND_MATCH` records the design; the model bundle records `band_match=True` so the contract detector prepares test audio identically.

**The 69 files without the wall** are not a different pipeline. Their 7.75 kHz band is still −58 dB (median); the cutoff just sits nearer 7.5 kHz (7.5 kHz band −27 dB against −56 dB for the rest). Three are band-limited well below 6.5 kHz (telephony-like) and three are genuinely full-band to 8 kHz. Band matching everything is therefore right for 1,665 of 1,671 files, and the compression detector's `bw_hz` flags the six (after band matching, 981 test files read 7,250 Hz, 670 read 7,500 Hz, 12 read 8,000 Hz, 8 sit at 5–7 kHz).

## Handcrafted spectral + prosody detector

**Features** (75, `hearsay.handcrafted.features`): high-band energy ratios (> 4/6/7 kHz, mean and std), 85%/95% roll-off, centroid, bandwidth, flatness, 6-band spectral contrast, spectral flux, 20 MFCC means + stds; YIN pitch (median, IQR and std of log F0, slope std), voiced fraction, loudness dynamics, quiet-frame level, pause fraction and count, zero-crossing rate. Duration, peak and absolute level are excluded (they are corpus shortcuts: peak alone has AUC 0.66 train-vs-test). Cost 0.63 s per clip, YIN dominated; 20,000 clips take 8 minutes on 6 workers.

**Preparation, three versions.**

| | Trim | Segment | Band match |
|---|---|---|---|
| v1 (`crop_mode="first4s"`) | yes | first 4 s | no |
| v2 (`crop_mode="segment"`) | yes | `hearsay.embed.prepare_segment`: random crop to a length drawn from the NSA test durations, seed `0 + row`, 8 s cap; test clips whole | no |
| v3 | yes | as v2 | `band_limit` on every clip |

v2's crop lengths and offsets are drawn exactly as `scripts/extract_embeddings.py --segment --crop test --seed 0` draws them (`test_duration_sampler(0)`, one draw per manifest row in order; checked against the first five of the main chat's `nsa_train_sample_v2` shard: identical to 6 decimals), so the handcrafted and deep detectors are fused on the same audio.

**Models.** Standardized class-balanced logistic regression (C = 0.1) and LightGBM (300 trees, 31 leaves), each scored by out-of-fold minDCF on the predefined inner folds; the better one is refit on all inner rows and read out once on the outer holdout. Selection is by inner CV only. Logistic regression won the CV for v1 and v2; LightGBM won it for v3.

**Results.**

| | Selected by CV | CV minDCF (logreg / lgbm) | Holdout minDCF | Holdout EER | Holdout AUC | π = 0.5 | Sponsor code, flipped | Test: mean score | Test: share > 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| v1 | logreg | 0.615 / 0.708 | 0.709 | 12.0% | 0.945 | 0.494 | 0.385 | 0.85 | 88% |
| v2 | logreg | 0.615 / 0.707 | 0.661 | 10.8% | 0.951 | 0.471 | 0.363 | 0.87 | 90% |
| **v3** | **lgbm** | 0.640 / **0.614** | **0.254** | **4.3%** | **0.993** | 0.168 | 0.127 | 0.43 | **42%** |

Sponsor code as shipped (higher = bona fide, unflipped) reads 1.0 for every version, as expected for a 1.0 = synthetic detector; that is the score-direction tripwire, not a model property.

**Untangling the v3 jump** (every cell: model refit on all inner rows, read out on the holdout; the CV-selected cells are the ones above):

| Version × model | Holdout minDCF | EER | AUC | playht | wavegrad2 | LibriSpeech | LJ | Test share > 0.5 |
|---|---|---|---|---|---|---|---|---|
| v2 logreg (selected) | 0.661 | 10.8% | 0.951 | 0.67 | 0.65 | 0.84 | 0.38 | 90% |
| v2 lgbm | 0.374 | 5.9% | 0.986 | 0.33 | 0.42 | 0.46 | 0.08 | 68% |
| v3 logreg | 0.687 | 12.1% | 0.948 | 0.63 | 0.73 | 0.83 | 0.37 | 83% |
| **v3 lgbm (selected)** | **0.254** | **4.3%** | **0.993** | 0.25 | 0.26 | 0.32 | 0.10 | **42%** |

Two things are true at once. LightGBM transfers to the held-out generators far better than logistic regression at either version (0.37 vs 0.66 on v2), while the inner CV, pooled over folds that hold out the three generators these features cannot see at all, preferred logistic regression until v3. And the band match is a real gain on top: for LightGBM it takes the holdout from 0.374 to 0.254 and, more importantly for the test set, it moves the share of test files above 0.5 from 68% to 42%; for logistic regression it does not help the holdout but still pulls the test share from 90% to 83%. The selection rule stayed inner-CV-only throughout (the holdout was read once per version), so v3 lgbm is a legitimate pick, and the disagreement between inner CV and outer holdout is itself worth knowing for fusion: inner-OOF scores understate what this detector does on generators it can see.

**Per generator and per bona fide source** (v3 lgbm; inner rows are out-of-fold, holdout rows from the final model; v2 logreg in parentheses):

| Generator (inner OOF) | minDCF | | Holdout | minDCF |
|---|---|---|---|---|
| unit_speech | 0.25 (0.30) | | playht | 0.25 (0.67) |
| diffgan_tts | 0.27 (0.22) | | wavegrad2 | 0.26 (0.65) |
| your_tts | 0.32 (0.66) | | | |
| xtts_v2 | 0.47 (0.43) | | **Bona fide source** | |
| openvoicev2 | 0.53 (0.17) | | LJ Speech (holdout chapters) | 0.10 (0.38) |
| elevenlabs | 0.80 (0.98) | | LibriSpeech (holdout speakers) | 0.32 (0.84) |
| pro_diff | 1.00 (0.81) | | | |
| grad_tts | 1.00 (1.00) | | | |

**What separated the classes** (single-feature AUC on inner rows, v3; < 0.5 means lower on synthetic): spectral contrast in the top band (`contrast4_mean` 0.26) and bottom band (`contrast0_mean` 0.30), the frame-to-frame variability of the upper MFCCs (`mfcc16–19_std` 0.28–0.35: synthetic spectra are steadier), pitch variability (`f0_iqr_log` 0.29, `f0_std_log` 0.35: synthetic voices are flatter), spectral flux (0.32), and, once the dead band was removed, spectral flatness (`flatness_std` 0.67, `flatness_mean` 0.64: synthetic speech is noisier between harmonics) plus a few MFCC means (`mfcc13_mean` 0.66, `mfcc5/6_mean` 0.63). Nothing in the prosody block beyond pitch variability mattered: pause fraction/count, voiced fraction, loudness dynamics and zero-crossing rate all sit within 0.05 of chance.

**What had no effect or hurt.** The first-4-s crop (v1 → v2 gained 0.05 holdout minDCF from length parity alone). Anything above 7 kHz (Trap 2). Logistic regression on these features (it never transferred to the held-out generators). The prosody block beyond pitch variability.

**Honest reading.** On the two held-out generators the v3 detector is respectable (minDCF 0.25, EER 4%), but three of the eight inner generators (grad_tts, pro_diff at 1.0; elevenlabs at 0.80) are invisible to it, and exactly those are the modern zero-shot and diffusion systems the test set is likeliest to contain. Treat the holdout number as "what it does on generators it can see", not as an estimate for the test set. Its value is (a) the rubric's spectral and prosody techniques with readable evidence per file, (b) a fusion column whose information differs from XLS-R's, and (c) the two traps above, which it exposed.

## Compression forensics detector

**Features** (19, `hearsay.compression.features`, one STFT, about 10 ms per clip): effective bandwidth (the highest 250 Hz band still within 40 dB of the 1–3 kHz level), four band levels between 4 and 8 kHz relative to 1–3 kHz, high-band tilt; and, in 3–7 kHz (below the band-match cutoff, which empties 7.25–8 kHz for every clip), deep holes (cells more than 75 dB below the clip peak) and local holes (cells more than 30 dB below the frame's band median), each as fraction, frame-to-frame flicker, persistence and runs per frame; floor depth (2nd and 10th percentile), valley depth; the 0.3–3 kHz floor; the whole-clip floor.

**Laundering-equalized training.** A random half of the training rows of *both* classes (10,171 of 20,000, drawn per row from `default_rng(seed + row)` and recorded in the `cmp_launder` column) were re-encoded with ffmpeg through MP3 (libmp3lame, three times in four) or AAC at 24–128 kbps and 16 / 22.05 / 44.1 kHz, then decoded back to 16 kHz before feature extraction. In DiffSSD only ElevenLabs and PlayHT are MP3 and every real clip is lossless, so without this step "lossy = fake" *is* the model and the holdout (PlayHT) would reward it.

**Does it see codec history?** Yes. A class-agnostic logistic regression on the same 19 features, trained to tell laundered from clean: inner OOF AUC 0.90, holdout AUC 0.94 (0.94 on real, 0.95 on fake). Sensitivity by round-trip on the holdout, as mean P(laundered): MP3 24–48 kbps 0.92–0.99, MP3 64 kbps 0.78, AAC 24 kbps 0.97, AAC 32–64 kbps 0.54–0.63, MP3/AAC 96–128 kbps 0.39–0.56, clean 0.16. Strongest cues: whole-clip floor depth (AUC 0.15: laundered clips have a shallower floor after level normalization), valley depth (0.73) and the local-hole fraction, flicker and runs (0.69–0.70). Applied to *unlaundered* training rows it also lights up `openvoicev2` (0.78: that vocoder's output has codec-like holes) and, weakly, native 128 kbps ElevenLabs MP3s (0.33).

**Does it see the class?** Barely, once codec history is equalized. Inner CV minDCF: logistic regression 1.00 (no better than "always real"), LightGBM 0.93 (EER 25%). Holdout: minDCF 0.90, EER 20%, AUC 0.89; on clean holdout rows 0.85, on laundered ones 0.91; playht 0.81, wavegrad2 0.97; LJ 0.64, LibriSpeech 0.91. Single-feature AUCs sit within 0.16 of chance (valley depth 0.34, local holes 0.35, deep-hole runs 0.64). Test set: mean score 0.72 and **81% of files above 0.5**, the "everything looks synthetic" signature the handcrafted detector had before the band match.

**What the test set looks like under this lens.** The laundering detector scores the test set at a mean P(laundered) of 0.62, with 72% of files above 0.5 (10th–90th percentile 0.34–0.85). Together with the 7.2 kHz wall this says the sponsor's pipeline passed the clips through a lossy or wideband-speech codec stage. On this test set, codec traces are pipeline, not class. **Recommendation: keep `compression.csv` out of fusion unless the leave-one-out ablation shows a gain; keep the detector for routing and evidence** (bandwidth, hole fraction, the laundering reading), which is what the rubric's technique asks for.

## ENF (mains hum) detector

`hearsay.detectors.enf`, rule-based, never learned. 2 s Hann windows (0.49 Hz bins, quadratic peak interpolation) at a 0.5 s hop on the trimmed clip capped at 8 s; for each of 50 and 60 Hz, track the strongest bin within ±1 Hz per frame and its SNR against the ±2–8 Hz neighbourhood. Present = median SNR ≥ 12 dB in ≥ 60% of frames. Stable = tracked-frequency std ≤ 0.25 Hz and range ≤ 0.5 Hz. Scores are deliberately mild: stable hum 0.4 (a genuine electrical recording environment), discontinuous hum 0.6 (a fabricated or spliced background), none 0.5. The training corpora (studio, audiobook, TTS) carry no hum, so the rule cannot be validated on labels and a strong swing would be "confidently wrong on a subpopulation".

**Inventory.** Test set: hum present in 24 of 1,671 files (15 at 50 Hz, 9 at 60 Hz; median SNR 14.7 dB, max 29.9 dB), stable in 16. Training sample: corpus-driven, not class-driven. A stable hum is present in 36% of LJ Speech clips (Linda Johnson's home studio) and 21% of LibriSpeech clips, in under 0.5% of nine DiffSSD generators, and, tellingly, `grad_tts` (trained on LJ) reproduces a 50/60 Hz component in 19% of its clips that is stable in only 1.6% of them, which is exactly the "discontinuous hum" rule firing on a vocoder that learned the hum but not the grid. As a score it cannot move minDCF alone (three discrete values; inner and holdout minDCF 1.0), and a fusion column would learn "hum = LJ = real" from training, a corpus artifact that fires on 1% of the test set. Keep it as evidence and a routing fact, not a fusion column.

## Splice / discontinuity detector

`hearsay.detectors.splice`, rule-based. (1) Clicks: a sample-to-sample step more than 15× the RMS of the steps in the surrounding 20 ms (the sample itself excluded, so a plosive burst does not count) and more than 15% of full scale. (2) DC-offset jumps: adjacent 100 ms window means differing by more than 2% of full scale and more than 6× the clip's typical window-to-window drift. (3) Background-floor variability (range of a 0.4 s running-minimum level tracker), reported as a feature only: in the 1 h box it could not be separated from ordinary pauses (a pause exposes the room floor; a speech-only stretch does not). Score 0.6 when a click or DC jump is found, else 0.5. The sponsor said no clip is partially synthetic, so a seam is evidence of editing, not of synthesis.

**Inventory.** Test set: 58 of 1,671 files (3.5%) show a seam, 16 by click and 44 by DC jump. On the training sample the seams are synthesis and codec artifacts rather than edits: single-sample clicks in 27% of `wavegrad2` clips and 14% of `diffgan_tts`; DC jumps in 13% of `unit_speech`, 12% of `playht` and 9% of `elevenlabs`, but also in 9% of LibriSpeech; against 0.7% of LJ Speech and 0.1% of `pro_diff`. Inner and holdout minDCF 1.0 (a two-valued score). The floor-range feature behaves as feared (median 28 dB on the test set, 33–99 dB on training, driven by pauses). Value: per-file evidence and a vocoder-artifact observation worth a line in the README; not a fusion column.

## Score export format (what M4 consumes)

`outputs/detector_scores/<name>.csv`, columns `path, fold, split, score, logit`.
- `split` ∈ {`inner_oof`, `holdout`, `test`}; `fold` is the fold-file value (`0`–`4`, `holdout`) or `test`.
- Learned detectors: inner rows are out-of-fold predictions from the predefined folds (every learned piece, the scaler included, fit fold-locally); holdout rows come from the model fit on all inner rows; test rows from the same model on `outputs/manifests/nsa_test.csv`, keyed by path in manifest order. `score` = P(synthetic), `logit` = log(score / (1 − score)) with score clipped to [1e-6, 1 − 1e-6].
- Rule-based detectors (`scripts/score_detector.py`): nothing is fitted, so all rows carry the detector's plain output; the same three splits are used for uniformity. A row whose result is not `ok` has NaN score/logit (fusion imputes it and sets the missing indicator). A `<name>_results.csv` beside it keeps status, evidence and features per row for the explanation report.
- All exports cover the same 21,671 paths (20,000 training + 1,671 test); paths are unique.

## Recommendations for the main chat

1. **Band-match the deep detector's training audio.** XLS-R sees the raw waveform; the 7.2 kHz wall is in every test clip and in no training clip. The cheapest fix is one line in `hearsay.embed.prepare_segment` (or in the extraction loop): `x = hearsay.handcrafted.band_limit(x)` before the trim, for training *and* test. Re-extract `nsa_train_sample_v2` and `nsa_test_v2` and compare the holdout minDCF and, more importantly, the test-score distribution (share of test files above 0.5 should move toward 30%). This is the single most likely cause of a draft-review surprise.
2. **Route on bandwidth.** The `compression` detector's `bw_hz` feature (and the per-file inventory) identifies the 69 test files without the wall and any telephony-band files; the orchestrator can log them and fusion can report minDCF on that subpopulation separately.
3. **Fusion ablation.** Use `handcrafted_v2.csv` vs `handcrafted_v3.csv` (same rows) to see what band matching does inside fusion, not only alone. Expect `container.csv` to contribute nothing (constant); keep it for the routing log and the documentation. Treat `compression.csv` as suspect: it calls 81% of the test set synthetic, and the test set reads as laundered throughout, so drop it unless leave-one-out shows a gain. `enf.csv` and `splice.csv` are near-constant on the test set; their training-side signal is corpus and vocoder artifacts.
4. **Do not learn anything from the container.** If the orchestrator ever adds container fields to fusion, Trap 1 applies.

## What worked / what had no effect (README-ready)

**Worked**
- Matching training audio to the test set's ~7.2 kHz low-pass (`band_limit`): the largest single gain of the track, and the difference between a detector that calls 90% of the test set synthetic and one that calls 42%.
- Length parity with the deep detector (random crops to test-duration lengths; no first-4-s crop): +0.05 holdout minDCF on its own.
- Gradient-boosted trees over the 75 handcrafted features (LightGBM): they transfer to unseen generators; logistic regression on the same features does not.
- Spectral contrast, upper-MFCC frame-to-frame variability, pitch variability and spectral flatness as the interpretable cues (synthetic speech: flatter pitch, steadier spectra, noisier between harmonics).
- Treating the container as routing information, not evidence, and measuring why.
- Laundering-equalized training for compression forensics (a random half of *both* classes re-encoded through MP3/AAC), so the detector cannot learn "MP3 = ElevenLabs/PlayHT".

**No effect or hurt**
- Every feature above 7 kHz before band matching (a corpus fingerprint, not a class cue).
- Any learned use of container fields (perfect on training, meaningless on the test set).
- Prosody beyond pitch variability: pause statistics, voiced fraction, loudness dynamics, zero-crossing rate all sit at chance on this data.
- Logistic regression as the handcrafted classifier.
- ENF and splice as *scores*: no training corpus carries mains hum or edits, so their scores are mild by design and their value is evidence and routing.
- The handcrafted features on diffusion TTS in LJ's voice (`grad_tts`, `pro_diff`) and on ElevenLabs: invisible at every version. Only the deep detector can cover those.

## State for the next chat

- **Tracked:** everything under `src/`, `scripts/`, `tests/` listed above, this report, the CLAUDE.md disclosure lines and the `docs/STATUS.md` rows. Nothing pushed.
- **At risk (gitignored, not archived):** `models/hc_*`, `models/cmp_*`, `outputs/handcrafted/*_v{2,3}.csv`, `outputs/compression/*.csv`, `outputs/detector_scores/*.csv`, `outputs/inventory/container_*.csv` and `nsa_test_highband.csv`. All regenerable with the commands below (about 25 minutes of CPU on 6 workers).
- **Owned by the main chat, untouched:** `embed.py`, `probe.py`, `submission.py`, the embedding/probe scripts, `submissions/`, `splits/nsa_folds.csv`.
- **Open:** the deep detector band match (Recommendation 1); speaker-embedding drift (technique 6) was never started; ENF and splice scores are untested against any labeled edited or hum-bearing audio.
- **Gotcha found late:** the first commits of this track were made while the full suite was crashing (a pipeline masked pytest's exit status); the crash was the lightgbm + torch OpenMP conflict above, fixed in the follow-up commit with `hearsay.trees`. The suite was green before the follow-up was committed.

## How to reproduce

```bash
cd ~/Projects/hearsay && uv sync                      # external drive mounted
uv run python scripts/inventory_container.py --manifest outputs/manifests/nsa_test.csv --name nsa_test
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_test.csv --name nsa_test_v3 --workers 6
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_train_sample.csv --name nsa_train_sample_v3 --crop test --seed 0 --workers 6
uv run python scripts/train_handcrafted.py --train nsa_train_sample_v3 --folds splits/nsa_folds.csv --test nsa_test_v3 --out-name handcrafted
uv run python scripts/extract_compression.py --manifest outputs/manifests/nsa_test.csv --name nsa_test --workers 6
uv run python scripts/extract_compression.py --manifest outputs/manifests/nsa_train_sample.csv --name nsa_train_sample --crop test --seed 0 --launder-frac 0.5 --workers 6
uv run python scripts/train_handcrafted.py --train nsa_train_sample --folds splits/nsa_folds.csv --test nsa_test --features-dir outputs/compression --model-prefix cmp --rung "D-track (compression forensics)" --out-name compression
for d in container enf splice; do uv run python scripts/score_detector.py --detector $d --workers 6; done
uv run pytest -q && uv run ruff check .
```

`--no-band-match` and `--crop-mode first4s` reproduce v2 and v1.
