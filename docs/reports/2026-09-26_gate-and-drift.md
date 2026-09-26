# Non-speech gate, default-answer policy, and speaker-embedding drift

Sat Sep 26, 2026, 06:00–07:00. Follows architecture.md section 10, decision 4 (default answer for undetermined files) and the v4 brief's section 8.7 (rubric technique 6, speaker-embedding consistency). Both are contract detectors in `src/hearsay/detectors/`, registered by `hearsay.detectors.engineered`, exported in the M4 format by `scripts/score_detector.py`. Neither is a fusion column: the gate's score is constant and its decision lives in `features["is_speech"]`; drift's score is mild and, on this data, points the wrong way for a stacker (below).

## 1. The problem the gate solves

The deep detectors are trained on speech and score non-speech near 1.0: the public-data M1 probe scored pure silence 0.99. With a false alarm at 9.33× a miss, a file with no speech to judge belongs at the real end of the ranking, deliberately and reproducibly, with a reason the explanation report can print.

## 2. The gate (`hearsay.detectors.speech_gate`)

Speech, for the gate, is material that is
1. **not silent**: RMS above −70 dBFS (digital silence is −180; the quietest NSA test file is −34.6);
2. **voiced**: at least 5% of frames within 30 dB of the loudest have a YIN pitch in 65–390 Hz (NSA test minimum 0.099);
3. **moving in pitch**: std of log F0 over voiced frames ≥ 0.02 (a tone or chord locks YIN to one value: 0.00; NSA test minimum 0.099);
4. **moving in loudness**: std of frame loudness ≥ 3 dB (a steady tone, hum or stationary noise: under 1 dB; NSA test minimum 6.0);
5. **not spectrally flat**: long-term spectral flatness < 0.5 (white noise 0.998; NSA test maximum 0.39).

Any failed cue gates the file; the failed cues are the reasons (`reason_silence`, `reason_unvoiced`, `reason_tone`, `reason_static`, `reason_noise`) and go into the evidence sentence. Score is always 0.5. Cost about 0.13 s per clip (YIN).

**Set from the test set, not guessed.** `outputs/inventory/nsa_test_speech_gate.csv` holds the five cues for all 1,671 test files; every threshold sits beyond that file's extreme, so **no NSA test file is gated**. A sixth cue, "a few spectral lines" (share of long-term energy in the top five bins ≥ 70%, meant for tones and chords), was tried and dropped: 77 real test files exceed it, because speech puts most of its long-term energy in a few low bins; tones and chords are caught by cues 3 and 4 anyway.

**Cases** (`tests/test_speech_gate.py`): the submission preflight's silence and three-note chord, a 220 Hz tone, white noise, a speech-like synthetic (moving pitch, syllable-rate loudness, a pause) that must pass, the same at −64 dBFS RMS that must not read as silence, and 60 real test clips of which ≥ 98% must pass (100% do).

**Known limit.** Rhythmic polyphonic music moves in pitch and loudness and can pass this gate. The submission preflight (silence, chord) and the flag on any fused score above 0.8 remain the backstop; a learned speech/music discriminator would be the next step if the draft review shows music in the test set.

## 3. The policy (`speech_gate.apply_default_answer`)

A gated file gets `DEFAULT_ANSWER` = 0.02 plus `1e-4 × its fused score`. So every gated file ranks **below every speech file** (speech scores are never that low after calibration), gated files keep a **deterministic order among themselves** (the fused score still breaks ties, which matters if the sponsor's ranking has to be total), and the explanation report can say "no speech to judge (silence): default answer". A decode failure is handled upstream by `safe_run` (error result) and should take the same path. The orchestrator applies this after fusion; fusion itself never sees the gate as a column.

Why 0.02 and not 0.0: the constant rollback submission is all 0.0, and a gated file should still be distinguishable from "no score at all" in the TSV and the log.

## 4. Speaker-embedding drift (`hearsay.detectors.speaker_drift`)

ECAPA-TDNN speaker embeddings (SpeechBrain `spkrec-ecapa-voxceleb`, Apache-2.0, 89 MB, fetched to `weights/spkrec-ecapa-voxceleb`; loaded from there, never from the Hub at run time) on 1.0 s windows at a 0.5 s hop over the silence-trimmed clip (cap 8 s), unit-normalized; features are the minimum, mean and std of the pairwise cosines and the minimum adjacent-window cosine. Rule: minimum pairwise cosine below **0.05** → score 0.6 ("voice drifts across the clip"), else 0.5. Cost about 0.25 s per clip on CPU; the encoder loads in under 2 s.

**Calibration** (`outputs/inventory/speaker_drift_calibration.csv`; inner rows only): 120 LibriSpeech real, 60 LJ real, 160 DiffSSD fakes (20 per generator), 80 synthetic two-speaker splices (the first half of one LibriSpeech speaker's clip glued to the first half of another's), 80 same-speaker splices as control.

| Threshold on minimum pairwise cosine | Single-voice clips flagged | Two-speaker splices flagged |
|---|---|---|
| 0.00 | 2.6% | 75% |
| **0.05** | **6.2%** | **94%** |
| 0.10 | 12.6% | 98% |
| 0.20 | 35% | 100% |

At 0.05 the flags fall on LibriSpeech real 11.7%, LJ real 6.7%, fakes 1.9%, same-speaker splices 12.5%, two-speaker splices 93.8%. One-second ECAPA windows are noisy for one speaker too (median minimum cosine 0.20 for real, 0.28 for fakes), which is why the rule sits near zero.

**A finding worth a line in the README: the fakes are the most consistent voices.** Every drift statistic points the same way: fakes have higher minimum, adjacent-minimum and mean cosines than real speakers (class AUCs in the table below). A cloned voice does not vary from second to second the way a person does. That makes drift *evidence of editing or of a real speaker*, not of synthesis; a fused column would learn "drift = real", the opposite of the rubric's framing, so the export is for the routing log and per-file evidence only.

| Statistic (real 180 vs fake 160, inner rows) | Class AUC (> 0.5 = higher on fakes) |
|---|---|
| mean pairwise cosine | 0.725 |
| minimum pairwise cosine | 0.692 |
| minimum adjacent-window cosine | 0.663 |
| std of pairwise cosines | 0.195 (fakes vary less) |

These are single-statistic AUCs on a 340-clip inner sample, not a validated detector; they say the direction, and they mark "speaker-embedding consistency" as a candidate v4-style feature family (`spk_*`, four columns, 0.25 s per clip) for whoever reopens the feature lane. Not pursued here: the lane is closed and the column would need the ECAPA encoder inside every extraction worker.

**Export summaries** (`scripts/score_detector.py`, all 21,671 rows, no failures; gate 405 s on 6 workers, drift 2,043 s on 4 threads).

| | NSA test (1,671) | DiffSSD spoof (10,000) | LibriSpeech real (4,000) | LJ real (6,000) |
|---|---|---|---|---|
| Gate: files gated | **0** | 43 (all "stationary noise") | 41 (all "stationary noise") | 0 |
| Drift: files flagged (min cosine < 0.05) | **16.1%** (269) | 1.4% | 8.7% | 6.0% |
| Drift: median minimum cosine | 0.27 | 0.28 | 0.21 | 0.18 |
| Drift: median windows per clip | 6 | 13 | 10 | 13 |

The 84 gated training clips are very noisy recordings and very noisy vocoder output (spectral flatness above 0.5); none of the test set is. The drift rate on the test set (16%) is above every training corpus; per generator in training the rule fires on 0–4% of fakes and 7% of real clips, so on a 70%-real test set with phone and field recordings a rate in the teens is what "drift marks real speakers and harder conditions" predicts. That is one more reason it is a routing and evidence detector, not a fusion column: fused, it would push 16% of the test set, mostly real, toward synthetic.

## 5. Exports and where they plug in

- `outputs/detector_scores/speech_gate.csv` (score 0.5 throughout) and `speech_gate_results.csv` with `is_speech` and the reasons per path; `outputs/detector_scores/speaker_drift.csv` / `_results.csv` with the cosine statistics per path. All 21,671 rows.
- Orchestrator: read `is_speech` from the gate's features (or the results file) and apply `apply_default_answer` to the fused score; print the gate's evidence for gated files.
- Docker: needs `speechbrain` (now in `pyproject.toml`) and `weights/spkrec-ecapa-voxceleb` copied into the image; `HF_HUB_OFFLINE=1` is fine because the detector loads from the local directory.
