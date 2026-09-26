# Feature engineering: where you can add (map for the spectral/prosody teammate)

Sat Sep 26, 2026, ~05:00. This is the "where do I plug in" companion to the brief ([2026-09-26_handcrafted-v4-brief.md](2026-09-26_handcrafted-v4-brief.md), which has the math, the metric and the gate). Read this one first, then the brief, then the code in the order at the bottom. Everything here was checked against the repo at commit `4b644c9` and the working tree after it. A repo-wide code map is being generated at `docs/code-map.md`.

## 1. The lane in one picture

```
audio file
  │ load_audio                       src/hearsay/audio.py          16 kHz mono float32
  │ band_limit → trim → crop → RMS   src/hearsay/handcrafted.py    _crop(), lines 55-69
  ▼
features(x, families=(...))          src/hearsay/handcrafted.py    lines 72-141
  ├─ 75 v3 features                  (spectral 64, prosody 11)     built in
  └─ ctx dict → hc_v4.FAMILIES       src/hearsay/hc_v4.py          ← YOU ADD HERE (line 300)
  ▼
outputs/handcrafted/<name>.csv       scripts/extract_handcrafted.py  --families lfcc,phase,...
  ▼
train_handcrafted.py                 scripts/train_handcrafted.py    logistic vs LightGBM, inner folds
  ├─ models/hc_<kind>_<stamp>/       bundle: trees + feature stats + families
  ├─ outputs/detector_scores/<out-name>.csv     fusion input (path, fold, split, score, logit)
  └─ meta.json                       per-generator, per-real-source, holdout, test share > 0.5
  ▼
detectors/handcrafted.py             HandcraftedDetector: recomputes your families at inference,
                                     writes the evidence sentence from LABELS
```

Five files. You will touch two of them for a new family (`hc_v4.py`, `detectors/handcrafted.py`), one line in a third (`train_handcrafted.py`), and add a test.

## 2. The one function you write

A family is a function of a context dict returning a flat `{name: float}`. Register it in `FAMILIES` at [hc_v4.py:300](../../src/hearsay/hc_v4.py). `features()` builds the context once and calls each requested family ([handcrafted.py:133-139](../../src/hearsay/handcrafted.py)).

What the context gives you, all computed on the prepared clip (band-limited to 7.25 kHz, silence-trimmed, cropped to a test-set length for training rows, RMS-normalized to 0.1):

| Key | Shape | What it is |
|---|---|---|
| `x` | `(n_samples,)` float32 | the waveform, 16 kHz |
| `Z` | `(257, T)` complex | STFT, n_fft 512, hop 160 (32 ms frames every 10 ms) |
| `S` | `(257, T)` float | power spectrogram, `abs(Z)**2` |
| `freqs` | `(257,)` | bin center frequencies in Hz, 0 to 8000 in 31.25 Hz steps |
| `voiced` | `(T,)` bool | frames with energy above −30 dB relative and YIN pitch in 65–390 Hz |
| `rms_db` | `(T,)` float | frame loudness in dB relative to the loudest frame (frame 400, hop 160) |
| `f0` | `(T,)` float | YIN pitch track, 60–400 Hz, frame 1024, hop 160 |

Minimal template, matching the existing families:

```python
# src/hearsay/hc_v4.py
def harmonic_noise(ctx: dict) -> dict[str, float]:
    S, freqs, voiced = ctx["S"], ctx["freqs"], np.asarray(ctx["voiced"], dtype=bool)
    band = freqs <= FMAX                      # never above 7 kHz
    ...                                       # compute per-frame values on voiced frames
    return _stats(values, "hnr_band1", pct=True)   # -> hnr_band1_mean/_std/_p10/_p90

FAMILIES["hnr"] = harmonic_noise              # or add it to the dict literal at line 300
```

`_stats(v, prefix, pct)` at [hc_v4.py:29](../../src/hearsay/hc_v4.py) gives mean, std and optional p10/p90. Use it; it also handles empty inputs.

## 3. Rules the pipeline enforces (and the ones it can't)

Enforced by code or tests:
- **Finite floats only.** `features()` maps non-finite values to 0.0 at the end, but don't rely on it; return real numbers.
- **Column names.** Not in `{path, label, generator, speaker, utt, source, filename, group, fold}` and not ending in `_flag`, `_launder` or `_crop_s`; the trainer treats those as metadata and drops them ([train_handcrafted.py:46-56](../../scripts/train_handcrafted.py)). Names must be disjoint from the 75 v3 names (test at [tests/test_hc_v4.py:48](../../tests/test_hc_v4.py)).
- **Offset invariance.** A shifted copy of the same clip must give the same values within tolerance ([tests/test_hc_v4.py:59](../../tests/test_hc_v4.py)). Training crops start at random offsets, test clips are whole utterances; a feature that depends on where the clip starts is a train-vs-test shortcut.
- **Family prefix map.** The trainer keeps `FAMILY_PREFIXES`, a family → column-prefix table ([train_handcrafted.py:51-52](../../scripts/train_handcrafted.py)), so a bundle only asks for the families its columns need. It falls back to the family name as the prefix (line 144), so a family whose columns start with its own name (`hnr` → `hnr_*`) needs no entry. Add one only if your column prefixes differ from the family name, as `phase` (`gd_`, `pc_`) and `jitter` (`jit_`, `shim_`) do.

Not enforceable by code, so they are on you:
- **Nothing above 7 kHz.** The test set is low-passed at 7.2 kHz and every clip you see is filtered to match. Bins above `FMAX` (7,000 Hz) are filter shape, identical for both classes. `cqcc` and `lfcc` stop at 7 kHz for this reason.
- **No level, no duration, no counts.** Level is normalized away; clip length is matched by construction. Use means, stds, percentiles, fractions and rates per second.
- **No onset or offset features.** See offset invariance.
- **Voiced-only where it matters.** Pitch-period and phase features are meaningless on unvoiced frames; use the `voiced` mask (with the fallback pattern in `_band_and_voiced`, [hc_v4.py:80](../../src/hearsay/hc_v4.py)).
- **Cost.** Measured from the extraction sidecars, in worker-seconds per clip: the 75 v3 features cost 0.14 s, and all six v4 families together add about 0.06 s (0.20 s for all 234 columns). Keep a new family under about 0.05 s per clip. Use at most 4 CPU workers until 05:15, 6 after.

## 4. Checklist for a complete family

1. Function + `FAMILIES` entry in [src/hearsay/hc_v4.py](../../src/hearsay/hc_v4.py) (line 300).
2. A `FAMILY_PREFIXES` entry ([scripts/train_handcrafted.py:51-52](../../scripts/train_handcrafted.py)) only if your column names do not start with the family name.
3. Plain-English `LABELS` for each column in [src/hearsay/detectors/handcrafted.py](../../src/hearsay/detectors/handcrafted.py) (the v4 block starts at line 55). This text becomes the evidence sentence judges read, e.g. "harmonic-to-noise ratio, 1–2 kHz 2.3 SD below real speech (toward synthetic)". Unlabeled columns fall back to their raw name.
4. Tests in [tests/test_hc_v4.py](../../tests/test_hc_v4.py). The finite/disjoint test (line 48) and the offset-invariance test (line 59) already run over every entry in `FAMILIES`, but the first one asserts your exact column-name set against the `EXPECTED` dict near the top of the file, so add your family's names there. Then one behavioral test, like `test_phase_family_separates_coherent_from_incoherent_phase` at line 67: a synthetic input where you know which way the feature should move.
5. `uv run pytest -q tests/test_hc_v4.py tests/test_handcrafted_detector.py` green, `uv run ruff check .` clean.
6. Gate it (section 6). Keep or kill.
7. If kept: full extraction, training variant, In-the-Wild readout (section 7).

## 5. Where the open slots are

What exists and how it did at the gate (details in [docs/reports/2026-09-26_handcrafted-v4.md](../reports/2026-09-26_handcrafted-v4.md)):

| Family | Columns | Best blind-generator AUC | State |
|---|---|---|---|
| `cqcc` | 72 | pro_diff 0.94, elevenlabs 0.82, grad_tts 0.72 | strong; 15 columns are corpus cues |
| `lfcc` | 60 | pro_diff 0.13 (`lfcc17_std`, which reads 0.75 on elevenlabs); elevenlabs 0.85 (`lfcc0_dstd`) | strong; 32 columns are corpus cues |
| `phase` (`gd_`, `pc_`) | 12 | grad_tts 0.80 on a corpus-neutral cue; pro_diff 0.21 | keep; the only family that sees grad_tts cleanly |
| `jitter` (`jit_`, `shim_`) | 5 | grad_tts 0.63, pro_diff 0.34 | marginal |
| `modulation` (`mod_`) | 6 | elevenlabs 0.66 | marginal, as expected on 3.4 s clips |
| `breath` | 4 | pro_diff 0.28 on one column | marginal |

Ranked places to add, given that:

1. **Harmonic fine structure, untried.** Per-band harmonic-to-noise ratio (say 0–1, 1–2, 2–4, 4–7 kHz) and the inter-harmonic noise floor relative to the harmonic peaks, on voiced frames. The v3 result that "synthetic speech is noisier between harmonics" came from whole-band flatness; nobody has asked which band. Use `S`, `freqs`, `f0` and `voiced`. Prefix `hnr_`.
2. **Vocoder excitation, untried.** LPC residual statistics per voiced frame (order 16 on the 7 kHz signal): residual kurtosis, peak-to-RMS, and the regularity of residual peaks at the pitch period. Neural vocoders produce an excitation that is too regular or too Gaussian. Uses `x`, `f0`, `voiced`. Prefix `lpc_`.
3. **Extend `phase`, which works.** `pc_` measures coherence at every peak pooled together. Split it by harmonic order (first three harmonics vs the rest) and by band; diffusion vocoders may fail at high harmonics only. Same function shape as `peak_coherence` ([hc_v4.py:118](../../src/hearsay/hc_v4.py)). Prefix `pc_`, new suffixes.
4. **Formant transitions, untried.** Formant bandwidths and the smoothness of F1/F2 trajectories across voiced runs (LPC roots per frame, then track). TTS coarticulation is often too smooth or too abrupt. Riskier: formant tracking on 3 s clips is noisy. Prefix `fmt_`.
5. **Fix rather than add: `breath`.** It currently measures the spectral shape of mid-level unvoiced frames. A real breath is broadband and 100–400 ms long; add duration and count-per-second of such events. Cheap.
6. **Corpus-cue cleanup, not a feature.** 83 of the 234 columns separate LJ from LibriSpeech real speech (`outputs/handcrafted/hc_gate_all_columns.csv`), and 34 v3 columns make LibriSpeech real look synthetic, which is the expensive error. Any family you add must pass the LJ-vs-LibriSpeech check at ≤ 0.15 deviation, and the trainer's `--drop-columns` ([train_handcrafted.py:118](../../scripts/train_handcrafted.py), exact names or `prefix*` globs) lets you train without the offenders.

Not in this lane: the speaker-embedding drift detector. It is a new contract detector, not a feature family (`src/hearsay/detectors/speaker_drift.py`, template `enf.py`, export with `scripts/score_detector.py`); section 8.7 of the brief has the recipe.

## 6. The gate, as it was actually run

Sample: `outputs/manifests/hc_gate_sample.csv`, 200 inner-fold clips per generator and per real source, 2,000 clips, never the holdout. Extraction on that sample with one family takes about 90 s on 4 workers.

```bash
cd ~/Projects/hearsay
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/hc_gate_sample.csv \
    --name hc_gate_<family> --crop test --seed 0 --workers 4 --families <family>
```

Then the per-column table (the brief, section 7, has the script): AUC of each blind generator vs all real, AUC of each real source vs all synthetic, LJ-vs-LibriSpeech corpus check. Keep rule per column: |AUC − 0.5| ≥ 0.15 on grad_tts, pro_diff or elevenlabs; corpus deviation ≤ 0.15; the two real sources within 0.10 of each other. A family survives if any column does. Existing tables to compare against: `outputs/handcrafted/hc_gate_all_columns.csv`.

Codec check for anything that passes on elevenlabs: launder 100 real clips through MP3 128 kbps with `hearsay.compression.launder`, recompute, and require the feature to move by less than 0.5 clean standard deviations.

## 7. From a kept family to a fusion column

```bash
# full extraction (full manifest, seed 0: crops match M1 row for row)
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_test.csv \
    --name nsa_test_v5 --workers 6 --families lfcc,phase,cqcc,<yours>
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_train_sample.csv \
    --name nsa_train_sample_v5 --crop test --seed 0 --workers 6 --families lfcc,phase,cqcc,<yours>
# train; never overwrite handcrafted.csv (fusion reads it) or handcrafted_v4.csv (the current candidate)
uv run python scripts/train_handcrafted.py --train nsa_train_sample_v5 --folds splits/nsa_folds.csv \
    --test nsa_test_v5 --out-name handcrafted_v5 [--drop-columns "contrast0_mean,mfcc2_std,..."]
# real-world false alarms, eval only
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/itw_stress.csv \
    --name itw_stress_v5 --crop test --seed 200 --workers 6 --families lfcc,phase,cqcc,<yours>
uv run python scripts/eval_bundle.py --bundle models/hc_lgbm_<stamp> \
    --features outputs/handcrafted/itw_stress_v5.csv --oof outputs/detector_scores/handcrafted_v5.csv --name itw_stress
```

How to read the trainer's `meta.json` (in `models/hc_<kind>_<stamp>/`):

| Key | What to look at |
|---|---|
| `cv.lgbm.min_dcf`, `cv.logreg.min_dcf` | inner out-of-fold minDCF; the selection number. v3 lgbm: 0.614 |
| `cv_by_generator` | the blind three: v3 has grad_tts 1.00, pro_diff 1.00, elevenlabs 0.80 |
| `cv_by_bonafide_source.librispeech` | false alarms on unfamiliar real voices; v3: 0.77. Must not get worse |
| `val_min_dcf`, `val_by_generator`, `val_by_bonafide_source` | the holdout, read once; gaps under 0.15 are noise |
| `test.frac_above_half` | share of test files called synthetic; prior is 0.30, v3 is 0.42 |
| `feature_auc_top15` | which of your columns the model leans on |

And `eval_<name>.json` from `eval_bundle.py`: minDCF, EER and P_FA on the 2,000 In-the-Wild real clips at the inner-fold threshold. The deep models sit at 1.2% (M1) and 5.7% (M1b) there; that number is now a gate for every column entering fusion.

## 8. Do not touch, and when

Owned by other lanes: `src/hearsay/embed.py`, `probe.py`, `submission.py`, `m5_*.py`, `spectra.py`; `scripts/extract_embeddings.py`, `train_probe.py`, `make_probe_csv.py`, `m5_*.py`, `score_spectra.py`; `submissions/`; `splits/nsa_folds.csv`; `outputs/detector_scores/handcrafted.csv` and `handcrafted_v4.csv`. Never `import lightgbm` outside the trainer. Stage only your own files with `git add <paths>`; never `-A`; don't push.

Deadlines: detector scores due **Sat 19:00**; fusion freezes **Sat 22:00**. A family that isn't in a scored `handcrafted_v*.csv` by 19:00 doesn't ship.

## 9. Reading order

1. This file, then the brief ([2026-09-26_handcrafted-v4-brief.md](2026-09-26_handcrafted-v4-brief.md)).
2. [src/hearsay/hc_v4.py](../../src/hearsay/hc_v4.py): six worked examples of exactly what you'll write. `peak_coherence` (line 118) is the best-commented.
3. [src/hearsay/handcrafted.py](../../src/hearsay/handcrafted.py) lines 56–141: what the context holds and how it's built.
4. [tests/test_hc_v4.py](../../tests/test_hc_v4.py): the test pattern.
5. [docs/reports/2026-09-26_handcrafted-v4.md](../reports/2026-09-26_handcrafted-v4.md): the gate tables and the corpus-cue lists.
6. [scripts/train_handcrafted.py](../../scripts/train_handcrafted.py) and [scripts/eval_bundle.py](../../scripts/eval_bundle.py) only when you get to section 7.
