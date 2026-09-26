# Brief: handcrafted v4, feature discovery for the three blind generators (Sat Sep 26, 2026, ~03:40)

For the teammate taking the spectral/prosody lane. This replaces the generic explainer with the project's actual numbers, code paths and rules. Read [STATUS.md](../STATUS.md), the [CPU detector report](../reports/2026-09-26_cpu-detectors.md) and [`src/hearsay/handcrafted.py`](../../src/hearsay/handcrafted.py) first; this brief tells you what in them matters for your job.

**The job in one sentence.** Add cheap acoustic statistics to `hearsay.handcrafted.features` that separate `grad_tts`, `pro_diff` and `elevenlabs` from real speech, without raising false alarms on LibriSpeech real speech, measured on the shared folds with the shared preprocessing, and exported through the existing trainer.

**Deadlines.** Detector scores due **Sat 19:00**. Fusion freezes **Sat 22:00**. After that, nothing new gets in.

## 1. What exists, and the bar you're extending

The pipeline is already `audio → features() → LightGBM → P(synthetic)`, fold-validated, exported for fusion, wrapped as a contract detector with plain-English evidence. You are adding columns to `features()`. Everything downstream is built.

**Current detector (v3, `models/hc_lgbm_20260926-0232`)**

| Readout | Value |
|---|---|
| Inner-fold OOF minDCF (LightGBM / logistic) | 0.614 / 0.640 |
| Outer-holdout minDCF, EER, AUC | 0.254, 4.3%, 0.993 |
| Test set: share of files scored > 0.5 | 42% (NSA says ~30% are synthetic) |
| Features, cost | 75, about 0.63 s per clip (YIN pitch dominates) |

**Per generator, inner out-of-fold, v3.** This is your target table. Lower is better; 1.0 means "no better than always saying real".

| Generator | Voice | minDCF | Verdict |
|---|---|---|---|
| unit_speech | multi-speaker | 0.25 | visible |
| diffgan_tts | LJ | 0.27 | visible |
| your_tts | multi-speaker | 0.32 | visible |
| xtts_v2 | multi-speaker | 0.47 | partly |
| openvoicev2 | multi-speaker | 0.53 | partly |
| **elevenlabs** | commercial, MP3 source | **0.80** | **blind** |
| **pro_diff** | LJ, diffusion | **1.00** | **blind** |
| **grad_tts** | LJ, diffusion | **1.00** | **blind** |
| playht (holdout) | commercial, MP3 source | 0.25 | visible |
| wavegrad2 (holdout) | LJ, diffusion vocoder | 0.26 | visible |

**Per real source** (real clips of that source vs all synthetic): inner OOF LJ 0.43, **LibriSpeech 0.77**; holdout LJ 0.10, LibriSpeech 0.32. Unfamiliar real voices draw the false alarms. That is the side that costs 4×.

**What already separates the classes** (single-feature AUC on inner rows; < 0.5 means lower on synthetic): spectral contrast bands 4 and 0 (0.26, 0.30), frame-to-frame variability of upper MFCCs `mfcc16–19_std` (0.28–0.35), pitch variability `f0_iqr_log` and `f0_std_log` (0.29, 0.35), spectral flux (0.32), spectral flatness (0.64–0.67). Reading: synthetic speech here has flatter pitch, steadier spectra frame to frame, and is noisier between harmonics. **Do not re-measure these.** Note what carried signal: the *variability over time* of spectral shape, more than its mean. Build your new families with that in mind.

**What sits at chance and should not be re-attempted as-is:** pause fraction and count, voiced fraction, loudness dynamics, zero-crossing rate, and every feature above 7 kHz.

**Why grad_tts and pro_diff are the cleanest target you'll ever get.** Four DiffSSD generators (grad_tts, diffgan_tts, pro_diff, wavegrad2) speak in LJ Speech's voice, and LJ Speech real is in every training fold. So "grad_tts vs LJ real" is the same voice, same text style, same nominal recording, different origin. A feature that separates that pair is measuring synthesis, not speaker or corpus. Both are diffusion TTS front ends with a GAN-type neural vocoder behind them. Vocoder phase incoherence is the documented weakness of that class, which is why phase and group delay are first on the list.

## 2. The preprocessing you inherit (and must not bypass)

`hearsay.handcrafted._crop` runs before any feature, in this order:

1. `band_limit(x)`: a 71-tap Kaiser low-pass, cutoff 7,250 Hz, 600 Hz transition, 45 dB, applied zero-delay (`fftconvolve`, `mode="same"`). **Linear phase, so it does not distort phase or group delay below 7 kHz.**
2. `hearsay.embed.prepare_segment(x, crop_s, seed)`: silence trim (35 dB below the loudest 20 ms frame), then for training rows a random crop to a length drawn from the NSA test-duration distribution (median 3.4 s), then an 8 s cap. Test clips are used whole, no crop.
3. RMS-normalize to 0.1.

So inside `features()` you receive `x`: float32, 16 kHz mono, band-limited to ~7.2 kHz, trimmed, about 3–8 s, at a fixed RMS. Consequences:

- **Nothing above 7 kHz is signal.** The test set is low-passed there (1,602 of 1,671 files fall 44 dB between 6.5 and 7.5 kHz) and the filter now does the same to training audio. Any filterbank, CQT or band feature you add must stop at 7,000 Hz. Bins above it are filter shape, identical for both classes, and just add noise columns.
- **Absolute level is gone.** Do not add level features; add ratios and shapes only.
- **Duration is not a feature.** Training crops and test clips share a length distribution by construction; do not add anything that scales with clip length (use means, stds, percentiles and rates per second, never counts).
- **Crop edges are arbitrary.** Training clips are cut mid-word at a random offset; test clips are whole utterances with natural onsets and offsets. Any feature that looks at how a clip begins or ends will separate train from test, not real from fake. Onset/offset shape, first-syllable energy, edge silence: don't.

The correction to the explainer: the 7 kHz wall is on **every** test file, real and fake alike, and on **no** training file of either class. It never showed up in fold validation (train and validation were both full-band). It showed up as the v2 detector calling 90% of the test set synthetic. Your fold numbers cannot catch this class of shortcut; only the test-set score distribution can (section 7).

**Cost budget.** Extraction is 0.63 s per clip; 20,000 clips take about 8 minutes on 6 workers. Keep any new family under about 0.2 s per clip. Use at most **6 workers**; the main chat's GPU job needs the other cores for decoding.

## 3. Where to plug in

Minimal-change path, all within the CPU lane:

1. New module `src/hearsay/hc_v4.py`. One function per family, each taking what it needs and returning a flat `dict[str, float]`:
   ```python
   def lfcc(x: np.ndarray, S: np.ndarray) -> dict[str, float]: ...      # S = power STFT, 512/160
   def phase(x: np.ndarray, Z: np.ndarray) -> dict[str, float]: ...     # Z = complex STFT, 512/160
   def modulation(rms_db: np.ndarray) -> dict[str, float]: ...
   FAMILIES = {"lfcc": ..., "phase": ..., "cqcc": ..., "modulation": ..., "jitter": ..., "breath": ...}
   ```
2. In `handcrafted.features`, add a keyword `families: tuple[str, ...] = ()`. Keep the complex STFT (`Z = librosa.stft(...)`, then `s = np.abs(Z) ** 2`, which is what the code already computes) so phase families reuse it, and at the end `for name in families: f.update(FAMILIES[name](...))`. The existing final line already maps non-finite values to 0.0.
3. `scripts/extract_handcrafted.py`: add `--families lfcc,phase` and pass it through `features_for_path` (it is a positional `ex.map`; add one more list). Record `families` in the `.meta.json` sidecar.
4. `scripts/train_handcrafted.py`: copy `families` from the sidecar into the saved bundle, next to `crop_mode` and `band_match`.
5. `src/hearsay/detectors/_learned.py`, in `run()`: pass `families=b.get("families", ())` to the feature function so test-time audio gets the same features. `src/hearsay/detectors/handcrafted.py`: add a `LABELS` entry per new feature (plain English; it becomes the evidence sentence judges read).
6. Tests under `tests/`: one hermetic test per family (a tone and a noise clip, as `tests/test_handcrafted_detector.py` does), asserting finite values, expected key names, and that a shifted copy of the same clip gives the same values within tolerance (offset invariance; see section 2).

**Naming rules.** Column names must be unique, must not be in the trainer's metadata set (`path, label, generator, speaker, utt, source, filename, group, fold`) and must not end in `_flag`, `_launder` or `_crop_s`, which the trainer treats as metadata and drops. Prefix by family: `lfcc3_std`, `gd_std_mean`, `mod_2_4_ratio`.

**Files you must not touch** (other lanes): `src/hearsay/embed.py`, `probe.py`, `submission.py`; `scripts/extract_embeddings.py`, `train_probe.py`, `make_probe_csv.py`; `submissions/`; `splits/nsa_folds.csv`; `outputs/detector_scores/handcrafted.csv` (fusion consumes it; you write `handcrafted_v4.csv`). Never `import lightgbm` anywhere but `scripts/train_handcrafted.py` (it crashes next to torch on this Mac; `hearsay.trees` runs the trees in numpy). `git add` only your own paths, never `-A`; don't push.

## 4. The metric, exactly

Not `4·FPR + FNR`. The project's number is the normalized detection cost at π_synth = 0.3, C_FA = 4, C_miss = 1:

```
normalized minDCF = 9.33 · P_FA + P_miss,   minimized over the threshold
```

`hearsay.metrics.min_cost(y, s)` computes it (y: 1 = synthetic; higher s = more synthetic). `hearsay.metrics.report(y, s)` adds EER, the π = 0.5 reading and both sponsor-code directions. Only the ranking matters; the threshold is swept.

The explainer's example, redone at the real weights: detector A with P_FA 2%, P_miss 20% scores 0.39; detector B with P_FA 10%, P_miss 5% scores 0.98, barely better than a constant answer. One false alarm costs as much as 9.3 misses. Judge every feature by whether it moves the **real** side.

## 5. Folds, exactly

`splits/nsa_folds.csv`: columns `path, label, generator, speaker, source, group, fold`; `fold` ∈ {0, 1, 2, 3, 4, holdout}. Spoof is grouped by generator, real by speaker (LibriSpeech, 80 speakers) or by chapter (LJ, one speaker). 20,000 rows: 16,142 inner, 3,858 holdout.

| Fold | Synthetic generators held out in that fold | Real clips |
|---|---|---|
| 0 | grad_tts, unit_speech | ~1,635 |
| 1 | diffgan_tts, openvoicev2 | ~1,634 |
| 2 | pro_diff, xtts_v2 | ~1,632 |
| 3 | your_tts | ~1,609 |
| 4 | elevenlabs | ~1,632 |
| holdout | playht, wavegrad2 (+ 26 real groups) | 1,858 |

So the out-of-fold score for grad_tts comes from a model that never saw grad_tts (trained on folds 1–4). The trainer does this for you with `PredefinedSplit` and picks logistic vs LightGBM by inner OOF minDCF. It reads the holdout **once** per run.

**Rules.** Select on inner OOF only. Read the holdout as a sanity check, not a target; its noise is about ±0.07–0.10 (two generators), so a holdout gap under 0.15 is not evidence. Never look at the holdout while choosing features. Never touch In-the-Wild; the main chat uses it as an eval-only stress set.

"Real multi-speaker audio" in the earlier message means **LibriSpeech**, 80 different readers, one per clip. It does not mean several people in one clip; no clip here has that. The check is: per-source minDCF for LibriSpeech real must not get worse. The trainer prints it (`cv_by_bonafide_source`, `val_by_bonafide_source`).

## 6. Shortcuts specific to this data

These are measured, not hypothetical. Each is a way a new feature can look great and be wrong.

| Cue | Fact | What it does to a naive feature |
|---|---|---|
| Codec | Only `elevenlabs` (MP3 128 kbps, 44.1 kHz) and `playht` (MP3 ~160 kbps, 24 kHz) are lossy; every real clip and the other 8 generators are lossless. | A phase or fine-structure feature that "detects ElevenLabs" may be detecting MP3. Test: `hearsay.compression.launder` round-trips a clip through MP3/AAC; if the feature moves on laundered **real** clips, it is a codec cue. |
| Mains hum | Stable 50/60 Hz hum in 36% of LJ clips and 21% of LibriSpeech, in under 0.5% of nine generators; 24 of 1,671 test files have any hum. | Any low-frequency feature (CQCC with a low `fmin`, sub-100 Hz bands) learns "hum = real", which fires on 1% of the test set. Keep `fmin` ≥ 65 Hz and run the LJ-vs-LibriSpeech check below. |
| Corpus identity | LJ is one speaker in one home studio. LibriSpeech is audiobooks. Four generators clone LJ's voice. | A feature with AUC far from 0.5 on **LJ real vs LibriSpeech real** is a corpus cue, whatever it does to the classes. |
| Crop edges | Training crops start mid-word; test clips are whole utterances. | Any onset/offset-sensitive feature separates train from test. |
| Already closed | Peak level (AUC 0.66 alone), leading silence (LibriSpeech 0.37 s vs test 0.06 s), duration, container format, the 7 kHz wall. | Closed by RMS normalization, trim, test-length crops, `load_audio`, `band_limit`. Do not re-open any of them with a level, silence, length or bandwidth feature. |

## 7. The 30-minute gate, concretely

Per family: hypothesis → implement → extract on a small stratified sample → per-generator AUC table → keep or kill. Do not run the full 20,000-clip extraction per idea.

**Sample manifest** (about 2,400 clips, ~4 minutes at 0.63 s/clip on 6 workers; less if you compute only the new family):

```bash
cd ~/Projects/hearsay && uv run python - <<'EOF'
import pandas as pd
m = pd.read_csv("outputs/manifests/nsa_train_sample.csv")
f = pd.read_csv("splits/nsa_folds.csv")[["path", "fold"]]
m = m.merge(f, on="path")
m = m[m.fold != "holdout"]                     # inner rows only; the holdout is never used for gating
s = pd.concat([g.sample(min(len(g), 200), random_state=0) for _, g in m.groupby(["source", "generator"])])
s.drop(columns="fold").to_csv("outputs/manifests/hc_gate_sample.csv", index=False)
print(s.groupby(["source", "generator"]).size())
EOF
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/hc_gate_sample.csv \
    --name hc_gate_<family> --crop test --seed 0 --workers 6 --families <family>
```

**Gate table**: per new column, AUC of each blind generator against all real, AUC of each real source against all synthetic, and the corpus check.

```bash
cd ~/Projects/hearsay && uv run python - <<'EOF'
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as auc
d = pd.read_csv("outputs/handcrafted/hc_gate_<family>.csv")
d = d[d.hc_flag.fillna("") == ""]
new = [c for c in d.columns if c.startswith("<prefix>")]       # e.g. "lfcc", "gd_", "mod_"
real, spoof = d[d.label == "bonafide"], d[d.label == "spoof"]
rows = []
for c in new:
    r = {"feature": c, "all": auc(d.label == "spoof", d[c])}
    for g in ["grad_tts", "pro_diff", "elevenlabs", "openvoicev2", "xtts_v2"]:
        sub = pd.concat([real, spoof[spoof.generator == g]])
        r[g] = auc(sub.label == "spoof", sub[c])
    for src in ["ljspeech", "librispeech"]:
        sub = pd.concat([real[real.source == src], spoof])
        r[f"real_{src}"] = auc(sub.label == "spoof", sub[c])
    r["LJ_vs_Libri_real"] = auc(real.source == "ljspeech", real[c])   # corpus check: want ~0.5
    rows.append(r)
t = pd.DataFrame(rows).set_index("feature")
t["sep_blind"] = (t[["grad_tts", "pro_diff", "elevenlabs"]] - 0.5).abs().max(axis=1)
print(t.sort_values("sep_blind", ascending=False).round(3).to_string())
EOF
```

**Proposed keep/kill rule** (numbers are a starting point; tighten if many pass):

- **Keep** a family if at least one feature has |AUC − 0.5| ≥ 0.15 on grad_tts, pro_diff or elevenlabs, **and** its `LJ_vs_Libri_real` AUC is within 0.5 ± 0.15, **and** its `real_librispeech` AUC is no worse than the same feature's `real_ljspeech` by more than 0.1.
- **Kill** a family if its best blind-generator separation is under 0.10, or if it separates LJ from LibriSpeech real more strongly than it separates real from fake.
- Codec check for anything that passes on elevenlabs: launder 100 real clips through MP3 128 kbps at 44.1 kHz (`hearsay.compression.launder`), recompute the feature, compare distributions to the unlaundered clips. If it moves by more than half a standard deviation, it is a codec cue; drop or note it.

**Then the real run** for the families that survive. Use the **full** manifest and `--seed 0`, so crop lengths match the deep detector row for row (that parity is what lets fusion compare the two on the same audio):

```bash
cd ~/Projects/hearsay
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_test.csv \
    --name nsa_test_v4 --workers 6 --families lfcc,phase
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_train_sample.csv \
    --name nsa_train_sample_v4 --crop test --seed 0 --workers 6 --families lfcc,phase
uv run python scripts/train_handcrafted.py --train nsa_train_sample_v4 --folds splits/nsa_folds.csv \
    --test nsa_test_v4 --out-name handcrafted_v4
```

The trainer prints the top single-feature AUCs, the inner CV for both models, the holdout once, per generator, per real source, and the test-set share above 0.5. **Success looks like:** inner CV minDCF below 0.614; grad_tts / pro_diff / elevenlabs OOF minDCF visibly below 1.00 / 1.00 / 0.80; LibriSpeech per-source minDCF not above 0.77; test share above 0.5 not moving away from 30%. It writes `outputs/detector_scores/handcrafted_v4.csv` in the fusion format (`path, fold, split, score, logit`; 21,671 rows; `score` = P(synthetic)) and a bundle under `models/hc_<kind>_<stamp>/`. You never write the export by hand, so the score direction cannot be inverted by accident.

Tell the main chat when `handcrafted_v4.csv` exists and what the per-generator table says. It decides whether fusion switches from v3 to v4.

## 8. The families, grounded

Order by expected value per half hour. Each says what to compute on the `x`, `Z`, `S` and `rms_db` you already have inside `features()`.

**8.1 LFCC (first; easiest; named in the plan and never built).** 40 triangular filters linearly spaced 0–7,000 Hz over the 257-bin power spectrum `S` (n_fft 512, hop 160), log, DCT-II (ortho), keep 20 coefficients. Summarize over frames: mean, std, and **std of the first difference** (the v3 signal was in upper-MFCC frame-to-frame variability, so include the deltas). 60 features. Cost is negligible. Expect modest gains on the visible generators and a chance on ElevenLabs; probably nothing on the diffusion pair.

**8.2 Phase and group delay (the main bet for grad_tts and pro_diff).** Use the complex STFT `Z` (same 512/160 frame). Two unwrap-free quantities, both restricted to voiced frames (the existing `voiced` mask) and to bins between 300 and 7,000 Hz:

- *Group delay without unwrapping.* With `Y = stft(n · x[n])` using the same window, `τ_g(k, t) = Re(Y / Z) = (Z_r·Y_r + Z_i·Y_i) / |Z|²`, in samples. Clip to ±256. Per frame take the std and the interquartile range across bins; summarize over frames as mean, std, 10th/90th percentiles. Natural speech has structured group delay near harmonics; GAN vocoders produce phase that is locally inconsistent, so the spread differs.
- *Instantaneous-frequency deviation.* `Δφ(k, t) = angle(Z(k, t+1) · conj(Z(k, t))) − 2π·k·160/512`, wrapped to [−π, π]. For a steady harmonic the deviation is nearly constant over time; incoherent phase makes it jump. Features: per-bin temporal std of Δφ, then mean and median across bins; and the fraction of bins with std above 0.5 rad.

Two cautions. (1) The random crop offset changes every raw phase value but not these derived quantities; test that with a shifted copy. (2) ElevenLabs is MP3-decoded; MP3 mangles phase in a codec-specific way. Run the laundering check before believing any ElevenLabs result from this family.

**8.3 CQCC.** `librosa.cqt(x, sr=16000, fmin=65, n_bins=162, bins_per_octave=24, hop_length=256)` reaches about 6,994 Hz, safely under the wall. Log power, DCT across the 162 bins, keep 24 coefficients, then mean, std and delta-std over frames (72 features). fmin 65 Hz keeps the 50/60 Hz fundamentals out, but the 100–180 Hz hum harmonics are in band, so run the LJ-vs-LibriSpeech check. Cost about 0.05–0.1 s per clip.

**8.4 Rhythm / modulation spectrum.** Envelope is the existing `rms_db` at 100 frames per second (frame 400, hop 160). Mean-subtract, Hann, rfft. Features: energy in 1–2, 2–4, 4–8 and 8–16 Hz as fractions of the 1–16 Hz total; peak modulation frequency; spectral entropy of 1–16 Hz. Expectations are **low**: a 3.4 s clip gives 0.3 Hz resolution and 10–15 syllables, and every pause/energy statistic v3 tried sits at chance. Keep it to one gate cycle.

**8.5 Jitter and shimmer.** The 10 ms YIN track already yields `f0_slope_std`, which is a frame-level jitter proxy and is *not* among the top cues, so frame-level variants are unlikely to add much. The version worth 30–60 minutes is cycle-level: band-pass `x` to 0.7–1.4× the clip's median F0, take positive-going zero crossings as cycle marks inside voiced runs of at least five cycles, periods `T_i` from consecutive marks. Jitter (local) = mean |T_i − T_{i+1}| / mean T; RAP with a 3-point average; shimmer from the peak |x| within each cycle on the unfiltered signal. Reject runs where any |Δ log F0| > log 1.5 (octave errors). Median-filter the YIN track (5 frames) before using it to define voiced runs. This is the riskiest cheap family; if the marks are unstable after 30 minutes, kill it.

**8.6 Breath.** Frames with −35 dB < `rms_db` < −20 dB are the between-speech material (below −35 is what `pause_frac` already counts). Features: their fraction, and the mean spectral centroid and flatness of those frames relative to the voiced frames' values. A breath is broadband and noisy; digital or room silence is not. One gate cycle; expectations low, given that pause statistics carry nothing here.

**8.7 Speaker-embedding drift (only if the above are done; rubric technique 6, never started).** ECAPA weights are **not** downloaded and SpeechBrain is not a dependency; budget `uv add speechbrain` plus the `speechbrain/spkrec-ecapa-voxceleb` download, and tell whoever owns Docker that the image needs both. Windows of 1.0 s at 0.5 s hop give 5–6 embeddings on a 3.4 s clip; features: minimum and mean pairwise cosine, std of cosines. Build it as a contract detector (`src/hearsay/detectors/speaker_drift.py`, see `enf.py` for a rule-based example) with a mild score like ENF's, export with `scripts/score_detector.py --detector speaker_drift`, and expect near-constant scores on 3 s clips. It is for the 20% technique-diversity rubric with per-file evidence, not for minDCF.

## 9. Writing it down (graded)

Add to `docs/reports/2026-09-26_cpu-detectors.md` or a new `docs/reports/2026-09-26_handcrafted-v4.md`: the gate table for every family tried, kept **and killed**, with the per-generator and per-real-source numbers, and one sentence on why each was kept or dropped. "What had no effect" is scored. Add one disclosure line under "AI use disclosure" in `CLAUDE.md` if you used an assistant.

## 10. Reading order

1. `docs/STATUS.md`
2. `docs/reports/2026-09-26_cpu-detectors.md`: the "Handcrafted spectral + prosody detector" section and "Trap 2".
3. `src/hearsay/handcrafted.py`, `scripts/extract_handcrafted.py`, `scripts/train_handcrafted.py`.
4. `src/hearsay/detectors/_learned.py` and `detectors/handcrafted.py` (where the evidence sentence comes from).
5. `src/hearsay/metrics.py`.
6. `tests/test_handcrafted_detector.py` for the hermetic test pattern.

Setup: `cd ~/Projects/hearsay && uv sync`, external drive mounted (`ls data/nsa/DiffSSD data/ljspeech data/librispeech`), `uv run pytest -q` → 215 passed, `uv run ruff check .` clean. Keep both green.
