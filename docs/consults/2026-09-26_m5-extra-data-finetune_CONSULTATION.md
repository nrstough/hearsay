# Consult: M5, XLS-R fine-tune data mix and recipe (Sat Sep 26, 2026, ~02:30 EDT)

**Target:** a frontier LLM, for strategic and technical opinion.
**Scope:** only the M5 fine-tune rung of HEARSAY: four questions on the extra-data mix and the fine-tune recipe.
**Blocking?** No. The data bundle and training code are being built in parallel. Answers become config: ratios, caps, unfreeze depth, augmentation set. A small inner-fold ablation has the final say.

Copy below the line.

---

## Your role

You are a speech anti-spoofing / audio deepfake detection researcher: SSL front ends, ASVspoof, In-the-Wild generalization. Push back hard if a premise below is wrong or we're over-engineering. We have about 30 hours left in a hackathon and about $13 of cloud GPU credit (roughly 6 H100-hours), so practical beats ideal.

## The problem

- **Task.** Binary clip-level detection: bona fide vs synthetic speech, for the NSA "HEARSAY" challenge at HackGT 13.
- **Output.** A probability per file, 1.0 = synthetic.
- **Scoring (60% of the judging).** Normalized minDCF with C_FA = 4 (calling real audio synthetic), C_miss = 1, and π_synth = 0.3; about 70% of the test files are real.
  - That works out to minDCF = 9.33·P_FA + P_miss, normalized.
  - **False alarms on real speech dominate.** Only the score ranking matters.
- **Test set.** 1,671 WAV files, 16 kHz PCM, unlabeled.
  - Durations: 3.0–13.6 s, median 3.41 s, p90 4.57 s, p99 7.1 s; 5.9% are over 5 s and 0.6% over 8 s.
  - The brief says the set may include:
    - bona fide speech from studio, smartphone, telephony and field recordings across codecs and bitrates
    - modern zero-shot TTS and neural vocoders
    - voice conversion
    - replay
    - scene manipulation
    - laundered fakes: transcoding, band-limiting, noise
    - metadata-spoofed containers
- **Deployment.** Scoring runs offline in a Docker image on CPU.

## What's fixed

- **Front end.** `facebook/wav2vec2-xls-r-300m`: 24 transformer layers, 1024-d.
- **Input path, identical for train and test:**
  1. FFmpeg decodes to 16 kHz mono float32.
  2. Leading and trailing silence are trimmed (relative 35 dB).
  3. Per-input zero-mean, unit-variance normalization.
  4. No tiling or repeat-padding.
  5. Test-time input is the whole trimmed clip, with a cap of about 14 s.
- **Shortcuts deliberately neutralized:**
  - **Level:** the LJ Speech real audio peaks around 0.54, while the NSA test set peaks around 1.0.
  - **Leading silence:** about 0.37 s in LibriSpeech vs about 0.06 s in the test set.
  - **Tiling seams**, from repeat-padding short clips.
  - **Container/codec:** all of it is decoded before training.
- **Training crops.** A per-batch crop length is sampled from the test-duration distribution, about 70%, mixed with uniform 3–14 s. Clips are length-bucketed with an attention mask and re-cropped at random each epoch.
- **Validation.** A grouped nested fold file.
  - **Outer holdout:** the `playht` and `wavegrad2` generators plus 26 bona fide speaker/chapter groups. It is never used for training or selection.
  - **Inner folds 0–4:** generator-grouped for spoof, speaker- or chapter-grouped for bona fide.
  - We produce 5-fold out-of-fold scores for a logistic stacker with engineered detectors, plus one "full" model trained on all inner folds for holdout and test.
- **Fine-tune plan (open to change).**
  - The CNN feature encoder stays frozen; the top N transformer layers and a pooled linear head are trained.
  - bf16, with class-balanced sampling.
  - A fixed epoch count, chosen in a pilot on inner fold 4.
- **Augmentation plan (open to change).** Applied to both classes. We always report a clean-only score next to the augmented one.
  - Precomputed codec round-trips: MP3, Opus, AAC, AMR-NB, μ-law.
  - On the fly: additive noise at 5–30 dB SNR, band-limits of 3.4, 4 and 7 kHz, simulated RIR (synthetic exponential-decay impulses, since we have no RIR corpus), and random gain before normalization.

## Core training data (sponsor-provided, in the fold file)

| Pool | Clips (sample / full available) | Notes |
|---|---|---|
| DiffSSD spoof, 10 generators: grad_tts, unit_speech, diffgan_tts, openvoicev2, pro_diff, xtts_v2, your_tts, elevenlabs, playht, wavegrad2 | 10k in the sample (1k per generator) / 70k | Mostly diffusion TTS, plus XTTS, YourTTS and 2 commercial generators. Some files are MP3 or 22 kHz; all are decoded to 16 kHz. |
| LJ Speech, bona fide | 6k / 13.1k | **One female speaker**; a 16 kHz resampled subset |
| LibriSpeech dev-clean + test-clean, bona fide | 4k / 5.3k | LibriVox audiobooks |

Per-fold spoof generators:
- fold 0: grad_tts, unit_speech
- fold 1: diffgan_tts, openvoicev2
- fold 2: pro_diff, xtts_v2
- fold 3: your_tts
- fold 4: elevenlabs
- holdout: playht, wavegrad2

Each inner fold has about 1.6k bona fide.

## Extra open data we can add to training (training only; never scored, never in the holdout)

| Pool | Size | Our concern |
|---|---|---|
| **Rest of the NSA manifest** | +50k DiffSSD spoof (inner generators only), +8.4k LJ and Libri real | Low risk; same domain |
| **ASVspoof 2019 LA, train + dev** | 5.1k bona fide (VCTK, about 100 speakers) + 45k spoof (A01–A06, 2018-era TTS and VC) | The real-speaker diversity looks valuable, since our real side is one LJ speaker plus LibriSpeech. The attacks are old, and ASV19 has a known silence shortcut, which our trimming mostly removes. |
| **MLAAD English** | 6.4k **spoof only**, from 143 TTS models: 30 clips per model, plus 300 each for ElevenLabs Turbo v2.5 / v2 Multilingual / v3, Gemini 3.1 Flash TTS, Qwen3-TTS (0.6B/1.7B/CustomVoice). Durations 2–26 s. | Its matching bona fide (M-AILABS audiobooks) is **not downloaded**. The generators are modern and commercial, which is exactly what the test likely contains. Family overlap with our folds (ElevenLabs → fold 4, OpenVoiceV2 → fold 1, the XTTS family → fold 2) is handled: a fold model never trains on the extra generators from its own validation families. |
| In-the-Wild (Müller 2022) | 16 GB | **Never trained on** (project rule); used only as a stress-test readout |

Planned batch mix (a first guess): balanced by label, NSA at 50% or more of every batch, MLAAD at 15% or less of the spoof side, ASV19 spoof about 1:1 with ASV19 real.

## Current numbers (for calibration of expectations)

- **The frozen XLS-R probe (M1) on NSA** is not in yet; it's running now. This is the bar M5 must beat, measured as outer-holdout minDCF at π = 0.3.
- **The frozen XLS-R probe on ASVspoof 2019 (public shakedown):**
  - generator-grouped CV minDCF 0.180 (π = 0.5), layer 7 best
  - A07–A19 eval minDCF 0.023
  - Probably inflated by the ASV19 silence shortcut.
- **Engineered detectors on NSA** (handcrafted spectral and prosody features + LightGBM), inner-fold CV at π = 0.3:
  - overall minDCF 0.61
  - per held-out generator: diffgan_tts 0.27, unit_speech 0.25, your_tts 0.32, xtts_v2 0.47, openvoicev2 0.53, elevenlabs 0.80, grad_tts 1.0, pro_diff 1.0
  - per bona fide source: LJ 0.43, LibriSpeech 0.77
  - So the hard cases are the diffusion vocoders and the commercial generators, and real LibriSpeech is the main false-alarm source.
- **Compression-forensics detector:** CV minDCF 0.93, near chance on these decoded inputs.

## What I want (ranked)

For each question: your recommendation, a runnable experiment we can finish in under 1 H100-hour, and EV × effort.

1. **Spoof-only MLAAD.** Adding 6.4k MLAAD spoof without its M-AILABS bona fide: how big is the corpus/recording-chain shortcut risk ("audiobook-TTS acoustics = fake")?
   - Do our mitigations handle it: laundering augmentation, a cap of 15% or less of the spoof side, silence trimming, level normalization?
   - Is LibriSpeech (also LibriVox) an adequate bona fide proxy, or should we fetch M-AILABS en_US real audio (a few GB) or drop MLAAD?
   - Is there a cheap diagnostic that reveals the shortcut? One idea: a classifier trained on MLAAD vs LibriSpeech, scored on held-out MLAAD-source real audio.
2. **ASVspoof 2019 LA.** Do its A01–A06 attacks help or hurt when the targets are modern diffusion and commercial TTS?
   - Use bona fide only, bona fide plus balanced spoof, or skip it?
   - Does ASV19 bona fide (VCTK read speech, a different mic chain) add false-alarm risk on phone or field real audio, or reduce it?
3. **Mixing and curriculum.**
   - What NSA share per batch, and what caps per source?
   - Should the extra data stay in the whole run, be used for warm-up then tapered, or come in only after an NSA-only warm-up?
   - Any per-source loss weighting?
   - With only about 16k NSA training clips per fold model, is more data likely to dominate the recipe?
4. **Fine-tune recipe for XLS-R 300M under this budget** (each run about 10–15 min on an H100):
   - how many top layers to unfreeze (vs full fine-tune with a frozen CNN)
   - learning rates, backbone vs head; layer-wise LR decay
   - epochs
   - **RawBoost** vs our codec/noise/RIR/band-limit set (or both, and at what probability)
   - pooling: mean, attentive statistics, or AASIST-style graph back ends
   - whether to use hidden states from middle layers (e.g. a learned weighted sum of layers) rather than the last layer, given that the frozen probe liked layer 7
   - whether one-class / OC-softmax loss is worth it for unseen generators

## Additional context

- **What already works (the bar):** the frozen-probe pipeline on the identical input path. M5 must beat it on the outer holdout, or we keep M1.
- **What we can A/B:** inner-fold minDCF from any config in about 15 min of H100 per run. The ablation budget is about 4–6 runs total, including the final 6 (5 fold models plus 1 full model).
- **Hard lines:**
  - no training on the outer holdout or In-the-Wild
  - no selection on the holdout
  - offline CPU inference: a single model, no big ensembles
  - 16 kHz mono everywhere
  - output 1.0 = synthetic
- **Physical replay** is not covered in training (simulated RIR only).
- **The false-alarm cost dominates.** A config that raises real-speech diversity may beat one that adds spoof diversity, even at equal EER. Tell us if you disagree.

## Output format

1. A one-line **verdict** on the overall data plan.
2. A **ranked options table**: approach | fit to our constraints | GPU cost | effort | verdict (do / maybe / skip).
3. For each of the four questions: a recommendation with concrete numbers (ratios, caps, N layers, LRs, epochs, augmentation probabilities) and one **runnable experiment**.
4. A direct **"are we over-engineering?" call**. What would you cut, given about 6 H100-hours and 30 hours of wall clock?
5. Any **failure mode we haven't named**, especially around the cost asymmetry (false alarms on real speech).
