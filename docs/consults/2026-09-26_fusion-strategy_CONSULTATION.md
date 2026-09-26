# Consult: fusion, selection, false alarms and the "default answer" (Sat Sep 26, 2026, ~03:30 EDT; updated ~04:30 with In-the-Wild and band-limit findings; ~05:50 with the band-matched v3 numbers, M3 and the fusion v0 readout)

**Target:** a frontier LLM, for strategic and technical opinion.
**Scope:** everything **except** the M5 fine-tune recipe, which has its own consult: `2026-09-26_m5-extra-data-finetune_CONSULTATION.md`.
**Blocking?** Partly. The fusion and selection answers decide what we send for NSA's one-time draft review on Saturday afternoon.

Copy below the line.

---

## Your role

You are an applied researcher in audio deepfake detection and in detection-cost evaluation (ASVspoof, NIST SRE-style DCF). Push back hard on any wrong premise. We have about 26 hours left in a hackathon (final TSV due in about 23). Practical beats ideal.

## The problem

- **Task.** The NSA "HEARSAY" challenge at HackGT 13: output P(synthetic) per audio file, with 1.0 = synthetic.
- **Test set.** 1,671 unlabeled 16 kHz mono PCM WAVs, 3.0–13.6 s long (median 3.4 s), English. NSA says "approximately 70% of the files are real."
- **Judging.**
  - **60% MinDCF.** Points are (1 − ours)/(1 − best) × 60, where "best" is the best team's minDCF.
  - **20%** for diversity and depth of techniques. The brief itemizes: container/metadata, spectral, prosody, ENF, compression, speaker-embedding consistency, deep anti-spoofing, splice, and a bonus for "agentic orchestration" (choosing analyses per file).
  - **20%** for GitHub documentation "in your own words", including what did not work and explainability.
- **Stated cost.** A false alarm (real called synthetic) costs 4× a miss. With π_synth = 0.3, our normalized minDCF is **9.33·P_FA + P_miss**. That is our selection metric; EER is logged alongside.
- **The sponsor's scoring code contradicts the instructions.** Their MinDCF code (shipped to us) is the ASVspoof5 Track 1 evaluation package, unmodified:
  - `Pspoof = 0.5, Cmiss = 1, Cfa = 4`
  - `compute_det_curve(bona, spoof)` treats a **higher score as bona fide**
  - The instructions say 1.0 = synthetic.
  - Scored with that code as shipped, every good detector of ours gets **minDCF 1.00**. Scored with our scores flipped, the same detectors get 0.09–0.13.
  - Our decision is to follow the instructions (1.0 = synthetic) and never flip. We have asked the sponsor; there's no answer yet.
- **The instructions also say:** "You will have to strategize and use some sort of Game Theory or Benchmark maxing to determine what the default answer to be if you cannot determine an answer for all of the files. This could be crucial!"
- **Deliverables.** The TSV (before 8 AM Sunday), a Docker image that runs inference offline on CPU, and a README. We also get **one optional early review of a draft TSV** against the test labels.

## Data and validation

**Training data (sponsor-provided plus public):**
- DiffSSD spoof: 10 generators, 70k clips total, of which the training sample uses 1k per generator:
  - LJ-voice TTS/vocoders: grad_tts, diffgan_tts, pro_diff, wavegrad2
  - multi-speaker: openvoicev2, unit_speech, xtts_v2, your_tts
  - commercial: elevenlabs, playht
- LJ Speech (real, **one speaker**): 6k in the sample.
- LibriSpeech dev+test (real, about 80 speakers): 4k in the sample.

**Input path (identical for train and test):**
1. Decode through FFmpeg to 16 kHz.
2. Trim silence (relative 35 dB).
3. Per-input z-normalization.
4. No tiling; training crops are drawn from the test-duration distribution; test-time uses the whole clip.

**Shortcuts found and neutralized:**
- **7.2 kHz band wall.** Every NSA test file is low-passed at about 7.2 kHz: the median energy share in 7.5–8 kHz is 1.6e-8, against 6e-4 to 6e-3 in LJ, LibriSpeech and DiffSSD. That band was a train-only cue. It's now removed by an identical Kaiser low-pass on train and test, in both the deep and engineered paths (since ~04:20). **All deep numbers below predate this fix** and are being re-measured.
- peak level (AUC 0.66 on its own)
- leading silence (LibriSpeech 0.37 s vs test 0.06 s)
- a tiling seam from padding short clips
- container format

**Folds** (`splits/nsa_folds.csv`, groups: spoof by generator; bona fide by speaker, or chapter for LJ):
- **Outer holdout:** playht + wavegrad2 + 26 bona fide groups (1,858 real / 2,000 spoof). Never used for training or selection.
- **Inner folds 0–4:** leave-generator-out on the spoof side.

**Leakage:** no exact file or PCM duplicates between the test set and LJ, the NSA LJ subset, or In-the-Wild.

## Current numbers

Normalized minDCF at π = 0.3 unless stated. Holdout = outer holdout.

| Detector | Inner-fold OOF minDCF | Holdout minDCF | Holdout EER | Sponsor code as-is / flipped |
|---|---|---|---|---|
| **M1:** frozen XLS-R 300M, layer 7 mean-pooled, logistic regression | 0.250 | **0.146** | 2.5% | 1.00 / 0.090 |
| Handcrafted: 75 spectral + prosody features, LightGBM | 0.614 | 0.255 | 4.3% | 1.00 / 0.127 |
| Compression forensics | 0.931 | 0.900 | 20% | 1.00 / 0.590 |
| Container/metadata, ENF, splice | 1.00 | 1.00 | 35–50% | no signal (inputs are uniformly decoded) |
| **Stack:** logistic on inner OOF of M1 + handcrafted | 0.250 | 0.146 | 2.5% | puts essentially all weight on M1 |
| **Mean** of z-scored logits, M1 + handcrafted (no fitting) | 0.258 | **0.066** | 1.3% | – / 0.043 |

**Detail:**
- M1 per held-out generator: playht 0.032, wavegrad2 0.214 (the LJ-voice vocoder is the hard one).
- M1 per bona fide source: **LJ 0.088 vs LibriSpeech 0.159**.
- Handcrafted per bona fide source: LJ 0.10 vs LibriSpeech 0.32. Unfamiliar real voices draw more false alarms.
- M1 layer sweep, inner CV: L6 0.284, **L7 0.250**, L8 0.282, L12 0.281, L20+ ≥ 0.40.

**M1 on the NSA test set:**
- Score distribution: 28% of files above 0.9 and 40% above 0.5 (NSA says about 30% synthetic).
- Preflight: **pure silence scores 0.9996 and a synthetic music chord scores 1.0**, i.e. confidently wrong on non-speech. The test set looks all-speech (median voiced fraction 0.8).

## Real-world stress test (In-the-Wild, evaluation only, never trained on)

The sample is 2,000 real clips from 54 speakers plus 1,000 fakes, web-sourced celebrity audio. "Inner threshold" is the threshold that minimizes inner-fold out-of-fold DCF.

| | M1 (NSA-only real) | M1b (+ASVspoof 2019 LA real: 40 VCTK speakers, 5.1k clips, plus a 2.5k A01–A06 anchor) |
|---|---|---|
| NSA holdout minDCF | 0.146 | 0.079 (better on every slice; inside holdout noise) |
| Inner-fold CV minDCF | 0.250 | 0.248 |
| **In-the-Wild minDCF** | **0.374** | 0.405 |
| In-the-Wild P_FA at the inner threshold | **1.2%** | 5.7% |

**Update ~05:40 (band-matched v3 and M3):**

| Detector | NSA holdout minDCF | In-the-Wild minDCF | In-the-Wild real P_FA | Test share > 0.5 (prior ~30%) |
|---|---|---|---|---|
| M1 v3 (band-matched) | 0.159 | 0.380 | 0.7% | 28.9% (v2 was 39.6%) |
| M1b v3 (band-matched, +VCTK) | 0.072 | 0.342 | 0.8% | 26.8% |
| **M3 Spectra-AASIST** (pretrained, frozen, band-matched) | pending (~06:00) | **0.065** | **0.0%** (0 of 2,000) | 26.5% (200-clip pilot) |
| Handcrafted v4 (234 features, LightGBM) | 0.170 | 1.00 (AUC 0.76) | 0.65% | 41% |

**Fusion v0 readout (05:32; rules pre-declared, nothing fit on holdout, In-the-Wild or test). Inputs: M1b v3 and handcrafted v4.** Each detector's logit is standardized on its inner out-of-fold rows. `zmean` = mean of standardized logits; `rankmean` = mean of per-detector ECDF ranks; `stack_nonlj` = logistic stacker fit on inner rows with LJ real speech excluded (one speaker on both sides of every fold), so its inner number is in-sample and not shown.

| Rule | Inner OOF | Holdout | Holdout LibriSpeech real | ITW minDCF | ITW real P_FA | ITW P_miss | Test share > 0.5 |
|---|---|---|---|---|---|---|---|
| M1b v3 alone | 0.301 (folds include the extra real rows) | 0.072 | 0.102 | 0.342 | 0.8% | 27.8% | 26.8% |
| handcrafted v4 alone | 0.434 | 0.170 | 0.217 | 1.00 | 0.65% | 97% | 56% |
| zmean | 0.259 | **0.018** | 0.022 | 0.387 | **0.15%** | 58.5% | **39.6%** |
| rankmean | 0.254 | **0.015** | 0.015 | 0.379 | **0.15%** | 59.5% | 31.8% |
| stack_nonlj | (in-sample) | 0.044 | 0.066 | **0.317** | 0.7% | 31.6% | 27.5% |

Reading: fusing the two cuts the holdout by 4× and In-the-Wild false alarms by 5×, but doubles In-the-Wild misses and pushes the test-set share of files called synthetic from 27% to 32–40%, against a 30% prior. The handcrafted column contributes almost no real-world detection (ITW P_miss 97%), so its In-the-Wild "gain" is a threshold effect. The stacker sits between.

**Diagnosis of the test-share rise (05:55).** The handcrafted column carries a residual train-to-test shift that band matching did not remove: the test recordings are darker below 7 kHz, codec-like. Its 234 columns tell train from test at AUC 0.99 (the 75 v3 columns alone: 0.97), and 84% of the shifted columns move in the "fake" direction, so under a 0.3-prior Platt map it calls 56% of the test set synthetic (v3: 49%; the deep models: 27%). No single feature family is responsible (leave-one-family-out: 54–57%), and pruning the shifted columns makes it worse (60–68%; the survivors still separate train from test at 0.98). Training-side augmentation as fit-only extra rows (codec round-trips, random tilt and low-pass; "v5b") helps partially: clean holdout 0.137 (v4 0.170), holdout-real false alarms at the calibrated threshold 6.3% (from 9.1%), In-the-Wild real P_FA 0.40% (from 0.65%), test share 51% (from 56%), at an inner-fold cost (0.418 → 0.469; pro_diff 0.30 → 0.63). So the fusion rules' test-share rise is this column's offset, not the deep models'. The remaining remedy on the table is re-centering that detector's logit on the unlabeled test distribution before fusion, or down-weighting it.

**Spectra-AASIST's training data is undisclosed.** Its card reports 1.46% EER on In-the-Wild, and it separates our NSA inner rows perfectly (AUC 1.0). So its In-the-Wild and inner-fold numbers may be in-sample. Band matching fixed the test-distribution shift but left the M1 In-the-Wild gap unchanged. **New question: how should we weight a detector whose validation rows may be in its training data?**

- **Both models are 3–5× worse on real-world audio** than on the NSA-domain holdout. The brief says the test's real audio may include smartphone, telephony and field recordings.
- **Adding VCTK read speech did not help on In-the-Wild.** An earlier consult predicted it would.
- Part of the gap may be the 7.2 kHz band mismatch (In-the-Wild isn't low-passed either); the re-measurement will show.

## What I want (ranked)

For each question: your recommendation, a concrete procedure we can run in under 1 hour of CPU, and EV × effort.

1. **Selection under disagreement.** The inner folds say the M1 + handcrafted mean is slightly worse than M1 (0.258 vs 0.250). The holdout says it is much better (0.066 vs 0.146). The logistic stacker trusts the inner OOF and zeroes out the handcrafted detector.
   - What is the principled choice?
   - What is the likely reason for the disagreement?
     - The inner folds are leave-one-generator-out on 8 generators.
     - The holdout has only 2 generators.
     - The handcrafted inner OOF is weak on grad_tts and pro_diff (1.0).
   - Would rank averaging, per-fold weight stability, a constrained stacker (non-negative weights, a shrink toward equal weights) or nested CV of the fusion rule resolve it without peeking at the holdout?
   - Is it acceptable to use the one-time draft review to choose between two candidates? Which two should we send?
   - **With the fusion v0 table above:** the unfitted rules (zmean, rankmean) win the holdout and In-the-Wild false alarms by a wide margin but raise In-the-Wild misses from 28% to 59% and the test share above 0.5 from 27% to 32–40%. The stacker fit on non-LJ rows is in between on everything. Which of the four do we ship at 14:00? The rising test share is now diagnosed as the handcrafted column's domain offset (above): is re-centering one detector's logit on the unlabeled test set's own mean and spread legitimate under our no-pseudo-label rule, and if so, per column or only for columns with a measured train-to-test shift?
   - **Weighting a pretrained detector whose validation rows may be in-sample.** M3 (Spectra-AASIST) separates our inner rows perfectly (AUC 1.0) and reports In-the-Wild real P_FA 0.0%, but its training data is undisclosed and LJ, LibriSpeech, DiffSSD and In-the-Wild are all public. A stacker fit on inner rows will give it all the weight. How should its weight be set: holdout only, rank mean with equal weight, a cap, or excluded from stacking and used only as a gate?
2. **The "default answer" / game-theory hint.**
   - With a rank-based MinDCF at π = 0.3 and C_FA = 4, where should files we "cannot determine" sit in the ranking? Examples: decode failures, non-speech, detector disagreement, low confidence.
   - Is there a smarter policy than the prior?
     - Push uncertain files to the real end, because false alarms dominate.
     - Break ties deliberately.
     - Account for the minDCF threshold sweep.
   - How does the answer change if NSA actually runs their shipped code unmodified (a higher score treated as bona fide, Pspoof = 0.5)?
   - Is there any hedge that is safe under both directions, or is the draft review the only real protection?
3. **Channel robustness: false alarms on real-world real speech.** This is now the top question. Our real training data is one LJ speaker, about 80 LibriSpeech readers, and optionally 40 VCTK speakers, all clean read speech. In-the-Wild shows 3–5× worse minDCF, and more read-speech speakers didn't help. The test's real audio may include smartphone, telephony and field recordings.
   - What is the cheapest effective lever in the next ~20 hours, using In-the-Wild real-speech P_FA as the proxy?
     - channel augmentation on both classes: telephony band-limit, codecs, RIR, noise, RawBoost convolutive
     - score normalization
     - test-statistics standardization or CORAL on the unlabeled test embeddings
     - which bona fide corpora actually add channel variety (not just more read speech)
     - center-loss or one-class objectives (another consult warned these are contraindicated with narrow bona fide data)
   - What is safe to do with the unlabeled test set? We have ruled out pseudo-labeling.
   - Is checking whether test real clips are LJ-like (nearest-neighbor to LJ embeddings) useful or a trap?
4. **Explainability and the diversity score.** Our engineered detectors carry no signal because every test file is identical in format. Should we:
   - keep them in the orchestrator anyway (to cover the rubric),
   - replace them with techniques that work on decoded 16 kHz audio (e.g. speaker-embedding drift within a clip, breath/pause statistics, formant or phase features), or
   - just document them honestly as "no effect"?
   - What would a judge value for "agentic orchestration" beyond rule-based routing?
5. **The non-speech failure.** Silence and music score about 1.0. What is the right gate (a VAD threshold on voiced fraction, a speech/music classifier), and what score should gated files get, given question 2?
6. **Time allocation.** About 26 h remain (23 to the final TSV), and one person owns the deep models, fusion, TSV and the draft review; a fine-tune, a pretrained-detector scoring run and the engineered detectors run in parallel lanes; Docker is unowned. What would you cut, and what single change do you expect to move holdout minDCF the most?

## Hard lines

- No training or selection on the outer holdout.
- Never train on In-the-Wild.
- No pseudo-labeling of the test set.
- Output 1.0 = synthetic (instructions).
- Offline CPU inference in Docker, no hosted APIs.
- Training on external datasets is allowed by the brief.

## Output format

1. A one-line **verdict** on our current direction.
2. A **ranked actions table**: action | expected minDCF impact | effort | risk | do / maybe / skip.
3. For each of the six questions: a recommendation with concrete numbers and one runnable procedure.
4. The **draft-review plan**: which TSV(s) to send, what to learn from the returned score, and how to act on it.
5. **Failure modes we haven't named**, especially around the 4× false-alarm cost and the sponsor-code direction.
