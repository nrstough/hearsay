# Consult: after the draft review — what the number means, what heads we are missing, what to train in the last hours (Sat Sep 26, 2026, drafted ~12:15 EDT; send once NSA returns the draft minDCF)

**Target:** a frontier LLM (VeriLM "Deep Think", two experts + synthesis), for strategic and technical opinion.
**Scope:** the whole system, from the returned test number to Sunday 08:00. Fusion selection was settled by the previous consult (`2026-09-26_fusion-strategy_CONSULTATION.md` / `_RESPONSE.md`); do not reopen it unless the number forces it.
**Blocking?** Yes for the afternoon plan: the answer decides whether the remaining hours go to a new detector head, a refit, or documentation.
**Fill in before sending:** every `___` below (the returned minDCF, the time sent, anything NSA said with it).

Copy below the line.

---

## Your role

You are an applied researcher in audio deepfake detection and detection-cost evaluation (ASVspoof, NIST-style DCF), and you have built systems under a deadline. Push back hard on any wrong premise, including ours about what the returned number means. We have about ___ hours left (final file due Sunday 08:00 EDT; it is now ___ Saturday). Practical beats ideal; a runnable experiment beats a literature pointer.

## The problem

- **Task.** The NSA "HEARSAY" challenge at HackGT 13: output P(synthetic) per audio file, 1.0 = synthetic, on 1,671 unlabeled 16 kHz mono WAVs (3.0–13.6 s, median 3.4 s, English). NSA confirmed in person today: **about 70% real**, the "analyst scenario", ASVspoof 5 track-1 DCF with a **false alarm costing 4× a miss**, and the score runs "from 0.0 for 100% confident real to 1.0 for 100% confident synthetic".
- **Judging.** 60% minDCF (points = (1 − ours)/(1 − best team) × 60); 20% diversity and depth of techniques (container/metadata, spectral, prosody, ENF, compression, speaker-embedding consistency, deep anti-spoofing, splice, plus a bonus for choosing analyses per file); 20% documentation in our own words, including what did not work and explainability.
- **Deliverable.** The TSV and the GitHub README only. NSA said today the Docker image is **not required** (we built one anyway; it is evidence, not a deliverable).
- **The one measurement we get.** NSA offers a one-time review of a draft TSV. We sent ours at ___ today and it returned **minDCF = ___** (nothing else: no EER, P_FA or P_miss). Everything below is what we knew before that number; the first ask is what the number tells us.
- **Selection metric on our side.** Normalized minDCF at π_synth = 0.3, C_FA = 4, C_miss = 1: **9.33·P_FA + P_miss**, swept over thresholds. We also report the sponsor script's own cost (its shipped code has Pspoof = 0.5 and treats a higher score as bona fide, which moves the 4× onto misses; "miss-averse" below). Only the ranking of our scores matters to either.

## What we shipped in the draft (fixed unless the number says otherwise)

**Rule "A3 w0.2 + E"** (`models/fusion_v2/constants.json`), chosen by a selection rule written before its results and ratified at 12:05:
1. Rank each of three detector logits against its own inner-fold out-of-fold distribution (ECDF).
2. `base = 0.6·rank(M1b v3) + 0.2·rank(handcrafted v5) + 0.2·rank(M5)`.
3. If Spectra-AASIST's margin < −3 ("strongly bona fide") and `base > 0.5`, `base *= 0.5`. Spectra can only pull a file toward real; nothing is fitted on it (its training data is undisclosed).
4. Platt map with a prior shift to 0.3 into [0.001, 1].
5. Files the speech gate rejects (non-speech, decode failure) are pinned strictly below every determinate score, in [0, 0.001), ordered by residual signal. This is the "default answer": it is where the 4× error is avoided under both readings of the score. 0 of 1,671 test files are gated.

**The three fused detectors and the suppressor:**

| Stream | What it is | Inner OOF | Holdout | In-the-Wild (brief / miss-averse) | ITW real P_FA at inner thr |
|---|---|---|---|---|---|
| M1b v3 | frozen XLS-R 300M, layer 7, mean-pooled, logistic regression; trained on NSA sample + ASVspoof19 LA bona fide (40 VCTK speakers) + a 2.5k A01–A06 anchor; band-matched input | 0.301 | 0.072 | 0.343 / 0.296 | 0.8% |
| handcrafted v5 | 234 spectral/prosody/LFCC/CQCC/phase/modulation/breath/jitter features, LightGBM, trained with augmented fit-only rows (codec round-trips, tilt, low-pass) | (v4 0.434) | 0.137 | 1.00 alone (AUC 0.76; contributes false-alarm control, not detection) | 0.65% |
| M5 | XLS-R 300M truncated to 12 layers, frozen, learned layer weights + attentive stats pooling + linear head; trained on vast.ai with laundering augmentation, MLAAD spoof capped at 12%, extra LJ/LibriSpeech real | 0.302 | 0.363 (short clips ≤ 4 s: 0.60; LibriSpeech real 0.43) | (alone) — | 0.45% |
| Spectra-AASIST (M3) | off the shelf, frozen, band-matched | separates inner rows perfectly (possibly in-sample) | 0.012 | 0.065 | 0.0% |

**The fusion sweep (pre-declared candidates, rule fixed before results):**

| Candidate | Inner | Holdout (brief) | Holdout [#FA, #miss] of 3,858 | ITW brief | ITW miss-averse | ITW P_FA / P_miss at inner thr |
|---|---|---|---|---|---|---|
| M1b v3 alone | 0.301 | 0.072 | [8, 63] | 0.343 | 0.296 | 0.8% / 27.8% |
| E on A α0.2 (previous shipped rule: 0.8 M1b + 0.2 handcrafted, M3 suppression) | 0.140 | 0.014 | [1, 18] | 0.260 | 0.267 | 1.4% / 14.8% |
| F: M5 as a second suppressor on top | 0.136 | 0.014 | [1, 18] | 0.258 | 0.268 | 1.4% / 14.4% |
| A3 w0.2 without the M3 step | 0.213 | 0.020 | [1, 30] | 0.274 | 0.251 | 0.3% / 32.4% |
| **A3 w0.2 + E (shipped)** | **0.135** | **0.0065** | **[0, 13]** | **0.228** | **0.239** | 1.3% / 12.2% |

The In-the-Wild column for M5-containing rules is crop-fair (M5 scored on the same test-length crops as the others; whole clips had flattered it by 0.013). Between the previous rule and the shipped one, the test-set ranking moved modestly: Spearman 0.974, 3 files cross 0.5, 31 cross 0.9 (22 down, 9 up), 21 files swap in and out of the top 30%.

**Router ablation (holdout / In-the-Wild, brief cost), measured on the previous rule:** router on 0.014 / 0.258; no M3 suppression 0.030 / 0.323; "fuse everything" (M3 inside the blend) 0.0085 / 0.276. So the suppression step is worth 2× on the holdout and 0.06 on In-the-Wild; putting M3 inside the blend helps the in-family holdout and hurts the wild set, which is why it is suppression-only.

## Data, validation and what we already know about the test set

- **Training data:** DiffSSD (10 TTS generators: grad_tts, diffgan_tts, pro_diff, wavegrad2, openvoicev2, unit_speech, xtts_v2, your_tts, elevenlabs, playht), LJ Speech (one real speaker), LibriSpeech dev/test (~80 real speakers), plus for M1b/M5 only: ASVspoof 2019 LA bona fide (VCTK), MLAAD spoof (M5), extra LJ/LibriSpeech (M5). Never trained on: our outer holdout (playht + wavegrad2 + 26 real speaker groups), In-the-Wild (2,000 real / 1,000 spoof, web audio), the test set.
- **Folds:** spoof grouped by generator, real by speaker; inner folds leave one generator out; the outer holdout is never used for training or selection.
- **Test-set facts (label-free):** every file is 16 kHz PCM WAV written by the same FFmpeg build (encoder tag `Lavf58.29.100`); every file is low-passed at about 7.2 kHz (no training corpus is); the envelope below 7 kHz is darker and codec-like across nearly every handcrafted column; median voiced fraction 0.8, no silence or tones; 3.0–13.6 s. We band-match every input (train and test) with a 71-tap Kaiser low-pass at the same edge. Our label-free guess is that the test set went through one curated re-encode pipeline rather than being heterogeneous field audio; the wild-domain weight λ is being estimated this afternoon.
- **Shortcuts already removed:** the 7.2 kHz band, peak level, leading silence, a tiling seam, container format.
- **Our two proxies disagree by 20×:** holdout 0.0065 (in-family generators, clean read speech) vs In-the-Wild 0.228 (real-world audio). The returned number sits somewhere between these regimes, and reading which is ask 1.

## What is established or dead (do not re-suggest)

- Equal-weight fusion (z-mean, rank-mean): doubles In-the-Wild misses. Rejected.
- M3 inside a fitted stacker: forbidden (undisclosed training data). Suppression-only is its ceiling.
- Container/metadata, ENF, splice as score streams: dead on this test set by mechanism (uniform FFmpeg decode erases container evidence; 16 kHz PCM with a 7.2 kHz wall and re-encoding leaves no usable mains component or splice discontinuity). They run and produce evidence and routing facts; they are documented as such.
- Speaker drift (ECAPA windows): evidence only; the finding is that fakes are the *most* consistent voices. Compression forensics: evidence only.
- "Share of test files above 0.5" as a selection signal: monotone maps cannot change minDCF; kept as a smoke alarm (27.3% now, prior 30%).
- Nearest-neighbour-to-LJ labelling, one-class objectives, more handcrafted feature engineering, holdout tuning below 0.05: all rejected in the previous consult with reasons we accept.
- Loudness, silence and duration as features: removed as shortcuts; they will not come back.

## In flight this afternoon (do not duplicate; do tell us if any is a waste)

A channel-robustness lane on the frozen deep path, time-boxed to 16:00: (A) λ̂, the test set's wild-domain weight from label-free channel statistics; (B) diagnosing the 7.2 kHz edge as a specific codec by matching roll-off and hole statistics against LJ clips laundered over an MP3/AAC grid; (C) if B matches, a symmetric codec round-trip refit of M1b (both classes, never one); (D) one wild non-read bona fide corpus into inner folds only if it downloads in 30 minutes; (E) three M3 leakage probes (perturbation sensitivity vs M1b, holdout-vs-test rank agreement, MLAAD spoof through M3). Any new column goes through the same pre-declared sweep. Constraint at the moment: the external drive holding the training corpora is unplugged until ~___, so (C) and (D) may slip.

## Resources

One Apple M3 Pro (18 GB; the runner does 1,671 files in ~28 min CPU; XLS-R embedding extraction ~22 min), optional vast.ai A100 at ~$0.7/h (spent $5.42 so far; tens of dollars more is fine), the corpora above on a local drive, and about ___ hours of wall clock with four parallel Claude Code sessions. Weights on disk: XLS-R 300M, WavLM Large and Base (both unused so far), Spectra-AASIST, ECAPA-TDNN. No hosted model may sit on the scoring path.

## What I want (ranked)

For each: your recommendation, a procedure we can run inside the time it deserves, and an EV × effort estimate in minDCF points and rubric points.

1. **Read the number.** Returned minDCF = ___. Against holdout 0.0065 and In-the-Wild 0.228: which regime is the test set in, and how confident can we be from one number on 1,671 files (about 500 synthetic)? Our pre-declared reaction table: 0.00–0.20 ship unchanged and spend the hours on documentation; 0.20–0.45 our direction but wild-like, so channel work matters; 0.45–0.90 ambiguous, fall back to M1b alone, do not flip; 0.95–1.00 the sponsor's script read our scores inverted, so ship the pre-flipped file. Is that table right, and does the number move anything else (the prior shift, the suppression threshold, the block placement)?
2. **Missing heads.** Given what the test set is (uniform re-encode, 7.2 kHz wall, short read-like speech, ~30% synthetic from unknown generators), which score streams are we most likely missing that could be built and validated in 3–6 hours on the hardware above? Candidates we can think of: a WavLM-based probe (weights are on disk, never tried) as a second SSL view; a probe on a different XLS-R layer or a multi-layer concatenation; a codec-artifact head trained on laundered pairs; a prosody/duration-normalized F0 head; a per-clip "generator family" classifier used as routing. For each you endorse: the training recipe under our fold discipline, the expected gain on the wild regime specifically, and how it enters fusion (it must go through the same pre-declared sweep, so say what weights you would pre-declare).
3. **Train or document?** The remaining rubric mass is 40 points of diversity and documentation against a minDCF share where the shipped rule is already within noise of our proxies' floor. Given the number, how would you split the hours between (a) the channel refit lane, (b) a new head from ask 2, (c) the README (worked examples, router-on/off ablation, honest negative results, disclosure), and (d) nothing (stop touching the file)? Be blunt about (d).
4. **NSA's spectral-features talk.** One of us attends a sponsor talk on spectral features for synthetic-speech detection this afternoon. What should we ask, and what should we listen for, that would change our system? Our current beliefs: magnitude features (LFCC, CQCC, modulation, band statistics) carry most of the signal; phase and group-delay features are fragile under re-encoding; everything above the test set's 7.2 kHz wall is unusable; the train/test channel gap is our largest weakness. Give us five questions in plain words and, for each, what answer would make us change something.
5. **Anything we have wrong.** You have the whole picture above. Where is the premise error, if there is one?

## Additional context

- What already works and sets the bar: the shipped rule reproduces from audio end to end (runner parity Spearman 1.0, max |Δp| 9e-4 on 1,671 files) and beats every simpler rule on every proxy under both costs. A new head must clear that on the pre-declared sweep, or it goes into the README as a negative result.
- We can A/B anything on the holdout and In-the-Wild in minutes from exported logits; anything needing new embeddings costs 22 minutes of CPU per corpus, or an A100 rental.
- Hard lines: no training or selection on the holdout, In-the-Wild or the test set; no pseudo-labels; augment both classes or neither; the block stays at the bottom; 1.0 = synthetic unless the returned number decodes to inversion; final file frozen by Sunday 05:00.

## Output format

1. One-line verdict on the returned number (regime, confidence).
2. A ranked table of options: approach / expected minDCF effect in the wild regime / expected rubric effect / effort in hours / verdict (do, maybe, skip).
3. For each "do": a runnable procedure with the fold discipline spelled out and the pre-declared fusion entry.
4. The five questions for the NSA talk with the answer that would change our plan.
5. A direct call on "are we over-engineering; should we stop touching the file?", no hedging.
