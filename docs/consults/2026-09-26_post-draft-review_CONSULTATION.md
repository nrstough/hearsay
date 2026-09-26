# Consult: after the draft review — the returned number is fixed; what do we build in the last 13 hours? (Sat Sep 26, 2026, drafted ~12:15 EDT, rewritten ~15:40 with NSA's returned numbers)

**Target:** a frontier LLM (VeriLM "Deep Think", two experts + synthesis), for strategic and technical opinion. Third consult; the first two are `2026-09-26_m5-extra-data-finetune_*.md` and `2026-09-26_fusion-strategy_*.md`.
**Scope:** everything that could move the test-set number before the Sunday 05:00 freeze: new heads, new engineered features, a better fusion layer. The score direction and the default-answer policy are settled by the returned number and are not reopened.
**Blocking?** Yes for the evening plan: the answer decides where four parallel sessions spend the night.

Copy below the line.

---

## Your role

You are an applied researcher in audio deepfake detection and detection-cost evaluation (ASVspoof, NIST-style DCF), and you have shipped systems under a deadline. Push back hard on any wrong premise, including ours about what the returned number means. It is Saturday 15:40 EDT; the final file freezes Sunday 05:00 (about 13 hours) and is handed in by 08:00. Practical beats ideal; a runnable experiment beats a literature pointer. We can A/B most ideas on exported scores in minutes and can rent an A100 for a few dollars.

## The problem

- **Task.** The NSA "HEARSAY" challenge at HackGT 13: output P(synthetic) per audio file, 1.0 = synthetic, on 1,671 unlabeled 16 kHz mono WAVs (3.0–13.6 s, median 3.4 s, English). NSA confirmed in person: **about 70% real**, the "analyst scenario", ASVspoof 5 track-1 DCF with a **false alarm costing 4× a miss**, score from 0.0 (confident real) to 1.0 (confident synthetic).
- **Judging.** 60% minDCF (points = (1 − ours)/(1 − best team) × 60); 20% diversity and depth of techniques (container/metadata, spectral, prosody, ENF, compression, speaker-embedding consistency, deep anti-spoofing, splice, plus a bonus for choosing analyses per file); 20% documentation in our own words, including what did not work and explainability.
- **Deliverable.** The TSV and the GitHub README. The Docker image is not required (NSA, Saturday noon); we built one anyway as evidence.
- **Selection metric on our side.** Normalized minDCF at π_synth = 0.3, C_FA = 4, C_miss = 1: **9.33·P_FA + P_miss**, swept over thresholds. We also report the sponsor script's own cost (its shipped ASVspoof5 code has Pspoof = 0.5 and treats a higher score as bona fide, which puts the 4× on misses: normalized **P_FA + 4·P_miss**; "miss-averse" below). Only the ranking of our scores matters to either.

## The one measurement we get, and it is now in hand

We sent our draft TSV at 12:30 (the exact artifact we intend to ship, gate and pinned block included). **NSA returned: minDCF = 0.0733, EER = 3.534%.** No P_FA, P_miss or threshold, and NSA did not say which cost weighting or which score-direction convention produced the number.

This number is fixed. It is the only labeled readout of the test set we will ever get, so we treat it as ground truth about the shipped file and as the baseline every change must beat by reasoning, not by re-measurement. Our two proxies for the same file: **outer holdout 0.0065** (in-family generators, clean read speech) and **In-the-Wild 0.228** (web audio, never trained on). The test set landed at 11× the holdout and a third of In-the-Wild, next to what our best single detector scores on the holdout (M1b v3 alone: 0.072).

What we read from it so far, for you to confirm or refute:
- **Direction is ours.** Under the sponsor's code as shipped (higher = bona fide) our file would have scored 1.00; 0.0733 means they read 1.0 = synthetic, as the instructions say. EER 3.5% confirms it (an inverted file reads 96.5%).
- **Regime.** By our pre-declared table (0.00–0.20: our direction, the test set behaves like the holdout), the shipped rule stays and no flip or fallback is triggered.
- **Error profile is ambiguous.** With about 1,170 real and 500 synthetic files: under the brief's cost 0.0733 is, for example, P_FA 0.3% (3–4 real files) plus P_miss 4.5% (about 22 fakes), or 0 false alarms plus 37 missed fakes. Under the sponsor code's cost it is, for example, P_FA 1.3% plus P_miss 1.5%. EER 3.5% says that at the equal-error point about 41 real and 18 fake files are on the wrong side, so the ranking has a tail of hard files on both sides.

## What we shipped in the draft (the baseline; fixed unless you argue otherwise)

**Rule "A3 w0.2 + E"** (`models/fusion_v2/constants.json`), chosen by a selection rule written before its results and ratified at 12:05:
1. Rank each of three detector logits against its own inner-fold out-of-fold distribution (ECDF).
2. `base = 0.6·rank(M1b v3) + 0.2·rank(handcrafted v5) + 0.2·rank(M5)`.
3. If Spectra-AASIST's margin < −3 ("strongly bona fide") and `base > 0.5`, `base *= 0.5`. Spectra can only pull a file toward real; nothing is fitted on it (its training data is undisclosed).
4. Platt map with a prior shift to 0.3 into [0.001, 1].
5. Files the speech gate rejects (non-speech, decode failure) are pinned strictly below every determinate score, in [0, 0.001). 0 of 1,671 test files are gated.

**The three fused detectors and the suppressor:**

| Stream | What it is | Inner OOF | Holdout | Holdout EER | In-the-Wild (brief / miss-averse) | ITW real P_FA at inner thr |
|---|---|---|---|---|---|---|
| M1b v3 | frozen XLS-R 300M, layer 7, time-mean, StandardScaler + class-balanced logistic regression + Platt; trained on the NSA sample + ASVspoof19 LA bona fide (40 VCTK speakers) + a 2.5k A01–A06 spoof anchor, inner folds only; band-matched input | 0.301 | 0.072 | 1.4% | 0.343 / 0.296 | 0.8% |
| handcrafted v5b | 234 spectral / prosody / LFCC / CQCC / phase / modulation / breath / jitter features on the band-matched, trimmed, RMS-normalized clip; LightGBM (300 trees, 31 leaves); trained with augmented fit-only twins (codec round-trips, ±4 dB/kHz tilt, 4.5–7 kHz low-pass) | 0.469 | 0.137 | 3.1% | 1.00 alone (AUC 0.77; it orders In-the-Wild fakes but its threshold does not transfer) | 0.40% |
| M5 | XLS-R 300M truncated to 12 frozen layers, learned softmax layer weights (peak at layers 5–6) + masked attentive statistics pooling + linear head, 136k trainable params; trained on an A100 with class-blind laundering augmentation, MLAAD spoof capped at 12%, extra LJ/LibriSpeech/VCTK real; the full 12-layer fine-tune lost to this on the unseen ElevenLabs fold (0.38–0.41 vs 0.31) | 0.302 | 0.363 (clips ≤ 4 s: 0.60; LibriSpeech real 0.43) | 6.0% | — | 0.45% |
| Spectra-AASIST (M3) | `lab260/Spectra-AASIST`, off the shelf, frozen, band-matched, zero-padded to one 4.04 s window | 0.0045 (possibly in-sample) | 0.012 | 0.16% | 0.065 / — | 0.0% |

**The pre-declared sweeps (rule fixed before results; the In-the-Wild column for M5 rules is crop-fair):**

| Candidate | Inner | Holdout (brief) | Holdout [#FA, #miss] of 3,858 | ITW brief | ITW miss-averse | ITW P_FA / P_miss at inner thr |
|---|---|---|---|---|---|---|
| M1b v3 alone | 0.301 | 0.072 | [8, 63] | 0.343 | 0.296 | 0.8% / 27.8% |
| A α0.2 (0.8 M1b + 0.2 handcrafted, no M3 step) | 0.236 | 0.030 | [3, 30] | 0.322 | 0.280 | 0.45% / 36.0% |
| E on A α0.2 (previous shipped rule) | 0.140 | 0.014 | [1, 18] | 0.260 | 0.267 | 1.4% / 14.8% |
| F: M5 as a second suppressor on top | 0.136 | 0.014 | [1, 18] | 0.258 | 0.268 | 1.4% / 14.4% |
| A3 w0.2 without the M3 step | 0.213 | 0.020 | [1, 30] | 0.274 | 0.251 | 0.3% / 32.4% |
| **A3 w0.2 + E (shipped; NSA: 0.0733 / EER 3.5%)** | **0.135** | **0.0065** | **[0, 13]** | **0.228** | **0.239** | 1.3% / 12.2% |
| Four-way equal rank mean incl. Spectra (ruled out before measuring) | — | 0.0045 | — | 0.214 | 0.196 | — |

**Router ablation on the shipped rule (holdout / In-the-Wild brief):** router on 0.0065 / 0.229; no M3 suppression 0.020 / 0.274; the suppression touched 83 holdout and 24 In-the-Wild files, every one of them real. The gate touched 0 test files. The equal four-way mean beats us on every proxy by giving Spectra a full vote; we refused that before seeing the numbers because its training data is undisclosed and our validation rows may be in it.

**Reproducibility:** the runner reproduces the shipped TSV live from audio on all 1,671 files (Spearman 1.0, max |Δp| 9.3e-4, about 28 minutes of Mac CPU). Any new column enters through the same export format and the same kind of pre-declared sweep.

## Data, validation and what we know about the test set

- **Training data:** DiffSSD (10 TTS generators: grad_tts, diffgan_tts, pro_diff, wavegrad2, openvoicev2, unit_speech, xtts_v2, your_tts, elevenlabs, playht), LJ Speech (one real speaker), LibriSpeech dev/test (about 80 real speakers); for M1b and M5 only, ASVspoof 2019 LA bona fide (VCTK) and, for M5, MLAAD spoof and extra LJ/LibriSpeech. Never trained on: the outer holdout (playht + wavegrad2 + 26 real speaker groups, 3,858 rows), In-the-Wild (2,000 real / 1,000 spoof), the test set.
- **Folds:** spoof grouped by generator, real by speaker (LJ by chapter); five inner folds leave whole generators out; every learned piece (scalers, Platt maps, trees, fusion references) is fit fold-locally.
- **Test-set facts (label-free):** every file is 16 kHz PCM WAV written by the same FFmpeg build; every file is low-passed at about 7.2 kHz (no training corpus is); below 7 kHz the envelope is darker and codec-like across nearly every handcrafted column (train-vs-test AUC 0.99 on the 234 features); median voiced fraction 0.8; no silence or tones.
- **Channel-robustness lane, finished at 13:20 (all diagnostic, nothing shipped):**
  - λ̂, the test set's wild-domain weight from label-free channel statistics: **0.51 (95% CI 0.12–0.64)**; the test set is between our clean holdout and In-the-Wild on about half the channel axes and outside both on the rest (beyond the wild end on noise-floor stationarity and flatness; beyond the clean end on loudness spread). It is its own recording population.
  - The 7.2 kHz wall is a smooth resampling low-pass, **not** MP3 or AAC (34 codec variants tried; the nearest was 3% closer than our own Kaiser filter where the rule needed 20%). So the symmetric codec refit of M1b did not run.
  - M3 memorisation probes: M3 is the most perturbation-stable detector by AUC (mean |ΔAUC| 0.0008 vs M1b 0.0041) but has the least stable rank order; lowest miss rate on 572 MLAAD clips from 143 unseen generators (14.9% vs M1b 22.7%, M5 28.7%, handcrafted 56.1%); the suppression step fired on no spoof clip under any perturbation and never raised the fused minDCF. Kept.
  - **New weak spot: additive noise.** At 20 dB SNR white noise, handcrafted v5 collapses (AUC 0.998 → 0.64) and fakes turn into misses: M1b's holdout minDCF 0.056 → 0.271, the shipped rule's 0.012 → 0.269. M3 is best under noise (0.116) but its step never fires there. MP3, ±2% speed and a one-sample shift move nothing.
- **Shortcuts already removed:** the 7.2 kHz band, peak level, leading silence, a tiling seam, container format. A "double low-pass stopband" (7–8 kHz after our filter) was found and measured: it moves no detector.

## What is established or dead (do not re-suggest)

- Equal-weight fusion of M1b and handcrafted (z-mean, rank-mean): doubles In-the-Wild misses. Rejected.
- M3 inside a fitted stacker or with a full vote: forbidden by our own rule (undisclosed training data). Suppression-only is its ceiling unless you can argue the returned number changes that.
- Container/metadata, ENF, splice as score streams: dead on this test set by mechanism (one FFmpeg decode erases container evidence; 16 kHz PCM with a 7.2 kHz wall leaves no usable mains component; the sponsor says no clip is partially synthetic). They run for evidence and routing and are documented as such.
- Speaker drift (ECAPA windows): evidence only; fakes are the *most* self-consistent voices (AUC 0.73 toward fake). Compression forensics: evidence only (it reads the pipeline, holdout 0.90).
- Full fine-tuning of the XLS-R layers: fit the seen generators, transferred worse to ElevenLabs than a frozen backbone. Head-only won.
- Pruning corpus-cue or domain-shifted handcrafted columns: worse every time (the survivors still separate train from test at AUC 0.98).
- Peak-normalizing Spectra's input; the WavLM weights (Large and Base are on disk, never run).
- "Share of test files above 0.5" as a selection signal; nearest-neighbour-to-LJ labelling; one-class objectives; holdout tuning below 0.05.
- We did not attend or record the sponsor's spectral-features talk, so nothing from it is in our system.

## Resources

One Apple M3 Pro (18 GB): the full runner does 1,671 files in about 28 minutes; XLS-R embedding extraction for the 20,000-row fold file is about 22 minutes per layer set; the handcrafted extraction about 17 minutes. Optional vast.ai A100 at about $0.7/h (spent $5.42 so far; tens of dollars more is fine; the M5 trainer, bundle and cloud scripts exist and a fold model trains in 12 minutes). Weights on disk: XLS-R 300M, WavLM Large and Base, Spectra-AASIST, ECAPA-TDNN. Exported per-detector score files for every row (inner OOF, holdout, test, In-the-Wild) let us A/B any fusion idea in seconds. Four parallel Claude Code sessions; one person (Nathan) owns the TSV and every ship decision. No hosted model may sit on the scoring path.

## What I want (ranked)

For each: your recommendation, a procedure we can run inside the time it deserves under our fold discipline, and an EV × effort estimate in minDCF points on the test set specifically, with the reasoning that connects it to the returned 0.0733.

1. **Read the number.** minDCF 0.0733 and EER 3.534% on the shipped file. Which regime is the test set in, and what error profile is most likely (a handful of false alarms and a tail of misses, or the reverse)? Does the pair (minDCF, EER) tell us whether NSA used the brief's cost or their script's cost? What do you infer about the number of files that separate us from, say, 0.03, and about whether the remaining errors are unseen-generator misses (where M3 and M5 are strongest) or real-speech false alarms (where the handcrafted column and M1b's short-clip behaviour are the suspects)? Say plainly whether one number on 1,671 files can support any of the changes below or whether we are now optimizing proxies only.

2. **More and different heads on the SSL backbones.** We have one frozen probe (layer 7, logistic) and one frozen-backbone head (12-layer, attentive pooling). Candidates, in our order of guessing: (a) a WavLM Large probe on the same recipe as M1b, as a second SSL view with different pretraining; (b) an M1b variant on a multi-layer concatenation or a learned weighting of layers 5–9 instead of layer 7 alone (embeddings for all 25 layers of XLS-R exist, so this is minutes); (c) averaging M5's five fold models with the full model at test time (free, variance reduction); (d) an attentive-pooling or statistics-pooling head on the frozen layer-7 frames instead of the time-mean (M5's pooling on M1's layer); (e) a probe trained on noise-augmented embeddings, given the noise weak spot. For each you endorse: the training recipe under our folds, the expected gain on a test set that scored 0.0733, how it enters fusion, and what weights you would pre-declare. Which two would you run first, and is any of them a waste?

3. **More or different engineered features.** The handcrafted column is 20% of the vote and is the weakest under noise and under the test set's envelope shift; it is also the only column a judge can read. The previous memo said "no further points there"; the returned number lets us revisit. Candidates: (a) noise-augmented twins in training (symmetric, like the codec twins), aimed at the 20 dB weak spot; (b) a delta/dynamics-only variant that drops the envelope means the test shift lives in (we tried pruning shifted columns and it hurt; we have not tried a model trained on deltas alone); (c) families we have not built: long-term average spectrum residuals against a real-speech template, vocoder-harmonic and inter-harmonic noise ratios, formant bandwidth and trajectory statistics, sub-band modulation of the 4–7 kHz band, bispectral or phase-cyclostationarity cues; (d) a "generator-family" multi-class head whose output is used as routing rather than as a score. For each: what it measures that XLS-R does not, the expected gain given that the test set is band-limited to 7.2 kHz and re-encoded, and the two-hour version of the experiment.

4. **A more elegant fusion layer.** The shipped rule is a fixed-weight rank blend chosen from six candidates, a one-way suppression step and a Platt map. What would you replace or extend it with that is admissible under our rules (no M3 inside a fitted model; nothing fit on holdout, In-the-Wild or test; every reading pre-declared)? Options we can see: a non-negative constrained stacker over the three admissible ranks fit on non-LJ inner rows; per-detector isotonic calibration before blending; a mixture-of-experts gated by measured file properties (voiced fraction, bandwidth, noise-floor level, clip length) so that, for example, M5 gets more weight on the short clips where M1b is weaker; a cascade that trusts M1b alone outside its ambiguous band; a Bayesian combination that treats M3's margin as a likelihood with a capped weight; a second suppression threshold (M3 margin < −3 halves; < −6 quarters). Which of these has a mechanism that could plausibly move a 0.0733 test number, which is holdout-fitting, and what sweep would you pre-declare?

5. **Time split, bluntly.** About 13 hours to the freeze, four sessions, one decision-maker. Split the hours across (a) heads from ask 2, (b) features from ask 3, (c) fusion from ask 4, (d) documentation (the README is drafted; a docs site and the disclosure are in flight), (e) stop touching the file. Say which single change you expect to move the test number the most, and whether the honest answer to "should we keep experimenting" is no.

6. **Anything we have wrong.** You have the whole picture above, including the returned number. Where is the premise error, if there is one?

## Additional context

- What already works and sets the bar: the shipped file scores 0.0733 on the real test set, reproduces from audio end to end, and beat every simpler rule on every proxy under both costs. A change ships only if it wins a pre-declared sweep on inner OOF, holdout and In-the-Wild under both costs, and Nathan ratifies it; otherwise it goes into the README as a negative result, which is itself graded.
- We can A/B anything on exported logits in seconds; new embeddings cost about 22 minutes of CPU per corpus and layer set; a new M5-style head costs about 12 minutes per fold model on an A100 plus setup.
- Hard lines: no training or selection on the holdout, In-the-Wild or the test set; no pseudo-labels; augment both classes or neither; the pinned block stays at the bottom; 1.0 = synthetic; final file frozen Sunday 05:00.

## Output format

1. One-line verdict on the returned number (regime, likely error profile, confidence).
2. A ranked options table across asks 2–4: approach / mechanism that could move the test number / expected minDCF effect / effort in hours / risk / verdict (do, maybe, skip).
3. For each "do": a runnable procedure with the fold discipline spelled out, the pre-declared fusion entry and the acceptance rule.
4. The time split for ask 5, with the single highest-value change named.
5. A direct call on "are we over-engineering; should we stop touching the file?", no hedging.
6. Failure modes we have not named.

---

## Round 0: VeriLM scope questions and our answers (Sat ~15:55)

Settings chosen: Deep Think, Standard tier at X-High effort (Opus 5 + GPT-5.6 Terra at X-High, Gemini 3.1 Pro at High; synthesis Opus 5; 210 credits). Grounding off, Verify off: every number the experts need is in the prompt, and any recommendation is validated on our side by a pre-declared sweep, which is a stronger check than a verify pass.

1. **Ratification.** Nathan ratifies every change before it replaces the shipped file; nothing auto-ships. Experiments run freely in parallel sessions. Recommend aggressively and label candidates "ready to ratify".
2. **Confidence bound.** State one explicitly. Granularity under the brief's cost with ~1,170 real / ~500 synthetic: one false alarm = 0.008 minDCF, one miss = 0.002, so 0.0733 is about 9 false-alarm equivalents or 37 misses. Under the sponsor code's cost: one false alarm = 0.00085, one miss = 0.008. Our holdout's own noise is ±0.07–0.10 (two generators), so holdout gaps under 0.15 are not evidence.
3. **Minimum bar.** Strong loss aversion. Our standing rule: a candidate replaces the shipped rule only if In-the-Wild brief-cost improves by ≥ 0.01, miss-averse worsens by ≤ 0.01, holdout worsens by ≤ 0.05, inner OOF worsens by ≤ 0.03, and Nathan ratifies. The last switch moved 3 of 1,671 files across 0.5 (Spearman 0.974); a change that reorders more than that needs a stronger case. Propose a stricter bar if one shot warrants it.
4. **Documentation.** Fixed and separately staffed: the README is drafted (397 lines) with its own session, a docs-site lane is running, the disclosure is current; the README polish window is Sun 05:00–07:30. A shipped rule change costs 1–2 hours of number updates. Experimental hours come from the other sessions plus Nathan's decision bandwidth, which is the binding constraint.
5. **GPU budget.** Not binding: tens of dollars is fine ($5.42 spent, A100 about $0.7/h). Wall clock is: a fold model trains in 12 minutes but the M5 rung took hours end to end because of bundle transfer (Cloudflare R2) and boxes that never booted (6 of 12). Anything on the A100 must fit the 13-hour clock with that overhead.

Anything else: the Mac is shared by four sessions, so CPU timings above stretch by about 40% under load; the plan's 22:00 fusion freeze is ours and can move, the 05:00 freeze cannot; the previous consult predicted A3 would lose on short clips and LibriSpeech real, and it did not.

**Added to "anything else" (Sat ~16:05): how close our proxies came to NSA's numbers.** Shipped rule (A3 w0.2 + E), recomputed from the exported logits (minDCF matches the sweep report; EER and the error rates at the brief-cost argmin are new):

| Split | Rows (real / synthetic) | minDCF (brief) | EER | P_FA / P_miss at the minDCF threshold |
|---|---|---|---|---|
| Outer holdout | 1,858 / 2,000 | 0.0065 | 0.21% | 0.00% / 0.65% (0 FA, 13 misses) |
| Inner OOF | 8,142 / 8,000 | 0.135 | 6.83% | 0.25% / 11.2% |
| In-the-Wild | 2,000 / 1,000 | 0.228 | 5.80% | 0.75% / 15.8% |
| **NSA test (returned)** | ~1,170 / ~500 | **0.0733** | **3.53%** | not given |

Reading: the holdout was optimistic by 11× on minDCF and 17× on EER; In-the-Wild was pessimistic by 3× on minDCF but only 1.6× on EER; inner OOF was pessimistic by 2× on both. On a log scale between the holdout and In-the-Wild the test sits at 0.68 for minDCF and 0.85 for EER, further toward the wild end than λ̂ = 0.51 (CI 0.12–0.64) suggested. Relative to In-the-Wild the test's minDCF is better than its EER, so the low-false-alarm end of the ranking (where minDCF is scored) held up better than the middle: consistent with a miss tail from unseen generators rather than a false-alarm problem, but that is an inference, not a measurement. Single-detector reference: M1b v3 alone scores holdout 0.072 / EER 1.4%, almost exactly the test number of the full rule.

**Added (Sat ~16:15): where we stand against the field.** NSA's draft-review feedback puts one team ahead of us: minDCF 0.0584, EER 2.5% (ours 0.0733, 3.53%). Under the rubric's formula, points = (1 − ours)/(1 − best) × 60, that gap is worth 0.95 of 60 points today (59.05 vs 60.00). If the leading team reaches 0.03 by the deadline the gap grows to 2.7 points. Matching 0.0584 means removing 0.015 minDCF, about 2 false alarms or 7 misses under the brief's cost. The other 40 points (technique diversity, documentation) are unaffected by any of this.
