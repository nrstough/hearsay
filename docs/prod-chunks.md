# HEARSAY: Prod Chunks (master doc, v1 → v2)

Started Thu Oct 1, 2026. This is the master doc for HEARSAY after the hackathon. Part A records v1, the system we submitted, chunk by chunk. Part B explains what v1 taught us, in plain English. Part C splits v2 into chunks that can each be planned, run and closed on their own.

- **v1 is frozen.** It is git tag `v1`, the system as submitted. Nothing in Part A changes except to fix a fact.
- **Sources of truth for v1 numbers:** `submissions/log.csv`, the reports under `docs/reports/`, and each model's `meta.json`. Every number here names its file. If this doc and one of those disagree, the source wins.
- **Plan of record for v2:** this file. `docs/plan.md` is the hackathon plan and stays as history. `docs/master-doc.md` is the pre-rescope doc, pinned unchanged by `tests/test_docs_consistency.py`.

## How a chunk works

Each chunk is one self-contained piece of work. A chunk is done when its **Done when** line is true and its report exists, not when time runs out.

| Field | Meaning |
|---|---|
| **Goal** | One sentence: what exists or is known afterwards that wasn't before |
| **Why** | The v1 evidence that makes this worth doing, with a file reference |
| **Questions** | What the chunk must answer (understanding chunks) |
| **Method** | How, at the level of a plan; the run spec goes in `docs/specs/` |
| **Deliverables** | Files, models, reports |
| **Done when** | A checkable condition |
| **Depends on** | Chunks that must close first |
| **Size** | S (≤ 1 day), M (2–4 days), L (≥ 1 week). Any M or L goes through `/plan-review` first. |

Status markers used below: ✅ done (v1), 🟡 built but thin, ⬜ not started.

---

# Part A: v1 as built (retrospective)

## A.0 v1 in one paragraph

Any audio file goes in.
- **Normalization.** It is decoded once to 16 kHz mono and low-passed at 7.25 kHz to match the test set.
- **Detectors.** Seven engineered detectors and three deep detectors score it.
- **Fusion.** Three scores are blended by rank: 0.6·M1b (frozen XLS-R probe) + 0.2·handcrafted v5 + 0.2·M5 (XLS-R head). Spectra-AASIST can then halve a score it thinks is real, but it can never raise one.
- **Calibration.** A Platt map turns the blended rank into a probability at the 0.3 prior.
- **Explanation.** The rule-based detectors (container, ENF, splice, speaker drift, speech gate) never move the score. They supply routing facts and the plain-English evidence in the per-file explanation JSON.

**Result.**
- **File.** `submissions/CrossExam_predictions.tsv`, sha256 `fb783076…`.
- **NSA's score.** minDCF **0.0733**, EER **3.534%**. One team was ahead at 0.0584 / 2.5% (`docs/reports/2026-09-26_sponsor-questions.md`, log row 15:40).
- **Our own proxies straddle that number.** Outer holdout 0.0065 was 11× optimistic. Inner CV 0.135 was 2× pessimistic, and In-the-Wild 0.228 was 3× pessimistic (README).

## A.1 v1 chunks

Each v1 chunk is listed with what it produced and what it taught. Detail lives in the linked report.

### V1-01 ✅ Foundation: loader, metric, writer, folds
- **Built.**
  - **Loader and writer.** One FFmpeg decode path (`hearsay.audio`). The TSV writer never overwrites a file and always logs (`hearsay.submission`).
  - **Metrics.** `hearsay.metrics` with C_FA = 4, C_miss = 1, π_synth = 0.3, so normalized minDCF = 9.33·P_FA + P_miss. `sponsor_min_dcf` re-implements NSA's ASVspoof5 code.
  - **Fold file.** `splits/nsa_folds.csv`, 20,000 rows. The outer holdout is the playht and wavegrad2 generators plus 26 real groups (3,858 rows). There are 5 inner folds grouped by generator and speaker, and LJ is grouped by chapter.
- **Learned.**
  - **Holdout noise is large** (σ ≈ 0.07–0.10), so gaps under about 0.15 are noise.
  - **LJ crosses every fold.** It is one speaker, so it sits on both sides of every fold.
  - **Folds are simpler than planned.** The code groups by plain generator and speaker keys, not the connected components that `docs/plan.md` describes.

### V1-02 ✅ Shortcut hunting
- **Found and closed** (`docs/architecture.md` §11). Each was a way to tell training data from test data without listening for fakes:
  - Peak level (AUC 0.66 alone).
  - Leading silence: 0.37 s in LibriSpeech vs 0.06 s in test clips.
  - Clip length and tiling: training clips run 5–9 s, test clips about 3.4 s.
  - Container format. PlayHT's MP3s carry the same `Lavf58.29.100` tag as every test file.
  - **The 7.2 kHz wall.** 1,602 of 1,671 test files drop 44 dB between 6.5 and 7.5 kHz, and no training corpus does.
  - Mains hum: present in LJ and LibriSpeech, almost never in generators.
  - The double low-pass stopband.
  - Non-speech: silence scored 99% synthetic.
- **Learned.** This was the most valuable work of the weekend. Before the band match, the handcrafted model called 90% of the test set synthetic.

### V1-03 ✅ M1 / M1b: frozen XLS-R probe (the score carrier)
- **Built** (`src/hearsay/probe.py`, `src/hearsay/embed.py`).
  - **Embedding.** `facebook/wav2vec2-xls-r-300m`, frozen. Hidden state 7 is averaged over time, giving 1,024 numbers per clip.
  - **Classifier.** A StandardScaler, then class-balanced logistic regression, then Platt calibration.
  - **M1b** adds ASVspoof 2019 real speech (5,128 clips, 40 VCTK speakers) and a 2,520-clip A01–A06 spoof anchor, in inner folds only.
- **Numbers** (`submissions/log.csv`, model `meta.json`):

  | Model | Inner | Holdout | EER | In-the-Wild | ITW P_FA / P_miss |
  |---|---|---|---|---|---|
  | M1 v3 | 0.257 | 0.159 | 2.85% | 0.380 | 0.7% / 35.8% |
  | M1b v3 (shipped) | 0.265 | 0.0717 | 1.40% | 0.342 | 0.8% / 27.8% |

- **Learned.**
  - Layer 7 is best. Layers 6 and 8 are close, and layers 19–24 are poor (0.40–0.43).
  - The NSA-domain holdout (0.07–0.16) badly understates the wild-audio gap (0.34–0.41).
  - More read speech made the holdout better but raised In-the-Wild false alarms (v2: 1.2% → 5.7%).

### V1-04 ✅ Handcrafted spectral + prosody (v1 → v5b)
- **Built.** The detector is `hearsay.detectors.handcrafted`, with v4 families in `hearsay.hc_v4`. It is a LightGBM model whose trees run in numpy at inference.
  - **v1–v3** have 75 features: high-band ratios, roll-off, centroid, bandwidth, flatness, spectral contrast, flux, 20 MFCCs, YIN F0 statistics, voicing, energy, pauses and ZCR.
  - **v4** adds six families, for 234 columns: LFCC, CQCC, group-delay and phase coherence, envelope modulation, jitter/shimmer, and breath.
  - **v5b** (shipped) adds fit-only augmented twins: codec round-trips plus random spectral tilt and low-pass.
- **Numbers.** v3 holdout 0.254 → v4a 0.170 → v5b 0.137 (EER 3.1%, inner 0.469) (`docs/reports/2026-09-26_handcrafted-v4.md`, `models/hc_selected/meta.json`).
- **Learned.**
  - grad_tts is undetectable when held out in every variant (1.00), and ElevenLabs is weak (0.67–0.92).
  - **The test set's spectral envelope is shifted:** darker and codec-like. Train-vs-test separability is AUC 0.991.
  - The model is useless on In-the-Wild (minDCF 1.0, misses 96–97%) and misses 56% of MLAAD.
  - v6 (noise twins) failed its gate.

### V1-05 ✅ Engineered forensic detectors
- **Built** (`src/hearsay/detectors/`): container, compression, ENF, splice, speech gate and speaker drift (ECAPA).
- **Learned** (`docs/reports/2026-09-26_cpu-detectors.md`, `docs/reports/2026-09-26_gate-and-drift.md`). Every one of them ends up measuring the dataset, not the fake, so they became evidence and routing only:
  - **Container.** The container is the training label.
  - **Compression.** The test set reads as laundered (mean P(laundered) 0.62).
  - **ENF.** Hum means LJ, which means real.
  - **Splice.** Seams in the training data are vocoder and codec artifacts.
  - **Speaker drift.** Fakes are the *most* self-consistent voices, so a learned column would conclude that drift means real.

### V1-06 ✅ M3: Spectra-AASIST (off the shelf)
- **Built** (`src/hearsay/spectra.py`, `docs/reports/2026-09-26_m3-spectra.md`).
  - **Model.** `lab260/Spectra-AASIST`, a complete anti-spoofing model: its own fine-tuned XLS-R plus an AASIST graph-attention back end. No training on our side.
  - **Input.** Clips are cut into 4.04 s windows. 81% of test clips are shorter than one window and get zero-padded.
- **Numbers.** Holdout 0.012 (EER 0.16%), inner 0.0045, In-the-Wild 0.065.
- **Admitted only as a suppressor.** Its training data is undisclosed and may include LJ, LibriSpeech and DiffSSD, so those numbers can't be trusted.

### V1-07 🟡 M5: XLS-R "fine-tune" (shipped as a frozen-backbone head)
- **Built** (`docs/reports/2026-09-26_m5-xlsr-finetune.md`).
  - **Model.** XLS-R cut to 12 layers, a learned layer mix (peaks at layers 5–6), attentive statistics pooling and a linear head.
  - **Data.** NSA + extra LJ/LibriSpeech + ASVspoof 2019 + MLAAD (6,180 clips, 143 TTS models).
  - **Augmentation.** Class-blind laundering: noise, reverb, band-limits, gain and clipping, RawBoost, and codecs (MP3, Opus, AAC, μ-law).
  - Trained on vast.ai A100s for $5.42.
- **Important.** The shipped M5 is **not** a fine-tuned backbone. The runs that actually fine-tuned the 12 layers were worse on the hardest folds: ElevenLabs 0.38–0.41 vs 0.31 frozen, and NSA-only full fine-tune 0.762. The shipped arm is the frozen backbone with a trained head.
- **Gate: failed.**

  | | M5 | M1 v3 |
  |---|---|---|
  | Holdout | 0.363 | 0.159 |
  | Inner OOF | 0.302 | 0.257 |
  | ITW false alarms (the one axis it won) | 0.45% | 0.7% |

  It still earned 20% of the fusion because it catches different fakes.
- **Learned.**
  - It is weak on short clips: ≤ 4 s scores 0.595, and the test median is 3.4 s. It was never trained on short crops.
  - It is weak on wavegrad2 (0.578).
  - It is weakest of all under noise (0.57).

### V1-08 ✅ Fusion, orchestration, default answer
- **Built.** Pre-declared sweeps, a rank blend, and the M3 suppression step "E": if M3's margin is below −3 and the base rank is above 0.5, the rank is halved.
  - Constants: `models/fusion_v2/constants.json`.
  - Reports: `docs/reports/2026-09-26_fusion-sweep-predeclared.md`, `docs/reports/2026-09-26_post-draft-sweep.md`.
- **Learned.**
  - Only the M3 suppression changes decisions (`outputs/fusion/orchestration_ablation_v2.md`).
  - Equal-weight fusion of all four looked *better* on every labeled readout, but it was rejected because Spectra's data is unknown.
  - The "logistic stacker" in the plan never shipped. Only the Platt map is fitted.
  - After the draft, four candidates were tried (WavLM column, M5 weight 0.4, a second Spectra tier, noise-augmented handcrafted). All failed the pre-declared gate, and v1 was kept.

### V1-09 ✅ Channel robustness
- **Built** (`docs/reports/2026-09-26_channel-robustness.md`).
- **Learned.**
  - The 7.2 kHz wall is a smooth resampler low-pass, not MP3 or AAC.
  - λ̂ (how "wild" the test set is) = 0.51, CI 0.12–0.64, so indeterminate. The test set is its own recording population.
  - **The biggest measured weakness: 20 dB white noise turns fakes into misses.** The shipped rule's holdout goes from 0.012 to 0.269, and M1b's P_miss from 0.07 to 0.88.

### V1-10 ✅ Runner, API, Docker, hosting, docs
- **Runner and API.** `scripts/run_pipeline.py`, `hearsay.pipeline` and `hearsay.api`. The live run matches the submitted file (Spearman 1.0).
- **Docker.** Built and verified, but still on the *previous* fusion rule (`fusion_v1`).
- **Hosting.** Vultr demo box (teardown still pending, see C.0).
- **Docs.** Docs site and 72-page PDF.

---

# Part B: Understanding v1

Each section marks which parts the repo **measured** and which are **background** (general knowledge, not tested here). The background parts are hypotheses for Part C to test.

## B.1 What a "backbone" is, and why we read layer 7

XLS-R and WavLM are **self-supervised speech models**.
- **How they were trained.** On huge amounts of unlabeled speech, to fill in hidden parts of the audio. No one told them about fakes.
- **What they produce.** Each of their 24 layers outputs a 1,024-number description of every 20 ms of audio.
- **What a probe is.** Freeze the model, average one layer over time, and fit a plain logistic regression on top.

**Measured.**
- XLS-R's best layer is 7. Layers 6 and 8 are close, layers 19–24 are much worse, and layers 0–1 are near chance.
- M5's learned layer mix independently put most weight on layers 5–6.

**Background.**
- Lower and middle layers keep fine acoustic texture: how the waveform was made.
- Upper layers are shaped toward phonetic content: what was said.
- Vocoder artifacts are texture, so the middle wins.

## B.2 XLS-R vs WavLM: same score, different mistakes

**Measured** (`docs/reports/2026-09-26_post-draft-heads.md`). Same recipe, same data, same crops.

| | WavLM Large L9 | XLS-R (M1b) L7 |
|---|---|---|
| Outer holdout minDCF | 0.0717 | 0.0717 (identical) |
| Best layer | 9, flat from 8 to 18 | 7, sharp |
| In-the-Wild minDCF | 0.752 | 0.342 |
| Its In-the-Wild failure | **false alarms** on wild real speech (7.05%) | **misses** on wild fakes (P_miss 27.8%) |
| Rank agreement (Spearman) with the other | holdout 0.86, test 0.80, ITW 0.77 | |
| Of the shipped rule's 158 ITW misses, how many it alone flags | **57** | 1 |

So the two backbones see different things. WavLM catches fakes XLS-R is blind to at any layer (`docs/reports/2026-09-26_post-draft-negatives.md`). But it also distrusts real wild speech, and as a fourth fusion column it made inner CV worse, so it failed the gate.

**Background (to test in V2-21).**
- **Pretraining data.** XLS-R was trained on about 436k hours of multilingual read and broadcast speech. WavLM Large was trained on about 94k hours of mostly English audiobooks and podcasts.
- **Pretraining objective.** XLS-R uses wav2vec 2.0's contrastive objective. WavLM predicts masked units *and* is trained to denoise speech mixed with other speakers and noise.
- **Hypothesis.** WavLM learned to separate speech from channel and background. That could make it notice odd "background" in clean TTS (catching fakes) and also in noisy real speech (false alarms).
- **Not yet measured.** Nothing in the repo tests why. That is a gap, not a finding.

## B.3 What "fine-tuning" changed (and didn't)

**Measured.**
- Real backbone fine-tuning **overfit the generators it saw**. Training loss fell to the label-smoothing floor while transfer to held-out ElevenLabs got worse.
- The shipped M5 is a frozen backbone. It differs from M1 in five ways:
  1. a learned mix of layers 0–12 instead of only layer 7
  2. attention pooling instead of a plain average
  3. more and more varied data (MLAAD's 143 TTS models)
  4. heavy augmentation
  5. a trained nonlinear head
- Net effect: fewer wild false alarms, but much worse on short clips, wavegrad2 and the NSA holdout.
- Some of its errors are complementary, which is why 20% of it helped the fusion (ITW 0.260 → 0.228).

**Open.**
- Which of those five differences caused which effect was never separated. V2-22 does that ablation.
- Fold 1 (diffgan + openvoicev2: 0.263 vs 0.090) was never diagnosed.

## B.4 Why Spectra-AASIST behaved so differently

"Spectra" is not a spectrogram model. It is **another XLS-R**, fine-tuned end to end by someone else on undisclosed data, with an AASIST graph-attention classifier on top.

**Measured.**
- **Its accuracy looks too good to trust:** holdout 0.012, inner 0.0045.
- **Its outputs are confident and two-humped:** spoof margins +10.5, real margins −7.7.
- **It agrees weakly with M1:** Spearman 0.57 on test, with 9.2% of files on opposite sides of the threshold.
- **Its ranking is the least stable under perturbation** (noise Spearman 0.875 vs M1b 0.913), even though its AUC barely moves.
- **It catches the most MLAAD fakes** (miss rate 14.9% vs M1b 22.7%).

**Most likely explanation (not proven).**
- It was trained on corpora overlapping ours (LJ, LibriSpeech, DiffSSD-like TTS), so it has partly *memorized our evaluation data*. That explains numbers too good to trust.
- It was also trained end to end on far more spoof variety, which explains the confident margins and the different fakes it catches.
- Zero-padding 81% of test clips is out of distribution for it. Its training most likely used repeat-padding.

V2-23 tests these.

## B.5 Why the handcrafted model drifts on the test set

**Measured.**
- 234 interpretable features separate training audio from test audio almost perfectly (AUC 0.991).
- The test set's spectral envelope is darker and codec-like. 84% of shifted columns move in the "fake" direction.
- Augmentation (v5b) cut the damage but didn't remove it. Calibrated test share went 56% → 51%, against a true share of about 30%.

**Reading.** Hand-designed spectral features measure the *recording chain* as faithfully as they measure the voice. Unless training covers the test chain, they report the chain. This is the main argument for augmentation and better real-speech data in v2 (V2-30, V2-31).

## B.6 Why every holdout number was optimistic

- **Same pipeline on both sides.** The outer holdout comes from the same pipeline as training: same real corpora (LJ, LibriSpeech) and same generator family.
- **In-the-Wild is wild.** It is real-world audio, and every deep probe drops to 0.34–0.41 there.
- **The test set is neither** (λ̂ indeterminate). NSA's 0.0733 sits between our holdout (0.0065) and In-the-Wild (0.228).
- **Lesson.** v2 needs **several** distinct evaluation populations and reports all of them (V2-01).

## B.7 How the metric decides what matters

normalized minDCF = 9.33·P_FA + P_miss. **One false alarm costs about 9 misses.** So:
- **Real speech gets most of the attention.** The real-speech side of the distribution decides the score.
- **Ranking is enough for minDCF.** It sets its own best threshold.
- **Real-world use needs calibration.** A real deployment has to pick a threshold in advance (actual DCF), which v1 never measured (V2-61).

---

# Part C: v2

## C.0 Before any v2 chunk (housekeeping)

| Item | Why | Done when |
|---|---|---|
| **Destroy the Vultr box** `hearsay-demo` 64.177.49.133 | Running since Sep 26 at about $0.055/h; no destroy row in `docs/reports/cloud-expense-ledger.md` | Destroyed in the console, ledger row added |
| **Commit or discard the stray changes** | `docs/specs/2026-09-26_m5-xlsr-finetune-audit.md`, `next-env.d.ts`, `tsconfig.json` are modified on `main` | Clean `git status` |
| **Mount check** | `data/nsa/DiffSSD`, `data/ljspeech` and `data/librispeech` are symlinks to the external Crucial drive | v2 README says so; manifests resolve |
| **Jev access** | V2-63 needs a TypeSafe API key and a team decision on sending derived scores to a hosted service | Key in `.env`; decision recorded in V2-63 |

## C.1 v2 rules (carry over, plus new)

Carried from v1:
- 16 kHz mono everywhere.
- Validate by generator and speaker, never by clip.
- Never train on a validation split, including scalers and calibration.
- Pre-declare selection rules.
- Keep a clean-only score beside any augmented one.
- No hosted model on the scoring path.

New for v2:
1. **v1 is untouchable.** Nothing under tag `v1` changes, and `submissions/` keeps v1's history. v2 experiments log to `experiments/log.csv`, created in V2-00. It has the same columns plus `benchmark_version`.
2. **One benchmark, versioned.** Every v2 number is reported on the V2-01 benchmark, all populations, both costs. A number from anywhere else is a note, not a result.
3. **A sealed set.** One held-out population is touched only to close a phase (V2-01). In-the-Wild can no longer be that set: v1 used it to *select* the fusion rule, so it is contaminated for selection.
4. **Understanding before building.** Phases 1–2 exist so that Phases 3–6 change one thing at a time for a stated reason.

## C.2 Phase map

```
Phase 0  Foundation        V2-00 repo  ─┬─ V2-01 benchmark ─┬─ V2-02 error atlas of v1
                                        │                   │
Phase 1  Understand data   V2-10 corpus atlas · V2-11 test-set profile
                                        │
Phase 2  Understand models V2-20 layer probing · V2-21 XLS-R vs WavLM · V2-22 M5 ablation · V2-23 Spectra anatomy
                                        │
Phase 3  Data              V2-30 new data · V2-31 augmentation library · V2-32 data mix
                                        │
Phase 4  Features          V2-40 handcrafted spectral v7 · V2-41 forensic upgrades
                                        │
Phase 5  Models            V2-50 backbone fine-tune done right · V2-51 new architectures · V2-52 short-clip
                                        │
Phase 6  Fusion + decision V2-60 neural fusion · V2-61 probabilistic decision layer · V2-62 orchestrator v2 · V2-63 Jev
                                        │
Phase 7  Ship              V2-70 explanations · V2-71 release v2 · V2-72 write-up
```

Phases 1 and 2 can run in parallel. Phases 3 and 4 can too, once V2-01 exists. Phase 6 waits on at least one Phase 3–5 result.

**Suggested first three:**
1. V2-0U, understanding v1 first. Then V2-00 + V2-01: nothing else is comparable without them.
2. V2-02, cheap and it uses only v1's exported scores.
3. V2-21, the XLS-R vs WavLM question you most want answered.

---

## Phase 0: Foundation

### V2-0U 🟡 Understand v1 (learning guide)
- **Goal:** Every team member can explain v1 end to end, including what each detector does, why its role was chosen, and where v1 fails, without reading the hackathon reports.
- **Why:** v1 was built in parallel AI-assisted lanes in 36 hours; the team's understanding lags the code.
- **Deliverables:**
  - `docs/learn/v1-explained.md`: 15 lessons with check-yourself questions, a glossary and hands-on exercises.
  - Its figures in `docs/learn/img/`, made by `scripts/learn_figures.py`.
- **Method:** Read in order and do the exercises. Go through one lesson per session with Claude, which quizzes on the check-yourself questions and answers follow-ups.
- **Done when:** Nathan can give the 1-minute explanation (lesson 0) and answer every check-yourself question. Any lesson that was confusing gets rewritten.
- **Size:** S–M.


### V2-00 ⬜ v2 workspace
- **Goal:** v2 work happens beside v1 without touching it.
- **Method:**
  - Add a `v2` section to `CLAUDE.md`.
  - Create `experiments/log.csv` and a `models/v2/` and `outputs/v2/` convention.
  - Fix the known `load_embeddings` gap: honor the `decode_error` flag (CLAUDE.md gotchas).
  - Add a one-command "reproduce v1 numbers" check from the exported scores.
- **Done when:** `uv run pytest` is green, the v1 reproduce check matches `submissions/log.csv`, and the log exists.
- **Size:** S.

### V2-01 ⬜ The v2 benchmark
- **Goal:** One script reports any detector or fused system on a fixed set of populations, under both costs, with confidence intervals.
- **Why:** v1's three readouts disagreed by 11× and 3× around the one real number (B.6). In-the-Wild is spent as a selection set.
- **Populations (proposed):**
  1. NSA outer holdout (as v1)
  2. In-the-Wild (report only)
  3. MLAAD held out by model family
  4. **ASVspoof 5 eval** or another modern public set (V2-30 picks it)
  5. Perturbation cohorts: noise 20/10 dB, reverb, codecs, telephony, short crops ≤ 3.5 s
  6. **Sealed set:** a population never used for any choice, touched once per phase
- **Also:**
  - Ask NSA whether the test key (labels) can be released after the event. If yes, it becomes the best sealed set we could have.
  - Speaker-cluster bootstrap CIs (`scripts/fuse_sweep_v3.py` already has one).
  - actDCF at a fixed threshold beside minDCF and EER.
- **Deliverables:** `scripts/benchmark.py`, `benchmarks/v1.yaml` (population manifests + hashes), `docs/reports/<date>_benchmark-v1.md` with v1's full scorecard as the baseline.
- **Done when:** the shipped v1 rule and each v1 column have a scorecard row, and re-running reproduces it bit for bit.
- **Depends on:** V2-00.
- **Size:** M.

### V2-02 ⬜ Error atlas of v1
- **Goal:** Know exactly which files v1 gets wrong and what they have in common.
- **Why:**
  - The 158 ITW misses.
  - grad_tts is blind in every handcrafted variant.
  - ElevenLabs is weak.
  - M5's fold-1 loss is undiagnosed.
  - 154 test files where Spectra and M1 disagree.
- **Questions:**
  - Are the misses short, noisy, a particular generator, a particular speaker?
  - Which column would have caught each one?
  - Do the false alarms share a recording chain?
- **Method:**
  - Join all exported columns (`outputs/detector_scores/*.csv`) with metadata (duration, SNR estimate, bandwidth, generator, speaker).
  - Fit shallow decision trees on "v1 wrong vs right" to describe the error regions.
  - Listen to 20 examples per region.
- **Deliverables:** `docs/reports/<date>_v1-error-atlas.md`, `outputs/v2/error_atlas.parquet`.
- **Done when:** every error region has a named hypothesis and a chunk that tests it.
- **Depends on:** V2-01.
- **Size:** S–M.

## Phase 1: Understand the data

### V2-10 ⬜ Corpus atlas
- **Goal:** A map of every corpus we have: what it sounds like, how it was recorded, and what separates each one from the test set.
- **Why:** Every v1 shortcut was a corpus difference found by accident (V1-02). Do it on purpose.
- **Questions:**
  - Per corpus and per generator: duration, level, SNR, bandwidth and wall shape, silence, codec traces, speaker count, language, speaking style.
  - Which features separate corpus A from B? This is a "domain classifier" per pair.
  - Where do the corpora sit in XLS-R and WavLM embedding space (UMAP/t-SNE plots coloured by corpus, label and generator)?
- **Deliverables:** `docs/reports/<date>_corpus-atlas.md` with plots, `outputs/v2/corpus_stats.parquet`, an updated shortcut ledger.
- **Done when:** each corpus has a profile card and every pair has its top separating features listed.
- **Size:** M.

### V2-11 ⬜ Test-set profile
- **Goal:** Describe the NSA test set as fully as possible without labels.
- **Questions:**
  - Does it cluster, for example by recording chain or by speaker?
  - Which training corpus is each cluster nearest to?
  - What is the residual difference in the 6.8–7.25 kHz transition band (unprobed in v1)?
  - What explains the envelope shift (B.5)?
- **Method:**
  - Clustering in several embedding spaces.
  - Nearest-corpus classifier.
  - Wall-shape fitting.
  - Never use v1's scores as labels.
- **Done when:** we can write one paragraph on "how the NSA test set was probably made", with evidence.
- **Size:** S–M.

## Phase 2: Understand the models

### V2-20 🟡 Layer-wise probing across backbones
- **Started Oct 3:** `docs/reports/2026-10-03_layer-diversity.md`. Per-layer probes for XLS-R and WavLM, error overlap, pairs and the noise test. Lead: XLS-R L7 + L20 beats L7 alone on inner (0.255 vs 0.265) and In-the-Wild (0.267 vs 0.342), pending a pre-declared confirmation.
- **Goal:** One chart per backbone showing which layers carry fake-vs-real information, and which carry corpus and channel information.
- **Method:**
  - For XLS-R, WavLM Large, WavLM Base and one more (e.g. HuBERT or wav2vec2-BERT), probe every layer for four targets:
    - real vs fake
    - generator identity
    - corpus identity (a shortcut)
    - noise or codec condition (a shortcut)
  - The best layer is the one high on the first target and low on the shortcut targets.
- **Deliverables:** `docs/reports/<date>_layer-probing.md`, reusable `scripts/probe_layers.py`.
- **Done when:** the "middle layers carry vocoder texture" claim (B.1) is either supported or refuted with numbers.
- **Depends on:** V2-01. Embeddings can reuse `scripts/extract_embeddings.py`.
- **Size:** M.

### V2-21 🟡 XLS-R vs WavLM: why they differ
- **Started Oct 3** (same report). WavLM's opposite failure is confirmed at every layer from 8 up. Noise triples its false alarms (3.5% → 12.7% from cleanest to noisiest third), but even its clean clips are 4–5× XLS-R's rate.
- **Goal:** Explain B.2's measured difference with evidence, not guesses.
- **Questions:**
  - Are the two representations similar layer by layer? Use CKA similarity matrices between every XLS-R and WavLM layer.
  - Which files does each get right that the other misses, and what do those files share (from V2-02's metadata)?
  - Is WavLM's false-alarm tendency on wild real speech tied to background noise level? This tests the denoising hypothesis.
  - Does a model that sees both (concatenated, or a learned mix) beat either alone on the benchmark?
- **Deliverables:** `docs/reports/<date>_xlsr-vs-wavlm.md`, a short plain-English section added to Part B here.
- **Done when:** each hypothesis in B.2 is marked supported, refuted, or untestable with what we have.
- **Depends on:** V2-01, V2-20.
- **Size:** M.

### V2-22 ⬜ M5 ablation: what each change did
- **Goal:** Separate M5's five differences from M1 (B.3) and know which helped and which hurt.
- **Method:**
  - Starting from M1b, add one change at a time, each on the benchmark:
    - multi-layer mix
    - attention pooling
    - MLAAD data
    - augmentation
    - nonlinear head
    - short-crop training (never tried)
  - Then a true fine-tune of a few top layers with early stopping on a held-out *generator*.
  - Use embeddings where possible (cheap, on the Mac). Use a GPU only for the fine-tune arm, and ask before renting.
- **Done when:** each change has a measured effect with a CI, and fold 1's loss is explained.
- **Depends on:** V2-01.
- **Size:** M–L.

### V2-23 ⬜ Spectra-AASIST anatomy
- **Goal:** Decide whether Spectra can be trusted, and learn from its design.
- **Questions:**
  - Does it score LJ or LibriSpeech training clips more confidently than unseen real speech from the same recording conditions? That is a memorization test.
  - Does repeat-padding instead of zero-padding change its test behaviour?
  - Why is its ranking unstable under noise when its AUC is not?
  - Does its back end (AASIST graph attention) add anything over its XLS-R alone (probe its internal XLS-R at layer 7)?
- **Done when:** a written verdict says whether to use it as a column, keep it as a suppressor, or drop it.
- **Size:** S–M.

## Phase 3: Data

### V2-30 ⬜ Better data
- **Goal:** Training data that covers the recording chains and generators v1 lacked, with licences recorded.
- **Why:**
  - v1's real speech is mostly read audiobooks (LJ, LibriSpeech, VCTK).
  - The wild-domain gap (B.6) and the envelope shift (B.5) both point to real speech that's too clean.
  - v1 dropped the wild real corpus because VoxCeleb needs a request form and SpoofCeleb's real speech is enhanced.
- **Candidates, each to vet for licence and overlap:**
  - **Real, wild:** VoxCeleb1/2 (request), Common Voice, podcast or broadcast sets, telephone speech.
  - **Spoof, modern:** ASVspoof 5, newer MLAAD releases, codec-based TTS (e.g. CodecFake-style sets), recent commercial voices.
  - **Matched pairs:** real and fake of the same speaker and text, if any set offers them; best for understanding.
- **Rules:**
  - Every new set gets a profile card (V2-10) and a shortcut check (`/validate-training-data`) before training.
  - Grouping goes by generator and speaker in the fold file.
  - MLAAD is non-commercial, which matters if v2 is ever a product.
- **Done when:** at least one wild real corpus and one modern spoof corpus are in manifests, profiled, and folded.
- **Size:** M.

### V2-31 ⬜ Augmentation library (shared, symmetric)
- **Goal:** One augmentation module every detector trains through, always applied to both classes.
- **Why:**
  - Noise is v1's biggest measured hole (V1-09: holdout 0.012 → 0.269 under 20 dB noise).
  - M1/M1b never saw any augmentation.
  - Augmentation was built three separate times (handcrafted v5, compression laundering, M5's `src/hearsay/augment.py`).
- **Method:**
  - Promote `src/hearsay/augment.py` to the shared module.
  - Add: speed and pitch, real noise and room-response libraries (e.g. MUSAN-style noise, measured room impulse responses), telephony (incl. AMR where an encoder is available), resampler walls shaped like the test set's.
  - Every model reports clean and augmented scores side by side (existing rule).
- **Done when:** M1b, retrained on augmented embeddings, is benchmarked, and the noise cohort improves without the clean score degrading beyond a pre-declared margin.
- **Depends on:** V2-01. V2-30 is helpful.
- **Size:** M.

### V2-32 ⬜ Data mix study
- **Goal:** Know how much each source contributes. Make the proportions of real speech, spoof families and augmentation a measured choice.
- **Why:** v1 cut MLAAD from 28% to 12% on one probe, and adding VCTK raised wild false alarms.
- **Done when:** the benchmark shows a mix curve for at least real-source share and spoof diversity.
- **Size:** M.

## Phase 4: Features

### V2-40 ⬜ Handcrafted spectral v7: the features we meant to build
- **Goal:** Build the spectral and source-filter features that were planned and never implemented, and make the handcrafted model robust to noise.
- **What exists already (don't rebuild):**
  - MFCC, LFCC, CQCC
  - flatness, roll-off, centroid, bandwidth, contrast, flux
  - group delay and phase coherence
  - envelope modulation, jitter/shimmer, breath
- **What was planned and never built** (`docs/handoffs/2026-09-26_feature-engineering-map.md` §5, `docs/reports/2026-09-26_post-draft-negatives.md`):
  1. **Per-band harmonic-to-noise ratio and inter-harmonic noise floor** (`hnr_`), ranked first in the map
  2. **LPC residual / vocoder excitation statistics** (`lpc_`): kurtosis, peak-to-RMS, pitch-period regularity
  3. **Phase coherence split by harmonic order and band** (extends `pc_`)
  4. **Formant bandwidths and F1/F2 trajectory smoothness** (`fmt_`)
  5. **Breath events:** duration and rate
  6. **Vocoder harmonic regularity** (plan technique 2)
  7. **LTAS residuals and sub-band modulation spectra**
  8. MFCC deltas and spectral-contrast std (small gaps)
  9. **Noise-robust variants:** voiced-frame-only statistics, and features computed after a fixed denoiser
- **Method:**
  - Same per-column gate as v4 (`docs/handoffs/2026-09-26_handcrafted-v4-brief.md` §7), committed this time as a module.
  - Add a noise-cohort column to the gate.
- **Done when:** each family has a gate verdict, and the handcrafted model's benchmark row (esp. grad_tts, ElevenLabs, noise) is updated.
- **Depends on:** V2-01, V2-31.
- **Size:** M.

### V2-41 ⬜ Forensic detector upgrades
- **Goal:** Close the planned-but-unbuilt parts of the forensic detectors and give them labeled validation.
- **Items:**
  - Quantization traces in compression.
  - Phase breaks in splice.
  - A labeled spliced and edited set to validate splice and drift (synthesize from real speech).
  - A labeled hum set for ENF.
  - Speaking-rate regularity.
- **Done when:** each evidence detector has a measured hit rate on labeled edits, so its explanation sentences can be trusted.
- **Size:** S–M.

## Phase 5: Models

### V2-50 ⬜ Backbone fine-tune, done right
- **Goal:** A fine-tuned backbone that beats the frozen probe on held-out generators.
- **Why:** v1's full fine-tune overfit seen generators (B.3).
- **Method candidates:**
  - partial fine-tune (top N layers) or LoRA adapters
  - early stopping on a held-out *generator* fold
  - short crops matching the test duration
  - the V2-31 augmentation
  - SAM or weight averaging for flatter minima
  - Apply to whichever backbone V2-21 favours.
- **GPU:** ask before renting. v1's vast.ai + R2 scripts are reusable (`scripts/cloud/`, `.claude/skills/fleet-campaign`).
- **Done when:** a pre-declared gate on the benchmark passes or fails, with the result recorded either way.
- **Depends on:** V2-01, V2-22, V2-31.
- **Size:** L.

### V2-51 ⬜ New architectures
- **Goal:** Try at least two published anti-spoofing designs on our benchmark.
- **Candidates (verify current state of the art first; these are starting points, not endorsements):**
  - SSL + AASIST trained by us, on our data, so it's comparable to Spectra
  - layer-selection heads over SSL hidden states (SLS-style)
  - state-space (Mamba-style) back ends on SSL features
  - a raw-waveform model (RawNet2/RawGAT family) as a non-SSL control
  - a spectrogram CNN as a second non-SSL control
- **Done when:** each has a benchmark row and a one-paragraph "what it sees that the others don't" from V2-02-style error overlap.
- **Size:** L.

### V2-52 ⬜ Short-clip robustness
- **Goal:** No detector is worse on clips ≤ 4 s than on longer ones by more than a pre-declared margin.
- **Why:** The test median is 3.4 s. M5 scored 0.595 on ≤ 4 s, and Spectra zero-pads 81% of test clips.
- **Done when:** the duration cohort in the benchmark is flat within the margin for the shipped columns.
- **Size:** S–M.

## Phase 6: Fusion and decision

### V2-60 ⬜ Neural fusion layer
- **Goal:** Replace the hand-set rank blend with a small learned fusion network that beats it on the benchmark and stays explainable.
- **Why:** v1's blend weights were chosen from a short pre-declared list, and the plan's stacker never shipped. Equal-weight fusion beat the shipped rule on every labeled readout (`outputs/fusion/orchestration_ablation_v2.md`), so the hand-set weights left performance on the table.
- **Design (proposed, hand-built in PyTorch):**
  - **Inputs:**
    - each detector's logit
    - a missing indicator per column
    - routing facts: duration, SNR estimate, bandwidth, voiced fraction, laundering probability
  - **Network:** a 1–2 hidden-layer MLP, or a **gated mixture**: a small gate network weights each detector by the clip's conditions. That *is* the orchestrator, made learnable.
  - **Monotonicity:** constrain the output to increase in each detector's logit, e.g. non-negative weights on the logit path, so the network can't learn "more synthetic-looking means more real".
  - **Training:** fit only on inner out-of-fold scores (nested CV, as v1's invariant 5).
  - **Interpretability:** report per-detector contributions for each file, feeding V2-70.
- **Baselines it must beat:** v1's A3 w0.2 + E, equal-weight, and a logistic stacker.
- **Done when:** it beats all three on the benchmark under both costs, and passes on the sealed set at phase close.
- **Depends on:** V2-01, and new columns from Phases 3–5 for it to be worth it.
- **Size:** M.

### V2-61 ⬜ Probabilistic decision layer (in house, offline)
- **Goal:** Turn the fused score into a calibrated probability and a decision with a stated cost, including an "undetermined" answer.
- **Candidates (any of these are in scope):**
  - **Bayesian calibration to log-likelihood ratios**, with the Bayes-optimal threshold for a given prior and cost. This is the standard probabilistic decision layer in anti-spoofing, and the baseline.
  - **PLDA** (probabilistic linear discriminant analysis): a generative back end common in speaker verification.
  - **GEV** (generalized extreme value) tail modelling, for open-set rejection.
  - **JEM** (joint energy-based models): a classifier that is also a density model, so it can say "this input is unlike anything I've seen".
- **Relation to Jev.** TypeSafe's Jev model is tested separately in V2-63, against this layer.
- **Why:**
  - v1 optimized ranking only; calibration was cosmetic (B.7).
  - Real use needs a threshold set in advance (actDCF).
  - An honest "unsure" for out-of-distribution audio is the natural home for v1's pinned-block idea.
- **Done when:** actDCF is within a pre-declared margin of minDCF on every benchmark population, and an out-of-distribution flag is measured on held-out corpora.
- **Depends on:** V2-60.
- **Size:** M.

### V2-62 ⬜ Orchestrator v2
- **Goal:** Per-file routing that measurably changes decisions for the better, beyond v1's single suppression rule.
- **Method:** It either falls out of V2-60's gated mixture, or is an explicit cascade: run expensive models only where cheap ones disagree. Report the ablation like v1's (`scripts/orchestration_ablation.py`).
- **Done when:** the ablation shows routing beats no-routing on the benchmark.
- **Size:** S–M.

### V2-63 ⬜ Jev experiment: a hosted decision model as the decision layer
- **What Jev is.** TypeSafe AI's "System One" decision model, in limited early access since Sep 15, 2026 (all from public coverage; verify against TypeSafe's own docs at chunk start).
  - **Input:** a state (a string or structured data) plus typed questions. It does not take audio.
  - **Output:** typed answers with a probability distribution and a confidence: Choice (categorical), Score (numeric) and Noul (yes/no).
  - **Training:** RLCD, "reinforcement learning for calibrated decisions", aimed at honest probabilities.
  - **Delivery:** hosted only, through an API; no local or offline version is mentioned.
  - **Stated limits:** unreliable counting, arithmetic and comparisons, and accuracy loss on large noisy state.
- **Goal:** Measure whether Jev, given each clip's detector outputs and evidence, produces a better-calibrated synthetic probability than our own decision layer (V2-61) and neural fusion (V2-60).
- **Why it might help:**
  - Its pitch is calibrated probabilities, which is exactly v1's gap (B.7).
  - It can read the evidence sentences and routing facts as well as the numbers. That is something a numeric fuser can't do.
- **Why to be careful:**
  - It is a hosted API, so it can't sit on the offline scoring path under the v1 rule. Using it in shipped v2 is a rule change for the team to decide.
  - Our fusion inputs are mostly numbers, and arithmetic and comparison are its stated weak spot.
  - Its confidence is the vendor's claim until our benchmark measures it.
- **Method:**
  1. **Build the state.** One structured state per clip: each detector's logit and status, routing facts (duration, SNR estimate, bandwidth, voiced fraction, laundering probability), and the evidence sentences.
  2. **Design a sealed-safe set of variants:**
     - numbers only
     - numbers plus evidence text
     - evidence text only
     - ranks instead of raw logits, which may suit its weakness with arithmetic
  3. **Ask one Noul question:** "Is this clip synthetic?" Take its probability as the score.
  4. **Keep labeled examples out of validation.** If Jev accepts in-context or few-shot labeled examples, they come from inner folds only, and the outer holdout and sealed set are never shown with labels.
  5. **Report on the V2-01 benchmark** beside V2-60 and V2-61:
     - minDCF and actDCF under both costs
     - calibration error and a reliability curve
     - test-retest stability on the same input
     - latency and cost per 1,000 clips
- **Preconditions:**
  - An API key in `.env`, never committed.
  - The team agrees on what may leave the machine. Only derived numbers and text are sent, never audio; NSA test data is excluded unless NSA's terms allow it.
- **Deliverables:** `scripts/jev_decide.py`, with responses cached to `outputs/v2/jev/`, and `docs/reports/<date>_jev-decision-layer.md`.
- **Done when:** Jev has a full benchmark row beside V2-60 and V2-61, plus a written verdict: adopt as the decision layer (with the rule change), keep as an optional online mode, or drop.
- **Depends on:** V2-01, V2-60, V2-61. A quick numbers-only pilot can run right after V2-01 on v1's exported scores.
- **Size:** S–M.

## Phase 7: Ship

### V2-70 ⬜ Explanations that are checked
- **Goal:** Every evidence sentence is backed by a detector with a measured hit rate (V2-41) and a contribution from V2-60.
- **Done when:** a sample of 50 explanations is reviewed against the audio, with the review recorded.
- **Size:** S.

### V2-71 ⬜ Release v2
- **Goal:** The v2 equivalent of v1's package: tag `v2`, Docker image on the v2 rule, README "v1 → v2" section, benchmark scorecard.
- **Done when:** the image reproduces the v2 benchmark offline, and the docs site has a v2 page.
- **Size:** S–M.

### V2-72 ⬜ Write-up
- **Goal:** A report or paper-style write-up: what v1 did, what v2 learned in Phases 1–2, and the measured gains.
- **Size:** M.

---

## Appendix: open facts and loose ends from v1

These are things v1 left unknown. Each is owned by a chunk above.
- NSA's cost convention: brief vs sponsor-code. The draft number didn't settle it (`docs/consults/2026-09-26_post-draft-review_RESPONSE.md`). Owner: V2-01, by asking NSA.
- Spectra-AASIST's training data and licence: Apache-2.0 vs MIT. Owner: V2-23.
- Why WavLM false-alarms on wild real speech. Owner: V2-21.
- M5's fold-1 loss. Owner: V2-22.
- grad_tts's blindness in every handcrafted variant. Owner: V2-02, V2-40.
- The test wall's transition-band shape. Owner: V2-11.
- `src/components/GuidedTour.tsx` still states unmeasured numbers (0.1983, 48 kHz, ...). Fix or drop before the frontend is reused. Owner: V2-71.
- The Docker image is on `fusion_v1`, not the shipped rule. Owner: V2-71.
- `docs/architecture.md` and `docs/code-map.md` are Saturday snapshots. Owner: V2-00 (mark historical) or V2-71 (rewrite).
