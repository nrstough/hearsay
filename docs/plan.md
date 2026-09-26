# HEARSAY: The Plan

HackGT 13, Sep 25 to 27, 2026. Written Fri Sep 25, 1:30 pm, from the VeriLM scoping memo (three models, max effort, key facts grounding) plus the master doc. This is the version the team works from. The master doc in Claude Docs is the long form; the memo (`VeriLM Memo.md` in this folder) is the evidence.

## The one-paragraph version

We build an audio deepfake detector and wrap it in a tabletop "listening head." One model, two outputs: a CSV of synthetic-likelihood scores on NSA's hidden test set, and a live head that turns toward whichever of two phones is talking and goes red on the cloned voice. The detector is a fine-tuned XLS-R 300M. We ship a valid CSV inside the first hour and never go without one. Fine-tuning happens on a rented H100, never on the laptop. The biggest risk is that audio through a phone speaker and a loud hall wrecks the detector, so Friday night includes a structured replay capture on the real rig and the demo has a disclosed wired fallback.

## What changed from the master doc

| Was | Now | Why |
|:---|:---|:---|
| M2 hand-crafted features before fine-tuning | Fine-tune (M5) promoted to right after the first CSV; M2 is one 30-minute gate | SSL features make hand features mostly noise; fine-tuning is where the gain is |
| Gradient-boosted stacker (M4) | Regularized logistic fusion + Platt scaling | Boosted stacker on a small validation set is an overfit trap |
| 3-window alert | 2-window alert (~6 s of evidence) | 8 s feels sluggish on stage |
| "Frozen WavLM or XLS-R" | `facebook/wav2vec2-xls-r-300m`, with a bake-off as safety valve | Best frozen front end (17.4% mean EER vs WavLM Large 20.6%), most codec- and noise-robust |
| 4-mic array | 2 mics, unless a ReSpeaker-class USB unit is already in hand | Two phones a couple of feet apart are ~10 samples of TDOA apart at 48 kHz; 2 mics separate them with margin. Hand-wiring 4 channels is complexity for no reason |
| Laser lock | Cut | Alignment, jitter, safety, adds nothing beyond the LED |
| Unstructured hour of replay recording | 480-clip matrix, ~32 min of playback | 2 phones × 3 positions × 2 levels × 2 classes × 20 clips |
| M6 type head "if labeled" | Cut unless labels are explicit and everything else is frozen | Distraction |
| No constant-prior CSV | Constant-prior CSV in hour 1 | Rollback artifact; never be without a submission |

Cut outright: scratch SSL pretraining (no world where it wins in 36 h), RL (no sequential decision problem; the batch-efficiency premise is wrong), custom AASIST back end, laser.

## Model ladder (reordered)

| Rung | Deliverable | M3 Pro | H100 | Cut rule |
|:---|:---|:---|:---|:---|
| M0 | Any format → 16 kHz mono float32; chunking; generator/speaker-disjoint split; **constant-prior CSV** | 1.5 to 2.5 h | | never |
| M1 | Frozen XLS-R 300M → mean pool → logistic regression → **first learned CSV** | 5k-clip probe 10 to 25 min; full pass ~1 to 2 h | minutes | never |
| M5 | Fine-tuned XLS-R (partial first, full if stable), clean/channel-mixed curriculum | ~53 h, infeasible | 0.6 to 1.5 h per 3-epoch run | cap at 1 run if behind |
| M3 | Off-the-shelf detector score (`lab260/Spectra-AASIST`, verify it exists) as a fusion stream | 45-min timebox | | drop on friction |
| M4 | Logistic fusion + Platt calibration | 0.5 to 2 h | | degrade to mean fusion |
| M2 | Hand-crafted side branch, one 30-min gate | ≤1 h | | first substantive cut |
| M6 | Type head | 1 to 1.5 h | | first overall cut |

Cut order if behind: M6 → M2 → M3 → M4 to mean fusion → M5 capped at one run. Never cut M0, M1, grouped validation, or calibration.

Live path: the best XLS-R 300M checkpoint at demo time (a fine-tuned 300M costs the same at inference as a frozen one). 4 s windows, 2 s hop, alert on 2 consecutive positive windows. Latency gate: 750 ms 95th-percentile per window. If XLS-R misses it, live drops to WavLM-base or off-the-shelf W2V2-AASIST while offline keeps XLS-R.

All M3 Pro timings are analytic (133 GFLOPs per 4 s window), not measured. Run a 100-window microbenchmark in the first two hours before planning around any of them.

## Data

Rule: sample by generator and channel condition, cap near 100k training clips, hold out by generator. More volume from the same sources does not help and costs three times (extraction, fine-tune, validation).

| Priority | Dataset | Size | Role |
|:---|:---|:---|:---|
| P0 | ReplayDF | 132 h full; pull a few-thousand-clip subset (~5 GB) | Models the exact speaker-air-mic failure. Live path needs it |
| P0 | MLAAD + M-AILABS | gated on HF, request access now | 54 languages, 127+ architectures; best unseen-generator value. Sample by architecture, not volume |
| P0 | ASVspoof 2019 LA | ~7 GB, 25k train clips | Canonical baseline |
| P0 | In-the-Wild | 8 GB | **Held-out stress set only. Never train on it** |
| P1 | ASVspoof 5 | 142 GB | Skip today; codec diversity if there is time |
| P2 | WaveFake, Codecfake | 29 GB / huge | Volume only; skip |

Today's pull is ~23 GB including weights. Do not route public data through R2; the vast box downloads it straight from source in parallel with the laptop. R2 or scp only for what exists nowhere else: the NSA set and our replay captures.

Leakage: hash-dedupe NSA train/test against everything public. In-the-Wild is celebrity-sourced and the highest contamination risk if NSA drew from public figures.

If outside data is disallowed: M0 to M4 never needed it; fine-tune M5 on NSA only with heavy augmentation (RawBoost, codec round-trips, RIR). Ask the sponsor explicitly whether pretrained anti-spoof checkpoints count as outside data.

## Weights to download before 8 pm

- `facebook/wav2vec2-xls-r-300m` (core)
- `microsoft/wavlm-large` (bake-off challenger and load-failure fallback)
- `microsoft/wavlm-base` (live-path speed spare)
- `lab260/Spectra-AASIST` (M3 stream; verify it exists and loads first)

~3 GB total.

## Bake-off (after the first CSV, ~1.5 to 3 h on the M3 Pro)

XLS-R 300M vs WavLM Large vs one of WavLM-base/HuBERT Large on an identical 2 to 5k grouped, class-balanced NSA subsample, identical 4 s crops, identical mean pooling and logistic head. Pass rule: the challenger replaces XLS-R only if it wins by ≥1.0 absolute EER point and loses ≤2 points on the replay stress subset. Ties go to XLS-R.

## Replay and the live demo

Physical replay drives a top detector's EER from 4.7% to 18.2%; RIR retraining only gets it back to ~11%. Failures are asymmetric: fakes pass as real while real stays stable, so lowering the threshold buys false alarms, not safety.

Friday-night capture (~23:15 to 00:15 with the EE): 2 phones × 3 positions × 2 levels × 2 classes × 20 clips = 480 captures. Split by source utterance and speaker before replaying so no clip crosses train/validation; hold out one phone-position cell.

Training mixture for M5: ~30% clean, ~40% RIR/ReplayDF-style channel simulation, ~20% codec round-trips (Opus, AMR, μ-law, AAC, MP3), ~10% RawBoost/noise/clipping. Applied to both classes. Keep a clean-only validation score because the NSA test set is probably clean.

Honest ceiling: 80 to 90% balanced accuracy on the fixed, matched two-phone rig; 70 to 80% on an unmatched phone or room. Go/no-go: ≥80% balanced accuracy on untouched local replay clips and ≤10% genuine false alarms. Below that, use a disclosed 3.5 mm wired detector feed and keep mic-based localization, and say so.

## Offline test-time compute (the CSV path has slack)

1. Score every speech window, aggregate mean + max of logits. Always on. Pick the aggregator from the sponsor's label semantics.
2. Two-to-three-model logistic fusion (fine-tuned XLS-R + frozen-LR stream + Spectra-AASIST). Only if locked validation improves ≥0.5 point.
3. Codec TTA, ~5 to 6 views, only if the drop-time inventory shows codec diversity in the test set. Skip if it looks clean.
4. Stochastic weight averaging of 3 to 5 neighboring fine-tune checkpoints, if training is stable.
5. XLS-R 1B offline stream: last priority, ~0.66 point gain.
6. Pseudo-labeling on the test set: do not.

## Hardware

Mic array: 2 mics, d ≈ 0.10 m, GCC-PHAT with parabolic interpolation (~1 to 2° resolution). Detector input is a selected raw channel; beamforming can mask the artifacts the detector keys on. Feed processed audio to detection only if a Saturday A/B on the replay holdout improves.

Add-ons for the two hardware people, ranked by story value per hour:

1. Servo analog gauge REAL → SYNTHETIC. 2 to 4 h, ~$10, unanimous.
2. Grandma-device mockup puck: RGB LED + coin vibration motor on the same serial protocol. 3 to 6 h. Turns the second product framing into hardware.
3. Fixed phone docks with per-dock source LEDs. 2 to 3 h. Stabilizes replay geometry, which protects detector accuracy.
4. Push-to-challenge button: a judge plays any clip and sees a live P(synthetic). 1 to 2 h, mostly software.
5. Minimal enclosure. 3 to 5 h, only if ahead.

Every add-on shares the laptop and the serial link. Check each for whether it needs a second capture path before assigning it.

## Grandma device (requirements only, no build this weekend beyond the puck)

Phone-call audio is codec-limited, not air-replayed; codec augmentation is the mitigation and XLS-R is the most codec-robust frozen front end. Requirements: robust at 8 to 16 kHz; 3 to 6 s windows with a rolling decision over 10 to 20 s of speech; alert within 8 to 12 s; multi-window confirmation with hysteresis and an explicit "uncertain" state; false positives on the order of one per tens of hours of genuine calls, because false alarms destroy an elderly user's trust. A 300M model runs on a laptop or mains hub; a phone app needs quantization; a microcontroller-class device is not feasible with this model.

## Multilingual

Yes, one model, bounded claim. XLS-R is cross-lingual; MLAAD covers 54 languages; ReplayDF adds six. Cheapest demo: English plus Spanish and German (both in M-AILABS), 10 to 20 held-out clips per language through the actual rig, ~1 h Saturday evening, zero training. Demonstrate that the pipeline handles multilingual audio; do not claim equal accuracy across languages.

## Risks that will bite

- **Metric mismatch.** Could be AUC, EER, thresholded accuracy, a DCF cost, or log-loss. Keep raw logits, calibrated probabilities, and hard decisions separately. Submit calibrated probabilities; have a rank-preserving raw-score CSV ready.
- **Class imbalance.** Anti-spoof sets run ~90% spoof. Weighted sampler at ~50% bona fide per batch. Never quote accuracy. Never assume test prior matches train.
- **Split hygiene.** Generator- and speaker-disjoint. Check the duration-by-class histogram in hour one; clip length correlating with label is a classic shortcut.
- **Silence and music.** Detectors shortcut on silence duration. VAD-gate the live scorer; offline, emit a prior-like score for non-speech and flag it. Test one music file and one silence file before every CSV.
- **Format pitfalls.** One FFmpeg path to 16 kHz mono float32 for train and test. Never denoise or loudness-normalize test audio; codec artifacts are signal. Unit-test empty, truncated, NaN, odd sample rate, MP3/M4A, VBR. Every file yields one finite score in exact row order.
- **Score direction.** Spectra-AASIST reports bona-fide-vs-spoof. Assert with known examples that the exported value increases with synthetic likelihood before any CSV ships.
- **GPU cost figures in the memo were graded wrong.** Order of magnitude (under $100) is safe; check live vast.ai prices. Use datacenter-verified hosts, checkpoint every run, cap spend near $75.

## First two hours after the drop (Fri 20:00 to 22:00)

| When | Do |
|:---|:---|
| 0:00 to 0:20 | Read metric, label semantics, outside-data policy, submission schema. Send the sponsor three written questions: outside data? metric? type labels? |
| 0:20 to 0:40 | Inventory: N, formats, sample-rate and duration histograms by class, class balance, decode failures on a stratified sample |
| 0:40 to 1:00 | Build grouped splits. Write the **constant-prior CSV** through the actual test loader. Save as rollback |
| 1:00 to 1:30 | Run Spectra-AASIST on a labeled sample, verify score direction, fit a provisional Platt map |
| 1:30 to 2:00 | Launch the M3 latency microbenchmark; start frozen-embedding extraction for the bake-off |

No custom architecture work in this window.

## Model-owner timeline (~26 h)

**Friday 20:00 to 02:00.** Two-hour runbook → constant + first learned CSVs. Bake-off on the M3 Pro. Replay capture with the EE ~23:15 to 00:15. Spin up and verify one datacenter H100. Kick full embedding extraction overnight.

**Saturday ~09:00 to 24:00.** Bake-off readout, front-end lock. Launch M5 run 1 (partial fine-tune) in the background. Build M4 fusion + calibration + all-window scorer while it trains. Run 2 with the augmentation mixture. Ingest the replay capture, lock its untouched split, run 3 with replay adaptation. Select one core checkpoint, fit separate clean and replay calibrators, freeze a stable CSV. Integrate the live scorer on the rig with the EE, tune the threshold. 30-min hand-feature gate, 45-min Spectra-AASIST fusion timebox. Evening: multilingual demo clips, conditional TTA. **No new architecture after ~22:00.**

**Sunday 06:00 to 08:00 (hacking ends 8 am).** Freeze weights and calibration. Final CSV: fused, all-window, calibrated, music/silence sanity files checked. Schema, row order, finite range, reproducibility. Set live threshold, rehearse clean/replay pairs, confirm the wired fallback. Archive code, hashes, licenses, attribution.

Hard gates: no learned CSV by Friday midnight → stop the bake-off, ship calibrated Spectra-AASIST. Fine-tune not beating the frozen probe by Saturday midday → freeze the probe, move to replay robustness. Live model misses 750 ms → WavLM-base or W2V2-AASIST live. Through-air balanced accuracy under 80% → disclosed wired fallback.

## Before 8 pm today

1. Run the Claude Code setup prompt (done: `~/Projects/hearsay` exists).
2. Request MLAAD access on Hugging Face.
3. Create a vast.ai account; find a datacenter-verified H100 with ≥100 GB disk.
4. Post Q&A: outside training data, pretrained anti-spoof checkpoints, pre-event weight downloads, Shipyard + NSA stacking.
5. Download weights (list above), ASVspoof 2019 LA, In-the-Wild, a ReplayDF subset.
6. Tell the hardware guys: 2 mics, gauge first, puck second, docks third, no laser.
7. Pick the two or three phones for replay capture and the demo.
8. Make or find a cloned-voice clip and its real source (a teammate's own voice, with consent).

## Rules that never move

- Always have a valid CSV. Log every CSV with its validation score in `submissions/log.csv`.
- Validate by generator, not by clip.
- Never train on In-the-Wild or the validation split.
- Keep a clean-only validation score alongside any replay-augmented one.
- Fine-tuning happens on the H100, never on the laptop.
- Cap any rung at a few hours. When a time box runs out, stop and report.
- Credit every model and dataset in Devpost. The rules require it.
- Everything on the live path runs locally. No hosted API.
