# NSA HEARSAY: rules and sponsor questions (Fri Sep 25)

**The sponsor brief is now in `docs/nsa-challenge.md` (PDF: `docs/nsa-challenge.pdf`) and is authoritative.** It adds two submission requirements: a Docker image that runs inference on the test set, and a README covering approach, architecture and run instructions. The 20% diversity score is itemized per detector type, with a bonus for agentic orchestration.

## Rules from the kickoff slides (photos taken by Nathan, Fri Sep 25 evening)

**Data**
- Audio files split into training and testing, about 50% real and 50% synthetic.
- Labels ("answers") are given for training.
- Clips are over 2 s and English only.
- Some countermeasures, e.g. added noise. Noise added to real or synthetic audio keeps its label.
- A held-out dataset is used for testing.
- No determination of intent, and no partially synthetic clips.

**Definitions**
- Miss: not identifying synthetic speech.
- False alarm / false positive: identifying real speech as synthetic.
- False alarms need analysis, so they cost more: the penalty factor is 4.

**Metric**
- Detection Cost Function (MinDCF) as in ASVspoof 5 (Track 1 baselines, `asvspoof5-baselines`).
- Formula as transcribed from the slide: `DCF = C_miss·(1−P(attack))·P(miss) + C_FA·P(attack)·P(FA)`, minimized.
- It is a minimum over thresholds, so **only the ranking of our scores matters**.

**Judging**
- 60% MinDCF.
- 20% diversity in techniques (quality, innovation, depth).
- 20% documentation in GitHub, "in your own words".

**Submission**
- A tab-separated file named `TeamName_predictions.tsv`.
- Two columns with header `filename<TAB>cm-score`. The score is a probability: 0.0 = real/bona fide, 1.0 = synthetic. No logarithms or log-likelihood ratios. Every test file must have an entry.
- Send it as a DM in the NSA x HexLabs Discord channel.
- Optional one-time feedback on a test-data submission.
- Final submission is due **no later than 8 AM** (Sunday).

**Generators named on the "Text to Speech: inexpensive and ubiquitous" slide** (likely what the NSA fakes come from):
- ElevenLabs: instant voice cloning from 10 s to 1 min of audio; professional voice cloning from 30+ min.
- Amazon Polly, Grok TTS, Google Gemini.
- Open-weight models on Hugging Face, e.g. Qwen-3 (Qwen3-TTS).

Black-box detectors are discouraged, not forbidden (said aloud, per Nathan).

**What they're looking for**
- Unique perspectives: synthetic-speech consistency, breathing, pauses, emotion, prosody; read transcript vs. real-time speech; correlation, stacking, signal processing, artifacts, metadata, statistics.
- Test yourself: generate your own synthetic speech.
- Tell us what didn't work, and what did work and how it can be explained.
- Minimize black-box detectors.

## What the presenter said aloud (Nathan's notes, Fri Sep 25)

- Class balance is **70% real / 30% synthetic**. The slide photo reads "~50% real, ~50% synthetic". We go with 70/30 (π_synth = 0.3) and will check it against the NSA train labels.
- Clips are 3 s or longer, with a likely upper bound around 30 s. The slide says "more than 2 seconds".
- Labels are discrete, pass or fail: every clip is fully synthetic or fully real, with nothing in between. Added noise does not change the label.
- One-time early submission to check whether we're on track against the test data.
- Diversity means "not just SSL": fine-tuning, depth, multiple layers, engineered features.
- Ranking preference, using the bullseye slide: accurate and precise > accurate but not precise > neither. The worst case is **high precision, low accuracy**, i.e. consistently and confidently wrong. For us: submit calibrated probabilities, not saturated 0/1 scores, and never be confidently wrong on a whole subpopulation (e.g. silence or a noise type).
- Black-box detectors are discouraged, not forbidden.
- Outside data was neither allowed nor forbidden, so we use it and disclose it. ElevenLabs was mentioned as a way to generate our own test fakes.

With π_synth = 0.3 and C_FA = 4, normalized DCF = (4·0.7·P_FA + 0.3·P_miss) / 0.3 = **9.33·P_FA + P_miss**. False alarms dominate. The best constant is "always real" (normalized 1.0).

## What this changes (implemented)

- `hearsay.metrics`: normalized minDCF with C_FA = 4, C_miss = 1, π_synth = 0.5 _(corrected Sep 26: now π_synth = 0.3, confirmed by the NSA instructions, "approximately 70% of the files are real")_. It is the selection metric for layer choice, bake-off, fusion and submissions. EER is logged alongside. The `validation_score` column in `submissions/log.csv` holds normalized minDCF.
- `hearsay.submission.write_submission` writes the TSV above.
- The constant rollback submission is all 0.0 ("always real"; 1.0 only if π_synth > 0.8). Its normalized minDCF is 1.0, which every model must beat.
- Submissions are calibrated posteriors, sigmoid(LLR + logit π). Because MinDCF sweeps the threshold, the −ln 4 decision shift is opt-in (`--cost-shift`) and does not change the score.
- The one-time feedback is valuable: use it on the first strong learned model, not the constant one.

## Conflicts to resolve

- **Class balance.** A pasted note (source not filled in) said 70% real / 30% synthetic. The slide says ~50/50. We use the slide's number until the NSA train labels are counted. _(Resolved Sep 26: the NSA instructions say about 70% real, so π_synth = 0.3.)_ If the train split really is 70/30, normalized minDCF becomes 9.33·P_FA + P_miss.
- **Prior in the DCF.** The slide weights the false-alarm term by P(attack). That is not the usual ASVspoof 5 form (miss/FA roles and the P(attack) value differ). At P(attack) = 0.5 both readings give the same number. _(Sep 26: the sponsor's scoring code uses Pspoof = 0.5, Cmiss = 1, Cfa = 4 and treats a higher score as bona fide; see `docs/plan.md`, "Score direction".)_

## Questions for the sponsor

Post these in writing (NSA x HexLabs Discord) and record the answers below.

1. **Outside data.** May we train on public datasets (ASVspoof 2019 LA, MLAAD, In-the-Wild, ReplayDF) as well as the provided training set?
2. **Pretrained checkpoints.** Do pretrained anti-spoofing checkpoints (e.g. `lab260/Spectra-AASIST`) count as outside data? Are general SSL speech models (XLS-R, WavLM) fine?
3. **Pre-event downloads.** We downloaded public weights and datasets before 8 pm Friday; no code was written or run on them. Is that OK?
4. **DCF parameters.** Which P(attack) goes into the MinDCF (0.5 to match the data, or ASVspoof 5's 0.05)? Is the 4× penalty on false alarms, i.e. real called synthetic?
5. **Filename column.** Basename or relative path? Extension included?
6. **Test balance.** Is the test set also ~50/50?
7. **Feedback.** What does the one-time feedback return: the MinDCF on the full test set?
8. **Track stacking.** Can a Shipyard (hardware) project also submit to NSA HEARSAY?

## Answers

| # | Answer | Source / time |
|---|--------|---------------|
| metric | MinDCF, false alarm ×4, label 1 = synthetic, not log-loss | kickoff slides, Fri Sep 25; confirmed by Nathan |
| format | TSV `filename<TAB>cm-score`, probability, no LLR | kickoff slides |
| direction | **1.0 = synthetic** ("Output a synthetic probability score (0.0 to 1.0), 1.0 means synthetic"). We submit this and never flip; the sponsor's ASVspoof5 code treats higher = bona fide, so the draft review doubles as a check (a strong model scoring near 1.0 → raise it with NSA) | NSA brief; reaffirmed by Nathan, Sat Sep 26 |
| balance | 70/30 real/synthetic said aloud; slide reads ~50/50; verify on NSA train labels | presenter, per Nathan |
| 1 | Yes: the brief lists ASVspoof 2019/2021/5, WaveFake, In-the-Wild, MLAAD, ADD, VCTK, LibriSpeech, VOiCES for training/calibration | `docs/nsa-challenge.md` |
| 2 | Yes: "any open-source or commercial tools"; AASIST, RawNet, wav2vec 2.0/WavLM/Whisper front ends suggested | `docs/nsa-challenge.md` |
| 3 | | |
| 4 | | |
| 5 | Filename including its extension | `docs/nsa-challenge.md` |
| 6 | | |
| 7 | | |
| 8 | | |
