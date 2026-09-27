# Devpost write-up (CrossExam, NSA HEARSAY at HackGT 13)

Verified against the repository records on Sat Sep 26, 2026, ~20:30, after the post-draft KEEP. Copy into Devpost as is; every number is cited in `README.md`, `docs/reports/2026-09-26_post-draft-sweep.md`, `docs/reports/2026-09-26_post-draft-negatives.md` and `docs/reports/2026-09-26_sponsor-questions.md`. Update the "final official score" sentence when NSA announces results.

---

## Inspiration

A convincing cloned voice can make a recording hard to trust. But calling a real person's voice fake can be just as damaging. The NSA HEARSAY challenge made that tradeoff concrete: its brief assigns a higher cost to false alarms. We built CrossExam to make a careful synthetic-voice assessment and show the evidence behind it.

## What it does

CrossExam takes an audio file, returns a score from 0 (real) to 1 (synthetic), and produces a per-file explanation. The report shows which analyses ran, what each found, and which findings affected the final score. It also records evidence that is useful to an analyst but too unreliable to treat as a vote.

## How we built it

We decode audio through one FFmpeg path to a consistent 16 kHz mono signal. Ten detectors then examine the same clip through deep-learning and audio-forensic techniques: learned voice representations, spectral and prosodic features, compression traces, container metadata, electrical hum, editing seams, speaker consistency, and a speech gate that abstains on non-speech.

The final score blends the ranks of three complementary models: a frozen XLS-R probe, a handcrafted acoustic-feature model, and a trained head on XLS-R. A separate anti-spoofing model, Spectra-AASIST, has a deliberately limited role: when it strongly recognizes real speech, it can lower a potential false alarm, but it cannot raise a file's synthetic score. Other forensic checks appear in the explanation without changing the score. A routing log makes those choices visible for each file.

We used generator- and speaker-grouped validation so clips from the same source would not leak across training and evaluation. The scoring pipeline runs locally on CPU; no language model or hosted API makes the audio decision.

## Challenges we ran into

The hardest problem was distinguishing a voice artifact from a dataset artifact. Our training and test audio differed in loudness, leading silence, clip length, container format, and frequency range. A model could appear accurate by learning those shortcuts instead of learning synthetic speech. We normalized the audio, matched clip lengths and frequency bands, and kept container facts out of the class score.

Combining detectors was another lesson in restraint. Equal weighting looked attractive on one holdout but missed substantially more fakes on a separate In-the-Wild stress set. We wrote down a selection rule before comparing fusion candidates and chose a blend that held up across both settings. The trained XLS-R head was weaker than the frozen probe on its own, yet improved the combined system because it made different mistakes.

We also could not verify the training data of the pretrained Spectra-AASIST model. Its validation numbers might therefore be optimistic. Limiting it to false-alarm suppression let us use its signal without giving it unrestricted control of the verdict.

## What we're proud of

CrossExam scored all 1,671 challenge test files in a full live run with no scorer errors, producing a prediction file and an explanation for each file. The live predictions preserved the submitted file's ranking and had no verdict flips.

NSA's one-time draft review scored our submitted file at a normalized minDCF of 0.0733 with an equal error rate of 3.53% on the held-out test set. Lower is better. On our own evaluation sets the same rule reached 0.0065 on an outer holdout of unseen generators and real-speech groups, and 0.228 on In-the-Wild speech we never trained on, so the real test set landed between our two proxies. The final official score is announced after judging.

After the draft review we tested four pre-declared improvements against acceptance rules written before their numbers existed: a second suppression tier, a heavier weight on the trained head, a WavLM Large probe as a second backbone, and noise-augmented handcrafted features. Two of them improved our proxies slightly, none cleared the bar we set for a one-shot decision, and the file we submitted is the file NSA scored.

## What we learned

More detectors do not automatically make a better detector. Metadata, hum, splices, and speaker drift can help explain a recording while being misleading as class evidence. A simpler model can outperform a fine-tuned one alone, while the fine-tuned model still adds value in a carefully tested ensemble. A second backbone can see the fakes the first one misses and still not improve the blend. Above all, an explainable system must say which evidence changed its answer and which evidence did not.

## What's next

Three things, in order. First, evaluate on independent labeled data beyond the challenge set, because our two proxies bracketed the real number by 11x on one side and 3x on the other. Second, noise-robust training across every detector: at 20 dB of added noise the handcrafted model loses most of its discrimination and the combined rule follows, and a last-hours attempt to fix only that model with symmetric noise augmentation recovered part of the loss while getting worse on clean audio, so it did not ship. Third, more varied real-world recordings on the real side, where false alarms cost the most. The challenge test set probably carries little added noise; recordings in the field will.

## AI use and credits

We used Claude Code to help implement and document substantial parts of the project, and Codex for plan reviews and audits. Three strategy questions went to an outside multi-model review service (VeriLM), answered by Claude, Gemini and GPT models; the prompts and answers are saved in the repository with a record of what we adopted and what we ignored. Our team directed the experiments, set the evaluation and selection rules, reviewed the evidence, and made the final decisions. No LLM runs in the audio-scoring path.
