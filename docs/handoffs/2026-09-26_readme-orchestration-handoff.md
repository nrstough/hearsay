# Handoff: the README, the explanation report and orchestration credit (Sat Sep 26, 2026, ~08:30)

**Paste this into an idle chat (the CPU chat is the best fit: it wrote most of the "what worked / what didn't" material and owns the engineered detectors).** Not a `/plan-review` rung; it is writing plus one small build.

**Purpose of this chat:** produce the two things worth 40% of the grade that nobody owns yet: the README in our own words (20%) and the evidence that our per-file orchestration and explanations are real (the 20% diversity-and-depth bucket, including the "agentic orchestration" bonus). The fusion consult's Item 8 says what judges want; `docs/consults/2026-09-26_fusion-strategy_RESPONSE.md` item 10 summarises it. Deadline: a complete draft by **Sat 20:00**, final polish Sun 05:00–07:30.

Read first: `README.md` (placeholder), `CLAUDE.md` (the AI-use disclosure and the "what we built vs what AI did" section, both graded), `docs/STATUS.md`, `docs/architecture.md`, `docs/code-map.md`, every report under `docs/reports/`, and `docs/handoffs/2026-09-26_frontend-contract.md` (the per-file JSON that already carries evidence sentences and a routing log).

## What the README must contain (in our own words, no boilerplate)

1. **What HEARSAY is and how it decides**, one screen: any audio in, P(synthetic) out, with the routing log and per-detector evidence; the system-flow figure from `docs/img/architecture-flow.svg`.
2. **Approach and architecture**: the detector contract, the deep detectors (frozen XLS-R probe with layer 7, Spectra-AASIST as a false-alarm suppressor, the fine-tune that did not clear its gate and why), the eight engineered techniques, the fusion rule as frozen at 08:13 and the reason equal-weight fusion was rejected (In-the-Wild misses doubling), the non-speech gate and the pinned-block default-answer policy with the direction-robust argument.
3. **Validation**: generator- and speaker-grouped folds, the outer holdout read once, In-the-Wild as the eval-only stress set, the resolution-floor argument (a holdout gap under 0.15 is noise), and the draft-review decoding table.
4. **What worked** and **what did not**, each with its mechanism. The material exists: the five shortcuts (level, silence, tiling, container, the 7.2 kHz wall), the container-is-the-label finding, band matching (90% → 42% of test files called synthetic), length parity, LightGBM over logistic, the v4 families that see pro_diff and ElevenLabs, the corpus-cue columns that could not be pruned, the augmentation trade-off, VCTK helping once band-matched, ENF and splice as evidence only, compression as pipeline-not-class, speaker drift finding that fakes are the most self-consistent voices, the fine-tune losing to the frozen probe, the equal-weight fusion rejected, the pretrained model's possible in-sample rows. Every null result gets one line with its mechanism.
5. **Numbers table**: one row per detector and per fusion rule with inner OOF, holdout, In-the-Wild (both cost weightings) and the test share; sourced from `submissions/log.csv` and the model `meta.json` files, never retyped from memory.
6. **Reproduce it**: setup, the runner command, the Docker command, the parity numbers, and where each artifact lives.
7. **AI use and credits**: copy the disclosure from `CLAUDE.md` after updating it (it stops at the CPU chat's work; add M1 TSVs, M1b, band-match, both consults, M3, M5, v4, the runner, API, Docker, and these docs), and fill "what we built vs what AI did" per component. Credit every framework, model and dataset with license.

## Orchestration credit: make routing visible and measured (about 2 h)

The runner already emits a routing log per file from detector features. Judges want proof that routing changed decisions and that the delta was measured:

- A short `scripts/orchestration_ablation.py` that runs the shipped pipeline on the holdout and In-the-Wild rows with the router on and off (off = every detector always runs, no gate, no M3 suppression, no default block) and reports minDCF for both, plus how many files each rule touched. A measured null delta is itself rubric-positive.
- Five to ten worked examples in `docs/reports/2026-09-26_worked-examples.md`: real test files, their routing log, each detector's evidence sentence, the fused score, and one paragraph on why the system said what it said. Pick them to show variety: a confident real, a confident fake, a file where M3 suppressed a false alarm, a file with a seam or hum flagged as evidence, a low-voiced-fraction file near the gate, a disagreement case.
- One paragraph in the README on the abstention path (the pinned block) as part of orchestration.

## Rules

Numbers come from files, never from memory; cite the file for each. No hardware or demo wording (the docs test bans it). Do not use the bare word CSV in living docs (write `.csv` or TSV) and mention no prior other than 0.3. Stage only your own files (`README.md`, `CLAUDE.md` disclosure sections, the new report and script, `docs/STATUS.md` rows); never `-A`; do not push. Keep `uv run pytest -q tests/test_docs_consistency.py` green after every README edit. Report to the oversight chat when the draft exists.
