# Handoff — post-draft CPU lane: the negative results, written up for the README (Sat Sep 26, 2026, ~17:25; revised after the Fable second opinion)

**Purpose of this chat:** turn tonight's measured negatives into one report the README chat can cite, `docs/reports/2026-09-26_post-draft-negatives.md`, by ~19:00. Optionally, only if that is done and the CPU is free, run the handcrafted noise-twin experiment (v6) as a candidate for the gate chat. CPU only; you never touch the GPU, fusion, the runner or the shipped bundle `models/hc_selected`.

**What changed since the first version of this handoff:** the noise twins were this lane's main job. Two independent opinions (`docs/consults/2026-09-26_post-draft-review_RESPONSE.md`, "What to do" and "Round 2") now put its test-set effect at zero: the channel lane's per-feature table shows the test set's noise-floor percentiles beyond the clean end, so additive noise is a measured weakness the test set does not exercise. It is worth a README paragraph, not a switch. The write-ups are worth more: 20 points of the grade are "what worked and what did not", and tonight produced five measured negatives nobody has written down.

## Context

- NSA's draft review returned **minDCF 0.0733, EER 3.534%** on the shipped rule; one team is ahead at 0.0584. The consult records (`docs/consults/2026-09-26_post-draft-review_{CONSULTATION,RESPONSE,RESPONSE_fable}.md`) hold every number below; cite them by file and section, and never invent a number.
- **The negatives to write up** (each with the number and the mechanism, README voice: plain, short sentences, no marketing):
  1. **More heads on the same backbone are dead on arrival.** XLS-R layer probes on the M1b recipe (Fable memo, option 10): layer-7 baseline reproduces M1b (inner 0.2649, holdout 0.0717, In-the-Wild 0.342); mean of layers 5–9 gives holdout 0.052 but In-the-Wild 0.368 and doubles the In-the-Wild false-alarm rate (0.8% → 1.55%); concatenations of 6–8, 5/7/9 and 5–9 give In-the-Wild 0.353 / 0.430 / 0.399. Mechanism: of the shipped rule's 158 In-the-Wild misses, M1b alone would flag 1, handcrafted 2, M5 33, Spectra 107; a head on M1b's backbone inherits its blind spot.
  2. **Fusion re-slicing has a measured ceiling.** Cheating on purpose (weights chosen on In-the-Wild itself, (0.3, 0.2, 0.5)) reaches In-the-Wild 0.186, a 0.04 gain that costs the holdout two false alarms; a logistic stacker fit on In-the-Wild scores 0.257, worse than the shipped rank blend; the admissible non-negative stacker on inner non-LJ rows gives weights (0.66, −0.04, 0.37) and In-the-Wild 0.222 with holdout 0.029. The second M3 suppression tier is inert at the metric on every split (consult chat verification, Round 2 table). Calibration cannot move minDCF at all.
  3. **The handcrafted column is the readable column, not a vote.** Admissible stacker coefficient −0.04; test-set Spearman with M1b 0.21 against 0.79 on the holdout, so on the test set it reads the envelope shift. Tie this to the existing README text on the test-domain offset.
  4. **The excluded detector could shrink the miss tail, and stays excluded.** Promoting files with Spectra margin > 6.9 and base < 0.5 recovers 76 of 158 In-the-Wild misses at zero added false alarms (In-the-Wild 0.152), but the holdout shows two real files promoted [2, 11], the evidence is a set Spectra's authors evaluated on, and the suppression-only rule is a principle stated before any number. Loosening the threshold from −3 to −1.5 improves every proxy for the same reason and is skipped for the same reason.
  5. **The cost convention is unknowable from the returned pair, and we hedge by requiring gains under both costs.** The sponsor's scorer was edited from ASVspoof5's defaults to Pspoof 0.5, C_FA 4, and in that code the 4× multiplies accepted spoofs; the spoken rule puts it on false alarms. (0.0733, 3.53%) fits both: e.g. 3 false alarms + 25 misses, or 5 misses + 40 false alarms. The EER forces the two profiles to be mutually exclusive (at a single threshold both error rates cannot sit below the EER), so our own earlier decomposition (1.3% / 1.5%) was impossible.
  6. **The proxies against the one measurement:** the table in the CONSULTATION's second addendum (holdout optimistic 11× on minDCF and 17× on EER; In-the-Wild pessimistic 3× and 1.6×; the test sits further toward the wild end than λ̂ = 0.51 suggested).
  7. **The M5 weight.** A3 with M5 at 0.4 improves In-the-Wild by 0.028 under both costs with the holdout unchanged and inner slightly worse (0.135 → 0.137 brief, 0.300 → 0.337 sponsor-cost); post hoc, because the pre-declared sweep stopped at 0.2. Report it as measured; whether it ships is the gate chat's and Nathan's, and your paragraph must say so and be updated by 21:30 either way.
- **The optional experiment (v6, noise twins)** if and only if the report is committed and the CPU is free before 18:20: the shipped v5b recipe (`docs/reports/2026-09-26_handcrafted-v4.md`, "v5: training-side augmentation") plus a 20 dB white-noise twin per class on fit-side rows via a new `--noise-frac` / `--noise-snr-db` option in `scripts/extract_handcrafted.py` (same per-row `default_rng(seed + row)` draw as laundering and tilt, recorded in `hc_augment`, applied before band match on both classes; `hearsay.augment.add_noise(x, rng, snr_db)` exists). Re-extract the training sample as `nsa_train_sample_v6aug` (~25 min at 4 workers), train exactly like v5b with `--extra-rows-from`, out-name `handcrafted_v6`, score In-the-Wild with `scripts/eval_bundle.py` on the seed-200 v4 features into `_itw_handcrafted_v6.csv`, and run the pre-declared diagnostic (the channel lane's 500-clip cohort under 20 dB noise through `scripts/m3_probes.py`'s perturbation path, handcrafted only; pass bar AUC ≥ 0.90 with clean AUC within 0.005 of 0.998). Numbers to sit beside, v5b clean rows: inner 0.469, LibriSpeech real 0.59, clean holdout 0.137, holdout LibriSpeech 0.16, ITW P_FA / P_miss 0.40% / 96%, ITW AUC 0.77. Do not re-pin `models/hc_selected`. If it is not exported by 18:50, report it as not run.

## Working branch / worktree

`main` in `~/Projects/hearsay`, shared. Your files: `docs/reports/2026-09-26_post-draft-negatives.md`; if the experiment runs, `scripts/extract_handcrafted.py` (the option only), one test, `outputs/handcrafted/nsa_train_sample_v6aug.csv`, `models/hc_lgbm_<stamp>/`, the two exports, and `docs/reports/2026-09-26_post-draft-hc-noise.md`. Stage only those; never `git add -A`; do not push; tell the README chat and the gate chat the commit hash.

## Environment / setup

```bash
cd ~/Projects/hearsay && uv sync
```

## What to do next

1. Read the three consult records and the Round 2 table. Write `docs/reports/2026-09-26_post-draft-negatives.md` with the seven items above, one short section each: what was tried, the number, the mechanism, the file it comes from. Mark item 7 "pending the 19:30 decision" and update it when the gate chat reports.
2. Message the README chat ("README" lane, address in `docs/handoffs/2026-09-26_docs-site-handoff.md`) with the path so it can fold the items into "What did not work"; the README chat edits the README, you do not.
3. Run `uv run pytest -q tests/test_docs_consistency.py` (52 passed at last count) after committing, because the README chat will cite you.
4. Only then, and only before 18:20: the optional v6 experiment above. Otherwise stop at 19:00.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test: `uv run pytest -q tests/test_docs_consistency.py` → 52 passed; full `uv run pytest -q` → all passing and `uv run ruff check .` clean if you touch a script.
- Self-check for the optional experiment: with `--noise-frac 0` the extraction is byte-identical to a v5-style run on the same rows.
- At-risk: nothing for the write-up (tracked). For the optional experiment: `outputs/handcrafted/nsa_train_sample_v6aug.csv` (gitignored, ~25 min to regenerate; NOT archived), the bundle (gitignored; NOT archived), the exports (regenerable); copy the bundle's `meta.json` numbers into its report.
- LightGBM and torch must never be imported in one process (macOS OpenMP crash); `train_handcrafted.py` is the only LightGBM importer.
- In flight: heads chat (GPU: WavLM), gate chat (manifest, sweep, packet 19:30, sleep-equals-ship copy), README and docs-site lanes.

## Analytical notes

- Score direction 1.0 = synthetic everywhere. Augment both classes or neither; readouts on clean rows; the holdout never trained on; In-the-Wild eval-only.
- Every number in the report is cited to a file; where a number exists only in the Fable memo, cite that memo and say the consult chat verified the M5-weight and second-tier rows from the exports and did not re-run the layer probes.
- Tone: the README's voice. A negative result with its mechanism is graded content; a negative result without the mechanism is not.

## Pointers

- `docs/consults/2026-09-26_post-draft-review_RESPONSE.md`, `…_RESPONSE_fable.md`, `…_CONSULTATION.md`; `docs/reports/2026-09-26_channel-robustness.md` (§A per-feature positions, §E the noise finding); `docs/reports/2026-09-26_handcrafted-v4.md`; `README.md` "What did not work" (the section your items extend); `src/hearsay/augment.py`, `scripts/{extract_handcrafted,train_handcrafted,eval_bundle,m3_probes}.py`.
