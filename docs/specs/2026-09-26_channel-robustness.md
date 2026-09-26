# Run spec: channel robustness on the frozen deep path (Sat Sep 26, 2026, 12:30)

**Rung:** channel robustness, from `docs/handoffs/2026-09-26_channel-robustness-handoff.md`. P1 frozen by Nathan ~12:25 with a compressed process: P2 and the plan written together, one Codex plan review in the background while A (read-only) starts, audits once at the end. **Hard stop 16:00.** Nothing here touches the 12:03 draft send or the shipped TSV.

## Problem

Both deep detectors reach 0.007–0.07 minDCF on the NSA holdout and 0.23–0.34 on In-the-Wild real speech. Training real speech is all clean read speech; the 1,671 test files all share a ~7.2 kHz wall that may be one codec round-trip. How much In-the-Wild should weigh in any rule choice depends on how wild the test set is, which nobody has measured. Both shipped rules (fusion_v1 and the now-shipped fusion_v2) use M3 (Spectra-AASIST, training data undisclosed) as a false-alarm suppressor, and nobody has checked whether its holdout skill is memorisation.

## Scope (from the frozen P1)

| # | Action | Runs | Output |
|---|---|---|---|
| A | λ̂: the test set's wild-domain weight from label-free channel statistics | yes, first | λ̂ with bootstrap interval, sent to the oversight chat before anything else |
| E | Three M3 leakage probes (perturbation sensitivity; M1b–M3 rank agreement on holdout vs test; M3 on unseen MLAAD spoof), plus the model card | yes | a one-line verdict on M3's false-alarm-suppression role |
| B | Is the 7.2 kHz wall a codec? LJ + LibriSpeech inner clips through a codec grid vs the test files | yes, beside E | codec-match table, or a recorded negative |
| C | Symmetric codec round-trip refit of M1b | **only if** B finds a match **and** the extraction can start by 13:45 (frozen P1, 12:16). Runs locally (MPS extraction, CPU refit), so no rental approval is needed; the column changes nothing until Nathan ratifies it through the main chat's `scripts/fuse_sweep_m5.py --refit-m1b` (acceptance contract: `docs/reports/2026-09-26_fusion-sweep-predeclared.md`, Addendum 2) | recorded as "not run" with the reason otherwise |
| D | One wild bona fide corpus | **dropped** (time box) | recorded as not attempted |

## Design decisions

- **D1 (A features).** Every clip goes through the same preparation: `load_audio` → `hearsay.compression.features` with no crop and the pipeline's band match (which runs `band_limit` → `trim_silence`), plus six added channel statistics computed on the same prepared signal: noise-floor level (10th-percentile frame energy relative to the 90th), noise-floor stationarity (std over 0.5 s blocks of each block's 10th-percentile energy), a reverberation proxy (median energy-decay slope in the 120 ms after speech offsets), spectral flatness, frame-loudness std, and pause fraction. Trimming removes corpus-specific lead silence (a known shortcut, `architecture.md` §11) from every set alike. No labels, filenames or durations enter.
- **D2 (A classifier).** Standardized logistic regression (C = 1, balanced) separating holdout real (`splits/nsa_folds.csv`, fold `holdout`, bona fide, 1,858 clips) from In-the-Wild real (2,000 clips, 54 speakers). Out-of-fold scores by 5-fold **grouped** CV (group = holdout `group` / ITW speaker); no speaker spans a train and a validation fold.
- **D3 (A estimators).** Two, both reported: (i) the memo's mean posterior P(wild) over the 1,671 test files; (ii) adjusted classify-and-count, λ̂ = clip((q − FPR) / (TPR − FPR), 0, 1), with q the test share above 0.5 and TPR/FPR from the out-of-fold scores; refused (NaN plus a message) when TPR − FPR < 0.2. Bootstrap 1,000× over test files (and over the out-of-fold rows for (ii)); 95% percentile interval. Also on the subset the shipped rule calls real (lowest 70% of the 12:03 draft's scores), and a per-feature position t = (median_test − median_clean) / (median_wild − median_clean). A novelty share (test files beyond the 99th percentile of the training Mahalanobis distance) says how much of the test set is neither. The memo's crossover is 0.32.
- **D4 (B grid).** 50 LJ + 50 LibriSpeech clips from inner folds (seed 0). Variants: raw; Kaiser `band_limit` alone; each of mp3 {32, 48, 64, 96} kbps and aac {32, 48, 64} kbps at {16, 44.1} kHz through `hearsay.compression.launder`, alone and followed by `band_limit`. Statistics on the unfiltered result (the diagnosis compares what the test file is, not what the pipeline does to it): band levels at 6.5–7.75 kHz relative to 1–3 kHz, the 7.5-vs-6.5 kHz drop, roll-off slope over 6.5–8 kHz, high-band (6–7 kHz) flatness, `deep_hole_frac`, `local_hole_frac`, `floor_p2_db`. The same statistics on all 1,671 test files. Distance = mean over statistics of |median_variant − median_test| / IQR_test.
- **D5 (B match rule, pre-declared).** A codec variant "matches" if its distance is at least 20% below the Kaiser-alone distance **and** it is closer to the test medians than Kaiser on both hole statistics. Otherwise B is negative and C does not run.
- **D6 (E perturbation probe).** 500 inner clips (250 real, 250 spoof, stratified by fold and generator, seed 0), each decoded, silence-trimmed and cropped once to a test-length segment (seeded; no band match yet), then: none, MP3 64 kbps at 16 kHz round-trip, white noise at 20 dB SNR, ±2% speed (resampling), one-sample circular shift. The perturbed raw segment is scored with the runner's own `Models` paths (`m1_logit`, `spectra_logit`, `m5_logit`; CPU, the shipped probe, zero-pad Spectra, M1 fp16 rounding), which apply their own band match and trim exactly as at test time. Readouts per model and perturbation: AUC, ΔAUC vs clean, Spearman(clean, perturbed), and the share of spoof clips crossing M3's suppression threshold (logit < −3) under each perturbation.
- **D7 (E agreement probe).** Spearman(M1b, M3) on holdout rows vs test rows from the existing exports, with a 1,000× bootstrap interval on the difference.
- **D8 (E unseen-spoof probe).** 600 MLAAD English spoof clips (stratified by model, seed 0, test-length crop), scored by M1b, M3 and M5 as in D6. Readouts: each model's miss rate at its own inner-OOF brief-cost threshold, and **the share of MLAAD spoof with M3 logit < −3**, which is exactly the shipped rule's exposure (suppression can only create misses). M5 trained on MLAAD, so its number is in-sample and reported as such.
- **D9 (E verdict rule, pre-declared).** M3's suppression role is **at risk** if any of: (a) M3's mean |ΔAUC| over perturbations exceeds twice M1b's and 0.02 absolute; (b) holdout-minus-test Spearman exceeds 0.2 with the interval excluding 0; (c) more than 5% of MLAAD spoof falls below −3. Otherwise **kept**. The verdict is advice to the main chat; changing the shipped rule goes through `scripts/fuse_sweep_m5.py` and Nathan.
- **D10 (discipline).** No training or selection on holdout, In-the-Wild or test labels beyond D2's domain classifier, which uses only domain membership of real clips (never spoof-vs-real labels on eval sets). Nothing written under `data/`. No edits to `embed.py`, `probe.py`, `spectra.py`, `pipeline.py`, fold files, extraction/training/fusion scripts.

## Acceptance criteria

1. A: λ̂ (both estimators) with intervals in the report and sent to the oversight chat; grouped-CV AUC of the domain classifier reported; the grouped CV provably never splits a group (test).
2. B: the full grid table and the pre-declared D5 verdict in the report.
3. E: the three probes' tables and the D9 verdict in the report; model-card check recorded.
4. C and D recorded with their disposition (not run / not attempted, and why).
5. New tests pass; `uv run pytest -q` and `uv run ruff check .` stay green; `tests/test_docs_consistency.py` green after doc edits.
6. Report `docs/reports/2026-09-26_channel-robustness.md`, a `docs/STATUS.md` row, a `CLAUDE.md` disclosure bullet.

## Failure modes → tests (`tests/test_channel_robustness.py`, hermetic)

| Stage | Failure mode | Test |
|---|---|---|
| A features | NaN/inf on silence, very short or clipped audio | all-zero, 0.3 s and full-scale-square inputs return finite dicts |
| A features | a statistic that does not move with its cause (vacuous feature) | added white noise raises floor level; an added exponential reverb tail raises the decay proxy (slower decay); a stationary vs gated noise floor orders stationarity |
| A features | nondeterminism | same input twice → identical dict |
| A features | level shortcut | scaling the input by 0.1 leaves every statistic within tolerance |
| A classifier | group leakage across CV folds | every group lands in exactly one validation fold; counterfactual: a clip-level split on the same data is detected as leaking |
| A estimators | biased mixture estimate | synthetic two-Gaussian domains with a known 0.3 mix: both estimators within ±0.08 |
| A estimators | degenerate classifier | TPR − FPR < 0.2 → NaN plus reason, never a number |
| A estimators | clipping | q below FPR → 0, above TPR → 1 |
| A per-feature t | sign/scale errors | known medians give the expected t, including t < 0 and t > 1 |
| B stats | codec wall not measured | a signal low-passed at 7 kHz shows a larger 7.5-vs-6.5 drop than one low-passed at 7.75 kHz |
| B distance | wrong normalisation | identical distributions → 0; shifted by one IQR → 1 |
| B match rule | guard removed | a variant 25% better on distance but worse on one hole statistic is **not** a match; one better on both is |
| E perturbations | a perturbation that does nothing, or changes the length or label | each perturbation changes the signal; noise hits 20 ± 0.5 dB SNR; shift is exactly one sample; speed changes length by ±2% ± 1 sample |
| E metrics | ΔAUC sign | a scorer that is perfect clean and random perturbed gives ΔAUC ≈ −0.5 |
| E verdict | guard removed | each of (a), (b), (c) alone flips the verdict to "at risk"; none → "kept" |
| E agreement | bootstrap interval wrong | identical rankings on both sets → difference 0 with a tight interval |

Scripts read real data and weights and are exercised by running them; their pure functions are what the tests pin.

## Documentation

- Create: this spec; `docs/reports/2026-09-26_channel-robustness-plan.md`; `docs/reports/2026-09-26_channel-robustness.md`.
- Update: `docs/STATUS.md` (one row), `CLAUDE.md` (one disclosure bullet).
- Conditional (confirm at execution): `docs/architecture.md` §11 shortcut ledger, only if A or B finds a new train-vs-test fingerprint. **Confirmed:** A found one (the double low-pass stopband); the entry was added after the Codex audit (below).

## Codex plan review (before commit a0018a8): findings and dispositions

Review: `docs/reports/2026-09-26_channel-robustness-plan-review.md`. Every finding is adopted; the design above is amended as follows (these amendments override D2, D3, D6–D9 where they differ). Several were only partly implemented at first; the post-commit Claude critique found the gaps and they were closed in the critique-fix commit (see "Post-commit review").

1. **Mean posterior is not a mixture estimate.** λ̂ is the adjusted classify-and-count estimate; the mean posterior is descriptive only. Tests cover overlapping domains and endpoint mixtures (0, 0.3, 0.7, 1 at two separations), and pin the mean posterior's bias.
2. **Synthetic files in the test set; no identifiability check.** Never-trained controls (VCTK clean read, DiffSSD spoof, inner real, inner real after one MP3 round-trip) are scored. A pre-declared rule (**D3b**) must pass before λ̂ is read against the crossover: AUC ≥ 0.8, VCTK and DiffSSD each read wild under 25%, LJ-excluded estimate within 0.15. Otherwise λ̂ is "unidentifiable". The likely-real subset is a sensitivity readout, not the estimate. Added after the first full run: refits without the dominant feature and without all floor/hole features, so a one-feature answer is visible.
3. **LJ grouped by chapter.** The domain classifier groups by speaker (`speaker` column; LJ = one speaker). Scaling is inside each fold (pipeline). The interval resamples reference speakers, not rows. LJ-excluded sensitivity reported.
4. **E baseline in-sample.** The perturbation probe uses 500 **outer-holdout** clips (250 real by source, 250 spoof by generator), out of sample for M1b v3, M5 and handcrafted v5; M3's exposure is the question being probed. The holdout is read, never selected on: no model, threshold or weight is chosen from it (the fused minDCF readouts are the rule's own, unchanged).
5. **D7 cannot establish memorisation.** The M1b–M3 Spearman gap is reported as descriptive and removed from the D9 verdict.
6. **Exposure mislabelled.** Exposure is now the actual conjunction, computed by `FusionConstants.fuse` on `models/fusion_v2/constants.json` (`e_applied`: M3 logit < −3 and base rank > 0.5). The M3-only share is reported as an upper bound. Fused minDCF with and without the step (base vs final), and minDCF and EER per model, under every perturbation.
7. **Codec delay/padding.** Codec round-trips in E are aligned to their input by cross-correlation within ±4,000 samples and cut or zero-padded to the input length (`align_to`, tested). B trims silence before its statistics.
8. **Degenerate cases.** `feature_position` returns NaN below a 1e-9 median gap; the Mahalanobis covariance carries a 1e-3 ridge (tested on a constant column); the adjusted estimate refuses TPR − FPR < 0.2; a bootstrap with more than 10% refused resamples has no interval; any NaN verdict input gives "inconclusive", never "kept" (tested).
9. **Submission/log disposition.** This rung is diagnostic only: it writes **no TSV and no `submissions/log.csv` row** (log rows record TSVs; the shipped TSV is the main chat's). Any new column goes through `scripts/fuse_sweep_m5.py` in the main chat.
10. **Cache identity.** A's cache is keyed by path plus codec spec. E's cache is keyed path|perturbation|seed, with the model identities (`Models.version()`) and constants path stored beside it; a mismatch refuses to resume.
11. **Bounds.** No new compute starts after 15:00; an unfinished E probe makes the verdict "inconclusive". Telephony and physical replay are untested and stated as such.

**D9 revised (E verdict).** "At risk" if (a) M3's mean |ΔAUC| over the four perturbations exceeds twice M1b's and 0.02, or (c) the shipped rule damps more than 5% of unseen MLAAD spoof (`e_applied`), or (d) under any perturbation the step raises the fused holdout minDCF by more than 0.01. "Inconclusive" on any missing number; "kept" otherwise.

## Deviations from the design, recorded

- **D1 / A:** `clip_floor_db` added to the >7 kHz drop list after the first full run (it reads the stopband). Added sensitivities beyond D3: fold-ensemble and in-sample rate variants, per-fold AUCs, and a duration-matched run (reference clips cropped to test-length draws, seeded per clip), because D1's "no durations enter" was not strictly true for whole clips. The existing feature cache was written by `channel_stats` before the identity file existed; its identity (`channel_stats_v1`) was backfilled, with the code unchanged in between.
- **D4 / B:** the grid also has MP3 at 24 kbps. B's distance includes `clip_floor_db`; a sensitivity without it is reported (Kaiser 0.660 vs best codec 0.647: no match either way).
- **D8 / E:** MLAAD is 572 clips (4 per model × 143 models; stratified by model), not 600.

## Results (13:20)

Report: `docs/reports/2026-09-26_channel-robustness.md`. Commits: a0018a8 (A), 32d6402 (B, λ̂ correction, E script), and the E commit.

| # | Outcome |
|---|---|
| A | λ̂ = 0.51 (95% CI 0.12–0.64), identifiable, but only just (VCTK control 24.7% vs a 25% limit); **indeterminate** against the 0.32 crossover; consistent variants 0.36 (duration-matched) to 0.57. The first run's 0.947 is withdrawn: `clip_floor_db` read the 7–8 kHz stopband and was added to the >7 kHz drop list (a deviation from D1's feature list, made under D1's own rule). |
| B | No codec match (Kaiser 1.081, best codec 1.045: 3% where D5 needs 20% and both hole statistics). |
| C | Not run: B negative. The laundering path was built and smoke-tested, then removed unused (`scripts/channel_launder.py` is not committed). |
| D | Not attempted (time box, frozen P1). |
| E | **Kept** under D9 revised (`verdict_from`; both probes complete): M3 mean \|ΔAUC\| 0.0008 vs M1b 0.0041 (but M3's clean-vs-perturbed rank stability is the lowest of the deep models); MLAAD damped 1.2%; the M3 step never raised fused minDCF (it lowered it by 0.008–0.020 in 4 of 5 conditions). Agreement gap 0.175 (descriptive). New: 20 dB noise drops handcrafted v5 to AUC 0.64 and the shipped rule to 0.269 holdout minDCF. |

Acceptance criteria:
1. λ̂, both estimators and intervals, sent to oversight first (12:33), with the correction at 12:57. Grouped CV tested.
2. B's grid table and D5 verdict are in the report.
3. E's three probes, the D9 verdict and the model-card check are in the report.
4. C and D dispositions are recorded.
5. Tests: `tests/test_channel_robustness.py`, 56 passing after the critique fixes. Full suite 552 passing at 73f6a5b; ruff is clean on this change's files (3 findings in `src/hearsay/analyzer.py`, another lane's commit 19925a3).
6. Report, STATUS row and CLAUDE.md bullet are done.

**No TSV and no `submissions/log.csv` row:** this rung is diagnostic only (plan-review disposition 9).

## Post-commit review

**Claude critique, round 1 (13:25, on 73f6a5b): Overall Needs work.** Plan adherence, Test coverage, Review compliance and Documentation were Needs work; Scope, Freeze and Regression were Acceptable. Every number in the report and STATUS matched the JSON outputs. Findings and fixes:

1. **The verdict could say "kept" with a missing number** (`max()` skips NaN; no completeness check). Fixed with `verdict_from`, which gathers every input and the cohort counts and returns "inconclusive" on any gap. Tested with a NaN step effect, a missing perturbation, and short cohorts.
2. **The one-class LJ fold** made the pooled AUC and FPR pessimistic, the "without the top feature" row was misread as a collapse, and q and the rates came from different models. The fix:
   - per-fold AUCs reported
   - fold-ensemble and in-sample rate variants added (0.48 and 0.57)
   - the report's reading and limits rewritten
3. **"Most perturbation-stable" contradicted the rank-stability numbers.** The report now gives both readings, qualifies M3's MLAAD exposure as unknown, and states that the verdict shows the step is safe, not that M3 generalises.
4. **`lambda_hat_reading` ignored the interval.** It now comes from `crossover_reading` ("indeterminate" when the interval contains the crossover). Tested.
5. **The STATUS per-feature claim was false.** The STATUS text and the report now list the features outside [0, 1].
6. **Partial dispositions:**
   - #6: fixed-inner-threshold P_FA and P_miss added per detector and perturbation.
   - #7: AAC and 44.1 kHz alignment tests and a failed-round-trip test added.
   - #10: identity required whenever a cache exists (`check_cache_identity`, shared by A and E); readouts use only the requested keys (`select_requested`); A's cache carries a feature version.
   - #11: short cohorts give "inconclusive".
7. **Test gaps closed:** `acc_ci` refusal, cache refusal, speaker grouping (`speaker_groups`), MP3 changing the signal, B's silence trim, key parsing and crops, strict JSON.
8. **The report now carries the full 34-row grid** (AC2).
9. **The D4 and D8 deviations are recorded above**, including B without `clip_floor_db`.
10. **Provenance corrected:** the −157 dB source, the Codex-review timing, and the 0.947 run's artifact.
11. **The duration confound is now measured:** a duration-matched sensitivity gives 0.36.
12. **Wording and strict JSON:** fixed. The fusion lane's λ̂ line was relayed to oversight.

**Claude critique, round 2 (13:50, on cec75ec): Overall Acceptable.** All 12 round-1 findings were verified closed in code, including a re-extraction check of 18 cached feature keys (maximum difference 1.4e-14), and every report and STATUS number matched the regenerated JSON. Remaining items, all fixed in the following commit:
1. The report said the suppression step "fires only on real clips", but it damps 1.2% of MLAAD spoof. Reworded.
2. Two table labels said "unseen MLAAD". Fixed.
3. The report said the LJ fold was the fifth; it is fold 0. Fixed.
4. `--readout-only` did not check the cache's identity. It now refuses a cache with no meta file or with other perturbations, and prints the scoring models.
5. `verdict_from` checked how many step effects there were, not their names, and accepted booleans. Both are now checked, with a test.
6. The report called M3 "near-perfect" under noise. It now says best of the four, with P_miss 0.32.
7. Two stale claims sit in another lane's `docs/reports/2026-09-26_fusion-sweep-predeclared.md` (lines 53 and 141), and README.md:225 repeats the "fires only on real clips" claim. Those are other lanes' files, relayed through oversight.

**Codex audit, round 1 (13:55, on 68cc08e): Overall Fail** (`docs/specs/2026-09-26_channel-robustness-audit.md`). Plan adherence, Test coverage, Review compliance and Documentation were Fail; Scope discipline Excellent; Freeze and Regression Acceptable. Codex confirmed the saved cohorts have no missing scores, so the recorded verdict stands. Findings and fixes:
1. **Completeness checked M1b only.** `perturb_readout` now counts a clip only if every detector, both fused scores and the step flag exist under all five perturbations. Tested through readout → verdict: one missing M3 score shrinks `n_paired`, and the verdict goes "inconclusive".
2. **Booleans were accepted in the scalar verdict inputs.** `m3_verdict` now rejects them, tested per input.
3. **Missing deliverables:**
   - The report now carries the mean-posterior interval (0.55–0.58, AC1).
   - The conditional shortcut-ledger entry is written (`architecture.md` §11, "Double low-pass stopband"). It is marked **open**: whether the detectors respond to the stopband depth is unmeasured.
4. **Reversed AUC reasoning.** The report now says a stable AUC means the classes stay apart, not that scores stay put. The claim about trigger (a) is removed.
5. **The final full suite predated the fixes.** It is re-run at the audit-fix commit, recorded below.

Full suite at the audit-fix commit: **572 passed** (`uv run pytest -q`, 199 s). Ruff is clean on this change's files; the 3 findings in `src/hearsay/analyzer.py` belong to another lane (19925a3).

**Codex audit, round 2 (on fb13a29): Overall Fail**, on two completeness gaps. Documentation was now Acceptable, and the saved readouts reproduced exactly.
1. **The MLAAD readout counted rows without checking scores.** A NaN M3 score or step flag left n = 572. It now uses `complete_rows`: finite values required in every detector, fused score and flag column.
2. **The perturbation cohort used `notna()`, which accepts ±inf.** It now uses `np.isfinite`.

Both are tested through readout → verdict ("inconclusive") for NaN, +inf and −inf, and for the M3, M1b and step-flag columns. The regenerated outputs are unchanged (500 and 572 clips, verdict "kept"). Full suite at this commit: 581 passed.
