# V2-33 stage 0: example-weighting bake-off (hardest vs easiest vs random), results

Run spec: `docs/specs/2026-10-13_v2-33-hardneg-stage0.md` (design of record: its "Phase 3 round 2, FINAL DESIGN" section plus the Codex round-1 dispositions). Plan: `docs/reports/2026-10-13_v2-33-hardneg-stage0-plan.md`. Code: `scripts/v2_hardneg_bakeoff.py`, tests `tests/test_hardneg.py`. Outputs: `outputs/v2/hardneg/` (gitignored: `report.json`, `csweep.json`, `cells/*.json|npz`, `rows.json`, run logs), so every number lives in this report. Log row: `experiments/log.csv`, rung `V2-33-stage0`. Branch `v2`.

**Framing, fixed before any number was read:** this is a **diagnostic on the legacy split, not a validation.** The fold file `splits/nsa_folds_plus_asv19.csv` is generator-disjoint and group-disjoint (0 of 158 groups cross a fold) but not speaker-disjoint: 41 of 113 speaker tokens span more than one fold, covering 19,699 of 23,790 rows (11 of 73 tokens and 13,115 of 16,142 rows on the NSA-only basis, led by `lj`, whose real clips and the fakes synthesised from her voice share one token). No validation conclusion is drawn here. **V2-33 stays open** pending a V2-01 benchmark row on generator- and speaker-disjoint folds, and inference throughout is **exploratory**: 132 gated cells plus a 64-cell descriptive sweep, N recorded, nothing corrected for multiplicity, no cell adopted.

## What this is

v1 trained its scoring probe with every training clip weighted equally. Other teams reportedly gained by focusing on the hardest clips, including an RL-driven variant. The Oct 10 pilot said hard-example training helps, but it was read under the brief cost and its selection step mined under that cost too. The pre-run exploration (spec W1 to W7) then found that the pilot's two "direction" arms were the same model at two regularization strengths, that the probe is perfectly separable in-sample so an in-sample "hardest" does not exist, and that the dominant lever in the whole family was the regularization constant `C`, not which examples were up-weighted. So the question was reframed on Nathan's instruction ("yeah reframe it, hard vs easy vs random"):

> At **matched regularization**, does up-weighting a 5% subset of training examples change the score, and does the selection rule matter: **hardest vs easiest vs random**?

Design, all pre-declared:

- **Arms.** `none` (base); `hardest_both`, `easiest_both`, `random_both` (the primary three-way question, four layers); `hardest_reals`, `hardest_fakes`, `easiest_reals`, `easiest_fakes` (direction probes, L7 only); `focal_g2` (continuous hard-example emphasis, L7 only, not on the matched-weight footing); `pilot_quantile` (reproduces the pilot's `mine_nested` row; L7 only; **not eligible for recommendation**).
- **Selection** on nested inner-inner out-of-fold margins, normalised per inner model by the IQR of that model's own training margins (Codex #6); a fold never mines the rows it scores. Top 5% per class, weight x5, `sum(sample_weight) = n` for every arm so no arm is confounded with `C`.
- **Grid.** Layers 6, 7, 8, 20 x pools `pool_full` (23,790 rows fitted, NSA + ASVspoof 2019 train/dev) and `pool_nsa` (16,142 NSA rows fitted) x `C` in {1, 8, 64}. 132 cells. `pool_nsa` x full basis is N/A by declaration.
- **Readings.** package-0.5 = P_FA + 4·P_miss (rule 5, primary); package-0.3 = P_FA + 1.714·P_miss; brief = 9.33·P_FA + P_miss (v1's); EER beside them. Every reading below is pooled over the five inner folds on the **NSA-only 16,142-row basis** unless a row says otherwise; thresholds are each cell's own package-0.5 optimum, found by an exact re-sweep.
- **Gate R1 to R10** per (arm, layer, pool, C) against the matched `none` cell; bands 0.010 (brief, folds), 10% relative (false alarms), P(better) >= 0.90 from a paired 1,000-replicate bootstrap over the fold file's `group` units (seed 0). The answer to the primary question is the four pre-declared paired contrasts aggregated over strata, never a cell.

## Verdict

**No recipe survives the pre-declared gate.** Of the 107 candidate cells, **8 pass** R1 to R4b, R6, R7 and R9, **6 are INDETERMINATE** (R1's P(better) within 2 SE of the 0.90 band with every other rule passing) and 93 fail. Each pass is on a single layer and none on L20, so R5 fails for every (arm, pool, C) and there is no survivor. One of the eight passes is the **random control** (`random_both`, L6, `pool_full`, C=8), which sets the scale: a cell-level pass at this gate is something a random 5% up-weighting can produce.

The three-way answer, read the way the design of record specified: **per C**, the pre-declared unit, each C's eight (layer, pool) strata averaged inside 1,000 shared `group`-resamples (seed 0), NSA-only basis, pooled package-0.5. Gain = base − arm, so positive means the arm is better; the interval is the 5th to 95th percentile of the replicate means.

| Contrast | C=1 (v1's value): P(better), median gain, 90% interval | C=8 | C=64 |
|---|---|---|---|
| hardest − none | **0.025**, −0.0086, [−0.0140, −0.0008] | 0.785, +0.0028, [−0.0018, +0.0089] | 0.415, −0.0012, [−0.0112, +0.0085] |
| easiest − none | **0.971**, +0.0211, [+0.0004, +0.0376] | **0.975**, +0.0290, [+0.0005, +0.0447] | **0.987**, +0.0198, [+0.0012, +0.0304] |
| hardest − random | 0.090, −0.0032, [−0.0082, +0.0008] | 0.413, −0.0012, [−0.0107, +0.0093] | 0.404, −0.0032, [−0.0185, +0.0150] |
| easiest − random | **0.998**, +0.0268, [+0.0011, +0.0441] | **0.986**, +0.0248, [+0.0006, +0.0380] | **0.984**, +0.0191, [+0.0014, +0.0337] |

- **Up-weighting the hardest 5% does not help, and at v1's C it hurts.** At C=1 the aggregate says the hardest arm is *worse* than base with P 0.975 (median −0.0086, the whole interval below zero) and worse than random with P 0.91. At C=8 and C=64 it is a wash (P(better) 0.785 and 0.415, intervals straddling zero). The sign depends on the layer: better at L6 in 6 of 6 strata, worse at L20 in 6 of 6.
- **Up-weighting the easiest 5% is better than base and than random at every C**, with P(better) 0.971 to 0.998 and median gains of 0.019 to 0.029 package-0.5. The lower ends of the 90% intervals sit at +0.0004 to +0.0014, so the effect is consistently signed but small: about one C-step's worth, and the interval nearly touches zero at C=1 and C=8. It holds on both pools and all four layers (22 of 24 strata individually). It is not the over-regularization artifact the design feared (spec W7), because every arm holds `sum(w) = n`. It is also **not a recommendation**: no (arm, pool, C) clears R5, and this is inner OOF on a split with speaker overlap.
- **The regularization constant is the larger lever**: 0.023 to 0.203 of package-0.5 between C=1 and C=256 depending on layer and pool. Descriptive only (below).
- **Nothing is adopted.** The direction finding (easiest, not hardest), the one consistent single-layer lead (`hardest_reals` on `pool_nsa`, L7) and the C curve go to the V2-01 benchmark lane, to be read on the holdout cohort and In-the-Wild under the read-once ledger, on speaker-disjoint folds.

The descriptive all-strata means, which pool C and were the first commit's headline, are +0.0040 (hardest − none), −0.0261 (easiest − none), +0.0043 (hardest − random) and −0.0258 (easiest − random); they are kept in `report.json` as `mean_delta_package05_all_strata` and carry no inference.

## Run record

| | Run 1 | Run 2 | Run 4 (final) |
|---|---|---|---|
| When | Mon Oct 13, 17:11 to 19:38 EDT | Sat Oct 17, 20:48 to 21:12 EDT | Sat Oct 17, 21:43 to 22:12 EDT |
| Script | the committed script minus every Oct 17 fix | after the three readout fixes (defects 1 to 3 below) | the committed script, after the audit-round-1 fixes |
| Cells | 132 of 132 declared; 131 OK, 1 INVALID | 132 of 132; 131 OK, 1 INVALID (same cell, same reason) | 132 of 132; 131 OK, 1 INVALID (same cell, same reason); 131 computed, 0 reused |
| Fit wall time | 17:11 to 17:29 (18 min) | 878 s, then the gate 21:02 to 21:12 | 868 s (14.5 min), then the gate 21:58 to 22:12 (14.5 min: per-cell bootstraps, the per-C aggregates for four contrasts, H3) |
| Self-check (base reproduces the shipped L7 probe) | passed: Spearman exactly 1.0, package-0.5 gap below 1e-12 | passed: Spearman 1.0, gap 0.0e+00 | passed: Spearman 1.0, gap 0.0e+00 (`fit_run4.log`) |
| Time box | 3 h persisted deadline, not reached | 3 h, not reached | 3 h, not reached (the gate now checks it too) |
| Preserved at | `outputs/v2/hardneg_run1_20261013/` (31 MB copy taken before run 2) | its `report.json`, beside run 1 as `report_run2_20261017.json` | `outputs/v2/hardneg/` |

A third attempt (run 3, Oct 17 21:58) was stopped after 64 cells, by PID, when the Claude critique
arrived with fixes that touch the script: at that point the fingerprint still hashed the whole file,
so any edit mid-run would have made the earlier cells "stale" under the new validation rule. The
fingerprint now hashes the fitting-path source instead (see the disclosures), and run 4 is the
complete run under the final script. The descriptive C sweep (`csweep.json`, 64 rows, complete) ran
once, in run 1; nothing in any Oct 17 fix touches it, and its rows are carried unchanged.

**Why there are several runs.** While writing this report from run 1's outputs, three defects were
found in the readout and gate code (table below); the Codex audit and the Claude critique of the
first commit then found six more in the gate and the inference layer (recorded in the spec's
scorecards). None touches a fitted model, a score or a pre-declared band, so the deterministic grid
was re-run with the fixed script and checked cell by cell against the preserved run 1 each time:
run 4 against run 1, all 131 OK cells byte-identical in `oof` and `mask` and identical in every primary reading (the renamed population fields aside), the INVALID cell INVALID for the same reason, so 132 of 132 agree; run 2 had given the same result. Fixing the code to compute what the pre-declaration said, and re-running
without changing any band or definition, is the disposition the manifest rule allows ("if the
self-check fails, no candidate is read until it is fixed"); redefining H5 after reading would not be,
so H5 is reported as frozen. One change is to the *inference layer* and is stated as such: the
primary answer is now aggregated per C inside shared bootstrap resamples, which is what the design
of record and the Codex plan review specified, where the first commit had reported stratum counts.

### The three defects, and what changed

| # | What was wrong in run 1 | Consequence in run 1 | Fix (Oct 17) | Test |
|---|---|---|---|---|
| **1. R7 read the population, not the mined set.** `read_cell` computed "group concentration" over every real (or fake) row of the basis; the mined rows were never characterised beyond per-fold counts. | Every cell reported 0.0316 / `lj:LJ028` (real) and 0.125 / `gen:pro_diff` (fake): a constant, so **R7 could never fail**, which is the spec's own E1 failure mode. H4's characterisation readout (source, generator, group, speaker) did not exist, and H3's arm-overlap readout promised by the plan was not implemented. | Picks are returned per outer fold and stored (as global row indices) in each cell's `.npz`; `characterise()` reports pooled pick counts, unique rows, top group and its share, the per-fold maximum group share, top speaker, source shares and (fakes) generator shares; `gate_cell` reads R7 from the mined set and treats a missing readout as a failure; the gate stage adds H3 Jaccard between arms. The population figures stay, renamed `pop_*`. | `test_e_r7_reads_the_mined_set_not_the_population`, `test_e_r7_missing_readout_is_a_failure_for_a_mined_arm`, `test_e_r7_fakes_only_arm_needs_a_readout_but_has_no_real_side_to_cap`, `test_h4_characterise_measures_the_mined_rows_not_the_population`, `test_c7_mined_picks_are_training_rows_of_their_fold`, `test_h3_jaccard_detects_collapsed_arms` |
| **2. R4a/R4b looked up integer fold keys on JSON-loaded cells.** Cells are written through `clean()` and read back with `json`, which turns the per-fold keys `0`/`4` into `"0"`/`"4"`; `.get(0)` returned `None`. | **R4a and R4b failed in all 107 non-base cells**, including cells whose fold 0 and fold 4 were better than base; 13 cells failed on R4 alone and would otherwise have passed. The gate's verdict column in run 1 is therefore wrong wherever R4 decided it. The unit tests passed because their fixture cells use integer keys. | Keys are normalised to `int` before the lookup. | `test_e_r4_survives_the_json_round_trip` (round-trips a passing cell through `clean()` + `json` and requires the same verdict) |
| **3. H5 "confidently wrong" saturates as frozen.** The frozen definition (Codex #6) counts false alarms whose normalised real-side margin is in the worst decile of all reals. Whenever the false alarms at the package-0.5 threshold already cover that decile, the count *is* the decile. | 815 in every NSA-basis cell (the decile of 8,142 reals) and 1,327 in every full-basis cell; the minimum false-alarm count in any cell is 996. The readout carried no information about what the arm did. | **Not redefined**: that would be a post-read change to a frozen definition. Reported here as saturated; `report.json` records the saturation check; a scale-anchored definition belongs to stage 0.5. | `test_h5_as_frozen_saturates_when_false_alarms_cover_the_real_decile` (pins the documented state) |

One promised readout is unavailable rather than wrong: the characterisation's **duration** field. The embedding manifests carry no duration column (`path, label, generator, speaker, utt, source, n_windows, flag`), and this lane may not touch audio, so duration is reported as not available.

## The primary answer: four paired contrasts

Each stratum compares the arm with the `none` cell at the same layer, pool and C on the NSA-only basis. P(better) is the share of 1,000 paired `group`-resamples (seed 0; 999 usable, 1 single-class draw skipped) in which the arm's pooled package-0.5 is lower than base's; its Monte-Carlo SE is 0.005 to 0.016. The one INVALID cell is a `focal_g2` cell and touches no contrast, so all 24 strata are present in each.

**The pre-declared inference, per C.** Within one C, the eight (layer, pool) strata are averaged inside each resample; the 24 strata of a contrast are never pooled across C. Gain = base − arm (positive = arm better):

| Contrast | C | strata | P(better) ± SE | median Δ | p5 | p95 |
|---|---|---|---|---|---|---|
| hardest − none | 1 | 8 | 0.025 ± 0.005 | -0.0086 | -0.0140 | -0.0008 |
| hardest − none | 8 | 8 | 0.785 ± 0.013 | +0.0028 | -0.0018 | +0.0089 |
| hardest − none | 64 | 8 | 0.415 ± 0.016 | -0.0012 | -0.0112 | +0.0085 |
| easiest − none | 1 | 8 | 0.971 ± 0.005 | +0.0211 | +0.0004 | +0.0376 |
| easiest − none | 8 | 8 | 0.975 ± 0.005 | +0.0290 | +0.0005 | +0.0447 |
| easiest − none | 64 | 8 | 0.987 ± 0.004 | +0.0198 | +0.0012 | +0.0304 |
| hardest − random | 1 | 8 | 0.090 ± 0.009 | -0.0032 | -0.0082 | +0.0008 |
| hardest − random | 8 | 8 | 0.413 ± 0.016 | -0.0012 | -0.0107 | +0.0093 |
| hardest − random | 64 | 8 | 0.404 ± 0.016 | -0.0032 | -0.0185 | +0.0150 |
| easiest − random | 1 | 8 | 0.998 ± 0.001 | +0.0268 | +0.0011 | +0.0441 |
| easiest − random | 8 | 8 | 0.986 ± 0.004 | +0.0248 | +0.0006 | +0.0380 |
| easiest − random | 64 | 8 | 0.984 ± 0.004 | +0.0191 | +0.0014 | +0.0337 |

The first commit of this report had no aggregate and read stratum counts instead ("22 of 24 strata better"); both the Codex audit and the Claude critique flagged that as stratum-counting the pre-declaration had ruled out. The counts are kept below as a descriptive view only.

**Descriptive: by C, layer and pool** (mean Δ = arm − base, negative better, and the number of strata where the arm is better):

| Contrast | C=1 | C=8 | C=64 | L6 | L7 | L8 | L20 | `pool_full` | `pool_nsa` |
|---|---|---|---|---|---|---|---|---|---|
| hardest − none | +0.0082 (3/8) | −0.0008 (5/8) | +0.0046 (5/8) | −0.0210 (6/6) | +0.0099 (3/6) | −0.0006 (4/6) | +0.0277 (0/6) | +0.0123 (5/12) | −0.0043 (8/12) |
| easiest − none | −0.0250 (7/8) | −0.0316 (8/8) | −0.0217 (7/8) | −0.0226 (6/6) | −0.0274 (6/6) | −0.0262 (6/6) | −0.0283 (4/6) | −0.0221 (10/12) | −0.0301 (12/12) |
| hardest − random | +0.0030 (3/8) | +0.0041 (4/8) | +0.0058 (4/8) | −0.0129 (5/6) | +0.0059 (3/6) | +0.0025 (2/6) | +0.0216 (1/6) | +0.0175 (2/12) | −0.0089 (9/12) |
| easiest − random | −0.0302 (8/8) | −0.0267 (7/8) | −0.0205 (7/8) | −0.0146 (6/6) | −0.0313 (6/6) | −0.0230 (6/6) | −0.0344 (4/6) | −0.0169 (10/12) | −0.0347 (12/12) |

The hardest arm's one systematic pattern is a layer effect: it helps at L6 and hurts at L20 in every stratum, with L7 and L8 split. The easiest arm's two misses are both L20 `pool_full` (C=1 +0.0004, C=64 +0.039), the layer with the worst base (0.59 to 0.66) and the most false alarms.

**Every stratum** (Δ = arm − base package-0.5, negative is better; P = that stratum's own P(better)):

| L | pool | C | hardest − none | P | easiest − none | P | hardest − random | P | easiest − random | P |
|---|---|---|---|---|---|---|---|---|---|---|
| 6 | pool_full | 1 | -0.0267 | 0.65 | -0.0305 | 0.82 | -0.0321 | 0.65 | -0.0360 | 0.76 |
| 6 | pool_full | 8 | -0.0403 | 0.92 | -0.0444 | 0.87 | -0.0178 | 0.89 | -0.0219 | 0.82 |
| 6 | pool_full | 64 | -0.0276 | 0.96 | -0.0324 | 0.97 | +0.0006 | 0.36 | -0.0041 | 0.59 |
| 6 | pool_nsa | 1 | -0.0194 | 0.93 | -0.0110 | 0.77 | -0.0115 | 0.98 | -0.0031 | 0.78 |
| 6 | pool_nsa | 8 | -0.0049 | 0.77 | -0.0080 | 0.63 | -0.0062 | 0.66 | -0.0094 | 0.79 |
| 6 | pool_nsa | 64 | -0.0068 | 0.82 | -0.0092 | 0.66 | -0.0106 | 0.72 | -0.0130 | 0.98 |
| 7 | pool_full | 1 | +0.0425 | 0.08 | -0.0455 | 0.93 | +0.0093 | 0.28 | -0.0788 | 0.93 |
| 7 | pool_full | 8 | +0.0165 | 0.20 | -0.0330 | 0.74 | +0.0258 | 0.30 | -0.0236 | 0.79 |
| 7 | pool_full | 64 | +0.0408 | 0.08 | -0.0177 | 0.67 | +0.0476 | 0.21 | -0.0109 | 0.66 |
| 7 | pool_nsa | 1 | -0.0075 | 0.89 | -0.0112 | 0.84 | -0.0091 | 0.90 | -0.0128 | 0.80 |
| 7 | pool_nsa | 8 | -0.0170 | 0.88 | -0.0276 | 0.90 | -0.0170 | 0.90 | -0.0276 | 0.90 |
| 7 | pool_nsa | 64 | -0.0160 | 0.88 | -0.0293 | 0.90 | -0.0209 | 0.81 | -0.0342 | 0.85 |
| 8 | pool_full | 1 | +0.0203 | 0.11 | -0.0199 | 0.76 | +0.0396 | 0.04 | -0.0006 | 0.55 |
| 8 | pool_full | 8 | -0.0100 | 0.64 | -0.0410 | 0.75 | +0.0070 | 0.40 | -0.0240 | 0.74 |
| 8 | pool_full | 64 | -0.0064 | 0.64 | -0.0393 | 0.74 | +0.0179 | 0.13 | -0.0150 | 0.73 |
| 8 | pool_nsa | 1 | +0.0122 | 0.19 | -0.0065 | 0.64 | +0.0009 | 0.45 | -0.0178 | 0.75 |
| 8 | pool_nsa | 8 | -0.0131 | 0.68 | -0.0241 | 0.76 | -0.0289 | 0.88 | -0.0398 | 0.97 |
| 8 | pool_nsa | 64 | -0.0065 | 0.57 | -0.0261 | 0.79 | -0.0213 | 0.77 | -0.0409 | 0.98 |
| 20 | pool_full | 1 | +0.0237 | 0.08 | +0.0004 | 0.54 | +0.0166 | 0.28 | -0.0068 | 0.66 |
| 20 | pool_full | 8 | +0.0586 | 0.27 | -0.0013 | 0.70 | +0.0611 | 0.09 | +0.0012 | 0.70 |
| 20 | pool_full | 64 | +0.0561 | 0.35 | +0.0393 | 0.24 | +0.0340 | 0.31 | +0.0172 | 0.12 |
| 20 | pool_nsa | 1 | +0.0206 | 0.36 | -0.0754 | 0.96 | +0.0103 | 0.39 | -0.0858 | 0.93 |
| 20 | pool_nsa | 8 | +0.0041 | 0.34 | -0.0733 | 0.92 | +0.0089 | 0.31 | -0.0685 | 0.90 |
| 20 | pool_nsa | 64 | +0.0032 | 0.45 | -0.0592 | 0.86 | -0.0012 | 0.51 | -0.0635 | 0.75 |

## The gate

Verdicts over the 132 cells, final gate: **24 BASE, 93 FAIL, 8 PASS, 6 INDETERMINATE, 1 INVALID.** (Run 1 had read 107 FAIL and 0 PASS because R4a/R4b failed in every cell, defect 2; the first commit read 9 PASS because an R1-indeterminate cell was counted as a pass, Codex audit round 1 #2.)

**The eight passing cells.** Each passes on exactly one layer: four at L6, four at L7, none at L8 or L20; so no (arm, pool, C) reaches R5's three layers including L20, and no arm is a survivor. The random control is among them.

| PASS cell | pkg-0.5 | base | Δ | P(better) ± SE | f0 (base) | f4 (base) | FA (base) | brief Δ | mined real top group (share) |
|---|---|---|---|---|---|---|---|---|---|
| hardest_both L6 pool_full C=8 | 0.2402 | 0.2805 | -0.0403 | 0.924 ± 0.008 | 0.4998 (0.5472) | 0.2527 (0.3057) | 1300 (1445) | -0.0132 | asvspoof2019:LA_0085 (0.070136) |
| hardest_both L6 pool_full C=64 | 0.2421 | 0.2697 | -0.0276 | 0.961 ± 0.006 | 0.5243 (0.5487) | 0.2603 (0.3079) | 1344 (1622) | -0.0049 | asvspoof2019:LA_0085 (0.069759) |
| hardest_both L6 pool_nsa C=1 | 0.2379 | 0.2573 | -0.0194 | 0.927 ± 0.008 | 0.4437 (0.4717) | 0.2524 (0.2677) | 1412 (1509) | -0.0081 | spk:libri_8555 (0.065806) |
| easiest_both L6 pool_full C=64 | 0.2374 | 0.2697 | -0.0324 | 0.968 ± 0.006 | 0.4746 (0.5487) | 0.2653 (0.3079) | 1273 (1622) | -0.0152 | asvspoof2019:LA_0078 (0.084465) |
| easiest_both L7 pool_full C=1 | 0.2851 | 0.3307 | -0.0455 | 0.931 ± 0.008 | 0.4232 (0.4735) | 0.3803 (0.3950) | 1426 (1642) | -0.0016 | asvspoof2019:LA_0108 (0.082202) |
| random_both L6 pool_full C=8 | 0.2579 | 0.2805 | -0.0225 | 0.967 ± 0.006 | 0.5321 (0.5472) | 0.2918 (0.3057) | 1400 (1445) | -0.0012 | lj:LJ028 (0.015837) |
| hardest_reals L7 pool_nsa C=1 | 0.2215 | 0.2405 | -0.0190 | 0.961 ± 0.006 | 0.3837 (0.4116) | 0.3443 (0.3394) | 1095 (1258) | -0.0057 | spk:libri_6295 (0.068266) |
| hardest_reals L7 pool_nsa C=8 | 0.1988 | 0.2219 | -0.0231 | 0.965 ± 0.006 | 0.3808 (0.4002) | 0.3368 (0.3825) | 1081 (1155) | -0.0120 | spk:libri_6295 (0.064576) |

**The six INDETERMINATE cells**, where every rule but R1 passes and P(better) lies within 2 SE of 0.90: `hardest_both` L7 `pool_nsa` at all three C (0.893 / 0.884 / 0.883), `easiest_both` L7 `pool_nsa` C=8 (0.904, the cell the first commit counted as a pass), `hardest_reals` L7 `pool_nsa` C=64 (0.900) and `pilot_quantile` L7 `pool_nsa` C=8 (0.887). Two further cells carry the indeterminate flag but fail R4a as well and are FAILs (`easiest_both` L7 `pool_nsa` C=64 at 0.901; `easiest_both` L20 `pool_nsa` C=8 at 0.917). The 0.90 band alone decides nothing in that region, as the plan said it would not.

**R5**, the cross-layer rule, for every primary-family (arm, pool, C) with at least one passing layer: `hardest_both|pool_full|C8` [6], `hardest_both|pool_full|C64` [6], `hardest_both|pool_nsa|C1` [6], `easiest_both|pool_full|C1` [7], `easiest_both|pool_full|C64` [6], `random_both|pool_full|C8` [6]; all `n = 1`, none includes L20, all **false**. The L7-only arms are descriptive with R5 N/A by declaration (Codex #7).

**Which rules did the work.** Over the 93 FAIL cells: R1 (P(better) ≥ 0.90 with a better pooled reading) fails in 88; R4a (fold 0 within 0.010) in 55; R3b (sign agreement across readings) in 41; R2 in 40; R4b (fold 4) in 39; R6 (false alarms within 10% relative) in 17; R3 (brief within 0.010) in 2. R7 never fails: now that it reads the mined set, the largest real-side top-group share in any mined set is 0.139 against the 0.25 cap (the fake side is reported, not gated; see the characterisation). R9 refuses only the INVALID focal cell, now at validation time rather than as a FAIL.

**R3b, the convention rule.** 41 cells show a sign flip between the package readings and the brief reading, i.e. they are better under rule 5's cost and worse under v1's, or the reverse. None could be a recommendation while the cost convention is open (D5).

## Direction probes, focal and the pilot-comparable arm (L7 only, descriptive)

Six arms run at L7 only, by design: four one-sided direction probes, focal loss, and the pilot-comparable quantile rule. R5 is N/A for them; they are read against the matched `none` cell like every other, and nothing here is a recommendation.

| arm | L | pool | C | pkg-0.5 | pkg-0.3 | brief | EER | f0 | f4 | FA@thr | Δpkg-0.5 vs none | P(better) ± SE | verdict | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hardest_reals | 7 | pool_full | 1 | 0.3814 | 0.2450 | 0.2937 | 9.10% | 0.5108 | 0.4154 | 1709 | +0.0507 | 0.100 ± 0.009 | FAIL | R1,R2,R3b,R4a,R4b |
| hardest_reals | 7 | pool_full | 8 | 0.2532 | 0.1851 | 0.2942 | 7.14% | 0.5170 | 0.3933 | 1288 | -0.0036 | 0.604 ± 0.015 | FAIL | R1,R4a,R4b |
| hardest_reals | 7 | pool_full | 64 | 0.2722 | 0.1969 | 0.2971 | 7.56% | 0.5323 | 0.4068 | 1500 | +0.0375 | 0.048 ± 0.007 | FAIL | R1,R2,R3b,R4a,R4b,R6 |
| hardest_reals | 7 | pool_nsa | 1 | 0.2215 | 0.1620 | 0.2509 | 6.28% | 0.3837 | 0.3443 | 1095 | -0.0190 | 0.961 ± 0.006 | PASS |  |
| hardest_reals | 7 | pool_nsa | 8 | 0.1988 | 0.1482 | 0.2532 | 5.85% | 0.3808 | 0.3368 | 1081 | -0.0231 | 0.965 ± 0.006 | PASS |  |
| hardest_reals | 7 | pool_nsa | 64 | 0.1914 | 0.1427 | 0.2544 | 5.71% | 0.3799 | 0.3319 | 1001 | -0.0265 | 0.900 ± 0.009 | INDETERMINATE | R1 |
| hardest_fakes | 7 | pool_full | 1 | 0.3012 | 0.2175 | 0.3126 | 8.21% | 0.4845 | 0.4166 | 1638 | -0.0295 | 0.649 ± 0.015 | FAIL | R1,R3,R3b,R4a,R4b |
| hardest_fakes | 7 | pool_full | 8 | 0.2229 | 0.1663 | 0.3047 | 6.76% | 0.4773 | 0.3851 | 1151 | -0.0339 | 0.654 ± 0.015 | FAIL | R1,R4a,R4b |
| hardest_fakes | 7 | pool_full | 64 | 0.2272 | 0.1643 | 0.3128 | 6.84% | 0.4762 | 0.3987 | 1158 | -0.0075 | 0.527 ± 0.016 | FAIL | R1,R3b,R4a,R4b |
| hardest_fakes | 7 | pool_nsa | 1 | 0.2432 | 0.1775 | 0.2631 | 6.95% | 0.4255 | 0.3172 | 1247 | +0.0026 | 0.321 ± 0.015 | FAIL | R1,R2,R4a |
| hardest_fakes | 7 | pool_nsa | 8 | 0.2276 | 0.1665 | 0.2676 | 6.71% | 0.4135 | 0.3548 | 1165 | +0.0057 | 0.163 ± 0.012 | FAIL | R1,R2,R4a |
| hardest_fakes | 7 | pool_nsa | 64 | 0.2234 | 0.1609 | 0.2698 | 6.54% | 0.4112 | 0.3698 | 1001 | +0.0055 | 0.207 ± 0.013 | FAIL | R1,R2,R4a |
| easiest_reals | 7 | pool_full | 1 | 0.3244 | 0.2251 | 0.3004 | 8.34% | 0.4861 | 0.4068 | 1538 | -0.0063 | 0.936 ± 0.008 | FAIL | R4a,R4b |
| easiest_reals | 7 | pool_full | 8 | 0.2507 | 0.1840 | 0.2990 | 7.27% | 0.4839 | 0.3878 | 1406 | -0.0061 | 0.669 ± 0.015 | FAIL | R1,R4a,R4b |
| easiest_reals | 7 | pool_full | 64 | 0.2318 | 0.1716 | 0.3020 | 6.83% | 0.4809 | 0.3803 | 1285 | -0.0029 | 0.602 ± 0.015 | FAIL | R1,R4a,R4b |
| easiest_reals | 7 | pool_nsa | 1 | 0.2308 | 0.1708 | 0.2389 | 6.70% | 0.4213 | 0.2532 | 1277 | -0.0097 | 0.712 ± 0.014 | FAIL | R1 |
| easiest_reals | 7 | pool_nsa | 8 | 0.1982 | 0.1518 | 0.2393 | 6.07% | 0.4138 | 0.2535 | 1056 | -0.0237 | 0.830 ± 0.012 | FAIL | R1,R4a |
| easiest_reals | 7 | pool_nsa | 64 | 0.1933 | 0.1472 | 0.2408 | 5.89% | 0.4120 | 0.2547 | 996 | -0.0246 | 0.870 ± 0.011 | FAIL | R1,R4a |
| easiest_fakes | 7 | pool_full | 1 | 0.2765 | 0.1969 | 0.3004 | 7.54% | 0.4109 | 0.4306 | 1498 | -0.0542 | 0.728 ± 0.014 | FAIL | R1,R4b |
| easiest_fakes | 7 | pool_full | 8 | 0.1975 | 0.1453 | 0.2937 | 5.92% | 0.3949 | 0.3966 | 1010 | -0.0592 | 0.724 ± 0.014 | FAIL | R1,R4b |
| easiest_fakes | 7 | pool_full | 64 | 0.1860 | 0.1374 | 0.2956 | 5.61% | 0.3892 | 0.3957 | 1022 | -0.0487 | 0.710 ± 0.014 | FAIL | R1,R4b |
| easiest_fakes | 7 | pool_nsa | 1 | 0.2334 | 0.1709 | 0.2575 | 6.73% | 0.4159 | 0.3544 | 1404 | -0.0071 | 0.664 ± 0.015 | FAIL | R1,R3b,R4b,R6 |
| easiest_fakes | 7 | pool_nsa | 8 | 0.2069 | 0.1530 | 0.2572 | 6.21% | 0.4094 | 0.3617 | 1119 | -0.0149 | 0.862 ± 0.011 | FAIL | R1 |
| easiest_fakes | 7 | pool_nsa | 64 | 0.2036 | 0.1495 | 0.2577 | 6.11% | 0.4086 | 0.3648 | 1059 | -0.0144 | 0.763 ± 0.013 | FAIL | R1,R4a |
| focal_g2 | 7 | pool_full | 1 | 0.3710 | 0.2392 | 0.3008 | 8.92% | 0.5151 | 0.3977 | 1767 | +0.0404 | 0.361 ± 0.015 | FAIL | R1,R2,R3b,R4a |
| focal_g2 | 7 | pool_full | 8 | 0.3574 | 0.2349 | 0.3023 | 8.72% | 0.5100 | 0.4103 | 1994 | +0.1006 | 0.118 ± 0.010 | FAIL | R1,R2,R3b,R4a,R4b,R6 |
| focal_g2 | 7 | pool_full | 64 | INVALID: ValueError: focal gradient norm 0.0112 > 0.01 on fold 2 | | | | | | | | | INVALID | |
| focal_g2 | 7 | pool_nsa | 1 | 0.3295 | 0.2176 | 0.2512 | 8.04% | 0.4618 | 0.2502 | 1608 | +0.0890 | 0.295 ± 0.014 | FAIL | R1,R2,R3b,R4a,R6 |
| focal_g2 | 7 | pool_nsa | 8 | 0.3109 | 0.2144 | 0.2553 | 7.93% | 0.4615 | 0.2435 | 1640 | +0.0891 | 0.295 ± 0.014 | FAIL | R1,R2,R3b,R4a,R6 |
| focal_g2 | 7 | pool_nsa | 64 | 0.3024 | 0.2113 | 0.2587 | 7.90% | 0.4647 | 0.2404 | 1554 | +0.0844 | 0.293 ± 0.014 | FAIL | R1,R2,R3b,R4a,R6 |
| pilot_quantile | 7 | pool_full | 1 | 0.3385 | 0.2285 | 0.3028 | 8.58% | 0.4952 | 0.4094 | 1710 | +0.0079 | 0.321 ± 0.015 | FAIL | R1,R2,R4a,R4b |
| pilot_quantile | 7 | pool_full | 8 | 0.2274 | 0.1721 | 0.3011 | 6.79% | 0.4648 | 0.3751 | 1269 | -0.0294 | 0.660 ± 0.015 | FAIL | R1 |
| pilot_quantile | 7 | pool_full | 64 | 0.2255 | 0.1696 | 0.3040 | 6.82% | 0.4830 | 0.3964 | 1205 | -0.0093 | 0.522 ± 0.016 | FAIL | R1,R4a,R4b |
| pilot_quantile | 7 | pool_nsa | 1 | 0.2626 | 0.1867 | 0.2554 | 7.03% | 0.3933 | 0.3138 | 1385 | +0.0221 | 0.305 ± 0.015 | FAIL | R1,R2,R3b,R6 |
| pilot_quantile | 7 | pool_nsa | 8 | 0.2122 | 0.1561 | 0.2586 | 6.25% | 0.3895 | 0.3518 | 1109 | -0.0096 | 0.887 ± 0.010 | INDETERMINATE | R1 |
| pilot_quantile | 7 | pool_nsa | 64 | 0.2104 | 0.1546 | 0.2585 | 6.18% | 0.3931 | 0.3493 | 1115 | -0.0075 | 0.852 ± 0.011 | FAIL | R1 |

- **`hardest_reals` is the one consistent single-layer lead, and it is pool-dependent.** On `pool_nsa` it is better at every C (−0.019 / −0.023 / −0.027, P(better) 0.961 / 0.965 / 0.900), passing the gate at C=1 and C=8 and INDETERMINATE at C=64 (P(better) exactly 0.900, within 2 SE of the band). On `pool_full` the same recipe is **worse** at C=1 and C=64 (+0.051, +0.038; P 0.10, 0.05). The mined set explains why: on `pool_full` 39% of the hardest reals are ASVspoof 2019 bona fide rows, and up-weighting them moves the boundary for a population the NSA basis does not contain. The pilot's "reals x4 wins" was a `pool_full` finding under a looser prior (W1); this is the matched-regularization, NSA-basis version, and it only holds without the ASVspoof rows in the fit.
- **`hardest_fakes` does nothing reliable** (`pool_full` −0.030 / −0.034 / −0.008 with P 0.65 / 0.65 / 0.53; `pool_nsa` slightly worse at every C).
- **`easiest_fakes` has the largest raw deltas on `pool_full`** (−0.054 / −0.059 / −0.049) with P(better) only 0.71 to 0.73: the gain concentrates in a few of the 14 spoof groups, so a group bootstrap is wide. `easiest_reals` is small and positive-direction throughout (−0.003 to −0.025, P 0.60 to 0.94).
- **Focal loss (γ=2) is worse everywhere**: +0.040 and +0.101 on `pool_full` (C=64 INVALID: final gradient norm 0.0112 > 0.01 on fold 2), +0.089 / +0.089 / +0.084 on `pool_nsa`, with 30 to 40% more false alarms at its own threshold and P(better) 0.12 to 0.36. The pilot had focal first *only* under the per-fold mean (spec D2); under pooled package-0.5 it is the worst arm in the grid. Focal is also the one arm not on the matched `sum(w) = n` footing, so part of this is a regularization difference; it is reported as descriptive for that reason.
- **The pilot-comparable arm does not reproduce the pilot's gain.** `pilot_quantile` is worse than base at C=1 on both pools (+0.008, +0.022) and modestly better at C=8 and C=64 (−0.029 / −0.009, −0.010 / −0.008) with P(better) at most 0.887 (INDETERMINATE at C=8, FAIL elsewhere). Its mined rows are two-thirds the same clips as `hardest_both` (Jaccard 0.66 to 0.71), so the pilot's `mine_nested` row was a hardest-example recipe on the full basis read under the per-fold mean; none of those three choices survives here.

## The regularization constant is the larger lever (descriptive; a lead for V2-01, not a selection)

The `none` arm, C in {0.25, 1, 2, 4, 8, 16, 64, 256}, every layer and pool, NSA-only basis. v1 inherited `C = 1.0` (`src/hearsay/probe.py:43-47`) and selected under the brief reading, where C barely matters; under rule 5's reading it is the largest single effect in this lane.

| L | pool | pkg-0.5 at C=0.25 | C=1 (v1) | C=8 | C=64 | C=256 | Δ C=1→256 | brief C=1→256 | FA at own thr, C=1→256 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | pool_full | 0.4286 | 0.3544 | 0.2805 | 0.2697 | 0.2705 | -0.0839 | 0.3026 → 0.3236 | 1660 → 1620 |
| 6 | pool_nsa | 0.3443 | 0.2573 | 0.2253 | 0.2197 | 0.2191 | -0.0383 | 0.2727 → 0.2802 | 1509 → 1238 |
| 7 | pool_full | 0.4360 | 0.3307 | 0.2568 | 0.2348 | 0.2313 | -0.0993 | 0.3013 → 0.3074 | 1642 → 1285 |
| 7 | pool_nsa | 0.3241 | 0.2405 | 0.2219 | 0.2179 | 0.2177 | -0.0228 | 0.2567 → 0.2646 | 1258 → 1060 |
| 8 | pool_full | 0.4825 | 0.4251 | 0.3285 | 0.2968 | 0.2911 | -0.1339 | 0.3041 → 0.3232 | 1983 → 1483 |
| 8 | pool_nsa | 0.3491 | 0.2940 | 0.2577 | 0.2550 | 0.2541 | -0.0399 | 0.2934 → 0.2869 | 1486 → 1230 |
| 20 | pool_full | 0.6779 | 0.6581 | 0.5884 | 0.4675 | 0.4549 | -0.2032 | 0.3292 → 0.3235 | 3013 → 2466 |
| 20 | pool_nsa | 0.5959 | 0.5910 | 0.4290 | 0.3920 | 0.3883 | -0.2027 | 0.4141 → 0.4022 | 2414 → 2042 |

Package-0.5 falls monotonically (or nearly) with C on every layer and pool, by 0.023 (L7, `pool_nsa`) to 0.203 (L20, `pool_nsa`); the brief reading moves the other way by 0.001 to 0.011 everywhere except L20 `pool_nsa`. The gain is in the error-carrying folds (f0 and f4 both improve at L7 `pool_full`: 0.4735 → 0.4552 and 0.3950 → 0.3624) and the false-alarm count at the cell's own threshold falls with it (L7 `pool_nsa` 1,258 → 1,060). All 64 rows are in Appendix B.

**Why this is a lead and not a selection.** Selecting C on inner OOF is the L7+WavLM trap (`docs/reports/2026-10-03_layer-diversity.md:64-80`): inner CV contains no wild real speech and cannot price wild false alarms, which is where v1's deep columns actually failed. The curve goes to the V2-01 benchmark lane, which can read it on the holdout cohort and In-the-Wild under the read-once ledger. Note also the convention dependence: package-0.5 and package-0.3 improve monotonically with C while brief gets slightly worse, so R3b would refuse a C change as a "candidate" until NSA's cost convention is settled.

The full 64-row sweep is in the appendix.

## Mined-set characterisation (R7, H4) and arm overlap (H3)

Every mined cell now carries a characterisation of the rows it actually up-weighted (pooled over the five outer folds; each pick counted once per fold that selected it, so a row can appear up to four times). Summary over the 102 cells that mine rows (the five OK `focal_g2` cells are a continuous loss with no picks, so they carry no mined set), with the full table in Appendix D:

| | hardest (both / reals) | easiest (both / reals) | random | pilot_quantile |
|---|---|---|---|---|
| Real side, top `group` share (R7 cap 0.25) | 0.042 to 0.139 | 0.066 to 0.113 | 0.016 to 0.031 | 0.098 to 0.101 |
| Real side, top speaker | `lj` at 23 to 38% | `lj` at 24% (`pool_nsa`) | `lj` at 63% (its population share) | `lj` at 24 to 25% |
| Real side, sources (`pool_nsa`, mean) | LibriSpeech 0.62, LJ 0.38 | LibriSpeech 0.75, LJ 0.25 | LJ 0.63, LibriSpeech 0.37 | LibriSpeech 0.76, LJ 0.24 |
| Fake side, top `group` share (reported, not gated) | **0.62 to 0.80, always `gen:grad_tts`** | 0.34 to 0.70 (`gen:your_tts` or `asv19_A05`) | 0.10 to 0.13 | 0.79, `gen:grad_tts` |
| Fake side, generators (`pool_nsa`, mean) | grad_tts 0.76, ElevenLabs 0.22 | your_tts 0.54, OpenVoice v2 0.19, pro_diff 0.15, XTTS v2 0.09 | flat, 0.12 to 0.13 each | grad_tts 0.79, ElevenLabs 0.19 |

Readings:

- **R7 holds for every mined set** (max 0.139 vs the 0.25 cap), and the real-side mined sets are *less* speaker-concentrated than the population: LJ, 63% of `pool_nsa` reals, is 23 to 38% of the hardest or easiest reals. Both extremes over-sample LibriSpeech, the multi-speaker source. So "the hardest reals" are not one voice, which is the H6 worry, and the characterisation readout is now a real check rather than a constant.
- **The hardest fakes are the two generators v1 struggled with.** 76% `grad_tts` and 22% ElevenLabs on `pool_nsa`; this is the W3 point made concrete (whole generators are groups, so a fake-side cap is unsatisfiable by construction) and it says the hardest-fake recipe is mostly "up-weight grad_tts". The easiest fakes are `your_tts`, OpenVoice v2 and `pro_diff`.
- **The arms are distinct (H3).** Jaccard over unique mined rows: hardest vs easiest 0.000 to 0.002 (disjoint, as they must be); hardest vs random 0.064 to 0.075 and easiest vs random 0.070 to 0.084 (the overlap two independent 5% samples pooled over four training folds produce); hardest vs the pilot-comparable arm 0.662 to 0.705. No arm collapsed onto another; the in-sample family that would have (W2) was withdrawn before the run.
- Per-fold maximum real group share is at most 0.27 (`pilot_quantile`, which mines only 2% of reals), so no single fold's mined set is dominated by one group either.
- **Duration** is not reported: the manifests carry no duration column and this lane may not read audio.

Full H3 table (72 pairs):

| pair | stratum | Jaccard |
|---|---|---|
| easiest_both vs random_both | L06 pool_full C1 | 0.074 |
| easiest_both vs random_both | L06 pool_full C64 | 0.076 |
| easiest_both vs random_both | L06 pool_full C8 | 0.073 |
| easiest_both vs random_both | L06 pool_nsa C1 | 0.076 |
| easiest_both vs random_both | L06 pool_nsa C64 | 0.076 |
| easiest_both vs random_both | L06 pool_nsa C8 | 0.077 |
| easiest_both vs random_both | L07 pool_full C1 | 0.075 |
| easiest_both vs random_both | L07 pool_full C64 | 0.075 |
| easiest_both vs random_both | L07 pool_full C8 | 0.074 |
| easiest_both vs random_both | L07 pool_nsa C1 | 0.070 |
| easiest_both vs random_both | L07 pool_nsa C64 | 0.072 |
| easiest_both vs random_both | L07 pool_nsa C8 | 0.071 |
| easiest_both vs random_both | L08 pool_full C1 | 0.073 |
| easiest_both vs random_both | L08 pool_full C64 | 0.072 |
| easiest_both vs random_both | L08 pool_full C8 | 0.072 |
| easiest_both vs random_both | L08 pool_nsa C1 | 0.072 |
| easiest_both vs random_both | L08 pool_nsa C64 | 0.074 |
| easiest_both vs random_both | L08 pool_nsa C8 | 0.075 |
| easiest_both vs random_both | L20 pool_full C1 | 0.073 |
| easiest_both vs random_both | L20 pool_full C64 | 0.073 |
| easiest_both vs random_both | L20 pool_full C8 | 0.072 |
| easiest_both vs random_both | L20 pool_nsa C1 | 0.081 |
| easiest_both vs random_both | L20 pool_nsa C64 | 0.083 |
| easiest_both vs random_both | L20 pool_nsa C8 | 0.084 |
| hardest_both vs easiest_both | L06 pool_full C1 | 0.001 |
| hardest_both vs easiest_both | L06 pool_full C64 | 0.002 |
| hardest_both vs easiest_both | L06 pool_full C8 | 0.001 |
| hardest_both vs easiest_both | L06 pool_nsa C1 | 0.001 |
| hardest_both vs easiest_both | L06 pool_nsa C64 | 0.001 |
| hardest_both vs easiest_both | L06 pool_nsa C8 | 0.001 |
| hardest_both vs easiest_both | L07 pool_full C1 | 0.001 |
| hardest_both vs easiest_both | L07 pool_full C64 | 0.001 |
| hardest_both vs easiest_both | L07 pool_full C8 | 0.001 |
| hardest_both vs easiest_both | L07 pool_nsa C1 | 0.000 |
| hardest_both vs easiest_both | L07 pool_nsa C64 | 0.000 |
| hardest_both vs easiest_both | L07 pool_nsa C8 | 0.000 |
| hardest_both vs easiest_both | L08 pool_full C1 | 0.002 |
| hardest_both vs easiest_both | L08 pool_full C64 | 0.002 |
| hardest_both vs easiest_both | L08 pool_full C8 | 0.002 |
| hardest_both vs easiest_both | L08 pool_nsa C1 | 0.000 |
| hardest_both vs easiest_both | L08 pool_nsa C64 | 0.000 |
| hardest_both vs easiest_both | L08 pool_nsa C8 | 0.000 |
| hardest_both vs easiest_both | L20 pool_full C1 | 0.001 |
| hardest_both vs easiest_both | L20 pool_full C64 | 0.001 |
| hardest_both vs easiest_both | L20 pool_full C8 | 0.001 |
| hardest_both vs easiest_both | L20 pool_nsa C1 | 0.001 |
| hardest_both vs easiest_both | L20 pool_nsa C64 | 0.001 |
| hardest_both vs easiest_both | L20 pool_nsa C8 | 0.001 |
| hardest_both vs pilot_quantile | L07 pool_full C1 | 0.662 |
| hardest_both vs pilot_quantile | L07 pool_full C64 | 0.668 |
| hardest_both vs pilot_quantile | L07 pool_full C8 | 0.668 |
| hardest_both vs pilot_quantile | L07 pool_nsa C1 | 0.705 |
| hardest_both vs pilot_quantile | L07 pool_nsa C64 | 0.703 |
| hardest_both vs pilot_quantile | L07 pool_nsa C8 | 0.704 |
| hardest_both vs random_both | L06 pool_full C1 | 0.069 |
| hardest_both vs random_both | L06 pool_full C64 | 0.070 |
| hardest_both vs random_both | L06 pool_full C8 | 0.070 |
| hardest_both vs random_both | L06 pool_nsa C1 | 0.073 |
| hardest_both vs random_both | L06 pool_nsa C64 | 0.073 |
| hardest_both vs random_both | L06 pool_nsa C8 | 0.073 |
| hardest_both vs random_both | L07 pool_full C1 | 0.067 |
| hardest_both vs random_both | L07 pool_full C64 | 0.069 |
| hardest_both vs random_both | L07 pool_full C8 | 0.068 |
| hardest_both vs random_both | L07 pool_nsa C1 | 0.074 |
| hardest_both vs random_both | L07 pool_nsa C64 | 0.075 |
| hardest_both vs random_both | L07 pool_nsa C8 | 0.075 |
| hardest_both vs random_both | L08 pool_full C1 | 0.065 |
| hardest_both vs random_both | L08 pool_full C64 | 0.064 |
| hardest_both vs random_both | L08 pool_full C8 | 0.064 |
| hardest_both vs random_both | L08 pool_nsa C1 | 0.072 |
| hardest_both vs random_both | L08 pool_nsa C64 | 0.074 |
| hardest_both vs random_both | L08 pool_nsa C8 | 0.072 |
| hardest_both vs random_both | L20 pool_full C1 | 0.069 |
| hardest_both vs random_both | L20 pool_full C64 | 0.070 |
| hardest_both vs random_both | L20 pool_full C8 | 0.069 |
| hardest_both vs random_both | L20 pool_nsa C1 | 0.067 |
| hardest_both vs random_both | L20 pool_nsa C64 | 0.067 |
| hardest_both vs random_both | L20 pool_nsa C8 | 0.067 |
| hardest_fakes vs easiest_fakes | L07 pool_full C1 | 0.000 |
| hardest_fakes vs easiest_fakes | L07 pool_full C64 | 0.000 |
| hardest_fakes vs easiest_fakes | L07 pool_full C8 | 0.000 |
| hardest_fakes vs easiest_fakes | L07 pool_nsa C1 | 0.000 |
| hardest_fakes vs easiest_fakes | L07 pool_nsa C64 | 0.000 |
| hardest_fakes vs easiest_fakes | L07 pool_nsa C8 | 0.000 |
| hardest_reals vs easiest_reals | L07 pool_full C1 | 0.002 |
| hardest_reals vs easiest_reals | L07 pool_full C64 | 0.002 |
| hardest_reals vs easiest_reals | L07 pool_full C8 | 0.002 |
| hardest_reals vs easiest_reals | L07 pool_nsa C1 | 0.000 |
| hardest_reals vs easiest_reals | L07 pool_nsa C64 | 0.000 |
| hardest_reals vs easiest_reals | L07 pool_nsa C8 | 0.000 |

## H5 as frozen

The frozen definition counts, per cell, the false alarms at the package-0.5 threshold whose normalised real-side margin lies in the worst decile of all reals. On the NSA basis the decile is 814 of 8,142 reals; the fewest false alarms any cell has at its own threshold is 996 (`easiest_reals`, L7, `pool_nsa`, C=64). Whenever the false alarms cover the decile, every worst-decile real is a false alarm and the count is the decile: **815 in all 131 OK cells on the NSA basis, 1,327 in every full-basis reading.** `report.json` records this (`h5_saturation.saturated = true`). The count therefore says nothing about any arm, and the question Nathan asked in P2, whether confidently-wrong false positives are *separately visible*, is answered "not by this definition." This was knowable when the definition was frozen: the spec's own reference table records false-alarm rates of 17.7% and 20.2% at the package-0.5 threshold, both above the 10% decile, so the saturation was predictable on Oct 13 and was not predicted (Claude critique, finding 4). It is left as frozen rather than redefined after the read. A definition that can move, for stage 0.5: the number of false alarms beyond a fixed normalised-margin cut set on the *base* arm (so arms are compared on one scale), or the count of reals scored above the base's 99th-percentile real score.

## Base reference and self-check

| Cell | Basis | package-0.5 | package-0.3 | brief | EER | threshold | FA | miss |
|---|---|---|---|---|---|---|---|---|
| `none`, L7, `pool_full`, C=1 | full 23,790 | 0.3104 | 0.2082 | 0.2649 | 7.74% | −9.9743 | 2,348 | 351 |
| `none`, L7, `pool_full`, C=1 | NSA-only 16,142 | 0.3307 | 0.2280 | 0.3013 | 8.45% | | 1,642 | |
| `none`, L7, `pool_nsa`, C=1 (**reference cell**) | NSA-only 16,142 | 0.2405 | 0.1766 | 0.2567 | 6.86% | | 1,258 | |

The first two rows reproduce the spec's verified reference values exactly (0.3104 / 2,348 and 0.3307 / 1,642). The `pool_full` L7 C=1 cell is the shipped L7 probe re-fitted here, and the blocking self-check held in both runs: Spearman with the shipped `outputs/v2/bakeoff/probes/L07.npz` OOF exactly 1.0 and a package-0.5 gap below 1e-12 (the shipped OOF is Platt-rescaled, so a raw max|diff| would read 57.4 on this identical ranking; that is the trap the pilot's `print` fell into). Note that `pool_nsa` at the same layer and C scores **0.2405**, better than the shipped probe's 0.3307 on the same rows: dropping the 7,648 ASVspoof 2019 train/dev rows from the fit helps the NSA basis at C=1. That is a pool effect, reported, not a candidate.

Run preconditions (R8) held: zero `decode_error` rows, zero all-zero embedding rows, every manifest row joined to the fold file with label, generator, speaker and source agreeing, the inner path set identical to `outputs/v2/bakeoff/rows.json` (23,790 rows in the same order), 0 of 158 groups crossing folds, and the speaker crossing documented above. The one INVALID cell (`focal_g2`, L7, `pool_full`, C=64: final gradient norm 0.0112 against the 0.01 bound on fold 2) is refused under R9 and counted, not scored.

## What this does not measure

- **Nothing here is a validation result.** Legacy split with speaker overlap; nested inner OOF only; the outer holdout, In-the-Wild, MLAAD and the test set were not read (D3, rules 3 and 7).
- **No wild real speech**, so wild false alarms, the axis on which v1's deep columns failed, are unpriced. **No laundering, telephony, codec, reverb, short-crop or physical-replay conditions.**
- **The cost convention is unresolved** (D5: the NSA document says a false alarm costs 4x a miss; the shipped code puts the 4 on the miss). All three readings are reported and R3b refuses any convention-dependent sign; **adoption of anything in this lane is blocked until NSA answers**, and the question stays on the open list in `docs/reports/2026-10-10_v2-fusion-research.md`.
- **A probe on frozen XLS-R embeddings only.** Nothing is said about example weighting inside a fine-tuned head (the M5 recipe) or the AntiDeepfake backbones; that is stage 2.
- **Release blocker, stated plainly:** V2-33 cannot close on this evidence. Its "done when" needs a V2-01 benchmark row for a mined model beside its unmined twin on speaker-disjoint folds, with wild-real false alarms reported separately.

## Decisions for Nathan (the plan's STOP 3 items, now with numbers)

1. **The C finding is bigger than the question.** At L7 on `pool_nsa`, package-0.5 goes from 0.2405 at C=1 to 0.2177 at C=256; at L20 from 0.591 to 0.388; on `pool_full` at L7 from 0.3307 to 0.2313 (NSA basis). Every weighting shape in this lane moves the score by less than the C=1 → C=8 step at the same layer. The brief reading worsens by up to 0.011 over the same range. Confirm the disposition: a lead handed to the V2-01 lane, not adopted here.
2. **132 gated cells against the "four candidates against one gate" cap.** The primary answer is one three-way comparison with a random control; no cell is adopted. Still a departure from the letter of that rule, recorded.
3. **The cost convention.** It disagrees about C's direction (package readings improve, brief worsens). Adoption stays blocked.
4. **New: the nine readout, gate and inference defects** (three found at close-out, six by the Codex audit and the Claude critique of the first commit). Fixed and re-run on Oct 17 with byte-identical scores. Confirm that fixing the code to match the pre-declaration, without touching any band, is acceptable post-read; H5 was deliberately left as frozen.
5. **New: what stage 0.5 should inherit.** (a) The hypothesis to carry is **easiest-side emphasis**, with hardest as the comparison and random as the control, not hard-negative mining as originally framed. (b) `hardest_reals` on the NSA-only pool is the one lead worth a benchmark row; it reverses when the ASVspoof 2019 rows are in the fit, so the pool is part of the recipe. (c) `focal_g2` should be dropped from stage 0.5's arms. (d) H5 needs the scale-anchored definition above before it is used as a readout. (e) C becomes an explicit axis of any stage-0.5 or stage-1 design, never inherited at 1.0. (f) Speaker-disjoint folds are a prerequisite for anything beyond a diagnostic (Codex #1).

## Disclosures

- **No pre-run commit exists.** P1 said every recipe, selection rule and band would be committed before any number was read; the script, spec and plan were first committed on Oct 17 (`7ba7faf`), after both runs. The pre-declaration's integrity therefore rests on the file history on disk (the spec's FINAL DESIGN and the plan's Codex dispositions predate run 1's 17:11 start on Oct 13) and on the preserved run-1 outputs, not on a commit, and cannot be independently verified from git (Codex audit round 1, #6). Every confirmation run from here on commits its declaration before it executes.
- **The inference layer changed after the first read, and this is stated as such.** The first commit's headline read stratum counts; the design of record and the Codex plan review (#2) had specified one aggregate inside shared group-resamples with its uncertainty, per C. Both post-commit reviews flagged it; `aggregate_contrast()` now computes exactly that, and the Verdict is restated on it. The direction of every conclusion is unchanged; what the aggregate adds is that the easiest-side gain, while consistently signed, has a 90% interval whose lower end is within 0.0015 of zero.
- **The gate changed after the first read, in three ways the pre-declaration already required:** an R1-indeterminate cell is INDETERMINATE and not a pass (one of the first commit's nine passes); R9 degeneracy makes a cell INVALID at validation rather than an ordinary FAIL; the gate validates every checkpoint's fingerprint and refuses a partial grid or an expired deadline. No band moved.
- **The checkpoint fingerprint changed.** It now hashes the embedding shard contents (not only the manifests) and the source of the fitting-path functions (not the whole script), so a gate or report edit no longer forces a refit while any change to how a cell is fitted or read does. Run 4 recomputed every cell under it.
- **`experiments/log.csv` carries two rows for this rung.** The 01:13 UTC row records the first commit's gate (9 passes, C-pooled means); the 02:13 UTC row records the final gate (8 passes, per-C aggregate P(better)) and supersedes it. Both stay: the log is append-only.
- Run 2 happened after run 1's numbers had been read. The fixes change what R7 and R4 *test*, not any score, and the identity check above is the evidence. The deadline file was reset for run 2 (run 1's is preserved with its outputs).
- The log row's `validation_score` is the reference cell (`none`, L7, `pool_nsa`, C=1, NSA-only basis), because no candidate passed and nothing was adopted; the notes name the basis and all three readings.
- Codex's eleven plan-review findings were all accepted; the gate and tests implement them (spec, "Codex plan review round 1, disposition"). The stage-1 bandit contract (G1), the one-sided-augmentation gate (G2) and the SNR cap (G4) are pre-declared in the spec but have no code yet; they are stage 0.5 / 1 work.
- The lane's test file grew from 82 to 94 tests on Oct 17: the eight named in the defects table, the three-seed C2 directional test the break check needed (it had been named in the inventory but never written), and an F4 test that the log row carries the basis and every reading. The full suite's one failure at the start of the close-out, `test_build_site.py::test_every_page_has_shell_invariants`, was this lane's: the plan and spec each had more than one top-level heading, so the docs site rendered two `<h1>`s; the extra headings are now `##`.

## Appendix A: every cell

Δ is against the `none` cell at the same layer, pool and C; all readings on the NSA-only basis at each cell's own package-0.5 threshold; f0 and f4 are per-fold package-0.5; P(better) ± its Monte-Carlo SE. Verdicts are the final gate's (PASS, INDETERMINATE, FAIL, BASE, INVALID).

| arm | L | pool | C | pkg-0.5 | pkg-0.3 | brief | EER | f0 | f4 | FA@thr | Δpkg-0.5 vs none | P(better) ± SE | verdict | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 6 | pool_full | 1 | 0.3544 | 0.2287 | 0.3026 | 8.47% | 0.5413 | 0.2907 | 1660 | +0.0000 |  | BASE |  |
| none | 6 | pool_full | 8 | 0.2805 | 0.1977 | 0.3197 | 7.76% | 0.5472 | 0.3057 | 1445 | +0.0000 |  | BASE |  |
| none | 6 | pool_full | 64 | 0.2697 | 0.1982 | 0.3228 | 7.74% | 0.5487 | 0.3079 | 1622 | +0.0000 |  | BASE |  |
| none | 6 | pool_nsa | 1 | 0.2573 | 0.1959 | 0.2727 | 7.65% | 0.4717 | 0.2677 | 1509 | +0.0000 |  | BASE |  |
| none | 6 | pool_nsa | 8 | 0.2253 | 0.1771 | 0.2805 | 7.04% | 0.4662 | 0.2735 | 1252 | +0.0000 |  | BASE |  |
| none | 6 | pool_nsa | 64 | 0.2197 | 0.1741 | 0.2804 | 6.88% | 0.4638 | 0.2637 | 1211 | +0.0000 |  | BASE |  |
| none | 7 | pool_full | 1 | 0.3307 | 0.2280 | 0.3013 | 8.45% | 0.4735 | 0.3950 | 1642 | +0.0000 |  | BASE |  |
| none | 7 | pool_full | 8 | 0.2568 | 0.1867 | 0.3050 | 7.11% | 0.4553 | 0.3695 | 1472 | +0.0000 |  | BASE |  |
| none | 7 | pool_full | 64 | 0.2348 | 0.1729 | 0.3077 | 6.70% | 0.4546 | 0.3630 | 1313 | +0.0000 |  | BASE |  |
| none | 7 | pool_nsa | 1 | 0.2405 | 0.1766 | 0.2567 | 6.86% | 0.4116 | 0.3394 | 1258 | +0.0000 |  | BASE |  |
| none | 7 | pool_nsa | 8 | 0.2219 | 0.1628 | 0.2652 | 6.43% | 0.4002 | 0.3825 | 1155 | +0.0000 |  | BASE |  |
| none | 7 | pool_nsa | 64 | 0.2179 | 0.1599 | 0.2646 | 6.37% | 0.3968 | 0.3856 | 1066 | +0.0000 |  | BASE |  |
| none | 8 | pool_full | 1 | 0.4251 | 0.2769 | 0.3041 | 10.28% | 0.5390 | 0.5027 | 1983 | +0.0000 |  | BASE |  |
| none | 8 | pool_full | 8 | 0.3285 | 0.2339 | 0.3235 | 8.96% | 0.5393 | 0.5155 | 1771 | +0.0000 |  | BASE |  |
| none | 8 | pool_full | 64 | 0.2968 | 0.2174 | 0.3239 | 8.52% | 0.5356 | 0.5173 | 1529 | +0.0000 |  | BASE |  |
| none | 8 | pool_nsa | 1 | 0.2940 | 0.2067 | 0.2934 | 8.00% | 0.4373 | 0.4558 | 1486 | +0.0000 |  | BASE |  |
| none | 8 | pool_nsa | 8 | 0.2577 | 0.1808 | 0.2869 | 7.20% | 0.4330 | 0.4515 | 1223 | +0.0000 |  | BASE |  |
| none | 8 | pool_nsa | 64 | 0.2550 | 0.1793 | 0.2867 | 7.06% | 0.4331 | 0.4500 | 1258 | +0.0000 |  | BASE |  |
| none | 20 | pool_full | 1 | 0.6581 | 0.3843 | 0.3292 | 14.61% | 0.6948 | 0.9054 | 3013 | +0.0000 |  | BASE |  |
| none | 20 | pool_full | 8 | 0.5884 | 0.3820 | 0.3320 | 14.25% | 0.6847 | 0.9063 | 3504 | +0.0000 |  | BASE |  |
| none | 20 | pool_full | 64 | 0.4675 | 0.3416 | 0.3247 | 12.79% | 0.6852 | 0.9084 | 2540 | +0.0000 |  | BASE |  |
| none | 20 | pool_nsa | 1 | 0.5910 | 0.3671 | 0.4141 | 13.56% | 0.6040 | 0.8612 | 2414 | +0.0000 |  | BASE |  |
| none | 20 | pool_nsa | 8 | 0.4290 | 0.2987 | 0.4004 | 11.28% | 0.6001 | 0.8682 | 2227 | +0.0000 |  | BASE |  |
| none | 20 | pool_nsa | 64 | 0.3920 | 0.2797 | 0.4034 | 10.68% | 0.5973 | 0.8633 | 1954 | +0.0000 |  | BASE |  |
| hardest_both | 6 | pool_full | 1 | 0.3277 | 0.2192 | 0.3101 | 8.32% | 0.5267 | 0.2693 | 1524 | -0.0267 | 0.646 ± 0.015 | FAIL | R1,R3b |
| hardest_both | 6 | pool_full | 8 | 0.2402 | 0.1675 | 0.3065 | 6.70% | 0.4998 | 0.2527 | 1300 | -0.0403 | 0.924 ± 0.008 | PASS |  |
| hardest_both | 6 | pool_full | 64 | 0.2421 | 0.1743 | 0.3179 | 6.91% | 0.5243 | 0.2603 | 1344 | -0.0276 | 0.961 ± 0.006 | PASS |  |
| hardest_both | 6 | pool_nsa | 1 | 0.2379 | 0.1817 | 0.2646 | 7.07% | 0.4437 | 0.2524 | 1412 | -0.0194 | 0.927 ± 0.008 | PASS |  |
| hardest_both | 6 | pool_nsa | 8 | 0.2204 | 0.1690 | 0.2631 | 6.68% | 0.4223 | 0.2442 | 1298 | -0.0049 | 0.770 ± 0.013 | FAIL | R1 |
| hardest_both | 6 | pool_nsa | 64 | 0.2129 | 0.1657 | 0.2627 | 6.58% | 0.4208 | 0.2349 | 1229 | -0.0068 | 0.822 ± 0.012 | FAIL | R1 |
| hardest_both | 7 | pool_full | 1 | 0.3732 | 0.2382 | 0.3031 | 8.83% | 0.5080 | 0.3985 | 1874 | +0.0425 | 0.077 ± 0.008 | FAIL | R1,R2,R4a,R6 |
| hardest_both | 7 | pool_full | 8 | 0.2732 | 0.1980 | 0.2919 | 7.59% | 0.5422 | 0.3594 | 1435 | +0.0165 | 0.199 ± 0.013 | FAIL | R1,R2,R3b,R4a |
| hardest_both | 7 | pool_full | 64 | 0.2756 | 0.1992 | 0.2994 | 7.61% | 0.5392 | 0.3690 | 1450 | +0.0408 | 0.083 ± 0.009 | FAIL | R1,R2,R3b,R4a,R6 |
| hardest_both | 7 | pool_nsa | 1 | 0.2330 | 0.1702 | 0.2538 | 6.55% | 0.4011 | 0.3107 | 1335 | -0.0075 | 0.893 ± 0.010 | INDETERMINATE | R1 |
| hardest_both | 7 | pool_nsa | 8 | 0.2049 | 0.1505 | 0.2582 | 5.98% | 0.3860 | 0.2988 | 1033 | -0.0170 | 0.884 ± 0.010 | INDETERMINATE | R1 |
| hardest_both | 7 | pool_nsa | 64 | 0.2019 | 0.1473 | 0.2516 | 5.85% | 0.3869 | 0.2837 | 1135 | -0.0160 | 0.883 ± 0.010 | INDETERMINATE | R1 |
| hardest_both | 8 | pool_full | 1 | 0.4454 | 0.2923 | 0.3113 | 10.91% | 0.5818 | 0.5554 | 2006 | +0.0203 | 0.106 ± 0.010 | FAIL | R1,R2,R4a,R4b |
| hardest_both | 8 | pool_full | 8 | 0.3185 | 0.2253 | 0.3191 | 8.84% | 0.6020 | 0.5475 | 1592 | -0.0100 | 0.639 ± 0.015 | FAIL | R1,R4a,R4b |
| hardest_both | 8 | pool_full | 64 | 0.2904 | 0.2110 | 0.3160 | 8.24% | 0.6106 | 0.4996 | 1481 | -0.0064 | 0.643 ± 0.015 | FAIL | R1,R4a |
| hardest_both | 8 | pool_nsa | 1 | 0.3062 | 0.2119 | 0.2897 | 8.12% | 0.4545 | 0.4855 | 1573 | +0.0122 | 0.188 ± 0.012 | FAIL | R1,R2,R3b,R4a,R4b |
| hardest_both | 8 | pool_nsa | 8 | 0.2446 | 0.1755 | 0.2939 | 6.86% | 0.4562 | 0.4672 | 1320 | -0.0131 | 0.677 ± 0.015 | FAIL | R1,R3b,R4a,R4b |
| hardest_both | 8 | pool_nsa | 64 | 0.2486 | 0.1757 | 0.2975 | 6.81% | 0.4575 | 0.4828 | 1238 | -0.0065 | 0.566 ± 0.016 | FAIL | R1,R3,R3b,R4a,R4b |
| hardest_both | 20 | pool_full | 1 | 0.6818 | 0.3899 | 0.3319 | 14.97% | 0.7186 | 0.9312 | 3015 | +0.0237 | 0.078 ± 0.008 | FAIL | R1,R2,R4a,R4b |
| hardest_both | 20 | pool_full | 8 | 0.6470 | 0.3890 | 0.3261 | 14.70% | 0.7125 | 0.9300 | 3513 | +0.0586 | 0.273 ± 0.014 | FAIL | R1,R2,R3b,R4a,R4b |
| hardest_both | 20 | pool_full | 64 | 0.5236 | 0.3593 | 0.3173 | 13.31% | 0.7065 | 0.9337 | 3013 | +0.0561 | 0.348 ± 0.015 | FAIL | R1,R2,R3b,R4a,R4b,R6 |
| hardest_both | 20 | pool_nsa | 1 | 0.6116 | 0.3764 | 0.4147 | 14.05% | 0.6276 | 0.8744 | 3579 | +0.0206 | 0.361 ± 0.015 | FAIL | R1,R2,R4a,R4b,R6 |
| hardest_both | 20 | pool_nsa | 8 | 0.4331 | 0.3095 | 0.4056 | 11.61% | 0.6164 | 0.8671 | 2476 | +0.0041 | 0.335 ± 0.015 | FAIL | R1,R2,R4a,R6 |
| hardest_both | 20 | pool_nsa | 64 | 0.3952 | 0.2872 | 0.4006 | 11.03% | 0.6203 | 0.8634 | 2049 | +0.0032 | 0.447 ± 0.016 | FAIL | R1,R2,R3b,R4a |
| easiest_both | 6 | pool_full | 1 | 0.3239 | 0.2119 | 0.2988 | 8.02% | 0.4857 | 0.2683 | 1611 | -0.0305 | 0.821 ± 0.012 | FAIL | R1 |
| easiest_both | 6 | pool_full | 8 | 0.2361 | 0.1674 | 0.3156 | 6.67% | 0.4714 | 0.2768 | 1230 | -0.0444 | 0.866 ± 0.011 | FAIL | R1 |
| easiest_both | 6 | pool_full | 64 | 0.2374 | 0.1663 | 0.3076 | 6.59% | 0.4746 | 0.2653 | 1273 | -0.0324 | 0.968 ± 0.006 | PASS |  |
| easiest_both | 6 | pool_nsa | 1 | 0.2464 | 0.1903 | 0.2676 | 7.42% | 0.4731 | 0.2760 | 1440 | -0.0110 | 0.768 ± 0.013 | FAIL | R1 |
| easiest_both | 6 | pool_nsa | 8 | 0.2173 | 0.1627 | 0.2840 | 6.72% | 0.4676 | 0.2765 | 1256 | -0.0080 | 0.632 ± 0.015 | FAIL | R1,R3b |
| easiest_both | 6 | pool_nsa | 64 | 0.2106 | 0.1603 | 0.2699 | 6.62% | 0.4634 | 0.2765 | 1230 | -0.0092 | 0.660 ± 0.015 | FAIL | R1,R4b |
| easiest_both | 7 | pool_full | 1 | 0.2851 | 0.1993 | 0.2996 | 7.55% | 0.4232 | 0.3803 | 1426 | -0.0455 | 0.931 ± 0.008 | PASS |  |
| easiest_both | 7 | pool_full | 8 | 0.2238 | 0.1688 | 0.3050 | 6.65% | 0.4162 | 0.3865 | 1281 | -0.0330 | 0.743 ± 0.014 | FAIL | R1,R4b |
| easiest_both | 7 | pool_full | 64 | 0.2171 | 0.1625 | 0.3091 | 6.65% | 0.4196 | 0.3930 | 1210 | -0.0177 | 0.668 ± 0.015 | FAIL | R1,R3b,R4b |
| easiest_both | 7 | pool_nsa | 1 | 0.2293 | 0.1669 | 0.2474 | 6.59% | 0.4124 | 0.2567 | 1240 | -0.0112 | 0.843 ± 0.012 | FAIL | R1 |
| easiest_both | 7 | pool_nsa | 8 | 0.1942 | 0.1449 | 0.2516 | 6.00% | 0.4102 | 0.2423 | 1154 | -0.0276 | 0.904 ± 0.009 | INDETERMINATE |  |
| easiest_both | 7 | pool_nsa | 64 | 0.1886 | 0.1413 | 0.2503 | 5.84% | 0.4096 | 0.2307 | 1080 | -0.0293 | 0.901 ± 0.009 | FAIL | R4a |
| easiest_both | 8 | pool_full | 1 | 0.4051 | 0.2727 | 0.3026 | 10.20% | 0.5584 | 0.5005 | 2175 | -0.0199 | 0.760 ± 0.014 | FAIL | R1,R4a |
| easiest_both | 8 | pool_full | 8 | 0.2875 | 0.2057 | 0.3121 | 8.23% | 0.5606 | 0.4987 | 1608 | -0.0410 | 0.747 ± 0.014 | FAIL | R1,R4a |
| easiest_both | 8 | pool_full | 64 | 0.2575 | 0.1910 | 0.3266 | 7.60% | 0.5553 | 0.5159 | 1396 | -0.0393 | 0.738 ± 0.014 | FAIL | R1,R3b,R4a |
| easiest_both | 8 | pool_nsa | 1 | 0.2875 | 0.2034 | 0.2828 | 7.91% | 0.4316 | 0.4211 | 1478 | -0.0065 | 0.640 ± 0.015 | FAIL | R1 |
| easiest_both | 8 | pool_nsa | 8 | 0.2336 | 0.1712 | 0.2783 | 6.83% | 0.4306 | 0.4273 | 1312 | -0.0241 | 0.760 ± 0.014 | FAIL | R1 |
| easiest_both | 8 | pool_nsa | 64 | 0.2289 | 0.1707 | 0.2795 | 6.76% | 0.4305 | 0.4359 | 1241 | -0.0261 | 0.794 ± 0.013 | FAIL | R1 |
| easiest_both | 20 | pool_full | 1 | 0.6584 | 0.3844 | 0.3302 | 14.53% | 0.6870 | 0.8915 | 3020 | +0.0004 | 0.537 ± 0.016 | FAIL | R1,R2 |
| easiest_both | 20 | pool_full | 8 | 0.5870 | 0.3675 | 0.3333 | 13.70% | 0.6794 | 0.9044 | 3029 | -0.0013 | 0.702 ± 0.014 | FAIL | R1,R3b |
| easiest_both | 20 | pool_full | 64 | 0.5068 | 0.3506 | 0.3242 | 13.01% | 0.6855 | 0.9292 | 2966 | +0.0393 | 0.243 ± 0.014 | FAIL | R1,R2,R3b,R4b,R6 |
| easiest_both | 20 | pool_nsa | 1 | 0.5155 | 0.3383 | 0.4082 | 12.61% | 0.6157 | 0.8212 | 2390 | -0.0754 | 0.961 ± 0.006 | FAIL | R4a |
| easiest_both | 20 | pool_nsa | 8 | 0.3558 | 0.2596 | 0.3996 | 10.14% | 0.6257 | 0.8210 | 1887 | -0.0733 | 0.917 ± 0.009 | FAIL | R4a |
| easiest_both | 20 | pool_nsa | 64 | 0.3328 | 0.2460 | 0.4034 | 9.66% | 0.6168 | 0.8173 | 1802 | -0.0592 | 0.862 ± 0.011 | FAIL | R1,R4a |
| random_both | 6 | pool_full | 1 | 0.3598 | 0.2277 | 0.3089 | 8.50% | 0.5409 | 0.2899 | 1806 | +0.0054 | 0.244 ± 0.014 | FAIL | R1,R2,R3b |
| random_both | 6 | pool_full | 8 | 0.2579 | 0.1850 | 0.3185 | 7.16% | 0.5321 | 0.2918 | 1400 | -0.0225 | 0.967 ± 0.006 | PASS |  |
| random_both | 6 | pool_full | 64 | 0.2415 | 0.1756 | 0.3243 | 6.81% | 0.5306 | 0.2906 | 1392 | -0.0282 | 0.968 ± 0.006 | FAIL | R3b |
| random_both | 6 | pool_nsa | 1 | 0.2494 | 0.1923 | 0.2785 | 7.54% | 0.4786 | 0.2643 | 1514 | -0.0079 | 0.671 ± 0.015 | FAIL | R1,R3b |
| random_both | 6 | pool_nsa | 8 | 0.2267 | 0.1727 | 0.2816 | 7.00% | 0.4727 | 0.2814 | 1243 | +0.0014 | 0.488 ± 0.016 | FAIL | R1,R2,R3b |
| random_both | 6 | pool_nsa | 64 | 0.2235 | 0.1703 | 0.2834 | 6.91% | 0.4716 | 0.2851 | 1189 | +0.0038 | 0.424 ± 0.016 | FAIL | R1,R2,R3b,R4b |
| random_both | 7 | pool_full | 1 | 0.3639 | 0.2436 | 0.3045 | 8.97% | 0.4937 | 0.4425 | 1941 | +0.0332 | 0.093 ± 0.009 | FAIL | R1,R2,R4a,R4b,R6 |
| random_both | 7 | pool_full | 8 | 0.2474 | 0.1848 | 0.3033 | 7.26% | 0.4782 | 0.4104 | 1347 | -0.0094 | 0.633 ± 0.015 | FAIL | R1,R4a,R4b |
| random_both | 7 | pool_full | 64 | 0.2280 | 0.1740 | 0.3065 | 6.92% | 0.4770 | 0.4108 | 1160 | -0.0068 | 0.604 ± 0.015 | FAIL | R1,R2,R3b,R4a,R4b |
| random_both | 7 | pool_nsa | 1 | 0.2421 | 0.1775 | 0.2577 | 6.93% | 0.4198 | 0.3441 | 1279 | +0.0016 | 0.542 ± 0.016 | FAIL | R1,R2 |
| random_both | 7 | pool_nsa | 8 | 0.2218 | 0.1643 | 0.2630 | 6.49% | 0.4113 | 0.3734 | 1167 | -0.0000 | 0.438 ± 0.016 | FAIL | R1,R2,R3b,R4a |
| random_both | 7 | pool_nsa | 64 | 0.2228 | 0.1625 | 0.2609 | 6.46% | 0.4100 | 0.3678 | 1049 | +0.0049 | 0.363 ± 0.015 | FAIL | R1,R2,R3b,R4a |
| random_both | 8 | pool_full | 1 | 0.4058 | 0.2763 | 0.3055 | 10.31% | 0.5543 | 0.5170 | 1879 | -0.0193 | 0.689 ± 0.015 | FAIL | R1,R3b,R4a,R4b |
| random_both | 8 | pool_full | 8 | 0.3115 | 0.2251 | 0.3252 | 8.86% | 0.5457 | 0.4903 | 1588 | -0.0170 | 0.684 ± 0.015 | FAIL | R1,R3b |
| random_both | 8 | pool_full | 64 | 0.2725 | 0.1991 | 0.3295 | 8.05% | 0.5425 | 0.4925 | 1661 | -0.0243 | 0.717 ± 0.014 | FAIL | R1,R3b |
| random_both | 8 | pool_nsa | 1 | 0.3053 | 0.2153 | 0.2942 | 8.24% | 0.4497 | 0.4766 | 1586 | +0.0113 | 0.057 ± 0.007 | FAIL | R1,R2,R4a,R4b |
| random_both | 8 | pool_nsa | 8 | 0.2735 | 0.1949 | 0.2947 | 7.69% | 0.4477 | 0.5068 | 1384 | +0.0158 | 0.030 ± 0.005 | FAIL | R1,R2,R4a,R4b,R6 |
| random_both | 8 | pool_nsa | 64 | 0.2698 | 0.1924 | 0.2951 | 7.64% | 0.4496 | 0.5108 | 1338 | +0.0148 | 0.040 ± 0.006 | FAIL | R1,R2,R4a,R4b |
| random_both | 20 | pool_full | 1 | 0.6652 | 0.3835 | 0.3366 | 14.62% | 0.6939 | 0.9059 | 3556 | +0.0072 | 0.437 ± 0.016 | FAIL | R1,R2,R3b,R6 |
| random_both | 20 | pool_full | 8 | 0.5858 | 0.3751 | 0.3324 | 13.94% | 0.6744 | 0.9241 | 3109 | -0.0025 | 0.699 ± 0.015 | FAIL | R1,R3b,R4b |
| random_both | 20 | pool_full | 64 | 0.4896 | 0.3475 | 0.3224 | 12.86% | 0.6697 | 0.9254 | 2655 | +0.0221 | 0.336 ± 0.015 | FAIL | R1,R2,R3b,R4b |
| random_both | 20 | pool_nsa | 1 | 0.6013 | 0.3749 | 0.4071 | 13.86% | 0.6124 | 0.8551 | 3007 | +0.0103 | 0.114 ± 0.010 | FAIL | R1,R2,R3b,R6 |
| random_both | 20 | pool_nsa | 8 | 0.4242 | 0.2972 | 0.3984 | 11.23% | 0.6205 | 0.8603 | 2078 | -0.0048 | 0.718 ± 0.014 | FAIL | R1,R4a |
| random_both | 20 | pool_nsa | 64 | 0.3963 | 0.2807 | 0.4026 | 10.74% | 0.6195 | 0.8606 | 1863 | +0.0043 | 0.365 ± 0.015 | FAIL | R1,R2,R3b,R4a |
| hardest_reals | 7 | pool_full | 1 | 0.3814 | 0.2450 | 0.2937 | 9.10% | 0.5108 | 0.4154 | 1709 | +0.0507 | 0.100 ± 0.009 | FAIL | R1,R2,R3b,R4a,R4b |
| hardest_reals | 7 | pool_full | 8 | 0.2532 | 0.1851 | 0.2942 | 7.14% | 0.5170 | 0.3933 | 1288 | -0.0036 | 0.604 ± 0.015 | FAIL | R1,R4a,R4b |
| hardest_reals | 7 | pool_full | 64 | 0.2722 | 0.1969 | 0.2971 | 7.56% | 0.5323 | 0.4068 | 1500 | +0.0375 | 0.048 ± 0.007 | FAIL | R1,R2,R3b,R4a,R4b,R6 |
| hardest_reals | 7 | pool_nsa | 1 | 0.2215 | 0.1620 | 0.2509 | 6.28% | 0.3837 | 0.3443 | 1095 | -0.0190 | 0.961 ± 0.006 | PASS |  |
| hardest_reals | 7 | pool_nsa | 8 | 0.1988 | 0.1482 | 0.2532 | 5.85% | 0.3808 | 0.3368 | 1081 | -0.0231 | 0.965 ± 0.006 | PASS |  |
| hardest_reals | 7 | pool_nsa | 64 | 0.1914 | 0.1427 | 0.2544 | 5.71% | 0.3799 | 0.3319 | 1001 | -0.0265 | 0.900 ± 0.009 | INDETERMINATE | R1 |
| hardest_fakes | 7 | pool_full | 1 | 0.3012 | 0.2175 | 0.3126 | 8.21% | 0.4845 | 0.4166 | 1638 | -0.0295 | 0.649 ± 0.015 | FAIL | R1,R3,R3b,R4a,R4b |
| hardest_fakes | 7 | pool_full | 8 | 0.2229 | 0.1663 | 0.3047 | 6.76% | 0.4773 | 0.3851 | 1151 | -0.0339 | 0.654 ± 0.015 | FAIL | R1,R4a,R4b |
| hardest_fakes | 7 | pool_full | 64 | 0.2272 | 0.1643 | 0.3128 | 6.84% | 0.4762 | 0.3987 | 1158 | -0.0075 | 0.527 ± 0.016 | FAIL | R1,R3b,R4a,R4b |
| hardest_fakes | 7 | pool_nsa | 1 | 0.2432 | 0.1775 | 0.2631 | 6.95% | 0.4255 | 0.3172 | 1247 | +0.0026 | 0.321 ± 0.015 | FAIL | R1,R2,R4a |
| hardest_fakes | 7 | pool_nsa | 8 | 0.2276 | 0.1665 | 0.2676 | 6.71% | 0.4135 | 0.3548 | 1165 | +0.0057 | 0.163 ± 0.012 | FAIL | R1,R2,R4a |
| hardest_fakes | 7 | pool_nsa | 64 | 0.2234 | 0.1609 | 0.2698 | 6.54% | 0.4112 | 0.3698 | 1001 | +0.0055 | 0.207 ± 0.013 | FAIL | R1,R2,R4a |
| easiest_reals | 7 | pool_full | 1 | 0.3244 | 0.2251 | 0.3004 | 8.34% | 0.4861 | 0.4068 | 1538 | -0.0063 | 0.936 ± 0.008 | FAIL | R4a,R4b |
| easiest_reals | 7 | pool_full | 8 | 0.2507 | 0.1840 | 0.2990 | 7.27% | 0.4839 | 0.3878 | 1406 | -0.0061 | 0.669 ± 0.015 | FAIL | R1,R4a,R4b |
| easiest_reals | 7 | pool_full | 64 | 0.2318 | 0.1716 | 0.3020 | 6.83% | 0.4809 | 0.3803 | 1285 | -0.0029 | 0.602 ± 0.015 | FAIL | R1,R4a,R4b |
| easiest_reals | 7 | pool_nsa | 1 | 0.2308 | 0.1708 | 0.2389 | 6.70% | 0.4213 | 0.2532 | 1277 | -0.0097 | 0.712 ± 0.014 | FAIL | R1 |
| easiest_reals | 7 | pool_nsa | 8 | 0.1982 | 0.1518 | 0.2393 | 6.07% | 0.4138 | 0.2535 | 1056 | -0.0237 | 0.830 ± 0.012 | FAIL | R1,R4a |
| easiest_reals | 7 | pool_nsa | 64 | 0.1933 | 0.1472 | 0.2408 | 5.89% | 0.4120 | 0.2547 | 996 | -0.0246 | 0.870 ± 0.011 | FAIL | R1,R4a |
| easiest_fakes | 7 | pool_full | 1 | 0.2765 | 0.1969 | 0.3004 | 7.54% | 0.4109 | 0.4306 | 1498 | -0.0542 | 0.728 ± 0.014 | FAIL | R1,R4b |
| easiest_fakes | 7 | pool_full | 8 | 0.1975 | 0.1453 | 0.2937 | 5.92% | 0.3949 | 0.3966 | 1010 | -0.0592 | 0.724 ± 0.014 | FAIL | R1,R4b |
| easiest_fakes | 7 | pool_full | 64 | 0.1860 | 0.1374 | 0.2956 | 5.61% | 0.3892 | 0.3957 | 1022 | -0.0487 | 0.710 ± 0.014 | FAIL | R1,R4b |
| easiest_fakes | 7 | pool_nsa | 1 | 0.2334 | 0.1709 | 0.2575 | 6.73% | 0.4159 | 0.3544 | 1404 | -0.0071 | 0.664 ± 0.015 | FAIL | R1,R3b,R4b,R6 |
| easiest_fakes | 7 | pool_nsa | 8 | 0.2069 | 0.1530 | 0.2572 | 6.21% | 0.4094 | 0.3617 | 1119 | -0.0149 | 0.862 ± 0.011 | FAIL | R1 |
| easiest_fakes | 7 | pool_nsa | 64 | 0.2036 | 0.1495 | 0.2577 | 6.11% | 0.4086 | 0.3648 | 1059 | -0.0144 | 0.763 ± 0.013 | FAIL | R1,R4a |
| focal_g2 | 7 | pool_full | 1 | 0.3710 | 0.2392 | 0.3008 | 8.92% | 0.5151 | 0.3977 | 1767 | +0.0404 | 0.361 ± 0.015 | FAIL | R1,R2,R3b,R4a |
| focal_g2 | 7 | pool_full | 8 | 0.3574 | 0.2349 | 0.3023 | 8.72% | 0.5100 | 0.4103 | 1994 | +0.1006 | 0.118 ± 0.010 | FAIL | R1,R2,R3b,R4a,R4b,R6 |
| focal_g2 | 7 | pool_full | 64 | INVALID: ValueError: focal gradient norm 0.0112 > 0.01 on fold 2 | | | | | | | | | INVALID | |
| focal_g2 | 7 | pool_nsa | 1 | 0.3295 | 0.2176 | 0.2512 | 8.04% | 0.4618 | 0.2502 | 1608 | +0.0890 | 0.295 ± 0.014 | FAIL | R1,R2,R3b,R4a,R6 |
| focal_g2 | 7 | pool_nsa | 8 | 0.3109 | 0.2144 | 0.2553 | 7.93% | 0.4615 | 0.2435 | 1640 | +0.0891 | 0.295 ± 0.014 | FAIL | R1,R2,R3b,R4a,R6 |
| focal_g2 | 7 | pool_nsa | 64 | 0.3024 | 0.2113 | 0.2587 | 7.90% | 0.4647 | 0.2404 | 1554 | +0.0844 | 0.293 ± 0.014 | FAIL | R1,R2,R3b,R4a,R6 |
| pilot_quantile | 7 | pool_full | 1 | 0.3385 | 0.2285 | 0.3028 | 8.58% | 0.4952 | 0.4094 | 1710 | +0.0079 | 0.321 ± 0.015 | FAIL | R1,R2,R4a,R4b |
| pilot_quantile | 7 | pool_full | 8 | 0.2274 | 0.1721 | 0.3011 | 6.79% | 0.4648 | 0.3751 | 1269 | -0.0294 | 0.660 ± 0.015 | FAIL | R1 |
| pilot_quantile | 7 | pool_full | 64 | 0.2255 | 0.1696 | 0.3040 | 6.82% | 0.4830 | 0.3964 | 1205 | -0.0093 | 0.522 ± 0.016 | FAIL | R1,R4a,R4b |
| pilot_quantile | 7 | pool_nsa | 1 | 0.2626 | 0.1867 | 0.2554 | 7.03% | 0.3933 | 0.3138 | 1385 | +0.0221 | 0.305 ± 0.015 | FAIL | R1,R2,R3b,R6 |
| pilot_quantile | 7 | pool_nsa | 8 | 0.2122 | 0.1561 | 0.2586 | 6.25% | 0.3895 | 0.3518 | 1109 | -0.0096 | 0.887 ± 0.010 | INDETERMINATE | R1 |
| pilot_quantile | 7 | pool_nsa | 64 | 0.2104 | 0.1546 | 0.2585 | 6.18% | 0.3931 | 0.3493 | 1115 | -0.0075 | 0.852 ± 0.011 | FAIL | R1 |

## Appendix B: the C sweep, all 64 rows

`none` arm, NSA-only basis, each row at its own package-0.5 threshold.

| L | pool | C | pkg-0.5 | pkg-0.3 | brief | EER | FA@thr | f0 | f4 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | pool_full | 0.25 | 0.4286 | 0.2585 | 0.3073 | 9.57% | 1747 | 0.5483 | 0.3230 |
| 6 | pool_full | 1 | 0.3544 | 0.2287 | 0.3026 | 8.47% | 1660 | 0.5413 | 0.2907 |
| 6 | pool_full | 2 | 0.3266 | 0.2226 | 0.3198 | 8.34% | 1621 | 0.5412 | 0.3052 |
| 6 | pool_full | 4 | 0.2898 | 0.2027 | 0.3186 | 7.83% | 1582 | 0.5451 | 0.3036 |
| 6 | pool_full | 8 | 0.2805 | 0.1977 | 0.3197 | 7.76% | 1445 | 0.5472 | 0.3057 |
| 6 | pool_full | 16 | 0.2737 | 0.1955 | 0.3226 | 7.72% | 1349 | 0.5480 | 0.3060 |
| 6 | pool_full | 64 | 0.2697 | 0.1982 | 0.3228 | 7.74% | 1622 | 0.5487 | 0.3079 |
| 6 | pool_full | 256 | 0.2705 | 0.1988 | 0.3236 | 7.73% | 1620 | 0.5481 | 0.3079 |
| 6 | pool_nsa | 0.25 | 0.3443 | 0.2374 | 0.2679 | 8.87% | 1757 | 0.4813 | 0.2428 |
| 6 | pool_nsa | 1 | 0.2573 | 0.1959 | 0.2727 | 7.65% | 1509 | 0.4717 | 0.2677 |
| 6 | pool_nsa | 2 | 0.2344 | 0.1841 | 0.2727 | 7.20% | 1387 | 0.4695 | 0.2523 |
| 6 | pool_nsa | 4 | 0.2271 | 0.1775 | 0.2803 | 7.01% | 1275 | 0.4680 | 0.2796 |
| 6 | pool_nsa | 8 | 0.2253 | 0.1771 | 0.2805 | 7.04% | 1252 | 0.4662 | 0.2735 |
| 6 | pool_nsa | 16 | 0.2220 | 0.1760 | 0.2806 | 6.94% | 1262 | 0.4644 | 0.2673 |
| 6 | pool_nsa | 64 | 0.2197 | 0.1741 | 0.2804 | 6.88% | 1211 | 0.4638 | 0.2637 |
| 6 | pool_nsa | 256 | 0.2191 | 0.1736 | 0.2802 | 6.86% | 1238 | 0.4638 | 0.2631 |
| 7 | pool_full | 0.25 | 0.4360 | 0.2653 | 0.3000 | 9.96% | 1828 | 0.5446 | 0.4315 |
| 7 | pool_full | 1 | 0.3307 | 0.2280 | 0.3013 | 8.45% | 1642 | 0.4735 | 0.3950 |
| 7 | pool_full | 2 | 0.3061 | 0.2113 | 0.3029 | 7.96% | 1609 | 0.4640 | 0.3816 |
| 7 | pool_full | 4 | 0.2734 | 0.1971 | 0.3032 | 7.40% | 1314 | 0.4562 | 0.3722 |
| 7 | pool_full | 8 | 0.2568 | 0.1867 | 0.3050 | 7.11% | 1472 | 0.4553 | 0.3695 |
| 7 | pool_full | 16 | 0.2425 | 0.1780 | 0.3074 | 6.90% | 1372 | 0.4546 | 0.3666 |
| 7 | pool_full | 64 | 0.2348 | 0.1729 | 0.3077 | 6.70% | 1313 | 0.4546 | 0.3630 |
| 7 | pool_full | 256 | 0.2313 | 0.1710 | 0.3074 | 6.65% | 1285 | 0.4552 | 0.3624 |
| 7 | pool_nsa | 0.25 | 0.3241 | 0.2142 | 0.2500 | 7.95% | 1796 | 0.4304 | 0.3029 |
| 7 | pool_nsa | 1 | 0.2405 | 0.1766 | 0.2567 | 6.86% | 1258 | 0.4116 | 0.3394 |
| 7 | pool_nsa | 2 | 0.2379 | 0.1713 | 0.2586 | 6.75% | 1237 | 0.4048 | 0.3473 |
| 7 | pool_nsa | 4 | 0.2245 | 0.1639 | 0.2604 | 6.49% | 1111 | 0.4004 | 0.3589 |
| 7 | pool_nsa | 8 | 0.2219 | 0.1628 | 0.2652 | 6.43% | 1155 | 0.4002 | 0.3825 |
| 7 | pool_nsa | 16 | 0.2191 | 0.1603 | 0.2646 | 6.41% | 1080 | 0.3988 | 0.3843 |
| 7 | pool_nsa | 64 | 0.2179 | 0.1599 | 0.2646 | 6.37% | 1066 | 0.3968 | 0.3856 |
| 7 | pool_nsa | 256 | 0.2177 | 0.1592 | 0.2646 | 6.34% | 1060 | 0.3962 | 0.3856 |
| 8 | pool_full | 0.25 | 0.4825 | 0.2991 | 0.2988 | 11.05% | 2255 | 0.5836 | 0.5643 |
| 8 | pool_full | 1 | 0.4251 | 0.2769 | 0.3041 | 10.28% | 1983 | 0.5390 | 0.5027 |
| 8 | pool_full | 2 | 0.3986 | 0.2687 | 0.3065 | 10.05% | 1792 | 0.5370 | 0.5055 |
| 8 | pool_full | 4 | 0.3691 | 0.2541 | 0.3134 | 9.50% | 1727 | 0.5358 | 0.5071 |
| 8 | pool_full | 8 | 0.3285 | 0.2339 | 0.3235 | 8.96% | 1771 | 0.5393 | 0.5155 |
| 8 | pool_full | 16 | 0.3093 | 0.2254 | 0.3233 | 8.69% | 1619 | 0.5363 | 0.5161 |
| 8 | pool_full | 64 | 0.2968 | 0.2174 | 0.3239 | 8.52% | 1529 | 0.5356 | 0.5173 |
| 8 | pool_full | 256 | 0.2911 | 0.2156 | 0.3232 | 8.47% | 1483 | 0.5350 | 0.5173 |
| 8 | pool_nsa | 0.25 | 0.3491 | 0.2409 | 0.2779 | 9.04% | 1727 | 0.4409 | 0.4589 |
| 8 | pool_nsa | 1 | 0.2940 | 0.2067 | 0.2934 | 8.00% | 1486 | 0.4373 | 0.4558 |
| 8 | pool_nsa | 2 | 0.2596 | 0.1870 | 0.2883 | 7.43% | 1344 | 0.4351 | 0.4582 |
| 8 | pool_nsa | 4 | 0.2522 | 0.1802 | 0.2875 | 7.14% | 1357 | 0.4340 | 0.4555 |
| 8 | pool_nsa | 8 | 0.2577 | 0.1808 | 0.2869 | 7.20% | 1223 | 0.4330 | 0.4515 |
| 8 | pool_nsa | 16 | 0.2562 | 0.1800 | 0.2866 | 7.12% | 1247 | 0.4336 | 0.4521 |
| 8 | pool_nsa | 64 | 0.2550 | 0.1793 | 0.2867 | 7.06% | 1258 | 0.4331 | 0.4500 |
| 8 | pool_nsa | 256 | 0.2541 | 0.1792 | 0.2869 | 7.07% | 1230 | 0.4331 | 0.4500 |
| 20 | pool_full | 0.25 | 0.6779 | 0.3878 | 0.3306 | 14.93% | 3451 | 0.7205 | 0.9148 |
| 20 | pool_full | 1 | 0.6581 | 0.3843 | 0.3292 | 14.61% | 3013 | 0.6948 | 0.9054 |
| 20 | pool_full | 2 | 0.6423 | 0.3832 | 0.3280 | 14.42% | 3345 | 0.6910 | 0.9182 |
| 20 | pool_full | 4 | 0.6432 | 0.3940 | 0.3329 | 14.73% | 3645 | 0.6844 | 0.9078 |
| 20 | pool_full | 8 | 0.5884 | 0.3820 | 0.3320 | 14.25% | 3504 | 0.6847 | 0.9063 |
| 20 | pool_full | 16 | 0.5429 | 0.3680 | 0.3288 | 13.58% | 3195 | 0.6838 | 0.9060 |
| 20 | pool_full | 64 | 0.4675 | 0.3416 | 0.3247 | 12.79% | 2540 | 0.6852 | 0.9084 |
| 20 | pool_full | 256 | 0.4549 | 0.3358 | 0.3235 | 12.51% | 2466 | 0.6852 | 0.9085 |
| 20 | pool_nsa | 0.25 | 0.5959 | 0.3643 | 0.4182 | 13.50% | 2763 | 0.6143 | 0.8340 |
| 20 | pool_nsa | 1 | 0.5910 | 0.3671 | 0.4141 | 13.56% | 2414 | 0.6040 | 0.8612 |
| 20 | pool_nsa | 2 | 0.5504 | 0.3495 | 0.3991 | 13.05% | 2861 | 0.6041 | 0.8759 |
| 20 | pool_nsa | 4 | 0.4923 | 0.3303 | 0.3909 | 12.31% | 2400 | 0.6028 | 0.8680 |
| 20 | pool_nsa | 8 | 0.4290 | 0.2987 | 0.4004 | 11.28% | 2227 | 0.6001 | 0.8682 |
| 20 | pool_nsa | 16 | 0.4091 | 0.2897 | 0.4020 | 10.99% | 2122 | 0.5988 | 0.8633 |
| 20 | pool_nsa | 64 | 0.3920 | 0.2797 | 0.4034 | 10.68% | 1954 | 0.5973 | 0.8633 |
| 20 | pool_nsa | 256 | 0.3883 | 0.2761 | 0.4022 | 10.61% | 2042 | 0.5967 | 0.8639 |

## Appendix C: `pool_full` cells on both bases

The `none` cells of `pool_full` read on both bases. The full basis includes the 7,648 ASVspoof 2019 train/dev rows that the NSA basis excludes; `pool_nsa` x full basis does not exist and is N/A by declaration.

| L | C | basis | pkg-0.5 | pkg-0.3 | brief | EER | FA@thr |
|---|---|---|---|---|---|---|---|
| 6 | 1 | NSA-only 16,142 | 0.3544 | 0.2287 | 0.3026 | 8.47% | 1660 |
| 6 | 1 | full 23,790 | 0.3517 | 0.2267 | 0.2843 | 8.40% | 2628 |
| 6 | 8 | NSA-only 16,142 | 0.2805 | 0.1977 | 0.3197 | 7.76% | 1445 |
| 6 | 8 | full 23,790 | 0.3011 | 0.2075 | 0.3045 | 8.06% | 2310 |
| 6 | 64 | NSA-only 16,142 | 0.2697 | 0.1982 | 0.3228 | 7.74% | 1622 |
| 6 | 64 | full 23,790 | 0.2925 | 0.2090 | 0.3089 | 8.10% | 2539 |
| 7 | 1 | NSA-only 16,142 | 0.3307 | 0.2280 | 0.3013 | 8.45% | 1642 |
| 7 | 1 | full 23,790 | 0.3104 | 0.2082 | 0.2649 | 7.74% | 2348 |
| 7 | 8 | NSA-only 16,142 | 0.2568 | 0.1867 | 0.3050 | 7.11% | 1472 |
| 7 | 8 | full 23,790 | 0.2514 | 0.1770 | 0.2728 | 6.71% | 2150 |
| 7 | 64 | NSA-only 16,142 | 0.2348 | 0.1729 | 0.3077 | 6.70% | 1313 |
| 7 | 64 | full 23,790 | 0.2333 | 0.1651 | 0.2768 | 6.30% | 1880 |
| 8 | 1 | NSA-only 16,142 | 0.4251 | 0.2769 | 0.3041 | 10.28% | 1983 |
| 8 | 1 | full 23,790 | 0.3812 | 0.2482 | 0.2811 | 9.17% | 2743 |
| 8 | 8 | NSA-only 16,142 | 0.3285 | 0.2339 | 0.3235 | 8.96% | 1771 |
| 8 | 8 | full 23,790 | 0.3154 | 0.2195 | 0.3008 | 8.39% | 2536 |
| 8 | 64 | NSA-only 16,142 | 0.2968 | 0.2174 | 0.3239 | 8.52% | 1529 |
| 8 | 64 | full 23,790 | 0.2958 | 0.2098 | 0.3077 | 8.07% | 2548 |
| 20 | 1 | NSA-only 16,142 | 0.6581 | 0.3843 | 0.3292 | 14.61% | 3013 |
| 20 | 1 | full 23,790 | 0.5943 | 0.3392 | 0.2944 | 12.89% | 3860 |
| 20 | 8 | NSA-only 16,142 | 0.5884 | 0.3820 | 0.3320 | 14.25% | 3504 |
| 20 | 8 | full 23,790 | 0.5554 | 0.3460 | 0.3026 | 12.91% | 5322 |
| 20 | 64 | NSA-only 16,142 | 0.4675 | 0.3416 | 0.3247 | 12.79% | 2540 |
| 20 | 64 | full 23,790 | 0.4677 | 0.3188 | 0.3013 | 11.82% | 3881 |

## Appendix D: mined-set characterisation, every mined cell

Picks pooled over the five outer folds (unique rows in parentheses); shares are of the pooled picks; `–` means the arm mines no rows of that class. 102 cells mine rows (the five OK `focal_g2` cells mine nothing).

| arm | L | pool | C | real picks (unique) | top real group (share) | per-fold max | top real speaker (share) | real sources | fake picks (unique) | top fake group (share) | fake generators (top 3) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hardest_both | 6 | pool_full | 1 | 2652 (1447) | asvspoof2019:LA_0085 (0.067496) | 0.154135 | lj (0.226244) | asvspoof2019 0.39, librispeech 0.38, ljspeech 0.23 | 2104 (993) | gen:grad_tts (0.713878) | grad_tts 0.71, elevenlabs 0.14, asv19_A04 0.09 |
| hardest_both | 6 | pool_full | 8 | 2652 (1469) | asvspoof2019:LA_0085 (0.070136) | 0.156015 | lj (0.218326) | asvspoof2019 0.40, librispeech 0.38, ljspeech 0.22 | 2104 (1012) | gen:grad_tts (0.704848) | grad_tts 0.70, elevenlabs 0.14, asv19_A04 0.10 |
| hardest_both | 6 | pool_full | 64 | 2652 (1476) | asvspoof2019:LA_0085 (0.069759) | 0.156015 | lj (0.226998) | asvspoof2019 0.39, librispeech 0.38, ljspeech 0.23 | 2104 (1014) | gen:grad_tts (0.686787) | grad_tts 0.69, elevenlabs 0.13, asv19_A04 0.12 |
| hardest_both | 6 | pool_nsa | 1 | 1626 (814) | spk:libri_8555 (0.065806) | 0.138462 | lj (0.393604) | librispeech 0.61, ljspeech 0.39 | 1600 (791) | gen:grad_tts (0.804375) | grad_tts 0.80, elevenlabs 0.16, pro_diff 0.02 |
| hardest_both | 6 | pool_nsa | 8 | 1626 (826) | spk:libri_8555 (0.063961) | 0.138462 | lj (0.415129) | librispeech 0.58, ljspeech 0.42 | 1600 (796) | gen:grad_tts (0.803125) | grad_tts 0.80, elevenlabs 0.17, pro_diff 0.02 |
| hardest_both | 6 | pool_nsa | 64 | 1626 (820) | spk:libri_8555 (0.063346) | 0.138462 | lj (0.419434) | librispeech 0.58, ljspeech 0.42 | 1600 (798) | gen:grad_tts (0.803125) | grad_tts 0.80, elevenlabs 0.17, pro_diff 0.02 |
| hardest_both | 7 | pool_full | 1 | 2652 (1449) | spk:libri_652 (0.044495) | 0.105263 | lj (0.255656) | librispeech 0.39, asvspoof2019 0.35, ljspeech 0.26 | 2104 (983) | gen:grad_tts (0.703422) | grad_tts 0.70, elevenlabs 0.17, asv19_A06 0.07 |
| hardest_both | 7 | pool_full | 8 | 2652 (1471) | asvspoof2019:LA_0085 (0.042609) | 0.112782 | lj (0.231523) | librispeech 0.40, asvspoof2019 0.37, ljspeech 0.23 | 2104 (1011) | gen:grad_tts (0.682985) | grad_tts 0.68, elevenlabs 0.17, asv19_A06 0.08 |
| hardest_both | 7 | pool_full | 64 | 2652 (1473) | asvspoof2019:LA_0085 (0.041855) | 0.110902 | lj (0.230015) | librispeech 0.40, asvspoof2019 0.37, ljspeech 0.23 | 2104 (1013) | gen:grad_tts (0.675856) | grad_tts 0.68, elevenlabs 0.17, asv19_A06 0.08 |
| hardest_both | 7 | pool_nsa | 1 | 1626 (825) | spk:libri_6295 (0.068266) | 0.132308 | lj (0.371464) | librispeech 0.63, ljspeech 0.37 | 1600 (794) | gen:grad_tts (0.7875) | grad_tts 0.79, elevenlabs 0.19, pro_diff 0.02 |
| hardest_both | 7 | pool_nsa | 8 | 1626 (839) | spk:libri_6295 (0.064576) | 0.132308 | lj (0.378844) | librispeech 0.62, ljspeech 0.38 | 1600 (791) | gen:grad_tts (0.79) | grad_tts 0.79, elevenlabs 0.19, pro_diff 0.02 |
| hardest_both | 7 | pool_nsa | 64 | 1626 (841) | spk:libri_6295 (0.064576) | 0.132308 | lj (0.378844) | librispeech 0.62, ljspeech 0.38 | 1600 (789) | gen:grad_tts (0.79) | grad_tts 0.79, elevenlabs 0.19, pro_diff 0.02 |
| hardest_both | 8 | pool_full | 1 | 2652 (1441) | asvspoof2019:LA_0085 (0.044872) | 0.109023 | lj (0.311463) | librispeech 0.38, ljspeech 0.31, asvspoof2019 0.31 | 2104 (987) | gen:grad_tts (0.70865) | grad_tts 0.71, elevenlabs 0.16, asv19_A06 0.08 |
| hardest_both | 8 | pool_full | 8 | 2652 (1452) | asvspoof2019:LA_0101 (0.047511) | 0.103383 | lj (0.286199) | librispeech 0.38, asvspoof2019 0.33, ljspeech 0.29 | 2104 (996) | gen:grad_tts (0.695342) | grad_tts 0.70, elevenlabs 0.16, asv19_A06 0.08 |
| hardest_both | 8 | pool_full | 64 | 2652 (1470) | asvspoof2019:LA_0101 (0.048265) | 0.103383 | lj (0.285445) | librispeech 0.38, asvspoof2019 0.34, ljspeech 0.29 | 2104 (995) | gen:grad_tts (0.694867) | grad_tts 0.69, elevenlabs 0.17, asv19_A06 0.08 |
| hardest_both | 8 | pool_nsa | 1 | 1626 (856) | spk:libri_5142 (0.066421) | 0.138462 | lj (0.420049) | librispeech 0.58, ljspeech 0.42 | 1600 (804) | gen:grad_tts (0.7525) | grad_tts 0.75, elevenlabs 0.23, diffgan_tts 0.01 |
| hardest_both | 8 | pool_nsa | 8 | 1626 (854) | spk:libri_5142 (0.062116) | 0.123077 | lj (0.428659) | librispeech 0.57, ljspeech 0.43 | 1600 (812) | gen:grad_tts (0.760625) | grad_tts 0.76, elevenlabs 0.22, diffgan_tts 0.01 |
| hardest_both | 8 | pool_nsa | 64 | 1626 (865) | spk:libri_5142 (0.060886) | 0.12 | lj (0.428044) | librispeech 0.57, ljspeech 0.43 | 1600 (811) | gen:grad_tts (0.763125) | grad_tts 0.76, elevenlabs 0.22, diffgan_tts 0.01 |
| hardest_both | 20 | pool_full | 1 | 2652 (1446) | spk:libri_5142 (0.070513) | 0.139623 | lj (0.271493) | librispeech 0.40, asvspoof2019 0.33, ljspeech 0.27 | 2104 (1013) | gen:grad_tts (0.645437) | grad_tts 0.65, elevenlabs 0.25, asv19_A06 0.09 |
| hardest_both | 20 | pool_full | 8 | 2652 (1451) | spk:libri_5142 (0.067873) | 0.135849 | lj (0.266591) | librispeech 0.38, asvspoof2019 0.35, ljspeech 0.27 | 2104 (1030) | gen:grad_tts (0.626426) | grad_tts 0.63, elevenlabs 0.25, asv19_A06 0.10 |
| hardest_both | 20 | pool_full | 64 | 2652 (1460) | spk:libri_5142 (0.067873) | 0.135849 | lj (0.268477) | librispeech 0.38, asvspoof2019 0.36, ljspeech 0.27 | 2104 (1033) | gen:grad_tts (0.620247) | grad_tts 0.62, elevenlabs 0.25, asv19_A06 0.11 |
| hardest_both | 20 | pool_nsa | 1 | 1626 (854) | spk:libri_5142 (0.138376) | 0.221538 | lj (0.318573) | librispeech 0.68, ljspeech 0.32 | 1600 (805) | gen:grad_tts (0.678125) | grad_tts 0.68, elevenlabs 0.30, diffgan_tts 0.02 |
| hardest_both | 20 | pool_nsa | 8 | 1626 (846) | spk:libri_5142 (0.138991) | 0.221538 | lj (0.333333) | librispeech 0.67, ljspeech 0.33 | 1600 (815) | gen:grad_tts (0.663125) | grad_tts 0.66, elevenlabs 0.31, diffgan_tts 0.03 |
| hardest_both | 20 | pool_nsa | 64 | 1626 (844) | spk:libri_5142 (0.138376) | 0.221538 | lj (0.331488) | librispeech 0.67, ljspeech 0.33 | 1600 (818) | gen:grad_tts (0.66125) | grad_tts 0.66, elevenlabs 0.31, diffgan_tts 0.03 |
| easiest_both | 6 | pool_full | 1 | 2652 (1486) | asvspoof2019:LA_0108 (0.07994) | 0.166667 | lj (0.084087) | asvspoof2019 0.58, librispeech 0.33, ljspeech 0.08 | 2104 (951) | asvspoof2019:asv19_A05 (0.419202) | asv19_A05 0.42, asv19_A03 0.27, asv19_A02 0.18 |
| easiest_both | 6 | pool_full | 8 | 2652 (1503) | asvspoof2019:LA_0078 (0.081071) | 0.159091 | lj (0.090121) | asvspoof2019 0.55, librispeech 0.36, ljspeech 0.09 | 2104 (965) | asvspoof2019:asv19_A05 (0.433935) | asv19_A05 0.43, asv19_A03 0.27, asv19_A02 0.17 |
| easiest_both | 6 | pool_full | 64 | 2652 (1518) | asvspoof2019:LA_0078 (0.084465) | 0.172348 | LA_0078 (0.084465) | asvspoof2019 0.60, librispeech 0.34, ljspeech 0.07 | 2104 (1024) | asvspoof2019:asv19_A05 (0.410646) | asv19_A05 0.41, asv19_A03 0.25, asv19_A02 0.18 |
| easiest_both | 6 | pool_nsa | 1 | 1626 (906) | spk:libri_2277 (0.079336) | 0.153846 | lj (0.230627) | librispeech 0.77, ljspeech 0.23 | 1600 (802) | gen:your_tts (0.69625) | your_tts 0.70, openvoicev2 0.12, xtts_v2 0.08 |
| easiest_both | 6 | pool_nsa | 8 | 1626 (918) | spk:libri_2277 (0.078106) | 0.156923 | lj (0.221402) | librispeech 0.78, ljspeech 0.22 | 1600 (806) | gen:your_tts (0.69125) | your_tts 0.69, openvoicev2 0.11, xtts_v2 0.08 |
| easiest_both | 6 | pool_nsa | 64 | 1626 (921) | spk:libri_2277 (0.078106) | 0.156923 | lj (0.218942) | librispeech 0.78, ljspeech 0.22 | 1600 (806) | gen:your_tts (0.68625) | your_tts 0.69, openvoicev2 0.11, xtts_v2 0.09 |
| easiest_both | 7 | pool_full | 1 | 2652 (1476) | asvspoof2019:LA_0108 (0.082202) | 0.145833 | lj (0.106712) | asvspoof2019 0.58, librispeech 0.32, ljspeech 0.11 | 2104 (878) | asvspoof2019:asv19_A05 (0.415399) | asv19_A05 0.42, asv19_A03 0.28, asv19_A02 0.21 |
| easiest_both | 7 | pool_full | 8 | 2652 (1516) | asvspoof2019:LA_0108 (0.072775) | 0.145833 | lj (0.097662) | asvspoof2019 0.57, librispeech 0.34, ljspeech 0.10 | 2104 (886) | asvspoof2019:asv19_A05 (0.408745) | asv19_A05 0.41, asv19_A03 0.27, asv19_A02 0.22 |
| easiest_both | 7 | pool_full | 64 | 2652 (1517) | asvspoof2019:LA_0078 (0.072398) | 0.142045 | lj (0.096908) | asvspoof2019 0.57, librispeech 0.33, ljspeech 0.10 | 2104 (884) | asvspoof2019:asv19_A05 (0.407319) | asv19_A05 0.41, asv19_A03 0.27, asv19_A02 0.22 |
| easiest_both | 7 | pool_nsa | 1 | 1626 (866) | spk:libri_2277 (0.111316) | 0.153846 | lj (0.236777) | librispeech 0.76, ljspeech 0.24 | 1600 (759) | gen:your_tts (0.5975) | your_tts 0.60, openvoicev2 0.25, pro_diff 0.09 |
| easiest_both | 7 | pool_nsa | 8 | 1626 (870) | spk:libri_2277 (0.111316) | 0.153846 | lj (0.241697) | librispeech 0.76, ljspeech 0.24 | 1600 (768) | gen:your_tts (0.60375) | your_tts 0.60, openvoicev2 0.25, pro_diff 0.08 |
| easiest_both | 7 | pool_nsa | 64 | 1626 (874) | spk:libri_2277 (0.112546) | 0.156923 | lj (0.242927) | librispeech 0.76, ljspeech 0.24 | 1600 (766) | gen:your_tts (0.60375) | your_tts 0.60, openvoicev2 0.25, pro_diff 0.08 |
| easiest_both | 8 | pool_full | 1 | 2652 (1511) | asvspoof2019:LA_0108 (0.090875) | 0.143126 | lj (0.116893) | asvspoof2019 0.59, librispeech 0.29, ljspeech 0.12 | 2104 (830) | asvspoof2019:asv19_A05 (0.354087) | asv19_A05 0.35, asv19_A03 0.26, asv19_A02 0.24 |
| easiest_both | 8 | pool_full | 8 | 2652 (1511) | asvspoof2019:LA_0108 (0.083333) | 0.136364 | lj (0.111614) | asvspoof2019 0.59, librispeech 0.29, ljspeech 0.11 | 2104 (847) | asvspoof2019:asv19_A05 (0.367395) | asv19_A05 0.37, asv19_A03 0.28, asv19_A02 0.23 |
| easiest_both | 8 | pool_full | 64 | 2652 (1511) | asvspoof2019:LA_0108 (0.084465) | 0.141243 | lj (0.105958) | asvspoof2019 0.60, librispeech 0.29, ljspeech 0.11 | 2104 (858) | asvspoof2019:asv19_A05 (0.378327) | asv19_A05 0.38, asv19_A03 0.27, asv19_A02 0.23 |
| easiest_both | 8 | pool_nsa | 1 | 1626 (903) | spk:libri_2277 (0.094711) | 0.16 | lj (0.279828) | librispeech 0.72, ljspeech 0.28 | 1600 (775) | gen:your_tts (0.523125) | your_tts 0.52, openvoicev2 0.26, pro_diff 0.15 |
| easiest_both | 8 | pool_nsa | 8 | 1626 (910) | spk:libri_2277 (0.091021) | 0.153846 | lj (0.287823) | librispeech 0.71, ljspeech 0.29 | 1600 (774) | gen:your_tts (0.52125) | your_tts 0.52, openvoicev2 0.26, pro_diff 0.16 |
| easiest_both | 8 | pool_nsa | 64 | 1626 (912) | spk:libri_2277 (0.091636) | 0.156923 | lj (0.289668) | librispeech 0.71, ljspeech 0.29 | 1600 (780) | gen:your_tts (0.520625) | your_tts 0.52, openvoicev2 0.26, pro_diff 0.16 |
| easiest_both | 20 | pool_full | 1 | 2652 (1595) | spk:libri_2277 (0.066742) | 0.122411 | lj (0.160633) | asvspoof2019 0.45, librispeech 0.39, ljspeech 0.16 | 2104 (839) | asvspoof2019:asv19_A05 (0.390684) | asv19_A05 0.39, asv19_A02 0.22, asv19_A01 0.19 |
| easiest_both | 20 | pool_full | 8 | 2652 (1597) | spk:libri_3081 (0.067119) | 0.122411 | lj (0.152338) | asvspoof2019 0.44, librispeech 0.41, ljspeech 0.15 | 2104 (861) | asvspoof2019:asv19_A05 (0.364068) | asv19_A05 0.36, asv19_A02 0.23, asv19_A01 0.19 |
| easiest_both | 20 | pool_full | 64 | 2652 (1614) | spk:libri_3081 (0.066365) | 0.120527 | lj (0.156863) | asvspoof2019 0.44, librispeech 0.40, ljspeech 0.16 | 2104 (875) | asvspoof2019:asv19_A05 (0.361692) | asv19_A05 0.36, asv19_A02 0.23, asv19_A01 0.19 |
| easiest_both | 20 | pool_nsa | 1 | 1626 (954) | spk:libri_2277 (0.086716) | 0.172308 | lj (0.258303) | librispeech 0.74, ljspeech 0.26 | 1600 (936) | gen:your_tts (0.39375) | your_tts 0.39, pro_diff 0.29, xtts_v2 0.19 |
| easiest_both | 20 | pool_nsa | 8 | 1626 (961) | spk:libri_2277 (0.084871) | 0.166154 | lj (0.274908) | librispeech 0.73, ljspeech 0.27 | 1600 (971) | gen:your_tts (0.336875) | your_tts 0.34, pro_diff 0.31, xtts_v2 0.20 |
| easiest_both | 20 | pool_nsa | 64 | 1626 (968) | spk:libri_2277 (0.084871) | 0.166154 | lj (0.275523) | librispeech 0.72, ljspeech 0.28 | 1600 (965) | gen:your_tts (0.33875) | your_tts 0.34, pro_diff 0.29, xtts_v2 0.21 |
| random_both | 6 | pool_full | 1 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 6 | pool_full | 8 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 6 | pool_full | 64 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 6 | pool_nsa | 1 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 6 | pool_nsa | 8 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 6 | pool_nsa | 64 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 7 | pool_full | 1 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 7 | pool_full | 8 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 7 | pool_full | 64 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 7 | pool_nsa | 1 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 7 | pool_nsa | 8 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 7 | pool_nsa | 64 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 8 | pool_full | 1 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 8 | pool_full | 8 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 8 | pool_full | 64 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 8 | pool_nsa | 1 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 8 | pool_nsa | 8 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 8 | pool_nsa | 64 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 20 | pool_full | 1 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 20 | pool_full | 8 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 20 | pool_full | 64 | 2652 (2464) | lj:LJ028 (0.015837) | 0.026415 | lj (0.377074) | asvspoof2019 0.40, ljspeech 0.38, librispeech 0.23 | 2104 (1964) | gen:diffgan_tts (0.10076) | diffgan_tts 0.10, xtts_v2 0.10, your_tts 0.10 |
| random_both | 20 | pool_nsa | 1 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 20 | pool_nsa | 8 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| random_both | 20 | pool_nsa | 64 | 1626 (1492) | lj:LJ028 (0.03075) | 0.058462 | lj (0.627306) | ljspeech 0.63, librispeech 0.37 | 1600 (1479) | gen:your_tts (0.133125) | your_tts 0.13, pro_diff 0.13, xtts_v2 0.13 |
| hardest_reals | 7 | pool_full | 1 | 2652 (1449) | spk:libri_652 (0.044495) | 0.105263 | lj (0.255656) | librispeech 0.39, asvspoof2019 0.35, ljspeech 0.26 | – | – (–) |  |
| hardest_reals | 7 | pool_full | 8 | 2652 (1471) | asvspoof2019:LA_0085 (0.042609) | 0.112782 | lj (0.231523) | librispeech 0.40, asvspoof2019 0.37, ljspeech 0.23 | – | – (–) |  |
| hardest_reals | 7 | pool_full | 64 | 2652 (1473) | asvspoof2019:LA_0085 (0.041855) | 0.110902 | lj (0.230015) | librispeech 0.40, asvspoof2019 0.37, ljspeech 0.23 | – | – (–) |  |
| hardest_reals | 7 | pool_nsa | 1 | 1626 (825) | spk:libri_6295 (0.068266) | 0.132308 | lj (0.371464) | librispeech 0.63, ljspeech 0.37 | – | – (–) |  |
| hardest_reals | 7 | pool_nsa | 8 | 1626 (839) | spk:libri_6295 (0.064576) | 0.132308 | lj (0.378844) | librispeech 0.62, ljspeech 0.38 | – | – (–) |  |
| hardest_reals | 7 | pool_nsa | 64 | 1626 (841) | spk:libri_6295 (0.064576) | 0.132308 | lj (0.378844) | librispeech 0.62, ljspeech 0.38 | – | – (–) |  |
| hardest_fakes | 7 | pool_full | 1 | – | – (–) | – | – (–) |  | 2104 (983) | gen:grad_tts (0.703422) | grad_tts 0.70, elevenlabs 0.17, asv19_A06 0.07 |
| hardest_fakes | 7 | pool_full | 8 | – | – (–) | – | – (–) |  | 2104 (1011) | gen:grad_tts (0.682985) | grad_tts 0.68, elevenlabs 0.17, asv19_A06 0.08 |
| hardest_fakes | 7 | pool_full | 64 | – | – (–) | – | – (–) |  | 2104 (1013) | gen:grad_tts (0.675856) | grad_tts 0.68, elevenlabs 0.17, asv19_A06 0.08 |
| hardest_fakes | 7 | pool_nsa | 1 | – | – (–) | – | – (–) |  | 1600 (794) | gen:grad_tts (0.7875) | grad_tts 0.79, elevenlabs 0.19, pro_diff 0.02 |
| hardest_fakes | 7 | pool_nsa | 8 | – | – (–) | – | – (–) |  | 1600 (791) | gen:grad_tts (0.79) | grad_tts 0.79, elevenlabs 0.19, pro_diff 0.02 |
| hardest_fakes | 7 | pool_nsa | 64 | – | – (–) | – | – (–) |  | 1600 (789) | gen:grad_tts (0.79) | grad_tts 0.79, elevenlabs 0.19, pro_diff 0.02 |
| easiest_reals | 7 | pool_full | 1 | 2652 (1476) | asvspoof2019:LA_0108 (0.082202) | 0.145833 | lj (0.106712) | asvspoof2019 0.58, librispeech 0.32, ljspeech 0.11 | – | – (–) |  |
| easiest_reals | 7 | pool_full | 8 | 2652 (1516) | asvspoof2019:LA_0108 (0.072775) | 0.145833 | lj (0.097662) | asvspoof2019 0.57, librispeech 0.34, ljspeech 0.10 | – | – (–) |  |
| easiest_reals | 7 | pool_full | 64 | 2652 (1517) | asvspoof2019:LA_0078 (0.072398) | 0.142045 | lj (0.096908) | asvspoof2019 0.57, librispeech 0.33, ljspeech 0.10 | – | – (–) |  |
| easiest_reals | 7 | pool_nsa | 1 | 1626 (866) | spk:libri_2277 (0.111316) | 0.153846 | lj (0.236777) | librispeech 0.76, ljspeech 0.24 | – | – (–) |  |
| easiest_reals | 7 | pool_nsa | 8 | 1626 (870) | spk:libri_2277 (0.111316) | 0.153846 | lj (0.241697) | librispeech 0.76, ljspeech 0.24 | – | – (–) |  |
| easiest_reals | 7 | pool_nsa | 64 | 1626 (874) | spk:libri_2277 (0.112546) | 0.156923 | lj (0.242927) | librispeech 0.76, ljspeech 0.24 | – | – (–) |  |
| easiest_fakes | 7 | pool_full | 1 | – | – (–) | – | – (–) |  | 2104 (878) | asvspoof2019:asv19_A05 (0.415399) | asv19_A05 0.42, asv19_A03 0.28, asv19_A02 0.21 |
| easiest_fakes | 7 | pool_full | 8 | – | – (–) | – | – (–) |  | 2104 (886) | asvspoof2019:asv19_A05 (0.408745) | asv19_A05 0.41, asv19_A03 0.27, asv19_A02 0.22 |
| easiest_fakes | 7 | pool_full | 64 | – | – (–) | – | – (–) |  | 2104 (884) | asvspoof2019:asv19_A05 (0.407319) | asv19_A05 0.41, asv19_A03 0.27, asv19_A02 0.22 |
| easiest_fakes | 7 | pool_nsa | 1 | – | – (–) | – | – (–) |  | 1600 (759) | gen:your_tts (0.5975) | your_tts 0.60, openvoicev2 0.25, pro_diff 0.09 |
| easiest_fakes | 7 | pool_nsa | 8 | – | – (–) | – | – (–) |  | 1600 (768) | gen:your_tts (0.60375) | your_tts 0.60, openvoicev2 0.25, pro_diff 0.08 |
| easiest_fakes | 7 | pool_nsa | 64 | – | – (–) | – | – (–) |  | 1600 (766) | gen:your_tts (0.60375) | your_tts 0.60, openvoicev2 0.25, pro_diff 0.08 |
| pilot_quantile | 7 | pool_full | 1 | 1064 (628) | spk:libri_652 (0.075188) | 0.150235 | lj (0.161654) | librispeech 0.51, asvspoof2019 0.33, ljspeech 0.16 | 2104 (983) | gen:grad_tts (0.703422) | grad_tts 0.70, elevenlabs 0.17, asv19_A06 0.07 |
| pilot_quantile | 7 | pool_full | 8 | 1064 (646) | spk:libri_652 (0.069549) | 0.150235 | lj (0.152256) | librispeech 0.50, asvspoof2019 0.34, ljspeech 0.15 | 2104 (1011) | gen:grad_tts (0.682985) | grad_tts 0.68, elevenlabs 0.17, asv19_A06 0.08 |
| pilot_quantile | 7 | pool_full | 64 | 1064 (649) | spk:libri_652 (0.067669) | 0.140845 | lj (0.150376) | librispeech 0.50, asvspoof2019 0.35, ljspeech 0.15 | 2104 (1013) | gen:grad_tts (0.675856) | grad_tts 0.68, elevenlabs 0.17, asv19_A06 0.08 |
| pilot_quantile | 7 | pool_nsa | 1 | 655 (347) | spk:libri_5683 (0.100763) | 0.244275 | lj (0.244275) | librispeech 0.76, ljspeech 0.24 | 1600 (794) | gen:grad_tts (0.7875) | grad_tts 0.79, elevenlabs 0.19, pro_diff 0.02 |
| pilot_quantile | 7 | pool_nsa | 8 | 655 (356) | spk:libri_8555 (0.09771) | 0.267176 | lj (0.248855) | librispeech 0.75, ljspeech 0.25 | 1600 (791) | gen:grad_tts (0.79) | grad_tts 0.79, elevenlabs 0.19, pro_diff 0.02 |
| pilot_quantile | 7 | pool_nsa | 64 | 655 (357) | spk:libri_5683 (0.100763) | 0.267176 | lj (0.242748) | librispeech 0.76, ljspeech 0.24 | 1600 (789) | gen:grad_tts (0.79) | grad_tts 0.79, elevenlabs 0.19, pro_diff 0.02 |
