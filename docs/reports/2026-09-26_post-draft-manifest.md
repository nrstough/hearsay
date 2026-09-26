# Post-draft candidate manifest and gate (pre-declared Sat Sep 26, 2026, ~17:45 EDT)

**Status:** frozen at commit. Approved by Nathan in the gate chat (P1 "freeze" ~17:30; W4 ruling ~17:40). Nothing below is changed after any candidate's new number is seen. Results go in `docs/reports/2026-09-26_post-draft-sweep.md`; the sweep script is `scripts/fuse_sweep_v3.py`, report `outputs/fusion/sweep_v3_report.json`.

**State at commit:**
- Shipped TSV `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv`, sha256 `fb7830762691d04d8be938f8e2615e349c998cdfd72c58e94d02385d57368607`. NSA draft review: minDCF 0.0733, EER 3.534%.
- "Sleep equals ship": a byte copy at `submissions/CrossExam_predictions.tsv` with the same sha256, logged 17:33:52 as FINAL unless replaced by a ratified candidate.
- No new candidate export exists at commit time: `outputs/detector_scores/wavlm_l.csv` and `handcrafted_v6.csv` are absent. The one exception is W4, whose numbers were seen first (post hoc, see below).
- Sources: gate verbatim from `docs/consults/2026-09-26_post-draft-review_RESPONSE.md` ("What to do (adopted)"); candidate list from the same file's "Round 2" and `docs/handoffs/2026-09-26_post-draft-gate-handoff.md` (revised 17:25).

## The candidates (four; the cap; never combined; no fifth after results)

All ranks are `searchsorted(sorted inner-OOF logits of that column, logit) / n`, exactly as `scripts/fuse_sweep_m5.py` and `hearsay.pipeline` compute them. The E step is the shipped one: if M3 logit < −3 and base > 0.5, base × 0.5. CURRENT is the shipped rule, A3 w0.2 + E = `0.6·rank(m1b_v3) + 0.2·rank(handcrafted_v5) + 0.2·rank(m5_xlsr_ft)`, then E. The Platt map (for the 0.5-crossing count only) is the M5 sweep's: class-balanced logistic on inner-OOF fused scores, plus the prior shift log(0.3/0.7).

| ID | Definition | Source | Governing test |
|---|---|---|---|
| `T2` | CURRENT with a second M3 tier: × 0.5 when M3 logit < −3 and base > 0.5 (as now), × 0.25 in total when M3 logit < −6 and base > 0.5. Nothing fitted. | shipped exports | stricter gate |
| `W4` | A3 (0.4, 0.2, 0.4) + E: `0.4·rank(m1b_v3) + 0.2·rank(handcrafted_v5) + 0.4·rank(m5_xlsr_ft)`, then E. **Post hoc** (disclosure below). | shipped exports | the W4 bake-off (below) |
| `P_wl` | `0.4·rank(m1b_v3) + 0.2·rank(wavlm_l) + 0.2·rank(handcrafted_v5) + 0.2·rank(m5_xlsr_ft)`, then E. The refit-seat variant is not run. | `outputs/detector_scores/wavlm_l.csv` (heads chat, due 18:50) | stricter gate |
| `H_noise` | CURRENT with `handcrafted_v6` in the handcrafted seat. | `outputs/detector_scores/handcrafted_v6.csv` and `_itw_handcrafted_v6.csv` (CPU chat, only if idle) | stricter gate |

A missing export marks its candidate **not run**; it is never an error and never a pass. `P_l59` (XLS-R layers 5–9) was dropped at 17:20: the Fable agent measured it as a negative (Round 2, item 1), before this manifest.

**W4 post-hoc disclosure.** W4's proxy numbers were computed by the Fable agent and re-verified by the consult chat before this manifest existed. They are inner 0.1373 / 0.3366, holdout 0.0065 / 0.0088, In-the-Wild 0.1997 / 0.2075 (brief / averse), and test Spearman 0.970 against the shipped file. It passes the standing four-condition rule. Under the stricter gate it fails rule 3: inner is worse by 0.002 (brief) and 0.037 (averse). Because those numbers are already seen, they cannot decide anything, so Nathan chose a fresh-evidence bake-off (next section) over either rulebook alone.

## The gate (verbatim from the RESPONSE record; governs T2, P_wl, H_noise)

Supersedes the standing rule (ITW ≥ 0.01 better, averse ≤ 0.01 worse, holdout ≤ 0.05 worse, inner ≤ 0.03 worse) for these three candidates tonight.

1. Each candidate's definition, weights and preprocessing go in a manifest before its proxy table exists; candidates are never combined tonight.
2. In-the-Wild brief-cost improvement ≥ 0.020; sponsor-cost regression ≤ 0.002.
3. Inner-OOF brief-cost improvement ≥ 0.010; no sponsor-cost regression at printed precision.
4. Holdout degradation ≤ 0.010 under either convention (a safety constraint, not evidence).
5. Both costs must improve on at least two of the three proxy populations, and the candidate's mechanism diagnostic must pass (noise twins must recover the 20 dB lane; a new probe must correct baseline errors rather than clone M1b's ranks).
6. If the candidate's test-ranking Spearman against the shipped file falls below 0.974, the required gains double; 0.5-crossings are counted for disclosure only.
7. If two candidates pass, ship neither unless one Pareto-dominates across all six split × cost cells. Nathan ratifies exactly one candidate from one consolidated packet; the runner must then reproduce the exact ordering and pinned block. Nothing auto-ships.

### How the gate is computed (operationalization, approved at P1)

- **Costs.** Brief = `min_cost(y, s, 0.3)` with C_FA 4, C_miss 1 (normalized 9.33·P_FA + P_miss). Averse, the sponsor code's convention = `min_cost(y, s, 0.5, c_fa=1, c_miss=4)` (P_FA + 4·P_miss). A test pins both.
- **Populations.** Inner = the inner-OOF rows (16,142). Holdout = the outer holdout (3,858). In-the-Wild = the 3,000-row stress set. Six cells = 3 populations × 2 costs.
- **Δ.** Δ = CURRENT − candidate, both rounded to 4 decimals. "Improve" means Δ > 0; "regression" means Δ < 0.
- **Rule 2.** ITW brief Δ ≥ 0.020, and ITW averse Δ ≥ −0.002.
- **Rule 3.** Inner brief Δ ≥ 0.010, and inner averse Δ ≥ 0.
- **Rule 4.** Holdout brief Δ ≥ −0.010, and holdout averse Δ ≥ −0.010.
- **Rule 5.** Under the brief cost, Δ > 0 on at least 2 of the 3 populations; the same under the averse cost; and the candidate's diagnostic (below) passes.
- **Rule 6.** Spearman between the candidate's fused test scores and the shipped TSV's `cm-score`, over the test rows outside the pinned block (shipped `cm-score` < 0.001 marks the pinned block). If it is below 0.974, the rule-2 threshold becomes 0.040 and the rule-3 threshold becomes 0.020; nothing else changes.
- **0.5 crossings.** The count of non-pinned test files whose candidate probability and shipped probability fall on opposite sides of 0.5. Disclosure only; **no one inspects which files cross** (selection on the test set).
- **Row sets.** Each candidate is compared with CURRENT rebuilt on the identical row intersection (every column the candidate uses, plus M3). Any shrink from the base set is disclosed with the row counts. The self-check runs on the base set.

### Mechanism diagnostics (pre-declared)

- **T2.** The deeper tier (M3 logit < −6 and base > 0.5) must fire on **zero** spoof rows in every proxy split (inner, holdout, In-the-Wild) and in every kind of the channel lane's perturbation cohort (`outputs/channel/m3_perturb.csv`, using its `fused_base`). Expected overall outcome: readouts identical to CURRENT (verified inert in Round 2). That fails rule 2 (no improvement), so T2's verdict is KEEP either way; it is run so the record shows it.
- **P_wl.** Two parts; both must pass.
  1. *New catches.* Take CURRENT's In-the-Wild misses at CURRENT's inner-OOF brief-cost threshold (158 in Round 2). The `wavlm_l` column alone, thresholded at its own inner-OOF brief-cost threshold, must flag at least **16** of them (10%). For scale: M1b flags 1, M5 33, M3 107.
  2. *Corrective, not a clone.* Take CURRENT's errors at the same threshold on inner OOF and, separately, on In-the-Wild. On a false alarm, the error counts as corrected if `rank(wavlm_l)` < `rank(m1b_v3)`; on a miss, if `rank(wavlm_l)` > `rank(m1b_v3)`. Ties do not count. The corrected share must be **> 0.5** on both populations. The Spearman of `wavlm_l` against `m1b_v3` on holdout and test is reported beside it.
- **H_noise.** The CPU chat's pre-declared bar (its handoff, step 5), read from its report because the check needs audio: under 20 dB white noise on the channel lane's 500-clip cohort, `handcrafted_v6` AUC ≥ 0.90, and clean AUC within 0.005 of v5's 0.998. If those numbers are not reported, the diagnostic fails and H_noise fails rule 5.
- **W4.** None under the gate; its bake-off is below.

## The W4 bake-off (fresh evidence only; pre-declared before any of it is computed)

W4 enters the packet as ratifiable only if **all three** hold:

1. **Standing rule** (already known to pass): ITW brief Δ ≥ 0.01; ITW averse Δ ≥ −0.01; holdout brief Δ ≥ −0.05; inner brief Δ ≥ −0.03.
2. **Perturbation robustness.** This evidence is unseen for W4. The cohort is the channel lane's 500 holdout clips (250 bona fide, 250 spoof) under each of five kinds: none, MP3 64 kbps, 20 dB white noise, ±2% speed, one-sample shift. Per-row logits come from `outputs/channel/m3_perturb.csv`, and ranks use the shipped inner-OOF references (`models/fusion_v2/constants.json`). On every kind, W4's brief minDCF and averse minDCF must each be no more than 0.010 worse than CURRENT's on the same rows: 10 cells, all with Δ ≥ −0.010.
3. **The ITW gain is not noise.** Resample In-the-Wild **by speaker** (the manifest's `speaker` column, since clips of one speaker are not independent): 2,000 cluster-bootstrap replicates, `numpy.random.default_rng(0)`, the same resample for both rules. The 5th percentile of (CURRENT − W4) must be **> 0** under the brief cost and, separately, under the averse cost.

If W4 fails any part, it is recorded as a measured negative (post hoc, did not survive fresh evidence) and is not in the ratifiable set. If W4 passes, it counts as a "pass" for rule 7: if another candidate also passes, neither ships unless one Pareto-dominates the other across the six split × cost cells (≤ in all six and < in at least one, at 4 decimals).

## Readouts per candidate (all go in the locked table)

- Inner, holdout and In-the-Wild minDCF under both costs (the six cells).
- Holdout argmin [#FA, #miss] under the brief cost.
- In-the-Wild P_FA and P_miss at the candidate's inner-OOF brief-cost threshold.
- Holdout LJ Speech and LibriSpeech brief cost (as in the M5 sweep).
- Test Spearman against the shipped file (non-pinned rows), and the 0.5-crossing count.
- The mechanism diagnostic, or for W4 the bake-off.
- Gate verdict per rule, and the overall verdict: PASS / FAIL / NOT RUN.

## Self-check (must hold before any candidate is read)

With no new exports present, CURRENT reproduces `outputs/fusion/sweep_m5_report.json`'s shipped row to four decimals:
- inner 0.1351
- holdout 0.0065 / 0.0087
- In-the-Wild 0.228 / 0.2385
- ITW P_FA / P_miss at the inner threshold 0.013 / 0.122
- holdout argmin [0, 13]

CURRENT's Spearman against the shipped TSV is ≥ 0.999 on non-pinned rows. W4 reproduces the consult chat's numbers above. If the self-check fails, no candidate is read until it is fixed.

## Decision rule (the 19:30 packet)

- **No candidate passes:** KEEP. `submissions/CrossExam_predictions.tsv` (sha256 `fb783076…`) is final. Record the negatives in the sweep report and a log note row, then stop.
- **Exactly one passes:** recommend it to Nathan, with its integration risk stated.
  - P_wl needs a second backbone in the runner and does not ship unless it is reproduced end to end by 21:30.
  - W4 needs the full 1,671-row parity re-run: Spearman ≥ 0.999, max |Δp| ≤ 1e-3, exact sorted IDs for near-ties.
  - H_noise is a bundle swap.
- **Two or more pass:** ship none unless one Pareto-dominates every other passer across the six cells.
- **In every case:** Nathan ratifies or not in one 30-minute window (19:30–20:00). No weight or threshold is changed after the table is seen. Nothing auto-ships.
