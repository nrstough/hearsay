# Run spec: post-draft gate sweep (`scripts/fuse_sweep_v3.py`)

**Date:** Sat Sep 26, 2026, ~17:55 EDT. **Owner:** Nathan (gate lane). **Branch:** `main`.
**Pre-declaration this implements:** `docs/reports/2026-09-26_post-draft-manifest.md` (committed `141b254`, 17:39, before any new candidate export existed). The manifest governs; where this spec and the manifest differ, the manifest wins.
**Plan:** `docs/reports/2026-09-26_post-draft-gate-sweep-plan.md`. **Handoff:** `docs/handoffs/2026-09-26_post-draft-gate-handoff.md` (revised 17:25).

## Problem

NSA scored the shipped rule (A3 w0.2 + E) at minDCF 0.0733. Two outside opinions call for a few bounded candidates, judged under a gate stricter than the standing rule and frozen before any number exists. The manifest froze four candidates:
- `T2`: a second M3 tier.
- `W4`: M5 weight 0.4, post hoc, judged by a fresh-evidence bake-off.
- `P_wl`: the WavLM Large probe as a fourth column.
- `H_noise`: handcrafted v6. The CPU lane will not run it (message ~17:42), so it is NOT RUN.

This change is the code that evaluates them mechanically and writes the locked table for Nathan's 19:30 decision.

## Solution

A new script, `scripts/fuse_sweep_v3.py`, is modeled on `scripts/fuse_sweep_m5.py` and reuses `scripts/fuse_sweep.py`'s loaders and cost functions. Every decision lives in pure, hermetically tested functions:

- **D1: shipped-file tripwire.** At start, the script asserts that the shipped TSV's sha256 starts `fb783076`, and aborts otherwise.
- **D2: row sets.** The base set is the per-split intersection of `m1b_v3`, `handcrafted_v5`, `spectra_aasist` and `m5_xlsr_ft`, as in the M5 sweep. Each candidate that adds a column is compared with CURRENT rebuilt on its own intersection, including rank references recomputed from that intersection's inner-OOF rows. Any shrink is recorded.
- **D3: missing export.** A missing export gives status `NOT RUN`. It is never an error and never a pass.
- **D4: direction guard.** A new column whose inner-OOF AUC is < 0.5 raises. This guards against the score-direction bug class.
- **D5: fusion.**
  - Ranks: `searchsorted(sorted inner-OOF logits, logit) / n`.
  - Weighted sum accumulated in manifest order.
  - E step: × 0.5 when M3 logit < −3 and base > 0.5.
  - T2: × 0.25 in total when M3 logit < −6 and base > 0.5.
- **D6: readouts.**
  - Costs: brief = `min_cost(y, s, 0.3)` and averse = `min_cost(y, s, 0.5, c_fa=1, c_miss=4)`, on inner OOF, holdout and In-the-Wild. Inner averse is new relative to the M5 sweep.
  - Holdout: argmin [#FA, #miss] under the brief cost, plus LJ Speech and LibriSpeech brief cost.
  - In-the-Wild: P_FA and P_miss at the inner-OOF brief threshold.
- **D7: test-set agreement.**
  - Spearman of the fused test scores against the shipped TSV's `cm-score`, matched by basename, over non-pinned rows (shipped < 0.001 is the pinned block).
  - The 0.5-crossing count uses each rule's own Platt map (class-balanced logistic on inner-OOF fused, plus the prior shift log(0.3/0.7)). It is a count only; no file list is printed or written.
- **D8: the gate, verbatim rules 2–7** as operationalized in the manifest.
  - Δ = CURRENT − candidate at 4 decimals.
  - Rule 2: ITW brief Δ ≥ 0.020 and ITW averse Δ ≥ −0.002.
  - Rule 3: inner brief Δ ≥ 0.010 and inner averse Δ ≥ 0.
  - Rule 4: holdout brief and averse Δ ≥ −0.010.
  - Rule 5: Δ > 0 on ≥ 2 of 3 populations under each cost, and the diagnostic passes.
  - Rule 6: Spearman < 0.974 doubles the thresholds in rules 2 and 3 (0.040 / 0.020).
  - Rule 7: with two or more passers, ship none unless one Pareto-dominates every other passer across the six cells (≤ in all, < in at least one).
- **D9: diagnostics.**
  - **T2:** zero spoof fires of the deeper tier (M3 < −6 and base > 0.5) on inner, holdout, ITW and all five perturbation kinds.
  - **P_wl:** two parts.
    1. At its own inner-OOF brief threshold, `wavlm_l` alone flags ≥ 16 of CURRENT's ITW misses (CURRENT's misses taken at CURRENT's inner-OOF brief threshold).
    2. The corrected share of CURRENT's errors is > 0.5 on inner OOF and on ITW. A false alarm counts as corrected if the new rank < M1b's rank; a miss, if the new rank > M1b's rank. Ties don't count.
  - **H_noise:** NOT RUN, so no diagnostic input.
- **D10: the W4 bake-off.** All three must hold.
  1. The standing rule: ITW brief Δ ≥ 0.01, ITW averse Δ ≥ −0.01, holdout brief Δ ≥ −0.05, inner brief Δ ≥ −0.03.
  2. Perturbation robustness on `outputs/channel/m3_perturb.csv`, using ranks from `models/fusion_v2/constants.json`: Δ ≥ −0.010 in all 10 kind × cost cells.
  3. A speaker-cluster paired bootstrap of the ITW gain: 2,000 replicates, `default_rng(0)`, the same draws for both rules. The 5th percentile of (CURRENT − W4) must be > 0 under each cost.
  - The perturbation tripwire: CURRENT recomputed on the cohort must equal that CSV's `fused` column to 1e-9. If it doesn't, the bake-off is invalid, not passed.
- **D11: self-check.** Before any candidate verdict, CURRENT on the base set must reproduce `outputs/fusion/sweep_m5_report.json` to 4 decimals:
  - inner 0.1351
  - holdout 0.0065 / 0.0087
  - ITW 0.228 / 0.2385
  - ITW P_FA 0.013 and P_miss 0.122 at the inner threshold
  - holdout argmin [0, 13]
  - Spearman against the shipped file ≥ 0.999

  On a failure, the script prints SELF-CHECK FAIL, writes the report with every verdict `INVALID`, and exits non-zero. T2 and W4 are also checked against their known numbers (T2 identical to CURRENT; W4 at 0.1373 / 0.3366, 0.0065 / 0.0088, 0.1997 / 0.2075). A mismatch there is reported, but it does not block the run.
- **D12: outputs.**
  - Always writes `outputs/fusion/sweep_v3_report.json`.
  - `--write` is refused unless `--ratified-by nathan --candidate NAME` is given and NAME's verdict is PASS (or, for W4, its bake-off passed).
  - It writes only `models/fusion_v3/constants.json`, in the `fusion_v2` shape plus the new column's model identity, and refuses if that file exists.
  - It never writes `models/fusion_v2/`, `submissions/` or the shipped TSV.
- **D13: import hygiene.** Importing the module never imports lightgbm or torch (the macOS OpenMP rule). `hearsay.pipeline` (for `M5_DIR`) is imported only inside the `--write` path; it does not import torch either.

## Acceptance criteria

- **A1.** `uv run pytest -q tests/test_fuse_sweep_v3.py` passes: the hermetic tests below, plus `needs_data` self-checks when data is present.
- **A2.** `uv run pytest -q` passes everything (510 before this change, with data), and `uv run ruff check .` is clean.
- **A3.** `uv run python scripts/fuse_sweep_v3.py` with no `wavlm_l.csv` behaves as follows:
  - the self-check passes and the perturbation tripwire holds;
  - T2's six cells are expected to be identical to CURRENT's (non-blocking check), with verdict FAIL (rule 2) and a zero spoof-fire diagnostic;
  - W4 reproduces its known numbers, its gate verdict is FAIL (rule 3), and its bake-off verdict is printed;
  - P_wl and H_noise are NOT RUN;
  - the report JSON is written.
- **A4.** The shipped TSV's and `submissions/CrossExam_predictions.tsv`'s sha256 are unchanged, and `models/fusion_v2/` is untouched (`git status` clean there).
- **A5.** Every gate rule, each diagnostic part and each bake-off part has a counterfactual test that fails when that rule is removed or inverted.

## Test inventory (failure mode → test)

| Stage | Failure mode | Test(s) |
|---|---|---|
| Costs | The sweep optimizes the sponsor's cost as if it were the brief's | `brief` equals `min_cost(y,s,0.3)`; `averse` equals `min_cost(y,s,0.5,c_fa=1,c_miss=4)`; single-FA and single-miss vectors recover the 9.33 : 1 and 1 : 4 weights; an asymmetric example where brief ≠ averse |
| Loading | A missing export crashes or passes | NOT RUN status from a tmp dir; the other candidates are still evaluated |
| | Inverted new column | AUC < 0.5 raises; the correct fixture doesn't |
| | A candidate covers fewer rows | Intersection used for both rules; shrink recorded; base CURRENT unchanged |
| Fusion | T2 tier arithmetic | Point cases (−7, 0.8) → 0.2; (−4, 0.8) → 0.4; (−7, 0.4) → 0.4; (−2, 0.8) → 0.8; (−3 exactly, 0.8) → 0.8 (strict <) |
| | Weights | Manifest weights sum to 1 and equal P_wl 0.4/0.2/0.2/0.2, W4 0.4/0.2/0.4, CURRENT 0.6/0.2/0.2 |
| | Rank reference | Hand-computed searchsorted fixture |
| Gate | Each rule vacuous | All-pass fixture → PASS; one fixture per sub-rule (2a, 2b, 3a, 3b, 4a, 4b, 5-brief, 5-averse, 5-diagnostic) → FAIL naming it |
| | Boundaries and rounding | Δ = 0.020 passes, 0.0199 fails; rounding precedes comparison; inner averse equal passes, 0.0001 worse fails |
| | Rule 6 | 0.9739 doubles; 0.974 does not; doubled thresholds applied to rules 2a and 3a only |
| | Rule 7 | Neither dominates → none; one dominates → it; all-equal → none; a single passer → it |
| Diagnostics | T2 fires counted wrong | Spoof at (−7, 0.8) counts; real rows and base ≤ 0.5 don't |
| | P_wl new catches | 15 → fail, 16 → pass; misses taken from CURRENT's threshold, catches at the column's own |
| | P_wl corrective share | Exactly 0.5 fails; 0.6 passes; ties don't count; must hold on both populations |
| Bake-off | Bootstrap not paired or not clustered | Same seed → identical output; a speaker's rows move together (single-speaker fixture); missing speaker raises |
| | Parts vacuous | Standing-rule fail, perturbation −0.0101, 5th percentile ≤ 0 → each FAIL; all pass → PASS |
| Test agreement | Pinned rows included, or basename mismatch | Pinned rows excluded; an unmatched filename raises |
| Outputs | `--write` without ratification, or overwrite | Refused without `--ratified-by nathan`; refused for a non-passing candidate; refused if `fusion_v3/constants.json` exists; target path is fusion_v3 |
| | Shipped TSV changed | A hash mismatch aborts |
| Hygiene | LightGBM or torch in the same process | Module import leaves `lightgbm` and `torch` out of `sys.modules` (run in a subprocess) |
| Self-check (`needs_data`) | Drift from the shipped numbers | CURRENT = the expected dict; T2 = CURRENT; W4 = the known numbers; Spearman ≥ 0.999; perturbation tripwire ≤ 1e-9 |

## Documentation

- **This run spec**, frozen after commit.
- **`docs/reports/2026-09-26_post-draft-sweep.md`:** the locked table, verdicts and packet, written after the WavLM export lands (~19:15).
- **`CLAUDE.md`:** one AI-use disclosure bullet for the gate lane.
- **Conditional:** a `submissions/log.csv` note row after Nathan's 19:30 ruling. `docs/STATUS.md` belongs to the oversight chat, which is told the outcome instead of this lane editing it.

## Pre-data clarifications and review resolutions (18:25)

The manifest clarifications at `b877760` (P_wl misses at the inner-OOF threshold = 122, absolute bar 16; empty pinned block; INVALID path; rule-7 common rows; H_noise diagnostic fails without evidence; bake-off rounding) and the plan's "Review resolutions" table are part of this spec. Added design points:
- **D14.** Export validation happens before `fs.load`'s dedup, and invalid or incomplete candidates are INVALID per candidate.
- **D15.** `--write` is bound to the locked report's input hashes and decision.
- **D16.** Close-out: 19:25 lock, 20:00 ruling cutoff, a KEEP log row or a ratified integration job with a 21:30 lapse.

## Out of scope

- Runner integration of a ratified candidate (handoff step 6; a separate job).
- Any change to `fusion_v2`, the shipped TSV or `hearsay.pipeline`.
- Any candidate beyond the manifest's four.

## Results (execution, 18:40–18:45; before the WavLM export)

- **Tests:**
  - `uv run pytest -q tests/test_fuse_sweep_v3.py`: 71 passed, including 4 `needs_data`.
  - `uv run pytest -q`: 722 passed (the "510" baseline was stale; other lanes have added tests since).
  - `uv run ruff check .`: clean.
- **A4, immutability:** sha256 of `models/fusion_v2/constants.json` (`0818e83e…`), the shipped TSV and `submissions/CrossExam_predictions.tsv` (both `fb783076…`) are identical before and after. No `models/fusion_v3/` exists.
- **A3, `uv run python scripts/fuse_sweep_v3.py`, no `wavlm_l.csv`:**
  - Self-check ok on every field (inner 0.1351, holdout 0.0065 / 0.0087, ITW 0.228 / 0.2385, P_FA / P_miss 0.013 / 0.122, argmin [0, 13]). Spearman vs shipped is 1.000000 on all 1,671 rows (the pinned block is empty). Inner averse (new) is 0.2997.
  - **T2:** six cells identical to CURRENT; test Spearman 0.9857; 0 crossings. Deeper tier fires on 0 spoof rows in every split and perturbation kind (bona fide: inner 133, holdout 34, ITW 2, perturbation 0–5). Gate **FAIL** (2a, 3a, 5-brief, 5-averse: no improvement), as pre-declared.
  - **W4:** known numbers reproduced exactly; test Spearman 0.9702 (doubling triggered); 4 crossings. Gate FAIL (2a/3a doubled, 3b, 5). **Bake-off FAIL:**
    - Part 1 (standing rule): passes.
    - Part 2 (perturbation) fails. Δ (CURRENT − W4) per kind, brief / averse: none −0.012 / 0.000; mp3 −0.004 / −0.004; noise20 +0.051 / +0.020; speed −0.004 / −0.012; shift1 −0.016 / −0.004. Three cells are below −0.010. Tripwire 1.1e-16.
    - Part 3 (ITW bootstrap) fails: 54-speaker cluster bootstrap, 2,000 replicates, 5th percentile brief +0.0044, averse −0.0023.
    - W4 is a measured negative (post hoc, did not survive fresh evidence) and is not ratifiable.
  - **P_wl:** NOT RUN pending `wavlm_l.csv`. **H_noise:** NOT RUN (withdrawn).
  - Decision so far: KEEP.
- **Deviations from the plan:**
  - W4's gate record uses `diag_ok = True` ("no diagnostic under the gate"), so its gate failures list only the metric rules. W4 is governed by the bake-off alone.
  - `ruff format` was not applied: the repo's scripts carry `# fmt: skip` markers and CI runs `ruff check` only.

## Post-commit audit, round 1 (Claude pre-audit, 19:00; Overall Fail on test coverage)

Every finding below was fixed in the follow-up commit.
1. **Exceptions could escape.** `run_candidates` now catches any exception per candidate as INVALID, and the rule-7 re-score falls back to KEEP with `rule7_error`. `load_raw` refuses exports that lack `path`, `split` or `logit`, and a declared ITW file that is absent makes the candidate INVALID.
2. **H_noise came out INVALID instead of FAIL.** A candidate's context is now the union of CURRENT's columns and the new one, so CURRENT is rebuilt on identical rows (D2).
3. **The report was not really locked.** Every run writes `sweep_v3_report.json` plus a timestamped archive copy and prints its sha256. `--write` requires `--report-sha256` equal to the locked report's hash (recorded in the sweep doc), and `precheck_write` makes every cheap refusal before evaluation.
4. **The separate ITW file was not picked up.** `resolve_itw` now uses `_itw_<col>.csv` for a new column when that file is present.
5. **Missing tests.** 15 tests were added on a hermetic synthetic world:
   - NOT RUN with the other candidates still evaluated;
   - the direction guard, and the catch wiring against both thresholds;
   - a broken export becoming INVALID without ending the run;
   - H_noise FAIL, not INVALID;
   - a shrunk row set rebuilding its rank references;
   - the separate ITW file;
   - rule 7 on common rows, and its KEEP fallback;
   - the T2 diagnostic counterfactual;
   - `pwl_diag_ok` composition;
   - the ITW brief side of the room rule;
   - an unmatched test name;
   - `clean()`;
   - the `precheck_write` and `do_write` refusals;
   - `main` exiting 2 on a self-check failure.
6. **Nits:** the report is JSON-safe (NaN written as null, `allow_nan=False`); W4's gate record carries `governing: false`; the `fusion_v3` payload carries a `status`; `--new-column-model` is refused for candidates without a new column.

The re-run verdicts are unchanged: T2 FAIL, W4 BAKEOFF FAIL, P_wl and H_noise NOT RUN, KEEP. Report sha256 `07090750…` (19:06, pre-WavLM). `tests/test_fuse_sweep_v3.py`: 86 passed.
