# Plan: post-draft gate sweep (`scripts/fuse_sweep_v3.py`)

Run spec: `docs/specs/2026-09-26_post-draft-gate-sweep.md` (design decisions D1–D13, acceptance criteria A1–A5). Manifest: `docs/reports/2026-09-26_post-draft-manifest.md` (`141b254`). Mode: standard, with one Claude critique of this plan beside the Codex plan review.

## Ground truth used (read 17:20–17:55)

- **`scripts/fuse_sweep.py`:**
  - Lines 26–51: `REPO`, `S`, `PI = 0.3`, `norm`, `load(name, itw)`. `load` maps split `stress` → `itw`, concatenates an ITW file of `path,logit`, dedups by path and indexes by the normalized path.
  - `brief(y,s) = min_cost(y,s,PI)` and `averse(y,s) = min_cost(y,s,0.5,c_fa=1.0,c_miss=4.0)`.
  - It imports only numpy, pandas, sklearn and `hearsay.metrics`: no torch, no lightgbm.
- **`scripts/fuse_sweep_m5.py`:**
  - Lines 45–64: loaders, labels (`splits/nsa_folds.csv` plus `outputs/manifests/itw_stress.csv`), and the per-split intersection over m1, hc, m3, m5.
  - Lines 66–76: `rank`, `lg`, `e_step`.
  - Lines 93–110: `readout`.
  - Lines 121–157: Platt and the `--write` constants shape.
  - The module-level `fs` is loaded via importlib.
- **`src/hearsay/metrics.py:54`:** `min_cost(y, s, pi_synth, c_fa, c_miss)` over `roc_curve`, normalized by the default cost; `cost_at` at line 45.
- **`models/fusion_v2/constants.json`:**
  - `rank_ref_inner_oof_sorted` for m1b_v3, handcrafted_v5, m5_xlsr_ft (16,142 each).
  - `weights` 0.6 / 0.2 / 0.2; `e_rule` (−3, 0.5, ×0.5).
  - `platt` a = 17.6928, b = −8.1515, prior_shift = log(0.3/0.7).
  - `m5_checkpoint` hashes.
- **`outputs/channel/m3_perturb.csv`:**
  - Columns `key,path,kind,label,m1b_v3,spectra_aasist,m5_xlsr_ft,handcrafted_v5,fused_base,fused,e_applied`, all logits.
  - `fused` = `FusionConstants.fuse(...).fused` = the post-E rank-domain score (`scripts/m3_probes.py:205–217`); `fused_base` is the pre-E base.
  - 250 bona fide + 250 spoof per kind; kinds none, mp3, noise20, speed, shift1.
- **`outputs/manifests/itw_stress.csv`:** `path,label,generator,speaker,source`, which provides the speaker column for the cluster bootstrap.
- **The shipped TSV:** `filename<TAB>cm-score`, basenames like `HGT1013455.wav`, 1,671 rows; pinned rows have cm-score < 0.001.
- **Test conventions:** load scripts via `importlib.util.spec_from_file_location` (as in `tests/test_channel_robustness.py:19–23`). The `needs_data` marker is declared in `pyproject.toml:44`.

## Steps

### 1. `scripts/fuse_sweep_v3.py`: module constants and pure functions (D1–D10, D13)

Module header docstring: purpose, manifest pointer, usage (`uv run python scripts/fuse_sweep_v3.py [--write --ratified-by nathan --candidate NAME [--new-column-model DIR]]`).

Imports at top: `argparse, hashlib, importlib.util, json, math, sys, pathlib`, `numpy`, `pandas`, `scipy.stats.spearmanr`, `sklearn.metrics.roc_auc_score`, `hearsay.metrics.min_cost, cost_at, sigmoid`. Load `fs` from `scripts/fuse_sweep.py` exactly as `fuse_sweep_m5.py:30–32` does. There are no torch or lightgbm imports at module level.

Constants:
```python
SHIPPED_TSV = REPO / "submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv"
SHIPPED_SHA_PREFIX = "fb783076"
PINNED_BELOW = 0.001
BASE_COLS = ("m1b_v3", "handcrafted_v5", "m5_xlsr_ft")   # ranked; spectra_aasist is the E-step input only
CURRENT_W = {"m1b_v3": 0.6, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.2}
E_TIERS = ((-3.0, 0.5),)   # defined before CANDIDATES (Codex 1 / critique 6)
CANDIDATES = {   # manifest order; weights accumulated in dict order
  "T2":      {"weights": CURRENT_W, "tiers": ((-3.0, 0.5), (-6.0, 0.25)), "new": None, "test": "gate"},
  "W4":      {"weights": {"m1b_v3": 0.4, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.4}, "tiers": E_TIERS, "new": None, "test": "bakeoff"},
  "P_wl":    {"weights": {"m1b_v3": 0.4, "wavlm_l": 0.2, "handcrafted_v5": 0.2, "m5_xlsr_ft": 0.2}, "tiers": E_TIERS, "new": ("wavlm_l", None), "test": "gate"},
  "H_noise": {"weights": {"m1b_v3": 0.6, "handcrafted_v6": 0.2, "m5_xlsr_ft": 0.2}, "tiers": E_TIERS, "new": ("handcrafted_v6", "_itw_handcrafted_v6.csv"), "test": "gate"},
}
EXPECTED_CURRENT = {"inner_brief": 0.1351, "holdout_brief": 0.0065, "holdout_averse": 0.0087, "itw_brief": 0.228,
                    "itw_averse": 0.2385, "itw_pfa_at_inner_thr": 0.013, "itw_pmiss_at_inner_thr": 0.122,
                    "holdout_argmin_FA_miss": [0, 13]}
EXPECTED_W4 = {"inner_brief": 0.1373, "inner_averse": 0.3366, "holdout_brief": 0.0065, "holdout_averse": 0.0088,
               "itw_brief": 0.1997, "itw_averse": 0.2075}
GATE = {"itw_brief_gain": 0.020, "itw_averse_reg": 0.002, "inner_brief_gain": 0.010, "holdout_reg": 0.010,
        "spearman_floor": 0.974}
STANDING = {"itw_brief_gain": 0.01, "itw_averse_reg": 0.01, "holdout_brief_reg": 0.05, "inner_brief_reg": 0.03}
PWL_MIN_CATCHES, CORRECTIVE_MIN = 16, 0.5
BOOT_N, BOOT_SEED, PERTURB_REG = 2000, 0, 0.010
CELLS = [(p, c) for p in ("inner", "holdout", "itw") for c in ("brief", "averse")]
```

Pure functions (all hermetically testable):
- `sha_ok(path) -> bool`.
- `brief(y, s)` / `averse(y, s)`: thin wrappers equal to `min_cost(y,s,0.3)` / `min_cost(y,s,0.5,c_fa=1.0,c_miss=4.0)`. They do not delegate to `fs`, so the test pins the literal arguments.
- `rank_vs(ref_sorted, x) = np.searchsorted(ref_sorted, x) / len(ref_sorted)`.
- `blend(ranks: dict, weights: dict)`: `b = 0.0; for k, w in weights.items(): b = b + w * ranks[k]`. This is the order `hearsay.pipeline` uses.
- `apply_tiers(base, m3, tiers)`: `mult = np.ones_like(base)`. Iterate the tiers from the least deep, `for thr, m in tiers: mult = np.where((m3 < thr) & (base > 0.5), m, mult)`, so the deepest satisfied tier wins, and the multiplier is the total one (T2's 0.25 is the total, not 0.5 × 0.25). Return `base * mult`.
- `inner_threshold(y_inner, s_inner)`: `min(np.unique(s), key=cost_at(y, s, t, 0.3))`, as in `fuse_sweep_m5.py:107`.
- `readout(y, src, sc)`: the M5 readout (`fuse_sweep_m5.py:93–110`) plus `inner_averse`. It returns the six cells plus argmin, the LJ/Libri split and ITW P_FA / P_miss.
- `deltas(cur, cand)`: `{cell: round(cur - cand, 4)}` over the six cells, using values already rounded to 4 decimals.
- `gate(d: dict, spearman: float, diag_ok: bool) -> dict`. It returns `{"rules": {"2a":…, "2b":…, "3a":…, "3b":…, "4a":…, "4b":…, "5_brief":…, "5_averse":…, "5_diag":…}, "doubled": bool, "verdict": "PASS"|"FAIL", "failed": [...]}`. Rule 6 doubles the 2a and 3a thresholds when `spearman < 0.974`. Comparisons are ≥ on the rounded Δ (`round(x, 4) >= thr - 1e-12` to absorb float representation). Rule 5 counts `d[(p, c)] > 0` over p ∈ {inner, holdout, itw}, requiring ≥ 2 for each c.
- `standing_rule(d) -> dict` (the W4 bake-off's part 1).
- `pareto_pick(passers: dict[name, dict cell→value]) -> str | None`: none if empty; the single passer if one; otherwise the passer that is ≤ every other passer in all six cells and < in at least one; else None.
- `t2_fires(y, base, m3) -> int`: `sum((y==1) & (m3 < -6) & (base > 0.5))`.
- `new_catches(y_itw, cur_itw, thr_cur, col_itw, thr_col) -> int`: CURRENT's misses are `(y==1) & (cur < thr_cur)`; the column catches those with `col >= thr_col`.
- `corrective_share(y, cur, thr_cur, r_new, r_m1b) -> float`: errors are FAs `(y==0)&(cur>=thr)` and misses `(y==1)&(cur<thr)`. Corrected is `r_new < r_m1b` on FAs and `r_new > r_m1b` on misses (strict). Returns corrected / errors; 0 errors → NaN, which fails.
- `cluster_bootstrap(y, s_cur, s_cand, groups, n, seed) -> {"brief": p5, "averse": p5, "n": n}`: raises on NaN or empty groups. Unique groups; per replicate `rng.choice(G, G, replace=True)`, concatenating each drawn group's row indices (precomputed index lists). The same index array feeds both rules; Δ = cost(cur) − cost(cand) per cost; returns `np.percentile(Δ, 5)`. A replicate with a single class is skipped and counted (reported).
- `test_agreement(test_paths, fused, platt, shipped_df) -> {"spearman", "crossings", "n_nonpinned"}`:
  - Map `Path(p).name` to the shipped rows and raise on any unmatched name.
  - Non-pinned = shipped cm-score ≥ 0.001.
  - Candidate p = `0.001 + 0.999 * sigmoid(a*fused + b + prior_shift)`.
  - Crossings = `sum((p >= 0.5) != (shipped >= 0.5))`. No file names are returned.
- `fit_platt(s_inner, y_inner)`: class-balanced `LogisticRegression` → (a, b), as in `fuse_sweep_m5.py:129`.
- `perturb_eval(df, consts, weights_by_rule)`: ranks from `consts["rank_ref_inner_oof_sorted"]`, E tiers, per kind × {brief, averse} per rule. It also returns `max |CURRENT − df.fused|` (the tripwire).

### 2. `main()` and the data path (D2, D3, D4, D11, D12)

1. Parse args: `--write`, `--ratified-by`, `--candidate`, `--new-column-model`. Validate early:
   - `--write` without `--ratified-by nathan` or without `--candidate` → `ap.error`.
   - `--candidate` not in `CANDIDATES` → `ap.error`.
2. `if not sha_ok(SHIPPED_TSV): sys.exit("shipped TSV changed …")`.
3. Load the base columns with `fs.load` (m1b_v3; handcrafted_v5 plus the `_itw` file; spectra_aasist plus `outputs/spectra/itw_stress/scores.csv`; m5_xlsr_ft plus `_itw_m5.csv`), and the labels, as in `fuse_sweep_m5.py:48–55`. Attach the speaker for ITW from `itw_stress.csv`.
4. `build(cols: dict[name → df]) -> ctx`, where ctx holds the per-split intersection over all given cols plus spectra, `y`, `src`, `spk`, and the rank dicts. Rank refs come from this ctx's inner-OOF rows.
5. Base ctx → CURRENT readout → self-check against `EXPECTED_CURRENT` (exact equality of the rounded values; argmin list equality) and Spearman ≥ 0.999. On a failure, print SELF-CHECK FAIL with the diff, write the report with `"self_check": {…, "ok": false}` and every verdict `"INVALID"`, and `sys.exit(2)`.
6. For each candidate in `CANDIDATES` order:
   - `new=None` → base ctx.
   - Otherwise, check `S / f"{name}.csv"` exists; if not, set status `NOT RUN` and continue.
   - Otherwise, load it (ITW via its `_itw` file, or its own `stress` or `itw` split), build its ctx, and run the direction guard (`roc_auc_score(y_inner, logit_inner) < 0.5` → raise ValueError naming the column).
   - Compute CURRENT on the same ctx and the candidate score; readouts for both; deltas; test agreement for both (each with its own Platt); the diagnostic; then the gate or the bake-off.
   - Record `rows`, and the shrink vs base: per-split counts, `shrunk: bool`.
7. Diagnostics:
   - **T2:** `t2_fires` on inner, holdout and itw (the base being the pre-tier blend on each ctx), plus each perturbation kind from `perturb_eval`'s base.
   - **P_wl:** `new_catches` on itw; `corrective_share` on inner and itw; Spearman(`wavlm_l`, `m1b_v3`) on holdout and test logits (disclosure).
   - **W4:** none (the bake-off instead).
8. **W4 bake-off:**
   - `standing_rule(d)`;
   - `perturb_eval` with CURRENT and W4 weights: tripwire ≤ 1e-9, else `"INVALID"`; all 10 cells have Δ ≥ −0.010;
   - `cluster_bootstrap` on ITW: p5 > 0 for both costs.
   - Also record `gate(d, …)` for W4 for the record (expected FAIL rule 3).
   - W4's final status is "BAKEOFF PASS" or "BAKEOFF FAIL". A W4 bake-off pass counts as a passer for rule 7.
9. Also check T2 against the base CURRENT (six cells identical) and W4 against `EXPECTED_W4`, printed as `known-number check: ok/mismatch` (non-blocking).
10. Decision: `passers = {k: six cells}` for verdict PASS (the gate) or BAKEOFF PASS; `pick = pareto_pick(passers)`. Print one table (a pandas DataFrame of the six cells plus Spearman, crossings, verdict per candidate, with CURRENT on the base set as the first row), then the decision line:
    - `KEEP (no passer)`
    - `RECOMMEND <pick>`
    - `KEEP (≥2 passers, none Pareto-dominant)`
11. Write `outputs/fusion/sweep_v3_report.json`: manifest commit `141b254`, rows, self_check, per candidate {status, rows, readout, current_readout, deltas, test_agreement, diagnostic, gate, bakeoff, known_number_check}, passers, decision.
12. `--write` path (D12):
    - Refuse (exit) if `--candidate` is not the decided pick, or is not a passer.
    - Refuse if `models/fusion_v3/constants.json` exists.
    - `from hearsay.pipeline import M5_DIR` imported here only.
    - Write the `fusion_v2` shape: `final`, `pi_synth`, `rank_ref_inner_oof_sorted` over the candidate's ranked columns, `weights`, `e_rule` (plus `tiers` when there are two), `platt`, `determinate_map`, `how`, `m5_checkpoint`, `ratified_by`, `manifest_commit`.
    - For a new column, `new_column_model: {"name", "dir": --new-column-model (required, must exist)}`.
    - Never writes elsewhere.

### 3. `tests/test_fuse_sweep_v3.py` (hermetic plus `needs_data`)

Load the module via importlib (as `tests/test_channel_robustness.py:19–23` does). Tests, per the run spec's inventory:
1. `test_brief_is_min_cost_pi03`
2. `test_averse_is_min_cost_sponsor`
3. `test_brief_weights_recover_9_33_to_1`: 1 FA in n_real vs 1 miss in n_spoof, at a fixed threshold via `cost_at`.
4. `test_brief_differs_from_averse_on_asymmetric`
5. `test_rank_vs_matches_hand`
6. `test_blend_order_and_weights`
7. `test_manifest_weights_sum_to_one`
8. `test_apply_tiers_point_cases` (parametrized over five cases, including exactly −3 and exactly −6, where the comparison is strict)
9. `test_e_tiers_equal_shipped_e_step`: equals `np.where((m3<-3)&(base>0.5), base*0.5, base)` on random arrays.
10. `test_gate_all_pass`
11. `test_gate_each_rule_fails` (parametrized over nine sub-rules)
12. `test_gate_boundary_0p020`
13. `test_gate_rounding_first`
14. `test_gate_inner_averse_equal_passes_worse_fails`
15. `test_gate_spearman_doubles` (0.9739 vs 0.974; doubled thresholds apply only to 2a and 3a)
16. `test_standing_rule_cases`
17. `test_pareto_pick` (parametrized: none, single, dominant, non-dominant, all-equal)
18. `test_t2_fires_counts_only_spoof_above_half`
19. `test_new_catches_15_fails_16_passes`
20. `test_corrective_share_half_fails`, `test_corrective_share_ties_do_not_count`, `test_corrective_share_no_errors_is_nan`
21. `test_bootstrap_deterministic`
22. `test_bootstrap_paired_and_clustered`: one speaker holding all the separation → the percentile reflects whole-speaker draws; the two rules share draws (a test hook returns the draws).
23. `test_bootstrap_missing_speaker_raises`
24. `test_bakeoff_parts_fail` (a pure `bakeoff_verdict(standing, perturb_cells, p5)` function; parametrized over three failures plus a pass)
25. `test_test_agreement_excludes_pinned_and_counts_crossings`, `test_test_agreement_unmatched_raises`
26. `test_direction_guard_raises_on_inverted`
27. `test_missing_export_not_run`: build_candidate with a tmp `S` → NOT RUN.
28. `test_shrunk_rows_recorded`
29. `test_write_refused_without_ratification` (subprocess `--write` → nonzero, stderr names `--ratified-by`)
30. `test_write_refuses_existing_target`: pure `write_constants(path, …)` raises FileExistsError on a tmp path.
31. `test_sha_mismatch_detected`
32. `test_import_hygiene` (subprocess: import the module; assert "lightgbm" and "torch" not in `sys.modules`)
33. `needs_data` `test_self_check_reproduces_shipped_numbers`: base ctx → CURRENT readout == `EXPECTED_CURRENT`; Spearman ≥ 0.999.
34. `needs_data` `test_t2_identical_and_w4_known_numbers`
35. `needs_data` `test_perturbation_tripwire`: max |recomputed − fused| ≤ 1e-9.

For the subprocess tests to be possible, the refactor keeps `main()` light: data loading is inside `main`/`build`, not at import.

### 4. Run and verify

```bash
uv run pytest -q tests/test_fuse_sweep_v3.py
uv run ruff check scripts/fuse_sweep_v3.py tests/test_fuse_sweep_v3.py
uv run python scripts/fuse_sweep_v3.py        # A3: self-check ok, T2 identical/FAIL, W4 known numbers + gate FAIL (rule 3) + bake-off verdict, P_wl / H_noise NOT RUN
shasum -a 256 submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv submissions/CrossExam_predictions.tsv   # both fb783076…
git status --short models/                    # empty
uv run pytest -q && uv run ruff check .       # A2
```

The W4 bake-off result is then **known before 19:15**. It is recorded in the sweep report as computed; nothing about it is revisited.

### 5. Docs and commit

- Run spec §Results: test counts, and the T2 / W4 outcomes from step 4.
- `CLAUDE.md` AI-use disclosure: one bullet ("the post-draft gate lane (Sat Sep 26, 17:20–): the manifest and pre-declared gate, `scripts/fuse_sweep_v3.py`, its tests, the W4 fresh-evidence bake-off, the locked table …").
- Stage explicitly: `scripts/fuse_sweep_v3.py tests/test_fuse_sweep_v3.py docs/specs/2026-09-26_post-draft-gate-sweep.md docs/reports/2026-09-26_post-draft-gate-sweep-plan.md docs/reports/2026-09-26_post-draft-gate-sweep-plan-review.md CLAUDE.md`. Commit `feat(gate): fuse_sweep_v3 — pre-declared gate, W4 bake-off, T2 audit`.
- Post-commit: an adversarial Claude critique on the audit rubric (loop), then `bash .claude/review-audit.sh docs/specs/2026-09-26_post-draft-gate-sweep.md`.

### 6. At the WavLM export (~19:15): one run, locked table

- Confirm `outputs/detector_scores/wavlm_l.csv` exists, and record the heads chat's commit hash.
- Run `uv run python scripts/fuse_sweep_v3.py` **once**.
- Write `docs/reports/2026-09-26_post-draft-sweep.md` with:
  - the table from the JSON, the gate verdict per rule and candidate, and the diagnostics;
  - the bake-off, Spearman and crossing counts, and row counts;
  - the NOT RUN note for H_noise (CPU lane, `6c3e12f` / `16b0376`);
  - one recommendation.
- Commit it, and present it to Nathan by ~19:30.

## Risks and mitigations

- **The self-check fails** (for example the inner averse or the rounding differs from the M5 report). EXPECTED_CURRENT holds only the fields the M5 report actually has, and inner averse is new, so it isn't in the dict. Diagnose before reading any candidate.
- **The bootstrap is slow** (2,000 × 2 rules × 2 costs × `roc_curve` on about 3,000 rows is about 8,000 calls). It should take about 20–40 s. If it is too slow, vectorize nothing and accept it: correctness over speed.
- **ITW speaker values missing or NaN.** It raises. The manifest has the speaker column, so the check runs at step 4.
- **WavLM export shape differs** (for example ITW under `stress` or in a separate file). `fs.load` handles `stress`. A separate `_itw_wavlm_l.csv` is picked up if present; otherwise the ITW rows must be in the export, and a missing ITW split yields an intersection of 0 → the candidate is INVALID with a message (never a silent pass).
- **The perturbation cohort's M5 logits are CPU-runner logits, while the rank references come from A100 exports.** This is the same for both rules, and it is exactly what the shipped `fused` column used. The tripwire confirms the recomputation matches the runner.
- **Winner's curse and test-set peeking.** No file names are printed for crossings, and no parameter is exposed on the command line except the ratification flags.

## Review resolutions (18:25; Codex plan review + Claude critique; supersede the steps above where they differ)

Codex: `docs/reports/2026-09-26_post-draft-gate-sweep-plan-review.md` (10 findings). Claude critique: 1 blocker, 7 should-fix, 5 nits. It verified the self-check reproduces exactly (inner averse 0.2997), the perturbation tripwire holds at 1.1e-16, import hygiene holds, and ITW has 54 speakers with no NaN. The manifest clarifications were committed at `b877760` before any new number.

| # | Finding | Resolution in the code |
|---|---|---|
| C1 / Cr6 | `E_TIERS` used before its definition | Moved above `CANDIDATES` (step 1). |
| C2 | Rule-7 Pareto compares different row sets | `pareto_pick` receives passers re-scored on the intersection of all passers' row sets (`rescore_common(passers)`). Test: two passers on different rows → compared on common rows. |
| C3 | Fold contract not validated; `fs.load` silently dedups | New `validate_export(df_raw, folds)` runs on the raw CSV **before** `fs.load`. It requires unique normalized paths, finite logits, and every inner_oof / holdout row's split agreeing with `splits/nsa_folds.csv` (holdout ↔ fold `holdout`; inner_oof ↔ fold 0–4). A failure makes the candidate INVALID. Provenance: the sha256 of every input CSV goes in the report, and the heads chat's commit hash is recorded in the sweep report by hand. |
| C4 | Partial test coverage | Test rows must be exactly the 1,671 shipped basenames, each once. ITW rows must be the 3,000. Otherwise INVALID. Tests: a missing row, and a duplicate basename. |
| C5 / Cr4 | A raise aborts the locked run; NaN Spearman bypasses doubling | Each candidate is evaluated in `try/except (ValueError, KeyError)` → `{"status": "INVALID", "reason": …}`, and the loop continues. A non-finite Spearman or diagnostic → INVALID. Test: an inverted P_wl fixture → INVALID while W4 and T2 still produce results, and the report is written. |
| C6 | `--write` not bound to the locked inputs | Non-write mode writes `sweep_v3_report.json` with an `inputs` block (sha256 of every CSV read, the shipped TSV, the fusion_v2 constants, the manifest commit). `--write` **does not rewrite the report**. It loads it, recomputes the input hashes and refuses on any difference. It re-runs the evaluation and refuses unless the recomputed decision equals the report's `decision.pick` and the candidate matches. It writes only `models/fusion_v3/constants.json` (refusing if it exists). For P_wl, it requires `--new-column-model DIR` whose `meta.json` exists, and records that file's sha256. A residual risk is stated: the tie between the model directory and the export is by the heads report, not by bytes. |
| C7 / Cr5 | H_noise diagnostic undefined | H_noise's `diag_ok` is hard-coded `False` with the reason "no noise-AUC evidence supplied (withdrawn by the CPU lane 17:42)" (manifest clarification 4). It cannot pass. |
| C8 | Threshold search omits "call everything real" | `inner_threshold` searches `np.r_[np.unique(s), np.inf]`. The self-check still reproduces 0.013 / 0.122; if it doesn't, stop and report. Test: a constant-score column → threshold `inf`, 0 catches. |
| C9 | `git status models/` can't see ignored files | Verification hashes `models/fusion_v2/constants.json`, the shipped TSV and `submissions/CrossExam_predictions.tsv` before and after. Write-path tests cover a failing candidate, a passing but non-picked candidate, an existing target, and successful serialization to a tmp target (a pure `write_constants(path, payload)`). |
| C10 | No closing step or stop behavior | New **step 7** below. |
| Cr1 | P_wl miss set: 158 vs 122 | Manifest clarification 1: inner-OOF threshold (122 misses), absolute bar 16. The report prints the miss count and each column's catches (M1b, M5, M3, wavlm_l) at that threshold for scale. |
| Cr2 | Empty pinned block | Manifest clarification 2: all 1,671 rows. |
| Cr3 | T2 not row-identical | The `needs_data` T2 check compares only the six rounded cells, and it is **non-blocking** in the script (it prints `known-number check`). The report also carries bona fide fire counts. A3 is reworded to "T2 six cells expected identical (non-blocking)". |
| Cr7 | Perturbation rounding | Cells are rounded to 4 decimals before Δ ≥ −0.010; the bootstrap uses `np.percentile` linear on unrounded Δ. Both are pinned in tests. |
| Cr8 | Inventory items without tests | Added tests: averse 1 : 4 recovery; other candidates still evaluated after NOT RUN / INVALID; a shrunk ctx rebuilds rank refs from its own inner rows while base CURRENT is unchanged; `--write` refused for non-pass or non-pick; tripwire failure → bake-off INVALID (hermetic, via `bakeoff_verdict(…, tripwire=1e-6)`); self-check failure → every verdict INVALID with exit 2 (pure `finalize(self_check_ok=False, …)`); T2 diagnostic aggregating every split and kind; boundaries 2b (−0.002 passes, −0.0021 fails) and 4a/4b (−0.010 passes, −0.0101 fails). |
| Nits | Tier order; `fused_base` column; P_wl constants vs `DETECTOR_ORDER`; D13 wording; test rows | `apply_tiers` asserts the thresholds are strictly decreasing. The T2 perturbation diagnostic reads the CSV's `fused_base`, as the manifest says. P_wl's integration risk is flagged in the packet (`FusionConstants` rejects `wavlm_l` today; `pipeline.py:180–183`). The run spec's D13 wording is corrected (`hearsay.pipeline` doesn't import torch). Test coverage below 1,671 is INVALID, not disclosed-and-scored. |

The test count rises from 35 to about 48.

### 7. Close-out (new)

- **19:25, lock.** If `outputs/detector_scores/wavlm_l.csv` is absent at 19:25, P_wl is NOT RUN, and the table is locked with T2, W4 and H_noise only. If the export lands, the one run happens immediately. If the export or the self-check fails validation, the candidate is INVALID; there is no rerun after a fix.
- **19:30–20:00, Nathan's window.** He either ratifies one candidate from the packet's ratifiable set, or says KEEP. No ruling by 20:00 means KEEP.
- **KEEP:**
  - Append a `submissions/log.csv` note row: KEEP, `submissions/CrossExam_predictions.tsv` final, sha256 `fb783076…` re-verified at write time, sweep report path.
  - Message the CPU lane (for its §7) and the oversight chat.
  - Stop.
- **Ratify:**
  - `--write --ratified-by nathan --candidate NAME` → `models/fusion_v3/constants.json`.
  - Runner integration, the full 1,671-row parity, the new immutable TSV and its log row, and a new final-name copy are the **integration job**, owned by this lane unless Nathan assigns it elsewhere, with a hard stop at 21:30. If the job cannot finish by 21:30, the ratification lapses to KEEP, and the step-1 file stays final (log note row).
  - Message the CPU lane, the oversight chat and the README chat.
