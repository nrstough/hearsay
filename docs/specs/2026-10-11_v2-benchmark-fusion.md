# Run spec: v2 benchmark harness (V2-01-lite) + pre-declared fusion test (V2-60)

**Stakes:** HIGH — it changes how the final synthetic probability is computed and judged, and it defines the yardstick the other two v2 lanes are scored by.

- Branch / worktree: `v2-benchmark-fusion` (cut from `v2` at 6edbf13), worktree `.claude/worktrees/v2-benchmark-fusion`. `models/`, `outputs/`, `data/*`, `weights/*` are symlinks into the main checkout (read-only except this lane's `outputs/v2/benchmark/` and `outputs/v2/fusion/`).
- Source handoff: `docs/handoffs/2026-10-10_v2-fusion-benchmark-handoff.md`. Research: `docs/reports/2026-10-10_v2-fusion-research.md`.
- Pipeline: `/plan-review`, P1 frozen and P2 frozen ("freeze deep") on Oct 11.
- Time box: none (Nathan override, Oct 11; CLAUDE.md updated). Report at the end of each part.

## Owner's statement

### P1
- **What it does:** Step one is benchmarking. So getting one harness that is the same every time so that we can track progress. And part B is different fusion experiments, our current and equal weights, a small learned model, and possibly a larger learned model after that if it works.
- **Most likely way it is wrong:** What's most likely to go wrong here is probably a sort of data leakage. I think we found that that had happened.

### P2
- **What it does:** These tests prove that the fusion layer will, well, the fusion layer tests will be correct. And it makes sure that each proposed run runs correctly. And make sure that they almost load right. All must. All must.
- **Most likely way it is wrong:** The most likely thing to go wrong here is one of the learned models to overfit to what it is trained on and not generalize well.

## P1 (frozen Oct 11)

**Conventions.** Run specs `docs/specs/`; plans and reports `docs/reports/`; no feature-spec layer (CLAUDE.md); super doc `docs/prod-chunks.md` (V2-01 and V2-60 sections only); no bug log (the sweep used past run specs, reports and memory: 29 classes); tests `uv run pytest -q`, lint `uv run ruff check .`; Codex via `.claude/review-plan.sh` / `.claude/review-audit.sh`.

**Problem.**
- No common yardstick exists; v1's readouts disagreed by up to 11×.
- The reported inner 0.1351 for the shipped rule was a leakage artifact (rows ranked against their own fold); the honest number is 0.1558.
- The hard-negative and backbone lanes are blocked waiting for a shared scorer.
- Whether learned fusion beats the shipped blend has never been tested under a pre-declared protocol.

**Approach.**
1. **Part 1, the benchmark.** One shared scorer takes any per-clip score file or fusion rule, scores it on every population we hold today with each population's role labelled (selectable, read once, report-only, in-sample), reports all three metric readings plus EER with speaker/generator cluster bootstrap intervals, pins every input by hash, and must reproduce the reference numbers before anyone trusts it. Its input format is published early (this spec, "Interface v1").
2. **Part 2, the fusion test, on option B's columns.** At most four candidates pre-declared and committed before any run: the shipped rule (control), fixed equal weights, one small shrunk learned weighting, and a fourth slot (set in P2). Spectra is never fitted and only applies the frozen veto afterwards. A larger learned model is a conditional follow-on in its own run, only if the small one wins. Adoption waits for a fresh sealed population.

**Decisions carried in.**
- Metric: NSA's minDCF at Pspoof 0.5 ("package-0.5", P_FA + 4·P_miss) selects; brief and package-0.3 always reported; no candidate may regress beyond its tie band on any reading.
- Discipline: train-fold rank references; In-the-Wild report-only; holdout read once per candidate set.
- Deferred: MLAAD by family, ASVspoof 5, reverb/codec/telephony/short-crop cohorts, the sealed set, actDCF (to V2-61).
- **Option B**, chosen by Nathan against the research recommendation (A), with an ElevenLabs-fold guard.
- No time box (Nathan override). One run spec with two parts; Part 1's interface committed first.

## P2 (frozen Oct 11, "freeze deep")

### Decisions (Nathan agreed to all five)
1. **C2 domain: train-fold ranks.** Output is a weighted rank in [0,1], so E's "base > 0.5" is measured on it exactly as for the shipped rule.
2. **handcrafted_v5: no cap.** C2 shrinks toward C0's weights (0.6 / 0.2 / 0.2, and 0 for WavLM and L20); hc reaches 0 only when α = 0 and the fit gives it 0.
3. **C3** = 0.8 × C0 weights + 0.1 WavLM + 0.1 L20, then E.
4. **Tie bands:** brief 0.02, package-0.5 0.05, package-0.3 0.03. **ElevenLabs guard:** f4 package-0.5 ≤ C0's f4 package-0.5 + 0.05.
5. **Cohort compute in scope, ask-first:** WavLM + L20 on both perturbation cohorts (MPS, ~40 min; drive now mounted); hc_v5, M5, Spectra on the ITW cohort (CPU, ~35 min).

### Candidates (option B; Spectra only ever as E)

| ID | Rule | Fitted params |
|---|---|---|
| C0 | Shipped: 0.6·m1b_v3 + 0.2·handcrafted_v5 + 0.2·m5_xlsr_ft on train-fold ranks, then E | 0 |
| C1 | Equal rank weights (0.2 each) over m1b_v3, handcrafted_v5, m5_xlsr_ft, wavlm_l, xlsr_L20, then E | 0 |
| C2 | Non-negative LR on the five columns' train-fold ranks, cost-weighted at π_eff 0.8 (package-0.5), weights normalized to sum 1, shrunk toward C0 weights with α ∈ {0, .25, .5, .75} chosen by inner-inner leave-one-fold-out; then E; then Platt fitted per training fold | ~1–2 effective |
| C3 | 0.48·m1b_v3 + 0.16·hc_v5 + 0.16·m5 + 0.1·wavlm_l + 0.1·xlsr_L20, then E | 0 |

E (frozen, `models/fusion_v2/constants.json` e_rule): if Spectra logit < −3.0 and base > 0.5 (both strict), base × 0.5.

**Verdict:** WIN = beats C0 beyond the package-0.5 band on pooled nested inner AND no reading regresses beyond its band AND the ElevenLabs guard holds. LOSE = worse beyond any band. TIE = otherwise. The larger learned model is a separate run, only if C2 = WIN.

### Failure modes → tests
Files: `tests/test_benchmark.py`, `tests/test_v2_fusion.py`. "Data" = marked `needs_data`.

**L. Leakage (owner's P1 risk; required)**
| Failure mode | Test |
|---|---|
| Row ranked against refs including its own fold | `test_nested_refs_exclude_scored_fold` |
| Inner-inner α selection sees the outer fold | `test_alpha_selection_never_sees_outer_fold` |
| Platt/scaling/weights fitted on the scored fold | `test_c2_fit_inputs_are_train_folds_only` |
| Spectra enters a fitted parameter | `test_spectra_refused_as_fit_feature`, `test_dominance_check_flags_column_over_share` |
| Selection on ITW, its cohort, MLAAD, test or holdout | `test_select_refuses_report_only_population` (per role) |
| Holdout read twice for one candidate set | `test_holdout_read_ledger_refuses_second_read`, `test_holdout_ledger_allows_new_candidate_set` |
| ASVspoof 2019 LA eval rows in any path | `test_loader_refuses_asv19_eval_paths` |
| 23,790-row basis mixed in | `test_inner_basis_is_16142_and_l20_filtered_by_path`, `test_basis_mismatch_raises` |
| Candidate added/changed after the spec is committed | `test_run_refuses_if_candidates_differ_from_spec_hash` |
| MLAAD read as selectable for rules containing M5 | `test_mlaad_role_in_sample_for_m5` |

**O. Overfitting (owner's P2 risk; required)**
| Failure mode | Test |
|---|---|
| Learned fit good in-sample, not out of fold | `test_c2_reports_in_sample_gap` |
| Verdict read from in-sample numbers | `test_verdict_uses_nested_oof_only` |
| Pure-noise column earns weight | `test_noise_column_gets_near_zero_weight_after_shrink` |
| Parameter count hidden | `test_c2_reports_effective_params_and_alpha_per_fold` |
| α unstable across folds | `test_alpha_per_fold_reported_and_spread_flagged` |
| Generalizes on inner but not off-distribution | `test_learned_readout_includes_holdout_and_itw_gap` |
| Larger model run without C2 winning | `test_followon_refused_unless_c2_verdict_win` |

**S1. Loader and score-file contract**
| Failure mode | Test |
|---|---|
| Mixed abs/rel paths silently drop rows | `test_canonical_key_abs_and_rel_forms_agree`, `test_canonical_key_from_worktree_and_main_agree`, `test_canonical_key_never_resolves_symlinks` |
| Path outside the repo | `test_path_outside_repo_raises` |
| Population shrinks after a join | `test_coverage_assert_after_join_raises_on_one_missing_row` |
| Duplicate paths dropped silently | `test_duplicate_path_raises` |
| Basename keys collide | `test_nsa_join_uses_full_path_not_basename` |
| Unknown split; stress vs itw | `test_unknown_split_raises`, `test_stress_maps_to_itw` |
| Fold read as float | `test_fold_coerced_to_str` |
| Non-finite logits | `test_nonfinite_logit_raises` |
| Inverted column | `test_direction_guard_inner_auc_below_half_raises` |
| `score` read instead of `logit` | `test_reader_uses_logit_column`, `test_score_logit_sigmoid_consistency_checked` |
| Wrong ITW source per column | `test_itw_override_sources_pinned`, `test_resolve_itw_missing_override_raises` |
| Missing/extra column in another lane's file | `test_schema_missing_column_raises`, `test_contract_example_file_round_trips` |

**S2. Population registry (`benchmarks/v1.yaml`)**
| Failure mode | Test |
|---|---|
| Input changed since pinned | `test_sha256_mismatch_refuses` |
| Row counts drift | `test_registry_counts_match` (per population) |
| Missing/unknown role | `test_every_population_has_known_role` |
| Deferred population quietly included | `test_deferred_populations_absent_and_listed` |
| Registry differs from disk | `test_registry_hashes_match_disk` (data) |

**S3. Metrics**
| Failure mode | Test |
|---|---|
| Brief default used for package-0.5 | `test_scorecard_primary_is_package05`, `test_no_bare_min_cost_default_in_benchmark` |
| package-0.3 unnamed | `test_package03_formula_hand_case` |
| Cost names swapped (bona fide side) | `test_package05_equals_sponsor_flip` (with ties) |
| Ties / all-real threshold | `test_min_cost_with_heavy_ties`, `test_all_real_threshold_included` |
| FA/miss count without threshold | `test_counts_carry_reading_and_threshold` |
| Single-class subset | `test_single_class_population_reports_na_not_crash` |

**S4. Rank and fusion primitives**
| Failure mode | Test |
|---|---|
| Rank side / denominator wrong | `test_rank_left_side_and_n_train_denominator` |
| Blend order breaks float parity | `test_blend_accumulates_in_weight_order` |
| E boundaries, base before Platt | `test_e_step_boundaries`, `test_e_uses_base_before_platt` |
| NaN through E | `test_e_nan_input_raises` |
| Platt changes ranking | `test_platt_monotone_metrics_unchanged` |
| Bad weights | `test_rule_weights_validated` |

**S5. Readouts**
| Failure mode | Test |
|---|---|
| Pooled-only verdict | `test_readout_has_pooled_perfold_macro_f0_f4` |
| No without-E row | `test_readout_has_with_and_without_e` |
| Dominance / weight shares missing | `test_dominance_spearman_on_real_rows` |
| Macro rounding differs | `test_macro_round_then_mean_matches_research` |
| Candidates on different rows | `test_all_candidates_scored_on_identical_rows` |

**S6. Bootstrap**
| Failure mode | Test |
|---|---|
| LJ split by chapter (CIs too narrow) | `test_lj_is_one_cluster` |
| Clusters not class-stratified | `test_inner_bootstrap_clusters` |
| Non-deterministic / unpaired | `test_bootstrap_seeded_and_paired` |
| All draws single-class | `test_bootstrap_all_skipped_raises` |
| Length mismatch | `test_bootstrap_length_mismatch_raises` |

**S7. Reference reproduction (data; acceptance bar; within 5e-5 of the rounded reference)**
`test_ref_m1b_inner`, `test_ref_m1b_holdout`, `test_ref_m1b_itw`, `test_ref_ship_transductive_inner`, `test_ref_ship_transductive_holdout`, `test_ref_ship_transductive_itw`, `test_ref_ship_nested_inner`, `test_rerun_bit_identical`.

| What | Basis | brief | package-0.5 | package-0.3 |
|---|---|---|---|---|
| m1b_v3 export | inner 16,142 | 0.3013 | 0.3307 | 0.2280 |
| m1b_v3 export | holdout | 0.0717 | 0.0571 | |
| m1b_v3 export | ITW | 0.342 | 0.2955 | |
| SHIP transductive | inner 16,142 | 0.1351 | 0.2997 | |
| SHIP transductive | holdout | 0.0065 | 0.0087 | |
| SHIP transductive | ITW | 0.228 | 0.2385 | |
| SHIP nested | inner 16,142 | 0.1558 | 0.3390 | |

**S8. Cohort scoring script**
| Failure mode | Test |
|---|---|
| Decode error zero rows | `test_cohort_flag_column_and_zero_audio_refused` |
| Rows ≠ manifest | `test_cohort_rows_match_cohort_rows_fn` |
| Perturb seed differs | `test_perturb_seed_matches_m3_probes` |
| Partial cohort accepted | `test_partial_cohort_rejected` |
| Threads > 4 | `test_threads_capped` |

**F. Fusion run**
| Failure mode | Test |
|---|---|
| Negative weight | `test_nnlr_weights_nonnegative` |
| Wrong cost prior | `test_c2_prior_is_0_8_for_package05` |
| α off-grid | `test_alpha_grid_fixed` |
| Full shrink ≠ C0 | `test_full_shrink_recovers_c0_weights` |
| Verdict edges | `test_verdict_win_tie_lose_boundaries`, `test_verdict_any_reading_regression_blocks_win`, `test_f4_guard_blocks_win` |
| Log row malformed | `test_log_row_columns_and_basis_in_notes` |

### Commands
```bash
uv run pytest -q tests/test_benchmark.py tests/test_v2_fusion.py
uv run pytest -q -m needs_data tests/test_benchmark.py
uv run pytest -q && uv run ruff check .
```

### Acceptance
- Every new test passes; all seven reference numbers match within 5e-5; two runs give identical scorecards.
- Interface v1 (below) committed before Part 2 runs.
- All four candidates have full readouts; C2's in-sample gap reported; verdict report written.

### Regression
Any existing test fails; count below 847 passed (12 skipped at baseline); ruff not clean; any file outside this lane's ownership list changes.

### Documentation committed to
- This run spec.
- `docs/reports/2026-10-11_v2-benchmark-fusion-plan.md` (plan).
- `docs/reports/2026-10-11_benchmark-v1.md` (v1 scorecard).
- `docs/reports/<run date>_v2-fusion-verdict.md`.
- `benchmarks/v1.yaml`.
- `docs/prod-chunks.md` V2-01 and V2-60 sections only (deliverables rename, actDCF → V2-61, status).
- `experiments/log.csv` (one row per pre-declared run).
- `CLAUDE.md` (time-box rule, already edited).
- Conditional, confirm at execution: `docs/code-map.md`, if it lists scripts.

### Break-check candidate
`test_nested_refs_exclude_scored_fold`; mutation: rank against all inner rows (transductive).

## Interface v1 (published for the other lanes)

Stable from this commit; any change bumps the benchmark version.

### Score file contract
- CSV, one row per clip. Required columns: `path`, `split`, `logit`. Optional: `fold`, `score`.
- `logit`: finite float, **increasing with synthetic likelihood**. A column whose pooled inner AUC is < 0.5 is refused (direction guard).
- `score`, if present, must equal `sigmoid(logit)` within 1e-6.
- `split` ∈ {`inner_oof`, `holdout`, `test`, `itw`}; `stress` is read as `itw`. Anything else is refused.
- `path`: absolute under the repo root (main checkout or any worktree) or repo-relative. The join key is the repo-relative POSIX path, **never symlink-resolved**. Paths outside the repo are refused. ASVspoof 2019 LA eval paths are refused.
- No duplicate paths (refused, never dropped).
- Coverage: a split present in the file must cover its population exactly (inner_oof = the 16,142 `splits/nsa_folds.csv` non-holdout rows; holdout = 3,858; test = 1,671; itw = 3,000 `outputs/manifests/itw_stress.csv` rows). Partial populations are refused; absent splits are reported as "not scored".
- `inner_oof` must be out-of-fold by `splits/nsa_folds.csv` folds (the 16,142 NSA basis). Numbers on the 23,790 basis are notes, not benchmark rows.
- ITW scores in a separate file are passed explicitly (`--itw`); there is no silent fallback.
- Cohort files: columns `path`, `kind`, `logit`; key (`path`, `kind`); kinds {`none`, `mp3`, `noise20`, `speed`, `shift1`}.

### Populations and roles (`benchmarks/v1.yaml`)
| Population | Rows | Role |
|---|---|---|
| inner (nested OOF) | 16,142 | selectable |
| holdout | 3,858 | read once per candidate set |
| itw | 3,000 | report-only |
| holdout perturbation cohort | 2,500 | report-only |
| itw perturbation cohort | 2,500 | report-only |
| mlaad | 572 (spoof only) | report-only; in-sample for M5 |
| test | 1,671 (unlabelled) | report-only: share above threshold, Spearman and flips vs shipped TSV |

### Entry points (names may be refined in the plan; the CLI and file contract are the contract)
```bash
uv run python scripts/v2_benchmark.py score --file <scores.csv> --name <name> [--itw <itw.csv>]
uv run python scripts/v2_benchmark.py rule --rule <rule.yaml>
uv run python scripts/v2_benchmark.py registry --check
```
Output: `outputs/v2/benchmark/<name>.json` + a markdown scorecard row. Every readout reports brief, package-0.5, package-0.3 and EER, pooled / per-fold / macro (inner), with and without E for rules, weight shares, the dominance check, and cluster-bootstrap intervals. Every FA/miss count names its reading and threshold.

## Amendments (dated; outside the frozen region)

- **2026-10-12, Nathan (STOP 3):** "No platt for now it'll overfit and not generalize". C2 drops the per-fold Platt step: C2 = E applied to the shrunk weighted rank. All candidates are scored on the post-E weighted rank. Calibration is deferred to V2-61. Recorded context: the ruling came after an early inner look in which C2 scored 0.376 package-0.5 without Platt and 0.514 with per-fold Platt (C0 0.3390). Other pre-run clarifications (K2–K13) are in `benchmarks/v2_60_candidates.json` and the plan.

## Test break check
_To be filled in post-commit._

## Results
_To be filled in after execution._
