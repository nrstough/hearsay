# Plan: v2 benchmark (V2-01-lite) + pre-declared fusion test (V2-60)

Run spec: `docs/specs/2026-10-11_v2-benchmark-fusion.md` (P1/P2 frozen; committed 8c66021). Mode: deep (architecture, file-impact and risk exploration, then critique). Branch `v2-benchmark-fusion`, worktree `.claude/worktrees/v2-benchmark-fusion`.

## Disclosure before anything runs
The risk-exploration agent computed preview inner-OOF readouts for the four frozen candidates (no holdout labels touched): C1 package-0.5 0.452 (loses), C3 0.358 (tie, f4 0.224 > guard 0.215), a C2 prototype 0.376 (tie). The candidate table was committed (8c66021) before this, so nothing can be re-picked on it. The verdict report records the early look. The numbers serve only as an independent cross-check of the implementation.

## Pre-run clarifications (committed in `benchmarks/v2_60_candidates.json` at Step 0.5, before any code touches data)
These settle points the frozen P2 left implicit. None changes a candidate, band or guard. K1 amends P2 by Nathan's ruling (Oct 12).
- **K1. No Platt for C2 (Nathan's ruling, Oct 12: "No platt for now it'll overfit and not generalize").** This amends P2's C2 row, which ended "then E; then Platt fitted per training fold". C2 = E applied to the shrunk weighted rank, and nothing after it. Every metric for every candidate is computed on that post-E weighted rank, as for C0 and all reference numbers. Calibration to a probability is deferred to V2-61. Context recorded: the early look gave C2 pooled package-0.5 0.376 without Platt (TIE) and 0.514 with per-fold Platt (LOSE); the ruling was made after that look and the verdict report says so.
- **K2. Inner-inner references exclude both the outer fold f and the inner fold g.** For each outer f, inner fold g ∈ train folds: references = folds ∉ {f, g}; fit on those rows' ranks; score g against the same references; pool the four g blocks.
- **K3. α criterion:** pooled package-0.5 over the four g blocks on the **pre-E base** (Spectra never touches α; P1). E applies only to the final output. Ties within 1e-12 go to the **larger** α (more shrinkage; the research code went the other way, recorded). Grid stays {0, .25, .5, .75}; an `alpha_at_grid_max` flag is reported per fold.
- **K4. Shrink after normalization:** w = α·w_C0 + (1−α)·w_fit/Σw_fit, w_C0 = [0.6, 0.2, 0.2, 0, 0] over (m1b_v3, handcrafted_v5, m5_xlsr_ft, wavlm_l, xlsr_L20). Σw_fit = 0 raises (no silent fallback). Intercept dropped (rank-invariant within a fold; verified).
- **K5. Off-inner C2 model** (holdout, ITW, test, cohorts): refit on all 16,142 inner rows with all-inner references, α chosen by the same procedure one level up (5-fold LOFO on inner). No Platt (K1).
- **K6. Effective parameters** = 1 (α) + (1−α)·(nnz(w_fit) − 1). Also reported: raw ‖w‖₁, intercept, optimizer `success` and `nit`, in-sample cost on the training folds vs nested OOF cost per fold.
- **K7. Bootstrap:** paired, class-stratified, B = 2000, seed 0. Primary: reals = LJ as one fixed stratum present once in every draw (never split by chapter; CIs are conditional on LJ, stated) + LibriSpeech speakers resampled; spoofs = generators resampled. Sensitivity row: LJ resampled as one cluster among the real clusters (draws missing LJ counted). ITW: speakers resampled within class. Degenerate, no CI, marked: holdout (2 spoof generators), inner f3 and f4 (1 generator each). Paired differences carry the statements; marginal CIs carry the caveat that generators dominate them.
- **K8. Holdout ledger:** append-only, tracked `benchmarks/holdout_ledger.jsonl`, mirrored to `outputs/v2/benchmark/holdout_ledger.jsonl`; refuse if either has the key. Key = sha256 of canonical JSON of the sorted rule definitions + the holdout population's label/fold sha (**not** the benchmark version, so a registry bump cannot mint a new read; input shas recorded as data). Reservation is atomic across processes and worktrees: `os.open(..., O_CREAT|O_EXCL)` on a per-key file `outputs/v2/benchmark/holdout_ledger/<key>.json` (shared by every checkout through the `outputs/` link) is the authority; the tracked JSONL and the mirror are audit copies rebuilt from the authority files if one lags. The entry is reserved before computing and completed with the readout and its digest. A crashed reservation may be resumed only with identical code sha and input shas. Later report regeneration (e.g. under v1.1) reuses the persisted holdout readout for an unchanged key rather than reading again. **Exemption:** a hard-coded allowlist of exactly two rule hashes with published holdout references (m1b_v3 column; SHIP with full-inner refs). The Part 1 holdout readout of the other v1 columns (hc_v5, m5, spectra, wavlm_l, xlsr_L20) is one ledgered set, "benchmark-v1 baseline". `v2_fusion.py` refuses a holdout read for any rule not in the committed candidates JSON. Every readout prints the cumulative ledger count. Holdout is off by default (`--holdout`). `HEARSAY_HOLDOUT_LEDGER` is honoured only when `PYTEST_CURRENT_TEST` is set; tests also have an autouse guard that the tracked ledger is byte-identical.
- **K9. Pandas default float parser** for every score CSV (the `constants.json` rank refs equal the sorted inner logits exactly only under it).
- **K11. Verdict scope:** readings = pooled nested inner with E, for brief, package-0.5, package-0.3, compared as integer ten-thousandths (round each cost to 4 decimals, convert to int, subtract ints; no float boundary error). WIN: C0 − cand > 0.05 (strict) on package-0.5, AND for every reading cand − C0 ≤ band, AND f4 package-0.5: cand ≤ C0 + 0.05 (inclusive). LOSE: cand − C0 > band on any reading. TIE otherwise. A C2 WIN is labelled **provisional**: the base columns' OOF scores carry second-order stacking leakage (their models saw the held-out fusion fold when scoring training rows), which favours fitted candidates only; a TIE or LOSE is robust to it. The follow-on may run on a provisional WIN, but its result is provisional too, and adoption needs the sealed population regardless. Per-fold, macro, EER, without-E, holdout, ITW and cohorts are report-only.
- **K12. Other pinned thresholds:** dominance flag = a column's Spearman with the fused score on inner real rows ≥ 0.95 while its weight share < 0.5; holdout/ITW "gap" = population cost − nested inner cost, per reading; the test-population cut = the primary-reading threshold chosen on inner (nested scores for the comparison table, all-ref scores for the off-inner model); no C2 probability is produced (K1).
- **K13. Log row fields:** `validation_score` = package-0.5 pooled nested inner; `validation_score_clean_only` = the same (no augmentation); `csv_path` = `outputs/v2/fusion/v2_60_readout.json`; `benchmark_version` = the registry version in force; `notes` = brief, package-0.3, EER, basis 16,142, verdict.
- **K10. Option-B coverage:** where an option-B column has no scores (holdout cohort until scored, MLAAD), C1/C2/C3 rows read "not scored" with the reason; never a partial blend.

**Time box:** none (Nathan's override, Oct 11, recorded in the run spec and CLAUDE.md). Generic rubric text asking for a few-hour limit does not govern this run. End-of-part reports to Nathan after Part 1 and after Part 2; ask-first stops at the cohort compute.

## Step 0: pre-flight
1. `git branch --show-current` = `v2-benchmark-fusion`; `pwd` = the worktree; `git status` clean except this plan.
2. Baseline: 847 passed, 12 skipped; ruff clean (recorded Oct 11).

## Step 0.5: commit the pre-declaration first
`benchmarks/v2_60_candidates.json`: C0–C3 weights, E rule, α grid, π_eff 0.8, K1 (with Nathan's ruling), K2–K13, tie bands, f4 guard, and `spec_frozen_region_sha256` (spec text from `## P2 (frozen Oct 11` up to `### Failure modes`, LF-normalized, trailing whitespace stripped, as of 8c66021). Commit it with this plan before any code reads data. `v2_fusion.py` refuses if the JSON has uncommitted changes and records its git blob sha in the readout.

## Part 1: the benchmark

### Step 1. `src/hearsay/benchmark.py` (new, pure: numpy, pandas, scipy, sklearn, `hearsay.metrics`; no torch/lightgbm)
Ports cite their source line. Nothing is imported from `scripts/`.
- Constants: `BENCHMARK_VERSION="v1"`, `SPLITS`, `SPLIT_ALIASES={"stress":"itw"}`, `ROLES={"selectable","read_once","report_only","in_sample","reference"}`, `COHORT_KINDS`, `READINGS={"brief":(0.3,4,1),"package05":(0.5,1,4),"package03":(0.3,1,4)}`, `PRIMARY="package05"`.
- Keys: `canonical_key(path)` — lexical, no `resolve()`; strips the main-checkout prefix or any `.claude/worktrees/<x>/` prefix; refuses paths outside the repo and ASVspoof 2019 LA eval (`ASVspoof2019_LA_eval`, `LA_E_`, `asv19_eval_probe`). Replaces `scripts/fuse_sweep.py:31-33`.
- `read_itw_override(path)`: `path` + `logit` required; `split` optional and ∈ {itw, stress}; extra columns ignored and listed in the scorecard (covers `_itw_m5.csv`, `_itw_handcrafted_v5.csv`, `outputs/spectra/itw_stress/scores.csv`, using its `logit`).
- `read_score_file(path, itw=None)` → `ScoreFile`: schema (`path`,`split`,`logit`), logit never score, `score≈sigmoid(logit)` within 1e-6 if present, stress→itw, unknown split raises, fold→str, finite, duplicates raise, ITW only from explicit `itw=`; pandas default parser. Mirrors `scripts/fuse_sweep_v3.py:307-328` checks.
- `read_cohort_file(path)`: key (path, kind), kinds ⊆ COHORT_KINDS, 250+250 per kind present; ignores any `fused*`/`e_applied` columns (transductive).
- `load_labels(root)`: `splits/nsa_folds.csv` + `outputs/manifests/itw_stress.csv` keyed by canonical path; `cluster` column (LJ → `lj`, LibriSpeech speaker, generator for spoofs, ITW speaker); `stratum` column (lj / libri / spoof / itw-real / itw-spoof).
- `align(sf, population)`: exact coverage or raise; absent split → None ("not scored").
- `check_inner_basis(sf, labels)`: fold ids equal `nsa_folds.csv`, exactly 16,142.
- `direction_guard(y, s)`: pooled inner AUC < 0.5 raises.
- Metrics: `reading(y, s, name)` always passes π, C_FA, C_miss explicitly; single-class → None. `readings()`, `counts_at()` (threshold search includes +inf; carries reading + threshold), `fold_table()` (pooled, f0–f4, macro round-then-mean as `extra.py:74-78`).
- Ranks/fusion: `rank_vs` (side left, /len(ref); `fuse_sweep_v3.py:105`), `nested_ranks`, `full_ranks`, `blend` (accumulate in weight order; `:109-113`), `validate_weights`, `apply_e` (strict, raises on non-finite; replaces `apply_tiers`), `fit_platt`/`to_p` (`:281-287`), `Rule`, `score_rule()` returning base and with-E scores per population.
- `cluster_bootstrap()` per K7; `dominance()` (weight shares + Spearman vs blend on real rows); `test_population_report()` (share above the primary threshold, Spearman and flips vs the shipped TSV, resolved from the main checkout via `git rev-parse --git-common-dir`); `scorecard()` → dict, `json.dumps(sort_keys=True)`, no timestamps inside.
- Registry: `load_registry`, `check_registry` (sha256 + counts), `HoldoutLedger` (K8), `require_selectable(pop, rule)` (MLAAD is `in_sample` for rules containing m5_xlsr_ft).

### Step 2. `scripts/v2_benchmark.py` (new CLI)
- `adapt-l20`: writes `outputs/v2/benchmark/scores/xlsr_L20.csv` from `outputs/v2/bakeoff/probes/L20.npz` + `rows.json`: oof filtered by path to the 16,142 NSA inner rows (folds checked), holdout in `nsa_folds` order, test via `data/nsa/HackGTHearsayTesting/<name>`, itw via `itw_path`; `score = sigmoid(logit)`; a header comment-free CSV plus a sidecar `.meta.json` noting the transductive Platt scale (rank use only).
- `registry --build` / `--check`: `benchmarks/v1.yaml` (JSON content, valid YAML 1.2, stdlib-parsed; no pyproject change).
- `score --file --name [--itw] [--holdout]`, `rule --rule <json>`; output `outputs/v2/benchmark/<name>.json` (refuse an existing name without `--force`; atomic write; records lane, input shas, registry sha) + a markdown row.
- Threads: sets `OMP/MKL/OPENBLAS_NUM_THREADS` ≤ 4 before importing numpy.

### Step 3. `benchmarks/v1.yaml`
Run order: `adapt-l20`, then `registry --build`, then `registry --check`. Populations, roles, row counts, sha256 of every input: the five detector exports, the three ITW overrides, the Spectra ITW file, `xlsr_L20.csv` (+ L20.npz, L20.joblib, rows.json), `emb_itw_*.npz`, `emb_mlaad.npz`, the `itw_stress_v3` embeddings manifest, `splits/nsa_test_durations.csv`, the model dirs used by the cohort job (shipped M1, M5, handcrafted, `models/m1_wavlm-large_L9_20260926-1906`; hashed file lists), `splits/nsa_folds.csv`, `outputs/manifests/itw_stress.csv`, `outputs/channel/m3_perturb.csv`, `outputs/channel/m3_mlaad.csv`, `models/fusion_v2/constants.json`, the shipped TSV. Deferred populations listed with reasons (MLAAD by family, ASVspoof 5, reverb/codec/telephony/short-crop cohorts, sealed set; actDCF → V2-61). Cohort score files for the option-B columns are declared `pending` in v1; when Step 6 lands them the registry becomes **v1.1** (spec: any change bumps the version) and the Part 1 scorecards are regenerated under v1.1.

### Step 4. `tests/test_benchmark.py`
All S1–S8 tests and the L tests that need no fusion code (the four that do — `test_alpha_selection_never_sees_outer_fold`, `test_c2_fit_inputs_are_train_folds_only`, `test_spectra_refused_as_fit_feature`, `test_run_refuses_if_candidates_differ_from_spec_hash` — live in `tests/test_v2_fusion.py`, Step 10; Part 1 acceptance excludes them), hermetic on a seeded synthetic world (pattern of `tests/test_fuse_sweep_v3.py:481-525`, injected roots instead of monkeypatched globals), data tests with their own `skipif` on file existence (no conftest exists). Added from exploration:
`test_inner_inner_refs_exclude_outer_and_inner_fold` (poison fold g, its ranks must not move), `test_pooled_metrics_scored_on_pre_platt_base`, `test_lj_stratum_always_present`, `test_reference_rules_exempt_from_ledger`, `test_ledger_written_before_holdout_metrics`, `test_tracked_ledger_unchanged_by_tests`, `test_constants_rank_refs_equal_sorted_inner_logits_exact`, `test_l20_adapter_rows_json_order_and_basename_prefix`, `test_option_b_candidates_report_not_scored_where_columns_missing`, `test_cohort_reader_ignores_precomputed_fused`, `test_cohort_deltas_vs_own_none_kind`, `test_c0_reproduces_shipped_tsv_on_test` (data, 1e-12), `test_shipped_tsv_resolved_from_main_checkout`, `test_score_name_collision_refused`, `test_import_hygiene_benchmark` (no torch/lightgbm), `test_reference_exemption_limited_to_spec_refs`, `test_ledger_env_override_only_under_pytest`, `test_ledger_concurrent_reservation_one_wins` (two subprocesses), `test_ledger_version_bump_cannot_mint_new_read`, `test_ledger_regeneration_reuses_persisted_readout`, `test_verdict_integer_arithmetic_exact_band_edges`, `test_itw_override_schema_accepts_path_logit_and_stress`, `test_k7_sensitivity_row_and_degenerate_folds_marked`, `test_rerun_bit_identical` (two subprocesses, different PYTHONHASHSEED).

### Step 5. Reproduce and write the v1 scorecard
- `uv run pytest -q tests/test_benchmark.py` then `-m needs_data`.
- Score each v1 column (m1b_v3, handcrafted_v5, m5_xlsr_ft, spectra_aasist, wavlm_l, xlsr_L20) and SHIP (transductive and nested) on every population where they exist. Holdout: m1b_v3 and SHIP under the two-hash `reference` allowlist; the other columns as the one ledgered "benchmark-v1 baseline" set.
- Write `docs/reports/2026-10-11_benchmark-v1.md` (one h1; links only to tracked files; backticked paths otherwise).
- Commit Part 1 (explicit paths). **End-of-Part-1 report to Nathan.**

## Step 6. Cohort scoring (`scripts/v2_cohort_scores.py`) — ask Nathan first
Pre-check `pgrep -fl "extract_embeddings|repair_decode_errors|score_spectra|m5_score|m3_probes|run_pipeline|bakeoff_layers|python.*v2_"` (PID 18825, another project's pytest, matched on Oct 11). One MPS job at a time; stop by PID only; threads ≤ 4 (enforced in-script; `pipeline.set_cpu_threads` caps at 6).
- **A. ITW cohort, XLS-R columns (CPU, seconds):** rows = `bakeoff_layers.cohort_rows()`; `emb_itw_{kind}.npz` rounded to fp16; L20 = a·decision(e[:,20]) + b; m1b via `Probe.llr` (layer 7). Parity on `none` vs stored scores.
- **Pilot first (every column):** 20 rows of kind `none`, compared with the stored clean scores. Tolerances: same-device recipes max |Δlogit| ≤ 0.02; MPS vs CPU ≤ 0.05; Spearman ≥ 0.999. A failed pilot aborts that column before the full run and is reported to Nathan (the `_itw_m5.csv` recipe is undocumented, "crop-fair"; if M5 parity fails, its ITW cohort is reported as not scored rather than guessed). Pilot results go in the meta JSON.
- **B. ITW cohort, hc_v5 / M5 / Spectra (CPU ~35 min):** `draw = test_duration_sampler(200); crop = [draw() for _ in range(3000)]`, in the order of `outputs/embeddings/wav2vec2-xls-r-300m/itw_stress_v3/manifest.csv` (the order `cohort_rows` indexes); `x = m3_probes.perturb(load_audio(path), kind, r)`; `seg = prepare_segment(x, crop[r], 200+r)`. Spectra: `synth_logit(score_clip(models.spectra, seg, ("zero",), 16)["zero"])`. M5: `m5_logit(seg)`. Handcrafted: `features(x, crop_s=crop[r], seed=200+r, ...)` + `predictor(bundle)` (never `.run()`).
- **C. ITW cohort, WavLM (MPS ~12 min):** same `x`/`seg`; `load_backbone("wavlm-large","mps")`, fp16, `Probe.llr` (layer 9). Parity vs `wavlm_l.csv` itw.
- **D. Holdout cohort, L20 + L7 parity (MPS ~10 min):** `mp.TEST_DURS = mp._test_durs()` first; `rows = mp.holdout_rows(500)`; `x = perturb(segment(path, seed), kind, seed)`; then the M1 preprocessing `embed_segment(bb, prepare_segment(x))` and fp16 rounding, as `Models.m1_logit`; full-depth XLS-R (not truncated); L7 vs `m3_perturb.csv` m1b_v3 within the MPS-vs-CPU tolerance.
- **E. Holdout cohort, WavLM (MPS ~10 min):** same inputs and preprocessing as D, WavLM backbone, layer-9 probe.
- Output one file per (cohort, column), matching Interface v1's cohort contract (key path + kind): `outputs/v2/benchmark/cohorts/{itw,holdout}/<column>.csv` with columns path, kind, logit, error; plus a meta JSON per cohort (model ids, shas, seeds, thread count, pilot parity). A test runs the real writer's output through `read_cohort_file`, including a missing kind. Refuse zero-audio rows and partial cohorts. Then re-pin in `benchmarks/v1.yaml`.
- Tests (S8) in `tests/test_benchmark.py`, hermetic with a fake scorer.

## Part 2: the fusion test

### Step 7. Verify the pre-declaration
The candidates JSON was committed at Step 0.5. A test parses the frozen spec region and asserts the JSON's candidates, bands and guard match it.

### Step 8. `src/hearsay/fusion_v2.py` (new, pure)
`sample_weights(y, prior_fake=0.8)` (`common.py:124-134` generalized), `fit_nnlr` (`common.py:143-156` verbatim plus `success`/`nit`), `shrink` (K4), `inner_inner_select_alpha` (K2, K3), `run_c2_nested` (per-fold α, weights, effective params, in-sample vs OOF gap, `alpha_at_grid_max`; no Platt), `fit_c2_offinner` (K5), `refuse_fit_feature` (Spectra), `verdict(c0, cand)` (bands + f4 guard, nested OOF only, rounds to 4 decimals before comparing as `fuse_sweep_v3.py:159-161`), `followon_allowed`, `check_candidates(json, spec)`.

### Step 9. `scripts/v2_fusion.py`
1. Refuse unless `check_candidates` passes and the registry checks.
2. Nested inner readouts for C0–C3 on identical rows: pooled, per fold, macro, all three readings + EER, with and without E, E-fire counts per fold, weight shares, dominance, paired bootstrap vs C0 (K7), C2 gap and α diagnostics.
3. Verdicts per candidate (inner only).
4. One holdout read for the candidate set (ledger), then ITW, cohorts (where columns exist), MLAAD (in-sample label for M5 rules), test agreement vs the shipped TSV: report-only.
5. Write `outputs/v2/fusion/v2_60_readout.json`; append one `experiments/log.csv` row per candidate (`validation_score` = package-0.5 nested inner; notes = brief, package-0.3, EER, basis 16,142, verdict).
6. Larger-model follow-on: only if C2 = WIN (`followon_allowed`); otherwise recorded as not triggered.

### Step 10. `tests/test_v2_fusion.py`
The spec's O and F tests plus `test_alpha_selected_on_package05_pre_e` (fixture where pre-E and post-E selection pick different α; the pre-E choice must win), `test_alpha_tie_breaks_to_larger_alpha`, `test_alpha_at_grid_edge_flagged`, `test_shrink_applied_after_normalization`, `test_zero_sum_fit_weights_raises`, `test_c2_offinner_model_fit_on_all_inner_only`, `test_spec_hash_ignores_results_section_edits`, `test_candidates_json_matches_frozen_spec_region`, `test_verdict_refuses_stale_scorecard_hashes`, `test_alpha_invariant_to_spectra_column` (permute Spectra logits; α unchanged), `test_v2_fusion_refuses_holdout_for_rule_not_in_json`, `test_candidates_json_must_be_committed`, `test_c2_has_no_platt_step`, and the four fusion-dependent L tests moved from Step 4. Plus a data test that reproduces the risk agent's preview within 5e-4 (C0 0.3390 exact; C1 ≈ 0.452; C3 ≈ 0.358) as an independent cross-check.

### Step 11. Reports and docs
- `docs/reports/<run date>_v2-fusion-verdict.md` (exactly one h1; links only to tracked existing files, other paths backticked; unique title): verdict table (no Platt, K1), the K1 ruling and its timing, the early-look disclosure, the disclosure that the option-B columns' holdout and ITW were already read in the bake-off, α degeneracy (α at grid max in every fold expected), E-fire counts, gaps, the basis caveat (WavLM and L20 trained on the 23,790 basis with ASVspoof 2019 train/dev; shipped columns not), sealed-population decision for Nathan (options from the handoff).
- Every benchmark and verdict report carries a limitations block: folds are generator-held-out but not wholly speaker-disjoint (LJ chapters cross folds in `scripts/make_folds.py`; generators share target voices); and a "not evaluated" list: telephony, reverb, codecs beyond mp3, short crops, physical replay, ASVspoof 5, MLAAD by family, plus the duration, level, silence, container and single-speaker shortcut limits from v1.
- **No submission.** v2 writes no TSV and never touches `submissions/log.csv` (handoff shared rules). The blocker to any adoption is the missing sealed population; `test_c0_reproduces_shipped_tsv_on_test` keeps the shipped TSV reproducible.
- `docs/prod-chunks.md` V2-01 (lines 352–370: status, deferred list, actDCF → V2-61, deliverables `src/hearsay/benchmark.py` + `scripts/v2_benchmark.py`) and V2-60 (626–647: option B, candidates, verdict, status stays 🟡). Commit only if `git diff docs/prod-chunks.md` shows only these hunks (worktree branch: no other lane's hunks).
- `docs/code-map.md`: conditional; ask at execution (not in the lane's ownership).
- Run spec: Results and Test break check sections only.

### Step 12. Verify, commit, post-commit
- `uv run pytest -q tests/test_benchmark.py tests/test_v2_fusion.py`; `-m needs_data`; full `uv run pytest -q`; `uv run ruff check .`.
- Doc-sync sweep restricted to owned files (spec, plan, reports, `benchmarks/`, prod-chunks V2-01/V2-60); `docs/code-map.md` ask-first; commit (explicit paths; no stash).
- Break check: `test_nested_refs_exclude_scored_fold` vs `nested_ranks` (rank against all inner rows) in a detached worktree under the session scratchpad, removed with `git worktree remove --force` afterwards.
- Claude critique loop, then `.claude/review-audit.sh docs/specs/2026-10-11_v2-benchmark-fusion.md`.

## Risks and mitigations
| Risk | Mitigation |
|---|---|
| Pooled metrics after per-fold Platt | K1: no Platt in C2 (Nathan); `test_c2_has_no_platt_step` |
| Spectra influencing α | K3 pre-E selection; `test_alpha_invariant_to_spectra_column` |
| Holdout reads multiplying | K8 two-hash allowlist, JSON-only rules, cumulative count |
| Inner-inner refs include fold g | K2; poison test |
| α degenerate at grid edge | K3 flag; reported, not "fixed" (grid frozen) |
| Near-separable NNLR (‖w‖ 15–37) | shrink after normalization (K4); report ‖w‖, success |
| ITW cohort crop mismatch | Step 6 bakeoff order + parity checks on `none` |
| `m3_probes.TEST_DURS` default 3.4 s | set `_test_durs()` before any `segment()` |
| Holdout ledger vs reruns/tests | K8 reference exemption, tmp ledger in tests |
| Shared `outputs/` collisions / stale inputs | name-collision refusal, registry shas, stale-hash refusal |
| Shipped TSV only in main checkout | resolve via git common dir |
| `experiments/log.csv` merge conflict at merge to v2 | expected; resolve by union at merge time |
| pgrep false positive / memory pressure | ask Nathan; one job at a time; check `flag` |
