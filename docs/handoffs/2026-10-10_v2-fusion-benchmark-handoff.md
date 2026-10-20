# Handoff: v2 benchmark + fusion lane (V2-01-lite, V2-60) (Sat Oct 10, 2026)

**Purpose of this chat:** build the shared v2 benchmark harness first, then run a pre-declared V2-60 fusion test. Use `/plan-review` for both.
- **The benchmark (V2-01-lite)** scores any per-clip score file or fused rule on every population we hold, under the three metric readings, with nested rank references and a cluster bootstrap. The other two lanes will call it.
- **The fusion test (V2-60)** has at most 4 candidates. Under option A (recommended, not yet chosen by Nathan) it uses the four shipped columns only, with Spectra outside every fitted parameter.

This is one of three parallel lanes started Oct 10. The others:
- hard negatives: `docs/handoffs/2026-10-10_v2-hard-negatives-handoff.md`
- bigger backbones: `docs/handoffs/2026-10-10_v2-backbone-handoff.md`

The shared rules block below is identical in all three; read it.

## Context

**Where things stand.** v1 is frozen at git tag `v1`. NSA scored it minDCF 0.0733 / EER 3.53%; the winner, SideQuests, scored 0.0317 (self-reported). The v2 plan of record is `docs/prod-chunks.md`: read C.1, rules 1–7.

**Research behind this lane:** `docs/reports/2026-10-10_v2-fusion-research.md` (eight researchers plus adversarial verifiers). Its verdicts:
- **Option A:** m1b_v3, handcrafted_v5 and m5_xlsr_ft, plus Spectra only as the frozen E step.
- **Option B is out.** WavLM and XLS-R L20 are blind on the ElevenLabs fold (brief 0.991 / 0.985). C3 (L20) would also need the drive for holdout-cohort scores.
- **No condition gate.** Inner rows hold no noisy real speech, so a gate collapses to constant weights.
- **Spectra must never be fitted.** Free fits hand it 54–88% of the weight.
- **High-capacity learners lose:** MLPs, MoE, unconstrained LR and condition interactions.

**On the selection metric, fusion is expected to lose or tie, not win.** These are nested inner numbers from the research table, 16,142 rows. The research report calls package-0.5 "averse".

| Candidate | brief | package-0.5 |
|---|---|---|
| SHIP + E (C0) | 0.1558 | **0.3390** |
| EQ3 + E (C1, rank domain) | 0.1541 | 0.4593 |
| Equal z-logit + E | 0.1633 | 0.4138 |
| NNLR z + E | 0.1625 | 0.3736 |
| NNLR ranks + E | 0.1629 | 0.3599 |

- **P(better than SHIP)** is ≤ 0.42 brief and ≤ 0.28 package-0.5. The research tie bands were 0.02 brief and 0.05 package-0.5. P2 sets the package-0.3 band.
- **The learned fits do help f4** (ElevenLabs): 0.03–0.05 vs 0.11 brief. That is the one place they win.
- **Measuring this is the point.** A clean "fusion can't beat SHIP until there are new columns or data" is the expected and useful outcome.

**Recommended candidates.** Finalize them in P2 with Nathan.

| ID | Candidate |
|---|---|
| **C0** | SHIP: rank blend 0.6·m1b_v3 + 0.2·handcrafted_v5 + 0.2·m5_xlsr_ft (weights from `models/fusion_v2/constants.json`), then E on the **blended rank** (`e_rule`: m3_logit_below −3.0, base_rank_above 0.5, ×0.5), then Platt. Read with nested (train-fold) rank references. |
| **C1** | Fixed equal weights over the same three, then E. Pick the score domain (rank or z) in advance. |
| **C2** | Non-negative LR on train-fold z-logits. Cost-weighted at the primary metric's effective prior: π_eff = 0.5·4 / (0.5·4 + 0.5·1) = **0.8** for package-0.5, not the 0.097 the research fits used. Shrunk toward a pre-declared target, α by inner-inner LOFO from {0, .25, .5, .75}, then E, then Platt. **Pre-declare where E's "base > 0.5" is measured** for a learned output, e.g. on the train-fold rank of C2's output, before Platt. |
| C3 (optional, better deferred) | Option B at a fixed small L20 weight. Its f4 guard (≤ SHIP + 0.03) must be restated on the primary reading. |

**Open P2 decision: handcrafted_v5.** "Shrink toward 0.2" conflicts with "cap and allow zero". Its test-set Spearman with the deep columns is only 0.10–0.25.

## Pipeline state (`/plan-review`)

- **Phase:** P1 was drafted only in the parent chat and is **not saved anywhere and not frozen**. Redraft it from this handoff and the research report, with the stakes line first.
- **Stakes:** HIGH. It changes how the final synthetic probability is computed.
- **Owner's statement:** none recorded. Run the P1 freeze gate with Nathan: one sentence on what this step does, and the most likely way it is wrong. Also get his option A vs B decision.
- **Break check:** not yet run.

## What to do next

1. **`/plan-review` for V2-01-lite, then V2-60.** Use separate run specs, or one spec with two parts. Set a time box in P1 (suggest ≤ 1 day per part). When it runs out, stop and report.
2. **V2-01-lite = V2-01 restricted to populations that exist today.**
   - **Deferred:** MLAAD held out by family, ASVspoof 5 eval, reverb/codec/telephony/short-crop cohorts, the sealed set, actDCF (to V2-61; say so in the spec).
   - **Deliverables:**
     - `src/hearsay/benchmark.py` + `scripts/v2_benchmark.py`, replacing V2-01's `scripts/benchmark.py`; update V2-01's deliverables line in `prod-chunks`
     - a tracked `benchmarks/v1.yaml` with population row lists and sha256s of every input file
     - `docs/reports/<date>_benchmark-v1.md`
   - **Publish the input contract and API in the run spec early.** The other lanes are waiting on it.

   | Population | Source | Notes |
   |---|---|---|
   | Nested inner OOF | 16,142 NSA rows (`splits/nsa_folds.csv`) | Pooled, per fold, macro. Rank references from training folds only. |
   | Outer holdout | 3,858 | Read once per pre-declared candidate set |
   | In-the-Wild | 3,000; labels and speakers in `outputs/manifests/itw_stress.csv` | Report-only, speaker-cluster bootstrap |
   | Holdout perturbation cohort | `outputs/channel/m3_perturb.csv` (5 kinds × 250 + 250, the four shipped columns) | |
   | ITW perturbation cohort | `outputs/v2/bakeoff/emb_itw_*.npz` | XLS-R only. See step 3. |
   | MLAAD | `outputs/channel/m3_mlaad.csv` | In-sample for M5: label it, never select on it |
   | NSA test | 1,671 | Share above the threshold; Spearman and flips vs SHIP (shipped TSV `submissions/CrossExam_predictions.tsv`, read only) |

   - **Column sources:** `outputs/detector_scores/{m1b_v3,handcrafted_v5,m5_xlsr_ft,spectra_aasist}.csv` (inner_oof 16,142 / holdout 3,858 / test 1,671; m1b_v3 also has itw 3,000).
   - **ITW for the others:** `outputs/detector_scores/_itw_handcrafted_v5.csv`, `outputs/detector_scores/_itw_m5.csv`, `outputs/spectra/itw_stress/scores.csv`.
   - **Loaders to reuse:** `scripts/fuse_sweep_v3.py` (`load_raw`, `resolve_itw`, `load_labels`, `build`, `validate_export`, `apply_tiers`, bootstrap).
   - **Every readout reports:** the three readings plus EER, with and without E, weight shares, and a dominance check (each column's Spearman with the fused score on real rows).
3. **ITW perturbation cohort for the other columns.** handcrafted_v5, M5 and Spectra need scoring on the same 500 rows × 5 kinds.
   - Rows come from `scripts/bakeoff_layers.py` `cohort_rows()`.
   - Perturbations come from `scripts/m3_probes.py` `perturb(x, kind, int(r))`.
   - Load the models through `hearsay.pipeline.Models(device="cpu", load_m5=True)` with threads ≤ 4, as `scripts/m3_probes.py` does.
   - Write the result with `scripts/v2_cohort_scores.py` to `outputs/v2/benchmark/itw_cohort_scores.csv`.
   - Estimate: about 35 min of CPU. Treat it like an MPS job (ask first), because loading M5 and Spectra adds memory pressure.
4. **Pre-declare the V2-60 candidates and rules in a run spec**, commit it, then run, report and commit. Update the V2-01 / V2-60 status in `prod-chunks`.
5. **Raise the sealed-population decision with Nathan.** Options:
   - the team records new real and fake clips
   - MLAAD models released after 2026-04-18 (new pulls)
   - Deepfake-Eval-2024 (evaluation-only licence)
   - NSA's per-clip cross-team results, if shared (see memory and the V2-01 note)

**Done when:**
- **V2-01-lite:** the harness reproduces every reference number below to 1e-4, its tests pass, and its interface is in the run spec.
- **V2-60 here:** the ≤ 4 pre-declared candidates have full readouts and a verdict report. V2-60 stays 🟡 because adoption waits for a sealed population (rule 7).

## Reference numbers the benchmark must reproduce

| What | Basis | brief | package-0.5 | Source |
|---|---|---|---|---|
| SHIP, transductive (all-inner ranks) | inner 16,142 | 0.1351 | 0.2997 | `outputs/v2/bakeoff/report.json` fusion.F(P0) |
| SHIP, transductive | holdout | 0.0065 | 0.0087 | same |
| SHIP, transductive | ITW | 0.228 | 0.2385 | same |
| SHIP, nested | inner 16,142 | 0.1558 | 0.3390 | research report §(4); `outputs/v2/research/inner-screening/` |
| m1b_v3 export | inner 16,142 | 0.3013 | 0.3307 | recomputed Oct 10 from `outputs/detector_scores/m1b_v3.csv` + `splits/nsa_folds.csv` (package-0.3: 0.228) |
| m1b_v3 export | holdout | 0.0717 | 0.0571 | same |
| m1b_v3 export | ITW | 0.342 | 0.2955 | same |

The bake-off's 0.2649 is on the 23,790-row basis. Don't use it as a benchmark reference.

## Environment / setup

```bash
cd ~/Projects/hearsay && git switch v2 && uv sync
uv run pytest -q && uv run ruff check .      # all passing, ruff clean, at the commit that added this handoff
tar czf ~/hearsay-v2-research-2026-10-10.tgz outputs/v2/research   # back up the non-regenerable research
```

## Pointers

- `docs/prod-chunks.md`: C.1, V2-01, V2-60, and B.7 (v1's brief reading, superseded for selection by rule 5).
- `docs/reports/2026-10-10_v2-fusion-research.md`: the whole report, especially the inner-screening table and "Claims that did not survive verification".
- `docs/specs/2026-10-03_layer-bakeoff.md`, `scripts/bakeoff_layers.py`: the pre-declaration template.
- `scripts/fuse_sweep.py`, `scripts/fuse_sweep_v3.py`, `models/fusion_v2/constants.json` (read only).
- `docs/learn/v1-explained.md`, lesson 11 (how v1's fusion works).

## Shared rules for the three v2 lanes (identical in all three handoffs)

### The metric (`docs/prod-chunks.md` C.1 rule 5, Nathan Oct 10: "It's the min DCF function")

**Primary: NSA's shipped minDCF, called "package-0.5".**
- Source: `data/nsa/HackGTMinDCF/asvspoof5/evaluation-package/calculate_metrics.py:16-21`, with Pspoof 0.5, Cmiss 1, Cfa 4.
- **The package names its costs from the bona fide side.** Its Cmiss is *our* false alarm (real called synthetic) and its Cfa is *our* miss.
- So for our 1.0 = synthetic scores it is **P_FA + 4·P_miss: misses cost 4×.**
- Code: `hearsay.metrics.min_cost(y, s, 0.5, c_fa=1.0, c_miss=4.0)`, identical to `sponsor_min_dcf(y, s, flip=True)` (checked to 2e-16, ties included). v1 scripts and the research report call it "averse".

**Always report two more readings:**
- **package-0.3:** `min_cost(y, s, 0.3, c_fa=1.0, c_miss=4.0)` = P_FA + 1.714·P_miss. Other teams say NSA scored with it.
- **brief:** `min_cost(y, s)` = 9.33·P_FA + P_miss. This is v1's reading, and every v1 report and the research pilots use it.

**Report EER beside them.** Any count of false alarms or misses must name the threshold it was taken at. The three readings put the threshold in very different places.

### Ownership

| Lane | Writes (create and edit) |
|---|---|
| Benchmark + fusion | `src/hearsay/benchmark.py`, `scripts/v2_benchmark.py`, `scripts/v2_fusion.py`, `scripts/v2_cohort_scores.py`, `tests/test_benchmark.py`, `tests/test_v2_fusion.py`, `benchmarks/`, `outputs/v2/benchmark/`, `outputs/v2/fusion/`; the V2-01 and V2-60 sections of `docs/prod-chunks.md` |
| Hard negatives | `scripts/v2_hardneg*.py`, `tests/test_hardneg*.py`, `outputs/v2/hardneg/` (including `outputs/v2/hardneg/emb/`); the V2-33 section |
| Backbones | `scripts/v2_backbone*.py`, `src/hearsay/antideepfake*.py` (only if a loader port is needed), `tests/test_backbone*.py`, `outputs/v2/backbone/`, `outputs/embeddings/<new backbone>/`, new dirs under `weights/`; the V2-53 section |

Each lane also writes its own specs, reports and handoffs under `docs/`, named with its lane slug.

### Read but never write

- `models/`, `submissions/`, `splits/`, `outputs/detector_scores/`, `outputs/embeddings/` (except the backbone lane's new sets), `outputs/v2/research/`, `outputs/v2/bakeoff/`, `outputs/v2/layer_diversity/`, and any other lane's files.
- **Do not run** `scripts/train_probe.py` or `scripts/export_probe_scores.py`. They write `models/` and `outputs/detector_scores/` and select layers on the brief reading. Build on `hearsay.probe` (`load_embeddings`, `make_clf`, `oof_scores`) the way `scripts/bakeoff_layers.py` does.
- **Do not edit** `src/hearsay/metrics.py` (read-only), `src/hearsay/augment.py`, `scripts/extract_embeddings.py` or the hackathon scripts.
- **`pyproject.toml` / `uv.lock`** change only with Nathan's OK, announced to the other lanes.

### Score files (the contract the benchmark reads)

- Per-clip scores go in a CSV shaped like `outputs/detector_scores/m1b_v3.csv`: columns `path,fold,split,score,logit`, with `split` ∈ {inner_oof, holdout, test, itw}. `logit` increases with synthetic likelihood.
- Write it under your lane's `outputs/v2/<lane>/`.
- **Paths:** the exports mix absolute and repo-relative paths, and `scripts/fuse_sweep.py` `norm()` follows the `data/` symlinks. So normalize both sides the same way and **assert full coverage after every join**. A silent row drop is the known trap.

### Inner basis

There are two inner populations. **State which one every inner number uses.**
- **23,790 rows:** 16,142 NSA rows plus 7,648 ASVspoof 2019 rows, from `splits/nsa_folds_plus_asv19.csv`. The bake-off and the research pilots use this one.
- **16,142 NSA-only rows**, from `splits/nsa_folds.csv`. The benchmark uses this one, as do the exports and fusion.

Reference values for M1b L7:

| Basis | Brief | package-0.5 | package-0.3 |
|---|---|---|---|
| 23,790 rows | 0.2649 | 0.3104 | |
| 16,142 rows (export) | 0.3013 | 0.3307 | 0.228 |

### Evaluation discipline

- **Select on inner OOF only.** Pre-declare candidates, pass rules and tie bands in a committed run spec before reading any other population. Template: `docs/specs/2026-10-03_layer-bakeoff.md`.
- **The outer holdout is read once** per pre-declared candidate set.
- **In-the-Wild and its perturbation cohorts are report-only** (rule 7: spent on v1 selection).
- **The local 572-clip MLAAD cohort is a readout only.** All of it is in M5's training manifest.
- **ASVspoof 2019 LA eval (A07–A19)** is barred from every fit, mining and readout path. That includes `outputs/embeddings/wav2vec2-xls-r-300m/asv19_eval_probe`.
- **Final adoption of anything** waits for a fresh sealed population (rule 7). The benchmark lane raises that decision with Nathan.

### Logging

- `experiments/log.csv` (created Oct 10, header only) gets one row per pre-declared run. Columns: `timestamp,rung,validation_score,validation_score_clean_only,csv_path,notes,benchmark_version`.
  - `validation_score` = package-0.5 on the population named in `notes`.
  - `notes` = brief, package-0.3, EER and the inner basis.
- Never write `submissions/log.csv`.
- The hackathon TSV / submission-log / Sep-27-freeze wording in `.claude/skills/plan-review/SKILL.md` does not apply to v2.

### Git

- All three lanes share one checkout on branch `v2`. Stage explicit paths only.
- **`docs/prod-chunks.md` is shared.** Edit only your sections, and commit it only when `git diff docs/prod-chunks.md` shows only your hunks. If another lane's hunks are pending, ask that lane or Nathan to commit first.
- **Never** run `git stash`, `git checkout .`, `git restore` or `git add -A` in the shared checkout.
- Never push without Nathan.
- **Not this lane's, leave alone:** the uncommitted `.claude/codex-audit-prompt.md`, `.claude/launch.json`, `docs/specs/2026-09-26_m5-xlsr-finetune-audit.md`, `next-env.d.ts` and `tsconfig.json`.

### Compute

- **CPU:** n_jobs ≤ 4 per lane on the 18 GB Mac. Note that `scripts/bakeoff_layers.py probes` hard-codes 6.
- **MPS:** one job on the whole Mac at a time.
  - Before any MPS or torch job, check `pgrep -fl "extract_embeddings|repair_decode_errors|score_spectra|m5_score|m3_probes|run_pipeline|bakeoff_layers|python.*v2_"`. If anything shows, ask Nathan first.
  - Memory pressure writes silent `decode_error` zero rows (CLAUDE.md), so check the manifest `flag` column after every extraction.
- **Processes:** stop them by explicit PID only, never with `pkill` patterns.
- **Downloads and GPU rental:** ask Nathan first. About 27–29 GiB was free on Oct 10. Every rental gets a row in `docs/reports/cloud-expense-ledger.md`.
- **The external Crucial drive is not mounted.** It holds the DiffSSD, LJ and LibriSpeech training audio. Only Nathan can mount it.

### At-risk, gitignored, NOT archived

- **`outputs/v2/research/` (48 MB): not regenerable from the repo.** The workflow lived in the parent chat. Back it up before starting: `tar czf ~/hearsay-v2-research-2026-10-10.tgz outputs/v2/research`.
- **`outputs/v2/bakeoff/`:**
  1. `uv run python scripts/bakeoff_layers.py extract` (MPS, ~13 min; ask first)
  2. `probes` (CPU)
  3. `report`
