# Handoff: v2 hard-negative mining lane (V2-33) (Sat Oct 10, 2026)

**Purpose of this chat:** run a pre-declared **stage-0 hard-example bake-off** on the saved frozen XLS-R embeddings (CPU only, no training audio), then plan stages 0.5 and 1. Go through `/plan-review` first; this is HIGH stakes and not started.
- **Stage 0.5:** a worst-op augmentation sweep on local ASVspoof 2019 train/dev audio. It needs MPS and Nathan's go-ahead.
- **Stage 1:** a bandit ("RL") over augmentation chains. It needs the external drive.

This is one of three parallel lanes started Oct 10. The others:
- benchmark + fusion: `docs/handoffs/2026-10-10_v2-fusion-benchmark-handoff.md`
- bigger backbones: `docs/handoffs/2026-10-10_v2-backbone-handoff.md`

The shared rules block below is identical in all three; read it.

## Context

**Where things stand.** v1 is frozen at tag `v1` (NSA minDCF 0.0733). The v2 plan of record is `docs/prod-chunks.md`: read C.1 rules 1–7 and V2-33.

**Organizer feedback:** other HEARSAY teams used "RL hard negative mining". No public recipe exists.

**Research:** `docs/reports/2026-10-10_v2-fusion-research.md`, section "Hard-negative mining and RL (V2-33)", with raw files in `outputs/v2/research/hard-negative-rl/`.
- "RL" for us most plausibly means a bandit (EXP3 or Thompson sampling) over augmentation/codec "arms", rewarded by detector error.
- The nearest published work, CRDA, is for faces, and its reward is miss-side. A false-alarm reward on augmented reals would be our own design.
- The literature prior is weak: spoof-emphasizing losses hurt on LA19 (CCE 1.39% → focal 1.67%, arXiv 2406.17246).

**What changed with rule 5.** The research pilots were scored on v1's brief reading. **Re-read on the primary metric (package-0.5, P_FA + 4·P_miss), the ranking changes.**

Source: re-derived Oct 10 from the saved OOFs (`outputs/v2/research/hard-negative-rl/{reweight,oneclass,groupweight}_oofs.npz`), with `inner_y` and `inner_fold` from `outputs/v2/bakeoff/rows.json`. Basis: **23,790 rows** (NSA plus ASVspoof 2019). "f4" here contains ElevenLabs plus ASVspoof 2019 A04 and 1,015 ASVspoof 2019 reals.

| Variant (pilot) | package-0.5 pooled | fold mean | f4 | brief pooled (old reading) |
|---|---|---|---|---|
| Base L7 | 0.3104 | 0.2040 | 0.3450 | 0.2649 |
| **Cost-weighted, reals ×4** | **0.2779** | **0.1995** | 0.3386 | 0.2705 |
| Focal γ = 2 | 0.3294 | 0.1984 | **0.3221** | 0.2502 |
| OHEM (in-sample margin, top 5% per class ×5) | 0.3139 | 0.2012 | 0.3348 | 0.2505 |
| Nested mine-and-retrain | 0.3065 | 0.2092 | 0.3479 | 0.2590 |
| Source-balanced weights | 0.2911 | 0.2066 | 0.3565 | 0.2633 |
| Speaker-balanced weights | 0.3330 | 0.2329 | 0.5059 | 0.2644 |
| OC-softmax | 0.5763 | 0.2354 | 0.3068 | 0.2299 |

**Which direction of mining pays under the new metric is unmeasured.** Reweighting reals ×4 wins pooled, while focal (fake-side emphasis) wins on f4 but loses pooled. **Stage 0 should test cost weighting in both directions** (reals ×k, fakes ×k).

**Hard-negative counts depend on the threshold.**
- At the brief threshold, L7 makes 61 false alarms out of 13,270 reals, on 17 of 103 speakers; the top 10 give 84% (`atlas.json`).
- At the package-0.5-optimal threshold the same OOF makes about **2,348 false alarms and 351 misses**. So "inner has few hard negatives" holds only under v1's reading.

**Inner false-alarm rate does not predict wild false-alarm rate.**
- Spearman 0.30 across 40 probes, at the brief threshold.
- The XLS-R and WavLM family medians are 0.41% vs 0.44% inner, but 1.2% vs 8.2% on ITW.
- The measured noise hole is a **miss** problem: the shipped holdout cohort goes 0.012 → 0.269 at 20 dB white noise (brief; `docs/reports/2026-09-26_channel-robustness.md`).

**Layer lead.** The L6+L20 and L8+L20 blends scored 0.447 and 0.425 vs L7's 0.485 on the 500-clip noise20 ITW cohort (brief, bake-off). Single L6 and L8 probes exist (`outputs/v2/bakeoff/probes/L06.npz`, `L08.npz`) but were never read under noise. Treat this as a lead.

## Pipeline state

- **Not started.** Run `/plan-review` from P1.
- **Stakes:** HIGH. It changes how a scoring model is trained.
- **Owner's statement:** none. Nathan gives his own-words sentences at each freeze gate.
- **P1 decision for Nathan:** may ASVspoof 2019 **train/dev** rows be mined and scored? This is open in the research report's Completeness check. CLAUDE.md describes them as "never scored or held out" in v1. Report stage 0 on both bases: 23,790 (comparable to the pilot) and the 16,142 NSA-only rows the benchmark uses.

## What to do next

1. **`/plan-review`, stage 0. Pre-declare everything.**
   - **Candidates:** base; cost weighting reals ×k and fakes ×k; focal; OHEM; nested mine-and-retrain; source-balanced weights. Run them on the L6, L7 and L8 probes.
   - **Selection rule:** per-fold mean on package-0.5 inner OOF, with f4 not worse than base and inner false alarms at the package-0.5-optimal pooled threshold ≤ base + a pre-declared band. Report the brief-threshold counts beside them.
   - **Guards vs readouts:**
     - ITW and the ITW perturbation cohort are report-only (rule 7).
     - The holdout (read once) and the MLAAD miss rate may be pre-declared guards with tie bands; P1 decides.
     - Test share and flips are reported.
   - **Mining rules:**
     - Mine only within inner training folds, nested: a fold never selects its own hard examples.
     - Assert no `decode_error` rows.
     - Never mine from ITW, holdout, test, the local MLAAD cohort, or LA eval (`asv19_eval_probe`).
     - Report the clean-only score beside every mined one.
   - **Never re-run `outputs/v2/research/hard-negative-rl/hn_*.py` in place.** They write into that directory and overwrite the pilot. Copy the logic into `scripts/v2_hardneg_*.py`, writing to `outputs/v2/hardneg/`.
   - **Time box: 3 h, then stop and report.** Stage 0 is done when the pre-declared table (candidates × L6/L7/L8 × three readings, both bases) is in `outputs/v2/hardneg/` and a report is in `docs/reports/`.
   - **What a stage-0 winner means:** a recommendation only. Adoption waits for a V2-01 benchmark row beside its unmined twin and for a sealed population.
2. **Plan stage 0.5** (ask Nathan before any MPS use). A worst-op sweep on local `data/asvspoof2019` **train/dev** audio:
   - **Ops:** noise, band-limit, reverb, codecs (local ffmpeg has AMR-NB, Opus, MP3, AAC and G.726), and speed. Speed is a guess: other teams' notes mention time-stretching, and our NSA documents only say "noise and other perturbations".
   - **Tooling:** write augmented embeddings under `outputs/v2/hardneg/emb/` with a lane script modeled on `scripts/bakeoff_layers.py` `stage_extract` (perturb in memory, then embed). Put new ops in `scripts/v2_hardneg_*.py`. Do not edit `src/hearsay/augment.py` or `scripts/extract_embeddings.py`.
   - **Requirements:**
     - Augment both classes equally.
     - Refit per fold under `splits/nsa_folds_plus_asv19.csv`. The ASVspoof 2019 bona fide rows are training rows for m1b_v3, L20, WavLM and M5, so measuring their noisy copies with the shipped models would be in-sample.
     - A shortcut gate: codec and noise features alone must give AUC < 0.6.
     - Afterwards, assert no `decode_error` flags or all-zero rows.
3. **Stage 1** (needs the drive, which Nathan mounts):
   - worst-of-K augmentation plus mine and retrain, with EXP3 or Thompson updates over augmentation arms
   - severity caps (SNR ≥ 5 dB)
   - **Downloads (ask Nathan first, then check `df -h ~`):** MUSAN (CC BY 4.0, 11 GB) and RIRS_NOISES (Apache-2.0, 1.3 GB). Neither is downloaded yet.
   - `weights/` has no vocoder for copy-synthesis.
4. **Stage 2** (GPU; ask Nathan before renting): an M5-style head with OHEM or focal loss plus the bandit. About $1–2 per configuration; ledger row.

**Expected failure modes** (each needs a check in P2):
- Inner hard reals don't transfer to wild false alarms.
- The model memorizes the few false-alarm speakers.
- It overfits 2019-era artifacts.
- Mining reals can't fix noise misses.
- It learns the shortcut "degraded audio means real".
- The search drifts to unrealistic transforms.
- The clean score regresses.
- Inner-only selection repeats the L7 + WavLM trap (`docs/reports/2026-10-03_layer-diversity.md`).

## Environment / setup

```bash
cd ~/Projects/hearsay && git switch v2 && uv sync
uv run pytest -q && uv run ruff check .
tar czf ~/hearsay-v2-research-2026-10-10.tgz outputs/v2/research   # if the benchmark lane hasn't already
ls outputs/embeddings/wav2vec2-xls-r-300m   # *_v3 sets: all 25 layers, decode_error-free (checked Oct 3); asv19_eval_probe is LA eval: barred
```

Re-derive the table above without touching the research directory:

```bash
uv run python -c "import json,numpy as np; from hearsay.metrics import min_cost; r=json.load(open('outputs/v2/bakeoff/rows.json')); y=np.array(r['inner_y']); f=np.array(r['inner_fold']).astype(int); d=np.load('outputs/v2/research/hard-negative-rl/reweight_oofs.npz'); [print(k, round(min_cost(y,d[k],0.5,c_fa=1.0,c_miss=4.0),4), round(min_cost(y[f==4],d[k][f==4],0.5,c_fa=1.0,c_miss=4.0),4)) for k in d.files]"
```

## Pointers

- `docs/prod-chunks.md`: C.1, V2-30, V2-31, V2-33.
- `docs/reports/2026-10-10_v2-fusion-research.md`: the hard-negative and data sections.
- `docs/reports/2026-10-03_layer-bakeoff.md`: readout template and the layer-blend noise numbers.
- `src/hearsay/probe.py`, `scripts/bakeoff_layers.py`, `scripts/m3_probes.py` (`perturb`), `src/hearsay/augment.py` (M5's ops; read only).
- CLAUDE.md gotchas: decode_error rows; never import LightGBM and torch in one process.

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
