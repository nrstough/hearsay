# Handoff: v2 bigger-backbone lane (V2-53) (Sat Oct 10, 2026)

**Purpose of this chat:** find out whether a larger pretrained anti-deepfake backbone improves HEARSAY, and whether its numbers can be trusted. Go through `/plan-review` first; this is HIGH stakes and not started.

1. Inventory the candidates.
2. Run membership and contamination tests **before** believing any score.
3. Score what can be scored locally.
4. Plan frozen-probe or fine-tune runs. These need the drive, and possibly a GPU; ask Nathan before renting.

This is one of three parallel lanes started Oct 10. The others:
- benchmark + fusion: `docs/handoffs/2026-10-10_v2-fusion-benchmark-handoff.md`
- hard negatives: `docs/handoffs/2026-10-10_v2-hard-negatives-handoff.md`

The shared rules block below is identical in all three; read it.

## Context

**Where things stand.** v1 is frozen at tag `v1` (NSA minDCF 0.0733). The v2 plan of record is `docs/prod-chunks.md`. Read:
- C.1 rules 1–7
- B.2–B.4: how XLS-R and WavLM differ, and why Spectra is untrusted
- V2-53

**Why this lane exists** (`docs/reports/2026-10-10_v2-fusion-research.md`, "What other HEARSAY teams did"; web-verified, scores self-reported)
- **The winner, SideQuests** (minDCF 0.0317 / EER 1.44%; github.com/kevinharvey2025/HactGT13AudioAuthentication), used three fine-tuned AntiDeepfake backbones (XLS-R-2B, XLS-R-1B, MMS-1B), equal-weight z-mean plus Platt calibration, a test-matched input view, and 11,900 copy-synthesis fakes.
- **The backbone was most of the win.** The untuned 2B backbone scored ITW 0.038 under package-0.3, on their own 2,500-clip ITW sample (their `docs/EVALUATION.md`; saved copy at `outputs/v2/research/other-teams/files/kevinharvey2025_HactGT13AudioAuthentication__docs_EVALUATION.md`).
- **For comparison:** our M1b L7 on our 3,000-clip ITW scores 0.342 brief / 0.2955 package-0.5 / 0.193 package-0.3. The samples differ, so this is not like-for-like.

**Contamination: known or likely in-sample for AntiDeepfake** (arXiv 2506.21090; `nii-yamagishilab/xls-r-2b-anti-deepfake` and siblings)
- **Our NSA inner and holdout spoofs.** They are DiffSSD, and roughly 140 of DiffSSD's ~146 h of fakes are in its post-training (SideQuests `DATA.md`, saved under `outputs/v2/research/other-teams/`).
- **Local MLAAD.**
- **ASVspoof 2019 LA,** train and probably eval. Another team reports the protocol lists LA19 train and eval plus all of ASVspoof 2021 LA/DF (`outputs/v2/research/other-teams/readmes/MeagerPotato_hearsay-dh-squad.md`).
- **Not yet checked:** our real corpora (LJ Speech, LibriSpeech) and ITW, against the paper's data table. SideQuests claims ITW was held out.

So, as with Spectra (B.4), **inner and holdout readouts for these models are in-sample until proven otherwise.**

**A test-set note.** Two teams say the NSA test reals look VCTK / ASVspoof 2019-like. We verified that 98.0% of test lengths fall on a 512-sample grid at 22.05 kHz (research report). **Do not** hash-match NSA test clips against the local LA eval copy without Nathan's explicit decision.

**Our v1 backbones, for comparison.** Brief reading unless marked.

| Backbone | Inner (basis) | Holdout | ITW brief | ITW package-0.5 |
|---|---|---|---|---|
| XLS-R 300M frozen probe, L7 (M1b) | 0.2649 (23,790 incl. ASVspoof 2019) / 0.3013 (16,142 NSA) | 0.0717 | 0.342 | 0.2955 |
| WavLM Large L9 probe | 0.2936 (23,790) | 0.0717 | 0.752 | 0.383 |
| Spectra-AASIST (likely memorized) | 0.0045 (16,142) | 0.012 | 0.065 | |

Sources: `outputs/v2/bakeoff/report.json`, `docs/reports/2026-09-26_post-draft-heads.md`, `docs/reports/2026-09-26_m3-spectra.md`.

## Pipeline state

- **Not started.** Run `/plan-review` from P1.
- **Stakes:** HIGH. It is a new scoring model and may cost money.
- **Owner's statement:** none.
- **Time box:** set one in P1 with Nathan.

**Stop and report, rather than push through, if:**
- (a) a checkpoint does not load under a strict key map
- (b) a model does not fit on MPS at fp16 with 8 s segments. No silent CPU fallback over thousands of clips without asking.
- (c) the membership tests cannot separate members from non-members on the ASVspoof 2019 positive control. **No benchmark verdict before (c) passes.**

## What to do next

1. **`/plan-review`, P1.** Agree the question with Nathan: is it "does a bigger backbone help", "can we trust it", or both? Contamination tests come first.
2. **Candidate inventory**, in the run spec.
   - **Candidates:**
     - AntiDeepfake **MMS-300M** (the Mac-feasible size)
     - AntiDeepfake XLS-R-1B, XLS-R-2B and MMS-1B (fine-tuned detectors that output scores directly)
     - raw `facebook/wav2vec2-xls-r-1b` / `-2b` as frozen probes
   - **For each, record:** licence (another team reports CC BY-NC-SA for the AntiDeepfake weights, `outputs/v2/research/other-teams/readmes/akashjainn_dispel.md`; verify against the HF cards), training-data list, parameter count, MPS memory at fp16 with 8 s segments, and download size. Also record third-party zero-shot readings from the same file.
   - **Loader.** AntiDeepfake checkpoints are **fairseq** files, and fairseq 0.12 does not install on Python 3.12 (our pin). The loader therefore has to be a strict transformers port: every tensor mapped, shapes checked. SideQuests did this in their `hearsay/antideepfake.py`; see their `docs/APPROACH.md`, saved under `outputs/v2/research/other-teams/files/`. Put it in `src/hearsay/antideepfake*.py`. Raw `facebook/wav2vec2-xls-r-1b` / `-2b` load through `hearsay.embed.load_backbone` (`AutoModel`) as they are. Do not `uv add` anything without Nathan's OK.
3. **Membership and contamination tests** (pre-declared):
   - **Positive control:** ASVspoof 2019 LA, where membership is known. Read only, and never fit on LA eval.
   - **Matched-condition comparison:** training-corpus clips vs unseen clips from the same conditions.
   - **Temporal holdout:** MLAAD models released after its training cutoff (release dates are unverified so far).
   - Check LJ Speech, LibriSpeech and ITW against the paper's data table.
4. **Local scoring, no drive needed.** Off-the-shelf detectors can score ITW, the NSA test set and MLAAD from local audio.
   - Mirror our band-matched input path (`hearsay.embed.prepare_segment`), and record both its native view and ours.
   - Write per-clip scores to `outputs/v2/backbone/<name>.csv` in the shared contract shape.
   - Every readout is "possibly in-sample" until step 3 passes.
5. **Training-side work needs the drive** (frozen probes on new backbones).
   - **Extraction:** use `scripts/extract_embeddings.py` with the flags recorded in each `outputs/embeddings/wav2vec2-xls-r-300m/*_v3/extract_meta.json` (commands in `docs/handoffs/2026-09-26_post-draft-heads-handoff.md`; swap `--model` and `--name`). Output lands in `outputs/embeddings/<model>/`.
   - **Probe:** **do not** run `train_probe.py` or `export_probe_scores.py`. Write `scripts/v2_backbone_probe.py` on `hearsay.probe`, writing only under `outputs/v2/backbone/`, with layer selection on package-0.5 pre-declared.
   - **Cost:**
     - Time: a 20k-row set took about 72 min of MPS at 0.215 s/clip for WavLM Large (`docs/reports/2026-09-26_post-draft-heads.md`, Timing). Expect several times that for 1B/2B.
     - Disk: all-layer fp16 sets are about 1 GB per 20k rows at 300M, roughly 2.5 GB at 1B and 3.8 GB at 2B, plus the weights. Budget it before asking Nathan for downloads.
   - **Verification:** `uv run python scripts/repair_decode_errors.py --model <weights dir> --name <set> --manifest outputs/manifests/<m>.csv --verify-only --twin outputs/embeddings/wav2vec2-xls-r-300m/<set>_v3`. It is an MPS job, and works only for AutoModel-loadable backbones.
   - **Fine-tunes** need a GPU plan and Nathan's approval. v1's M5 lesson: full fine-tunes overfit the generators they saw (`docs/reports/2026-09-26_m5-xlsr-finetune.md`).
6. **Verdict report:** column, suppressor only, or drop.
   - Include the benchmark row under the three readings, once the benchmark lane's harness exists. Until then, use bake-off-style readouts. `scripts/bakeoff_layers.py` is a template only: XLS-R 300M is hard-coded, and it has no package-0.3 reading.
   - Per rule 7, "column" is provisional until a sealed population outside AntiDeepfake's post-training list exists.
   - Then update the V2-53 status.

## Environment / setup

```bash
cd ~/Projects/hearsay && git switch v2 && uv sync
uv run pytest -q && uv run ruff check .
ls weights/        # local: wav2vec2-xls-r-300m, wavlm-large, wavlm-base, Spectra-AASIST, spkrec-ecapa-voxceleb
df -h ~            # free space before any download
tar czf ~/hearsay-v2-research-2026-10-10.tgz outputs/v2/research   # if not already backed up
```

## Pointers

- `docs/prod-chunks.md`: B.2–B.4, C.1, V2-50, V2-51, V2-53.
- `docs/reports/2026-10-10_v2-fusion-research.md`: other teams, Spectra treatment, data shortlist.
- `outputs/v2/research/other-teams/`: saved copies of other teams' READMEs and docs.
- `docs/reports/2026-09-26_m3-spectra.md`, `src/hearsay/spectra.py`, `scripts/score_spectra.py`: how v1 ran an off-the-shelf detector, with pad-mode and direction checks.
- `scripts/extract_embeddings.py`, `src/hearsay/embed.py`, `src/hearsay/probe.py`.

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
