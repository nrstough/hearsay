# Handoff — post-draft heads lane: WavLM Large probe (Sat Sep 26, 2026, ~17:25; revised after the Fable second opinion)

**Purpose of this chat:** produce one candidate score column for the gate lane by ~18:50: a frozen **WavLM Large** probe on the exact M1b recipe, exported as `wavlm_l`. You own the Mac's GPU (MPS) for this window. You do not change fusion, the runner, the TSV or any shipped file.

**What changed since the first version of this handoff:** the XLS-R layers 5–9 probe is **dropped**. An independent Fable agent ran it from the on-disk embeddings (`docs/consults/2026-09-26_post-draft-review_RESPONSE_fable.md`, option 10): with the layer-7 baseline reproducing M1b exactly (inner 0.2649, holdout 0.0717, In-the-Wild 0.342), the mean of layers 5–9 gives holdout 0.052 but In-the-Wild 0.368 with the false-alarm rate doubling (0.8% → 1.55%); the concatenations of 6–8, 5/7/9 and 5–9 give In-the-Wild 0.353 / 0.430 / 0.399. Every head on the same backbone inherits M1b's blind spot: of the 158 In-the-Wild misses under the shipped rule, M1b would flag 1, M5 33, Spectra 107. Only a different backbone can see the miss tail, which is why WavLM is the one head experiment left.

## Context

- NSA's draft review returned **minDCF 0.0733, EER 3.534%** on the shipped rule (A3 w0.2 + E). Our proxies: holdout 0.0065, In-the-Wild 0.228. One team is ahead at 0.0584 (0.95 of 60 points). Both outside opinions (`docs/consults/2026-09-26_post-draft-review_RESPONSE.md`, read "What to do" and "Round 2") put WavLM's expected test effect near zero with a real mechanism behind it; a clean negative result is graded README content, so run it properly and report either way.
- **Pre-declared fusion entry** (fixed by the gate lane, not yours): `0.4·rank(M1b v3) + 0.2·rank(wavlm_l) + 0.2·rank(handcrafted v5) + 0.2·rank(M5)`, then the unchanged M3 step. One candidate. You produce the column; the gate lane judges it. The alternative "refit seat" (WavLM replacing M1b via `fuse_sweep_m5.py --refit-m1b`) is **not** run tonight.
- **M1b v3 recipe you must mirror exactly** (`models/m1_wav2vec2-xls-r-300m_L7_20260926-0521/meta.json`): segment mode, band match, silence trim, test-length crops (seed 0 for training rows, seed 200 for In-the-Wild), 8 s cap, per-input normalization; time-mean per layer; train sets `nsa_train_sample_v3,asv19_addon_v3` with folds `splits/nsa_folds_plus_asv19.csv` (ASV19 rows inner-only); layer chosen by inner CV; StandardScaler + class-balanced logistic (C = 1) + Platt, all fold-local; holdout read once; stress set `itw_stress_v3` never fit on.
- **Repo facts checked:** `scripts/extract_embeddings.py --model <dir under weights/>` loads through `transformers.AutoModel` with `output_hidden_states=True`, so `wavlm-large` (24 layers, 25 hidden states, 1024-d, same as XLS-R 300M) needs no code change. Each v3 set's `extract_meta.json` under `outputs/embeddings/wav2vec2-xls-r-300m/` records the flags used; copy them.

## Working branch / worktree

`main` in `~/Projects/hearsay`, shared by several chats. Uncommitted files belong to other lanes; leave them. Your files: `outputs/embeddings/wavlm-large/*` and one `models/m1_wavlm-large_L*_<stamp>/` (both gitignored), `outputs/detector_scores/wavlm_l.csv` (gitignored), and the tracked report `docs/reports/2026-09-26_post-draft-heads.md`. No source changes are expected; if one is needed, keep it to `scripts/train_probe.py` / `scripts/export_probe_scores.py` and add a test. Stage only your files (`git add <paths>`), never `git add -A`; do not push; tell the gate chat the commit hash.

## Environment / setup

```bash
cd ~/Projects/hearsay && uv sync
pgrep -fl "extract_embeddings|score_spectra|m5_score|run_pipeline"   # the GPU must be free before you start
```

## What to do next

1. **Extraction (MPS, ~40 min total; ~55 under load).** Mirror each v3 set's `extract_meta.json` flags; the expected commands are:
   ```bash
   uv run python scripts/extract_embeddings.py --model wavlm-large --manifest outputs/manifests/nsa_train_sample.csv --name nsa_train_sample_wl --segment --crop test --seed 0
   uv run python scripts/extract_embeddings.py --model wavlm-large --manifest outputs/manifests/asv19_addon.csv --name asv19_addon_wl --segment --crop test --seed 0
   uv run python scripts/extract_embeddings.py --model wavlm-large --manifest outputs/manifests/nsa_test.csv --name nsa_test_wl --segment
   uv run python scripts/extract_embeddings.py --model wavlm-large --manifest outputs/manifests/itw_stress.csv --name itw_stress_wl --segment --crop test --seed 200
   ```
   Check `asv19_addon_v3/extract_meta.json` before the second line and use whatever crop flags it records. Row counts must be 20,000 / 7,648 / 1,671 / 3,000. Run the training-sample extraction first; if it is not done by 18:10, skip the ASV19 add-on and train NSA-only (say so: it is then an M1-style probe, compared by the gate against M1 v3: holdout 0.159, In-the-Wild 0.380).
2. **Probe.** `uv run python scripts/train_probe.py --model wavlm-large --train nsa_train_sample_wl,asv19_addon_wl --folds splits/nsa_folds_plus_asv19.csv --stress itw_stress_wl`. Layer by inner CV only. Record the CV-by-layer table.
3. **Export.** `uv run python scripts/export_probe_scores.py --probe models/<the new dir> --name wavlm_l --train nsa_train_sample_wl,asv19_addon_wl --folds splits/nsa_folds_plus_asv19.csv --test nsa_test_wl --stress itw_stress_wl` → `outputs/detector_scores/wavlm_l.csv` with splits inner_oof 16,142 / holdout 3,858 / test 1,671 / itw 3,000 (the shape of `m1b_v3.csv`).
4. **Report** `docs/reports/2026-09-26_post-draft-heads.md`: recipe, CV table, holdout and In-the-Wild readouts under both costs (the trainer prints them), Spearman of `wavlm_l` against `m1b_v3` on holdout and test (a clone of M1b's ranks is the failure mode the gate checks; M1b vs Spectra is 0.57 on the test set), and how many of the shipped rule's 158 In-the-Wild misses WavLM alone would flag at its inner threshold (M1b 1, M5 33, Spectra 107: that count is the mechanism diagnostic). **Do not** compute fused numbers; the gate chat does that.
5. Message the gate chat ("post-draft gate") when the export lands, with the commit hash. Hard stop 18:50: if the export is not there, report what was measured and stop.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test: `uv run pytest -q` → all passing (510 at cc6670c with data present); `uv run ruff check .` → clean. Only needed if you touch a script.
- Reference numbers the recipe must sit beside: M1b v3 inner CV 0.2649 (folds include the ASV19 rows), holdout 0.072 / EER 1.4%, In-the-Wild 0.342 (averse 0.296), In-the-Wild real P_FA at the inner threshold 0.8%.
- At-risk: `outputs/embeddings/wavlm-large/*` (about 1 GB per 20k-row set, gitignored, ~40 min to regenerate; NOT archived), the new probe dir (gitignored; NOT archived), the export (regenerable from the probe in a minute). Copy the probe's `meta.json` numbers into the report so they survive a cleanup.
- In flight, other lanes: the gate chat (manifest, sweep, packet at 19:30), the CPU chat (negative-result write-ups), the README chat, the docs-site lane. Nothing else should use the GPU tonight; if `pgrep` shows a job, ask the oversight chat before killing anything.

## Analytical notes

- Score direction: 1.0 = synthetic; `export_probe_scores.py` writes `logit = probe.llr`, increasing with synthetic likelihood; the trainer defines the label, so it cannot invert.
- Fold discipline: ASV19 rows inner-only; the holdout never trained on or selected on; In-the-Wild eval-only; selection by inner CV only.
- The gate the column must clear after fusion: In-the-Wild ≥ 0.020 better under the brief's cost and ≤ 0.002 worse under the sponsor's, inner ≥ 0.010 better, holdout ≤ 0.010 worse, plus the miss-tail diagnostic above. For the column alone, the interesting readouts are In-the-Wild below M1b's 0.343 and a test-set rank correlation with M1b well below 1.

## Pointers

- `docs/consults/2026-09-26_post-draft-review_RESPONSE.md` (plan, gate, Round 2), `…_RESPONSE_fable.md` (the layer-probe negative result and the miss-tail counts), `…_CONSULTATION.md`.
- `docs/reports/2026-09-26_m1b-asv19-bonafide.md`, `src/hearsay/embed.py`, `src/hearsay/probe.py`, `scripts/{extract_embeddings,train_probe,export_probe_scores}.py`, `CLAUDE.md`.

**Correction (19:05):** the miss counts above (158; M1b 1, M5 33, Spectra 107) are at the In-the-Wild argmin. The gate's diagnostic uses CURRENT's inner-OOF threshold: 122 misses, M5 19, Spectra 79 (`b877760`). Report both, and use the inner-threshold count as the one the gate reads.
