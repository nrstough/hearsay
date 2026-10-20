# Handoff: V2-24, what sees the misses (mechanistic probe of the carrier's blind spot)

**Date:** 2026-10-20. **Chunk:** `docs/prod-chunks.md` → V2-24 (Phase 2). **Branch to start from:** `v2` (pushed, `origin/v2`). **Stakes:** medium (no model ships from this; it decides which lever the next lanes pull). **Recommended model:** Opus (design-heavy reading of ambiguous probe results).

## The question

The shipped v1 carrier (XLS-R 300M, frozen, layer-7 linear probe, `models/m1_wav2vec2-xls-r-300m_L7_20260926-0521`) misses 24 of 480 fakes on the NSA test key. Of those 24, Spectra-AASIST ranks 18 in its synthetic top-28.7%, WavLM Large (frozen layer-9 probe, `models/m1_wavlm-large_L9_20260926-1906`) ranks 16, and the carrier ranks 3 (`docs/reports/2026-10-20_v2-02-error-atlas.md` §4.2). The misses share a style: voiced fraction ~0.54 vs 0.71, speaker-drift min-cosine ~0.16 vs 0.27, wide-band, clean, not short (§4.1).

**Decide which of two statements is true:**
- (a) XLS-R already encodes what separates these clips, at some layer N or along some direction the L7 probe does not read. Then the fix is a probe change (layer, layer mix, or direction), testable in V2-20's bake-off form, cheap.
- (b) It does not, at any layer. Then the levers are data (V2-30) and new columns (V2-53), and no re-probing of XLS-R will help.

## Rules that bind this lane

- **The sealed key is not opened.** The 24 test-miss identities (filenames) come from the persisted Phase 0 readout `outputs/v2/benchmark/sealed/SHIP_tsv.json` (`readout.errors_at_decision.misses`) and may be used as identities only. No label of any other test clip is used; no number is computed on the test set that needs labels beyond those 24. `benchmarks/sealed_ledger.jsonl` must be unchanged at the end.
- **Statistics come from In-the-Wild and the inner folds**, not from 24 clips: the 158 ITW misses at the ITW-optimal threshold (`docs/reports/2026-10-03_layer-bakeoff.md`; recompute from `outputs/detector_scores/m1b_v3.csv` ITW rows + `outputs/manifests/itw_stress.csv`), matched sets of ITW fakes the carrier catches, and ITW reals. Inner folds for fitting probes (`splits/nsa_folds.csv`, folds 0–4), never the holdout.
- **Rule 7:** ITW is a readout population. Nothing is selected on it; this lane produces a verdict about mechanism, not a candidate.
- **Rule 5:** report package-0.3 primary, package-0.5 and brief beside it, where you report a metric at all; most of this lane's outputs are effect sizes and AUCs, which is fine.
- **MPS is one job at a time.** On the evening of Oct 20 the V2-53 chain (`outputs/v2/backbone/logs/chain_20261020.sh`) holds it until ~21:10; check `pgrep -fl "v2_backbone_score|extract_embeddings|bakeoff_layers"` before any torch job. Stored embeddings cover a lot of the work (below).
- No cloud GPU without asking Nathan. No time box; report at the end of each part.

## What exists to build on

- **Stored per-layer material:** the V2-20 layer bake-off (`docs/specs/2026-10-03_layer-bakeoff.md`, `docs/reports/2026-10-03_layer-bakeoff.md`, `scripts/bakeoff_layers.py`, `outputs/v2/bakeoff/`: per-layer probes, `rows.json`, `emb_itw_*.npz`, `probes/L20.npz`). Read its report first; it already found L7+L20 does not confirm as a blend. Reuse its extraction path rather than writing a new one.
- **XLS-R L7 embeddings** for every population: `outputs/embeddings/wav2vec2-xls-r-300m/{nsa_train_sample_v3,asv19_addon_v3,nsa_test_v3,itw_stress_v3}/` (`hearsay.probe.load_embeddings`; check the `flag` column for decode errors, see CLAUDE.md gotchas).
- **The probe itself:** `hearsay.probe` (fit/apply), the model dir above holds the weight vector; `hearsay.embed.prepare_segment` is the shared band-matched input path.
- **Per-file atlas features** for the test set and the atlas CSV: `outputs/v2/error_atlas.csv`, `outputs/runner/v2_full/results/<file>.json` (voiced fraction, drift, bandwidth, duration), `scripts/v2_error_atlas.py` for how they were joined.
- **Benchmark harness** if a probe variant is to be scored: `scripts/v2_benchmark.py score` (Interface v1 files; holdout reads are ledgered, do not take one in this lane).

## Plan (adjust, but keep the verdict shape)

1. **Define the sets.** M_test (24 miss identities), M_itw (ITW misses), C_itw (ITW fakes the carrier catches, matched on duration), R_itw (ITW reals), and inner folds for fitting. Record sizes.
2. **Layer sweep on XLS-R** (24 transformer layers, mean-pooled, same segment policy as M1b): fit a per-layer logistic probe on inner folds (nested, as `bakeoff_layers.py` does), read on M_itw vs R_itw and C_itw vs R_itw separately. The gap between the two curves, per layer, is the finding: a layer where M_itw separates from reals is statement (a).
3. **Same sweep on WavLM Large** (it catches them): which layers carry it, and is it the same depth as XLS-R's best layer for M_itw?
4. **Direction contrast at L7:** project M_itw, C_itw and R_itw L7 embeddings onto the shipped probe's weight vector w and onto the top principal directions of the complement. If M_itw separates from R_itw in the complement but not along w, the probe reads the wrong direction (a); if it separates nowhere at L7 but does at layer N, it is a layer question (a); if nowhere, (b).
5. **Input facts:** a shallow tree (depth ≤ 3) on "miss vs caught" over the atlas features for M_itw ∪ C_itw, to name the region in words.
6. **Verdict** with cluster-bootstrap intervals (speakers for ITW), written as (a) or (b) with the evidence beside it, plus the M_test picture as a 24-clip footnote.

## Deliverables

- `docs/reports/2026-10-20_v2-24-what-sees-the-misses.md` (every number cited to a file).
- `outputs/v2/mechinterp/` (probe tables, projections, the tree).
- Tests for any new library code (`tests/`), ruff clean, suite passing.
- `docs/prod-chunks.md` V2-24 status line; `experiments/log.csv` row (rung `V2-24`).
- Commit on a branch cut from `v2`; no push unless Nathan says so.

## Not this lane's

- Training anything beyond linear probes. Changing the shipped rule. Reading the holdout ledger or the sealed ledger. The AntiDeepfake backbones (V2-53 has them).
