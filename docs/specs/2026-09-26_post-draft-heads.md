# Run spec: post-draft heads lane, frozen WavLM Large probe (Sat Sep 26, 2026, 18:15)

**Rung:** post-draft heads, from `docs/handoffs/2026-09-26_post-draft-heads-handoff.md` (revised ~17:25: the XLS-R layers 5–9 probe is dropped, since Fable ran it on the on-disk embeddings and it failed on In-the-Wild). P1 frozen by Nathan ~17:35 and P2 ~18:05, compressed process: no code plan, so no Codex plan review; the Claude critique and Codex audit run on this spec and the report after the export lands. The 18:50 hard stop was lifted by Nathan (~18:10); the export is due ~19:15, before the gate packet at 19:30. Nothing here touches fusion, the runner, the TSV or any shipped file.

This spec was written before any WavLM probe number was seen. The checks and the fallback below are pre-declared.

## Problem

NSA's draft review returned minDCF 0.0733 on the shipped rule (A3 w0.2 + E); one team is ahead at 0.0584. The shipped rule misses 158 of 1,000 In-the-Wild fakes at its In-the-Wild brief argmin (15 FA). Of those, M1b alone flags 1 at its inner threshold, handcrafted 2, M5 33, Spectra 107: heads on the XLS-R backbone share M1b's blind spot. A different frozen backbone is the one head experiment left. The expected test effect is near zero; a clean negative result is graded README content.

## What runs

| Step | What | Output |
|---|---|---|
| 1 | WavLM Large extraction of the four v3 sets, each with its v3 twin's `extract_meta.json` flags | `outputs/embeddings/wavlm-large/{nsa_train_sample_wl, asv19_addon_wl, nsa_test_wl, itw_stress_wl}` |
| 1b | Repair of rows FFmpeg failed to decode under parallel load (new `scripts/repair_decode_errors.py`) | the same shards, flags cleared |
| 2 | `scripts/train_probe.py --model wavlm-large --train nsa_train_sample_wl,asv19_addon_wl --folds splits/nsa_folds_plus_asv19.csv --stress itw_stress_wl` | `models/m1_wavlm-large_L*_<stamp>/` |
| 3 | `scripts/export_probe_scores.py` with the same sets, `--test nsa_test_wl`, `--name wavlm_l` | `outputs/detector_scores/wavlm_l.csv` |
| 4 | Diagnostics from exports only: Spearman vs `m1b_v3`; shipped-rule miss-tail count | report |
| 5 | Report | `docs/reports/2026-09-26_post-draft-heads.md` |

## Design decisions

- **D1 (recipe = M1b v3).** Segment mode, band match, silence trim, 8 s cap, per-input normalization (`hearsay.embed`; WavLM's own `do_normalize: true` is the same operation, so nothing differs), time-mean per layer, all 25 hidden states stored in fp16. Crops: test-length, seed 0 for the NSA train sample, **seed 100** for the ASV19 add-on (what `asv19_addon_v3/extract_meta.json` records; the handoff's command said 0), none for the test set, seed 200 for In-the-Wild. Probe: layer by inner-CV brief minDCF only; StandardScaler + class-balanced logistic (C = 1) + Platt, all fold-local; ASV19 rows are inner-only; holdout read once; In-the-Wild never fit on. No code in the recipe changes.
- **D2 (parallel extraction and decode repair).** To fit the time box, the four sets were extracted by three concurrent processes after the serial chain was cut (the training sample resumed from its five completed shards; resumes are bit-identical because crop lengths and per-row seeds are regenerated deterministically). Under that load FFmpeg failed transiently on files that decode fine on retry (30 rows by 18:10; 0 in every v3 set; 0 in the five serial shards). `scripts/repair_decode_errors.py` re-embeds only flagged rows through the extractor's own segment path (stored crop length, seed = meta seed + row), after an identity check that re-embeds unflagged rows of the same set and refuses to write if they do not reproduce (relative error ≥ 2e-3). Run on each set only after its extraction has finished.
- **D3 (fallback, pre-declared).** If the ASV19 add-on is not extracted when the training sample is, train NSA-only, call it an M1-style probe, and compare it against M1 v3 (holdout 0.159, In-the-Wild 0.380), not M1b.
- **D4 (diagnostics, pre-declared).** (a) Spearman of `wavlm_l` vs `m1b_v3` logits on holdout, test and In-the-Wild; reference Spectra vs M1b on test (0.62 by this lane's computation; the handoff quotes 0.57). (b) Miss tail: reproduce the shipped rule on In-the-Wild from the exports (must give exactly 15 FA / 158 miss at the ITW brief argmin, else stop); each column's inner threshold = brief-cost argmin over its own inner-OOF logits; count of the 158 each column alone flags. Reference, reproduced by this lane's script: M1b 1, handcrafted 2, M5 33, Spectra 107. No fused numbers: the gate chat computes those.
- **D5 (ownership).** Tracked changes: this spec, the report, `scripts/repair_decode_errors.py`, `tests/test_repair_decode_errors.py` and `scripts/heads_diagnostics.py` (D4's diagnostics, promoted from a scratch script after the run so every reported number cites a tracked file; it reads exports only and hard-asserts the 15 / 158 reproduction). No change to `embed.py`, `probe.py`, the extraction, training or export scripts, the fold files or any fusion file. Staged by path; not pushed.

## Acceptance checks (fail → stop and report)

| # | Check | Pass |
|---|---|---|
| A1 | `extract_meta.json` of each `*_wl` set equals its v3 twin's | identical dicts |
| A2 | Row counts | 20,000 / 7,648 / 1,671 / 3,000; manifest paths in the same order as v3 |
| A3 | Decode errors after repair | 0 per set (v3 has 0), or each remaining row named in the report |
| A4 | Embeddings | shape (N, 25, 1024), all finite |
| A5 | Repair identity check | relative error < 2e-3 on the verified rows of every repaired set |
| A6 | Layer selection | trainer's inner-CV argmin; full CV table in the report |
| A7 | Direction | trainer's holdout AUC > 0.5 assertion passes |
| A8 | Export shape | splits inner_oof 16,142 / holdout 3,858 / test 1,671 / itw 3,000; path set per split identical to `m1b_v3.csv`; no NaN |
| A9 | Shipped-rule reproduction before counting the miss tail | exactly 15 FA / 158 miss |
| A10 | Tests | `uv run pytest -q` all pass; `uv run ruff check .` clean |

`tests/test_repair_decode_errors.py` covers D2's tool (6 tests):
- flagged rows only, with the stored crop and the row seed
- NaN crop for uncropped sets
- identity mismatch refuses and writes nothing (fails if the check is removed)
- a still-failing row stays flagged
- a no-op on clean sets
- windows-mode refusal

## Readouts reported (no pass/fail: the gate chat judges)

CV-by-layer table; holdout minDCF, EER, act DCF, per-generator and per-source; In-the-Wild minDCF (brief) and averse cost, P_FA and P_miss at the inner threshold; Spearman (D4a); miss-tail count (D4b); timing. For the column alone, the interesting outcomes are In-the-Wild below M1b's 0.342 and a test-set Spearman with M1b well below 1.

## Results (19:10)

Full readouts, CV table, diagnostics and the decode-repair incident are in `docs/reports/2026-09-26_post-draft-heads.md`. Summary:

**Run.**
- Layer 9 by inner CV (0.2936).
- Holdout 0.0717 / EER 1.45%, level with M1b v3.
- In-the-Wild 0.752 brief / 0.383 averse, against M1b's 0.342 / 0.296. P_FA at the inner threshold is 7.05%, against 0.8%.

**Diagnostics.**
- Spearman vs M1b: test 0.795, holdout 0.863, In-the-Wild 0.774.
- Miss tail: WavLM flags 57 of the shipped rule's 158 In-the-Wild misses. References: M1b 1, handcrafted 2, M5 33, Spectra 107.

**Checks.**
- A1–A9 pass. 99 transiently failed rows were repaired, identity relative error 0.0, and none are still failing.
- A10: pytest **745 passed**. `ruff check` is clean on this lane's files; the repo-wide run fails on 3 findings in `src/hearsay/analyzer.py` from commit 19925a3, which belongs to another lane.

**Deviation from D2's plan.** The three-process layout was abandoned at ~18:15 because of swapping, and the train-sample process was restarted clean at 18:34. Neither changes any output: resumes are deterministic, and every repaired set passed the identity check.

**Gate verdict** (gate chat, commit 57b9c90): the pre-declared entry failed both the gate and the room. WavLM does not ship; the result is recorded as a negative.
