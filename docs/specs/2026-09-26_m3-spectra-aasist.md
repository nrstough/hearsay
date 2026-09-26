# Run spec: M3, Spectra-AASIST as the second deep score stream (Sat Sep 26, 2026, ~04:20)

Status: P1 and P2 frozen (deep mode). Plan: `docs/reports/2026-09-26_m3-spectra-aasist-plan.md`.
- **Worktree:** branch `main`, `~/Projects/hearsay`, shared with the main chat, the CPU-detector chat and the M5 chat. Stage only M3 files; never `git add -A`; never push without Nathan.
- **Handoff:** `docs/handoffs/2026-09-26_m3-spectra-aasist-handoff.md`.
- **Rung:** M3 (`docs/plan.md` ladder: 45 min, "drop on friction").
- **Time box:** 45 min of build from freeze (~04:20 → ~05:05 EDT), then the scoring run on MPS once the main chat releases it (their estimate 05:00–05:15). Export expected by ~07:00.
- **Hard stop:** 08:00 EDT. Whatever exists then is reported; nothing is pushed through.
- **Coordination (Sat 04:05–04:20):** the main chat confirmed band match now lives in `hearsay.embed.prepare_segment` (commit 133e536), asked for an In-the-Wild stress readout before the full export, and will read the exported `logit` column directly in fusion. The oversight chat asked for the exact shared crop recipe (D3), the direction → In-the-Wild → export order, a contamination check on the Spectra card (done: In-the-Wild is listed only as an evaluation set; training data undisclosed), and a raw-logit export (D6).

## Problem

Fusion (M4) has one deep score, the M1 XLS-R probe, plus the engineered detectors. A second deep detector trained differently disagrees in useful ways, and the stacker can exploit that. Spectra-AASIST (`lab260/Spectra-AASIST`, XLS-R 300M front end + AASIST back end, off the shelf) already loads from `weights/Spectra-AASIST` and passed its direction check on ASVspoof 2019 eval (AUC 1.0, 0.169 s per clip on MPS). Nobody has scored the NSA fold file, the test set or the In-the-Wild stress set with it. This change produces that score stream in the fusion format, the holdout readout, and a short report. No training, no fine-tuning, no TSV, nothing under `submissions/`.

Two facts shape the design:
- **81% of NSA test clips are shorter than one Spectra window** (64,600 samples, 4.04 s; test median 3.41 s). How a short clip is padded is the test condition, not an edge case. Training-side clips are 5–9 s, so without matched crops the padding would apply almost only to test rows, the tiling seam M1 v2 removed.
- **Every NSA test clip is low-passed at ~7.2 kHz** and no training corpus is (`docs/reports/2026-09-26_cpu-detectors.md`). Since 133e536 the shared input path band-matches every clip by default.

## Solution

### D1. Scope
A scoring rung. One new script, one module extended, one additive helper in `hearsay.metrics`, one hermetic test file, one report. Nothing is fit except a two-parameter diagnostic that never reaches the export. The "always a valid TSV" rule is satisfied by the main chat's existing submissions; M3 emits no TSV.

### D2. Input path (one function scores every row)
`load_audio` (16 kHz mono, existing) → `hearsay.embed.prepare_segment(x, crop_s, seed)` with default args, i.e. band match (7.25 kHz Kaiser low-pass, from 133e536) → `trim_silence` → optional crop → cap at 8 s → short-clip handling (D4) → 64,600-sample windows at 50% hop, at most 16 → `torchaudio.functional.preemphasis` → Spectra forward → mean of per-window logits → `synth_logit = logit_spoof − logit_bonafide`. No filter of our own is added on top of `prepare_segment`. Inner, holdout, test and In-the-Wild rows all go through this function.

### D3. Crop parity with M1 and the handcrafted detector
Fold rows (inner **and** holdout) are cropped to test-duration lengths with the exact draws the other detectors used: `hearsay.embed.test_duration_sampler(0)`, one draw per row of `outputs/manifests/nsa_train_sample.csv` in order (same 20,000 paths and order as `splits/nsa_folds.csv`, verified), per-row offset seed `0 + row index`, keyed by path so a `--limit` pilot crops each file identically to the full run. In-the-Wild rows: seed 200, one draw per `outputs/manifests/itw_stress.csv` row in order (the main chat's `itw_stress_v2` seed). Test rows are scored whole, no crop. Rationale: the stacker sees the same audio per file across M1, handcrafted and M3, the holdout is honest about the padding mode, and the pad-mode pilot on inner rows is meaningful because inner rows are now as short as test rows.

### D4. Short-clip handling: three modes, chosen on inner rows only
For a prepared clip shorter than 64,600 samples:
- `repeat`: tile to one window (the model card's own `pad_random`, in-distribution for Spectra, but a seam).
- `zero`: zero-pad to one window (the handoff's option (b)).
- `whole`: feed the clip at its own length (the AASIST positional table covers 400 SSL frames = 8 s), with a 1 s zero-pad floor.
Clips at or above one window are windowed identically in all modes. **Selection:** a pilot on the first 200 fold rows (about 160 inner) plus the first 200 test rows, on CPU while MPS is held. Primary criterion inner-row minDCF (π_synth = 0.3); modes within 0.02 tie, and the tie goes to `zero`. A winner whose test share above one half exceeds 0.6 is flagged as a false-alarm signature, the next mode is preferred, and both are recorded. The holdout is never used for selection and is read once, after the full run. One extra run with `band_match=False` on the winner is a diagnostic reading for the report only, run last and dropped if the box is tight.

### D5. Direction gate
Before any fusion file is written, the scorer computes AUC(synth_logit, is_spoof) and the class medians on the inner rows it scored. AUC ≤ 0.5 or spoof median ≤ bona fide median exits non-zero, writes nothing under `outputs/detector_scores/` or `models/`, and leaves the raw file under `outputs/spectra/` for inspection. Flipping is never applied silently. The untouched `scripts/spectra_direction.py` is also run once on 300 inner rows (150 per class, sampled from fold ≠ holdout, seed 0) as a baseline reading of the raw path (no trim, no crop, repeat-pad).

### D6. Export: raw margins, no fitted calibration (Nathan, Sat 04:15)
`outputs/detector_scores/spectra_aasist.csv`, header exactly `path,fold,split,score,logit`, one row per fold-file row (20,000; `split` = `inner_oof` for folds 0–4, `holdout`) plus one per test-manifest row in manifest order (1,671; `fold = test`, `split = test`), path strings verbatim from their source files, unique. `logit` = raw `synth_logit`; `score` = `sigmoid(logit)`. Rows that fail decode or the model keep NaN in both columns (fusion imputes) and are counted; more than 5% failures aborts the export. **No Platt or other fitted map is applied to the file.** Reason: a monotone map cannot change minDCF, but it moves thresholds on shifted data and, at Spectra's ±10–17 margins, an expansive map would saturate the 1e-4 clip in fusion v1 and destroy ranking in the bona fide tail. The main chat confirmed fusion will read `logit` directly, standardized per detector on inner rows. Companion `outputs/detector_scores/spectra_aasist_raw.csv` carries `logit_spoof, logit_bonafide, synth_logit, n_windows, n_samples, crop_s, flag` per row in the same order.

### D7. Readouts in `models/m3_spectra_<stamp>/meta.json`
- Holdout: `hearsay.metrics.report` on `synth_logit` (EER, minDCF at π_synth = 0.3, actual DCF, minDCF at π = 0.5, `sponsor_code_asis`, `sponsor_code_flipped`), `by_group` per held-out generator (playht, wavegrad2) and per bona fide source (librispeech, ljspeech), AUC.
- Inner rows pooled: the same report, as a diagnostic (Spectra is not fit on them, so inner and holdout are both out-of-sample for the model).
- Test: n, share with `synth_logit > 0` (prior about 0.30), mean and std of score, distinct scores.
- Direction dict (D5), the pilot table and the chosen mode (D4).
- Platt (a, b) fit class-balanced on inner rows, labeled `in_sample_diagnostic: true`, with the share of test rows the map would put above one half. Never applied to the export.
- Level-shortcut measurement: within-class Spearman correlation between clip peak level and `synth_logit` on inner rows.
- In-the-Wild stress (`--stress itw=<scores.csv>`): n, minDCF, EER, per generator and source where more than one value exists, and P_FA / P_miss at Spectra's own inner-row minDCF threshold and at its inner-row EER threshold, flagged "possibly optimistic, training data undisclosed".
- Config: pad mode, band match, max windows, seed, device, crop policy, git sha, row counts, seconds per clip, wall time.
- M1 comparison in the report reads the newest `models/m1_*/meta.json` at write time.

### D8. Operations
`scripts/score_spectra.py`: `--folds`, `--test-manifest`, `--device` (default mps if available), `--pad-mode`, `--band-match/--no-band-match` (default on, passed through to `prepare_segment`), `--max-windows 16`, `--seed 0`, `--limit N` (pilot: outputs only under `outputs/spectra/pilot/`, never the fusion file or `models/`), `--manifest path --name n` (labeled stress set: outputs under `outputs/spectra/<name>/`, never the fusion file), `--stress name=path` (repeatable, post-hoc readouts in meta), `--out-name` (default `spectra_aasist`), `--resume`, `--prefetch 4`. Decode in threads (the `extract_embeddings.py` pattern), model forward on the main thread, `band_limit` warmed once on the main thread before the pool starts. Progress with ETA every 100 rows; raw rows checkpointed every 500 to `outputs/spectra/<out-name>_partial.csv` with a config hash; `--resume` skips scored paths and refuses on a hash mismatch. Full run at `nice -n 10` under `nohup`, log `outputs/spectra/score_spectra.log`. Never on vast.ai. MPS only after the main chat releases it.

### D9. `by_group` moves to `hearsay.metrics` (additive)
Copied verbatim from `scripts/train_handcrafted.py`; the trainer is left untouched (it imports LightGBM and is another chat's file). `hearsay.spectra` and the scorer never import LightGBM.

### D10. Order of operations
1. Build and tests (box).
2. CPU pilot with `--limit 200`, three modes, at `nice -n 10` while MPS is held; choose the mode.
3. When MPS is released: direction baseline (D5), then In-the-Wild (3,000 clips, about 10 min), post the numbers to Nathan and the main chat.
4. Full 21,671-row export in the background (about 75 min at 0.2 s per clip). Default is to proceed; Nathan or the main chat can stop it.
5. Report, STATUS row, disclosure line, commit, audits.

## What will change

| File | Change |
|---|---|
| `src/hearsay/spectra.py` | Extend: input prep, short-clip modes, `spectra_logits(pad_mode=)`, direction check, crop plan, scoring loop helpers, export, meta. Existing signatures keep working. |
| `src/hearsay/metrics.py` | Add `by_group` (verbatim from the handcrafted trainer). |
| `scripts/score_spectra.py` | New: argparse and orchestration only. |
| `tests/test_spectra.py` | New: hermetic tests with a fake torch module; one weights-gated CPU test. |
| `docs/reports/2026-09-26_m3-spectra.md` | New report. |
| `docs/STATUS.md` | M3 row. |
| `CLAUDE.md` | Disclosure: Claude Code bullet gets an M3 line; Spectra-AASIST "how used" = M3 score stream, license still unclear. |
| `docs/specs/2026-09-26_m3-spectra-aasist.md` | This file. |
| `docs/reports/2026-09-26_m3-spectra-aasist-plan.md` (+ `-plan-review.md`) | Plan and its Codex review. |
| Conditional | `docs/plan.md` (expect no change); staging the untracked handoff with the commit (confirm with Nathan). |

Not touched: `splits/nsa_folds.csv`, `scripts/spectra_direction.py`, `scripts/train_handcrafted.py`, `src/hearsay/embed.py`, anything the main, CPU or M5 chats own, `submissions/`.

## Tests (P2, frozen)

Stage A input path: A1 bitwise parity with `prepare_segment` (crop on/off); A2 band-match counterfactual (7.8 kHz tone attenuated > 40 dB by default, unchanged with `band_match=False`); A3 no crop for test rows, 8 s cap; A4 crop parity pin (fold row crop equals the draw at its manifest index, first seed-0 draws pinned to 6 decimals, `--limit` changes no crop, In-the-Wild seed 200 independent); A5 repeat tiles (autocorrelation peak at lag = clip length); A6 zero pads zeros (no peak); A7 whole keeps own length, 1 s floor; A8 long clips identical across modes, 50% hop, last flush, cap; A9 window-count formula; A10 pre-emphasis counterfactual; A11 degenerate inputs finite (all zeros, 0.1 s, NaN-bearing WAV); A12 float32 contiguous; A13 unknown mode raises.

Stage B scoring: B1 direction contract; B2 window mean; B3 cap reaches the model; B4 device and no grad; B5 default signature unchanged (equals repeat); B6 model error → NaN + `model_error`, run continues; B7 decode error → NaN + `decode_error`, row kept.

Stage C export and gates: C1 counts and splits; C2 relative paths resolved for loading, verbatim in the file; C3 test order preserved; C4 header byte-equal; C5 no calibration in the file (logit equals raw, score equals sigmoid, meta Platt non-identity); C6 duplicate paths refused; C7 NaN rows kept and counted; C8 failure-rate gate (6% aborts, 5% passes); C9 direction gate blocks export, raw file kept; C10 `--limit` never writes fusion or `models/`; C11 `--manifest` mode writes only under `outputs/spectra/<name>/`, accepts `bona-fide`, errors without labels; C12 `--out-name`; C13 raw companion aligned; C14 checkpoint/resume byte-identical, hash mismatch refuses; C15 no LightGBM import.

Stage D metrics and meta: D1–D3 `by_group` semantics; D4 holdout numbers unaffected by other rows; D5 Platt diagnostic uses inner rows only; D6 test share above half; D7 config recorded; D8 one-class slice skipped gracefully; D9 stress thresholds come from inner rows, P_FA/P_miss at both; D10 level-shortcut measurement recorded.

Weights-gated: W1 real model on CPU, 2 s and 6 s sines, all modes, finite logits of the expected shape.

Standing failure modes: (1) laundering — nothing trained, band match measured, physical replay not covered; (2) generator/speaker — nothing fit, held-out groups read once, selection on inner rows only; (3) prior — all four readings, a flip between them is flagged; (4) shortcuts — silence, duration, tiling and bandwidth via the shared path and the pilot, level measured, container waived (PlayHT MP3 is a holdout generator; noted per generator); (5) loader — existing `load_audio`, already pinned.

## Acceptance criteria

- AC1 `uv run pytest -q tests/test_spectra.py` passes; `uv run pytest -q` has no new failures (215 passing before); `uv run ruff check .` clean.
- AC2 `scripts/spectra_direction.py` exits 0 on the 300 inner rows; the scorer's gate passes for the chosen mode.
- AC3 fusion file: 21,671 rows, unique paths, exact header, split counts 16,142 / 3,858 / 1,671, ≤ 5% NaN, `score == sigmoid(logit)` on every finite row.
- AC4 meta has every D7 block.
- AC5 the pilot ran before the full run; the selection is recorded in meta and the report.
- AC6 report, STATUS row and disclosure lines written; docs-consistency tests pass.
- AC7 `git status` shows only M3 files changed.
- AC8 the full run finished, or is running with a log and the report says so.

Regression: any previously passing test failing, a ruff finding, an edit outside the file list, a change to any existing export, or a change in `spectra_direction.py`'s default behavior.

## Codex plan review

Run Sat 04:55 on the second draft of the plan (after a Claude critique pass fixed 6 critical items in the first draft: tests writing into the real `outputs/`, three vacuous tests, resume semantics, per-mode pad counts, the stress threshold sweep, the build box). Review file: `docs/reports/2026-09-26_m3-spectra-aasist-plan-review.md`. Nine findings, all addressed in the plan's amendments 8 and 13–20:

| # | Finding | Resolution |
|---|---|---|
| 1 Critical | Resume hash omits the training manifest (its order is the crop recipe) | Hash the training manifest, the stress manifest, the scoring script and the model's `model.py`/`config.json` (amendment 8); C16 reorders the manifest and expects a refusal |
| 2 Critical | Non-finite model outputs pass the failure gate | Output shape and finiteness validated in `score_clip`, else `model_error` (13); B8 |
| 3 Critical | NaN rows crash `report`/`by_group`/AUC | `finite_rows` mask with `n_used`/`n_dropped` in every readout, status dicts instead of exceptions (14); D11 |
| 4 Critical | Publication not atomic, stale export possible | Run directory first, publish last by `os.replace`, `<out-name>_summary.json` carries the stamp and hash (16); C14 |
| 5 Critical | Provisional stress readout not producible before the export | `--readout` mode on saved scores (17); D9 |
| 6 Critical | No completion condition for the TSV boundary | M3 emits no TSV by design (handoff); completion = published file + summary + numbers handed to the main chat and recorded with the time, or a named blocker (19) |
| 7 Suggestion | Global 5% gate hides concentrated failures | Per-split gate, per-class and per-generator coverage in meta (15); C8 |
| 8 Suggestion | C14/C10 conflict with the lifecycle | `--max-chunks` makes an interrupted run; C10 enumerates created paths instead of snapshotting shared trees (18) |
| 9 Suggestion | Laundering checks untested | Recorded as a deferred limitation of a scoring rung, next to physical replay (20) |

Amendments 1–12 came from the exploration and critique agents: chunked decode pool, raw companion under `outputs/spectra/`, one-pass pilot with `--limit 400 --test-limit 200` and holdout rows skipped, consecutive-decode abort, the M1 comparator rule, the secondary tripwire, the `ValueError` forward guard, the resume rules, `--repo-root`, explicit threshold sweep, direction recorded (not gated) outside the full run, and the build order for a box that is 2–3 h rather than 45 min.

## Amendments (Sat Sep 26, 05:05, before commit)

Recorded from the plan (`docs/reports/2026-09-26_m3-spectra-aasist-plan.md`, "Amendments to the run spec", items 1–20); the design decisions D1–D10 above stand except where noted.
- **D4 pilot:** `--limit 400 --test-limit 200`, inner and test rows only (holdout rows in the head slice are skipped), all three modes in one decode pass; tie-break order inner minDCF, inner AUC, then `zero`. Direction is recorded per mode, not gated, in the pilot and in manifest mode.
- **D6 export:** the raw companion lives at `outputs/spectra/<out-name>_raw.csv` (five base columns first, then labels and raw columns), not under `outputs/detector_scores/`. Everything is computed into `models/m3_spectra_<stamp>/{raw.csv, fusion.csv, meta.json}` first; the fusion file is then published by atomic replace together with `outputs/detector_scores/<out-name>_summary.json` (stamp, config hash, git sha, counts). The 5% failure gate applies per split; non-finite or wrong-shape model outputs are failures; every readout applies one finite-row mask and records `n_used` / `n_dropped`.
- **D7 readouts:** a secondary test tripwire (share with `logit_bonafide` at or below the model card's classify threshold, −1.0625009) beside the primary share with a positive margin; stress thresholds come from an explicit ROC sweep on inner rows with `None` plus a flag at the +inf entry or when nothing beats the constant decision; `--readout` computes the stress readout from saved scores without inference, so the In-the-Wild P_FA at the pilot's inner threshold exists before the full export (labeled provisional); failures are counted per split, per class and per generator. The M1 comparator is the newest `models/m1_*` meta with `folds == splits/nsa_folds.csv`, a single training set and `val_min_dcf` present.
- **D8 operations:** `--repo-root` drives every output path and relative-path resolution (the hermetic tests run in a temp directory); the decode pool maps per chunk (500 rows) so decoded clips never pile up in RAM; `band_limit` is warmed on the main thread before the pool (its lazy FIR init is a two-statement race); 25 consecutive decode errors abort with a mount hint; the resume hash also covers the training manifest, the stress manifest, the scoring script and the model's `model.py`/`config.json`; resumed rows reuse their stored margin verbatim (read back with `float_precision="round_trip"`); the partial and its sidecar are deleted on completion and `--resume` without a partial exits 2; `--max-chunks N` stops after N chunks with exit 5 (the resume test's interrupted run, and a dry run); the forward guard is a `ValueError`, not an `assert`.
- **D10 completion:** M3 emits no TSV by design; the rung is complete when the published file, its summary sidecar, the run stamp and the holdout numbers have been handed to the main chat and recorded below with the time. Codec laundering, telephony and simulated replay are untested in this scoring rung and are listed in the report next to physical replay.
- **Time box:** the frozen P2 test list was a 2–3 h build, not 45 min. Build order put the gates first; the diagnostics were built last (all landed).
- **D5 raw-file location (superseded by amendment 16 of the plan):** on a direction or failure-gate refusal the raw scores and the gate meta stay in `models/m3_spectra_<stamp>/` (`raw.csv`, `meta.json`, no `fusion.csv`), not under `outputs/spectra/`; nothing is published. On a single-mode run the gate path writes the same mode-frame layout as the success path.
- **D7 readouts after the run (06:00–06:25, oversight chat's asks):** `--readout` also accepts `test=<scores.csv>` with `m1_tsv=<submission.tsv>` (Spearman of the two score columns on the test set and the files each detector alone calls synthetic at its own midpoint), `pilot=<pilot meta.json>` (carries the pad-mode table and the selection into the full run's meta), and `--final` (unsets `provisional` when the inner scores are the full run's). `--merge-into <meta.json>` appends one `readout` block plus the `caveats` block (`inner_rows_possibly_in_sample`, `in_the_wild_possibly_optimistic`) to the run's `meta.json`; the immutable run artifacts are `raw.csv` and `fusion.csv`, and the readout is a derived, reproducible addition. The full run's meta therefore carries the pilot table and the chosen mode through the readout, not through the scoring pass.
- **D8 progress cadence:** progress with ETA prints per decode chunk (500 rows), not every 100 rows; `--limit 0` and `--test-limit 0` are honored as zero rows.
- **Tests declared by amendment:** A14 (`peak_normalize` and the `--peak-norm` plumbing, config-hash separation, pre-normalization `peak` kept), C14b (a run that trips the gate on its last chunk publishes nothing and leaves raw plus gate meta in the run directory), D11 as written (one undecodable file planted in an inner, a holdout and a test row, two in a stress set; every readout keeps `n_used` / `n_dropped`; `failures.by_split` and per-generator coverage recorded). The unreachable `argmin_at_inf_entry` branch of `sweep_thresholds` was removed: roc_curve's +inf entry costs exactly the constant decision, so the existing flag covers it.
- **Peak-normalization arm (05:12, main chat's proposal after the pilot's level flag):** `--peak-norm` scales every prepared clip to peak 0.95 after `prepare_segment` and before windowing and pre-emphasis, applied to every row alike (Spectra sets `normalize_waveform=False`, so raw level reaches the model; training real audio peaks ~0.5 while NSA test clips and some spoof generators sit ~1.0, a synthetic cue that would push loud real test clips toward "fake", the 4× side). The pilot arm runs on the same 322 inner + 200 test rows with `zero`. Rule: ship it if inner minDCF stays within the 0.02 tie band of the raw arm and the within-spoof level Spearman drops; if it ships, the In-the-Wild set is re-run with it so the stress readout matches the export. The flag is in the config hash, meta and summary sidecar; the raw companion keeps the pre-normalization `peak`. Two full runs inside one minute no longer share a run directory (collision suffix).

## Results (Sat Sep 26, filled 06:10 before commit)

- **Baseline before edits:** 301 tests passing (the P2 figure of 215 predated tonight's commits), ruff clean.
- **Build:** `hearsay.metrics.by_group` (additive), `hearsay.spectra` extended (input prep, three pad modes, forward guard, per-clip scoring, crop plan, finite-row readouts, threshold sweep, stress readout, Platt and level diagnostics, peak normalization), `scripts/score_spectra.py` (full / pilot / manifest / readout modes, chunked decode pool, per-split failure gate, direction gate, run directory then atomic publish, resume with config hash, `--max-chunks`, `--repo-root`), `tests/test_spectra.py`: 40 collected after the audit fixes (39 hermetic plus the weights-gated W1, which ran on CPU; several IDs share one function), all passing; full suite: 351 passing over the tracked test files at the fix commit (M3's file has 40 after the audit fixes; the count moves as other chats commit), no failures; the working tree's three failures at commit time are all in another chat's untracked `tests/test_speech_gate.py`; ruff clean.
- **Pilot (CPU, 05:02–05:09, 322 inner + 200 test rows, three modes in one pass):** inner minDCF repeat 0.0066 / zero 0.0000 / whole 0.0000, AUC 1.0 in all; test share above a zero margin 0.310 / 0.265 / 0.280. Pick by the frozen rule: `zero`. Level flag: within-spoof Spearman(peak, margin) 0.27–0.36.
- **Peak-normalization arm (MPS, 05:18, main chat's proposal):** inner minDCF 0.0331 (LJ Speech real 0.033), test share 0.265, within-spoof Spearman rose to 0.40. Rejected by the rule in the amendments; raw zero-pad shipped.
- **Direction baseline (untouched `spectra_direction.py`, MPS, 05:19):** 300 inner rows, AUC 1.0, EER 0, medians +10.5 / −7.7, exit 0.
- **In-the-Wild (CPU, 05:10–05:28, 3,000 clips, seed-200 crops):** minDCF 0.065, EER 3.2%, AUC 0.996; at the full run's inner thresholds real P_FA 0.0%, spoof P_miss 7.8% (EER threshold 5.41) / 13.8% (minDCF threshold 6.92). Flagged possibly optimistic (training data undisclosed). M1 v3 on the same set: 0.380.
- **Full export (MPS, 05:22–06:00, 2,256 s, 0.104 s per clip):** 21,671 rows, 0 failures in every split; direction on 16,142 inner rows AUC 1.0. Published `outputs/detector_scores/spectra_aasist.csv` + `spectra_aasist_summary.json` (stamp 20260926-0522), raw companion `outputs/spectra/spectra_aasist_raw.csv`, run dir `models/m3_spectra_20260926-0522/`. AC3 verified: header exact, 21,671 rows, unique paths, splits 16,142 / 3,858 / 1,671, `score == sigmoid(logit)` on every row, test order and fold paths verbatim.
- **Outer holdout (read once):** minDCF **0.0120**, EER 0.16%, AUC 1.0, actual DCF 0.065, minDCF at π = 0.5 0.0063, sponsor code as-is 1.0 / flipped 0.0022; playht 0.010, wavegrad2 0.012, LibriSpeech real 0.011, LJ real 0.008. Comparator M1 v3 (`m1_…_0518`, same fold file): 0.159. Inner pooled: 0.0045 (diagnostic; possibly in-sample).
- **Test set:** 29.3% of files above a zero margin (prior ~30%; M1 v3 28.9%); 28.3% under the card's threshold. Disagreement with M1 v3: Spearman 0.57, both synthetic 409, both real 1,108, Spectra only 80, M1 only 74.
- **Caveat recorded in meta and report:** `inner_rows_possibly_in_sample = true` (oversight chat's ask): Spectra's training data is undisclosed and the inner rows score AUC 1.0.
- **Completion handoff:** published file, sidecar, stamp, holdout numbers and the "fusion reads `logit`" note sent to the main chat 06:02; In-the-Wild numbers sent 05:29 (before the export) to the main and oversight chats. No TSV by design.
- **Deviations from the plan:** the In-the-Wild set ran on CPU (0.36 s per clip) instead of waiting for MPS, so it finished while the main chat still held the GPU; the `--stress` flag was not passed to the export (the stress file would have been read at the end of a 38-minute run while its producer was still writing), the same readout was produced afterwards by `--readout … --merge-into meta.json`, which is the reproducible path; the `--no-band-match` diagnostic run was dropped (not needed for any decision, GPU handed back instead); the manifest-mode and pilot checkpoints are left inside their run directories (only full-mode partials are deleted). `docs/plan.md` needed no change. Cut from P2: nothing; every test ID landed (A1–A14, B1–B8, C1–C16 plus C14b, D1–D11, W1). The pre-audit critique (06:20) found D11 hollow, the C14 publication-order case missing and A2's band-off half vacuous; all three were written for real in the fix commit, plus the readout `--final` / `pilot=` additions so AC4's pilot table and chosen mode reach the full run's meta and the In-the-Wild block is no longer labeled provisional.

## Audits

- **Claude pre-audit critique (06:20–06:35, adversarial, audit rubric):** round 1 Overall **Fail** on Test coverage (D11 hollow, C14 publication-order case missing, A2 band-off half vacuous; AC4 partly met; `--merge-into` and the readout keys undeclared). Fixed in ed31cb9. Round 2 Overall **Acceptable** (Regression Excellent; three cosmetic leftovers fixed in 1747286: tracked test count label, `--final`/`pilot=` assertions, `--limit 0` empty-rows guard).
- **Codex audit round 1 (06:45, `.claude/review-audit.sh`, file `docs/specs/2026-09-26_m3-spectra-aasist-audit.md`):** Overall **Fail**. Findings and resolutions: (1) the per-split failure gate compared a rate rounded to 4 places, so 193 of 3,858 (5.0026%) passed → gate now on integer counts (`splits_over_gate`), test C8b pins 193/3,858 fails and 192/3,858 passes; (2) `crop_plan` drew from `hearsay.embed.TEST_DURATIONS` (the source checkout) while the resume hash fingerprinted the durations file under `--repo-root` → `crop_plan(manifest, seed, durations_csv)` draws from the root-resolved file with the same generator, the CLI tests no longer monkeypatch the module path, and A4 pins that the real manifest plus the real durations file give the pinned seed-0 draws; (3) W1 ran the 6 s clip in one mode with a shape check only → all three modes on both clips, finite, and the modes agree on the ≥ one-window clip; (4) the report's fusion note said inner and holdout are "both out-of-sample for the model", contradicting the in-sample caveat → reworded (nothing fit locally; pretrained overlap unknown for both splits). Codex could not run pytest in its sandbox (cache writes denied); it verified the published artifacts directly (21,671 unique paths, split counts, exact sigmoid equality, byte-identical run and published files).
- **Codex audit round 2 (06:40):** Overall **Fail** on one new finding, the four round-1 items resolved (Documentation **Excellent**): `score_clip` accepted a model output with the wrong number of rows (two finite rows for one window averaged silently). → the check is now `lg.shape == (n_windows, 2)`; B8 covers extra and missing rows on one- and two-window clips, and the script-level failure test counts them as `model_error` with NaN kept. Codex again could not run pytest in its sandbox and verified the published artifacts directly.
- **Codex audit round 3:** _pending._
