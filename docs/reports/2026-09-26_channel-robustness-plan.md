# Plan: channel robustness (A, E, B; C gated) — Sat Sep 26, 2026, 12:35

Run spec: `docs/specs/2026-09-26_channel-robustness.md` (design decisions D1–D10, acceptance criteria, failure-mode table). Hard stop 16:00.

## Ground truth read before writing

- `hearsay.compression.features(x, crop_s, seed, crop_mode, band_match)` (`src/hearsay/compression.py:46`) calls `hearsay.handcrafted._crop` → `band_limit` (`handcrafted.py:43`) → `prepare_segment` (`embed.py:80`: band match off since already filtered, `trim_silence`, optional crop, 8 s cap) → RMS normalize; returns 19 features incl. `bw_hz`, four band levels, `hf_slope_db_per_khz`, hole statistics, `floor_p2_db`.
- `hearsay.compression.launder(x, codec, kbps, sr_enc)` (`compression.py:105`): FFmpeg encode/decode, 16 kHz float32 out, raises `DecodeError`.
- `hearsay.audio.trim_silence` (`audio.py:90`), `load_audio` (`audio.py:22`).
- `hearsay.pipeline.Models` (`pipeline.py:~390`): CPU only; `m1_logit(x)` = `prepare_segment` → `embed_segment` → fp16 round → `Probe.llr`; `spectra_logit(x)` = `prepare_input` → `score_clip(zero pad, 16 windows)` → spoof − bonafide; `m5_logit(x)` with `load_m5=True`. Loading its handcrafted detector is part of the constructor (harmless, no LightGBM import).
- Exports `outputs/detector_scores/{m1b_v3,spectra_aasist,m5_xlsr_ft}.csv`: `path, fold, split ∈ {inner_oof, holdout, test, stress/itw}, score, logit`.
- Manifests: `splits/nsa_folds.csv` (path, label, generator, speaker, source, group, fold), `outputs/manifests/itw_stress.csv` (path relative, label, speaker), `outputs/manifests/nsa_test.csv` (filename, path), `outputs/manifests/mlaad_en.csv` (path, label, generator, model_name, …).
- Shipped draft TSV: `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv` (log row 12:03).
- Test pattern for scripts: `importlib.util.spec_from_file_location` (as `tests/test_spectra.py:115`).

## Steps

1. **Codex plan review in the background** on this file (`.claude/review-plan.sh`); findings folded in when they land, without blocking A.
2. **`scripts/channel_lambda.py` (A).** Pure functions: `channel_stats(x) -> dict` (D1: calls `compression.features(x, None, None, "segment", True)` and adds the six statistics on the same prepared signal, obtained by the same `_crop` call); `grouped_oof(X, y, groups, k=5, seed=0)`; `lambda_mean_posterior(p)`; `lambda_acc(q, tpr, fpr)` (NaN below a 0.2 margin); `feature_position(med_test, med_clean, med_wild)`; `bootstrap_ci(fn, *arrays, n=1000, seed=0)`. `main()`: build the three row sets, extract with a 6-process pool (resumable cache at `outputs/channel/lambda_features.csv`), fit, estimate, write `outputs/channel/lambda.json` and print the readout.
3. **Tests for A** in `tests/test_channel_robustness.py` (the A rows of the spec's failure-mode table). Run them.
4. **Run A** (~5 min extraction). **Send λ̂ to the oversight chat immediately.**
5. **`scripts/m3_probes.py` (E).** Pure: `perturb(x, kind, rng)` for `none | mp3 | noise20 | speed+2 | speed-2 | shift1`; `delta_auc(...)`; `spearman_gap(a_hold, b_hold, a_test, b_test, n=1000)`; `m3_verdict(m3_dauc, m1_dauc, gap, gap_lo, mlaad_below)` (D9). `main()` with subcommands `perturb`, `agreement`, `mlaad`; each loads `Models(device="cpu", load_m5=True, threads=3)` where needed, caches rows to `outputs/channel/m3_*.csv` (resumable), writes a JSON summary. The `perturb` and `mlaad` runs go in the background, two processes at 3 threads each.
6. **Tests for E** (the E rows of the table). Run them.
7. **`scripts/channel_codec.py` (B)** while E runs. Pure: `highband_stats(x) -> dict` (D4 statistics, no filtering of its own); `variant_grid()`; `match_distance(var_stats, test_stats)`; `codec_match(table, kaiser_row)` (D5). `main()`: 100 inner clips × grid, 1,671 test files, writes `outputs/channel/codec_grid.csv` and `codec_match.json`.
8. **Tests for B.** Run them.
9. **Gate C.** If B matches, message Nathan here with the time left and what C would cost (local MPS ~60 min extraction vs a vast box), and stop there for his answer. If B is negative, record C as not run.
10. **Report** `docs/reports/2026-09-26_channel-robustness.md`: A, E, B tables with sources, C and D dispositions, verdicts. `docs/STATUS.md` row, `CLAUDE.md` disclosure bullet. Spec "Results" section filled. Doc-sync grep for anything this resolves (e.g. "M3 leakage probes" as open, "λ unmeasured").
11. **Regression:** `uv run pytest -q`, `uv run ruff check .`, `uv run pytest -q tests/test_docs_consistency.py`.
12. **Commit** only this rung's files (`git add <paths>`), conventional prefix, no push.
13. **Claude critique** against `.claude/codex-audit-prompt.md` (loop until Acceptable), then **Codex audit** (`.claude/review-audit.sh docs/specs/2026-09-26_channel-robustness.md`); record both in the spec; fix and re-commit.
14. **Report to the oversight chat** at each milestone (λ̂; E verdict; B verdict; done).

## Time budget (from 12:35)

A code + tests 30 min, run 10 → λ̂ ~13:15. E code + tests 35 min, runs ~40 min in the background → ~14:30. B code + tests 25 min during E's run, run 10 → ~14:15. Report, docs, commit, audits 14:30–15:45. Slack 15 min.

## Risks

- **Domain classifier learns content, not channel** (LJ is one speaker; ITW is celebrities): the per-feature positions and the grouped CV make it visible; the report states it.
- **E runtime on CPU** (three deep models, ~3,600 clips): capped by sampling; if it runs long, MLAAD drops to 300 clips and the report says so.
- **Shared working tree:** other lanes have uncommitted files; stage by path only.
- **External drive**: remounted 12:15; a dropout fails extraction loudly (resumable caches).
