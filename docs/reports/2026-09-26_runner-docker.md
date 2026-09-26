# Runner, API and Docker status (Sat Sep 26, 2026, runner lane; written 07:35, v2 at 08:35)

## v2 (08:35): the shipped fusion rule changed to `e_on_a`; parity re-established

The pre-declared sweep (`docs/reports/2026-09-26_fusion-sweep-predeclared.md`, commit 299cab3)
replaced zmean with **"E on A alpha 0.2"**, persisted in `models/fusion_v1/constants.json` by
`scripts/fuse_sweep.py --write`. The runner now defaults to it (`--rule e_on_a`, the constants
file's `final` = `E_on_A_alpha0.2`):

1. `rank_d = searchsorted(inner_oof_sorted[d], logit_d) / len` for m1b_v3 and handcrafted_v5;
2. `base = 0.8 * rank_m1b + 0.2 * rank_hc`;
3. M3 as false-alarm suppression only: if the spectra_aasist margin < -3 and base > 0.5, base *= 0.5
   (M3 never promotes; nothing is fit on M3; a missing M3 means no suppression);
4. `p = sigmoid(a * base + b + logit(0.3))` with the file's Platt map;
5. the determinate map `0.001 + 0.999 * p` and the pinned block below 0.001 for gated files are
   the **policy**, done by `hearsay.detectors.speech_gate.apply_default_answer` (e2d5291) and
   applied **exactly once**: `FusionConstants.fuse` returns the Platt `p` (`fusion.p_fused`), then
   `final_score` calls `apply_default_answer(p, is_speech, order_by=sigmoid(m1b LLR), keys=[filename],
   failed=[decode failure])`. Gated files land in [0.0001, 0.001) ordered by the weak M1b signal
   with a hash jitter of the filename; decode failures and missing files below 0.0001.
   `tests/test_pipeline.py::test_analyze_with_the_shipped_rule_applies_the_map_exactly_once` pins
   the single application; the exact test below would drift by ~1e-3 if it were applied twice.

`--fusion models/fusion_v0/constants.json --rule zmean|stack_nonlj` still selects the 06:02 rules;
a rule the chosen file does not define is refused before scoring (`rule 'e_on_a' is not defined
by models/fusion_v0/constants.json (it has ('zmean', 'stack_nonlj'))`). `--flip` emits the
pre-flipped variant (1 - p through the same map, block still at the minimum) as
`<team>_predictions_FLIPPED.tsv`. `--policy none` disables the block only; the determinate map is
step 5 of the rule and always applies, so scores are in [0.001, 1] in every mode. The cache
identity already covered the constants file's sha, so a `--out` scored under fusion_v0 is moved
aside and rescored; `--flip`, like `--rule` and `--policy`, re-fuses cached logits in 0.1 s.
Every pre-existing flag and environment variable is unchanged. The API defaults to `e_on_a`
(`HEARSAY_RULE`; `HEARSAY_FUSION=models/fusion_v0/constants.json` for the old rules) and answers
500 with the file's rule list when asked for a rule the file lacks.

**Parity v2** (Mac, CPU, 50 template files, gate on: 0 gated, 0 decode errors; a busier machine
than at 06:50, 1.7 s/file):

| Comparison | Spearman | max abs diff | mean abs diff |
|---|---|---|---|
| exported logits -> fusion_v1 constants -> policy vs `20260926-0813_..._our_direction.tsv` (1,671 rows, test at 1e-6) | | 4.4e-16 | |
| same, `--flip`, vs `20260926-0813_..._FLIPPED_only_if_NSA_scores_inverted.tsv` | | 4.4e-16 | |
| live from audio, `e_on_a`, vs the our-direction TSV | 1.000000 | 2.7e-5 | 6.5e-7 |
| live from audio, `--flip` (re-fused from the cache), vs the FLIPPED TSV | 1.000000 | 2.7e-5 | 6.5e-7 |
| live from audio, `--fusion fusion_v0 --rule zmean`, vs the 06:02 zmean TSV | 1.000000 | 1.0e-3 | 7.7e-4 |

The zmean row's 1.0e-3 is the determinate map itself (the 06:02 TSVs predate it: p vs
0.001 + 0.999 p differs by at most 0.001); the underlying Platt p still matches to 2.4e-5 (v1
section below). The rank rule is far less sensitive to the CPU-vs-MPS logit noise than the z
rules were: a 5e-4 M1 logit difference moves a rank only when it crosses one of 16,142 reference
values, hence 2.7e-5 on p (one or two files crossing a step) with a mean of 6.5e-7. Per-detector
logit parity is unchanged from the 06:50 table (same score paths).

Commits: see the end of this file's commit list (v2 commit noted in the handoff message).

---


Owner of this lane: the runner chat. Files: `src/hearsay/pipeline.py`, `scripts/run_pipeline.py`,
`src/hearsay/api.py`, `tests/test_pipeline.py`, `tests/test_api.py`, this report. The Docker image
itself (`Dockerfile`, `.dockerignore`, `docker/*`) is the K Docker chat's (run spec
`docs/specs/2026-09-26_k-docker-image.md`); this lane only checked that its image runs this runner.

Commits: `1394ca0` runner, `b97bbb0` preflight tolerance, `6c2d1ca` API. Suite after them:
`uv run pytest -q` 416 passed (359 at 06:20), `uv run ruff check .` clean.

## What was built

**`hearsay.pipeline`** — one clip in, one `AnalyzeResponse` out (frontend contract v0,
`docs/handoffs/2026-09-26_frontend-contract.md`), through `analyze_clip(ctx, models, consts, rule)`:

- decode once into a `ClipContext`; every registered engineered detector through `safe_run`
  (`import hearsay.detectors.engineered`: compression, container, enf, handcrafted, speaker_drift,
  speech_gate, splice), each timed, with roles fused / routing / gate / evidence;
- **m1b_v3**: `prepare_segment -> embed_segment -> Probe.llr` on the frozen XLS-R backbone, the
  probe at `models/m1_wav2vec2-xls-r-300m_L7_20260926-0521`. The probe's own `segment=False` flag
  is not consulted: it was wrong (train_probe.py looked for `extract_meta.json` under the
  comma-joined train name; fixed upstream in `541afdb`), and the export was produced from
  segment-mode embeddings. The embedding is rounded through float16, the dtype the extracted
  shards were stored in (`M1_EMBED_FP16`, chosen by parity, table below). The encoder is
  truncated to `layers[:8]` (`truncate_backbone`): `hidden_states[7]` is bit-identical on the real
  backbone (max |d| = 0.0 on two test files, `tests/test_pipeline.py::test_truncated_backbone_keeps_layer_7_identical`),
  M1 per file 0.82 s -> 0.28 s; `--no-truncate` keeps all 24 layers;
- **spectra_aasist**: `hearsay.spectra.prepare_input -> score_clip(..., ("zero",), 16)`, no peak
  normalization, `synth_logit`: the published export's configuration
  (`outputs/detector_scores/spectra_aasist_summary.json`);
- **handcrafted_v5**: the registered detector pinned to `models/hc_selected`
  (-> `hc_lgbm_20260926-055451`, the bundle behind `handcrafted_v5.csv`); its `hc_logit` feature
  is the export's logit;
- **fusion**: `FusionConstants.load(models/fusion_v0/constants.json)` (written by
  `scripts/fuse.py`, never refit here): `z = (logit - mean) / std` per detector from the inner-OOF
  rows, `zmean` = mean of z (default) or `stack_nonlj` = weights . z + intercept, then
  `p = sigmoid(a * fused + b + prior_shift)` with that rule's Platt map. A fused detector that
  errored is imputed at z = 0 (its inner mean) and named in the routing log. `--detectors m1b`
  is the probe-only mode: `p = sigmoid(LLR + logit(0.3))`, `scripts/make_probe_csv.py`'s arithmetic;
- **gate**: `speech_gate.apply_default_answer` after fusion (`--policy none` leaves the fused
  probability alone, for parity against pre-gate TSVs). A decode failure or a template name with
  no file is gated the same way (`undetermined`, flag `decode_error` / `missing_file`);
- **routing_log** from the features: container `lossy` / `is_pcm_wav` / `ffmpeg_written`,
  compression `bw_hz` against the 7.25 kHz band match, enf `enf_present` / `enf_stable`, splice
  `n_seams`, speaker_drift `drift` / `cos_min`, speech_gate `is_speech` / `voiced_frac`, the fusion
  rule and any imputation; **version**: git sha (`HEARSAY_GIT_SHA` in the image), probe dir,
  resolved hc bundle, Spectra id, constants file, rule, policy, truncation depth.

**`scripts/run_pipeline.py`** — the Docker entrypoint's runner, also the Mac-side parity tool.

```
uv run python scripts/run_pipeline.py --in <audio dir> --out <dir> [--template <tsv>] [--team NAME]
    [--limit N] [--device cpu] [--rule zmean|stack_nonlj] [--policy speech_gate|none]
    [--detectors m1b|m1b,spectra,handcrafted] [--fusion|--constants <constants.json>]
    [--m1-mode segment|windows|auto] [--no-truncate] [--fresh] [--threads N]
    [--require-offline] [--app-root DIR] [--probe DIR] [--hc DIR] [--no-preflight] [--no-tsv]
    [--compare-tsv <logged.tsv>]
env: HEARSAY_TEAM HEARSAY_TEMPLATE HEARSAY_RULE HEARSAY_POLICY HEARSAY_FUSION HEARSAY_DETECTORS OMP_NUM_THREADS
```

- Rows: the template's order (`filename` column, tab-only parse, duplicates fail before scoring);
  each name resolves as a relative path under `--in`, else by basename against a recursive index
  (nested layouts; an ambiguous basename fails before scoring); a name with no file keeps its row.
  Without a template: sorted audio files. An empty listing fails with no TSV.
- Outputs under `--out`, never `submissions/`, never `append_log`: `<team>_predictions.tsv` via
  `hearsay.submission.write_submission` (a stamped name if it exists), `results/<filename>.json`
  (one AnalyzeResponse per file), `results.jsonl` (the resumable cache; its header line records
  git sha, probe, hc bundle, constants sha, scorers, m1 mode, fp16, truncation, and a resume
  requires an exact match, else the old cache is moved to `results.jsonl.stale-<stamp>`;
  `--fresh` forces that), `<team>_predictions.sidecar.csv` (per file: p, flag, is_speech, the
  three logits, fused), `timings.csv` (per file and per stage), `run_meta.json`, `preflight/`.
- A rerun with another `--rule` or `--policy` re-fuses the cached logits without loading a model
  (0.1 s for 50 files) and rewrites the per-file JSON.
- Preflight before the first file: silence and a chord through the whole pipeline twice; both
  finite, the two passes within 1e-6 (`preflight_consistent`; multithreaded x86 BLAS is not
  bit-reproducible: the amd64 image differed by 8e-10), both gated at the default answer.

**`hearsay.api`** — `uv run uvicorn hearsay.api:app --port 8000`. `POST /analyze` (multipart
`file` -> AnalyzeResponse; models loaded once on first use, one clip at a time),
`GET /results/{filename}` and `GET /results` (a runner `--out` given by `HEARSAY_RESULTS`),
`GET /health`. Honours `HEARSAY_RULE`, `HEARSAY_POLICY`, `HEARSAY_FUSION`, `HEARSAY_DETECTORS`.
Off the scoring path; not in the image.

**Tests** — `tests/test_pipeline.py` (31: toy constants in fuse.py's file layout, the two rules'
arithmetic to 1e-15, imputation, response shape and roles, routing log, default answer, gate
error, decode failure, re-fuse, M1-only, template listing, cache identity, preflight tolerance,
truncation on tiny stable/non-stable configs; `needs_data` exact checks that the exported logits
through `constants.json` reproduce both logged TSVs to 1e-6 (measured 1.1e-16); one
`slow`/`needs_weights` truncation identity test on the real backbone) and `tests/test_api.py` (7,
fake models behind `/analyze`, a temp results dir behind `/results`).

## Parity (Mac, CPU, 6 threads, 50 template files, gate on: 0 gated, 0 decode errors)

| Comparison | Spearman | max abs diff | mean abs diff |
|---|---|---|---|
| live p vs logged `20260926-0602_M4_fusion_stack_nonlj_m1bv3_hcv5_m3.tsv` | 1.000000 | 2.4e-7 | 8.9e-9 |
| live p vs logged `20260926-0602_M4_fusion_zmean_m1bv3_hcv5_m3.tsv` | 1.000000 | 2.4e-5 | 2.3e-6 |
| fusion from the exported logits through `constants.json` vs both TSVs | | 1.1e-16 | |
| `--detectors m1b --policy none` vs logged `20260926-0522_M1_..._0521.tsv` (20 files) | 1.0 | 1.5e-5 | 2.0e-6 |

Per detector, live logit vs the export fusion was fit on (`outputs/detector_scores/*.csv`, test rows):

| Detector | max abs dlogit | mean | note |
|---|---|---|---|
| m1b_v3 (fp16 rounding, default) | 5.5e-4 | 1.1e-4 | CPU fp32 vs the MPS-extracted, fp16-stored shards |
| m1b_v3 without fp16 rounding | 4.7e-3 | 1.7e-3 | 10x worse, so fp16 stays the default |
| m1b_v3 via the `windows` path | 3.95 | 0.95 | confirms segment mode is the export's path |
| handcrafted_v5 | 4.4e-16 | 1.8e-17 | identical (numpy trees, same features) |
| spectra_aasist | 1.1e-4 | 9.3e-6 | CPU vs MPS |

zmean's Platt slope (11.7) amplifies the M1 noise more than stack_nonlj's (1.76), hence 2.4e-5
vs 2.4e-7 on p. Both are far inside the 0.05 / 0.99 gate.

## Timing (Mac, CPU)

| Setting | s/file | m1b_v3 | spectra | engineered (7 detectors) |
|---|---|---|---|---|
| 6 threads, all 24 layers (first 50-file run) | 2.31 (median 2.17, max 6.0) | 0.82 | 1.04 | 0.35 |
| 6 threads, truncated to layer 7 | 0.77 | 0.28 | 0.36 | 0.27 |
| 4 threads, truncated, full test set (1,671 files) | 0.78 (median 0.68, p95 1.25, max 5.7) | 0.14 | 0.33 | 0.23 |

Model load: 5-8 s (XLS-R 0.5 s, Spectra 1.7-4.9 s). Full test set, 1,671 files, 4 threads,
`--rule zmean`, gate on: **1,319 s wall (22.0 min)**, 1,306 s scoring, 8 s model load, 0 gated,
0 decode errors, share > 0.5 = 0.2932 (the logged zmean TSV's share), and vs the logged zmean TSV
over all 1,671 rows: Spearman 1.000000, max abs diff 1.2e-4, mean 2.3e-6, 0 rows over 0.01.
Outputs (gitignored): `outputs/runner/full_20260926/` with `HEARSAY_predictions.tsv`,
`results/<filename>.json` for every test file (the frontend's static dump, every field
populated including `fusion.weights` and the full `routing_log`), `results.jsonl`, the sidecar,
`timings.csv` and `run_meta.json`.

API latency (`curl -F file=@... localhost:8000/analyze`, 6 threads): first request 11.4 s
(7.8 s model load + warm-up), then 1.7 s (HGT1046947) and 1.1 s (HGT1013455);
`GET /results/<filename>` in milliseconds.

## Docker

Not built by this lane: the K Docker chat owns `Dockerfile`, `.dockerignore`, `docker/*` and
was building concurrently (images `hearsay:20260926-0637`, `hearsay:20260926-0648` = `latest`,
linux/amd64, 2.66 GB content / 7.49 GB on disk, `BUILD_INFO` sha=9301666 with the 0521 probe,
`hc_lgbm_20260926-055451`, `cmp_lgbm_20260926-0233`, `fusion_v0/constants.json`; inside,
`platform.machine()` = `x86_64` under Rosetta). Their `docker/build.sh` runs
`docker buildx build --platform linux/amd64 --load` and retags `hearsay:latest`; running it from
this lane while they rebuild after "runner final" would have raced their tags and their VM's six
CPUs, so this lane did not run it (the main chat's instruction: run it only if it works with this
runner, otherwise stop). What this lane did check on their `hearsay:latest`, `--network none`,
3 test files, a reversed template, `--no-preflight` (that image predates `b97bbb0`, so its
preflight still compares the two passes with `==`): the image decoded and scored two of the
three files, then died on the third with `OSError: [Errno 5] Input/output error: 'ffmpeg'` while
containerd logged `write /var/lib/containerd/.../meta.db: input/output error` (exit 125): the
Colima VM disk on the external drive, not the runner (the same files score on the Mac). The two
files that did score, vs the logged zmean TSV: HGT1046947 0.685507 vs 0.685650 (1.4e-4),
HGT1013455 0.00089800 vs 0.00089792 (8e-8). Under Rosetta emulation the image took 22-28 s per
file (m1b_v3 7.6-9.6 s, spectra 8-10 s), about 30x the Mac's 0.77 s: an emulated full-set run
would take about 11 h, a native amd64 box would not pay that. So the image runs this runner end
to end (listing from a mounted template, decode, all detectors, fusion from the shipped
constants, gate) and the remaining image checks are blocked on the VM disk and the rebuild.

Their entrypoint execs `python /app/scripts/run_pipeline.py --in /data --out /out --threads N "$@"`
with `HEARSAY_TEAM` / `HEARSAY_TEMPLATE`; every flag and variable it relies on is unchanged in
`b97bbb0`/`6c2d1ca`, and the additions (`--detectors`, `--policy`, `--fusion`, `--m1-mode`,
`--no-truncate`, `--fresh`, `--require-offline`, `--app-root`) are additive.

## Findings worth knowing

- The 0521 probe's `segment` flag was False for the wrong reason (comma-joined `--train` name);
  `scripts/make_probe_csv.py` would have scored it through the tiled 4 s window path, which is
  off by up to 3.95 in LLR against the export. The runner never trusts the flag.
- The extracted embeddings are float16 on disk; rounding the live embedding the same way is
  what makes M1 parity 5e-4 instead of 5e-3.
- Encoder truncation is safe: both Wav2Vec2 encoder variants append `hidden_states[k]` before
  running layer k, so `layers[:k+1]` leaves it untouched (bit-identical on the real backbone).
- Two runs of the same clip differ by ~1e-9 on x86 with 6 BLAS threads; anything that asserts
  `==` on scores (including `hearsay.submission.preflight`) will fail in the amd64 image.
- `speaker_drift` (ECAPA, SpeechBrain) runs in the pipeline as evidence only and costs
  0.1-0.4 s per file; the ENF detector emits a numpy divide-by-zero warning on digital silence
  (harmless: the SNR of an all-zero frame).

## Not done

- No image built by this lane (above). The in-image 50-file parity, the 3-file smoke with the
  fixed preflight and the 1,671-file in-image wall time are the K chat's checks with its rebuild.
- The full-set run above was done on the Mac (22 min), not in the image.
- The API is not in the image and has no auth or rate limiting (demo only).
- The runner scores files one at a time (decode + engineered detectors + two deep models on
  one thread pool); a decode/engineered prefetch thread would overlap the ~0.3 s of CPU
  detectors with the deep models and buy ~30%.
