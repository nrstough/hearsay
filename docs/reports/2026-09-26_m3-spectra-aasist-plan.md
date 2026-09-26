# Plan: M3, Spectra-AASIST score stream (Sat Sep 26, 2026)

Run spec: `docs/specs/2026-09-26_m3-spectra-aasist.md` (P1/P2 frozen, deep mode). Handoff: `docs/handoffs/2026-09-26_m3-spectra-aasist-handoff.md`. Worktree: `main` in `~/Projects/hearsay`, shared. Stage only the files in step 9; never `git add -A`.

Deep-mode inputs: an architecture survey, a file-impact survey with current line numbers, a risk analysis (24 items), a critique pass (13 findings, 6 critical) on the first draft of this plan, and the Codex plan review (`docs/reports/2026-09-26_m3-spectra-aasist-plan-review.md`, 9 findings, 6 critical) on the second draft. All are folded in; the ones that changed the design are listed under "Amendments to the run spec" (13–20 are the Codex resolutions).

## Amendments to the run spec (recorded here and, dated, in the spec before commit)

1. **Chunked decode pool.** `ThreadPoolExecutor.map` over all rows submits every task eagerly; decode (~20 ms) outruns the model (~0.2 s), so up to 20,000 float32 segments would sit in RAM on an 18 GB box already running the main chat's XLS-R job. Map per 500-row chunk, aligned with the checkpoint cadence, the way `scripts/extract_embeddings.py:78-102` maps per shard.
2. **Raw companion moves to `outputs/spectra/<out-name>_raw.csv`**, five base columns first. A second header shape inside `outputs/detector_scores/` would break any future `glob("detector_scores/*.csv")` consumer.
3. **Pilot scores all pad modes in one decode pass** (`--pad-modes repeat,zero,whole`): each pilot clip is decoded and prepared once; clips at or above one window get one forward shared by all modes, short clips one forward per mode. Model loaded once. **Pilot rows are inner rows and test rows only**: holdout rows inside the head-N slice are skipped (the holdout is read once, after the full run). Fold side `--limit 400` (322 inner rows, 171 bona fide / 151 spoof), test side `--test-limit 200`; at 200 fold rows one bona fide false alarm would move normalized minDCF by 0.106, five times the tie band. **Tie-break order:** inner minDCF, then inner AUC, then `zero`.
4. **Abort on 25 consecutive decode errors** with a mount hint (the fold audio is on the external drive; an unmount would otherwise be discovered 75 min later at the 5% gate). `decode_error` and `model_error` are counted separately.
5. **M1 comparator rule.** The newest `models/m1_*` (0348) trained on `splits/nsa_folds_plus_asv19.csv`; the comparable run is the newest meta with `folds == "splits/nsa_folds.csv"`, a single `train` set and a `val_min_dcf` key (today: `m1_wav2vec2-xls-r-300m_L7_20260926-0347`, holdout 0.1458; a band-matched v3 directory landing during the run is picked by the same rule). Holdout keys are `val_min_dcf`, `val_eer`, `val_by_generator`, `val_by_bonafide_source`.
6. **Secondary tripwire.** Primary test readout: share of `synth_logit > 0` (score > 0.5). Secondary: share with `logit_bonafide ≤ −1.0625009`, the model card's `classify` threshold (`weights/Spectra-AASIST/model.py:785`; the README text says −1.140625, discrepancy noted).
7. **Forward guard.** `forward_windows` raises `ValueError` when `w.shape[1] > 128,000`: the AASIST positional table covers 400 SSL frames, frame 401 appears at 128,400 samples, and the 8 s cap leaves 80 samples of slack. (`ValueError`, not `assert`: `python -O` strips asserts.)
8. **Resume.** Config hash covers sha256 of the fold file, test manifest, **training manifest (its row order is the crop recipe)**, durations file, the stress manifest in manifest mode, the sources of `hearsay/spectra.py`, `hearsay/embed.py`, `hearsay/audio.py`, `hearsay/handcrafted.py` (the input path belongs to the main chat and changed tonight), `scripts/score_spectra.py` itself, and `weights/Spectra-AASIST/model.py` + `config.json`; pad mode, band match, max windows, seed, crop policy, device, torch and torchaudio versions, `model.safetensors` size and mtime. Partial files are written to `.tmp` then `os.replace`. pandas 3 reads an empty `flag` cell as NaN: `fillna("")` before comparing. **Resumed rows reuse their stored `synth_logit` verbatim** (never recomputed from the parsed float64 columns; `synth_logit` is a float32 subtraction). **On completion the partial and its sidecar are deleted**; `--resume` with no partial exits 2 ("nothing to resume") so a finished run is never silently re-exported under a new stamp. `--limit` and `--manifest` runs keep any checkpoint inside their own output directory and never touch `outputs/spectra/<out-name>_partial.csv`.
9. **`--repo-root PATH`** (default the repo). Every output path and every relative-path resolution derive from it, so the hermetic tests run against a fake repo in `tmp_path` and can never write into the real `outputs/` or `models/` (the post-commit audit runs the test suite while the real export may be running).
10. **Stress thresholds are an explicit sweep** (see step 2, `stress_readout`), and `--stress` is accepted in pilot mode too, labeled provisional, so an In-the-Wild readout exists before the full run's inner rows do.
11. **Direction outside the full run is recorded, never gated:** per mode in the pilot table, and in manifest mode.
12. **Build box.** The frozen P2 test list is a 2–3 h build, not 45 min. Build order (step 4) puts the gates first so the box can close with a scorer that refuses to ship a bad file; diagnostics (`platt_diagnostic`, `level_shortcut`, the band-off run, W1) are last and are computable post hoc from the raw companion. Whatever is cut is recorded in the spec's Results.
13. **Non-finite model outputs are failures** (Codex 2). `score_clip` validates the forward output: shape `(n, 2)`, every logit finite, and the derived `synth_logit` finite; anything else is flag `model_error`, NaN in the export, and counted. A NaN-returning fake model is a test (B8).
14. **One finite-row mask for every readout** (Codex 3). A helper `finite_rows(y, s, frame)` returns the mask, `n_used` and `n_dropped`; every readout (`report`, `by_group`, `direction_check`, `sweep_thresholds`, `platt_diagnostic`, `level_shortcut`, `stress_readout`) applies it first and records the two counts, with `{"status": "skipped_one_class"}` / `{"status": "empty"}` instead of an exception. Tests plant a decode failure in an inner, a holdout, a test and a stress row (D11).
15. **Per-split failure gate and coverage** (Codex 7). The 5% gate applies to inner, holdout and test separately (and to the stress set in manifest mode); meta records failures per split and, on fold rows, per class and per generator, so selectively missing hard rows cannot flatter a readout. C8 plants 10% failures inside the test rows with the overall rate under 5% and expects exit 3.
16. **Publication order and run identity** (Codex 4). Everything is computed into the stamped run directory first: `models/m3_spectra_<stamp>/{raw.csv, fusion.csv, meta.json}` (immutable run artifacts). Only after the gates and the meta succeed is the fusion file published to `outputs/detector_scores/<out-name>.csv` by `.tmp` + `os.replace`, together with `outputs/detector_scores/<out-name>_summary.json` (stamp, config hash, git sha, counts, holdout minDCF; the same sidecar pattern as `container_summary.json`). The raw companion under `outputs/spectra/` is a copy of the run directory's `raw.csv`. A failed rerun therefore never half-writes the shared file, and the summary sidecar names the run that produced it.
17. **Readout-only mode** (Codex 5). `--readout inner=<scores.csv> stress=<name>=<scores.csv> [--out PATH]` computes `stress_readout` (thresholds from the given inner scores) and writes a JSON, no inference. Step 7 uses it with the pilot's selected-mode scores and the In-the-Wild scores to post the provisional P_FA before the full export; the full run's meta then recomputes it from the full inner rows. D9 exercises exactly this workflow on saved fake files.
18. **`--max-chunks N`** (Codex 8): stop after N chunks, keep the partial, return 5. It is how C14 makes an interrupted run without a test-only hook, and it doubles as a dry run. C10 asserts the set of paths this scorer creates under `--repo-root` (and that `detector_scores/`, `models/` and the full-run partial are absent after a pilot); it no longer snapshots the real repo trees, which other chats legitimately write to.
19. **Completion condition** (Codex 6). M3 produces no TSV by design (handoff: TSVs and `submissions/` belong to the main chat). The rung is complete when the published file, its summary sidecar, the run stamp, the holdout numbers and the "fusion reads `logit`" note have been sent to the main chat and recorded in the run spec's Results with the time; if the export cannot finish by the hard stop, the Results name that as the blocker.
20. **Laundering checks are deferred** (Codex 9). Band match and the In-the-Wild readout do not establish robustness to codec laundering, telephony or simulated replay; nothing is trained here, so there is no augmented-versus-clean pair to report. The report lists this next to physical replay as a limitation of a scoring rung, and points at M5's laundering augmentation as where it is measured.

## Step 0. Pre-flight

```bash
cd ~/Projects/hearsay && git branch --show-current && pwd
git status --short        # today: docs/reports/2026-09-26_handcrafted-v4.md (CPU chat, untracked) plus the M3 spec and plan; hc_v4 and M5 were committed (9288d73, 982aa11)
uv run pytest -q 2>&1 | tail -3   # baseline: 301 collected at last look; record the pass count and any pre-existing failures
uv run ruff check .
```
Record the baseline in the run spec's Results (the spec's "215 passing" predates tonight's commits). Do not touch `src/hearsay/embed.py`, `src/hearsay/audio.py`, `src/hearsay/handcrafted.py`, `scripts/spectra_direction.py`, `scripts/train_handcrafted.py`, `splits/*`, `submissions/*`, or anything under another chat's lane.

## Step 1. `src/hearsay/metrics.py`: add `by_group` (additive)

Current state: 106 lines, ends with `report()` at lines 95-106. Imports at 18-23 (`annotations`, `math`, `numpy`, `roc_curve`); pandas is not imported. The file is imported by the CPU trainer and the M5 scripts, so the change is strictly additive.

- After line 23 add a type-only import so ruff does not flag the annotation:
  ```python
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      import pandas as pd
  ```
- Append after line 106, copied from `scripts/train_handcrafted.py:83-96` (the function moved down after the hc_v4 commit; 75-81 is now `decision()`), with the signature `by_group(dv: pd.DataFrame, yv: np.ndarray, s: np.ndarray, pi: float = PI_SYNTH) -> tuple[dict, dict]`; the only change is the default for `pi`. Docstring keeps "minDCF/EER per generator (its spoofs vs every bona fide clip) and per bona fide source (its bona fide clips vs every spoof)" and adds "Copied here from scripts/train_handcrafted.py for M3; the trainer keeps its own copy."
- `min_cost` and `eer` are already in scope. No LightGBM anywhere near this file.

## Step 2. `src/hearsay/spectra.py`: extend (current 63 lines)

Keep lines 1-50 (`load_spectra`) and 62-63 (`synth_logit`) as they are. Replace lines 53-59 (`spectra_logits`) with the block below and add the new helpers. Update the module docstring (lines 1-10) to describe the three pad modes, the shared input path and the raw export.

New imports: `from hearsay import SR`, `from hearsay.embed import prepare_segment, test_duration_sampler` (this pulls `transformers` at import time via `embed.py:16`; acceptable for a scorer, a few seconds for the test suite), `from hearsay.metrics import C_FA, C_MISS, PI_SYNTH, eer, min_cost, sigmoid`, `import pandas as pd`, `from sklearn.metrics import roc_auc_score, roc_curve`.

New constants after line 27:
```python
PAD_MODES = ("repeat", "zero", "whole")
MAX_INPUT_SAMPLES = 128_000   # AASIST pos_T covers 400 SSL frames; frame 401 appears at 128,400
MIN_WHOLE_SAMPLES = SR        # 1 s floor for whole mode (Encoder max_pool2d(3,3) needs >= 3 frames)
```

Functions, in file order:

```python
def prepare_input(x, crop_s=None, seed=None, band_match=True):
    """The shared M1 path: band match -> trim -> optional crop -> cap 8 s (hearsay.embed.prepare_segment)."""
    return prepare_segment(x, crop_s, seed, band_match=band_match)


def pad_windows(x, pad_mode="repeat", max_windows=16):
    """(n, L) float32 windows. Clips >= WIN: 50%-hop windows, identical in every mode.
    Shorter clips: repeat = tile to WIN (model-card pad_random; the tiling seam),
    zero = zero-pad to WIN, whole = the clip at its own length with a 1 s zero-pad floor."""
    if pad_mode not in PAD_MODES:
        raise ValueError(f"pad_mode must be one of {PAD_MODES}, got {pad_mode!r}")
    x = np.asarray(x, dtype=np.float32)
    if x.size >= WIN or pad_mode == "repeat":
        return windows(x, WIN, HOP)[:max_windows]          # today's path, byte-identical (B5; verified for empty, 1-sample, short, long, int16 inputs)
    if pad_mode == "zero":
        return np.pad(x, (0, WIN - x.size))[None]
    if x.size < MIN_WHOLE_SAMPLES:
        x = np.pad(x, (0, MIN_WHOLE_SAMPLES - x.size))
    return np.ascontiguousarray(x[None])


def pad_samples(n_samples, pad_mode) -> int:
    """Samples added by pad_windows for a clip of n_samples in this mode (0 for clips >= WIN)."""
    if n_samples >= WIN:
        return 0
    return WIN - n_samples if pad_mode in ("repeat", "zero") else max(0, MIN_WHOLE_SAMPLES - n_samples)


@torch.inference_mode()
def forward_windows(model, w):
    """(n, 2) logits for equal-length windows. Pre-emphasis on CPU before .to(device): the order
    scripts/spectra_direction.py's ASVspoof numbers were produced with."""
    if w.shape[1] > MAX_INPUT_SAMPLES:
        raise ValueError(f"{w.shape[1]} samples exceed the 400-frame AASIST positional table (8 s)")
    device = next(model.parameters()).device
    t = torchaudio.functional.preemphasis(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)))
    return model(t.to(device)).float().cpu().numpy()


@torch.inference_mode()
def spectra_logits(model, x, max_windows=16, pad_mode="repeat"):
    """Window-averaged (logit_spoof, logit_bonafide) for one 16 kHz mono clip."""
    return forward_windows(model, pad_windows(x, pad_mode, max_windows)).mean(axis=0)


def score_clip(model, x, pad_modes=("repeat",), max_windows=16) -> dict:
    """Per-mode logits for one prepared clip: {"n_samples", "n_windows", "<mode>": (2,) logits,
    "n_pad_samples_<mode>": int}. Clips >= WIN forward once and every mode gets the same logits."""


def crop_plan(manifest_csv, seed) -> dict[str, tuple[float, int]]:
    """path -> (crop_s, offset seed): one test-duration draw per manifest row in order from
    test_duration_sampler(seed), offset seed = seed + row index. Reproduces
    extract_embeddings --segment --crop test --seed <seed> and extract_handcrafted exactly.
    Draws for every row before any --limit so a pilot crops each file as the full run does.
    (nsa_train_sample.csv has 20,000 unique paths in the fold file's order; verified.)"""


def direction_check(y, s) -> dict:
    """AUC and class medians on finite rows; ok iff AUC > 0.5 and spoof median > bona fide median.
    One class or no finite rows -> {"status": "skipped_one_class", "ok": None}."""


def resolve_path(p, repo) -> str:      # score_detector.py:54 idiom; `repo` is always passed explicitly (--repo-root)
    return p if Path(p).is_absolute() else str(Path(repo) / p)


def fusion_frame(path, fold, split, synth) -> pd.DataFrame:
    """path, fold, split, score = sigmoid(synth), logit = synth. No calibration (spec D6).
    with np.errstate(over="ignore"); NaN stays NaN in both columns; raises on duplicate paths."""


def sweep_thresholds(y, s) -> dict:
    """Thresholds from inner rows only: fpr, tpr, thr = roc_curve(y, s);
    c = C_FA*fpr*(1-PI_SYNTH) + C_MISS*(1-tpr)*PI_SYNTH; thr_dcf = thr[argmin(c)];
    thr_eer = thr[argmin(|fnr - fpr|)] (scripts/spectra_direction.py:38-42).
    sklearn sets thr[0] = inf and min_cost clamps to the constant decision, so an argmin at index 0
    or a sweep that never beats the constant decision yields None plus a flag, never Infinity in JSON."""


def stress_readout(inner_y, inner_s, y, s, frame) -> dict:
    """minDCF, EER, AUC, by_group where the frame has >1 generator or >1 bona fide source, and
    P_FA = mean(s_bona >= thr), P_miss = mean(s_spoof < thr) at thr_dcf and thr_eer from
    sweep_thresholds(inner_y, inner_s) (sklearn's >= convention, as hearsay.metrics.cost_at)."""


def platt_diagnostic(y, s) -> dict:    # class-balanced LogisticRegression on finite inner rows only; {"in_sample_diagnostic": True, a, b, ...}
def level_shortcut(peak, s, y) -> dict:   # within-class Spearman(peak, synth_logit) on inner rows, scipy.stats.spearmanr, lazy import
def git_sha() -> str:                    # the scripts/m5_build_bundle.py:61-65 pattern, "unknown" on failure
```

The loop with the checkpoint files lives in the script (step 3); the pure per-clip pieces above are what the hermetic tests hit directly.

## Step 3. `scripts/score_spectra.py` (new)

Docstring: purpose, the three modes of use (full export, `--limit` pilot, `--manifest` stress), output files, a `Usage:` block in the repo style. Tests load this file with `importlib.util.spec_from_file_location` and call `run(argv, loader=fake_loader)`; nothing in the repo imports `scripts/` (the cloud tests insert `scripts/cloud` on `sys.path` for the same reason), and this stays that way.

Arguments (`argparse`, booleans via `argparse.BooleanOptionalAction`):
`--repo-root` (default `Path(__file__).resolve().parents[1]`), `--folds` (`splits/nsa_folds.csv`), `--test-manifest` (`outputs/manifests/nsa_test.csv`), `--train-manifest` (`outputs/manifests/nsa_train_sample.csv`, the crop-draw order), `--device` (default `hearsay.embed.default_device()`; tests pass `cpu`), `--pad-mode` (choices `PAD_MODES`, default `zero`), `--pad-modes` (comma list, pilot only), `--band-match/--no-band-match` (default on), `--max-windows 16`, `--seed 0`, `--limit N` (fold rows), `--test-limit N` (test rows, default `--limit`), `--manifest PATH --name NAME --stress-seed 200`, `--stress name=path` (append), `--out-name spectra_aasist`, `--resume`, `--max-chunks N`, `--readout inner=PATH stress=NAME=PATH [--out PATH]`, `--prefetch 4`, `--chunk 500`, `--threads 4` (CPU: `torch.set_num_threads`), `--consecutive-decode-abort 25`. Relative defaults resolve under `--repo-root`.

`run(argv=None, loader=load_spectra) -> int`, exit codes: 0 ok; 2 usage/config (missing label column, path not in the crop plan, stale or missing partial); 3 failure-rate gate (per split) or consecutive-decode abort; 4 direction gate; 5 stopped by `--max-chunks` (partial kept).

1. **Rows.** Default mode: `f = pd.read_csv(folds)`, `t = pd.read_csv(test_manifest)`; `rows = concat([f.assign(split=where(fold=="holdout","holdout","inner_oof")), t.assign(fold="test", split="test", label=nan, generator=nan, source=nan)])` (`score_detector.py:50-53`). `--limit` takes `f.iloc[:N]` and `t.iloc[:test_limit]` after the crop plan is built, and in pilot mode drops holdout rows. Manifest mode: `m = pd.read_csv(manifest)`; no `label` column → return 2; labels `.str.lower().str.replace("-", "")` (`spectra_direction.py:53-54`); `fold = split = "stress"`. `resolved = [resolve_path(p, root) for p in rows.path]`; exported `path` stays verbatim (existing exports mix absolute fold paths and relative test paths).
2. **Crops.** Default mode: `plan = crop_plan(train_manifest, seed)`; every fold row must be in the plan (else return 2 naming the first missing path); test rows `None`. Manifest mode: `plan = crop_plan(manifest, stress_seed)`.
3. **Config hash and output dirs.** Pilot: `outputs/spectra/pilot/<stamp>_<name>/` (checkpoint inside it). Manifest mode: `outputs/spectra/<name>/` (checkpoint inside it). Full run: `outputs/detector_scores/<out-name>.csv`, `outputs/spectra/<out-name>_raw.csv`, `outputs/spectra/<out-name>_partial.csv` + `.json` sidecar with the hash, `models/m3_spectra_<stamp>/meta.json`. All under `--repo-root`. Stamp: `datetime.now().astimezone().strftime("%Y%m%d-%H%M")`.
4. **Model.** `model = loader(device)`; `band_limit(np.zeros(SR, np.float32))` once on the main thread (warms `hearsay.handcrafted._FIR`, whose two-statement lazy init races across decode threads and would yield a float64 filter for a few rows); `torch.set_num_threads(threads)` when device is cpu.
5. **Resume.** With `--resume`: no partial → return 2; sidecar hash mismatch → return 2 ("stale partial: delete it or drop --resume"); else read it, `flag = flag.fillna("")`, keep rows whose flag is `""` or `decode_error` with their stored values verbatim, re-score `model_error`.
6. **Loop** in chunks of `--chunk` over the rows still to score: `with ThreadPoolExecutor(prefetch) as pool: for r, x, peak, flag in pool.map(prep, chunk_rows)`, where `prep` does `load_audio` → `peak = float(abs(x).max())` → `prepare_input(x, crop_s, seed, band_match)`, returning flag `decode_error` on `DecodeError`; the main thread calls `score_clip` inside `try/except RuntimeError` → flag `model_error`; a forward whose output is not `(n, 2)` or not finite (logits or `synth_logit`) is also `model_error` (amendment 13). A run of `--consecutive-decode-abort` decode errors aborts with the mount hint (return 3, partial kept). After each chunk: append raw rows, write the partial (`.tmp` + `os.replace`), print `done/total  elapsed  s/clip  eta` (`extract_embeddings.py:110` style plus ETA).
7. **Gates.** Failure rate per split (`inner_oof`, `holdout`, `test`; `stress` in manifest mode): any split above 5% → raw file written into the run directory, counts printed per split, return 3, nothing published. Every readout goes through `finite_rows` first (amendment 14). Direction (full run only, inner rows only): `direction_check(y_inner, synth_inner)`; `ok` is `False` or `None` → raw file kept under `outputs/spectra/`, return 4, nothing under `detector_scores/` or `models/`. Pilot: direction recorded per mode; manifest mode: recorded. Neither gates.
8. **Outputs.** Full run, in this order (amendment 16): `models/m3_spectra_<stamp>/raw.csv` (five base columns then `logit_spoof, logit_bonafide, synth_logit, n_windows, n_samples, n_pad_samples, crop_s, peak, flag, seconds`), the gates, `models/m3_spectra_<stamp>/fusion.csv` via `fusion_frame` (row count equals folds + test rows; unique paths; header exact), `meta.json`; then publish: copy `fusion.csv` to `outputs/detector_scores/<out-name>.csv` and write `<out-name>_summary.json` next to it, both by `.tmp` + `os.replace`; copy `raw.csv` to `outputs/spectra/<out-name>_raw.csv`; finally delete the partial and its sidecar. Pilot: per-mode `scores_<mode>.csv` (base columns + raw columns with `n_pad_samples_<mode>`) and `meta.json` with the pilot table (per mode: inner minDCF, EER, AUC, direction, test share above half, whole-mode floor hits) and the rule's pick with the granularity note (one bona fide false alarm = 4·(1/n_bona)·0.7/0.3 in normalized minDCF). Manifest mode: `scores.csv` + `meta.json` (report, by_group where applicable, direction, timing).
9. **Meta** (`json.dumps(indent=2)`): `rung: "M3"`, model path and safetensors size, config (pad mode, band match, max windows, seed, crop policy text, device, torch/torchaudio versions, git sha, config hash), counts (`rows, ok, decode_error, model_error`, each per split, plus fold-row failures per class and per generator), timing (`seconds, sec_per_clip`), `holdout` = `report(y, synth)` + `auc` + `by_generator` + `by_source` (`by_group`), `inner` = the same (diagnostic), `test` = `n, frac_above_half (synth > 0), frac_bonafide_le_card_threshold, score_mean, score_std, distinct`, `direction`, `thresholds` (`sweep_thresholds` on inner rows), `platt_diagnostic`, `level_shortcut`, and for each `--stress name=path`: `stress_readout` on that scores file (`provisional: true` when computed from pilot inner rows).

## Step 4. `tests/test_spectra.py` (new, hermetic) and build order

Fixtures and helpers at the top of the file (no conftest in this repo):
- `_wav(path, x)` writes 16 kHz mono float32 WAVs with `soundfile` (a direct dependency; ffmpeg decodes them through `load_audio` as every other test does).
- `_clip(seconds, seed, amp)` white noise; `_tone(seconds, hz)` for A2.
- `FakeSpectra(torch.nn.Module)`: one parameter (`forward_windows` reads the device from it); `forward(x)` asserts `x.ndim == 2`, `x.dtype == torch.float32`, `x.device == self.w.device`, `not torch.is_grad_enabled()`, `not self.training`; records `x.shape` and a copy of `x` in `self.calls`; returns `torch.stack([gain * m, 0.1 * m], dim=1)` with `m = x.abs().mean(1)`, so louder = more spoof and the two logits are not exactly representable as a difference (that is what lets C14 catch a recomputed `synth_logit`); `gain=-1` flips direction for C9; a `raise_on_len` attribute raises `RuntimeError` for B6.
- `_load_script()` loads `scripts/score_spectra.py` via `importlib.util.spec_from_file_location`; every run passes `--repo-root <tmp_path> --device cpu`.
- A fake repo in `tmp_path`: `splits/nsa_folds.csv` (8 rows: folds 0, 1 and holdout, both labels, generators `g1`, `g2`, sources `src_a`, `src_b`, absolute paths), `outputs/manifests/nsa_test.csv` (4 rows, relative paths, `filename,path`), `outputs/manifests/nsa_train_sample.csv` (same 8 paths, same order), `splits/nsa_test_durations.csv` (a few values; `hearsay.embed.TEST_DURATIONS` monkeypatched to it). C10 snapshots the mtimes of the real `outputs/` and `models/` trees before a run and asserts nothing changed.

Tests by stage (IDs from the run spec plus the Codex additions): A1–A13, B1–B8, C1–C16, D1–D11, W1, about 52 plus W1. Notes on the non-obvious ones:
- A2: 7.8 kHz tone through `prepare_input` drops by more than 40 dB in RMS with defaults and is unchanged (bitwise) with `band_match=False`; `hearsay.handcrafted._FIR.dtype == np.float32` after a warm-up call.
- A4: `crop_plan` on the fake manifest with seed 0 gives lengths equal to `test_duration_sampler(0)` draws in order and seeds `0..n-1`; with the real `splits/nsa_test_durations.csv` present, the first five seed-0 draws are pinned to `3.552688, 3.25075, 3.041875, 3.1115, 3.3205` (skip if missing); a fake `--limit 2` run and a full fake run write identical `crop_s` for the shared rows; the In-the-Wild plan uses seed 200 and differs from seed 0.
- A5/A6/A7: 2 s clip; repeat → length WIN and autocorrelation at lag 32,000 above 0.9; zero → length WIN, tail zeros, autocorrelation below 0.05; whole → length 32,000; 0.3 s clip in whole → length 16,000; `pad_samples` per mode: 32,600 / 32,600 / 0 for the 2 s clip, 11,200 for whole on the 0.3 s clip; 128,000 samples pass `forward_windows`, 128,400 raise `ValueError`.
- A8/A9: 4.5 s, 6 s and 8 s clips give identical windows in all modes; counts **2, 2, 3** (6 s = 96,000 samples: starts `[0]` plus the flush window at 31,400) with the last window flush; `max_windows=1` caps.
- B3: a 40 s array fed straight to `spectra_logits` (bypassing the cap; 19 windows) → the fake sees exactly 16; a 12 s array with `max_windows=2` → exactly 2.
- A10: the captured model input equals `torchaudio.functional.preemphasis` of the windows; a DC step differs from the raw windows.
- A11: all-zeros 3 s, 0.1 s, and a float WAV with NaN samples → finite logits in all modes.
- B5: a frozen copy of the pre-change `spectra_logits` in the test file; equality on a 2 s (tiled) and a 6 s (two-window) clip; positional call `(model, x, 4)` still works.
- C8: 20 fake rows with 2 undecodable (10%) → exit 3 and no fusion file; 1 undecodable (5%) → exit 0; 40 rows with 2 undecodable both inside the 4 test rows (overall 5%, test 50%) → exit 3 (per-split gate).
- B8: a fake whose forward returns NaN for a marker clip → that row is NaN with flag `model_error`, counted, run continues; a fake returning shape `(n, 3)` → the same.
- C16: reorder two rows of the fake training manifest between a run and `--resume` → exit 2 (the crop recipe is in the hash).
- D11: one decode failure planted in an inner, a holdout, a test and a stress row → every readout still exists in meta with `n_used`/`n_dropped` matching; no exception.
- C9: `FakeSpectra(gain=-1)` → exit 4, no `detector_scores` file, raw file present under the fake repo; default gain → exit 0 and the file exists.
- C10: `--limit` run creates files only under `<root>/outputs/spectra/pilot/` (the created set is enumerated); `<root>/outputs/spectra/<out-name>_partial.csv`, `<root>/outputs/detector_scores/` and `<root>/models/` do not exist afterwards. The real repo is protected by construction (`--repo-root`), not by a snapshot.
- C14: fake full run (chunk 2) → fusion file A and summary sidecar; a second fake run with `--max-chunks 1` returns 5 and leaves a partial; `--resume` → fusion file B byte-equal to A; after completion the partial is gone and `--resume` returns 2; `--pad-mode` changed against a fresh `--max-chunks 1` partial → 2. Also: the published file appears only after `meta.json` exists in the run directory (a fake that fails on the last chunk leaves no published file).
- C15: after importing `hearsay.spectra` and loading the script, `"lightgbm" not in sys.modules`; no line of `scripts/score_spectra.py` or `tests/test_spectra.py` starts with the LightGBM import (the `tests/test_trees.py:69-75` rule extended to the script).
- D9: `--readout inner=<pilot scores> stress=itw=<manifest-mode scores>` on saved fake files (no inference); `thr_dcf` equals the hand-computed argmin; P_FA/P_miss recomputed by hand match; an inner set whose argmin is index 0 yields `None` and the flag, and the JSON has no `Infinity`.
- W1: `needs_weights` + `slow`; skip if `weights/Spectra-AASIST/model.safetensors` is missing; `load_spectra("cpu")`; 2 s and 6 s sines; finite `(2,)` logits in all three modes; the 2 s clip's modes differ from each other.

**Build order** (so the box can close with gates in place): (1) `metrics.by_group` + D1–D3; (2) `pad_windows`, `pad_samples`, `forward_windows`, `spectra_logits`, `score_clip`, `crop_plan`, `direction_check`, `fusion_frame` + A4–A9, A13, B1–B5, C4–C6; (3) the script: rows, crops, chunked loop, the per-split failure gate, non-finite outputs, the direction gate, the run directory and publication order, `--limit`/`--pad-modes`/`--test-limit`, `--repo-root` + C1–C3, C7–C10, C12, C13, B6–B8; (4) manifest mode + C11; (5) resume, hash, `--max-chunks` + C14–C16; (6) `finite_rows`, `sweep_thresholds`, `stress_readout`, `--readout`, `platt_diagnostic`, `level_shortcut` + D4–D11; (7) A1–A3, A10–A12, W1. Steps 6–7 are computable post hoc from the raw companion if the box closes first.

## Step 5. Lint and tests

```bash
uv run ruff check .
uv run pytest -q tests/test_spectra.py
uv run pytest -q
```
Fix until the new file is green and the full run shows no failure that was not in the step-0 baseline.

## Step 6. CPU pilot (while MPS is held)

Direction-check manifest for the untouched script (150 per class, inner rows only, seed 0):
```bash
uv run python -c "
import pandas as pd
f = pd.read_csv('splits/nsa_folds.csv'); f = f[f.fold != 'holdout']
s = pd.concat([f[f.label == l].sample(150, random_state=0) for l in ('spoof', 'bonafide')])
s[['path', 'label']].to_csv('outputs/manifests/nsa_inner_direction300.csv', index=False)"
```
Pilot, three modes in one pass, inner and test rows only, at low priority, logged:
```bash
nice -n 10 uv run python scripts/score_spectra.py --limit 400 --test-limit 200 --pad-modes repeat,zero,whole --device cpu --threads 4 2>&1 | tee outputs/spectra/pilot.log
```
About 600 decodes and 1,560 forwards; 15–25 min under the current load (0.24 s per clip unloaded for XLS-R on this Mac, 3–5× under load). Apply the rule: inner minDCF, then inner AUC, then `zero`; a test share above 0.6 flags. Record the pilot table, the granularity note and the pick in the run spec.

## Step 7. GPU work, after the main chat releases MPS

Check first: `pgrep -fl "extract_embeddings|train_probe"` must be empty and the main chat must have said so.
```bash
uv run python scripts/spectra_direction.py --manifest outputs/manifests/nsa_inner_direction300.csv --name nsa_inner300
nice -n 10 uv run python scripts/score_spectra.py --manifest outputs/manifests/itw_stress.csv --name itw_stress --pad-mode <pick> --device mps
```
Provisional P_FA/P_miss at the pilot's inner thresholds, without rerunning inference (amendment 17):
```bash
uv run python scripts/score_spectra.py --readout inner=outputs/spectra/pilot/<dir>/scores_<pick>.csv stress=itw=outputs/spectra/itw_stress/scores.csv --out outputs/spectra/itw_stress/readout_provisional.json
```
Post the In-the-Wild numbers (minDCF, EER, per generator, the provisional P_FA/P_miss) to Nathan and the main chat. Then, by default:
```bash
nohup nice -n 10 uv run python scripts/score_spectra.py --pad-mode <pick> --device mps --stress itw=outputs/spectra/itw_stress/scores.csv > outputs/spectra/score_spectra.log 2>&1 &
```
About 75 min at 0.2 s per clip. Watch the log's ETA line; `--resume` restarts it if killed. The `--no-band-match` diagnostic run on the pick (`--limit 200`) goes last, on MPS, only if time allows.

## Step 8. Report and docs

- `docs/reports/2026-09-26_m3-spectra.md`: what M3 is, the input path and crop parity, the pilot table and the pick (noting `repeat` is the in-distribution choice and `zero`/`whole` are out of distribution for the model), the direction baseline, holdout numbers against the comparable M1 (amendment 5) per generator and per source, the test-set share above half against the 0.30 prior (both thresholds), the In-the-Wild readout flagged "possibly optimistic, training data undisclosed", the level-shortcut measurement, timing per device, what Spectra is blind to, and that fusion reads `logit` directly (commit 944380c). Write `.csv`, never the bare word; write "π = 0.5" for the secondary prior, never "π_synth = 0.5".
- `docs/STATUS.md`: the oversight chat already replaced line 40 with an "In flight" M3 row (commit 9288d73); replace it with the shipped row (numbers, file name) and update the test-count row (line 44) to the post-change count. Keep the header timestamp (main chat's snapshot).
- `CLAUDE.md`: line 124 area, add a Claude Code sub-bullet for M3 (Sat Sep 26): the Spectra-AASIST score stream, its tests and report; line 134: Spectra-AASIST "how used" → "M3 score stream (`outputs/detector_scores/spectra_aasist.csv`), scored through the shared band-matched input path; license still unclear (repo header Apache-2.0, card text MIT)".
- Run spec: Results (baseline test count, pilot table, pick, direction, In-the-Wild, holdout, test share, timing, deviations, anything cut) and the dated amendments above.
- `docs/plan.md`: no change expected; confirm.
- Completion handoff (amendment 19): message the main chat with the published file name, the run stamp and config hash, the holdout numbers, the test share, and the "fusion reads `logit`" note; record the time in the run spec's Results. The report lists codec laundering, telephony and simulated replay as untested here (amendment 20).

## Step 9. Commit

```bash
git add src/hearsay/spectra.py src/hearsay/metrics.py scripts/score_spectra.py tests/test_spectra.py \
  docs/specs/2026-09-26_m3-spectra-aasist.md docs/reports/2026-09-26_m3-spectra-aasist-plan.md \
  docs/reports/2026-09-26_m3-spectra-aasist-plan-review.md docs/reports/2026-09-26_m3-spectra.md \
  docs/STATUS.md CLAUDE.md
# plus docs/handoffs/2026-09-26_m3-spectra-aasist-handoff.md if Nathan confirms
git commit -m "feat(M3): Spectra-AASIST score stream through the shared input path; pad-mode pilot, direction gate, raw-logit export"
```
No push. Then the post-commit audits (Claude critique loop, Codex audit) per the skill.

## Risks and mitigations

| Risk | Mitigation in this plan |
|---|---|
| Hermetic tests write into the real `outputs/` or `models/` during the post-commit audit | `--repo-root`; every test uses `tmp_path`; C10 snapshots the real trees |
| Decode pool holds thousands of clips in RAM | Chunked map (amendment 1) |
| `_FIR` lazy-init race across decode threads | Warm once on the main thread; A2 asserts the float32 FIR |
| pos_T 400-frame limit | `ValueError` guard in `forward_windows`; A7 pins 128,000 passes and 128,400 raises |
| Zero padding unmasked, out of distribution | Pilot decides; `n_pad_samples_<mode>` recorded; report states `repeat` is in-distribution |
| No level normalization in Spectra | Level-shortcut measurement in meta (D10); no normalization added |
| MPS not bit-comparable with CPU | Device and versions in the config hash; byte-identity tests use the fake model on CPU; W1 uses `allclose` |
| Recomputed `synth_logit` on resume breaks byte identity | Stored value reused verbatim; the fake's second logit makes C14 sensitive to it |
| Crop desync under sampling | `crop_plan` draws all rows first, keyed by path; A4 pins draws |
| Pilot too coarse for the 0.02 tie band | 322 inner rows, holdout rows skipped, AUC tie-breaker, granularity note in the table |
| Sigmoid overflow warnings, saturated scores | `errstate(over="ignore")`; fusion reads `logit` (944380c) |
| Raw companion picked up by a glob | Lives under `outputs/spectra/` |
| One-class, all-NaN or index-0 threshold | `direction_check` and `sweep_thresholds` report `None` plus a flag; no `Infinity` in JSON |
| External drive unmount mid-run | 25 consecutive decode errors abort with a hint |
| Stale partial merged on resume, or a finished run re-exported | Config hash sidecar; pandas `fillna("")`; partial deleted on completion; `--resume` without a partial exits 2 |
| Wrong M1 comparator | Rule: newest with `folds == splits/nsa_folds.csv`, single train set, `val_min_dcf` present |
| Others' uncommitted files swept into the commit | Explicit `git add` paths |
| LightGBM next to torch | C15, plus the existing `test_trees` scan |
| GPU released later than 05:15 | CPU pilot first; the export waits; hard stop 08:00 in the spec |
| Box closes before the test list is done | Build order in step 4; diagnostics post hoc from the raw companion; cuts recorded |
| Non-finite forward output passes as ok | Output validated in `score_clip`; B8 |
| NaN rows crash a readout | `finite_rows` in every readout with counts; D11 |
| Concentrated failures hide behind a global 5% | Per-split gate; per-class and per-generator coverage in meta; C8 |
| Half-written or stale shared export | Run directory first, publish last by `os.replace`, summary sidecar with the stamp; C14 |
| Training manifest reordered between runs | Its sha is in the resume hash; C16 |
| Provisional stress readout impossible before the export | `--readout` mode on saved scores; D9 |

## Verification

- AC1–AC8 from the run spec, checked in order at the end of steps 7 and 8, with AC1 read as "no failure absent from the step-0 baseline".
- `head -1 outputs/detector_scores/spectra_aasist.csv` is exactly `path,fold,split,score,logit`; `wc -l` is 21,672; a pandas check of split counts 16,142 / 3,858 / 1,671 and unique paths.
- `git status --short` after the commit shows only others' files.
