# Plan: K, the offline CPU Docker image and `scripts/run_pipeline.py` (Sat Sep 26, 2026)

Run spec: `docs/specs/2026-09-26_k-docker-image.md` (decisions D1–D14, tests B/R/L/D/M/F/G/C/O/P, acceptance T1–T9). Branch `main`, worktree `~/Projects/hearsay`, shared with five chats: stage only the files listed in "Files" by path. Time box: build from approval (~06:50) to 09:30, hard stop 10:00, gate 14:00. The emulated amd64 build runs in the background from step 8 onward.

Preconditions verified at 06:10: Colima running (vz, Rosetta, 6 CPU, 6 GB, VM disk on the external drive with 30 GB free), `docker buildx` v0.37.1, an amd64 `python:3.12-slim-bookworm` container reports `x86_64`; `codex` CLI 0.154.0; 301 tests passing; ffmpeg 7.1.1 on the Mac.

## Execution note (06:35): the runner moved to the main chat

Steps 2 and 3 below (the runner and its tests) were superseded before execution: the main chat shipped `scripts/run_pipeline.py` + `src/hearsay/pipeline.py`, so the image wraps their runner (run spec, "Scope change at 06:35"). Steps 1, 4, 5 (image tests only), 6 (`pcm-hash` only; score parity is their `--compare-tsv`), 7, 8, 9 and 10 were executed as written with these adjustments: the Dockerfile copies weights before code (the critique's finding 2); every `-v` source is absolute and under `$HOME`; in-image checks use `--entrypoint python`; the build started at 06:37 with the weights layers cached for later rebuilds. The Codex review's findings and how each was handled are in the run spec. Intermediate stops: 08:00 commit and Claude critique, 09:00 Codex audit, 09:30 T8 addendum, 10:00 hard stop and report.

## Anchors (current code, verified 06:20)

| Symbol | Where | Note |
|---|---|---|
| `score_files(paths, score_fn, fallback) -> ScoredFiles` | `src/hearsay/submission.py:44` | flags `decode_error`, `score_error:<Exc>`, `nonfinite_score`; one row per input |
| `write_submission(ids, scores, out_path, *, id_col, score_col, score_max, sep)` | `submission.py:73` | `FileExistsError` at :85, validates, reads back at :105–109 |
| `list_test_files(test_dir, manifest, id_col)` | `submission.py:139` | comma reader at :142 (unusable for the tab template); no-manifest branch at :144 is the sorted rglob |
| `preflight(score_fn, fallback, tmp)` | `submission.py:148` | silence + chord, scored twice, asserts equal and finite |
| `AUDIO_EXT` | `submission.py:136` | |
| `load_backbone(name, device)` | `src/hearsay/embed.py:30` | `AutoModel.from_pretrained(REPO/"weights"/name).eval().to(device)` |
| `prepare_segment(x, crop_s, seed, max_s, band_match)` | `embed.py:80` | band → trim → crop → cap |
| `embed_segment(model, x)` | `embed.py:105` | `model(t, output_hidden_states=True).hidden_states`, time-mean per layer |
| `Probe` (`backbone, layer, clf, platt_a, platt_b, win_s, max_windows, segment`) / `llr` / `load` | `src/hearsay/probe.py:67, 78, 98` | `segment` is wrong for 0521 (spec) |
| `load_spectra(device)`, `prepare_input`, `score_clip(model, x, pad_modes, max_windows)`, `synth_logit` | `src/hearsay/spectra.py:47, 73, 139, 135` | `score_clip` returns `{n_samples, n_windows, <mode>: (2,), n_pad_samples_<mode>}` |
| `score_spectra.run(argv, loader=load_spectra)` | `scripts/score_spectra.py:458` | the injectable-loader pattern the runner copies; test rows: crop None, pad zero, no peak-norm, 16 windows (:100–105, :198, :260–264, :293) |
| `ClipContext`, `safe_run`, `DetectorResult` | `src/hearsay/detectors/base.py:43, 163, 109` | |
| `FeatureModelDetector(model_dir)`, `.run` clip 1e-6 and `hc_logit` | `src/hearsay/detectors/_learned.py:77, 102, 105` | |
| `HandcraftedDetector`, `NAME="handcrafted"` | `src/hearsay/detectors/handcrafted.py:101, 18` | `latest_model_dir` prefers `models/hc_selected/model.joblib` (`_learned.py:34`) |
| `speech_gate.DETECTOR`, `analyze`, `apply_default_answer(fused, is_speech, default)`, `DEFAULT_ANSWER=0.02`, `TIE_BREAK=1e-4` | `src/hearsay/detectors/speech_gate.py:126, 53, 115, 48, 49` | `features["is_speech"]` is 1.0/0.0 |
| fusion arithmetic | `scripts/fuse.py:93–100` (mu, sd = inner_oof mean, std+1e-9), `:110` zmean, `:112–116` stack (`decision_function` on Z), `:137–138` Platt + prior shift | the D8 bundle freezes these numbers |
| M1 export logit = `probe.llr` | `scripts/export_probe_scores.py:58, 65` | |
| hc export logit clip 1e-6 | `scripts/train_handcrafted.py:77–80` | |
| `hash_one(p) -> (p, file_sha, pcm_sha, n)` | `scripts/hash_audio.py:31` | for `parity.py pcm-hash` |
| `sigmoid`, `PI_SYNTH` | `src/hearsay/metrics.py:74, 31` | |
| `FakeSpectra`, `_wav`, `_clip`, `_tone`, `_load_script` | `tests/test_spectra.py:75, 69, 59, 64, 105` | reuse by import |
| `_make_bundle(out, kind)` | `tests/test_handcrafted_detector.py:55` | toy lgbm bundle from `tests/fixtures/lgbm_toy.json` |
| `ffmpeg_sine(out, *, sr, ch, dur, args)` | `tests/test_audio_and_submission.py:18` | |
| `bash -n` pattern | `tests/test_cloud_scripts.py:21–26` | |
| no-lightgbm pattern | `tests/test_spectra.py:486` | |
| Spectra `model.py` imports | `weights/Spectra-AASIST/model.py:1–5` | torch, transformers, huggingface_hub only; hub id at :755 patched by `spectra.py:56–65` |
| lock: torch CUDA markers | `uv.lock:2086–2103` | `cuda-toolkit`, `nvidia-*`, `triton` under `sys_platform == 'linux'` |
| transformers hidden-state capture | `transformers/models/wav2vec2/modeling_wav2vec2.py:907, 1265` | `_can_record_outputs` records each encoder layer's output; `tie_last_hidden_states=False`, so `hidden_states[L]` is the raw output of layer L−1 and keeping `L+1` modules preserves it (verified on a tiny stable-LN model: keep 3 and keep 4 both identical for L=3) |

## Step 0: pre-flight (5 min)

1. `git branch --show-current` → `main`; `pwd` → `~/Projects/hearsay`; `git status --short` (other chats' edits stay untouched).
2. `df -h "/Volumes/Crucial P3 NVME Gen 3 2TB" /` → at least 15 GB free on the VM drive, else stop and report.
3. `docker context show` → `colima`; `docker run --rm --platform linux/amd64 python:3.12-slim-bookworm python -c "import platform; print(platform.machine())"` → `x86_64`.
4. Read the toolchain probe result (below) once, and paste the exact RUN block into the Dockerfile in step 5.

## Step 1: `docker/assets.py` (10 min)

Standalone module, no hearsay imports (it runs inside the build before the venv exists).
- `freeze(root: Path, dirs: list[str], out: Path)`: for each relative dir, every regular file, sorted: `{"path": rel, "sha256": ..., "size": ...}`; writes JSON `{"created": iso, "files": [...]}`. Backbone files (`pytorch_model.bin`, `model.safetensors`) are included; hashing 2.5 GB takes about 10 s in the build.
- `verify(root: Path, manifest: Path, *, hash_over_mb: float | None = 100) -> list[str]`: returns a list of problems (missing, size mismatch, sha mismatch, extra file in a model dir); files larger than `hash_over_mb` are checked by size only at run time unless `--full` (so startup stays under a second); an empty list means OK.
- CLI: `python docker/assets.py freeze --root /app --dirs models/m1_shipped models/hc_selected weights/wav2vec2-xls-r-300m weights/Spectra-AASIST weights/spkrec-ecapa-voxceleb --out /app/models/assets.json` and `verify --root /app --manifest /app/models/assets.json [--full]`; non-zero exit and one line per problem.
Tests (step 6, `test_docker_image.py` B10): tmp tree with small files; freeze → verify OK; tamper a byte, delete a file, add a file → each reported by name.

## Step 2: `scripts/run_pipeline.py` (60 min)

Module layout, top to bottom:

1. Docstring: purpose, CLI, env vars, outputs, the never-flip rule.
2. Imports: stdlib; `numpy`; `hearsay.SR`; `hearsay.audio.DecodeError, load_audio`; `hearsay.metrics.PI_SYNTH, sigmoid`; `hearsay.submission.AUDIO_EXT, list_test_files, preflight, score_files, write_submission`. **No torch import at module top**: torch is imported inside the loaders so listing, cache and fusion tests run without it and `--help` is instant. No `lightgbm` anywhere (B9).
3. Constants: `APP = Path(__file__).resolve().parents[1]`; `DEFAULT_PROBE = "models/m1_shipped"`; `DEFAULT_HC = "models/hc_selected"`; `DETECTOR_ORDER = ("m1b", "spectra", "handcrafted")`; `SPECTRA_CFG = {"pad_mode": "zero", "peak_norm": False, "band_match": True, "max_windows": 16}`; `FALLBACK_LOGIT = float("nan")`; `PLACEHOLDER_TEAM = "teamName"`.
4. `truncate_backbone(model, layer: int)`: `n = len(model.encoder.layers)`; `layer > n` → `ValueError`; `keep = min(layer + 1, n)`; `model.encoder.layers = model.encoder.layers[:keep]`; `model.config.num_hidden_layers = keep`; return `keep`. (B11, B12, M6.)
5. Listing: `@dataclass Item(id: str, path: Path | None, sha256: str | None)`.
   - `read_template(path) -> list[str]`: `csv.DictReader(f, delimiter="\t")`; header must contain `filename` (else `SystemExit` naming the header); duplicates → `SystemExit` (L2, L7).
   - `index_audio(data_dir) -> dict[str, list[Path]]`: basename → paths, from `sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT)` (the same filter as `list_test_files`).
   - `list_items(data_dir, template, limit) -> list[Item]`: with a template, each id resolves via the index (0 hits → `path=None`, flag later `missing_file`; 2+ hits → `SystemExit` ambiguous); without, `list_test_files(data_dir, None, "filename")` ids (relative paths); `limit` slices; empty → `SystemExit` (L1–L9). `sha256` of each existing file computed here (C5).
   - `manifest_sha(items)`: sha256 over `"\t".join((id, sha or "missing"))` lines in order.
6. Cache: `class Cache` with `path`, `header: dict`; `open()` reads the header line, discards the file if `manifest_sha`/`config_sha` differ (prints a notice), parses the rest with a `try/except json.JSONDecodeError` that drops only a corrupt final line (C4); `rows: dict[tuple[str, str], dict]` keyed by `(id, sha256)`; `append(row)` writes one JSON line and flushes (C1–C5).
7. Scorers (each a small class with `name`, `config() -> dict`, `load()`, `unload()`, `logit(x: np.ndarray) -> float`):
   - `M1Scorer(app_root, probe_dir, mode, truncate, device, loaders)`: `load()` → `Probe.load(probe_dir)` (M4 checks: dir exists, backbone dir exists under `app_root/weights`, `0 <= layer <= 24`), `load_backbone(probe.backbone, device)` via the injectable `loaders["backbone"]`, `truncate_backbone` when `truncate` (records `kept_layers`), `torch.manual_seed(0)`; mode resolution: `segment` default, `windows`, or `auto` = the probe flag; if the flag disagrees with the resolved mode, print `WARNING: probe.segment=<flag> disagrees with --m1-mode <mode>; using <mode>` (M3). `logit(x)`: segment → `embed_segment(model, prepare_segment(x))`; windows → `embed_clip(model, x, probe.win_s, probe.max_windows)`; return `float(probe.llr(e)[0])` (M2, M5). `config()` → `{probe_sha, layer, kept_layers, mode, backbone}`. `unload()` → drop references, `gc.collect()`.
   - `SpectraScorer(app_root, device, loaders)`: `load()` → `loaders["spectra"](device)` (default `hearsay.spectra.load_spectra`); `logit(x)` → `x = prepare_input(x)` (band match on, no crop); `out = score_clip(model, x, (SPECTRA_CFG["pad_mode"],), SPECTRA_CFG["max_windows"])`; return `synth_logit(out[pad_mode])` (F5). `config()` → `SPECTRA_CFG | {"weights_sha": sha of model.safetensors from assets.json if present}`.
   - `HandcraftedScorer(app_root, model_dir, loaders)`: `load()` → `HandcraftedDetector(model_dir=model_dir)` (F6); `logit(x)` → `res = safe_run(det, ClipContext.from_array(x))`; status not `ok` → raise `RuntimeError(res.error)` (so `score_files` flags it `score_error:RuntimeError`); return `res.features["hc_logit"]`. Note: `ClipContext.from_array` skips a second decode; the array is the one `score_files` already decoded.
   - `SpeechGate`: not a scorer; `gate(x) -> (is_speech: bool, reasons: dict)` via `safe_run(speech_gate.DETECTOR, ClipContext.from_array(x))`; error → gated, reason `gate_error`.
8. Scoring loop `score_detector(scorer, items, cache, fallback_logit) -> dict[id, row]`: `preflight(scorer.logit, float("nan"), tmpdir)` first (D2; the runner records the two printed values by calling `score_files` on the same two files once more and storing them in `run_info["preflight"][scorer.name]`; the preflight itself already asserts reproducibility). Then for each item not in the cache: `path is None` → row `{logit: nan, flag: "missing_file"}`; else `res = score_files([path], scorer.logit, fallback=float("nan"))` → `logit = res.scores[0]`, `flag = res.flags[0]`; `row = {id, sha256, logit, score: sigmoid(logit) if flag == "" else nan, flag, seconds}`; `cache.append(row)`. Per-detector wall time recorded.
9. Gate pass `gate_items(items) -> dict[id, (is_speech, reasons)]`: decode once per file (`load_audio`), `DecodeError` → gated with reason `decode_error` (G3). Cached like a detector (`cache/speech_gate.jsonl`).
10. Fusion `load_bundle(path) -> dict` (validates keys, detectors ⊆ producible set else `SystemExit`, F4) and `fuse_row(bundle | None, logits: dict[str, float], shift) -> tuple[float, str]`:
    - no bundle: `llr = logits["m1b"]`; NaN → `(sigmoid(shift), "fusion_missing:m1b")` (the detector's own flag is kept beside it); else `(sigmoid(llr + shift), "")` (F2, D5).
    - bundle: `z[d] = (logits[d] - mu[d]) / sd[d]`; any NaN → fallback + `fusion_missing:<d>` (F3); rule `zmean` → `mean(z)`, `{"type": "linear"}` → `Σ w[d]·z[d] + intercept`, `identity` → the single detector's logit; `p = sigmoid(a·fused + b + shift)` with `platt = [a, b]` (F1).
11. Policy: `apply_default_answer(p, is_speech)` from `hearsay.detectors.speech_gate` when `--policy speech_gate` (G1–G4).
12. Output `write_outputs(out_dir, team, ids, scores, rows, run_info, limit)`: name `f"{team}_predictions{'-limit'+str(limit) if limit else ''}.tsv"`; if it exists, insert `-<YYYYmmdd-HHMMSS>` before `.tsv` (O2); scores outside [0, 1] are replaced by the fallback with flag `out_of_range` before `write_submission` (O6); `write_submission(ids, scores, path)`; sidecar CSV (`id, score, flag, is_speech, gate_reasons, <det>_logit...`) and `<stem>.run.json` (D10 fields, R4). Print the summary: file written, rows, flags by kind, share above 0.5, preflight values, wall time. Return 0.
13. `parse_args(argv)` and `run(argv=None, *, loaders=None) -> int` (the same shape as `score_spectra.run`): environment checks first (`--require-offline` → both env vars must be `"1"`, R1; `shutil.which("ffmpeg")`, R2; threads: `OMP_NUM_THREADS` if set, else `len(os.sched_getaffinity(0))` when available, else `os.cpu_count()`; `torch.set_num_threads` right after the torch import inside the loaders, R3), then listing, `assets.verify` when `<app>/models/assets.json` exists (R5), then the detector passes in `DETECTOR_ORDER` restricted to `--detectors`, the gate pass, fusion, policy, output. `main()` → `sys.exit(run())`.
14. Exit codes: `SystemExit(2)` for usage/environment/listing failures before any TSV; `0` after a TSV (O5).

## Step 3: `tests/test_docker_runner.py` (45 min, written alongside step 2)

Load the runner with the `_load_script` pattern (`tests/test_spectra.py:105`). Fixtures:
- `fake_backbone()`: `torch.nn.Module` with `encoder = types.SimpleNamespace(layers=nn.ModuleList([nn.Identity() for _ in range(24)]))`, a dummy `Parameter`, and `forward(x, output_hidden_states=True)` returning an object whose `.hidden_states` is a tuple of `len(layers)+1` tensors `(1, T, 4)` computed deterministically from `x` (mean, std, min, max broadcast), so `embed_segment` and `truncate_backbone` run for real.
- `fake_probe(layer=7)`: a real `Probe` with `clf` = an object whose `decision_function(X)` returns `X[:, 0] * 3.0` and `platt_a=1.0, platt_b=0.0`, saved with `joblib` into `tmp/models/m1_shipped/probe.joblib` (+ `meta.json`).
- `fake_spectra`: `FakeSpectra` from `tests/test_spectra.py`.
- `hc_bundle`: `_make_bundle(tmp, "lgbm")` renamed to `models/hc_selected`.
- `data_dir(n)`: WAVs via `_wav` (noise and tone clips at 16 kHz), plus one MP3 via `ffmpeg_sine`, a README.txt, and an empty `broken.wav`.
- `template(ids, order)`: writes a tab-separated file.
- `run(args, loaders)`: calls `mod.run([...], loaders={"backbone": ..., "spectra": ..., "hc_dir": ...})` with `--app-root tmp --device cpu --policy none` unless a test says otherwise; captures stdout.

Tests, by ID (one function each unless parametrized): L1 (shuffled template order), L2 (tab parse; missing header fails), L3 (no template → sorted relative ids), L4 (nested dir; ambiguous basename fails), L5 (`missing_file` row, exit 0, warning), L6 (empty → exit 2, no TSV), L7 (duplicate template ids fail before any scorer call: the fake loader records calls), L8 (`--limit 2` name and order), L9 (README.txt ignored; template naming it → `decode_error`); D1 (broken.wav → `decode_error`; fake scorer raising on the 2nd call → `score_error:RuntimeError`; NaN → `nonfinite_score`), D2 (preflight precedes real files; a fake whose 2nd preflight call differs → no TSV), D3 (preflight values in run.json; WARNING when > 0.8 via a fake that scores silence high), D4 (`/out` contents), D5 (fallback equals 0.3); M1 (exact posterior values and monotonicity), M2 (spies on `hearsay.embed.prepare_segment`/`embed_segment`/`embed_clip`), M3 (probe flag False + default → WARNING and segment path; `--m1-mode auto` → windows path), M4 (missing probe dir; layer 99), M5 (cache logit == fake llr), M6 (kept layers 8 by default; 24 with `--no-truncate`), M7 (loader called with `cpu`), M8 (two runs, identical TSVs); F1 (`fuse_row` unit tests for zmean and linear against hand-computed numbers, including the Platt and prior shift), F2 (no bundle equals the M1 formula), F3 (`fusion_missing:spectra` on a failed file), F4 (bundle naming `m5` fails before scoring), F5 (spectra logit equals `synth_logit(score_clip(...)["zero"])` computed directly with the same fake), F6 (hc logit equals `log(p/(1−p))` from the detector's own result), F7 (loader/unloader order and at most one live model); G1 (silence file gated → `0.02 + 1e-4·fused`), G2 (a speech-like clip passes `analyze` and keeps its score), G3 (broken.wav gated), G4 (`--policy none` leaves fused scores); C1 (cache files and header), C2 (pre-seeded partial cache → only the rest scored; byte-identical TSV), C3 (manifest or config change discards the cache; same values reuse it), C4 (corrupt trailing line skipped), C5 (changed bytes → rescored); O1 (name and format), O2 (second run gets a stamped name; first byte-identical), O3 (order read back), O4 (sidecar columns and run.json keys), O5 (exit codes), O6 (a fake returning 1.5 → `out_of_range` row, TSV still written), O7 (`partial: true`), O8 (timings present); R1 (`--require-offline` without env fails; with env proceeds), R2 (no ffmpeg on PATH fails first), R3 (thread env), R4 (run.json fields), R5 (tampered probe refused, intact accepted).

## Step 4: `Dockerfile`, `.dockerignore`, `docker/entrypoint.sh`, `docker/build.sh` (25 min)

`.dockerignore` (no wholesale `weights` exclusion: a `!` re-include under an excluded parent is not something to bet a 14:00 gate on; the wavlm dirs are excluded by name and `models/` is excluded entirely because the shipped bundles are copied from the staging dir):
```
.git
.venv
.cache
.pytest_cache
.ruff_cache
**/__pycache__
data
outputs
submissions
models
weights/wavlm-*
weights/*/.cache
docs
tests
*.wav
*.mp3
*.flac
*.ogg
*.m4a
.env
.env.example
.claude
```
The hc bundle and the probe are copied from a staging directory `docker/build/` that `build.sh` populates (`cp -RL <resolved hc> docker/build/hc_selected`, `cp -RL <probe dir> docker/build/m1_shipped`), so `.dockerignore` re-includes only fixed paths and `docker/build/` (gitignored via `docker/.gitignore`). This keeps the Dockerfile free of stamps.

`Dockerfile` (single stage; verified sequence from the toolchain probe, 06:30):
```
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project \
      --no-emit-package torch --no-emit-package torchaudio --no-emit-package lightgbm \
    | grep -vE '^(nvidia-|cuda-|triton)' > /app/requirements-cpu.txt \
 && uv venv /opt/venv --python 3.12 \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /app/requirements-cpu.txt \
 && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchaudio==2.11.0 \
 && uv pip check --python /opt/venv/bin/python \
 && rm -rf /root/.cache/uv
COPY src ./src
COPY scripts ./scripts
COPY docker ./docker
RUN uv pip install --python /opt/venv/bin/python --no-deps -e /app && rm -rf /root/.cache/uv
COPY weights/wav2vec2-xls-r-300m ./weights/wav2vec2-xls-r-300m
COPY weights/Spectra-AASIST ./weights/Spectra-AASIST
COPY weights/spkrec-ecapa-voxceleb ./weights/spkrec-ecapa-voxceleb
COPY docker/build/m1_shipped ./models/m1_shipped
COPY docker/build/hc_selected ./models/hc_selected
ARG BUILD_INFO=""
RUN printf '%s\n' "$BUILD_INFO" > /app/BUILD_INFO \
 && python docker/assets.py freeze --root /app --dirs models/m1_shipped models/hc_selected weights/wav2vec2-xls-r-300m weights/Spectra-AASIST weights/spkrec-ecapa-voxceleb --out /app/models/assets.json
VOLUME ["/data", "/out"]
ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
```
Why each non-obvious line: `--no-deps` on the requirements install is load-bearing (without it `speechbrain` resolves `torch` from PyPI as `2.14.0+cu130` with the whole CUDA tree, 5.8 GB, and the later CPU step becomes a no-op because `==2.14.0` is already satisfied); the export's `--no-emit-package torch` drops only the torch line, so the `grep` removes its 19 CUDA-only lines while keeping torch's pinned pure-Python deps (jinja2, sympy, networkx, setuptools, mpmath, markupsafe), which `--prune` would have unpinned; `README.md` is required by the editable install (`readme = "README.md"` in pyproject); `uv pip check` proves the closure is consistent; `rm -rf /root/.cache/uv` in the same layer saves 1.4 GB; the base is pinned by digest. Measured on this Mac: apt 56 s, requirements 9 s, torch 13 s (187 MB wheel), editable 1 s, venv 1.4 GB, ffmpeg 5.1.9 with mp3/flac/aac/opus/vorbis/pcm decoders, `torch.backends.cpu.get_cpu_capability()` = AVX2 under Rosetta. The `NNPACK ... Unsupported hardware` warning under Rosetta is harmless.

The weight COPYs come last so code changes do not invalidate the 2.5 GB layers. The torch versions are written literally in the Dockerfile and test B3 parses `uv.lock` to assert they match.

`docker/entrypoint.sh`: `set -euo pipefail`; if `OMP_NUM_THREADS` is unset, derive it: `n=$(nproc)`; read `/sys/fs/cgroup/cpu.max` and, when its first field is not `max`, `n=min(n, ceil(quota/period))`; `export OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n`; print `threads=$n`. Then `exec python /app/scripts/run_pipeline.py --data /data --out /out --require-offline --app-root /app ${HEARSAY_TEMPLATE:+--template "$HEARSAY_TEMPLATE"} --team "${HEARSAY_TEAM:-teamName}" --detectors "${HEARSAY_DETECTORS:-m1b}" --policy "${HEARSAY_POLICY:-speech_gate}" ${HEARSAY_FUSION:+--fusion "$HEARSAY_FUSION"} "$@"`. A template mounted at `/data/*.tsv` is also auto-detected by the runner when `--template` is absent and exactly one `.tsv` exists directly under `/data` (so NSA can drop the key file beside the audio); two or more → ignored with a notice. An empty `/data` exits non-zero (L6), which also catches a bad bind mount.

`docker/build.sh`: `set -euo pipefail`; `PROBE_DIR=${PROBE_DIR:-models/m1_wav2vec2-xls-r-300m_L7_20260926-0521}`; `HC_DIR=$(readlink -f models/hc_selected)` (must exist and contain `model.joblib`, else exit 1); `rm -rf docker/build && mkdir -p docker/build && cp -RL "$PROBE_DIR" docker/build/m1_shipped && cp -RL "$HC_DIR" docker/build/hc_selected`; `STAMP=$(date +%Y%m%d-%H%M)`; `SHA=$(git rev-parse --short HEAD)`; `DIRTY=$(git status --porcelain | grep -q . && echo dirty || echo clean)`; `docker buildx build --platform linux/amd64 --load --build-arg BUILD_INFO="sha=$SHA $DIRTY stamp=$STAMP probe=$(basename "$PROBE_DIR") hc=$(basename "$HC_DIR")" -t hearsay:$STAMP -t hearsay:latest .`; prints `docker image inspect hearsay:latest --format '{{.Size}} {{.Architecture}}'`. `docker/.gitignore` holds `build/`.

## Step 5: `tests/test_docker_image.py` (25 min)

B1–B6 static asserts on the Dockerfile text (base tag, apt ffmpeg + cleanup in one RUN, the three ENV vars, `--frozen`, `--no-dev`, the three `--no-emit-package`, the nvidia/cuda/triton filter, the CPU index URL, torch/torchaudio versions equal to `uv.lock`'s, no COPY of `data`/`outputs`/`submissions`/`wavlm`/`.venv`/`.git`, COPY destinations equal to `hearsay.embed.REPO`-relative `weights/<name>` and `models/hc_selected`); B4/B5 with a 30-line dockerignore matcher (rules: leading `!` re-includes, `**` any depth, `*` within a segment, a bare name matches the path prefix) checked against a list of must-ignore and must-keep paths; B7 `build.sh` resolution via a `--print-args` dry-run mode on a tmp tree with a symlink and with a dangling one; B8 `bash -n` and grep for `set -euo pipefail`, `--require-offline`, the env names; B9 no lightgbm; B10 assets freeze/verify; B11/B12 truncation on a tiny `Wav2Vec2Config(do_stable_layer_norm=True, num_hidden_layers=6, hidden_size=32, ...)`: identity for L=3 with 4 modules kept, `ValueError` for L=7, `--no-truncate` path leaves 6; B13 `@pytest.mark.slow @pytest.mark.needs_weights`: real XLS-R, one 3 s clip, `embed_segment` layer 7 equal (`np.array_equal`) between the full model and the 8-module copy; skips without weights.

## Step 6: `docker/parity.py`, `docker/smoke.sh` and P1–P4 (25 min)

`parity.py compare A.tsv B.tsv [--n N] [--spearman-min 0.99] [--max-abs 0.05] [--ignore-order]`: reads both with `csv` (tab), aligns on `filename` (mismatch → exit 2 naming the first offender), takes the first N of A's order, prints Spearman (`scipy.stats.spearmanr`), max abs diff, the count over `--max-abs`, and whether B's order equals A's; exit 1 on a threshold failure. `pcm-hash DIR --ids id1 id2 ... [--template T --n N] --out hashes.json`: `hash_one` per file; `pcm-diff a.json b.json` → lists differing ids, exit 1 if any.

`smoke.sh`: builds a 3-file set under `outputs/docker/smoke/` (not `/tmp`: Colima mounts only `$HOME`) (`sine.wav` 16 kHz via ffmpeg, `sine.mp3`, `sine.flac`) plus a reversed template; `docker run --rm --network none -v $tmp/data:/data:ro -v $tmp/out:/out -e HEARSAY_TEMPLATE=/data/template.tsv hearsay:latest`; asserts 4 lines, header, order equals the template, exit 0; runs again into `out2` and `cmp`s the two TSVs; runs a third time with `-e HEARSAY_DETECTORS=m1b,spectra,handcrafted` and checks three cache files exist. Prints the run.json preflight block. Before the runs, three one-line image self-checks: `python -c "import hearsay.embed as e; assert (e.REPO/'weights'/'wav2vec2-xls-r-300m'/'config.json').exists()"` (REPO resolves to /app), `python -c "import numpy as np, librosa; librosa.feature.mfcc(y=np.zeros(16000,'f4'), sr=16000)"` (numba JIT under Rosetta), and `python docker/assets.py verify --root /app --manifest /app/models/assets.json --full`.

Tests P1–P4 in `test_docker_image.py`: synthetic TSV pairs for `compare`; two tiny WAVs for `pcm-hash`; static greps on `smoke.sh` and `build.sh`.

## Step 7: hermetic gates T1–T3 (10 min)

`uv run pytest -q tests/test_docker_image.py tests/test_docker_runner.py`; `uv run ruff check .`; `uv run pytest -q` (all); `uv run pytest -q -m "needs_weights and slow" tests/test_docker_image.py`. Fix until green.

## Step 8: build in the background, Mac parity in the foreground (T4, T5)

`bash docker/build.sh > outputs/docker_build.log 2>&1 &` (background; expected 30–60 min: 2.5 GB context, Rosetta pip installs). Meanwhile T4: `uv run python scripts/run_pipeline.py --data data/nsa/HackGTHearsayTesting --template data/nsa/HearsayScoreKey4TeamX.tsv --out outputs/docker/k-mac --probe models/m1_wav2vec2-xls-r-300m_L7_20260926-0521 --policy none --limit 50 --device cpu` then `uv run python docker/parity.py compare outputs/docker/k-mac/teamName_predictions-limit50.tsv submissions/20260926-0522_M1_m1_wav2vec2-xls-r-300m_L7_20260926-0521.tsv --n 50` and `pcm-hash` of the 50 into `outputs/docker/k-mac/pcm.json`. If Spearman or max abs fail, stop: the fault is in the runner (compare the cached logits against `outputs/detector_scores/m1b_v3.csv` test rows for the same files to localise it), not in Docker.

## Step 9: image checks (T6, T7, T8)

Colima bind-mounts only paths under `$HOME` (verified by the probe: a `/private/tmp` mount appeared empty inside the container), so every `-v` source is under the repo (`outputs/docker/...`, gitignored) or `$HOME`; `smoke.sh` creates its temp set under `outputs/docker/smoke/`.

When the build finishes: `docker run --rm --platform linux/amd64 hearsay:latest python -c "import platform, torch; print(platform.machine(), torch.__version__)"`; `bash docker/smoke.sh`; T7: `docker run --rm --network none -v data/nsa/HackGTHearsayTesting:/data:ro -v $PWD/outputs/docker/k-img:/out -e HEARSAY_TEMPLATE=/data/../HearsayScoreKey4TeamX.tsv ... hearsay:latest --limit 50 --policy none` (mount the template read-only at `/tmpl/key.tsv` instead of relying on `/data`), then `parity.py compare` against the logged TSV and `pcm-diff` against the Mac hashes; T8: the full set in the background with `time`, written to `outputs/docker/k-full`. Under Rosetta the risk agent measured 1.56 s per 8 s clip through 8 layers on the Mac's native CPU and expects 3–8× that under emulation, so the full set is a 1.5–4 h background job; if it has not finished by the hard stop, the run spec records the measured per-file rate from T7 and the extrapolation, marked as such, and the native amd64 timing is deferred to the Sunday rebuild (a Linux box or NSA's own run). The `NNPACK` and `LOAD REPORT` (unexpected `quantizer.*` keys of the pretraining checkpoint) lines in the logs are expected noise.

## Step 10: docs, run spec results, commit, audits (30 min)

README Docker section (build, run, env vars, mounts, outputs, smoke and parity commands, the placeholder team name, the note that the TSV is not auto-logged); conditional edits (STATUS row, CLAUDE.md bullet, architecture 10.7) staged as my hunk only via `git diff -- <file> > /tmp/p.diff` edited to my hunk and `git apply --cached`, or deferred with a dated note; run spec Results filled with T1–T9 numbers; `git add` by path: `Dockerfile .dockerignore docker/ scripts/run_pipeline.py tests/test_docker_image.py tests/test_docker_runner.py README.md docs/specs/2026-09-26_k-docker-image.md docs/reports/2026-09-26_k-docker-image-plan.md docs/reports/2026-09-26_k-docker-image-plan-review.md`; commit `feat(K): offline CPU amd64 Docker image and scripts/run_pipeline.py (audio -> detectors -> fusion -> TSV)`; Claude critique loop; `bash .claude/review-audit.sh docs/specs/2026-09-26_k-docker-image.md`; fix findings; re-audit; report.

## Risks and mitigations (risk agent, 06:40; all VERIFIED unless marked)

| # | Risk | Mitigation in this plan |
|---|---|---|
| 1 | A hashed export leaves orphan `--hash` lines after the grep and forces `--require-hashes` | `--no-hashes` on the export (present); 72 requirement lines, 0 hash lines verified |
| 2 | Every weights/models lookup is `REPO`-relative (`embed.py:21`, `spectra.py:38`, `_learned.py:26`, `speaker_drift.py:32`): a non-editable install would resolve to site-packages | editable install plus `PYTHONPATH=/app/src`; smoke self-check asserts `hearsay.embed.REPO/weights/...` exists |
| 3 | Rosetta throughput: full 1,671 files is hours per heavy model | 50-file parity is the gate; the full set runs in the background and is extrapolated if unfinished (step 9); 8-layer truncation for M1 |
| 4 | Colima bind-mounts only `$HOME`; other sources mount empty without error | all `-v` sources under the repo; the runner exits non-zero on an empty `/data` |
| 5 | `docker run --cpus=N` quota vs `os.cpu_count()` oversubscription | entrypoint derives threads from `cpu.max` and affinity; exports `OMP_NUM_THREADS`, `MKL_NUM_THREADS` |
| 6 | numba/llvmlite JIT (librosa, on the handcrafted and speech-gate paths) unverified under Rosetta; default cache dir may be read-only | `NUMBA_CACHE_DIR=/tmp/numba` in the image; smoke self-check runs `librosa.feature.mfcc`; `NUMBA_CPU_NAME=generic` documented as the fallback |
| 7 | `append_log` targets `/app/submissions/log.csv`, absent in the image | the runner never calls `append_log` (D10) |
| 8 | `COPY` of the symlinked `models/hc_selected` copies a dangling link when the tree is copied; `!` re-includes under an excluded parent re-include only the link | staging dir `docker/build/` populated with `cp -RL`; `models/` excluded from the context; weights included by not excluding them |
| 9 | `compression` detector needs `models/cmp_selected`, not shipped | the runner names its detectors explicitly and never iterates the registry |
| 10 | transformers 5.17 and a local `pytorch_model.bin` offline | loads with `weights_only=True`, `mmap`, `local_files_only` forced by the env; verified against a dead endpoint (5.6 s) |
| 11 | Spectra `model.py` and the hub id | patched by `load_spectra`; `PyTorchModelHubMixin.from_pretrained` takes the local-dir branch; verified against a dead endpoint (13.0 s) |
| 12 | Memory in a 5.8 GiB VM | XLS-R (8 layers, 8 s clip) 1.07 GiB peak, Spectra with a 16-window batch 1.51 GiB; one model resident at a time |
| 13 | Import-time side effects | none; `splits/nsa_test_durations.csv` is not needed at test time |
| 14 | joblib pickles across arm64 → x86_64 | explicit little-endian dtypes, same sklearn/numpy; the `Probe` class must be importable (editable install) |
| 15 | ffmpeg 5.1.9 vs 7.1.1 decode | md5 of the decoded f32 stream identical for an NSA WAV on both; PCM parity (T7) confirms on 50 files |
| 16 | buildx `--load` size limits, context upload | containerd snapshotter, no tar export; 1.26 GB COPY measured at 60 s wall, so 2–3 min for the weights |
| 17 | Rosetta CPU features | `avx2`, `fma`, `f16c` exposed, no avx512; torch reports AVX2 |

## Toolchain probe result (06:30, amd64 container under Rosetta, 6 vCPU)

| Step | Wall | Size |
|---|---|---|
| apt update + ffmpeg | 59 s | +473 MB (libavdevice pulls X11/SDL; acceptable) |
| uv export + grep | 1 s | 271 → 252 lines, 72 package lines, 65 install on linux/py3.12 |
| requirements install (`--no-deps`, cold cache) | 9 s | venv 608 MB |
| CPU torch + torchaudio from the PyTorch index | 13 s | venv 1.3 GB (torch 711 MB) |
| editable install | 1 s | needs README.md |
| leftover uv cache | | 1.4 GB, removed in-layer |

Verified inside the venv: `torch 2.14.0+cpu, torchaudio 2.11.0+cpu, transformers 5.17.0, sklearn 1.9.1, numpy 2.5.3, scipy 1.18.1`; `torch.cuda.is_available()` False; no nvidia/cuda/triton/lightgbm packages; `torchaudio.functional.preemphasis` present; `platform.machine()` x86_64; matmul 2048² median 0.30 s; a 768-wide transformer layer on 49 frames 27 ms. Notes: `tzdata` (the Python package) does not install on Linux, so the runner avoids pandas timezone features (it uses the `csv` and `json` modules and stdlib `datetime`); `torchaudio.load` would need `torchcodec`, which the project never calls (all decoding is FFmpeg via `hearsay.audio`).
