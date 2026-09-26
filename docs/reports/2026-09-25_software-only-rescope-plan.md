# Plan: software-only rescope (deep mode)

Run spec: `docs/specs/2026-09-25_software-only-rescope.md`.
Base: `main` @ `e1a1e88`, worktree `~/Projects/hearsay`.
Inputs: architecture, file-impact and risk exploration agents (Opus) and a critique agent (Fable). **The "Critique amendments" section at the end supersedes any step it conflicts with.**
Time box: about 1 h. The TSV path is not touched (no code under `submission.py`/`audio.py` changes), so the "always a valid TSV" rule is not at risk from this change.

## Step 0. Pre-flight
- `git branch --show-current` must be `main`; `pwd` must be `~/Projects/hearsay`.
- `git status --short`: the only untracked files should be the run spec and this plan.

## Step 1. Detector contract: `src/hearsay/detectors/__init__.py` and `base.py` (new)

Idiom to match:
- `from __future__ import annotations`, and a module docstring citing the brief.
- `@dataclass`.
- `ValueError` for validation.
- `# noqa: BLE001 - <reason>` on the one broad except, following `submission.py:60`.
- Import `REPO` from `hearsay.submission` rather than re-deriving `parents[3]`.

### 1.1 `ClipContext` (`@dataclass`)
- Fields:
  - `path: Path`
  - private `_audio: np.ndarray | None`, `_audio_err: DecodeError | None`, `_probe: dict | None`
  - `cache: dict[str, Any]`
- `.audio`
  - Calls `hearsay.audio.load_audio` once.
  - Marks the array `flags.writeable = False`, so one detector cannot mutate another's input.
  - Caches a `DecodeError` and re-raises it on later accesses; there is no second ffmpeg call.
- `.probe`: calls `probe_audio` once. `probe_audio`'s return shape is unchanged (the loader test and `scripts/inventory.py` depend on it).
- `.memo(key, fn)`: per-clip shared cache, so detectors share expensive intermediates such as XLS-R windows.
- `ClipContext.from_array(x, probe=None, path="<array>")`: for tests; no ffmpeg.
- Threading: documented as "decode before fan-out"; there is no lock.

### 1.2 `DetectorResult` (`@dataclass(frozen=True)`)
- Fields:
  - `name: str`
  - `score: float`
  - `evidence: str`
  - `features: dict[str, float] = field(default_factory=dict)`
  - `status: Literal["ok", "skipped", "error"] = "ok"`
  - `error: str | None = None`
- `__post_init__` raises `ValueError` when:
  - `name` is not a non-empty str.
  - `score` is a `bool`/`np.bool_`, is not a `numbers.Real`, is non-finite, or is outside [0, 1]. The value is coerced to a Python `float` and stored via `object.__setattr__`.
  - `evidence` is not a str, or is empty after `.strip()`.
  - `features` is not a Mapping; a key is not a non-empty str; or a value is a bool, not `numbers.Real`, or non-finite. Ints and numpy scalars are coerced to float. A plain `dict` copy is stored (picklable, for joblib/multiprocessing fan-out).
  - `status` is not one of the three values.
  - The error field doesn't match the status: `ok` requires `error is None`; `error` requires a non-empty str; `skipped` requires `error is None`.
- The docstring states the direction: **score increases with synthetic likelihood**.

### 1.3 `Detector` (`@runtime_checkable Protocol`)
- `name: str`
- `applies(self, ctx) -> bool`
- `run(self, ctx) -> DetectorResult`
- Load heavy models lazily in `run`.

### 1.4 `safe_run(det, ctx) -> DetectorResult`
Never raises `Exception`; `KeyboardInterrupt`/`SystemExit` still propagate.
- `name = getattr(det, "name", type(det).__name__)`.
- `applies()` is evaluated first:
  - False: return `status="skipped"`, score 0.5, evidence `"not applicable: <name>"`. `run` is not called.
  - Raises: status `error`.
- Otherwise call `run`. It is an error if the return is not a `DetectorResult`, or if `result.name != name`.
- The error result has score 0.5, empty features, and `error=f"{type(e).__name__}: {e}"`. Its evidence is built from the type name so it is never empty (`str(ValueError())` is `""`).
- Wall time is recorded in `result.features`? **No.** Features are for fusion. Instead, `safe_run` returns the result unchanged, and timing is the pipeline's job (M4); this keeps the contract minimal.

### 1.5 `Registry` class
- `register(det) -> det`: usable as a decorator on instances. A duplicate name raises `ValueError`.
- `get(name)`: raises `KeyError`.
- `names()`: sorted, giving a stable fusion column order independent of import order.
- `all_detectors()`: in sorted-name order.
- A module-level `REGISTRY = Registry()` plus the function aliases `register`, `get` and `all_detectors`.

### 1.6 `__init__.py`
Re-exports:
- `ClipContext`
- `DetectorResult`
- `Detector`
- `safe_run`
- `Registry`
- `REGISTRY`
- `register`
- `get`
- `all_detectors`

## Step 2. `tests/test_detector_contract.py` (new; about 22 tests)
- A fixture snapshots and restores `REGISTRY` so no test leaks global state. Most tests use a fresh `Registry()`.

**B1: invalid scores**
- `score ∈ {nan, inf, -0.01, 1.01, np.float32("nan")}` → `ValueError`.
- `True` and `np.bool_(True)` → `ValueError`.
- `"0.5"` → `ValueError`.
- `np.float32(0.25)` is accepted and stored as a Python `float`.
- 0.0 and 1.0 are accepted (boundaries).

**B2: direction**
- A toy `EnergyTailDetector` scores the fraction of energy above 4 kHz. A white-noise "synthetic-marked" array scores higher than a 200 Hz tone "real-marked" array.
- Direction comes from the fixture definition, and the docstring states the convention. This pins that the contract carries the direction rule. It is not a claim about real audio.

**B3: errors**
- A detector that raises `RuntimeError("boom")` → `safe_run` gives `status="error"`, score 0.5, `"RuntimeError: boom"` in `error`, and empty features.
- Counterfactual: calling `det.run(ctx)` directly raises.
- `raise ValueError()` with an empty message still gives a valid error result.
- `run` returning `None`, a dict, or a result with the wrong name → error.
- `applies` raising → error.
- `KeyboardInterrupt` propagates.

**B4: evidence**
- `""`, `"   "` and `None` → `ValueError`.

**B5: features**
- Nested dict, str value, bool value, NaN, empty-string key, non-str key → `ValueError`.
- An int value is coerced to float.
- Mutating the caller's source dict afterwards does not change the result.

**B6: registry**
- A duplicate name raises.
- `get` of an unknown name raises `KeyError`.
- `names()` is sorted regardless of registration order.

**B7: determinism**
- The same `ClipContext` scored twice gives equal results.
- `ctx.audio` is read-only: an in-place write raises `ValueError`.
- `ClipContext` decodes once: `load_audio` is monkeypatched to count calls, and a real file's `.audio` is accessed twice → exactly 1 call.
- A failing decode raises `DecodeError` on both accesses, with 1 call.

**B8: skipped**
- `applies()` False → `status="skipped"`, and a `run` that would raise is never called.

**Status/error coherence**
- `status="ok", error="x"` → `ValueError`.
- `status="error", error=None` → `ValueError`.
- `status="bogus"` → `ValueError`.

## Step 3. MP4 fixture: `tests/test_audio_and_submission.py`
- Add `("aac.mp4", 48_000, 2, ("-c:a", "aac", "-f", "mp4"))` after line 35 (the `opus.ogg` case) in the parametrize at lines 25–36.
- `ffmpeg_sine` (lines 18–22) puts `args` before the output path, and the aac encoder and mp4 muxer are present.

## Step 4. Rewrite `docs/plan.md` (full replacement of all 168 lines)

Sections, in order. The A2, A3 and A5 tests key on the exact headings.

1. **Title and one-paragraph version.** A software-only, multi-detector audio authentication system; deliverables are the TSV, Docker and README; hardware line `(cut)`.
2. **`## Scoring and deliverables`**
   - 60% MinDCF (`C_FA = 4`, `C_miss = 1`, `π_synth = 0.3`; normalized 9.33·P_FA + P_miss).
   - 20% diversity; 20% documentation.
   - `teamName_predictions.tsv` (`filename<TAB>cm-score`).
   - Docker image, README, one-time draft review.
   - Final DM by 8 AM Sunday.
   - Metric-reading risk: every candidate logs normalized minDCF under three readings (π = 0.3 standard, π = 0.5, and the slide-literal form) plus EER. A decision that flips between readings is flagged. This is implemented in `hearsay.metrics` by the next M1/M4 change, not in this one.
3. **`## Architecture`**: ingest → orchestrator → detectors (contract in `src/hearsay/detectors/base.py`) → fusion → TSV + explanation report.
   - Fusion reads each detector's logit/LLR-type features and missing indicators, not raw feature dumps. It uses L2 logistic regression, imputes with the train mean (never safe_run's 0.5 placeholder), and reports per-source minDCF.
4. **`## Forensic techniques`**: exactly one bullet per brief category, each with its owner, one-line method and time box:
   - container/metadata
   - spectral
   - prosody
   - ENF
   - compression
   - speaker-embedding consistency
   - deep anti-spoofing
   - splice/discontinuity
   - orchestration bonus

   Rules:
   - Metadata uses embedded ffprobe/ExifTool tags only, never filesystem MAC times or filenames (git, Docker and unzip rewrite them).
   - Forensic detectors may read audio at its native rate via `ctx.memo`; this is the only exception to 16 kHz.
5. **`## Ladder`**: table M0, M1, D-track, M3, M5, M4, ORCH, K, DOC, with owner, time box and gate. Teammates own the D-track. **K (Docker) is owned by a teammate from Saturday morning** (risk H1), with the M1 model as its first payload.
6. **`## Data`**
   - NSA train is primary.
   - **Fold file:** `splits/nsa_folds.csv`, published within 1 h of the data arriving. Every learned detector and fusion must use it, or stacking leaks. Grouping falls back from speaker to generator to near-duplicate cluster.
   - MLAAD English (6,390 pulled) plus a multi-speaker bona fide source (LibriSpeech/VCTK, not yet downloaded). Public-data numbers for engineered features are confounded by corpus.
   - ASVspoof is baseline only; In-the-Wild is stress only.
   - Augmentation for both classes: codec/telephony, noise, RIR/replay, band-limit, random ffmpeg round-trips, which also neutralize container shortcuts.
   - Leakage hashing.
   - ECAPA weights to download (speaker drift and Docker).
7. **`## Timeline`**: gates from the risk report, all relative to data arrival (T0) with absolute backstops:
   - T0+1 h: inventory (including container-field AUCs; above 0.85 counts as a shortcut), fold file, constant TSV.
   - T0+3 h: M1 NSA TSV.
   - Sat 09:00: H100 run 1, pending approval. If NSA data hasn't arrived by 09:00, cut M5 to one run and shift every gate by the delay.
   - Sat 10:00: Docker skeleton (teammate).
   - Sat 14:00–18:00: draft TSV sent for the one-time review.
   - Sat 16:00: teammate detectors pass the contract tests and deliver OOF scores on the fold file.
   - Sat 20:00: fusion frozen.
   - Sat 22:00: no new detectors.
   - Sun 00:00: final Docker build, then an amd64 smoke test on 3 files, and a CPU-vs-MPS parity check on 50 files.
   - Sun 05:00: final TSV, with the silence/tone/music/noise preflight (flag any score above 0.8).
   - Sun 05:00–07:30: README, leave-one-detector-out ablation table, AI disclosure.
   - Before 08:00: DM sent by a named person.
8. **`## Cut order`**
   - Cut order: ENF → splice → speaker drift → M5 to one run → orchestrator to static routing → fusion to mean.
   - `Never cut:` a valid TSV at all times; generator- and speaker-grouped validation; the deep SSL detector; Docker; README.
9. **`## Docker`**:
   - Base `python:3.12-slim-bookworm` + apt ffmpeg.
   - CPU torch 2.14.0 / torchaudio 2.11.0 from the PyTorch CPU index. The lockfile's CUDA wheels are skipped with `--no-install-package`.
   - Editable install; `HF_HUB_OFFLINE=1`.
   - Weights: XLS-R + Spectra (both needed) + the probe joblib. The fine-tune goes on the HF Hub or a GitHub release, not git LFS.
   - One model loaded at a time (memory).
   - Per-file resumable cache; new output file per run.
   - `.dockerignore` excludes the WavLM weights, `.venv`, `data/`, `outputs/` and `.cache`.
   - Tested with `--network none`.
10. **`## Risks`**: from the risk agent, High and Medium, one line each with the mitigation.
11. **`## Rules that never move`**
    - Always a valid TSV; never overwrite one; log every TSV.
    - Validate by generator/speaker, not by clip.
    - Never train on In-the-Wild or the validation split, including scalers, calibration and stacking folds.
    - Fine-tune on the H100 only; ask before renting.
    - Time-box every rung.
    - Freeze Sunday morning.
    - Scoring path offline, no hosted API or LLM.
    - Credit every model and dataset.
    - README in our own words.
    - Before every TSV: silence/music/noise preflight, schema, row order, duplicate basenames.

## Step 5. `CLAUDE.md` (edit; keep the Pre-event work and AI disclosure sections)
- **Lines 3–5, "What HEARSAY is":** rewrite as software-only; TSV; new ladder.
- **Lines 7–16, team table:**
  - Nathan: deep detectors, fusion, orchestrator, TSV, M5.
  - Three teammates (names to fill in): engineered-feature detectors (D-track), with one taking K (Docker) from Saturday.
- **Lines 18–24:** replace `## Data contract` with `## Detector contract`, summarizing D3 and pointing to `src/hearsay/detectors/base.py`.
- **Lines 26–28:** new ladder and model rules; "valid TSV"; "log every TSV". The clean-only rule is reworded to "keep a clean-only score beside any augmented one".
- **Line 32:** state `C_FA = 4`, `C_miss = 1`, `π_synth = 0.3` explicitly (A3).
- **Line 34:** "CSV path" → "submission path". The column name `csv_path` stays; it is code-owned (`submission.py:31`).
- **Line 35:** "no hosted API or LLM on the scoring path; the Docker image runs offline".
- **Lines 37–38:** TSVs; the test suite exists.
- **Lines 46 and 50:** TSV; drop "demo".
- **Lines 52–58, Sources of truth:**
  - `docs/nsa-challenge.md` is authoritative.
  - `docs/plan.md` is the working plan.
  - `docs/scoping.md` and `docs/master-doc.md` are historical (pre-rescope).
- **Line 71** (historical "firmware"): keep, marked `(cut)`.
- **Line 85:** WavLM Base becomes a "bake-off/speed spare".
- **Line 97:** "per component (each detector, fusion, orchestrator, Docker)".
- **Add to the disclosure:** MLAAD English (6,390 clips, gated HF, non-commercial notice), the kickoff-slide rules, and Claude Code/Codex use in this rescope. ECAPA and LibriSpeech/VCTK go in when they are used.
- **Lines 107–113:** "/test-runner (suite exists)"; "/zoom-out … someone else's detector".

## Step 6. `README.md`
- **Line 3:** a two-sentence approach summary.
- **Line 5:** "detector contract".
- **Line 14:** tests exist.
- **Lines 18 and 27:** TSVs.
- **New section `## Docker`:** placeholder run command (`docker run --network none -v <test_dir>:/data:ro -v <out>:/out hearsay ...`), marked "planned; see docs/plan.md".
- The README narrative is to be rewritten by the team in their own words before submission; noted in `plan.md` DOC.

## Step 7. `.claude/skills/plan-review/SKILL.md`
- **Lines 30–50, HEARSAY localization:**
  - Ladder names.
  - A 33 h budget to 8 AM Sun.
  - Always a valid TSV + log row.
  - The detector contract replaces the data contract; a change to `DetectorResult` fields must name the affected detector owners.
  - Sources of truth: `nsa-challenge.md` + `plan.md`.
  - Cloud GPU: ask first.
- **Lines 126–143, standing failure modes**, rewritten:
  1. **Replay/laundering/telephony in the test set:** keep a clean-only score beside the augmented one; measure on an augmented validation slice.
  2. **Generator/speaker overfit:** hold out whole groups via the fold file.
  3. **Class prior / metric reading:** π = 0.3, and check the three-reading flip.
  4. **Shortcuts:** metadata/format/silence; per-field AUC; clean-vs-transcoded.
  5. **Loader format errors:** fixtures per format, including MP4.
- **Line 121** ("models/hardware", generic compute sense): leave. The A1 term list doesn't include "hardware".

## Step 8. Codex rubrics
- **`.claude/codex-review-prompt.md`:**
  - Lines 30–32: TSV.
  - Lines 33–34: timeline.
  - Lines 35–36: detector contract.
  - Lines 40–41: scoring path / Docker offline.
  - Lines 42–44: the standing failure modes, mirrored from Step 7.
- **`.claude/codex-audit-prompt.md`:**
  - Lines 59–61: TSV.
  - Lines 62–63: detector contract; scoring path offline.

## Step 9. Other project skills (HEARSAY-localized lines only; generic text left alone)
- **consult:**
  - Lines 24–25: sources.
  - Lines 39–48: the project summary.
  - Line 68: generic, keep.
- **diagnose, lines 12–15:** "ingest → orchestrator → detector → fusion". Line 32 ("replay a captured trace") is generic; keep.
- **doc-sync, lines 19–25:** doc set + detector contract. Line 77 ("issue tracker") is generic; keep.
- **zoom-out, line 9:** pipeline stages.
- **p3:**
  - Lines 47–49: TSV.
  - Lines 52–53: budget.
  - Line 54: "TSV-breaking fixes".
  - Lines 55–56: detector contract.
- **handoff:**
  - Lines 23–26: TSV; `submissions/*` ignore rule.
  - Lines 34–38: ladder / TSV / detector contract.
  - Lines 77–79: template fields.
- **validate-training-data, line 39:** replay/laundering augmentation.
- **test-runner, lines 42–46:** the suite exists (stale text; doc-sync sweep item).

## Step 10. Banners (one added line each; nothing else changes)
- **`docs/scoping.md`:** new line 1 `> **Historical (pre-rescope, Fri Sep 25):** the working plan is docs/plan.md; the spec is docs/nsa-challenge.md.` Line 1 is currently a base64 image, and the banner goes before it.
- **`docs/master-doc.md`:** the same banner as a new line 2, after the title.

## Step 11. `tests/test_docs_consistency.py` (new)

**Setup**
- `LIVING` is the explicit list: `CLAUDE.md`, `README.md`, `docs/plan.md`, plus a glob of `.claude/skills/*/SKILL.md` and `.claude/codex-*.md`.
- Assert that every named file exists and the glob returns at least 9 skill files, so the test can't pass vacuously.

**A1: hardware terms**
- Word-boundary regex, case-insensitive: `listening head|pan-tilt|servo|mic array|4-mic|head driver|direction finding|laser|grandma|live demo|replay capture|firmware|puck|gauge`, plus the case-sensitive `\bLED\b`.
- A line is exempt only if it contains the exact token `(cut)`.
- Cap: at most 6 exempt lines per file, so the exemption can't absorb everything.
- Self-test: an inline string with "pan-tilt servo" must be flagged; "header, labeled, microsoft, issue tracker, demonstrates" must not.

**A2: plan coverage**
Within the `## Forensic techniques` section of `plan.md` (section text sliced between headings):
- Each of 8 regexes must match:
  - `metadata|container`
  - `spectral`
  - `prosod`
  - `ENF`
  - `compression`
  - `speaker[- ]embedding`
  - `anti-spoof`
  - `splice`
- Plus `orchestrat`.

The whole file must match `Docker`, `README`, `TSV`, `MinDCF` and `8 AM`.

**A3: metric constants**
- `metrics.C_FA == 4`, `C_MISS == 1`, `PI_SYNTH == 0.3`.
- `CLAUDE.md` and `plan.md` each match `C_FA\s*=\s*4(\.0)?\b` and `π_synth\s*=\s*0\.3\b`.
- No living doc matches `(π|pi)_?synth\s*=\s*0\.5` except on a line containing `reading`, which is where the three-reading rule names 0.5 deliberately.

**A4: CSV wording**
- Case-sensitive regex `\bCSVs?\b` in `CLAUDE.md`, `README.md` and `plan.md`, excluding lines that contain `log.csv`, `csv_path`, `.csv` or `(cut)`.

**A5: never-cut set**
- The `## Cut order` section of `plan.md` contains a `Never cut:` line mentioning `TSV`, `grouped`, `SSL`, `Docker` and `README`.

**A7: historical docs**
- Uses `git`. Skipped with a reason if `.git` or `git` is absent (Docker).
- The handoff file's blob at `HEAD` equals its blob at `e1a1e88`.
- For `scoping.md` and `master-doc.md`: `git diff e1a1e88 -- <file>` has exactly one added line and zero removed lines.

## Step 12. Doc-sync sweep (before commit)
- Reconcile the P2 doc list against the diff.
- `docs/reports/2026-09-25_sponsor-questions.md`, "What this changes" (says π = 0.5 implemented): add a dated correction line. The report is a record, so annotate it rather than rewrite it. The A3 negative regex only scans living docs, so this is hygiene.
- `.env.example:2`: "live demo path" → "scoring path". `.gitignore:9`: comment CSVs → TSVs. These are one-word fixes found by file-impact; they're outside A1 scope.
- Grep the living docs for `M6`, `data contract`, `valid CSV`, `wired fallback` and `replay capture`; fix any hits.
- Leave `.py` docstrings saying "CSV" (`embed.py:2`, `submission.py:8`, …) for the next code change; not in scope.

## Step 13. Verify
```bash
cd ~/Projects/hearsay && uv run pytest -q
```
```bash
cd ~/Projects/hearsay && uv run ruff check .
```
- Expected: 35 existing tests plus about 22 contract tests, plus the docs tests, plus 1 MP4 test, all passing.
- `git diff --stat e1a1e88` touches only the files listed here.

## Step 14. Commit (no push without approval)
`docs: rescope HEARSAY to software-only multi-detector system; add detector contract`

## Step 15. Post-commit
1. Claude critique loop against `.claude/codex-audit-prompt.md`.
2. Then the Codex audit: `bash .claude/review-audit.sh docs/specs/2026-09-25_software-only-rescope.md`.
3. Record both results in the run spec's Results section; fix and re-audit.

## Risks and mitigations (for this change)
| Risk | Mitigation |
|---|---|
| The doc tests false-positive on generic words | Word-boundary phrases, no bare "head", "demo" or "led"; self-test fixture |
| The doc tests pass vacuously | File-existence and glob-count asserts; section slicing for A2/A5; exemption cap |
| The rewrite drops a rule that must survive | Step 4 §11 lists them; the critique agent checks the old plan's "Rules that never move" and the AI disclosure against the new text |
| `DetectorResult` too strict for real detectors (e.g. an int feature) | Ints and numpy scalars are coerced; only bool, non-numeric and non-finite values are rejected |
| Registry state leaks across tests | Snapshot/restore fixture; a fresh `Registry()` in most tests |
| A7 breaks inside Docker | Skipped with a reason when `.git` is absent; the Docker plan excludes `tests/` |

## Critique amendments (Fable critique; these supersede the steps above)

| # | Severity | Amendment |
|---|---|---|
| C1 | Blocker | `features` uses `field(default_factory=dict)`; a `{}` default fails at import. Applied in Step 1.2. |
| C2 | Major | No `MappingProxyType`: it's unpicklable, and it breaks the frozen dataclass `__hash__`. Store a `dict(features)` copy. Set `eq=True` and `unsafe_hash=False`, so results are not hashable (documented). The B5 "mutation raises" test is dropped; the caller-dict isolation test stays. |
| C3 | Major | Ruff defaults enable TRY004, BLE001 and RUF100. Each `isinstance` → `ValueError` branch gets `# noqa: TRY004 - contract (D3) mandates ValueError`. `safe_run` wraps `applies` and `run` in **one** `try` with a single `# noqa: BLE001`. |
| C4 | Major | `base.py` uses `import hearsay.audio as audio` and calls `audio.load_audio` / `audio.probe_audio`. The B7 test patches `hearsay.audio.load_audio`, so the call counter works. `np.bool_` is already rejected by the `numbers.Real` check (no special branch), but the test stays. `from hearsay.submission import REPO` is dropped (unused). |
| C5 | Major | Step 4 §11 "Rules that never move" adds five rules. (a) Keep a clean-only validation score beside any augmented one. (b) Never denoise or loudness-normalize test audio. (c) Submit calibrated probabilities, never saturated 0/1, and never be confidently wrong on a whole subpopulation. (d) Assert the exported score direction on known clips before any TSV ships (Spectra index 0 = spoof). (e) GPU spend cap about $75, and ask before renting. |
| C6 | Major | A4 runs over the same `LIVING` list as A1 (skills and codex rubrics included). The allowlist is lines containing `log.csv`, `csv_path`, `.csv` or `(cut)`. Steps 8–9 must therefore replace "valid CSV" and "submitted CSV" wording in p3, plan-review, handoff and both codex rubrics. |
| C7 | Major | New A8 test: slice the plan-review SKILL.md block that starts at its standing-failure-modes heading. It must contain `launder\|telephony`, `fold file\|grouped`, `π\|prior`, `shortcut` and `MP4`, and must not contain `if unknown`. |
| C8 | Major | A7: first `git cat-file -e e1a1e88^{commit}`, and skip with a reason if that fails (shallow clone or no git). Assert that the file text, with the one banner line removed, equals `git show e1a1e88:<path>`. The handoff file must equal `git show e1a1e88:<path>` exactly. The docstring says historical docs are frozen by design. |
| C9 | Major | A1 regex uses plural and hyphen variants: `servos?`, `mic[- ]arrays?`, `live[- ]demo`, `listening[- ]head`, `replay[- ]capture`, `head[- ]driver`, `pan[- ]tilt`, `lasers?`, `4[- ]mic`. Section slicing for A2, A5 and A8 splits on `^## ` with `re.MULTILINE`, so `###` subsections stay inside. |
| C10 | Major | Timeline, Step 4 §7. (a) Sleep blocks for Nathan: Sat 02:00–06:00 and Sun 01:00–04:30. (b) If NSA data hasn't arrived by Sat 12:00, the draft review uses the public-data M1 TSV, the OOF gate moves to 19:00, the fusion freeze to 22:00, and ENF and splice are cut immediately. (c) First full amd64 Docker image (M1 payload) by Sat 14:00, built by the teammate on an amd64 machine or via buildx; Sun 00:00 is only a rebuild. (d) The leave-one-detector-out ablation runs on the stored OOF columns at Sat 22:00. (e) The draft TSV is sent by Sat 14:00; NSA's turnaround time is unknown. |
| C11 | Minor | Step 9 edits to the other skills are **required by A1/A4** (the glob covers all skills). `.env.example`, `.gitignore` and the sponsor-questions annotation are **conditional hygiene**: done in the doc-sync sweep and listed in the run spec's Results as hygiene. |
| C12 | Minor | D3 skipped and error results have `features == {}`. Fusion keys on `status == "ok"` and imputes missing values with the train mean; it never feeds the 0.5 placeholder in as a score (stated in `plan.md` §3). |
| C13 | Minor | `ClipContext` private fields use `field(default=None, init=False, repr=False)`. |
| C14 | Minor | `CLAUDE.md` pre-event line 71 is reworded to "no application code (loader, pipeline, model, scoring) was written before the event". No `(cut)` token in text that will be copied to Devpost. |
| C15 | Minor | A3 positive regexes gain a self-test string. `CLAUDE.md` and `plan.md` write the literal `π_synth = 0.3` and `C_FA = 4`. The three-reading line writes `π = 0.5`, not `π_synth = 0.5`, so the negative regex never fires on it; the `reading` exemption is dropped. A2 matches `8 AM` case-insensitively. |
| C16 | Minor | `safe_run`: a `run()` result whose `status != "ok"` is converted to `status="error"` (`"run returned status=<x>"`), so a detector can't smuggle a score through an error status. Test added in B3. |
| C17 | Minor | The A6 verdict (timeline feasibility, from the critique and Codex) is written into the run spec's Results. |

Test count estimate after the amendments: about 24 contract tests plus about 9 doc tests plus 1 MP4 test.

## Codex plan-review amendments (`docs/reports/2026-09-25_software-only-rescope-plan-review.md`; supersede everything above)

| # | Codex | Amendment |
|---|---|---|
| X1 | Critical: fold fallback leaks generators | **Group key = connected components** of the graph linking clips that share a generator (spoof), a speaker, or a near-duplicate hash. Folds then hold out whole generators **and** whole speakers at once. The fold-file builder reports which links were available (e.g. "no speaker IDs in NSA labels"), so any guarantee that doesn't hold is stated, not assumed. This replaces the "speaker → generator → cluster" fallback in Step 4 §6. |
| X2 | Critical: stacking validation not clean | **Nested scheme** in `plan.md` §3/§6. The fold file assigns an untouched **outer holdout** (about 20% of groups) plus 5 inner folds over the rest. Every learned piece (detector training, layer choice, calibration, imputation means, scaler, fusion) is fitted fold-locally on inner training folds. Exported stacking features are inner-fold OOF predictions. Fusion is fitted on inner OOF rows and scored **once** on the outer holdout, where detectors are refit on all inner rows. The logged `validation_score` is the outer-holdout minDCF. |
| X3 | Critical: `safe_run` can raise | `Registry.register` rejects names that aren't non-empty str. `safe_run` resolves the name **inside** the single protected `try`. The fallback name is `"<" + type(det).__name__ + ">"`, which is always valid. New tests: a name property that raises, `name=None`, and `name=""` each give a valid `status="error"` result; registering `name=""` raises. |
| X4 | Critical: native-rate path bypasses the loader | **Deferred.** No native-rate exception in this change. "16 kHz mono everywhere" stays in `CLAUDE.md` and `plan.md` unchanged. `plan.md` lists "native-rate loader API (`hearsay.audio`, with conversion tests)" as a separate follow-up change for the spectral and compression detectors, going through its own review. Step 4 §4's native-rate sentence is removed. |
| X5 | Critical: fusion input unspecified | **Fusion input convention (D3 docstring and `plan.md` §3).** Fusion v1 uses exactly one column per detector: `<name>.logit = logit(clip(score, 1e-4, 1 − 1e-4))` for `status == "ok"`. Otherwise the value is missing, imputed with the fold-local train mean, plus a `<name>.missing` indicator column. Columns are ordered by detector name, logit then missing. `features` is used for explanations and logs only in v1. If every detector is missing for a clip, it gets the constant prior-decision score and a flag. Teammates only need a meaningful, direction-correct `score`. |
| X6 | Critical: no TSV artifact | This change doesn't modify any scoring or writer code. **Blocker: the NSA test set hasn't been downloaded yet**, so no real TSV can be made. Step 13 adds a smoke run of the existing learned-TSV path (`scripts/make_probe_csv.py` with the public M1 probe) on 4 scratch files, then removes the output and the log row. It proves the path still works without logging a non-NSA submission. This is recorded in the run spec's Results. |
| X7 | Suggestion: imbalance vs prior; replay | Standing failure mode 3 becomes "**class imbalance and prior**". The fold builder checks class counts, requires at least 30 clips per class in every inner fold and in the outer holdout (else fewer folds, reported), and requires class-balanced training (`class_weight="balanced"` or a weighted sampler). Failure mode 1 says **physical replay is not covered**: hardware is gone, so the only replay coverage is simulated RIR plus an optional ReplayDF subset. This is a named residual risk in `plan.md` Risks. |
| X8 | Suggestion: hashing | `DetectorResult` sets `__hash__ = None` explicitly. Test: `hash(result)` raises `TypeError`. |
| X9 | Suggestion: post-construction mutation | `safe_run` re-validates the returned result with `dataclasses.replace(res)`, which re-runs `__post_init__` and copies the features. Test: a detector that mutates `result.features["x"] = nan` after construction → `safe_run` gives `status="error"`. |
| X10 | Suggestion: owners and gates | `CLAUDE.md` and `plan.md` name owners as "Nathan" plus "Teammate A/B/C (fill in names)", with the split proposed to Nathan: A = metadata + compression, B = spectral + prosody, C = speaker drift + splice + Docker from Sat. **Moving vs fixed gates:** gates relative to data arrival (inventory, fold file, M1 TSV) move with the data. **Hard cutoffs never move:** Sat 14:00 draft sent (public M1 TSV if there is no NSA data), Sat 19:00 teammate OOF, Sat 22:00 fusion freeze, Sun 00:00 Docker rebuild, Sun 05:00 final TSV, Sun 08:00 DM. The "shift every gate by the delay" wording is deleted. |
