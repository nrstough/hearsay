# Run spec: software-only rescope (Fri Sep 25, 2026, ~23:00)

Status: P1 and P2 frozen (deep mode). Plan: `docs/reports/2026-09-25_software-only-rescope-plan.md`.
Branch `main`, worktree `~/Projects/hearsay`. Time box: about 1 h (planning and docs, plus one small contract module).

## Problem

The working plan (`docs/plan.md`, written 1:30 pm Fri) is built around a hardware demo (pan-tilt listening head, mic array, replay capture, live tracker) driven by essentially one detector. The NSA brief (`docs/nsa-challenge.md`, received ~22:40) and the kickoff notes (`docs/reports/2026-09-25_sponsor-questions.md`) grade something else:

- **60% detection performance.** MinDCF with false alarms (real called synthetic) weighted 4×; 70/30 real/synthetic said aloud (slide reads ~50/50).
- **20% forensic diversity.** "Points for each distinct technique your system actually leverages": container/metadata, spectral, prosody/phonetics, acoustic environment (ENF), compression forensics, speaker-embedding consistency, deep-learning anti-spoofing, splice/discontinuity, plus an **agentic orchestration bonus**.
- **20% documentation and presentation.** Includes "can the algorithm explain why a clip is synthetic/real".
- **Required deliverables.** GitHub repo with README (approach, architecture, run instructions), a **Docker image** that runs inference on the test set, and `teamName_predictions.tsv` (`filename<TAB>cm-score`, filename with extension, probability 0–1, 1.0 = synthetic).
- **Test data may include** bona fide audio over studio, smartphone, telephony and field channels; zero-shot TTS; voice conversion; replay; scene manipulation; laundered fakes (transcoding, band-limiting, noise); spoofed metadata. Clips are at least 2 s and English.

The team decided (Nathan, Fri ~23:00) to drop all hardware. The three teammates now work on feature engineering.

## Solution

### D1. Scope: software only
Cut: the listening head, pan-tilt, servos, mic array, direction finding, tracker, head driver, laser, LED, gauge, puck, docks, enclosure, live demo, replay-capture session, the four-stage hardware data contract, and the "grandma device" requirements. Keep the offline detector and everything that produces, explains, and packages its output.

### D2. Architecture
One pipeline per file:
1. **Ingest.** `hearsay.audio.probe_audio` (container/codec facts) plus `load_audio` (16 kHz mono float32).
2. **Orchestrator.** Rule-based: chooses which detectors run from container, codec, sample rate, duration, and early findings; records why. This earns the orchestration bonus without an LLM on the scoring path.
3. **Detectors.** Each emits a `DetectorResult` (D3).
4. **Fusion.** Regularized logistic stacking on grouped-CV out-of-fold detector outputs. Missing or skipped detectors are imputed, with a missing-indicator feature.
5. **Outputs.** The TSV, plus a per-file explanation report: top contributing detectors and their evidence strings.

### D3. Detector contract (`src/hearsay/detectors/base.py`; the only code in this change)
- **`ClipContext`.** Path, decoded 16 kHz mono audio, probe metadata. Audio is decoded lazily and at most once.
- **`DetectorResult`.**
  - `name`: str
  - `score`: float, finite, in [0, 1], **increasing with synthetic likelihood**
  - `evidence`: non-empty str
  - `features`: flat `dict[str, float]` of finite numbers, for fusion
  - `status`: `"ok" | "skipped" | "error"`
  - `error`: str or None
- **`Detector` protocol.**
  - `name`: str
  - `applies(ctx) -> bool`
  - `run(ctx) -> DetectorResult`
- **`safe_run(detector, ctx)`.**
  - Not applicable: returns `status="skipped"` without calling `run`.
  - Exception or contract violation: returns `status="error"` with a neutral score of 0.5, empty features, and the error message.
  - Never raises, so a file never loses its row.
- **Registry.** `register(detector)`, `get(name)`, `all_detectors()`. A duplicate name raises.
- **Validation.** Enforced in `DetectorResult.__post_init__`. Violations raise `ValueError`, which `safe_run` catches and turns into `status="error"`. Values are never clipped silently.

### D4. Ladder (replaces M0–M6)
| Rung | Deliverable | Owner |
|---|---|---|
| M0 | Loader + TSV writer + constant rollback (built; needs NSA data) | Nathan |
| M1 | Frozen XLS-R probe on NSA train → first learned TSV (built on public data) | Nathan |
| D-track | Engineered-feature detectors against the D3 contract: metadata/container, spectral, prosody, compression, speaker drift, splice, ENF | Teammates |
| M3 | Spectra-AASIST as a second deep detector (wrapper built) | Nathan |
| M5 | XLS-R fine-tune on the H100 (codec, telephony, noise, replay, band-limit augmentation) | Nathan |
| M4 | Fusion + orchestrator + explanation report | Nathan |
| K | Docker image (CPU inference on the test set) | Nathan + one teammate |
| DOC | README "in our own words", what worked / what didn't, experiment log | All |

### D5. Metric
Normalized MinDCF with C_FA = 4, C_miss = 1, π_synth = 0.3 (= 9.33·P_FA + P_miss), from `hearsay.metrics`. Every selection decision uses it; EER is reported alongside.

### D6. Data
- NSA train is primary, split generator- and speaker-disjoint where the labels allow.
- Public: MLAAD English (6,390 clips pulled), plus a multi-speaker bona fide source (new; LibriSpeech/VCTK suggested by the brief).
- ASVspoof 2019 is baseline only; In-the-Wild is a stress test only.
- Augmentation targets the brief's attack list: codec/telephony, noise, room/replay, band-limiting, transcoding.

### D7. Timeline
Now through 8 AM Sunday, with the Docker and README time boxes on Saturday and the one-time draft review on Saturday.

### D8. Cut order and never-cut set
- **Never cut:** a valid TSV at all times; generator- and speaker-grouped validation; the deep SSL detector; Docker; README.
- **Cut order:** ENF → splice → speaker drift → M5 to one run → orchestrator back to run-all (static routing) → fusion to mean.

### D9. Docs
- `docs/plan.md` is rewritten.
- `CLAUDE.md` is updated: purpose, ownership, detector contract instead of the data contract, ladder, rules.
- `README.md` is updated.
- The HEARSAY section of the project plan-review skill and the Codex rubrics are rewritten for software-only.
- Historical docs get a banner only.

### D10. Consistency tripwire
`tests/test_docs_consistency.py` pins the doc–code agreements listed below (A1–A5) so they can't drift silently.

### Plan-review amendments to D3 (critique + Codex; see plan C1–C17, X1–X10)
- `features` is a plain `dict` copy, with `field(default_factory=dict)`.
- `__hash__ = None` is set explicitly.
- `safe_run` resolves the detector name inside its one `try` and falls back to `"<ClassName>"`.
- `safe_run` re-validates the returned result via `dataclasses.replace`.
- A result from `run()` whose status isn't `ok` becomes `error`.
- The registry rejects names that aren't non-empty str.
- **Fusion input convention:** one column per detector, `logit(clip(score, 1e-4, 1-1e-4))` when `status == "ok"`, else missing plus a missing indicator.
- Native-rate audio is deferred to a separate change; "16 kHz mono everywhere" stands.

## Acceptance criteria

**Docs (`tests/test_docs_consistency.py` unless noted)**

| # | Criterion |
|---|---|
| A1 | No hardware term in the living docs (`CLAUDE.md`, `README.md`, `docs/plan.md`, `.claude/skills/*/SKILL.md`, `.claude/codex-*.md`) except on lines that explicitly mark it as cut |
| A2 | `docs/plan.md` names all 8 techniques, the orchestration bonus, Docker, README, TSV, MinDCF, and 8 AM |
| A3 | `hearsay.metrics.C_FA == 4`, `PI_SYNTH == 0.3`, and `CLAUDE.md` and `docs/plan.md` state the same values |
| A4 | No submission "CSV" wording in living docs (allowlist: `log.csv`, `.csv` manifest/sidecar paths, historical docs) |
| A5 | `docs/plan.md` lists the never-cut set from D8 |
| A6 | Timeline feasible, including a Docker slot before 8 AM Sunday (manual plus Codex plan review) |
| A7 | `docs/scoping.md`, `docs/master-doc.md`: diff is a banner line only; `docs/handoffs/2026-09-25_first-two-hours-handoff.md` is unchanged (git diff check) |
| A8 | The plan-review skill's standing failure modes are rewritten for software-only (covered by A1 on skill files) |

**Contract (`tests/test_detector_contract.py`)**

| # | Criterion |
|---|---|
| B1 | Non-finite or out-of-range score raises `ValueError` in `DetectorResult`; never clipped |
| B2 | Score-direction convention documented; a toy detector scores a synthetic-marked fixture above a real-marked one |
| B3 | `safe_run` turns an exception into `status="error"`, score 0.5, message kept; counterfactual: calling `run` directly raises |
| B4 | Empty or whitespace evidence rejected |
| B5 | Non-flat, non-numeric, or non-finite feature values rejected |
| B6 | Duplicate registry name raises |
| B7 | Same clip scored twice gives an identical result |
| B8 | `applies()` false gives `status="skipped"` and `run` is not called |
| B9 | `safe_run` never raises for a bad detector name (property raises, `None`, `""`); the registry rejects bad names; `hash(result)` raises `TypeError`; post-construction feature mutation is caught by `safe_run` re-validation; a non-`ok` status from `run` becomes `error` |

**Also:** MP4 audio container added to the loader format test (standing failure mode 5). All previously passing tests still pass; ruff clean.

## Test commands
```bash
cd ~/Projects/hearsay && uv run pytest -q
```
```bash
cd ~/Projects/hearsay && uv run ruff check .
```

## Standing failure modes (project plan-review skill)
1. **Live replay:** replaced by replay/laundering in the NSA test set, handled in the plan via augmentation plus a validation slice. No test in this change.
2. **Generator overfit:** grouped validation stays in the never-cut set (A5).
3. **Class imbalance:** π = 0.3 is pinned (A3).
4. **Metric:** MinDCF is known and pinned (A3).
5. **Loader formats:** MP4 fixture added.

## Results (execution, Sat Sep 26 ~00:10)

**Tests.** `uv run pytest -q` → 145 passed; `uv run ruff check .` clean.
- Contract: 57 parametrized cases in `tests/test_detector_contract.py`, covering B1–B9.
- Docs: 52 in `tests/test_docs_consistency.py`, covering A1–A5, A7 and A8. A counterfactual check confirmed the A1 and A4 regexes flag the pre-rescope versions of `plan.md` (15/15 hits), `CLAUDE.md` (6/5), `p3` (0/3) and `codex-review-prompt` (1/2).
- Loader: an MP4 case added to the format test.

**A6 (timeline feasibility).** Reviewed by the Fable critique (C10) and Codex (X10). Sleep blocks, moving gates versus hard cutoffs, a no-data fallback at Sat 12:00, and an amd64 Docker image by Sat 14:00 are all in `docs/plan.md`.

**TSV (Codex X6).** No scoring or writer code changed. The NSA test set arrived during execution, but no NSA-trained model exists yet, so a real learned TSV is blocked on training data (DiffSSD still downloading). A smoke run of `scripts/make_probe_csv.py` on 4 scratch files wrote a valid TSV; it was deleted and the log row reverted, so no non-NSA submission was logged.

**Deviations from the plan.** New facts arrived during execution (NSA instructions PDF, test set, scoring code, inventory):
- `docs/plan.md` now records:
  - the score-direction blocker: the sponsor's ASVspoof5 code treats higher = bona fide, and an unmodified run gives minDCF 1.0 for a good detector submitted as 1.0 = synthetic
  - a fourth metric reading (the sponsor code as shipped)
  - LJ Speech + DiffSSD as the NSA training data
  - the test-set inventory: all WAV, 16-bit PCM, 16 kHz mono, 3.0–13.6 s, peaks near full scale
  - duration and peak-level shortcuts against LJ
- The "one-flag flip in `write_submission`" is planned for the next change; it is **not** implemented here.

**Hygiene (conditional; C11).**
- `.env.example` ("live demo path" → "scoring path").
- `.gitignore` comment (CSVs → TSVs).
- `docs/reports/2026-09-25_sponsor-questions.md`: three dated correction notes (π_synth 0.5 → 0.3; the sponsor code's cost model).
- `validate-training-data` SKILL line 20 (generic "parquet/CSV" → "parquet or .csv tables") to satisfy A4 over all living docs.

**Not changed.** The `.py` docstrings that still say "CSV" (`embed.py`, `submission.py`, …) are left for the next code change; they are outside A1/A4 scope.

**Post-commit audit.**
- **Claude pre-audit critique (Fable), round 1: Acceptable overall.**
  - Freeze integrity and Regression check: Excellent. Every other dimension: Acceptable.
  - Findings fixed:
    - `safe_run` message building is now guarded; an exception whose `__str__` raises is contained (test added).
    - `ClipContext.from_array` rejects non-1-D input.
    - `ClipContext` private fields set `compare=False` (test added).
    - The final DM sender is named.
    - The preflight threshold of 0.8 is restored.
    - The `CLAUDE.md` claim about what `write_submission` enforces is corrected.
    - A "planned, not yet downloaded" disclosure stub is added.
  - Kept as is: the external-drive path added to the handoff skill (the team's only archive location).
- **Logged for the next submission-path change, not fixed here (out of scope):** `list_test_files` reads `--manifest` with a comma `csv.DictReader`, so the TSV template `data/nsa/HearsayScoreKey4TeamX.tsv` would fail. That change also adds the score-direction flip flag.
- **Codex audit:** _pending._
