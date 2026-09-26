# Handoff: wiring the frontend to HEARSAY, with the fusion section (Sat Sep 26, 2026, ~09:20)

**Purpose of this chat:** build Hrushi's UI against the live `hearsay.api` and the per-file JSON it returns, and make the **fusion section** of each result page show, in plain words, how the score was made: which detectors were ranked, with what weights, whether Spectra-AASIST pulled the file toward real, and where the final probability landed. Everything the UI shows must come from the pipeline's JSON; the UI never computes a score.

Read first: `docs/handoffs/2026-09-26_frontend-contract.md` (the JSON contract, v0), then this file. Skim `docs/architecture.md` for the picture and `docs/reports/2026-09-26_fusion-sweep-predeclared.md` for why the rule is what it is.

## Context

- The backend is done. `hearsay.api` (FastAPI, `src/hearsay/api.py`) serves `POST /analyze` (multipart upload → one `AnalyzeResponse`), `GET /results/{filename}` (a precomputed response from a runner output directory), `GET /results` (list) and `GET /health`. It wraps the same `hearsay.pipeline.analyze_clip` the Docker image runs, so what the UI shows is what NSA gets.
- The shipped fusion rule is `e_on_a`, frozen at 08:13 and read from `models/fusion_v1/constants.json`. The API and the runner default to it since commit `9614c18`.
- A full dump of all 1,671 test files exists at `outputs/runner/full_20260926/results/<filename>.json`, but it was made at 07:30 with the **rejected** `zmean` rule. Same JSON shape, wrong numbers: fine for layout work, wrong for any demo, screenshot or number that anyone will read. Regenerate it (step 2 below) before showing it to a person.
- No web framework code exists for the UI yet. Nothing under `web/`.

## Working branch / worktree

`main` in `~/Projects/hearsay`, shared by several parallel Claude chats. Other lanes commit their own files and the main chat pushes. Uncommitted files belong to whichever lane is editing them; do not touch them.

## Environment / setup

```bash
cd ~/Projects/hearsay && uv sync
```

Start the API (models load on the first request; ~15 s, then 1.5–4 s per file on CPU):

```bash
cd ~/Projects/hearsay && HEARSAY_RESULTS=outputs/runner/full_20260926 uv run uvicorn hearsay.api:app --port 8000
```

Check it:

```bash
curl -s localhost:8000/health
```

```bash
curl -s localhost:8000/results/HGT1013455.wav | python3 -m json.tool | head -60
```

Environment variables the API reads (all optional): `HEARSAY_RESULTS` (runner output dir for `GET /results`), `HEARSAY_RULE` (default `e_on_a`), `HEARSAY_FUSION` (a constants file; default `models/fusion_v1/constants.json`), `HEARSAY_DETECTORS` (default `m1b,spectra,handcrafted`), `HEARSAY_POLICY` (default `speech_gate`).

**CORS: there is none.** A browser app on its own dev port cannot call `localhost:8000` directly. Do not add CORS to `api.py`; it is runner code and the Docker lane rebuilds on every change. Proxy instead. Vite example, `web/vite.config.ts`:

```ts
export default defineConfig({
  server: { proxy: { "/analyze": "http://localhost:8000", "/results": "http://localhost:8000", "/health": "http://localhost:8000" } },
});
```

## What to do next

1. **Scaffold `web/`** with whatever stack Hrushi prefers (Vite + React is the obvious one). Put the dev-server proxy in first so `fetch("/results/HGT1013455.wav")` works on day one. Commit a package lockfile; nothing else in the repo is JavaScript.
2. **Regenerate the results dump under the shipped rule.** About 22 minutes on the Mac (0.78 s/file). Run it while the Docker chat is not doing a full in-image run, since both want the CPU:
   ```bash
   cd ~/Projects/hearsay && uv run python scripts/run_pipeline.py --data data/nsa/HackGTHearsayTesting --out outputs/runner/full_e_on_a --template data/nsa/HearsayScoreKey4TeamX.tsv
   ```
   Then point the API at it: `HEARSAY_RESULTS=outputs/runner/full_e_on_a`. Verify one file's `fusion.rule` says `e_on_a` and `version.git_sha` is current. The runner is resumable; a second invocation with the same `--out` continues.
3. **Copy the regenerated `results/` into the UI's fixtures** (e.g. `web/fixtures/results/`) so the UI works without the API running. `outputs/` is gitignored; the fixtures copy is not, and 1,671 JSONs are ~10 MB, which is fine to commit once. Do not commit the zmean dump.
4. **Build the result page.** Sections, in order: verdict + probability; the fusion section (below); the detector table (name, role, status, score, evidence sentence, seconds); the routing log as a list; the version block folded away.
5. **Build the upload flow** against `POST /analyze` (field name `file`, any audio format FFmpeg reads; 16 kHz mono is enforced inside). Show a spinner for 2–4 s. The response is the same shape as `GET /results`.
6. Ask Nathan before pushing. Stage with `git add web/`, never `git add -A`.

## The fusion section: what the JSON says and what the UI should show

The rule, in one sentence: **rank M1b v3 and the handcrafted detector against their own training distributions, blend 80/20, and if Spectra-AASIST is confident the voice is real, halve the blended rank; then map to a probability.** Spectra never pushes a file toward synthetic.

`response.fusion` for `e_on_a`:

| Field | Meaning | Show as |
|---|---|---|
| `rule` | `"e_on_a"` | "Rule: 80% XLS-R probe, 20% handcrafted, Spectra as false-alarm check" (constant text keyed on the rule) |
| `inputs` | raw logit of each of `m1b_v3`, `handcrafted_v5`, `spectra_aasist`; `null` when a scorer failed | the raw number, small, with a "higher = more synthetic" note (for `spectra_aasist` the value is the model's margin: negative = real) |
| `terms` | for `e_on_a`: the rank (0–1) of `m1b_v3` and `handcrafted_v5`. Missing scorers get 0.5 | two bars, 0–1: "where this file sits among the training clips that detector saw" |
| `weights` | `{"m1b_v3": 0.8, "handcrafted_v5": 0.2, "spectra_aasist": 0.0}` | the 80/20 split; Spectra's 0 weight is worth showing with the words "suppression only" |
| `detail.base` | the blended rank before Spectra: `0.8·rank_m1b + 0.2·rank_hc` | one bar, labelled "blended rank" |
| `detail.e_applied` | `true` when Spectra's margin was below −3 **and** `base` was above 0.5, so `base` was halved | a badge: "Spectra pulled this toward real (margin −4.2)" when true; omit when false. Never invent the converse: Spectra cannot pull toward synthetic |
| `detail.e_rule` | the thresholds `{"m3_logit_below": -3, "base_rank_above": 0.5, "multiply_by": 0.5}` | tooltip on the badge |
| `fused` | `base`, or `base × 0.5` if suppression applied | the number behind `p_fused` |
| `p_fused` | the probability after the Platt map, in [0.001, 1] | this is `probability_synthetic` unless the gate fired |
| `imputed` | list of scorers that failed and were replaced by the inner-fold median (0.5 rank) or, for Spectra, skipped | a warning chip per name: "detector unavailable, neutral value used" |
| `block_top`, `failure_top` | the ceilings of the pinned block (files that are not speech or failed to decode sit below `block_top`; decode failures below `failure_top`) | only needed to explain a tiny score on an `undetermined` file |

Top-level fields that frame the section: `probability_synthetic` (what the TSV carries), `verdict` (`synthetic` ≥ 0.5, `real` below, `undetermined` when not speech or decode failed), `is_speech`, `default_answer_applied` (true means the file was pinned below every determinate score, and the fusion numbers are informational only). The routing log's last line is a ready-made sentence for the section footer, e.g. `fusion: e_on_a over m1b_v3, handcrafted_v5; M3 margin −7.07, no suppression (M3 never promotes)`.

Things the UI must not do: recompute `probability_synthetic` from `inputs`; show `weights` as if Spectra had a share of the score; describe the 0.5 verdict cut as the operating point (NSA's cost puts the real threshold near 0.9; say "more likely synthetic than not" rather than "flagged").

Wording that is safe to hardcode for the roles in `detectors[].role`: `fused` "enters the score"; `evidence` "shown, never scored" (compression, ENF, splice, speaker drift); `routing` "decides what runs" (container); `gate` "decides whether the file is scoreable at all" (speech gate).

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test (backend, hermetic): `uv run pytest -q tests/test_api.py tests/test_pipeline.py` → all pass at `9614c18` (part of the 444-test suite). Run it if anything about the API's responses looks off; do not edit these files.
- Test (docs): `uv run pytest -q tests/test_docs_consistency.py` → 52 passed. Run after editing any doc under `docs/` or the README.
- There are **no frontend tests** yet. Add at least a fixture-shape test (every fixture JSON has `fusion.rule == "e_on_a"` and `contract == "v0"`) so a stale dump cannot sneak into a demo.
- At-risk: `outputs/runner/full_20260926/` (the zmean dump; regenerable, NOT archived, and fine to lose once step 2 is done).
- At-risk: `outputs/runner/full_e_on_a/` once created (regenerable in 22 min; copy `results/` into `web/fixtures/` and commit, which is the archive).
- In flight, other lanes: the Docker chat's full in-image run under `outputs/docker/k-full/` (competes for CPU; check with `ls outputs/docker/k-full/results | wc -l`), the main chat's M5 fusion candidates (`scripts/fuse_sweep_m5.py`; if a candidate is ratified the constants file and the rule name may change before the 22:00 freeze, and the dump would need regenerating again, so keep the fixture copy step cheap and scripted).

## Analytical notes

- Score direction: 1.0 = synthetic everywhere in the JSON and the TSV. Never flip in the UI. The pre-flipped TSV exists for one contingency and is not the UI's business.
- Expected numbers on the test set under `e_on_a`: 27.4% of files above 0.5 (458 of 1,671, from the 08:13 TSV); determinate scores in [0.001, 1]; 0 of 1,671 files gated; `e_applied` true on a minority of files (Spectra's margin is below −3 on most real speech but `base` is above 0.5 on few of those).
- Timings: engineered detectors ~0.3 s per file after warm-up; M1b ~1–3 s; Spectra ~0.5 s. The first request pays the model load.
- The frontend contract's sample JSON predates the shipped rule: its `fusion` block shows `stack_nonlj` weights and lacks `terms`, `detail`, `block_top`, `failure_top`. The table above is authoritative for `e_on_a`.
- Container evidence is constant on this test set (all 1,671 files are FFmpeg-written 16 kHz WAV); do not design the routing display around it being interesting.

## Boundaries

- The UI lives in `web/` only. Never edit `src/hearsay/`, `scripts/`, `docker/`, `tests/` or `models/`; owners are listed in `docs/code-map.md`. If the API needs something (CORS, a new field), ask the oversight chat, which relays to the runner's owner.
- No hosted model on the scoring path. A language model may reword evidence text in the UI; every number comes from the JSON.
- Stage only `web/` (`git add web/`); never `git add -A`; do not push without asking Nathan.
- The TSV, the Docker image and the README are the graded deliverables. The UI must not pull Nathan or the Docker owner off them before Sunday 08:00.

## Pointers

- `docs/handoffs/2026-09-26_frontend-contract.md`: the JSON contract and the detector-role rules.
- `src/hearsay/api.py`: the three endpoints, ~170 lines, readable in five minutes.
- `src/hearsay/pipeline.py`: `FusionConstants.fuse` (the rule's arithmetic) and `fusion_block` (what lands in the JSON).
- `models/fusion_v1/constants.json`: the frozen constants; `how` is a one-line description of the rule.
- `docs/reports/2026-09-26_fusion-sweep-predeclared.md`: the sweep that chose the rule, with the numbers.
- `docs/reports/2026-09-26_worked-examples.md`: eight test files walked through by hand; good copy for an "examples" page.
- `docs/architecture.md` and `docs/img/architecture-flow.svg`: the diagram to reuse on the landing page.
