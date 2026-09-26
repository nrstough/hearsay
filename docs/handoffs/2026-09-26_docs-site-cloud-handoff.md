# Handoff — docs-site lane, continued in a Claude Code cloud session (Sat Sep 26, 2026, ~13:35)

**Purpose of this chat:** continue the docs-site lane from where the Mac session stopped: run `/plan-review` on `docs/handoffs/2026-09-26_docs-site-handoff.md` (the full brief; read it first, it is the spec) and then build the site under `docs/site/`. The Mac session got as far as Phase 0/1 reading; nothing was written.

## Context

- The brief is `docs/handoffs/2026-09-26_docs-site-handoff.md` (committed, on `origin/main` at `41141db`). Everything below is what the Mac session learned on top of it.
- **Phase 0 conventions, resolved** (state them in P1): run specs `docs/specs/`, plan files `docs/reports/YYYY-MM-DD_<slug>-plan.md`, no feature-spec layer (CLAUDE.md: "No feature-spec layer"), super docs `README.md` + `docs/architecture.md` + `docs/STATUS.md` (this lane may **not** edit them; route changes to the oversight chat), bug log none (skip the historical-bug sweep and say so), full test command `uv run pytest -q`, lint `uv run ruff check .`, Codex scripts `.claude/review-plan.sh` / `.claude/review-audit.sh`.
- **Codex is not available in the cloud** (the CLI and its login live on the Mac). CLAUDE.md's rule applies: "without it, the plan-review Claude critique is the gate." So in Phase 3c run a Claude critique agent against `.claude/codex-review-prompt.md` instead of the script, and in Phase 4 run the Claude critique loop against `.claude/codex-audit-prompt.md` and record in the run spec that the Codex pass is pending on the Mac. Do not fake a Codex output.
- **Source docs already read on the Mac** (re-read in the cloud; they are the numbers' source): `README.md` (397 lines; sections: How it decides, Results at a glance, Approach and architecture, Validation, What worked, What did not work, Numbers, Reproduce it, Repository layout, AI use and credits), `docs/architecture.md` (five Mermaid blocks: XLS-R/M1/M5, Spectra-AASIST, engineered detectors, validation, fusion; plus `docs/img/architecture-flow.{mmd,svg}`), `docs/STATUS.md`, `docs/reports/2026-09-26_worked-examples.md` (eight files; file 3 `HGT3739624.wav` is the one where M3 suppression fired; file 4 `HGT1046947.wav` is the README's end-to-end trace under the shipped rule), `docs/reports/2026-09-26_fusion-sweep-predeclared.md`, `docs/reports/2026-09-26_draft-review-branches.md`, `docs/consults/2026-09-26_fusion-strategy_RESPONSE.md`, `docs/nsa-challenge.md`.
- **Facts checked on the Mac:** `docs/site/` does not exist yet; `markdown` is not installed (`uv run python -c "import markdown"` fails), so a generator must be standard library only; Mermaid from jsDelivr is the allowed live renderer, `docs/img/architecture-flow.svg` (33 KB) is the offline fallback for the main flow; `.gitignore` does not touch `docs/`; ruff `src = ["src", "tests"]`, line length 100.
- **Numbers the Home page must carry, checked against the README:** shipped rule `A3 w 0.2 + E` inner 0.135 / holdout 0.0065 / In-the-Wild 0.228 (brief cost) and 0.239 (sponsor-code cost); previous rule `E on α 0.2` 0.140 / 0.014 / 0.260 / 0.267; M1b v3 alone 0.301 / 0.072 / 0.343; Spectra ITW 0.065; test share above 0.5 = 27.4% for the shipped rule (the README's Numbers table; the orchestration-ablation table says 27.5% for "router on", which is a different readout, so cite each to its own table); 1,671 files, 0 gated; NSA's returned draft minDCF still pending. Router on vs off (README, Orchestration): holdout 0.0065 vs 0.0200, In-the-Wild 0.229 vs 0.274. Where the brief and the README differ, the README wins.
- **Recent, since the brief was written:** the channel-robustness lane finished (Sat 13:20): λ̂ = 0.51 (CI 0.12–0.64), the codec hunt was negative (the 7.2 kHz wall is a resampling low-pass, not MP3/AAC), M3 suppression kept (most perturbation-stable detector, lowest miss rate on MLAAD), new weak spot: 20 dB white noise takes the shipped rule to 0.27 holdout. Its report is `docs/reports/2026-09-26_channel-robustness.md`; part of it is still uncommitted on the Mac (see at-risk), so the cloud checkout may have an older version. Copy only what the committed text says.

## Working branch / worktree

The Mac session was on `main` in `~/Projects/hearsay`, 1 commit ahead of origin (`73f6a5b`, another lane's channel-robustness commit) with 7 uncommitted files from other lanes (`CLAUDE.md`, `docs/STATUS.md`, the channel-robustness report and spec, the M5 audit spec, `scripts/m3_probes.py`, `tests/test_channel_robustness.py`). None of those belong to this lane and none were touched. The cloud session starts from `origin/main` (`41141db`), which has the brief and everything this lane needs.

In the cloud: work on the branch the cloud session gives you. Stage only `docs/site/` and, if written, `scripts/build_site.py` and your run spec / plan / this lane's report; never `git add -A`. Push that branch, not `main`; Nathan merges. Tell the oversight chat the commit hash when a version is ready.

## Environment / setup

```bash
uv sync
```

The cloud has no `data/`, `weights/`, `models/` or `outputs/`; this lane needs none of them. Tests needing them skip by marker. If `uv sync` is slow (torch), it is still required for `tests/test_docs_consistency.py`, which imports `hearsay.metrics`.

Preview: `cd docs/site && python3 -m http.server 8080`, and use whatever browser tool the cloud session has; if none, check the HTML structurally (link check below, and `python3 -m html.parser`-level sanity) and say so in the run spec.

## What to do next

1. Re-read the brief `docs/handoffs/2026-09-26_docs-site-handoff.md` in full. It has the page list (eight pages, first four carry the grade), the diagram list, the design rules, the sync-pass schedule and the "never invent a number" rule.
2. Invoke `/plan-review` with that file as the argument. Present P1 (problem, approach, the resolved conventions above) and stop for "freeze". Then P2 with a failure-mode table (stale number vs its source, broken internal link, Mermaid CDN unreachable → SVG fallback, dark mode, phone width, a number with no source link, a page that contradicts the README, the footer's commit sha wrong), then the run spec at `docs/specs/2026-09-26_docs-site.md`, the plan at `docs/reports/2026-09-26_docs-site-plan.md`, the Claude critique standing in for Codex, stop for approval, execute.
3. Recommended build shape (decide in the plan, not here): hand-written HTML with one shared `site.css` and `site.js`, Mermaid from jsDelivr with the committed SVG as fallback for the main flow, and a tiny standard-library Python checker (links, numbers-with-sources, footer sha) rather than a full markdown-to-HTML generator, since `markdown` is not installed and a generator would have to handle the README's tables and Mermaid blocks.
4. Sync passes: after the first complete version, and again late tonight, re-read `README.md` and `docs/STATUS.md` on the current `main` and copy any changed number (expected: NSA's returned minDCF, the channel lane's Spectra-probe verdict, a possible final-rule note). Tell the README chat what the site says.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test: `uv run pytest -q tests/test_docs_consistency.py` → 52 passed (Mac, before this session; re-run in the cloud to confirm the checkout). It does not look at `docs/site/`, but run it after every sync.
- Full suite: `uv run pytest -q` → 510 passing on the Mac at cc6670c with data; in the cloud many skip. Lint: `uv run ruff check .` → clean.
- Link check (no dependency), from `docs/site`: `uv run python -c "import re,pathlib,sys; bad=[(p,l) for p in pathlib.Path('.').rglob('*.html') for l in re.findall(r'href=\"([^\"#:]+)', p.read_text()) if not (p.parent/l).exists()]; print(bad or 'links OK'); sys.exit(bool(bad))"` → `links OK`.
- At-risk on the Mac, not this lane's: the 7 uncommitted files listed above (channel-robustness lane). Not archived; that lane or the oversight chat commits them. Do not recreate them in the cloud.
- At-risk, this lane: nothing yet. Everything this lane makes is committed under `docs/site/`.
- In flight, other lanes: README polish until ~05:00 Sunday; NSA's returned draft minDCF (placeholder until then); Hrushi's frontend rewire (`web/`, `src/app/`, `src/components/`, `package.json`: do not touch).

## Analytical notes

- Score direction 1.0 = synthetic everywhere; on a score-line diagram, synthetic at the top, the pinned block below 0.001.
- Ten detectors: fused `m1b_v3`, `handcrafted_v5`, `m5_xlsr_ft`, `spectra_aasist` (suppression only); evidence `compression`, `enf`, `splice`, `speaker_drift`; routing `container`; gate `speech_gate`. Rubric mapping is in the brief's Analytical notes.
- The fusion arithmetic for a worked file (brief page 2): base = 0.6·rank(M1b) + 0.2·rank(handcrafted) + 0.2·rank(M5); if Spectra margin < −3 and base > 0.5, base ×0.5; p = 0.001 + 0.999·sigmoid(a·base + b + logit 0.3) with the Platt constants in `models/fusion_v2/constants.json` (not in the cloud; the fusion_v1 constants 17.816 / −8.302 / −0.847 are quoted in the worked-examples report, the v2 constants are not quoted in any markdown, so show the formula with the v1 numbers only if labelled as v1, or show it symbolically).
- Freeze is Sunday morning; this lane never touches the TSV.

## Pointers

- `docs/handoffs/2026-09-26_docs-site-handoff.md`: the brief (read first).
- `CLAUDE.md`: conventions, the AI-use disclosure to copy for the Credits page, the working rules.
- `README.md`, `docs/architecture.md`, `docs/STATUS.md`, `docs/reports/2026-09-26_worked-examples.md`, `docs/reports/2026-09-26_fusion-sweep-predeclared.md`, `docs/reports/2026-09-26_draft-review-branches.md`, `docs/consults/2026-09-26_fusion-strategy_RESPONSE.md`, `docs/nsa-challenge.md`, `docs/code-map.md`.
- `.claude/skills/plan-review/` (or `~/.claude/skills/plan-review/` on the Mac), `.claude/codex-review-prompt.md`, `.claude/codex-audit-prompt.md`: the pipeline and the two rubrics the Claude critiques must use.
