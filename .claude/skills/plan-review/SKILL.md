---
name: plan-review
description: "Full change pipeline — P1, P2, Spec + Plan (native + Codex-reviewed), Execute with doc propagation"
user_invocable: true
---
<!-- GLOBAL-SKILL:BEGIN (synced from ~/.claude/skills, do not edit inside) -->

# /plan-review — Full Change Pipeline

**Repo additions:** if the current repo has `.claude/skills/plan-review/SKILL.md` containing
`GLOBAL-SKILL:BEGIN`/`END` markers, also follow everything in that file *outside* the
markers. It holds this project's additions to this skill and wins where they conflict.
(If you are already reading that file, there is nothing extra to load.) A repo copy
without markers is ignored. **Path rule:** wherever this skill names
`~/.claude/ownership-checks.md`, use the repo's `.claude/ownership-checks.md` if the
global file doesn't exist (cloud sessions).

Four phases, three user stops.

```
P1 (stakes triage + problem) → freeze gate → user freezes
P2 (tests + doc impact) → freeze gate → user freezes ("freeze" or "freeze deep")
Write specs → Plan (standard or deep + Codex-reviewed) → user approves (HIGH: core-logic summary first)
Execute → tests → commit → break check (HIGH) → Claude critique → Codex audit → present results
```

## Ownership checks (why the extra friction exists)

The failure this guards against: the AI drafts P1 and P2, the user freezes them without
engaging, and the pipeline runs with nobody owning the decisions. **Read
`~/.claude/ownership-checks.md` before P1**: it defines the checks; this skill says where
they happen. Three checks put friction only where it matters:

1. **Stakes triage** (start of P1) — LOW or HIGH. Decides how much of the rest applies.
2. **Freeze gate** (P1 and P2) — the user states the phase in their own words before it
   can freeze. Saved verbatim as the **Owner's statement** in the run spec.
3. **Test break check** (HIGH only, post-commit) — prove the key test can actually fail.

Plus, for HIGH changes, a plain-English core-logic summary at STOP 3. None of these are
optional, and none may be satisfied on the user's behalf.

## When to use this vs `/workflow-review`

`/plan-review` is for surgical, linear changes — one ordered edit list executed by the main
loop. If the work is **parallelizable or large-scale** (a migration, a broad audit/sweep,
"apply X across N files", multi-dimension review), use `/workflow-review` instead: same
P1/P2/spec discipline, but the reviewed deliverable is a multi-agent `Workflow` script.

## Relief valve

If the user says "just fix it", "quick fix", or the change is clearly trivial (one-line,
typo, config), skip the full pipeline. Make a short checklist, code it, test it, commit it.

The relief valve is for changes that would triage **LOW**. If a "quick fix" touches HIGH
territory (see Stakes triage), say so in one line and recommend the full pipeline; the
user may still insist, and then you proceed.

---

## Phase 0 — Resolve project conventions (automatic, no user stop)

This skill is project-agnostic. Before Phase 1, resolve the following from the project
itself and state the resolved values in P1 so the user can correct them. Read `CLAUDE.md`
/ `AGENTS.md` first — a project that documents its own conventions overrides every
fallback below.

| Thing | How to resolve | Fallback |
|-------|----------------|----------|
| **Run spec dir** | Existing dir of per-change design docs | `docs/specs/` (create) |
| **Feature spec dir** | Existing dir of living per-feature docs | `docs/features/` (create) |
| **Plan file dir** | Where durable plan files live | `docs/reports/` (create) |
| **Super docs** | The top-level living docs named in `CLAUDE.md` / a doc map / `README.md` | `README.md` + any `docs/ARCHITECTURE.md` |
| **Bug log** | The project's known-issues file (`BUGS.md`, issue tracker, …) | skip the historical-bug sweep and say so |
| **Full test command** | `CLAUDE.md` → `package.json` `scripts.test` → `Makefile` `test` → `pyproject.toml`/`pytest.ini` → `Cargo.toml` → `go.mod` | none — say so explicitly rather than silently skipping regression |
| **Codex scripts** | `<repo>/.claude/review-plan.sh` and `review-audit.sh` if present | `~/.claude/review-plan.sh`, `~/.claude/review-audit.sh` |

If a project has no doc layers at all, collapse to a single run spec and say so in P2 —
do NOT invent a three-layer doc tree for a project that doesn't have one.

---

## Phase 1 — Problem & Approach

1. Confirm the working branch AND worktree (`git branch --show-current` + `pwd`)
2. **Stakes triage** (ownership-checks §1). Put it on the first line of P1, with a
   one-line reason. Unsure means HIGH; only the user can downgrade. If the P2 inventory
   or plan exploration shows a LOW change reaches HIGH territory, reclassify upward.
3. Discuss the problem with the user:
   - What is the problem?
   - What existing docs are relevant?
   - What is the approach, in plain English?
4. You may read files, explore, research — but do NOT write any files yet
5. Present P1 to the user (stakes line first, then the Phase 0 resolved conventions)

### >>> STOP 1: Freeze gate, then the user freezes P1 <<<

Run the **freeze gate** below. Once the user's sentences pass, **show the complete P1
in-message** — stakes, conventions, problem, approach, decisions carried in, the Owner's
statement verbatim, and the risks fed forward — so the user reads the whole thing as it
will be saved. P1 is frozen only when the user confirms that full view. Then move to
Phase 2. (2026-10-10, user directive: the user must see the entire P1 before it freezes,
not just the gate exchange.)

**P1 rules:** Plain English, no implementation details, no test details.

### Freeze gate (used at STOP 1 and STOP 2)

The Owner's statement gate (ownership-checks §2), applied to a phase:

1. **What this phase does.** P1: the problem and the approach. P2: what the tests will prove.
2. **Most likely way it is wrong** (HIGH only). P1: how the approach could be the wrong
   fix. P2: a failure the tests could miss.

Specific to this pipeline:
- **Feed the risk forward.** The P1 risk becomes a required row in the P2 failure-mode
  inventory. The P2 risk must map to a named test before P2 freezes; add the test if
  it is missing.
- No files exist yet, so carry the sentences in the conversation and write them into
  the run spec in 3a.
- At STOP 2 the user also says "freeze" or "freeze deep". If they give only the
  sentences, ask which mode.

---

## Phase 2 — Test Design & Doc Impact

1. **Failure-mode inventory FIRST** (2026-07-30, user directive — "we write all this
   new code and write five tests for it, and then we find 1 million problems in three
   weeks"). Before listing a single test, think really deeply about what tests are
   actually NEEDED, not the minimal set that demonstrates the happy path:
   - Enumerate, per component/stage the change touches, **all the things that could
     go wrong** — then derive the tests from that inventory. The inventory appears in
     P2 itself (a failure-mode → test table per stage), not just the test names.
   - **The owner's P1 risk** ("most likely way it is wrong", HIGH only) is a required
     row with at least one test.
   - **Map every historical bug class in the touched area to at least one test.**
     Search the project's bug log, run specs, and memory for what has ALREADY bitten in
     this area (decode/orientation bugs, silent stale artifacts, cascading transforms,
     vacuous guards, …) — each prior class gets a pin.
   - **Counterfactual tests for every guard/branch**: at least one test that FAILS if
     the guard is removed (a guard nothing can fail is not tested).
   - **Negative tests for every gate** (wrong artifact, missing asset, degenerate
     input → must refuse loudly, never silently fall back).
   - Where platform twins exist, **share/port the exact vector sets** across
     platforms rather than writing parallel approximations.
   - For anything touching models/hardware: **determinism, artifact-identity
     (tree-sha) gating, and config-drift tripwires** are mandatory test categories.
   - **Scale test count to code size**: a new subsystem warrants a per-stage
     failure-mode table and typically dozens of tests; "5 tests for a new module" is
     an automatic P2 rejection.
2. Then specify:
   - What specific tests validate the change? (exact commands)
   - What are acceptance criteria? (pass/fail definitions)
   - What constitutes a regression?
3. Identify the documentation layers affected (per the Phase 0 resolution):
   - **Run spec** (`{run-spec-dir}/{CHANGE_NAME}.md`): Always created. The design doc for THIS change. Frozen after commit — never modify a previous run spec.
   - **Feature master spec** (`{feature-spec-dir}/{FEATURE_NAME}.md`): The living truth for the feature being changed. Created if it doesn't exist, updated if it does.
   - **Super master docs**: Updated if the change affects flow, architecture, or shipped status.
4. For HIGH changes, name the **break-check candidate**: the one test most tied to the
   core logic, which the post-commit test break check will mutate against.
5. Present P2 to the user — include which docs will be created/updated

### >>> STOP 2: Freeze gate, then the user freezes P2 <<<

Run the **freeze gate** (see Phase 1). Once the sentences pass, **show the complete P2
in-message** — the failure-mode table, every test with its command, acceptance criteria,
the doc list, the break-check candidate, and the Owner's statement verbatim — so the
user reads it whole before freezing. P2 is frozen only when the user confirms that full
view and has said "freeze" or "freeze deep". Then move to Phase 3.

- **"freeze"** — standard planning (single-agent exploration → plan)
- **"freeze deep"** — deep planning (parallel multi-agent exploration + critique agent → plan)

**P2 rules:** Be specific — exact test commands, expected output. For docs, list ONLY the files you commit to editing (the post-commit audit grades Documentation against this list, so an unfulfilled entry is an automatic finding). If unsure a doc is affected, mark it "conditional — confirm at execution" rather than promising it.

---

## Phase 3 — Specs, Plan & Review

After P2 freeze, do ALL of the following without stopping:

### 3a. Write specs (automatic — no user stop)

1. Create the **run spec** (`{run-spec-dir}/{CHANGE_NAME}.md`). It holds the frozen P1
   and P2 **as shown to the user at STOP 1 and STOP 2** (problem, approach, failure-mode
   table, tests, doc list), the Owner's statements, and the technical spec of what will
   change. It must contain:
   - `**Stakes:** LOW | HIGH — <reason>` (plus any user override and why)
   - `## Owner's statement` with `### P1` and `### P2` subsections holding the user's
     freeze-gate sentences **verbatim**, labelled "What it does" and "Most likely way it
     is wrong" (the second omitted for LOW)
   - For HIGH: `## Test break check` with the chosen test, to be filled in post-commit
2. Create or update the **feature spec** (`{feature-spec-dir}/{FEATURE_NAME}.md`)
3. Update **super docs** if affected

### 3b. Write implementation plan

#### Standard mode (user said "freeze")

1. Explore the codebase — read files that will be changed, check current state (line numbers, current code)
2. Write the plan to a durable file `{plan-dir}/YYYY-MM-DD_{slug}-plan.md`:
   - Step-by-step execution order
   - Grounded in actual file contents (specific line numbers, current code state)
   - Files to create/modify
   - Test commands to run
   - Verification steps
   (A durable file — NOT native plan mode — because Codex review needs a file path and the run spec is the persistent record. Present the plan in-message at STOP 3 and wait for explicit approval.)

#### Deep mode (user said "freeze deep")

1. Spawn 3 parallel exploration agents (using the Agent tool) to investigate simultaneously:
   - **Architecture agent** — understand the relevant existing code, patterns, and reusable functions
   - **File impact agent** — find all files that will need modification, with current line numbers and code state
   - **Risk agent** — identify potential risks, edge cases, dependencies, and regression vectors
2. Synthesize all three agents' findings into a unified implementation plan
3. Spawn a **critique agent** to review the synthesized plan for missing steps, risks, and mitigations
4. Incorporate critique feedback, then write the final plan to a durable file `{plan-dir}/YYYY-MM-DD_{slug}-plan.md`:
   - Step-by-step execution order
   - Grounded in actual file contents (specific line numbers, current code state)
   - Files to create/modify
   - Test commands to run
   - Verification steps
   - Risks and mitigations (from risk + critique agents)

### 3c. Codex plan review

1. Run Codex plan review: `bash {codex-review-script} "<plan-file-path>"`
2. Read review, address findings in the plan

### >>> STOP 3: Present plan + Codex review to user <<<

3. **HIGH changes: lead with a core-logic summary** (ownership-checks §4), before the
   plan. If it doesn't match the P1 Owner's statement, say so here.
4. Present EVERY step of the plan to the user — not a summary, the complete plan
5. Present the Codex review findings and how they were addressed
6. **STOP. Do NOT proceed to execution.** Wait for explicit approval ("go ahead", "implement", "do it")

**Plan rules:** The plan describes HOW and IN WHAT ORDER, with specific line numbers and code patterns. Codex reviews for implementation gaps, missing edge cases, stale file paths. Codex review runs ONLY on plans — never on specs or feature docs. The plan lives as a durable file in the plan dir; the **run spec is the persistent audit record** of what was planned. The Codex audit checks adherence against the run spec, not the plan file.

---

## Phase 4 — Execute

### Pre-flight

1. `git branch --show-current` + `pwd` — hard stop if wrong branch or wrong worktree
2. Scope check: `git status` — stash/commit unrelated changes first

### Execute (no stopping until done)

- Code every planned step
- Run all P2 tests
- Run the full regression suite (the command resolved in Phase 0)
- Update any docs that need final data (e.g., test results in run spec)
- **Doc-sync sweep (before commit).** Do the sync NOW so the post-commit audit *verifies* rather than *discovers* it (this is what turns a 3-round audit into a 1-round pass):
  1. **Reconcile** the P2 doc list against the actual diff — edit each listed doc, or strike it from the run spec with a one-line reason (deferred / not-needed). Leave no unfulfilled promise.
  2. **Staleness grep** — for every flag/field/function/threshold/behavior you changed, AND every "TODO / not-yet / blocked / pending / defaults-to-None" this change *resolves*, grep the feature specs, the super docs, and the run specs for now-contradicted claims. Fix living docs; add a dated note to frozen run specs (don't rewrite them — see freeze rule).
  3. For cross-cutting changes, launch a doc-sync agent (this project's `doc-sync-checker` if it has one) to do 1–2 automatically.
  - The frozen-spec forward-reference (a doc that said "TODO X" when X is now done) is the most common miss — grep for the thing you just shipped.
- Commit with conventional commit prefix

### Post-commit (REQUIRED — do not skip)

**Claude critique FIRST (loop until it passes), Codex audit LAST (one confirmation pass).**
Codex is rate-limited, so spend the cheap Claude pass on the obvious/avoidable failures and
reserve Codex for the final grade. (This ordering came from a real run where Codex burned 2
quota-limited rounds on things a Claude critique caught for free.)

0. **Test break check — HIGH changes only, once per run, after tests pass.** Follow
   ownership-checks §3 exactly. Use the break-check candidate named in P2; if the
   implementation made a different test the better fit, switch and say why in one line.
   A decorative test is a finding: fix, commit, and re-check before the critique starts.
   Record the result in the run spec's `## Test break check`.

1. **Claude pre-audit critique — loop until Acceptable.** Launch a Claude critique agent
   (`general-purpose`) that grades the change the way the Codex audit does — and instruct it
   to be **maximally adversarial**: assume the implementer cut corners, and **bias toward
   Fail / Needs-work whenever a claim is not verifiable in the code.** (A soft critique that
   passes work Codex then fails wastes a rate-limited Codex round — the whole point of running
   this first.)
   - It reads the **AUDIT rubric** (the `codex-audit-prompt.md` that `review-audit.sh`
     resolves — project `.claude/` first, else `~/.claude/`; the SAME rubric it feeds Codex,
     NOT the lighter plan-oriented `codex-review-prompt.md`), the run spec, the plan +
     `*-review.md`, and the diff (`git diff <base>...HEAD`).
   - It emits the 8-dimension scorecard — Plan adherence, Scope discipline, Test coverage,
     Review compliance, Freeze integrity, Regression check, Documentation, Test sensitivity,
     Overall (= worst) — and must VERIFY each finding in the code (`file:line`), not take
     the docs' word. Require specifically:
     - **Test coverage** — enumerate EVERY acceptance criterion in the run spec and confirm
       each maps to a test that actually exercises it; an untested criterion is a Fail.
     - **Plan adherence** — verify each design decision (D1, D2, …) is implemented AS STATED,
       including its **edge / fallback path** (fallback engines, nil/empty inputs, boundary
       values), not just the happy path. Actively try to break each new / changed function.
     - **Freeze integrity** — the run spec has a Stakes line and an `## Owner's statement`
       with P1 and P2 entries. This pipeline always triages, so a missing classification
       or statement is a Fail here (not N/A as the generic rubric allows).
     - **Test sensitivity** — for HIGH, the break-check record exists, the mutation was
       realistic and in the core logic, the test is the one most tied to it, and the
       mutation appears in no commit (`git log -p <base>..HEAD`).
   - **Fix every finding** (code/docs/tests), commit, and **re-run the critique, looping
     until Overall is Acceptable** (or only non-blocking risks remain). Continue the SAME
     agent via `SendMessage` so it re-checks its own prior findings.
   - **Stop the loop when the deliverable stops moving** — if a round produces findings but
     changes nothing material, the loop is done; record the open items rather than spinning.
2. **Codex audit (final confirmation):** `bash {codex-audit-script} <run-spec-path>`
3. If the audit fails (network error, empty output): retry once. If it fails again,
   **hard stop** and tell the user — do NOT silently skip.
   - **On a Codex quota / usage-limit failure specifically:** the passing Claude critique
     (step 1) stands in as the gate, so do NOT block the change. Instead **auto-schedule the
     confirmation pass**: parse the reset time from the Codex error (`"try again at <date>
     <time>"`, the user's LOCAL time) and call `mcp__scheduled-tasks__create_scheduled_task`
     with a one-time `fireAt` ≈ 5 min after that reset. The scheduled prompt must be
     self-contained: `cd` to this worktree, `source ~/.zshrc`, run
     `bash {codex-audit-script} <run-spec-path>`, then on Acceptable record the
     scorecard in the run spec + commit, or on findings auto-fix/commit/re-audit — and NEVER
     push/merge. Record in the run spec that the Codex pass is **pending (scheduled for
     <reset time>)**, and tell the user it's queued.
4. Read the audit results. Update the run spec with the scores — BOTH the Claude critique
   verdict and the Codex grade.
5. **Automatically fix all audit findings.** Do not ask — just fix them (code, docs, tests).
   Commit the fixes, then re-run the audit to confirm resolution.
6. Present final results to user (Claude critique outcome + Codex scorecard).

### Present results

- Test results (pass/fail counts)
- Break check result line (HIGH), including any fix-and-recheck rounds
- Codex audit/scorecard results (REQUIRED — if missing, explain why)
- Audit findings fixed (list what was found and how it was resolved)
- Any deviations from plan and why
- Issues encountered

**Execution rules:**
- Go step by step. Do not skip or reorder without documenting why.
- Do NOT stop to ask questions after "go ahead". Only exception: scope-level blocker.

---

## Documentation model

Every plan-review run touches up to three documentation layers:

| Layer | Location | Lifecycle | When |
|-------|----------|-----------|------|
| **Run spec** | `{run-spec-dir}/{CHANGE}.md` | One per change, **frozen after commit** | Always created |
| **Feature master spec** | `{feature-spec-dir}/{FEATURE}.md` | Living document, updated over time | Created if new, updated if exists |
| **Super master docs** | The project's top-level living docs | Top-level overviews | Updated if affected |

Run specs are frozen after commit — never modify a previous run spec. Feature specs are the living truth and get updated with each change. Changes propagate upward: run spec → feature spec → super docs.

---

## User interactions summary

| Phase | What happens | User says |
|-------|-------------|-----------|
| P1 | Stakes triage + problem discussed (+ resolved conventions) | Owner's statement (what it does; + most likely way it's wrong if HIGH), then frozen |
| P2 | Tests + doc impact discussed | Owner's statement, plus "freeze" or "freeze deep" |
| Specs + Plan | Specs written; HIGH: core-logic summary; plan + Codex review presented | "go ahead" |
| Execute | Results + break check (HIGH) + audit presented | direction |

Three stops. Everything between stops is automatic.

---

## Rules

- Plans are written to a durable markdown file in the plan dir (Codex review requires a file path; native plan mode is not used). Deep mode adds parallel exploration agents + critique before writing the plan.
- Codex plan review: `bash {codex-review-script} <plan-file-path>` — **only on plans, never on specs**
- Codex audit: `bash {codex-audit-script} <run-spec-path>`
- **Prefer the project's own `.claude/review-*.sh` and `codex-*-prompt.md` when they exist** — they encode project-specific review context. The `~/.claude/` copies are the generic fallback.
- **The run spec is the audit artifact** — the Codex audit checks plan adherence and review compliance against the run spec. The plan file persists alongside it.
- **Run specs are frozen after commit** — never modify a previous run spec
- **NEVER skip the plan presentation stop.** The user MUST see and approve the full plan before execution.
- **Stakes triage is never skipped; unsure means HIGH.** Only the user can downgrade.
- **A bare "freeze" / "looks good" never passes a freeze gate.** The Owner's statement is the user's own words, saved verbatim, never written for them.
- **The break-check mutation never leaves its scratch worktree.** Delete the worktree afterwards.
- **Codex post-commit audit is REQUIRED** — never skip it. If it fails after retry, hard stop and report.
- **Audit findings are automatically fixed** — do not present findings and wait. Fix them immediately (code, docs, tests), commit, re-audit, then present the final results.
- **NEVER push without user approval.**

<!-- GLOBAL-SKILL:END -->

## HEARSAY additions

This project's additions to the shared skill above, moved here verbatim from the old copy. Each says which part of the skill it belongs to.

**Applies to: everything below — read before Phase 0**

## HEARSAY localization (project-specific — read before Phase 0)

- **When to use it here:** any rung or change expected to take more than an hour
  (CLAUDE.md working rule). Under an hour → relief valve.
- **Budget:** the build ends 8 AM Sunday (Sep 27); see the timeline and hard cutoffs in
  `docs/plan.md`. Every P1 names the rung (M0, M1, D-track, M3, M5, M4, K, DOC), a **time
  box** (a few hours max per rung), and the stop point. When the box runs out, stop and
  report — do not push through.
- **Always have a valid TSV:** from M1 onward, no plan may leave the repo unable to emit a
  valid `teamName_predictions.tsv`. Every execution that touches the scoring pipeline ends
  with a fresh valid TSV **and** a new row in `submissions/log.csv` (timestamp, rung,
  outer-holdout minDCF, submission path), or names the blocker. Never overwrite a
  previously submitted TSV.
- **Freeze on Sunday morning (Sep 27):** no new plan-review runs start that cannot land
  before the freeze. After the freeze, only TSV- or Docker-breaking fixes, via the relief
  valve.
- **Detector contract:** a change to `DetectorResult` / `ClipContext` / `safe_run`
  (`src/hearsay/detectors/base.py`) must say so in P1 and name the detector owners it
  affects (CLAUDE.md team table). Contract fields do not change silently.
- **Model rules (every rung):** validate by generator and speaker via the fold file, not
  by clip; never train on the validation split; log every TSV with its validation score;
  keep a clean-only validation score alongside any augmented one.
- **Sources of truth:** `docs/nsa-challenge.md` + the NSA instructions (rules, scoring) and
  `docs/plan.md` (plan and scope) are the super docs; `docs/scoping.md` and
  `docs/master-doc.md` are historical.
- **Cloud GPU:** ask before renting one (M5 is GPU-only; cap about $75).

**Applies to: Phase 0 — Resolve project conventions (automatic, no user stop)**

HEARSAY: this repo ships `.claude/review-plan.sh`, `.claude/review-audit.sh`, and HEARSAY-
localized rubrics (`.claude/codex-review-prompt.md`, `.claude/codex-audit-prompt.md`), so
the project copies are the ones to use. They need the Codex CLI installed and logged in
(`codex --version`). If a teammate has no Codex CLI, say so at P1; the Claude critique loop
(Phase 4, step 1) stands in as the gate and the plan is presented without a Codex review.
Plan files go in `docs/reports/`, run specs in `docs/specs/`; this project has no feature-
spec layer — collapse to run specs plus the super docs.

**Applies to: Phase 2 — Test Design & Doc Impact**

   - **HEARSAY standing failure modes** — every P2 in this project must address each of
     these (a test, a measurement, or an explicit one-line waiver saying why it doesn't
     apply to this change):
     1. **Laundering, telephony and replay in the test set** — the brief lists transcoded,
        band-limited, noised and replayed fakes; report a clean-only score beside any
        augmented one and measure on an augmented validation slice. Physical replay is not
        covered (no hardware; simulated RIR only) — say so rather than claim it.
     2. **Overfitting to known generators or speakers** — validation must hold out whole
        generator and speaker groups via the fold file (grouped, nested); a gain that only
        appears on seen groups is not a gain.
     3. **Class imbalance and prior** — about 70% of test files are real (π_synth = 0.3);
        check class counts per fold, class-balanced training, and whether a decision flips
        between the metric readings in `docs/plan.md` (including the sponsor-code reading).
     4. **Shortcuts** — duration, peak level, silence, container fields, the single LJ
        speaker; check per-feature AUC against the label and compare clean vs transcoded
        before trusting a gain.
     5. **Format conversion errors in the loader** — sample rate, channel count,
        int/float scaling, codec decoder padding, clipping. Everything must arrive as
        16 kHz mono; pin it with fixture clips in each input format we expect (incl. MP4).
