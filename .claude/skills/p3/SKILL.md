---
name: p3
description: "Execute an approved plan — code, test, commit"
user_invocable: true
---
<!-- GLOBAL-SKILL:BEGIN (synced from ~/.claude/skills, do not edit inside) -->

# /p3 — Execute

**Repo additions:** if the current repo has `.claude/skills/p3/SKILL.md` containing
`GLOBAL-SKILL:BEGIN`/`END` markers, also follow everything in that file *outside* the
markers. It holds this project's additions to this skill and wins where they conflict.
(If you are already reading that file, there is nothing extra to load.) A repo copy
without markers is ignored. **Path rule:** wherever this skill names
`~/.claude/ownership-checks.md`, use the repo's `.claude/ownership-checks.md` if the
global file doesn't exist (cloud sessions).

Execute an already-approved implementation plan. The plan should exist as a durable
markdown file (this project's plan directory — commonly `.claude/plans/` or `docs/reports/`)
and have been reviewed by the user.

## Pre-flight

1. `git branch --show-current` + `pwd` — verify correct branch AND worktree
2. Read the plan file
3. Scope check: `git status` — stash/commit unrelated changes first
4. **Resolve the project's full test command** — do not assume. In order: an explicit
   command in the plan → `CLAUDE.md` / `AGENTS.md` → project config (`package.json`
   `scripts.test`, `Makefile` `test` target, `pyproject.toml` / `pytest.ini`,
   `Cargo.toml`, `go.mod`). If no test command exists, say so explicitly in the results
   rather than silently skipping regression.
5. **Ownership checks** (read `~/.claude/ownership-checks.md`). Find the plan's run spec
   and read its `**Stakes:**` line and `## Owner's statement`.
   - No Stakes line: triage now (§1) and add it to the plan file.
   - HIGH with no Owner's statement (the plan was approved outside `/plan-review`):
     give the plain-English summary (§4), then run the Owner's statement gate (§2) on the
     plan before starting. Record the sentences verbatim in the plan file under
     `## Owner's statement`.
   This all happens before execution starts. The no-questions rule applies after it.

## Execute

Complete ALL planned steps without stopping:
- Code every step in the plan
- Run all tests specified in the plan
- Run the full regression suite (the command resolved in pre-flight)
- Update affected documentation (doc-sync)
- Commit with conventional commit prefix
- **HIGH: test break check** (ownership-checks §3) after the commit. A decorative test is
  fixed, committed, and re-checked before you report results.

## Present results

- Test results (pass/fail counts)
- Break check result line (HIGH)
- Any deviations from plan and why
- Issues encountered

## Rules

- Go step by step. Do not skip or reorder without documenting why.
- Do NOT stop to ask questions after starting. Only exception: scope-level blocker.
- If tests fail, fix and re-run before committing. Document the fix as a deviation.

<!-- GLOBAL-SKILL:END -->

## HEARSAY additions

This project's additions to the shared skill above, moved here verbatim from the old copy. Each says which part of the skill it belongs to.

**Applies to: Rules (project rules for every execution)**

## HEARSAY rules (project-specific)

- If the change touches a detector or the submission path, finish with a fresh **valid
  TSV** and a new row in `submissions/log.csv` (timestamp, rung, validation score,
  submission path), or name the blocker. Never overwrite a previously submitted TSV; write
  a new file.
- Report validation by held-out generator and speaker groups (fold file), with the
  clean-only score next to any augmented one. Never train on the validation split.
- Respect the plan's time box (rungs are capped at a few hours; the build ends 8 AM
  Sunday). When it runs out, stop and report the state — do not push through.
- After the Sunday-morning freeze, execute only TSV- or Docker-breaking fixes.
- Do not change the detector contract (`src/hearsay/detectors/base.py`) unless the plan
  says so and names the affected detector owners.
