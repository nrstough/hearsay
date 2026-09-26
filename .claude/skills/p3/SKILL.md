---
name: p3
description: "Execute an approved plan — code, test, commit"
user_invocable: true
---

# /p3 — Execute

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

## Execute

Complete ALL planned steps without stopping:
- Code every step in the plan
- Run all tests specified in the plan
- Run the full regression suite (the command resolved in pre-flight)
- Update affected documentation (doc-sync)
- Commit with conventional commit prefix

## Present results

- Test results (pass/fail counts)
- Any deviations from plan and why
- Issues encountered

## Rules

- Go step by step. Do not skip or reorder without documenting why.
- Do NOT stop to ask questions after starting. Only exception: scope-level blocker.
- If tests fail, fix and re-run before committing. Document the fix as a deviation.

## HEARSAY rules (project-specific)

- If the change touches the detector or CSV path, finish with a fresh **valid CSV** and a
  new row in `submissions/log.csv` (timestamp, rung, validation score, CSV path). Never
  overwrite a previously submitted CSV; write a new file.
- Report validation by generator (held-out generators), with the clean-only score next
  to any replay-augmented one. Never train on the validation split.
- Respect the plan's time box (rungs are capped at a few hours inside a 24–30 hour
  budget). When it runs out, stop and report the state — do not push through.
- After the Sunday-morning freeze, execute only demo/CSV-breaking fixes.
- Do not change a data-contract field (see CLAUDE.md) unless the plan says so and names
  the teammate on the other side.
