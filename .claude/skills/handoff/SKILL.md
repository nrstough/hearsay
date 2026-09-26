---
name: handoff
description: "Generate a self-contained handoff prompt to start a NEW chat that continues this project's work. Inspects current state (branch, worktree, uncommitted/at-risk artifacts, recent commits/results) and writes a ready-to-paste prompt covering the new chat's purpose, context, next steps, and — critically — the important tests + where run data lives (run data gets LOST when a job dir is cleaned). Use whenever the user says 'write a handoff', 'hand this off', 'prompt for a new chat', 'continue this in a new session', 'wrap up for next time', or is ending a work session and wants the next chat to pick up cleanly with nothing lost."
user_invocable: true
---

# /handoff — New-chat handoff prompt

Produce a **ready-to-paste prompt for a fresh chat** that continues the current work with zero context loss.

Why this matters: run data, uncommitted outputs, and test state are the things that actually get lost between sessions — a job-dir cleanup deletes mid-run results silently, and nobody notices until the next chat can't reproduce a number. So a good handoff isn't just "what's next" — its real job is making the next chat able to **find, re-run, and preserve everything that matters.** Treat the tests/at-risk section as the part you must not skimp on.

## Step 0 — Resolve where handoffs live (automatic)

| Thing | How to resolve | Fallback |
|-------|----------------|----------|
| **Handoff dir** | An existing dir of handoff docs (`docs/handoffs/`, `docs/sessions/`, …), or the location named in `CLAUDE.md` / `AGENTS.md` | `docs/handoffs/` (create) |
| **Test command** | `CLAUDE.md` / `AGENTS.md` → `package.json` `scripts.test` → `Makefile` `test` → `pyproject.toml`/`pytest.ini` → `Cargo.toml` → `go.mod` | say explicitly that there is no test command |
| **Archive location** | Where large/regenerable artifacts are kept (external drive, object storage, a data dir named in the project docs) | note "NOT archived" against each at-risk path |

If the handoff dir already holds handoffs, read the newest one and match its shape.

**HEARSAY:** handoffs live in `docs/handoffs/`. There is no archive location for large
artifacts yet — mark `data/`, `weights/`, `outputs/`, and any submission CSV "NOT archived"
unless someone has copied them off the laptop. Note: `submissions/*.csv` is **gitignored**
(only `submissions/log.csv` is tracked), so submitted CSVs are always at-risk artifacts.

## Step 1 — Inspect state (read it; don't guess)
- **Branch + worktree:** `git branch --show-current`, and note which worktree you're in (`pwd`; `git worktree list` if the project uses them).
- **Uncommitted / at-risk:** `git status --short` — anything uncommitted is at risk.
- **Recent commits:** `git log --oneline -8`.
- **At-risk artifacts:** run outputs, result JSONs, datasets, models living in job-tmp (`$CLAUDE_JOB_DIR/tmp`, `~/.claude/jobs/*/tmp`), a scratchpad dir, or anywhere regenerable-but-not-committed. Record their **paths** and whether they're **archived**. These are exactly what gets lost — a job-dir cleanup deletes them silently.
- **In-flight work:** background jobs, cloud/remote runs, scheduled tasks, open PRs — anything that will still be running after this chat ends, plus how the next chat checks on it.
- **HEARSAY status:** the current model-ladder rung (M0–M6), the last few rows of
  `submissions/log.csv`, the path of the newest **valid** CSV, the best validation score
  (clean-only and replay-augmented, by held-out generator), and hours left before the
  Sunday-morning freeze. Flag any pending data-contract change and which teammate it
  affects.

## Step 2 — Capture the tests (non-negotiable)
For the work in flight, list the tests/validation that matter:
- The **exact** command(s), copy-pasteable.
- The **last-known / expected result** (the number to reproduce).
- Any test or run artifact at risk + where it lives + archive status.

If there genuinely are no tests yet, say so explicitly so the next chat doesn't assume coverage that isn't there.

## Step 3 — Write the handoff prompt
Use this structure (adapt content, keep the sections):

```
# Handoff — <slug> (<date>)

**Purpose of this chat:** <one line — what the new chat should achieve / convey>

## Context
<what's been done, current state — tight>

## Working branch / worktree
<branch> in <worktree>; <clean | N uncommitted>

## Environment / setup
<the exact commands to get a working session: env activation, bootstrap script, env vars>

## What to do next
<concrete, ordered next steps>

## IMPORTANT — tests & at-risk artifacts (make sure these survive)
- Test: `<exact command>` → expected <result>
- At-risk: <path> (<archived where? / NOT archived>)
- In flight: <job/run> — check with `<command>`
<repeat as needed>

## Analytical notes
<key metrics, recent findings, gotchas, where data/models live>

## HEARSAY status
Rung: <M?> · Latest valid CSV: <path> · Val (held-out generators): clean <x> / replay <y>
Time left to freeze: <h> · Contract changes pending: <none | field → owner>

## Pointers
<relevant records / specs / memory files to read first>
```

## Step 4 — Save + present
- Save to `{handoff-dir}/<YYYY-MM-DD>_<slug>-handoff.md`.
- Present the full prompt text in-chat for copy-paste.
- **Reuse the current worktree/branch** — do NOT spin up a new branch/worktree for a handoff, unless the project's own rules say otherwise. If the handoff records anything doc-worthy beyond itself, follow the project's doc-sync rule.
