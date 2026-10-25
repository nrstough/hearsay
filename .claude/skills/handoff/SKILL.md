---
name: handoff
description: "Generate a self-contained handoff prompt to start a NEW chat that continues this project's work. Inspects current state (branch, worktree, uncommitted/at-risk artifacts, recent commits/results) and writes a ready-to-paste prompt covering the new chat's purpose, context, next steps, and — critically — the important tests + where run data lives (run data gets LOST when a job dir is cleaned). Use whenever the user says 'write a handoff', 'hand this off', 'prompt for a new chat', 'continue this in a new session', 'wrap up for next time', or is ending a work session and wants the next chat to pick up cleanly with nothing lost."
user_invocable: true
---
<!-- GLOBAL-SKILL:BEGIN (synced from ~/.claude/skills, do not edit inside) -->

# /handoff — New-chat handoff prompt

**Repo additions:** if the current repo has `.claude/skills/handoff/SKILL.md` containing
`GLOBAL-SKILL:BEGIN`/`END` markers, also follow everything in that file *outside* the
markers. It holds this project's additions to this skill and wins where they conflict.
(If you are already reading that file, there is nothing extra to load.) A repo copy
without markers is ignored. **Path rule:** wherever this skill names
`~/.claude/ownership-checks.md`, use the repo's `.claude/ownership-checks.md` if the
global file doesn't exist (cloud sessions).

Produce a **ready-to-paste prompt for a fresh chat** that continues the current work with zero context loss.

Why this matters: run data, uncommitted outputs, and test state are the things that actually get lost between sessions — a job-dir cleanup deletes mid-run results silently, and nobody notices until the next chat can't reproduce a number. So a good handoff isn't just "what's next" — its real job is making the next chat able to **find, re-run, and preserve everything that matters.** Treat the tests/at-risk section as the part you must not skimp on.

## Step 0 — Resolve where handoffs live (automatic)

| Thing | How to resolve | Fallback |
|-------|----------------|----------|
| **Handoff dir** | An existing dir of handoff docs (`docs/handoffs/`, `docs/sessions/`, …), or the location named in `CLAUDE.md` / `AGENTS.md` | `docs/handoffs/` (create) |
| **Test command** | `CLAUDE.md` / `AGENTS.md` → `package.json` `scripts.test` → `Makefile` `test` → `pyproject.toml`/`pytest.ini` → `Cargo.toml` → `go.mod` | say explicitly that there is no test command |
| **Archive location** | Where large/regenerable artifacts are kept (external drive, object storage, a data dir named in the project docs) | note "NOT archived" against each at-risk path |

If the handoff dir already holds handoffs, read the newest one and match its shape.

## Step 1 — Inspect state (read it; don't guess)
- **Branch + worktree:** `git branch --show-current`, and note which worktree you're in (`pwd`; `git worktree list` if the project uses them).
- **Uncommitted / at-risk:** `git status --short` — anything uncommitted is at risk.
- **Recent commits:** `git log --oneline -8`.
- **At-risk artifacts:** run outputs, result JSONs, datasets, models living in job-tmp (`$CLAUDE_JOB_DIR/tmp`, `~/.claude/jobs/*/tmp`), a scratchpad dir, or anywhere regenerable-but-not-committed. Record their **paths** and whether they're **archived**. These are exactly what gets lost — a job-dir cleanup deletes them silently.
- **In-flight work:** background jobs, cloud/remote runs, scheduled tasks, open PRs — anything that will still be running after this chat ends, plus how the next chat checks on it.
- **Pipeline state (if mid `/plan-review`, `/workflow-review`, or any skill using `~/.claude/ownership-checks.md`, including a `/consult` whose response is still `UNREVIEWED`):** the phase reached, the **stakes classification** (LOW/HIGH, reason, any user override), every **Owner's statement** sentence given so far, copied **verbatim**, and, for HIGH, whether the test break check has run and its result. Before 3a writes the run spec, the Owner's statement exists only in this conversation, so the handoff is its only copy.

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

## Pipeline state (only if mid a skill with ownership checks)
- Phase: <P1 drafted | P1 frozen | P2 frozen | plan approved | executing | auditing>
- Stakes: <LOW | HIGH> — <reason> <(user override: …)>
- Owner's statement (verbatim, do not edit):
  - P1 what it does: "<…>"  /  most likely wrong: "<…>"
  - P2 what it does: "<…>"  /  most likely wrong: "<…>"
- Break check (HIGH): <not yet run | CAUGHT | NOT CAUGHT → fix pending>
A phase counts as frozen ONLY if its Owner's statement is recorded above. If it is missing, run that phase's freeze gate again; do not take "frozen" on this handoff's word.

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

## Pointers
<relevant records / specs / memory files to read first>
```

## Step 4 — Save + present
- Save to `{handoff-dir}/<YYYY-MM-DD>_<slug>-handoff.md`.
- Present the full prompt text in-chat for copy-paste.
- **Reuse the current worktree/branch** — do NOT spin up a new branch/worktree for a handoff, unless the project's own rules say otherwise. If the handoff records anything doc-worthy beyond itself, follow the project's doc-sync rule.

<!-- GLOBAL-SKILL:END -->

## HEARSAY additions

This project's additions to the shared skill above, moved here verbatim from the old copy. Each says which part of the skill it belongs to.

**Applies to: Step 0 — Resolve where handoffs live (automatic)**

**HEARSAY:** handoffs live in `docs/handoffs/`. There is no archive location for large
artifacts except the external drive (`/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/`) —
mark `data/`, `weights/`, `models/`, `outputs/`, and any submission TSV "NOT archived"
unless someone has copied them there. Note: `submissions/*` is **gitignored** (only
`submissions/log.csv` is tracked), so submitted TSVs are always at-risk artifacts.

**Applies to: Step 1 — Inspect state (read it; don't guess)**

- **HEARSAY status:** the current ladder rung (M0, M1, D-track, M3, M5, M4, K, DOC), the
  last few rows of `submissions/log.csv`, the path of the newest **valid** TSV, the best
  validation minDCF (clean-only and augmented, on the outer holdout), and hours left
  before 8 AM Sunday. Flag any pending detector-contract change and which owners it
  affects.

**Applies to: Step 3 — the handoff template (add this section after "Analytical notes")**

## HEARSAY status
Rung: <M?> · Latest valid TSV: <path> · Val minDCF (outer holdout): clean <x> / augmented <y>
Time left to 8 AM Sun: <h> · Detector-contract changes pending: <none | field → owners>
