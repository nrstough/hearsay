---
name: zoom-out
description: Tell the agent to zoom out and give broader context or a higher-level perspective. Use when you're unfamiliar with a section of code or need to understand how it fits into the bigger picture.
disable-model-invocation: true
---
<!-- GLOBAL-SKILL:BEGIN (synced from ~/.claude/skills, do not edit inside) -->

**Repo additions:** if the current repo has `.claude/skills/zoom-out/SKILL.md` containing
`GLOBAL-SKILL:BEGIN`/`END` markers, also follow everything in that file *outside* the
markers. It holds this project's additions to this skill and wins where they conflict.
(If you are already reading that file, there is nothing extra to load.) A repo copy
without markers is ignored. **Path rule:** wherever this skill names
`~/.claude/ownership-checks.md`, use the repo's `.claude/ownership-checks.md` if the
global file doesn't exist (cloud sessions).

I don't know this area of code well. Go up a layer of abstraction. Give me a map of all the relevant modules and callers, using whatever domain vocabulary the codebase already uses.

<!-- GLOBAL-SKILL:END -->

## HEARSAY additions

This project's additions to the shared skill above, moved here verbatim from the old copy. Each says which part of the skill it belongs to.

**Applies to: the map you produce**

In HEARSAY, organize the map around the pipeline stages in CLAUDE.md (ingest/loader → orchestrator → detectors → fusion → TSV writer / explanation report), the owner of each detector, and the ladder rung it is on.
