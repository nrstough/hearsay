---
name: zoom-out
description: Tell the agent to zoom out and give broader context or a higher-level perspective. Use when you're unfamiliar with a section of code or need to understand how it fits into the bigger picture.
disable-model-invocation: true
---

I don't know this area of code well. Go up a layer of abstraction. Give me a map of all the relevant modules and callers, using whatever domain vocabulary the codebase already uses.

In HEARSAY, organize the map around the pipeline stages in CLAUDE.md (ingest/loader → orchestrator → detectors → fusion → TSV writer / explanation report), the owner of each detector, and the ladder rung it is on.
