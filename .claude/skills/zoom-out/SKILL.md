---
name: zoom-out
description: Tell the agent to zoom out and give broader context or a higher-level perspective. Use when you're unfamiliar with a section of code or need to understand how it fits into the bigger picture.
disable-model-invocation: true
---

I don't know this area of code well. Go up a layer of abstraction. Give me a map of all the relevant modules and callers, using whatever domain vocabulary the codebase already uses.

In HEARSAY, organize the map around the data-contract stages in CLAUDE.md (audio front end → detector → tracker → head driver / dashboard), the owner of each, and the model-ladder rung the detector is on.
