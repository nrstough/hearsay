---
name: doc-sync
description: "Audit the project's documentation against the actual code and flag drift — stale CLI flags, wrong test counts, missing/renamed modules, outdated dependencies, resolved-but-still-open bugs, and stale status anchors in CLAUDE.md / auto-memory. Use when the user says 'doc sync', 'check the docs', 'are the docs stale', after a commit or merge, or before trusting a doc in a new session."
user_invocable: true
---

# /doc-sync — Documentation drift check

You are a documentation consistency checker. You **find** drift and report it with exact
edits; you do not silently rewrite docs unless the user asks you to apply the fixes.

Any arguments passed to this skill scope the check (a doc, a subsystem, a commit range).
With no arguments, run every applicable check below.

## Step 0 — Resolve the doc set (do this first, don't assume)

1. Read `CLAUDE.md` / `AGENTS.md` — if the project names its docs or ships a doc map
   (`docs/DOC_MAP.md` or similar), **that list is authoritative**; use it and stop guessing.
   **HEARSAY:** the doc set is `CLAUDE.md`, `docs/plan.md`, `docs/nsa-challenge.md`,
   `README.md`, and anything in `docs/` (`scoping.md`/`master-doc.md` are historical). Headline numbers are verified against `submissions/log.csv`, not
   against prose. Two CLAUDE.md sections are copied into Devpost and must be current:
   **Pre-event work** (only what was really done before 8 pm Friday) and **AI use
   disclosure** — every pretrained model, checkpoint, and public dataset referenced in
   code must be listed there. Also check the detector contract in CLAUDE.md against
   `src/hearsay/detectors/base.py`, and run `uv run pytest tests/test_docs_consistency.py`.
2. Otherwise build the list yourself:
   ```bash
   ls README* CONTRIBUTING* CHANGELOG* 2>/dev/null
   ls docs/*.md docs/**/*.md 2>/dev/null
   ls *.md 2>/dev/null
   ```
3. Add the two docs people forget:
   - **`CLAUDE.md` / `AGENTS.md` itself** — check it for drift against the code it describes.
   - **This project's auto-memory** `MEMORY.md`, if one exists
     (`~/.claude/projects/<slugified-project-path>/memory/MEMORY.md`) — status claims,
     "current" markers, branch lists and headline metrics go stale fastest there.
4. Note which docs are **frozen by convention** (per-change run specs, ADRs, dated
   reports). Those get a dated correction note, never a rewrite — flag them separately.

## Step 1 — Scope to the diff when there is one

```bash
git diff --name-only; git diff --name-only HEAD~1
```
Prioritize checks touching the changed files. A full sweep is the no-argument fallback,
but a diff-scoped sweep finds real drift far faster.

## Checks to perform

### 1. CLI / public-interface consistency
Read the argument parsers or command definitions of the project's entry points
(`argparse`/`click`/`commander`/`clap`/flag packages, `package.json` `bin`, `Makefile`
targets) and compare flag names, defaults, and choices against every usage example in the
docs. Flag missing, renamed, removed, or default-changed options.

### 2. Test counts and test inventory
Collect the real number, don't trust the doc:
- pytest: `pytest --co -q -o "addopts=" 2>/dev/null | tail -1`
- jest/vitest: `npx jest --listTests 2>/dev/null | wc -l`
- go: `go test ./... -list '.*' 2>/dev/null | grep -c '^Test'`
- cargo: `cargo test -- --list 2>/dev/null | grep -c ': test$'`

Compare against the counts and test-file lists in the testing docs. A doc that names a
count is the single most reliably stale line in any repo.

### 3. Module / file inventory
List the project's source files and compare against any architecture or file-inventory
doc. Flag both directions: **documented-but-missing** (deleted or renamed) and
**present-but-undocumented** (new module the architecture doc never learned about).

### 4. Dependencies
Compare the manifest (`requirements.txt`, `pyproject.toml`, `package.json`, `Cargo.toml`,
`go.mod`) against the setup/install docs — versions, added packages, dropped packages,
and the language/runtime version the setup doc tells people to install.

### 5. Bug log
Read the bug log (`BUGS.md`, `KNOWN_ISSUES.md`, the issue tracker if that's the
convention). For each open entry, check whether the code it describes still behaves that
way, and whether its `file:line` references still point at the described code. Flag
**silently fixed** bugs (code changed, log never updated) and **drifted line refs**.

### 6. Stale status anchors (the highest-value check)
Grep the living docs, `CLAUDE.md`, and auto-memory for claims that go stale by nature,
and verify each against reality:
- **Headline metrics** — accuracy/AUC/latency/benchmark numbers, "current best", "SOTA",
  "baseline is X". Verify against the newest result record; flag any number the latest run
  contradicts.
- **"Current" / "active" / "pending" / "TODO" / "blocked" markers** — anything the code
  now resolves. A doc that says "TODO: X" when X shipped is the most common miss;
  **grep for the thing you just shipped.**
- **Algorithm / code-path claims** — a doc naming a module or approach that has since been
  replaced or removed. Check the named symbol still exists: `git grep -n '<symbol>'`.
- **Debug / workflow guidance** — a doc recommending a script or flag that is now the
  wrong path.
- **Branch and version lists** — verify against
  `git branch --all --sort=-committerdate | head -20` and the current tags.
- **Counts of hand-curated data** (label counts, dataset sizes, fixture counts) — recount
  from the source of truth; these drift between docs and disagree with each other.
- **Model / tool version pins** — a commit trailer, config, or doc pinned to a superseded
  model or CLI version.
- **Superseded forward-looking guidance** — old consults, plans, or predictions cited as
  current direction after their premise was settled.

## Output format

One table for the whole sweep, most-load-bearing drift first:

```
| Doc | Stale reference | Current reality | Fix (old → new) |
|-----|-----------------|-----------------|-----------------|
```

Then, separately: **frozen docs needing a dated note** (not a rewrite), and
**undocumented-but-present** items.

If nothing is stale, say so in one line: `✅ Docs in sync with the checked scope (<scope>).`

## Rules

1. **Be precise** — quote the exact stale text and the exact current code/output. No vague
   "may be out of date".
2. **Only flag real contradictions.** A doc vague enough to still be true is not drift.
3. **Verify before reporting** — re-check each finding against the code and drop the false
   positives. A findings table with noise in it stops getting read.
4. **Skip missing docs silently.** Don't report a doc that doesn't exist as a finding.
5. **Find, don't fix** — output the table. Apply the edits only when the user asks (then
   respect the frozen-doc convention: dated note, never a rewrite).
6. For a large or cross-cutting sweep, launch the `doc-sync-checker` agent (the project's
   own if it has one, else the global one) and report its table.
