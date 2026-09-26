---
name: test-runner
description: "Run the project's test suite the way the project actually runs it, then diagnose every failure down to a root cause in the source. Use when the user says 'run the tests', 'run the suite', 'why is this test failing', 'check for regressions', or asks for a pass/fail count."
user_invocable: true
---

# /test-runner — Run and diagnose the suite

Any arguments passed to this skill scope the run (a file, a marker, a keyword filter).
With no arguments, run the full suite.

## Step 0 — Resolve the environment and the command (never assume)

Read `CLAUDE.md` / `AGENTS.md` FIRST. A project that documents its test command has
already encoded the traps — a wrong invocation can pick up the wrong interpreter, the
wrong config, or a stale virtualenv, and every number you report afterwards is fiction.

Resolution order for the command:
1. An explicit command in `CLAUDE.md` / `AGENTS.md` (including any config-override flags
   the project says to pass).
2. Project config — `package.json` `scripts.test`, `Makefile` `test` target,
   `pyproject.toml` / `pytest.ini` / `tox.ini`, `Cargo.toml`, `go.mod`, a `scripts/test*`
   wrapper.
3. The ecosystem default, stated as an assumption in your report.

Resolution order for the environment:
- Python: the project's own venv, **by the exact name the project uses** — a repo with
  both `.venv` and `venv` almost always has one that is broken; check the docs, don't
  guess. Verify with `python -V` before running anything.
- Node: the lockfile's package manager (`npm ci` / `pnpm` / `yarn`), and the `.nvmrc`
  version if present.
- Env vars: if the project needs shell env (API keys, paths), source the project's
  documented startup (`source ~/.zshrc` and any bootstrap script) first.
- Worktrees: `pwd` and `git branch --show-current` **before** running. A backgrounded
  command does not carry `cd` forward, so prefix each call with `cd <worktree> &&` —
  otherwise you will silently test a different branch. A test count far off the expected
  number is the tell.

State the resolved command and interpreter in your report. If the project has **no** test
command, say so explicitly rather than inventing one.

**HEARSAY:** the environment is the uv-managed `.venv` (Python 3.12, pinned in
`.python-version`); run things with `uv run …` from the repo root. The test command is
`uv run pytest` (config in `pyproject.toml`, tests in `tests/`); as of the pre-event setup
the suite is empty, so pytest collects nothing (exit code 5) — report that as "no tests",
not as a pass. Markers: `needs_data`, `needs_weights`, `slow`. Tests
that need datasets or weights (gitignored, not on every teammate's laptop) must be marked
and reported as "skipped because gated", never as passes.

## Step 1 — Collect before you run

Get the count first (`pytest --co -q`, `jest --listTests`, `go test -list`,
`cargo test -- --list`). A collection error is a different bug from a test failure, and
the count tells you immediately whether you're pointed at the right tree.

Note the project's test **markers/tags** (slow, integration, requires-network,
requires-model, requires-device) and which the default run includes or excludes.

## Step 2 — Run

Run the resolved command. Capture full output. Don't stop at the first failure unless the
suite is unusably slow — the failure *set* is more informative than the first element.

If a single test hangs, re-run it alone with a timeout rather than letting the suite stall.

## Step 3 — Diagnose every failure at the source

For each failure:
1. Read the test — what contract does it assert?
2. Read the source it exercises. **Trace the actual code path.** Do not guess from the
   assertion message.
3. Classify:
   - **Real regression** — the code broke the contract. Fix the code.
   - **Stale test** — the test pinned an old contract the project deliberately moved off.
     Only then may the test change, and you must be able to say in one sentence: "this
     test asserted X; the change intentionally moves behavior to Y." Say it in the report.
   - **Environmental** — missing asset/model/device/network, wrong interpreter. Say what's
     missing and how to get it; do not paper over it.
   - **Flake** — re-run to confirm non-determinism; report the rate, not "flaky".
4. Give the fix at `file:line`.

**Never change a test to make it pass** outside the stale-test case above. If you cannot
state the contract move in one sentence, debug the failure instead.

## Output format

```
| Result | Count |
|--------|-------|
| passed / failed / skipped / errors | … |
```
Then per failure:
```
| Test | Failure (one line) | Class | Root cause (file:line) | Fix |
|------|--------------------|-------|------------------------|-----|
```

Always include: the exact command you ran, the interpreter/runtime version, the branch and
worktree, and — if the project has a known-good baseline count — the delta against it.

## Rules

1. Report the real numbers, including failures. Never round a failing suite up to "passing".
2. Root cause from the code, not the traceback text alone.
3. Distinguish "skipped because gated" from "skipped because broken".
4. If a pre-existing failure is unrelated to the current work, say so — and still count it.
5. If the run was partial (scoped, or cut short), label the numbers as partial.
