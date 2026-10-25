# Ownership checks (shared by every judgment skill)

The failure this guards against: the AI drafts the thinking, the user approves without
engaging, and the work proceeds with nobody owning the decisions. The fix is friction at
the few points that matter, not everywhere. Each skill names **where** these checks
happen; this file defines **what** they are. Read it whenever a skill points here.

None of these checks are optional, and none may be satisfied on the user's behalf.

In a repo without `~/.claude` (a cloud session), this file lives at `.claude/ownership-checks.md`.

## 1. Stakes triage

Classify the work and state it on one line, with a reason, before anything else:

> **Stakes: HIGH** — changes how lift scores are computed, which users see directly.

| Class | Covers |
|-------|--------|
| **LOW** | Plumbing, tooling, docs, config, refactors with no behavior change, quick factual questions |
| **HIGH** | Anything touching core logic, model output, scoring or judging, user-facing results, data handling, or money spent |

- **If unsure, classify HIGH.** Never pick LOW to save effort.
- The user can override at any time; record the override and their reason.
- If later work shows a LOW item reaches HIGH territory, reclassify upward and say so.
  Only the user can downgrade.

## 2. Owner's statement gate

At each gate point a skill names, ask the user to type, **in their own words**:

1. **One sentence on what this step does / decides.**
2. **The most likely way it is wrong.** Required for HIGH; skip it for LOW.

Rules:
- **"freeze", "looks good", "lgtm", "yes", "go", "ship it" or similar, on their own, are
  not approval.** Reply in one line asking for the sentence(s); do not proceed.
- **Do not write the sentence for them**, and do not offer a model answer to copy. Their
  wording is the point.
- **Check the sentence against what you drafted.** If it describes something different,
  misses the core of it, or contradicts it, say plainly where they differ. Then either
  your draft is wrong (fix it and re-present) or their reading is (next rule).
- **If it doesn't match, or they say they can't write it:** explain the step in plain
  English, no jargon, no file names, a few short sentences. Then ask again.
- **Feed the risk forward.** A "most likely way it is wrong" is not a formality: turn it
  into a test, a check, or an explicit question in the next step, and say where it went.
- **A partial answer** that covers a minor piece but misses the core gets the plain
  explanation and a re-ask, the same as a wrong one.

**Answers that never pass.** Judge by comparing what the answer *claims* against what you
drafted, never by how fluent or reasonable it sounds. Reject, and name which of these it is:

| Pattern | Example |
|---------|---------|
| Bare approval with filler | "fix everything", "do it all", "yes that", "it does the thing" |
| Pointing elsewhere | "same as before", "as discussed", "what you said", "see the plan" |
| Restating the question | "it does what this phase does" |
| Pasted text | 8+ consecutive words lifted from the draft, spec or plan, even if accurate. Ask for it in their words |
| Non-specific risk | "might have bugs", "could break", "something could go wrong" |
| **Plausible but wrong** | Fluent and confident, but names a different problem, approach or mechanism than the draft. Name the exact mismatch ("you said it caches results; it actually recomputes them") |

- **Keep the sentences verbatim**, typos and all, under a heading `## Owner's statement`
  in the skill's record (run spec, consult response, report). Once written they are
  frozen: never edit, paraphrase, or tidy them. If no file exists yet, carry them in the
  conversation until one does. `/handoff` carries them across chats.

## 3. Test break check (HIGH, whenever code and tests change)

Proves the key test can actually fail. Runs once per change, after tests pass, after the
commit (a worktree is built from a commit, so it then contains the change).

1. Pick the **one test most tied to the core logic** of the change. Say which, and why,
   in one line.
2. Make a throwaway worktree **outside the repo**, detached so no branch is created:
   `git worktree add --detach <scratch-dir>/breakcheck-<slug> HEAD`
3. In that worktree only, introduce **one deliberate, realistic bug** in the code the test
   covers (not in the test): a flipped comparison, an off-by-one, a wrong threshold or
   sign, swapped arguments, a skipped branch. For a bug fix, reverting the fix is the
   natural mutation. A syntax error, an import crash or a deleted function does not count.
4. Run **only that test** in the worktree. Point it at untracked assets (deps, venv,
   models) read-only if needed. Never mutate the main checkout. If the test cannot run at
   all, that is a finding, not a skip.
5. Report one line:
   `Break check: <test> vs <file:line> (<the bug>) → CAUGHT` or `→ NOT CAUGHT (decorative)`
6. Clean up: `git worktree remove --force <path>`; confirm with `git worktree list` and
   that `git status` in the main checkout is unchanged. **Never commit, merge, cherry-pick
   or copy the mutation.**
7. **If NOT CAUGHT:** the test is decorative. Strengthen it in the main checkout, commit,
   re-run steps 2–6 with the same mutation. Loop until caught.
8. Record under `## Test break check`: test, mutation, result, any fix-and-recheck rounds.

## 4. Plain-English summary before HIGH approvals

Before the user approves anything HIGH (a plan, a fix, a fanout, a train decision, acting
on outside advice), lead with 3–5 plain-English bullets: what will behave differently,
before → after, and who or what sees the difference. No file paths, line numbers,
function names or jargon. The user should be able to explain what they are approving from
this alone. If it doesn't match their earlier Owner's statement, say so.

## Where each skill applies them

| Skill | Triage | Owner's statement gate | Break check | HIGH summary |
|-------|--------|------------------------|-------------|--------------|
| `plan-review` | start of P1 | P1 freeze, P2 freeze | post-commit | STOP 3 |
| `workflow-review` | start of P1 | P1 freeze, P2 freeze | post-commit | STOP 3 |
| `p3` | from run spec, else now | before executing, if no statement exists | post-commit | before executing |
| `consult` | Step 1 | before drafting, before sending, each clarifying round, on the response | — | before acting on the advice |
| `diagnose` | Phase 1 | before the fix (Phase 5) | the regression test, by reverting the fix | at the post-mortem |
| `bug-investigate` | per bug | before any fix or ledger patch is applied | per HIGH fix | per HIGH bug |
| `fleet-campaign` | always HIGH | success contract, and before fanout | contract check on bad output | before fanout |
| `validate-training-data` | Step 0 | at the train / don't-train verdict | — | at the verdict |
| `improve-codebase-architecture` | per candidate | when a candidate is chosen to build | via `/plan-review` | via `/plan-review` |
| `ios-review`, `test-runner` | only when changing code or tests | before a HIGH fix, or before rewriting a test as "stale" | per HIGH fix | before a HIGH fix |

Mechanical skills (`doc-sync`, `shell-audit`, `handoff`, `zoom-out`, `graphify`) don't
run these checks. `handoff` carries their state between chats.
