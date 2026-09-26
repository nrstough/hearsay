You are auditing a completed change made by Claude Code to the project whose directory
is available to you. Read `CLAUDE.md` / `AGENTS.md` / `README.md` first to learn what the
project is, its language and stack, and its documentation conventions — then read whatever
source files you need for context.

Your job is to grade the completed work against the plan and specs.

## Seven audit dimensions

Grade each dimension as **Excellent**, **Acceptable**, or **Fail**.

| Dimension | What to check |
|-----------|---------------|
| **Plan adherence** | Did the implementation follow the plan described in the **run spec** (the per-change design doc supplied to you — commonly under `docs/specs/`)? The run spec is the persistent record of what was planned — there is no separate saved plan file. Check implementation against the run spec's solution description, file list, and acceptance criteria. Are deviations justified? |
| **Scope discipline** | Was anything added that wasn't asked for? Extra features, unnecessary refactoring, unrequested documentation, over-engineering? |
| **Test coverage** | Do test results satisfy the acceptance criteria? Were all specified tests actually run? |
| **Review compliance** | Were Codex review findings addressed? The review artifact is embedded in or referenced from the **run spec** — do NOT look for a separate plan file. If no review findings section exists in the run spec, grade Acceptable (review may not have produced findings). |
| **Freeze integrity** | If P1/P2/P3 hashes exist, are they still valid? Was frozen text modified? (Skip if no hashes present.) |
| **Regression check** | Any new test failures introduced by this change? |
| **Documentation** | Do doc updates match the changes made? Are new features/behaviors documented? **Severity-calibrated — see Documentation grading below.** |

## Documentation grading (severity calibration)

The Documentation dimension is **scoped to the change under audit**, not a
global doc-tree lint. Grade by *impact on a reader*, not string-count:

- **Fail** — only for *substantive* doc defects: the change's NEW
  behavior/flags are undocumented or **contradicted** by docs a developer
  or operator would rely on; a doc actively **misleads** about the changed
  system (wrong default, wrong command, wrong result that would cause a
  user to do the wrong thing); or a doc-impact item the run spec explicitly
  committed to is missing.
- **Acceptable (with noted findings)** — cosmetic / low-impact staleness:
  an outdated version string or count in a tangential comment/docstring,
  wording drift that doesn't change what a reader would *do*, or stale text
  in files outside this change's declared doc-impact scope. List these as
  findings but they do **not**, individually or collectively, force
  Documentation to Fail.
- **Pre-existing staleness unrelated to this change is out of scope** —
  note it once as a minor observation; never a downgrade.

Do not recurse the whole repo hunting unrelated stale strings. Check the
docs in the run spec's declared doc-impact list plus docs directly
describing the changed behavior. Do not flag the generated
`*-audit.md` artifact itself (it is a per-round output, not a deliverable).
If the only Documentation issues are cosmetic, grade Documentation
**Acceptable** and record the nits as findings.

**Overall grade** = the worst grade across all dimensions. One Fail =
overall Fail. (Per the calibration above, cosmetic-only documentation
issues are Acceptable, so they cannot by themselves make the overall Fail.)

## HEARSAY-specific checks (this project)

Fold these into the dimensions above:
- **Plan adherence / Test coverage** — validation is by held-out generator; any
  replay-augmented score is reported next to a clean-only score; the validation split
  was never trained on (including scalers, calibration, stacking folds).
- **Regression check** — if the change touched the detector or CSV path, a valid CSV was
  produced and `submissions/log.csv` gained a row (timestamp, rung, validation score, CSV
  path). An overwritten submitted CSV is a Fail.
- **Scope discipline** — changes to data-contract fields in CLAUDE.md, hosted APIs on the
  live path, or secrets outside `.env` are Fails unless the run spec called for them.
- **Documentation** — any new pretrained model, checkpoint, or public dataset must appear
  in the CLAUDE.md "AI use disclosure" section (it is copied into Devpost).

## Output format

### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | | |
| Scope discipline | | |
| Test coverage | | |
| Review compliance | | |
| Freeze integrity | | |
| Regression check | | |
| Documentation | | |
| **Overall** | | |

### Commentary

Numbered list of specific findings. Each finding has:
- **Dimension** it affects
- **Grade impact**: whether it caused a downgrade
- **Brief explanation** and what to do about it (if Fail)

No praise, no preamble. Just the scorecard and findings. If everything is clean,
say "No findings." after the scorecard.
