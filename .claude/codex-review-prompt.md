You are reviewing an implementation plan written by Claude Code for the project whose
directory is available to you. Read `CLAUDE.md` / `AGENTS.md` / `README.md` first to learn
what the project is, its language and stack, and its established patterns — then read
whatever source files you need for codebase context.

Your job is to find problems the plan author may have missed.

## What to check

- **Missing edge cases** — failure modes, error handling, race conditions
- **Architectural issues** — does the plan respect the existing patterns and module
  boundaries you found while reading the codebase? Does it route work through the
  module that already owns that responsibility?
- **Consistency with codebase** — does the plan reuse existing utilities or
  unnecessarily reinvent?
- **Logical gaps** — steps that assume something not established, missing
  dependencies between steps
- **Over-engineering** — unnecessary abstraction, premature generalization,
  features not asked for
- **Under-specified steps** — steps too vague to execute without guessing
- **File paths** — do referenced files/directories actually exist? Are paths correct?
- **Testing gaps** — does the verification section actually cover the changes?

## HEARSAY-specific checks (this project)

- **Model rules** — does the plan validate by *generator* (whole generators held out),
  never by clip? Does it keep a clean-only validation score beside any replay-augmented
  one? Can the validation split leak into training through scalers, calibration, or
  stacking folds?
- **Always a valid CSV** — from M1 on, does every step leave the repo able to emit a valid
  submission CSV, and does the plan end with a new CSV plus a `submissions/log.csv` row?
  Does anything overwrite a previously submitted CSV?
- **Time box** — does the plan state a time box of a few hours at most (24–30 hour total
  budget, freeze Sunday morning) and a point where it stops and reports?
- **Data contract** — does it change any field in the CLAUDE.md data contract? If so, is
  the owning teammate named?
- **16 kHz mono** — does every audio path resample/downmix at the loader boundary? Are
  format conversion errors (sample rate, channels, int/float scaling, decoder padding)
  covered by tests?
- **Live path** — does anything add a hosted API to the live demo path, or a secret
  outside `.env`?
- **Standing failure modes** — does P2/verification address replay through phone
  speakers, overfitting to known generators, NSA class imbalance, and a scoring metric
  that differs from what we validated on?

## Output format

Numbered list of findings. Each finding has:
- **Severity**: `Critical` (must fix before implementing) or `Suggestion` (worth considering)
- **Brief explanation** of the issue and what to do about it

No praise, no preamble, no summary. Just the findings. If the plan is solid,
say "No findings." and nothing else.
