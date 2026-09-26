### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | An entirely missing perturbation crashes instead of producing “inconclusive.” |
| Scope discipline | Excellent | Diagnostic scope preserved; deviations recorded; no scoring-path changes. |
| Test coverage | Fail | Missing-perturbation coverage does not exercise the failing readout path. |
| Review compliance | Fail | The adopted incomplete-probe requirement remains partly unimplemented. |
| Freeze integrity | Acceptable | Skipped: no P1/P2/P3 hashes present. |
| Regression check | Acceptable | Saved E readouts reproduce exactly; full-suite verification restricted by environment. Diagnostic-only submission exemption recorded. |
| Documentation | Acceptable | Required report, STATUS entry, disclosure and shortcut-ledger entry present; 52 documentation tests pass. |
| **Overall** | **Fail** | One reproducible incomplete-probe handling defect remains. |

### Commentary

1. **Plan adherence / Test coverage / Review compliance — downgrade to Fail.** Removing every MP3 row from the saved perturbation cohort makes `perturb_readout` raise `KeyError: 'mp3'` at [scripts/m3_probes.py:330](/Users/nathanstough/Projects/hearsay/scripts/m3_probes.py:330). The completeness check correctly finds zero paired clips, but metric computation then indexes the absent column. This violates the adopted requirement that an unfinished E probe yield “inconclusive.” Reindex pivots to all expected perturbations or return an incomplete readout before computing metrics. Add a readout → verdict regression test for an entirely absent perturbation, including rows excluded because scoring failed.

2. **Regression check — no failure downgrade.** Recomputed perturbation and MLAAD readouts exactly match the saved JSON: 500 and 572 clips, verdict “kept.” The 34-variant codec table also reproduces the recorded negative match. Finding 1 does not invalidate these completed measurements.

3. **Test coverage / Regression check — verification limitation, no additional downgrade.** On `main` at `fd779fb`, Python 3.12.13, the documented `uv` commands were blocked by read-only cache permissions. Direct `.venv` execution collected 537 tests before four environment-related collection errors. A scoped channel/docs run produced **107 passed, one cache-permission failure, 15 deselected**; a separate documentation run passed **52 tests**. The recorded **581-pass full suite was not independently reproduced**. Changed files pass Ruff; repository-wide Ruff reports the three documented, unrelated `analyzer.py` findings.
