### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | Checkpoint integrity requirements remain incomplete. |
| Scope discipline | Excellent | Changes stay within lane-owned files; detector contract and submissions untouched. |
| Test coverage | Fail | Tests miss incomplete score masks and truncated checkpoint JSON at gate time. |
| Review compliance | Fail | Accepted checkpoint-validation finding #3 is only partially resolved. |
| Freeze integrity | Acceptable | No freeze hashes; owner statements present. Pre-run declaration cannot be verified from git. |
| Regression check | Acceptable | 31 targeted tests and lint passed independently; full-suite success is recorded. |
| Documentation | Acceptable | Results match saved artifacts; diagnostic limitations and unavailable duration readout are disclosed. |
| Test sensitivity | Excellent | Independently reproduced both mutation catches in memory; neither mutation appears in the change’s commits. |
| **Overall** | **Fail** | Remaining defects concern checkpoint refusal, not evidence that the saved results are wrong. |

### Commentary

1. **Plan adherence / Review compliance / Test coverage — causes Fail.** [validate_cells()](/Users/nathanstough/Projects/hearsay/scripts/v2_hardneg_bakeoff.py:864) trusts the checkpoint’s own mask when checking score coverage. In a read-only reproduction, removing one NSA row from that mask and setting its score to NaN still returned **OK**: 16,141 scored rows against the stored reading’s 16,142. The gate can consequently bootstrap a different population from the cached readings. Validate array dimensions, Boolean mask type, and exact mask equality against the authoritative row index and training pool. Add a regression test requiring this checkpoint to become INVALID.

2. **Plan adherence / Review compliance / Test coverage — causes Fail.** [load_cells()](/Users/nathanstough/Projects/hearsay/scripts/v2_hardneg_bakeoff.py:844) parses every JSON checkpoint before `validate_cells()` can handle errors. A truncated JSON raises `JSONDecodeError`, aborting the gate without producing the promised refusal report. The [truncation test](/Users/nathanstough/Projects/hearsay/tests/test_hardneg.py:1102) corrupts only the NPZ file; JSON truncation is tested only for fit resume. Handle malformed JSON during loading, identify the affected declared cell, and test the gate’s INVALID or explicit-refusal result.

3. **Freeze integrity — limits grade to Acceptable.** The first declaration commit follows execution, as [disclosed](/Users/nathanstough/Projects/hearsay/docs/reports/2026-10-13_v2-33-hardneg-stage0.md:389). No P1/P2/P3 freeze hashes exist. The owner statements contain both required risks, but the claimed pre-run text cannot be independently authenticated. Future confirmation runs need the declaration committed before execution.

4. **Regression check — limits grade to Acceptable; no observed regression.** Independent checks found 132 cells, matching current fingerprints, 131 unchanged OOF/mask arrays versus run 1, and no saved mask/shape mismatch. Verdict counts match the report: 24 BASE, 93 FAIL, 8 PASS, 6 INDETERMINATE, 1 INVALID. The 31 targeted tests and repository lint passed. The recorded 1,248-pass full suite was not rerun in this read-only environment.
