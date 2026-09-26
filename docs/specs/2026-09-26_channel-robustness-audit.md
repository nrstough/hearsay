### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | All-failed E probes still crash instead of producing “inconclusive.” |
| Scope discipline | Excellent | Changes remain diagnostic; deviations are recorded. |
| Test coverage | Fail | Missing regression coverage for error-only caches with no score columns. |
| Review compliance | Fail | The adopted incomplete-probe contract remains partly unimplemented. |
| Freeze integrity | Acceptable | Skipped: no P1/P2/P3 hashes found. |
| Regression check | Acceptable | Saved E results and codec verdict reproduce; full-suite rerun limited by sandbox. |
| Documentation | Acceptable | Required report, STATUS entry, disclosure and shortcut-ledger entry exist. |
| **Overall** | **Fail** | One remaining completeness defect. |

### Commentary

1. **Plan adherence / Test coverage / Review compliance — causes Fail.** When every clip fails decoding or scoring, `run_scoring` can produce an error-only cache without score columns. Both [perturb_readout](/Users/nathanstough/Projects/hearsay/scripts/m3_probes.py:314) and [mlaad_readout](/Users/nathanstough/Projects/hearsay/scripts/m3_probes.py:377) then raise `KeyError: 'm1b_v3'`. Reproduced both cases. The latest tests retain score columns, so they miss this failure. Normalize missing columns or return an explicit empty readout; test error-only caches through readout → verdict, requiring “inconclusive.”

2. **Regression check — no failure downgrade.** Recomputed perturbation, MLAAD and verdict JSONs match the saved outputs exactly: 500 paired clips, 572 MLAAD clips, verdict “kept.” The codec calculation also reproduces “no match,” with all 34 variants containing 100 reference clips and 1,671 test rows. Finding 1 does not invalidate those recorded results.

3. **Test coverage / Regression check — verification limitation, not an additional defect.** The spec records 583 passing tests. This read-only sandbox blocked `uv` cache initialization; direct targeted execution produced 117 passes, seven cache-related failures and one temporary-directory error. Documentation tests independently passed **52/52**. Changed files pass Ruff; repository-wide Ruff reports only the three recorded findings in `src/hearsay/analyzer.py`.
