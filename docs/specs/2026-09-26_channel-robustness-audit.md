### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | D9’s incomplete-probe refusal remains incomplete. |
| Scope discipline | Excellent | Diagnostic scope preserved; no detector-contract, training, or submission changes. |
| Test coverage | Fail | Tests miss two reproducible completeness failures. |
| Review compliance | Fail | Review requirements for missing/non-finite scores are only partly implemented. |
| Freeze integrity | Acceptable | Skipped: no P1/P2/P3 hashes present. |
| Regression check | Acceptable | Saved results reproduce; no scoring-path changes. Full-suite rerun restricted by sandbox. |
| Documentation | Acceptable | Required report, intervals, STATUS, disclosure, and shortcut-ledger entry exist. |
| **Overall** | **Fail** | Incomplete inputs can still yield “kept.” |

### Commentary

1. **Plan adherence / Review compliance / Test coverage — causes Fail.** [MLAAD readout](/Users/nathanstough/Projects/hearsay/scripts/m3_probes.py:358) counts rows without checking score completeness. Replacing one cached row’s M3 score and suppression flag with NaN leaves `n = 572`; pandas silently excludes the missing flag from its mean, and the verdict remains **“kept.”** Require finite values for every required score and flag before counting a clip, and add a readout-to-verdict regression test requiring **“inconclusive.”**

2. **Plan adherence / Review compliance / Test coverage — causes Fail.** [Perturbation completeness](/Users/nathanstough/Projects/hearsay/scripts/m3_probes.py:322) uses `notna()`, which accepts infinity. Replacing one M3 score with infinity leaves `n_paired = 500` and produces **“kept,”** although metric calculations discard that score. Use a finite-value check consistently across cohort selection and metrics; test both positive and negative infinity.

3. **Regression check / Test coverage — verification limitation, no additional downgrade.** The saved cohorts contain complete finite scores, and the perturbation readout reproduces its saved JSON exactly; these defects do **not** invalidate the recorded verdict. The codec grid contains all 34 variants × 100 reference clips and 1,671 test clips. Documentation tests independently passed **52/52**. Changed-file lint passes; repository lint retains three findings from `19925a3` in `analyzer.py`. The recorded **572-pass** full suite could not be independently repeated: the read-only sandbox blocks uv caching and temporary files. A direct targeted run produced 106 passes, seven environment-related failures, and one deselected filesystem-writing test.
