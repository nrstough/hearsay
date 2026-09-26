### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | Perturbation checks do not enforce all five required kinds / ten cells. |
| Scope discipline | Excellent | Changes stay within the amended manifest and gate lane. |
| Test coverage | Fail | A5 lacks a counterfactual for the brief-cost bootstrap requirement. |
| Review compliance | Acceptable | Recorded resolutions implemented; model/export identity remains an explicitly acknowledged residual risk. |
| Freeze integrity | Acceptable | No P1/P2/P3 hashes found; original manifest text preserved, amendments appended. |
| Regression check | Acceptable | Protected hashes match; lint passes. Independent testing was limited by the read-only environment. |
| Documentation | Fail | AI-use disclosure contradicts the completed WavLM evaluation; mandated README negative is missing. |
| **Overall** | **Fail** | |

### Commentary

1. **Plan adherence — causes Fail.** [Perturbation evaluation](/Users/nathanstough/Projects/hearsay/scripts/fuse_sweep_v3.py:406) iterates only over kinds present in the CSV, and [the verdict](/Users/nathanstough/Projects/hearsay/scripts/fuse_sweep_v3.py:264) checks only supplied cells. I reproduced **PASS** with just `noise20_brief` and `noise20_averse`, plus passing standing-rule/bootstrap inputs. T2 likewise checks only present perturbation kinds. Require all five kinds, the expected cohort coverage, and all ten cost cells; missing evidence must produce INVALID. The current CSV contains 500 rows per kind, so this defect does not invalidate the recorded KEEP result.

2. **Test coverage — causes Fail.** [The bake-off counterfactuals](/Users/nathanstough/Projects/hearsay/tests/test_fuse_sweep_v3.py:280) test a nonpositive **averse** bootstrap percentile, but never a nonpositive **brief** percentile with everything else passing. Removing `boot["brief_p5"] > 0` would survive these tests, contrary to A5. Add independent zero/negative cases for both costs, plus missing-kind/cell cases for finding 1.

3. **Documentation — causes Fail.** [CLAUDE.md’s disclosure](/Users/nathanstough/Projects/hearsay/CLAUDE.md:141) still says WavLM Large was “not used in any run”; [README.md](/Users/nathanstough/Projects/hearsay/README.md:414) says “never run.” Both contradict the locked P_wl evaluation. The governing manifest also explicitly requires a README negative when P_wl fails; proposed wording in the sweep report has not reached README. Update the disclosure and README to distinguish evaluated-but-rejected WavLM Large from the shipped models.

4. **Regression check / Test coverage — verification limitation, no additional failure attributed.** Independent checks produced **87 passed, 1 skipped, 14 deselected**, including the data self-checks, and repository-wide Ruff passed. Tests requiring temporary writes could not be rerun under the read-only sandbox; the complete final suite was therefore not independently verified. The shipped TSV, final-name copy, fusion_v2 constants, and locked report match their recorded hashes. The final KEEP log row remains conditional on the pending ruling/H_noise close-out, so its absence at this audit is not a regression finding.
