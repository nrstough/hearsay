### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | D6’s holdout argmin readout omits the always-real operating point. |
| Scope discipline | Excellent | Changes stay within the amended manifest’s scope. |
| Test coverage | Fail | A5 lacks an independent counterfactual for P_wl’s ITW corrective-share threshold. |
| Review compliance | Acceptable | Previous audit fixes are present; model/export identity remains an acknowledged residual risk. |
| Freeze integrity | Acceptable | No P1/P2/P3 hashes found; original manifest preserved with appended amendments. |
| Regression check | Acceptable | Available tests and lint pass; protected artifact hashes match. Full rerun limited by read-only access. |
| Documentation | Acceptable | Required disclosures updated; historical pending wording remains above the final result. |
| **Overall** | **Fail** | |

### Commentary

1. **Plan adherence — causes Fail.** [Holdout threshold selection](/Users/nathanstough/Projects/hearsay/scripts/fuse_sweep_v3.py:141) searches only observed scores, excluding `+inf`. With two real and two spoof rows all scoring 0.5, I reproduced `holdout_brief = 1.0` alongside argmin **[2, 0]**; the actual optimum is **[0, 2]**. Include the always-real threshold, as `inner_threshold` already does, and add a regression test. This does not change the recorded candidates’ results.

2. **Test coverage — causes Fail.** [P_wl composition tests](/Users/nathanstough/Projects/hearsay/tests/test_fuse_sweep_v3.py:644) reject an inner corrective share of 0.5 and an ITW NaN, but never a finite ITW share ≤ 0.5 with the other conditions passing. I removed the ITW threshold check in memory while retaining finiteness checks; the composition test still passed. Add independent ITW cases at 0.5 and below, with sufficient catches and inner share > 0.5, to satisfy A5.

3. **Test coverage / Regression check — verification limitation, no additional failure attributed.** Independent execution produced **91 passed, 1 skipped, 15 deselected**, including the data self-checks. Temporary-write tests were excluded under the read-only sandbox. Repository-wide Ruff passed. Both protected TSVs, fusion_v2 constants, and the final report match their recorded hashes. The complete final repository suite was not independently rerun.

4. **Review compliance — limits grade to Acceptable.** The recorded C6 resolution explicitly accepts that the new model’s relationship to its export is established through the heads report rather than verified model bytes. `--write` records the supplied model’s metadata hash but does not prove that model produced the evaluated export. No candidate passed, so this residual risk did not affect KEEP.

5. **Documentation — cosmetic downgrade to Acceptable only.** The sweep report’s opening and pending sections still describe H_noise as forthcoming, although its appended final section records FAIL and KEEP. Update the opening status to point directly to the final run; the appended result already makes the outcome recoverable.
