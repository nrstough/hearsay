### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | Primary analysis and gate differ materially from the final design. |
| Scope discipline | Excellent | Changes stay within lane-owned files; detector and submission paths untouched. |
| Test coverage | Fail | Passing tests miss required gate behavior and aggregate inference. |
| Review compliance | Fail | Accepted findings #2, #3 and #8 remain incompletely implemented. |
| Freeze integrity | Acceptable | No freeze hashes recorded. Owner statements include both risks; pre-run frozen text cannot be independently verified. |
| Regression check | Acceptable | Recorded full suite: 1,234 passed, 3 skipped. Independently verified 26 targeted tests and repository lint; full suite not rerun in this read-only environment. |
| Documentation | Fail | Report and status document propagate an incorrect pass count and unsupported aggregate conclusions. |
| Test sensitivity | Excellent | Recorded C2 mutation caught on the first attempt across three seeds; mutation absent from script history. |
| **Overall** | **Fail** | |

### Commentary

1. **Plan adherence / Review compliance / Documentation — causes Fail.** The primary comparison is not implemented as specified. [The aggregation](/Users/nathanstough/Projects/hearsay/scripts/v2_hardneg_bakeoff.py:823) averages all 24 strata, including different C values, while its output says “C … never averaged over.” Bootstrap results are summarized separately for each stratum; no replicate-level aggregate distribution or aggregate uncertainty is produced. The report nevertheless uses these averages to conclude “does nothing” and “indistinguishable from random.” Compute each contrast separately at each C, averaging layers and pools within shared bootstrap replicates, and update the report, status and log.

2. **Plan adherence / Test coverage / Documentation — causes Fail.** [R1 indeterminacy does not prevent PASS](/Users/nathanstough/Projects/hearsay/scripts/v2_hardneg_bakeoff.py:531). The saved `easiest_both`, L7, `pool_nsa`, C=8 cell is marked both indeterminate and PASS at P(better)=0.903904. This violates the explicit final rule. Enforcing it reduces the reported nine passes to eight; the no-survivor conclusion remains unchanged. Make indeterminacy block PASS and R5 eligibility. The [existing test](/Users/nathanstough/Projects/hearsay/tests/test_hardneg.py:759) checks only the flag, allowing this defect to pass.

3. **Plan adherence / Review compliance — causes Fail.** Checkpoint protection remains incomplete. [The fingerprint](/Users/nathanstough/Projects/hearsay/scripts/v2_hardneg_bakeoff.py:579) hashes manifests but not embedding contents; repaired embeddings can therefore reuse stale scores. NPZ writes are non-atomic, and `stage_gate()` neither validates fingerprints nor refuses incomplete grids. Current readable checkpoint fingerprints match, so this is a reproducibility defect rather than evidence that this run mixed inputs. Hash embedding contents, validate checkpoint pairs before reuse and gating, and test changed shards, missing/truncated NPZ files and incomplete grids.

4. **Plan adherence / Test coverage — causes Fail.** [R9](/Users/nathanstough/Projects/hearsay/scripts/v2_hardneg_bakeoff.py:567) turns detected degeneracy into ordinary FAIL, not INVALID. It also omits explicit constant-score refusal and complete checks across all readings. Such cells retain `status="OK"` and remain eligible for the primary contrasts. Centralize refusal before gate evaluation and aggregation; test both INVALID classification and exclusion from contrasts.

5. **Review compliance — contributes to Fail.** The persisted deadline is consulted by fitting and the C sweep, but [gate execution](/Users/nathanstough/Projects/hearsay/scripts/v2_hardneg_bakeoff.py:743) never checks it. Bootstrap work can continue beyond the declared cutoff, contrary to accepted finding #8. Check the deadline during bootstrap scheduling and produce an incomplete report when it expires. The recorded runs finished within the limit, so no actual overrun is established.

6. **Plan adherence — contributes to Fail.** P1 required committing recipes, selection rules and bands before reading results. Git history first adds the script, run spec and review artifacts in `7ba7faf` on October 17, after both recorded runs. This does not prove retrospective editing, but it does establish that the promised pre-run commit record is missing. Record that limitation explicitly; future confirmation runs need a committed declaration before execution.
