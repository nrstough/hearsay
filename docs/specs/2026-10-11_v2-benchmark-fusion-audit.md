### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | Full-readout acceptance remains unmet; cohort integration and some required statistics are missing. |
| Scope discipline | Excellent | Changes stay within the documented implementation and documentation scope. |
| Test coverage | Fail | Named tests exist, but they do not establish complete candidate readouts or cohort integration. |
| Review compliance | Fail | The provisional-WIN follow-on finding remains unresolved. |
| Freeze integrity | Excellent | Frozen text is unchanged; candidate/spec hashes validate. Owner statements include both required risks. |
| Regression check | Acceptable | Lint passes; reference values, TSV parity and determinism reproduce. Full pytest execution is sandbox-blocked. |
| Documentation | Acceptable | Pending work and limitations are disclosed; minor bookkeeping is stale. |
| Test sensitivity | Excellent | Realistic transductive-rank mutation is caught immediately; no committed mutation found. |
| **Overall** | **Fail** | Acceptance and review requirements remain open. |

### Commentary

1. **Plan adherence / Test coverage — causes Fail.** The cohort work is incomplete beyond awaiting compute approval. [v2_fusion.py](/Users/nathanstough/Projects/hearsay/.claude/worktrees/v2-benchmark-fusion/scripts/v2_fusion.py:227) loads only the four legacy holdout columns and unconditionally marks ITW cohorts pending. Producing the new CSVs alone cannot complete the readouts. Connect the new files to the registry and both reporting paths, add an integration test, then complete the approved compute and regenerate reports. Until then, “all four candidates have full readouts” is unmet.

2. **Plan adherence / Test coverage — causes Fail.** Already-scored populations also lack required readout fields. [The fusion ITW output](/Users/nathanstough/Projects/hearsay/.claude/worktrees/v2-benchmark-fusion/scripts/v2_fusion.py:214) contains only pooled metrics for every candidate; it lacks speaker-cluster intervals and without-E results. The saved JSON confirms this. Holdout likewise lacks without-E results. Reuse the benchmark reporting functions and test the complete output schema, including explicit degenerate-interval markers where applicable.

3. **Review compliance — causes Fail.** [followon_allowed](/Users/nathanstough/Projects/hearsay/.claude/worktrees/v2-benchmark-fusion/src/hearsay/fusion_v2.py:252) still accepts `WIN (provisional)` despite the documented second-order stacking leakage. This is the unresolved critical review finding; C2’s current TIE merely leaves the branch unused. Preserve the historical declaration, but add a dated amendment requiring independent validation before authorizing a follow-on.

4. **Plan adherence — additional Fail finding.** The reports acknowledge repeated holdout-cohort evaluations outside the ledger, including during the original dry run. The current locking fix protects future runs but cannot establish read-once compliance for this completed experiment. Retain the historical exception and treat these readouts as exploratory; confirmatory evidence requires fresh, prospectively controlled evaluation.

5. **Regression check — limits grade to Acceptable.** The recorded full-suite result is **991 passed, 12 skipped**. Independent pytest execution failed during temporary-directory setup because this sandbox is read-only, not because of an implementation assertion. Direct read-only execution reproduced all seven reference cases, shipped-TSV parity and the determinism check; Ruff passed. These checks do not substitute for independently running the full suite.

6. **Documentation — limits grade to Acceptable; no substantive downgrade.** The [run-spec results](/Users/nathanstough/Projects/hearsay/.claude/worktrees/v2-benchmark-fusion/docs/specs/2026-10-11_v2-benchmark-fusion.md:287) still say ledger count **2**, while current reports and saved readouts say **4**. Its final paragraph also says empty reservations fail closed, whereas current `reserve()` recovers them under the lock. Update these non-frozen bookkeeping statements.
