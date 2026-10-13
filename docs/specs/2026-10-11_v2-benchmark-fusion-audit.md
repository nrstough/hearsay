### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | Read-once and candidate-freeze enforcement have gaps; required cohort work remains incomplete. |
| Scope discipline | Excellent | Changes stay within the lane; no detector-contract, hosted-scoring, or submission changes. |
| Test coverage | Fail | Four required S8 tests are absent; concurrency and candidate-integrity tests miss realistic violations. |
| Review compliance | Fail | Atomic reservation and second-order leakage findings remain incompletely addressed. |
| Freeze integrity | Excellent | P1/P2 text is unchanged; recorded hash verifies; owner statements include both required risks. |
| Regression check | Acceptable | Reference, TSV-parity, determinism checks and lint pass. Full-suite verification was blocked by the read-only environment. |
| Documentation | Fail | The conditionally committed code-map update is missing. |
| Test sensitivity | Excellent | Recorded mutation was never committed; independently repeating it in memory caused the selected test to fail immediately. |
| **Overall** | **Fail** | |

### Commentary

1. **Plan adherence / Review compliance / Test coverage — causes Fail.** In `src/hearsay/benchmark.py:1004`, an existing reservation with identical code and input hashes returns `resumed` without establishing that its owner crashed. Two concurrent executions of the same command can therefore both compute holdout results. The concurrency test at `tests/test_benchmark.py:857` assigns different code hashes, avoiding this case. I reproduced the incorrect `resumed` result using an existing-reservation fixture. Hold a process lock through computation and completion; test simultaneous readers with identical metadata.

2. **Plan adherence / Test coverage — causes Fail.** `src/hearsay/fusion_v2.py:266` verifies the frozen spec’s hash but does not compare candidate definitions against their original declaration. `committed_blob_sha` only requires the current candidate file to be committed. Changing and committing C1 weights therefore passes these guards while the original spec remains unchanged. I confirmed that modified C1 weights pass `check_candidates`. Pin the original candidate artifact and explicitly validate authorized amendments; test changes to weights, bands, candidate membership, and E settings.

3. **Review compliance / Plan adherence — causes Fail.** Plan-review finding 1 required fully nested base-model provenance or an exploratory comparison with confirmatory WIN/follow-on decisions blocked. The implementation retains second-order stacking leakage and `followon_allowed` at `src/hearsay/fusion_v2.py:252` accepts `"WIN (provisional)"`. Disclosure and a sealed-set adoption gate do not implement the requested follow-on restriction. Block that decision until independently validated, or generate correctly nested base scores. Existing speaker-disjointness limitations are disclosed and are not treated as newly introduced regressions.

4. **Plan adherence / Test coverage — causes Fail.** The run spec’s Results section explicitly leaves Step 6 and four S8 tests unfinished. Consequently, the required cohort coverage and “all four candidates have full readouts” acceptance criteria are unmet. The ask-first compute restriction explains the stop, but does not make the change complete. Finish the authorized cohort work and tests, or obtain an explicit scope amendment before claiming completed acceptance.

5. **Documentation — causes Fail.** The run spec commits to updating `docs/code-map.md` if it lists scripts; it does. The new benchmark and fusion entry points are absent, and Results explicitly records the update as unfinished. Under the requested grading rule, this is a missing committed doc-impact item. Resolve the ownership restriction and add the entries, or explicitly amend that commitment.

6. **Plan adherence — contributes to Fail.** Both reports disclose that the holdout cohort was evaluated outside the ledger before the correction, including during a dry run. Current enforcement cannot retroactively establish read-once execution. Preserve that disclosure and describe this run’s historical exception separately from guarantees for future runs.

7. **Regression check — limits grade to Acceptable; not a demonstrated regression.** The targeted pytest invocation produced 133 fixture-setup errors because no temporary directory was writable. Independent direct invocation passed all seven reference checks, shipped-TSV parity, and deterministic rerun checks; Ruff also passed. A full writable-environment suite result is still needed to verify the 847-pass regression floor. The missing sealed population is documented as the submission blocker; no new TSV is required for this audit.
