### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | Required cohort readouts remain incomplete; new cohort inputs bypass hash pinning. |
| Scope discipline | Excellent | Changes stay within the declared implementation and documentation scope. |
| Test coverage | Fail | Full-readout acceptance remains unmet; cohort provenance enforcement lacks coverage. |
| Review compliance | Fail | Provisional-WIN follow-on gate remains unresolved. |
| Freeze integrity | Excellent | Frozen text and candidate hashes match; amendments are separate; both owner risk statements exist. |
| Regression check | Acceptable | Reference, TSV-parity and determinism assertions pass directly; ruff passes. Full pytest independently blocked by sandbox permissions. |
| Documentation | Acceptable | Scoped documentation records pending work and historical evaluation exceptions. |
| Test sensitivity | Excellent | Recorded realistic leakage mutation caught on first attempt; mutation absent from commit history. |
| **Overall** | **Fail** | |

### Commentary

1. **Plan adherence / Test coverage — causes Fail.** Required cohort evaluations remain unfinished. C1/C3 lack holdout-cohort results, and all candidates lack ITW-cohort results, as acknowledged in the run spec (`docs/specs/2026-10-11_v2-benchmark-fusion.md:301`). Waiting for the specified compute approval is appropriate, but “all four candidates have full readouts” is still unmet. Complete the approved scoring, pin its inputs, and regenerate the readouts before claiming acceptance.

2. **Plan adherence / Test coverage — causes Fail.** load_cohort_frames (`scripts/v2_benchmark.py:287`) loads every cohort CSV without requiring a registry entry or verifying its hash. Registry construction uses fixed input lists that omit these files. The fusion reservation metadata (`scripts/v2_fusion.py:208`) also omits their hashes; `cohort_sha` hashes only paths and labels. Consequently, changed cohort logits can alter ITW results without changing recorded provenance, or resume a holdout reservation despite changed inputs. Require registry pins for consumed cohort files, record their hashes in reservation metadata, and test refusal of unpinned and modified files.

3. **Review compliance — causes Fail.** followon_allowed (`src/hearsay/fusion_v2.py:252`) accepts `WIN (provisional)`, despite the critical review finding about second-order stacking leakage. The current TIE prevents execution here but does not resolve the gate. Preserve the historical pre-declaration and record a prospective amendment requiring independent validation before authorizing a follow-on.

4. **Plan adherence — contributes to Fail.** The verdict disclosures (`docs/reports/2026-10-12_v2-fusion-verdict.md:81`) establish that C2’s base-model training provenance is not fully nested and that earlier holdout-cohort evaluations occurred outside the ledger. Later locking cannot retroactively establish read-once compliance. Keep those results exploratory; confirmatory evidence requires independent validation or fully nested base-model training and a fresh, ledgered evaluation. The inherited speaker-overlap limitation is disclosed and is not a separate regression.

5. **Regression check — limited to Acceptable, not evidence of a code failure.** The spec records **995 passed, 12 skipped**. During this audit, `uv` could not initialize its cache, and direct pytest encountered temporary-directory setup errors for all 148 new tests. Direct invocation of the seven reference assertions, shipped-TSV parity, nested-reference check, and deterministic rerun assertion passed; `ruff check --no-cache .` passed. The full-suite result remains independently unverified.
