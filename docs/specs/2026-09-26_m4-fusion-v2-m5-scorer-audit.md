### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | A5 forbids copying/logging the PARITY artifact when acceptance fails; both remain. |
| Scope discipline | Excellent | Capability added without switching the default or changing the detector contract. |
| Test coverage | Fail | A5’s full-set M5-logit tolerance is exceeded on five rows. |
| Review compliance | Acceptable | Plan-review fixes implemented; prior audit blockers remain unresolved below. |
| Freeze integrity | Acceptable | Skipped: no P1/P2/P3 freeze hashes found. |
| Regression check | Acceptable | Final-code suite reports 510 passed; six focused tests independently passed. Draft TSV hash matches. |
| Documentation | Excellent | Scoped docs describe the implementation, unchanged default, and outstanding deviation. M5 is already disclosed. |
| **Overall** | **Fail** | A5 remains unmet and unwaived. |

### Commentary

1. **Test coverage — downgrade to Fail.** Independently recomputing M5 live-versus-export differences across 1,671 cached rows gives **maximum 0.0350065**, with **five rows exceeding 0.02**. Probability parity and zero flips do not satisfy the separate logit gate. Meet the existing gate, or obtain Nathan’s explicit acceptance of a documented amendment; the proposed replacement thresholds are not yet approved.

2. **Plan adherence — downgrade to Fail.** [A5](/Users/nathanstough/Projects/hearsay/docs/specs/2026-09-26_m4-fusion-v2-m5-scorer.md) explicitly requires no PARITY copy or log row upon failure. The artifact and [log row](/Users/nathanstough/Projects/hearsay/submissions/log.csv) remain. Relabelling them “validation evidence” does not resolve that requirement. Pending an authorized amendment, retain evidence under `outputs/`, remove the submission-directory copy, and withdraw its log entry with an audit trail.

3. **Regression check / Test coverage — no additional downgrade.** Independent verification passed six focused tests covering constants identity, both candidate TSV polarities, v1 arithmetic, strict E boundaries, and error thresholds. The full-suite result of **510 passed** is recorded evidence, not independently rerun here. Repository lint still reports three issues solely in the unrelated `src/hearsay/analyzer.py`; these are not attributed to this change.
