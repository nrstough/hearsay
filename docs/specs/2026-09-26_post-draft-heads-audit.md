### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Acceptable | Outputs and diagnostics match; calibration deviation explicitly reconciled with the unchanged M1b recipe. |
| Scope discipline | Excellent | Changes confined to the experiment, repair tooling, tests, and associated documentation. |
| Test coverage | Acceptable | Recorded suite: 775 passed, 1 skipped. Independent full rerun blocked by read-only environment. |
| Review compliance | Excellent | Prior findings addressed, including both architecture calibration statements. |
| Freeze integrity | Acceptable | Skipped: no P1/P2/P3 hashes supplied. |
| Regression check | Acceptable | No regression identified; diagnostics reproduce, experiment logged, shipped TSV hash unchanged. |
| Documentation | Excellent | Report, disclosure, and calibration corrections describe the completed change. |
| **Overall** | **Acceptable** | |

### Commentary

1. **Plan adherence — Acceptable downgrade.** Platt calibration uses pooled inner-OOF labels rather than being fold-local as originally specified. The post-run correction explains this inherited M1b behavior. The saved positive slope preserves ranking metrics; outer-holdout, stress, and test labels remain excluded from fitting.

2. **Test coverage / Regression check — Acceptable verification limitation.** The recorded full-suite result could not be independently repeated: uv cache access and pytest temporary/cache writes were blocked by the read-only sandbox, with four collection errors. Using Python 3.12.13 directly, the documentation tests passed **52/52**, and `ruff check . --no-cache` passed. Independent checks reproduced the reported diagnostics, export counts and path sets, finite scores, and identical shipped/final TSV hashes. These environment errors are not evidence of a change-induced regression.
