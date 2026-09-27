### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Acceptable | Matches M1b recipe; deviations and pooled-calibration clarification recorded. |
| Scope discipline | Excellent | Changes confined to experiment tooling, tests, evidence and documentation. |
| Test coverage | Acceptable | A1–A4 and A8 independently verified; A5 logs record 426 exact matches; A6–A9 supported. Final suite recorded as 775 passed, 1 skipped. |
| Review compliance | Acceptable | Prior report/spec findings addressed; related architecture contradiction remains. |
| Freeze integrity | Acceptable | Skipped: no P1/P2/P3 hashes present. |
| Regression check | Acceptable | Diagnostics reproduce; lint passes; shipped TSV validates and matches its logged SHA-256. Independent pytest rerun blocked. |
| Documentation | Fail | Architecture still incorrectly guarantees fold-local Platt calibration. |
| **Overall** | **Fail** | Substantive calibration documentation contradiction remains. |

### Commentary

1. **Documentation — downgrade to Fail.** [docs/architecture.md:53](/Users/nathanstough/Projects/hearsay/docs/architecture.md:53) says every learned component, including Platt maps, is fit fold-locally. That contradicts both the corrected diagram at line 189 and the [report’s recipe](/Users/nathanstough/Projects/hearsay/docs/reports/2026-09-26_post-draft-heads.md:42): M1-family calibration uses pooled inner-OOF labels and is applied back to those rows. This concerns the exact validation-provenance correction declared in round 3, rather than unrelated historical staleness. A developer relying on the stated invariant could treat calibrated inner logits as fully out-of-fold. Update the invariant to state the pooled-calibration exception and distinguish it from untouched outer validation.

2. **Plan adherence — limited to Acceptable.** The original fold-local calibration requirement was clarified after execution to preserve the existing M1b recipe. The persisted WavLM Platt slope is positive, supporting the explanation that rankings and rank-based results remain unchanged. Outer-holdout and In-the-Wild labels are excluded from fitting. This resolves the implementation discrepancy through a documented deviation.

3. **Test coverage / Regression check — limited to Acceptable.** The final full-suite result is recorded, but could not be independently rerun here: `uv` cannot initialize its cache, and direct pytest cannot create temporary files in the read-only environment. These are environmental failures, not demonstrated regressions. Independent lint, export checks and diagnostics passed; both shipped TSV copies match the logged hash.
