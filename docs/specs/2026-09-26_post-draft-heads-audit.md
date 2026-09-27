### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | D1’s fold-local calibration requirement is not implemented. |
| Scope discipline | Excellent | Changes stay within the experiment and documented follow-up fixes; shipped inference code is untouched. |
| Test coverage | Acceptable | A1–A4 and A8 independently verified; A5 logs record 426 exact matches; A6–A9 supported. A10 rerun limited by sandbox. |
| Review compliance | Acceptable | No Codex plan review was required; embedded critique fixes are present. |
| Freeze integrity | Acceptable | Skipped: no P1/P2/P3 hashes supplied. |
| Regression check | Acceptable | No demonstrated regression. Diagnostics reproduce; logged shipped TSV remains valid and matches its recorded SHA-256. |
| Documentation | Fail | The new report incorrectly describes Platt calibration as fold-local. |
| **Overall** | **Fail** | Calibration implementation and stated validation contract disagree. |

### Commentary

1. **Plan adherence — downgrade to Fail.** D1 requires StandardScaler, logistic regression and Platt calibration to be “all fold-local.” However, [train_probe.py:78](/Users/nathanstough/Projects/hearsay/scripts/train_probe.py:78) fits one Platt calibrator using **all inner-OOF labels**, and [export_probe_scores.py:56](/Users/nathanstough/Projects/hearsay/scripts/export_probe_scores.py:56) applies it back to those same inner rows. The classifier and scaler are fold-local; calibration is not. This follows the existing M1b implementation but contradicts the run’s explicit requirement. Resolve that conflict through a documented, approved recipe clarification or implement and test fold-local calibration. **Outer holdout and In-the-Wild labels remain excluded from fitting.** The positive affine calibration also preserves rankings, so this finding does not invalidate the reproduced minDCF, Spearman or threshold-selected miss counts.

2. **Documentation — downgrade to Fail.** The [report’s recipe description](/Users/nathanstough/Projects/hearsay/docs/reports/2026-09-26_post-draft-heads.md:42) repeats the unsupported “all fold-local” claim. This is a substantive validation-provenance error: a developer could incorrectly treat the calibrated inner exports as fully out-of-fold. Describe pooled inner-OOF calibration explicitly and distinguish it from untouched outer validation.

3. **Test coverage / Regression check — limited to Acceptable.** The spec records **770 passed, 1 skipped** before round-two changes and describes 22 repair tests afterward, but does not record a final full-suite result. Independent rerunning on `main`, Python **3.12.13**, was blocked: `uv` could not initialize its cache; direct pytest collection encountered filesystem/cache errors. These are environmental, not demonstrated regressions. `.venv/bin/ruff check --no-cache .` passed. Record a full-suite result for the final revision in a writable environment.
