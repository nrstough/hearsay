### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | Completeness safeguard and explicit reporting requirements remain incomplete. |
| Scope discipline | Excellent | Changes confined to diagnostics, tests and documentation; deviations recorded. |
| Test coverage | Fail | Tests miss incomplete per-model cohorts that still produce “kept.” |
| Review compliance | Fail | Adopted completeness and invalid-input protections are only partially implemented. |
| Freeze integrity | Acceptable | Skipped: no P1/P2/P3 hashes present. |
| Regression check | Acceptable | No change-attributable regression established; full rerun blocked by sandbox restrictions. Diagnostic-only TSV/log exemption recorded. |
| Documentation | Fail | Required documentation omissions and an incorrect explanation of the verdict trigger. |
| **Overall** | **Fail** | |

### Commentary

1. **Plan adherence / Review compliance / Test coverage — causes Fail.** In [m3_probes.py:315](/Users/nathanstough/Projects/hearsay/scripts/m3_probes.py:315), paired-cohort completeness checks only M1b. Other models’ missing scores are silently excluded from metrics. Reproduction: replacing one cached Spectra score with NaN still yields `n_paired=500` and verdict `"kept"`. Require finite scores across every required model and perturbation before counting complete clips; test this through readout → verdict. The actual saved cohorts contain no missing scores, so this finding does **not** invalidate their recorded verdict.

2. **Review compliance — contributes to Fail.** Boolean rejection covers step effects but not the scalar inputs in [m3_probes.py:136](/Users/nathanstough/Projects/hearsay/scripts/m3_probes.py:136). Replacing M3’s mean ΔAUC with `False` still returns `"kept"`, despite the recorded round-two fix. Reject booleans consistently and test every verdict input.

3. **Plan adherence / Documentation — causes Fail under the explicit-deliverable rule.** The [report:40](/Users/nathanstough/Projects/hearsay/docs/reports/2026-09-26_channel-robustness.md:40) omits the descriptive mean-posterior interval required by AC1; the saved JSON contains **0.5493–0.5786**. Also, [architecture.md:254](/Users/nathanstough/Projects/hearsay/docs/architecture.md:254) lacks the conditionally promised shortcut-ledger update for the newly measured double-filter stopband fingerprint. Add the interval and explain why band matching alone did not remove that cue.

4. **Documentation — contributes to Fail.** The [report:104](/Users/nathanstough/Projects/hearsay/docs/reports/2026-09-26_channel-robustness.md:104) says clean AUC 1.000 leaves “almost no room to fall,” making trigger (a) barely possible. This reverses the mathematics: AUC 1.000 has maximum downward room. Explain instead that rank movement can preserve class separation; remove the unsupported claim about the trigger.

5. **Test coverage / Regression check — verification limitation, no additional downgrade.** Independently ran **30 channel tests** and **52 documentation tests**, all passing; changed-file lint passes. Full pytest could not complete because the read-only sandbox blocks cache/temp creation. Repository lint reproduces the three unrelated `analyzer.py` findings already disclosed. The recorded **552-pass** full run predates the critique fixes, so it is not final-revision verification.
