### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Excellent | Implements the governing manifest and amendments; final report records KEEP. |
| Scope discipline | Excellent | No detector-contract, scoring-path, or shipped-artifact changes. |
| Test coverage | Acceptable | Independently ran 92 tests successfully; 1 skipped, 15 filesystem-dependent cases deselected. Ruff passes. |
| Review compliance | Acceptable | Previous fixes are present; model/export identity remains an explicitly accepted limitation. |
| Freeze integrity | Acceptable | No P1/P2/P3 hashes present; original manifest text remains intact, with appended amendments. |
| Regression check | Acceptable | Data self-checks pass. Both TSV hashes and fusion_v2 constants match recorded values; no fusion_v3 exists. Full-suite rerun unavailable. |
| Documentation | Acceptable | Final outcome and WavLM disclosure updated; minor historical wording remains. |
| **Overall** | **Acceptable** | |

### Commentary

1. **Test coverage / Regression check — downgrade to Acceptable.** The read-only sandbox blocks `uv` cache writes and filesystem test fixtures. Independent verification produced **92 passed, 1 skipped, 15 deselected**, including the available data self-checks. The recorded **107 passed, 1 skipped** targeted run and earlier **722 passed** full-suite run were not independently reproduced in full.

2. **Review compliance — downgrade to Acceptable.** The supplied model directory is not cryptographically tied to the evaluated export; `precheck_write` checks for `meta.json`, whose hash is recorded only when writing. This is the explicitly accepted C6 residual risk. No candidate passes, so it does not affect this KEEP decision.

3. **Documentation — downgrade to Acceptable, cosmetic only.** The script’s opening describes H_noise as “withdrawn,” and the sweep report’s historical “Decision rule applied” paragraph still says “unless H_noise passes at ~20:30.” The opening and final Run 2 section clearly establish that H_noise failed. Update or mark those remaining sentences as historical.

4. **Regression check / Documentation — no additional downgrade.** The conditional KEEP log row is absent, but this audit occurred before the specified 19:30 ruling window. Append it after Nathan’s ruling or the applicable cutoff. The existing final TSV remains byte-identical to the shipped file.
