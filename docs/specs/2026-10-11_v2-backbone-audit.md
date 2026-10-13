### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | A3 checks a potentially different volume from the download destination; full scoring deliverables remain unfinished. |
| Scope discipline | Excellent | Committed changes stay within the lane; detector contract and submission ledger unchanged. |
| Test coverage | Fail | Scoring/overfit suites and required evaluation results are absent. |
| Review compliance | Acceptable | Loader corrections landed; scoring-related findings remain deferred with the blocked stages. |
| Freeze integrity | Acceptable | No freeze hashes present; hash verification skipped. Owner statements include both risk entries. |
| Regression check | Acceptable | Ruff passes; ledger hashes match baseline. Full pytest verification was blocked by this audit’s read-only environment. |
| Documentation | Acceptable | Disclosure and incomplete status documented; minor frozen-text inconsistency remains. |
| Test sensitivity | Excellent | Recorded q/k mutation caught on first attempt; source history contains no committed mutation. |
| **Overall** | **Fail** | |

### Commentary

1. **Plan adherence / Test coverage — causes Fail.** [v2_backbone_fetch.py:135](/Users/nathanstough/Projects/hearsay/scripts/v2_backbone_fetch.py:135) checks `HF_HOME`, but the download at line 162 supplies no `cache_dir`. Hugging Face can instead use `HF_HUB_CACHE`, bypassing both placement and capacity checks on the actual destination. A mocked invocation reproduced preflight against an external path followed by download selection of the internal cache. Resolve and validate the actual cache directory, pass it explicitly to `snapshot_download`, and test conflicting cache settings.

2. **Plan adherence / Test coverage — causes Fail for the full change under audit.** Only `tests/test_backbone_port.py` exists; the specified score and overfit suites do not. There are no four-candidate readouts, paired bootstrap results, contamination differentials, perturbation comparisons, H9 rejection demonstration, or provisional verdicts. W3 remains deferred, and three passing tests exercise test-local contracts rather than application behavior. The recorded drive blocker explains the stop but does not satisfy these acceptance criteria. Keep V2-53 open and complete the remaining implementation and evaluation before presenting the full change as complete.

3. **Documentation — limits grade to Acceptable.** [The run spec:344](/Users/nathanstough/Projects/hearsay/docs/specs/2026-10-11_v2-backbone.md:344) says frozen P1/P2 text is unchanged, but the diff from `735d0a3` shows H5 edited inside Frozen P2. The revised rule is explicitly identified and does not mislead scoring decisions, so this is not a Documentation Fail. Preserve the original frozen row and place the correction in the amendment section.

4. **Regression check — verification limitation, no Fail attributed.** Both ledger hashes match the recorded baseline, and `.venv/bin/ruff check . --no-cache` passes. Pytest could not start: `uv` required writable cache access, and direct pytest required a writable temporary directory. Consequently, the reported 76 passes and four unrelated site-test failures were not independently reproduced in this audit. The run spec names the scoring blocker, so the absent new TSV/log row does not independently fail the submission requirement.
