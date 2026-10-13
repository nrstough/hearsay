1. **Critical — C2 is not fully nested at the base-model level.** Excluding folds from rank references and NNLR fitting does not remove their influence from existing OOF exports: training-row scores were produced by base models trained on the held-out fusion fold. The research report explicitly identifies this second-order stacking leakage. Generate base scores with the outer fold excluded throughout training, or label this an exploratory fusion comparison and block confirmatory WIN/follow-on decisions pending independent validation. Test training provenance, not just arrays passed to NNLR.

2. **Critical — The cohort writer and reader have incompatible contracts.** Step 6 writes multiple columns per `(path, kind)` using a `column` field, while Step 1 and the published interface require unique `(path, kind)` keys. Specify either separate files per detector or an explicit long-format adapter keyed by `(path, kind, column)`. Test the actual writer output through the benchmark reader, including missing detector/kind combinations.

3. **Critical — The holdout ledger does not specify atomic reservation across processes/worktrees.** Checking two JSONL files and then appending permits concurrent readers to reserve the same key; “exclusive create” does not explain how multiple entries work in an existing append-only file. Use a shared lock or atomic per-key reservation, define recovery when only one mirror updates, and test simultaneous attempts from separate processes.

4. **Critical — Registry regeneration conflicts with read-once holdout semantics.** Step 3 requires regenerating Part 1 scorecards under v1.1, but K8 includes the benchmark version in the ledger key. This authorizes another baseline holdout read merely because cohort files arrived. Reuse persisted holdout results when rules and relevant inputs are unchanged; distinguish report regeneration from a new evaluation and test that version-only changes cannot bypass the ledger.

5. **Critical — The α-selection tests contradict the declared algorithm.** K3 selects α on the pre-E base, while Step 10 requires `test_alpha_selected_on_package05_with_e`. Resolve this before implementation. Add a fixture where pre-E and post-E selection choose different α values, and assert the declared pre-E choice alongside Spectra invariance.

6. **Critical — The required submission endpoint is missing.** The plan ends with JSON reports and `experiments/log.csv`, without a new TSV, `submissions/log.csv` row, or an explicit submission blocker. Since adoption awaits a sealed population, explicitly name that blocker and verify the existing TSV remains reproducible; otherwise add a uniquely named TSV through `hearsay.submission.write_submission` and the corresponding submission-log row.

7. **Suggestion — Report the existing speaker-disjointness limitations explicitly.** `scripts/make_folds.py` splits LJ by chapter and documents shared target voices across generators. Treating LJ as one bootstrap cluster does not make validation speaker-held-out. Carry these limitations into every benchmark report and avoid describing the folds as wholly speaker-disjoint.

8. **Suggestion — Pin verdict arithmetic at the declared precision.** Rounding inputs to four decimals and then subtracting binary floats can misclassify an exact 0.05 boundary. Compare integer ten-thousandths or decimal values, with equality tests for every tie band and the inclusive f4 guard.

9. **Suggestion — Resolve the time-box conflict explicitly.** The frozen spec and current `CLAUDE.md` authorize no v2 time box, but this review request asks for a few-hour limit and a stop point. State which requirement governs execution; if retaining the override, cite it directly in the plan and preserve end-of-part reporting.

10. **Suggestion — Make the deferred robustness limits visible in the verdict.** Telephony is deferred and physical replay is untested; the proposed cohorts do not establish either capability. Add explicit “not evaluated” entries and preserve the duration, level, silence, container, and single-speaker shortcut limitations alongside the numerical verdict.

---

## How the findings were addressed (Oct 11)

| # | Finding | Response |
|---|---|---|
| 1 | Second-order stacking leakage | Labelled, not fixed: a C2 WIN is marked provisional (the leakage favours fitted candidates only; TIE/LOSE are robust). Full fix needs retraining every base model per outer fold (M5 on GPU); out of scope. Adoption already needs the sealed population. |
| 2 | Cohort writer/reader contracts | One file per (cohort, column) under `outputs/v2/benchmark/cohorts/`, keyed (path, kind); writer output tested through the reader. |
| 3 | Ledger not atomic | Per-key `O_CREAT|O_EXCL` reservation files in the shared `outputs/` dir are the authority; JSONL copies rebuilt from them; concurrent-process test. |
| 4 | Version bump mints a read | Ledger key excludes the benchmark version; regeneration reuses the persisted readout; test added. |
| 5 | α test contradicts K3 | Renamed to `test_alpha_selected_on_package05_pre_e` with a pre-E vs post-E fixture. |
| 6 | No submission endpoint | Rejected: v2 writes no TSV and never `submissions/log.csv` (handoff rule). Blocker named (sealed population); shipped-TSV reproduction test kept. |
| 7 | Speaker-disjointness limits | Limitations block in every report. |
| 8 | Boundary arithmetic | Integer ten-thousandths; exact-edge tests. |
| 9 | Time box | Nathan's no-time-box override governs; stated in the plan. |
| 10 | Not-evaluated list | Added to every report. |
