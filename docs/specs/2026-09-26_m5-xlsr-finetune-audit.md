### Scorecard

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Plan adherence | Fail | Cleanup can abandon a running box; assembly can select a diagnostic run instead of the final model. |
| Scope discipline | Excellent | M5 scope and documented credential exception respected. |
| Test coverage | Fail | Missing regression coverage for both failure paths below. |
| Review compliance | Fail | The launcher fix addresses round 3, but the broader cleanup guarantee remains incomplete. |
| Freeze integrity | Acceptable | No P1/P2/P3 hashes found; fold-file SHA matches its pin. |
| Regression check | Fail | The added resume probe changes which fold-4 run the assembler selects. Existing delivered scores remain intact. |
| Documentation | Acceptable | Deviations disclosed; minor spend figures remain stale. |
| **Overall** | **Fail** | Two implementation defects remain. |

### Commentary

1. **Plan adherence / Regression check / Test coverage — causes Fail.**  
   [The assembler selects runs by modification time](/Users/nathanstough/Projects/hearsay/scripts/m5_assemble.py:58). Executing its selection function against the current artifacts, even with `train_top=0`, selects **`outputs/m5_runs/resume_probe` (600 steps)** for fold 4 instead of the delivered **`abl_cons/fold4_frozen` (2,500 steps)**. Rerunning assembly would replace that fold’s scores in the canonical export. Pin the six final run directories explicitly, validate their configurations and training budgets, and test that newer diagnostic runs cannot replace them.

2. **Plan adherence / Review compliance / Test coverage — causes Fail.**  
   [The reaper writes `DESTROYED` unconditionally after `kill_`](/Users/nathanstough/Projects/hearsay/scripts/cloud/reaper.sh:45), even when the Vast destroy command returns an error. Subsequent polls skip that instance, including at the deadline, leaving it potentially billable indefinitely. Record destruction only after confirming the instance is absent; retain failed attempts for retries. Add a mocked destroy-failure-then-success test. The existing cloud test checks branch strings, not this behavior.

3. **Test coverage / Regression check — verification limitation; no additional downgrade.**  
   Lint passed. Eight cloud checks passed; another encountered a sandbox temporary-directory error. `uv run pytest` was blocked by read-only cache permissions, so the recorded full-suite result was not independently reproduced. Independently verified score counts, unique paths, fold assignments, test order, finite values, and checkpoint hashes. Saved evidence also contains the 300→600-step resume and passing parity results.

4. **Documentation — limits grade to Acceptable.**  
   `CLAUDE.md` still reports $5.09, while the report and ledger total $5.42 including resume probes; model metadata also retains $5.09. Update these figures or label them “through final training.” This is low-impact accounting staleness, not a Documentation failure.
