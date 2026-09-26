# Handoff — post-draft gate lane: manifest, stricter gate, the sweep, the ratification packet (Sat Sep 26, 2026, ~17:25; revised after the Fable second opinion)

**Purpose of this chat:** be the one place where tonight's candidates are judged. Freeze the candidate manifest and the gate **before any new candidate number exists**, make "sleep equals ship" true tonight, extend the pre-declared sweep so each candidate enters its fixed seat, and at ~19:30 hand Nathan one locked table with a ratify-or-keep recommendation. Nothing ships without his word; the shipped TSV is immutable throughout.

## Context

- NSA's draft review: **minDCF 0.0733, EER 3.534%** on the shipped file `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv` (sha256 prefix `fb783076`). Decoded per the pre-declared table: our direction, ship unchanged. One team is ahead at 0.0584 / EER 2.5% (0.95 of 60 points; matching it means ~2 false alarms or ~7 misses under the brief's cost). Records: `docs/reports/2026-09-26_sponsor-questions.md`, `submissions/log.csv` (15:40 row).
- Two outside opinions agree on the shape of tonight (`docs/consults/2026-09-26_post-draft-review_RESPONSE.md`, read "What to do" and **"Round 2"**): no fusion redesign; a few bounded candidates under a gate stricter than the standing rule; every candidate must improve under **both** costs because NSA's convention is unknowable from the pair; stop by ~21:30. The seven-rule gate is in that file; copy it verbatim into the manifest.
- **Candidates and their seats** (one per slot, never combined; two are already measured and are listed for the record):
  - `T2`: second M3 suppression tier (quarter when margin < −6, in addition to halving below −3). **Verified inert at the metric** on every split (Round 2 table); run it anyway as a ten-second audit so the record shows the identical readouts.
  - `W4`: **A3 (0.4, 0.2, 0.4) + E**, the M5 weight raised from 0.2 to 0.4, same M3 step and Platt. **Post hoc**: its numbers were computed by the Fable agent and re-verified by the consult chat before this manifest exists (inner 0.1373 / 0.3366, holdout 0.0065 / 0.0088, In-the-Wild 0.1997 / 0.2075, test Spearman 0.970 vs shipped). It passes the standing four-condition rule and **fails the stricter gate's rule 3** (inner must improve ≥ 0.010 with no sponsor-cost regression; it worsens by 0.002 brief and 0.037 averse). Put it in the manifest with that disclosure, and get Nathan's decision on which gate governs it **before** the packet, never after.
  - `P_wl`: `0.4·rank(M1b v3) + 0.2·rank(wavlm_l) + 0.2·rank(handcrafted v5) + 0.2·rank(M5)`, then the unchanged M3 step. Column from the heads chat (`outputs/detector_scores/wavlm_l.csv`), due 18:50. The refit-seat variant (`--refit-m1b`) is not run tonight.
  - `H_noise`: the shipped weights with `handcrafted_v6` in the handcrafted seat. Only if the CPU chat produces it (it is re-scoped to write-ups; both opinions put this candidate's test effect at zero). If absent, mark "not run".
- **Sweep pattern to extend:** `scripts/fuse_sweep_m5.py`. Its CURRENT row for the shipped rule must reproduce `outputs/fusion/sweep_m5_report.json`: inner 0.1351, holdout 0.0065 / 0.0087, In-the-Wild 0.228 / 0.2385, P_FA / P_miss at the inner threshold 1.3% / 12.2%, holdout argmin [0, 13].

## Working branch / worktree

`main` in `~/Projects/hearsay`, shared. Your files: `scripts/fuse_sweep_v3.py`, `tests/test_fuse_sweep_v3.py`, `docs/reports/2026-09-26_post-draft-manifest.md` (commit it before any new candidate export exists), `docs/reports/2026-09-26_post-draft-sweep.md` (results and packet), `outputs/fusion/sweep_v3_report.json`, one log row for the final-name copy, and only on Nathan's word `models/fusion_v3/constants.json` (a new directory; never overwrite `models/fusion_v2/`). Stage only those; never `git add -A`; do not push; the oversight chat pushes on Nathan's standing instruction.

## Environment / setup

```bash
cd ~/Projects/hearsay && uv sync
shasum -a 256 submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv   # must start fb783076
```

## What to do next

1. **17:25–17:45, sleep equals ship.** Copy the shipped TSV to `submissions/CrossExam_predictions.tsv` (a copy, never a rename; `write_submission` semantics: no overwrite), record its sha256 in a `submissions/log.csv` note row marked "FINAL unless replaced by a ratified candidate", and tell Nathan and the oversight chat. If everyone falls asleep, this is the file that goes to NSA.
2. **17:45–18:05, the manifest.** `docs/reports/2026-09-26_post-draft-manifest.md`: the four candidates exactly as above with their seats and sources, the post-hoc disclosure for `W4`, the seven-rule gate verbatim, the readouts (inner / holdout / In-the-Wild under both costs; holdout argmin [#FA, #miss]; P_FA / P_miss at the inner threshold; Spearman of test scores vs the shipped file; count of 0.5 crossings for disclosure only; each candidate's mechanism diagnostic: `P_wl` must flag a meaningful share of the shipped rule's 158 In-the-Wild misses, `H_noise` must recover the 20 dB lane, `W4` has none), and the decision rule (if two pass, ship neither unless one Pareto-dominates all six split × cost cells; if none passes, KEEP). Commit it. Nathan approves the manifest and rules on which gate governs `W4` (recommendation: the stricter gate, so `W4` is KEEP by construction unless he says the standing rule applies).
3. **18:05–18:40, the sweep script.** `scripts/fuse_sweep_v3.py`: loads the shipped columns plus any candidate export that exists (a missing column marks its candidate "not run", never an error); builds CURRENT and each candidate on identical row sets; prints one table; applies the gate in code; writes `outputs/fusion/sweep_v3_report.json`. **Cost-function check (a named failure mode):** a test asserts the brief cost is `min_cost(y, s, 0.3)` with `C_FA = 4` (9.33·P_FA + P_miss) and the averse cost is `min_cost(y, s, 0.5, c_fa=1, c_miss=4)`. Self-check test: with no new candidates, CURRENT reproduces the numbers above to four decimals and `T2` reproduces them identically. `--write` refused unless `--ratified-by nathan`, then writes `models/fusion_v3/constants.json` in the `fusion_v2` shape plus any new column's model identity.
4. **18:50–19:30, the locked table.** When the heads chat delivers (or the 18:50 stop passes), run the sweep once and write `docs/reports/2026-09-26_post-draft-sweep.md`: table, gate verdict per candidate, diagnostics, Spearman and crossing counts, one recommendation. No iteration on weights or thresholds after seeing results. **Never inspect which test files cross 0.5 to decide anything**: that is selection on the test set.
5. **19:30–20:00, Nathan's window (30 min hard cap).** Present the packet. If he ratifies one candidate, integrate (step 6); if not, record KEEP in the sweep report and a log note row, confirm the step-1 file is the final one, and stop.
6. **If ratified (20:00–21:30):** the runner must reproduce the new ordering from audio. `W4` is a constants change only (`fuse_sweep_m5.py` grid → `--write` into `fusion_v3`), but **the full 1,671-row parity must be re-run**: M5's CPU-vs-A100 logit gap (5 rows already over the 0.02 mark) scales with its weight; accept only Spearman ≥ 0.999 and max |Δp| ≤ 1e-3, and compare exact sorted file IDs for near-ties. `P_wl` needs a second backbone in `hearsay.pipeline.Models` (~1.3 GB, about +0.5 s per file); if it cannot be reproduced end to end by 21:30 it does not ship, whatever the table says. `H_noise` is a bundle swap. Then a new TSV under `submissions/` (both polarities), a log row, README numbers to the README chat, and a new final-name copy replacing step 1's with a new log row. Never overwrite the shipped TSV.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test: `uv run pytest -q` → all passing (510 at cc6670c with data); `uv run ruff check .` → clean.
- Self-check: `uv run python scripts/fuse_sweep_v3.py` with no new exports → CURRENT = inner 0.1351, holdout 0.0065 / 0.0087, ITW 0.228 / 0.2385; `T2` identical; `W4` = inner 0.1373 / 0.3366, holdout 0.0065 / 0.0088, ITW 0.1997 / 0.2075 (the consult chat's verification, `scratchpad/verify_w04.py` logic: ranks against inner-OOF references, E step, `fs.brief` / `fs.averse`).
- Baseline hash `fb783076…`; any change to that file is a bug. The final-name copy must hash identically until a candidate is ratified.
- At-risk: `outputs/fusion/sweep_v3_report.json` (gitignored; copy the table into the sweep report), the other chats' candidate exports (gitignored; NOT archived; copy their key numbers into your report). The manifest and the sweep report are tracked.
- In flight: heads chat (GPU; `wavlm_l` by 18:50), CPU chat (negative-result write-ups; `handcrafted_v6` only if idle), README chat and docs-site lane (untouched by this work).

## Analytical notes

- Per-file worth on the test set under the brief cost: one false alarm ≈ 0.008 minDCF, one miss ≈ 0.002 (inverted under the sponsor's cost: 0.00085 / 0.008); improvements under 0.008 are one-file quanta. Holdout noise ±0.07–0.10; a "0.0065 → 0.0060" is not evidence and must not appear as such.
- The cost convention is unknowable from (0.0733, 3.53%): the sponsor's edited scorer puts its 4× on accepted spoofs; NSA's spoken rule puts it on false alarms. Hence "improve under both costs".
- Premise corrections both opinions made: M5 is worse on short clips (no short-clip expert); order-preserving calibration cannot move minDCF; M3 in suppression can only move files toward real, so nothing on the M3 step can shrink a miss tail; the handcrafted column's admissible stacker weight is −0.04 and its test-set rank correlation with M1b is 0.21 (it reads the envelope shift; it is the readable column, not a vote).
- Winner's curse: four candidates against one gate is the cap; do not add a fifth after seeing results. Ratification bandwidth is the binding constraint: one packet, one decision, then stop.

## Pointers

- `docs/consults/2026-09-26_post-draft-review_RESPONSE.md` (plan, gate, Round 2), `…_RESPONSE_fable.md` (the measured ceilings: reweighting, stacker, M3 threshold and promotion), `…_CONSULTATION.md`; `docs/reports/2026-09-26_fusion-sweep-predeclared.md` (how earlier sweeps and addenda were written); `scripts/fuse_sweep.py`, `scripts/fuse_sweep_m5.py`; `models/fusion_v2/constants.json`; `src/hearsay/pipeline.py`; `docs/reports/2026-09-26_runner-docker.md` (parity procedure); `docs/handoffs/2026-09-26_m4-fusion-handoff.md`.
