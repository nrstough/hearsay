# Handoff: the test-set TSV, the draft review and the final submission (Sat Sep 26, 2026, ~10:00)

**Purpose of this chat:** own the one file NSA grades. Send the draft for the one-time review, decode the number that comes back, pick the final TSV by the pre-declared table, run the preflight, and hand Nathan the file to DM before Sunday 08:00. This chat copies and checks files; it does not change the fusion rule (that is the M4 lane, `docs/handoffs/2026-09-26_m4-fusion-handoff.md`) and it never overwrites a submitted TSV.

Read first: `docs/reports/2026-09-26_draft-review-branches.md` (the decoding table with a file per band), then the "Score direction" part of `docs/plan.md`, then this file.

## Context

- **What NSA grades (60% of the score):** minDCF on 1,671 unlabeled test WAVs, false alarms costing 4× a miss, about 70% real. Only the ranking of our scores matters; any monotone rescaling changes nothing.
- **What we know about our error:** holdout 0.014 (two unseen generators + 26 unseen real speakers), In-the-Wild 0.260 (3,000 real-world clips, never trained on), inner 0.140. The test number is unknown; NSA provided no labels. The draft review is the only labeled measurement we will ever get on the test set: one file, one time, one number back.
- **The draft is the shipped rule, not an experiment.** Since ~12:05 that rule is **A3 w0.2 + E** (ratified by Nathan once NSA said the image is not required): rank blend 0.6·M1b v3 + 0.2·handcrafted v5 + 0.2·M5, Spectra-AASIST used only to suppress false alarms, Platt at the 0.3 prior into [0.001, 1], non-speech and decode failures pinned below 0.001. Constants `models/fusion_v2/constants.json`; the runner reproduces the file live from audio at Spearman 1.0 / max 9.3e-4 on all 1,671 rows. The previous rule `e_on_a` (fusion_v1) is the fallback; the evidence image `hearsay:20260926-0916` reproduces that one.
- **Score direction:** we submit 1.0 = synthetic, as the brief says. NSA's scoring code is ASVspoof5 code that treats a higher score as bona fide. The draft's returned number tells us which one they actually use; the decoding table below turns that number into an action. The pinned block stays at the bottom in both polarities, by design (consult item 5): that is where the 4× error is avoided under either reading.
- **The M5 candidate (A3 w0.2 + E)** beat the frozen rule on every readout but is shelved until the draft number is back; artifacts under `submissions/20260926-0914_*CANDIDATE*`, `outputs/fusion/fusion_v2_candidate/` and the runner's `models/fusion_v2/constants.json`. The runner can already execute it (8316e50, opt-in via `--fusion`; default and the draft file unchanged, reproduced to 1.1e-16), so reopening it is Nathan's call and, now that the Docker image is not required (NSA, Sat ~12:00), costs nothing but a file copy and a README line. Nathan is deciding at ~12:15 whether the draft itself goes out as A3.

## The file to send

| | |
|---|---|
| Logged file | `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv` (A3 w0.2 + E; supersedes the 08:13 `e_on_a` file, which stays logged as the fallback) |
| sha256 (first 16) | `fb7830762691d04d` |
| Copy in Downloads | `~/Downloads/CrossExam_predictions.tsv`, verified byte-identical at 12:03 (the earlier `e_on_a` copy is beside it as `CrossExam_predictions_e_on_a_superseded.tsv`; do not send that one) |
| Rows | 1,671 + header, in the template's order (`data/nsa/HearsayScoreKey4TeamX.tsv`) |
| Header | `filename<TAB>cm-score` |
| Range | min 0.0014, max 0.9996, no NaN, 0 files gated |
| Share ≥ 0.5 | 27.3%; a smoke alarm, not a selection signal |

**Team name: Cross Exam.** The file goes out as **`CrossExam_predictions.tsv`** (the brief's pattern is `teamName_predictions.tsv`). Made at 11:20: `~/Downloads/CrossExam_predictions.tsv`, byte-identical to the logged 08:13 file (sha256 `096f3f0c9e3cfa9a…`), checker output `OK rows 1671 min 0.0012 max 0.9997 share>=0.5 0.274`. The runner and the image take the name as `--team CrossExam` / `HEARSAY_TEAM=CrossExam` (their default is `HEARSAY`, which is wrong for the deliverable).

**How to send:** as a DM in the NSA × HexLabs Discord channel (kickoff notes, `docs/reports/2026-09-25_sponsor-questions.md`). Say it is the optional draft for review. **NSA returns minDCF only** (confirmed Sat ~11:50): no EER, P_FA or P_miss, so the decoding table below is the whole decoder. Its weak spot is the 0.45–0.90 band, which one number cannot resolve; if the number lands there, ask NSA directly whether they run `calculate_metrics.py` as shipped (higher = bona fide) or invert first, and do not flip on a guess.

**Before the DM, re-check the exact bytes that leave the machine:**

```bash
cd ~/Projects/hearsay && cmp ~/Downloads/HEARSAY_predictions.tsv submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv && echo IDENTICAL
```

## When the number comes back

Copy the file for the band, rename it to `<TeamName>_predictions.tsv`, run the check below, append one log row that records the decode, tell the M4 lane and the Docker lane which branch fired, and hand the file to Nathan. Every branch is already a logged TSV; nothing is computed under time pressure.

| NSA's minDCF | Meaning | Final TSV | Then |
|---|---|---|---|
| **0.00–0.20** | our direction; test behaves like our holdout | the A3 file, unchanged | remaining hours to README and diversity |
| **0.20–0.45** | our direction; test is wild-like | the A3 file, unchanged (it has the best In-the-Wild score under both costs of every rule we have) | the channel-robustness report (16:00) becomes the lever |
| **0.45–0.90** | ambiguous; do not flip | `submissions/20260926-0928_BRANCH_C_M1b_alone_our_direction.tsv` (sha `4ae685ff9824543d1`) | ask NSA the direction question directly before doing anything else; note the rule in the README |
| **0.95–1.00** | NSA's code reads our scores inverted | `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_FLIPPED_only_if_NSA_scores_inverted.tsv` | raise it with NSA at the booth or on Discord (their own document says 1.0 = synthetic); the runner's `--flip` produces this file |

If only EER comes back: 2–5% → first row; 95–98% → last row. If a number lands on a boundary or NSA sends something else (a rank, a plot, "looks fine"), do not guess: post it to Nathan and the M4 lane verbatim and ask NSA for minDCF and EER.

**Reject any temptation to re-tune on the returned number.** It is one number on 1,671 files; the table was fixed before it existed so that it cannot be argued with afterwards.

## Sunday, before 08:00

1. Take the final file from the table (or the ratified A3 file if Nathan reopened it and the M4 lane shipped it).
2. Run the check:
   ```bash
   cd ~/Projects/hearsay && uv run python - <<'EOF'
   import pandas as pd, sys
   f = sys.argv[1] if len(sys.argv) > 1 else "submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv"
   t = pd.read_csv("data/nsa/HearsayScoreKey4TeamX.tsv", sep="\t")
   d = pd.read_csv(f, sep="\t")
   assert list(d.columns) == ["filename", "cm-score"], d.columns
   assert len(d) == 1671 and list(d.filename) == list(t.filename), "row count or order"
   s = d["cm-score"]
   assert s.notna().all() and (s >= 0).all() and (s <= 1).all(), "range"
   print("OK", f, "min %.4f max %.4f share>=0.5 %.3f" % (s.min(), s.max(), (s >= 0.5).mean()))
   EOF
   ```
3. Direction check: the file's share ≥ 0.5 should be near 0.27 in our direction and near 0.73 flipped. If that does not match the branch you chose, stop.
4. Rename the copy to `CrossExam_predictions.tsv`; `cmp` it against the logged file; Nathan DMs it with the repo link (the Docker image is optional evidence, not a deliverable).
5. Append a log row (`submissions/log.csv`: `timestamp,rung,validation_score,validation_score_clean_only,csv_path,notes`) saying "FINAL, sent HH:MM, sha …, branch …".

## Working branch / worktree

`main` in `~/Projects/hearsay`, shared by several chats. This lane owns nothing in `src/` or `scripts/`; it writes `submissions/log.csv` rows and this doc. `submissions/*.tsv` are gitignored; only `log.csv` is tracked. Stage by path; never `git add -A` (the repo root now carries a Next.js tree from the frontend push). Ask Nathan before pushing.

## Environment / setup

```bash
cd ~/Projects/hearsay && uv sync
```

Nothing else is needed for this lane. Regenerating any TSV from audio is the M4 lane's job (`scripts/run_pipeline.py`, 22 min), never this lane's.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test: `uv run pytest -q tests/test_audio_and_submission.py` (or `uv run pytest -q -k submission`) → 42 passed at 28d26fe; the writer enforces header, unique filenames, finite [0, 1], no overwrite.
- Test: `uv run pytest -q tests/test_docs_consistency.py` → 52 passed; run after editing any doc.
- Check: the pandas snippet above on whichever file is about to leave the machine; expected `OK … min 0.0012 max 0.9997 share>=0.5 0.274` for the draft.
- At-risk: `submissions/*.tsv` (gitignored). Archived to `/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/submissions/` (13 files at 09:55). The external drive is at 100%; TSVs are 57 KB each, so the archive is safe, but check `df -h` before writing anything larger there. `~/Downloads/HEARSAY_predictions.tsv` is a convenience copy, not the record.
- At-risk: `submissions/log.csv` is tracked; every send gets a row, and the row is the source of truth on numbers when docs disagree.
- In flight: the M4 lane (branches done; idle until the number), Docker (image verified; not required, lane stood down), channel robustness (report 16:00; a symmetric refit would change the exports and the M4 lane would re-run both sweeps, producing new logged TSVs that this table would then need to point at), README (20:00).

## Analytical notes

- minDCF is a ranking metric; a 0.5 cut is not the operating point (the brief's Bayes threshold is near 0.9). Do not describe "flagged" files by the 0.5 share.
- Points from the 60% weight are relative: (1 − ours) / (1 − best team) × 60. At 0.014 against a best of 0.01, ~59.8; at 0.26 against 0.05, ~46.7.
- Under NSA's inverted code the 4× lands on misses instead of false alarms; the frozen rule was also checked under that cost (0.267 In-the-Wild) so the choice does not depend on the direction.
- The test set has a uniform 7.2 kHz cutoff and one FFmpeg encoder tag on every file, which reads as one curated re-encode pipeline rather than field audio; the consult's prior is therefore the 0.00–0.20 band, but that is a guess until the number arrives.

## Pointers

- `docs/reports/2026-09-26_draft-review-branches.md`: the table, with a file per band.
- `docs/consults/2026-09-26_fusion-strategy_RESPONSE.md`, items 4 and 5: why the table looks like this and why the block stays at the bottom.
- `docs/reports/2026-09-26_fusion-sweep-predeclared.md`: how the frozen rule was chosen; M5 addendum.
- `docs/handoffs/2026-09-26_m4-fusion-handoff.md`: the fusion lane, including the shelved A3 candidate.
- `docs/nsa-challenge.md` lines 186–200: the file format; `docs/reports/2026-09-25_sponsor-questions.md`: the Discord DM channel and the kickoff answers.
- `submissions/log.csv`: every TSV ever written, with its numbers.
