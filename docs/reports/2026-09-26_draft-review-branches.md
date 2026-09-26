# Draft-review reaction branches (pre-computed Sat Sep 26, ~09:30)

**Sent for review (12:03, superseding the 09:25 plan):** `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv` (sha256 `fb783076…`), copied as `~/Downloads/CrossExam_predictions.tsv` (team Cross Exam).
- **Rule:** A3 w0.2 + E. M1b v3 60% + handcrafted v5 20% + M5 20% in rank space, with M3 used only to suppress false alarms. Ratified by Nathan at 12:20 (GO A3) after NSA said the Docker image is not required; the pre-declared rule said replace. The runner's default is `models/fusion_v2/constants.json` since 12:35 and reproduces this file (exports: 7e-16; live: Spearman 1.0, max 9.3e-4). _The 09:25 plan sent the 08:13 `e_on_a` file (`096f3f0c…`); that file is the fallback, loadable via `--fusion models/fusion_v1/constants.json`._
- **Ask NSA for:** minDCF, P_FA, P_miss and EER.

Every branch is already a logged TSV in 1,671-row template order, so reacting to NSA's number is a file copy.

| NSA's returned minDCF | Meaning | Final TSV to submit (copy, rename to `TeamName_predictions.tsv`) |
|---|---|---|
| **0.00–0.20** | Our score direction; the test set behaves like our holdout. | `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv` (the file sent at 12:03; unchanged). _Until 12:20 this row pointed at the 08:13 `e_on_a` file._ |
| **0.20–0.45** | Our direction; the test set is more like In-the-Wild or out of family. | The same A3 file. "Most FA-robust" means lowest ITW brief-cost minDCF, the false-alarm-weighted, threshold-swept metric the rules were chosen on; A3 w0.2 + E has the best ITW brief (0.228) and averse (0.239) of every ratified rule. |
| **0.45–0.90** | Ambiguous. Do **not** flip. | `submissions/20260926-0928_BRANCH_C_M1b_alone_our_direction.tsv` (M1b v3 alone, same 0.001 + 0.999·p map; unchanged). |
| **0.95–1.00** | NSA's code is reading our scores inverted. | `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_FLIPPED_only_if_NSA_scores_inverted.tsv`. |

**If NSA returns EER instead:** 2–5% means our direction, so use the first row. 95–98% means their code's polarity, so use the last row.

**Also pre-computed, not tied to a band:**
- `…BRANCH_C_M1b_alone_FLIPPED.tsv`, for the case where NSA's number points to inversion and M1b alone is still preferred.
- `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_*.tsv`: the A3 candidate (positive result, not shipped).

All of these files are archived at `/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/submissions/`.

**Docker / runner (12:35):** the runner points at `models/fusion_v2/constants.json` (A3 w0.2 + E) by default; the image (not a graded deliverable, NSA ~12:00) stays on `e_on_a` as reproducibility evidence. It already supports `--flip` (since `9614c18`: `scripts/run_pipeline.py`, `pipeline.py`), which maps 1 − p through the same determinate map and keeps the gate block at the bottom, and the Docker smoke test covers it. So an inversion verdict changes only which TSV is sent.
