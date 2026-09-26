# Draft-review reaction branches (pre-computed Sat Sep 26, ~09:30)

**Sent for review:** `submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv` (sha256 `096f3f0c…`), copied as `~/Downloads/HEARSAY_predictions.tsv`.
- **Rule:** E on A α 0.2. M1b v3 80% + handcrafted v5 20% in rank space, with M3 used only to suppress false alarms. Frozen by Nathan at 09:25.
- **Ask NSA for:** minDCF, P_FA, P_miss and EER.

Every branch is already a logged TSV in 1,671-row template order, so reacting to NSA's number is a file copy.

| NSA's returned minDCF | Meaning | Final TSV to submit (copy, rename to `TeamName_predictions.tsv`) |
|---|---|---|
| **0.00–0.20** | Our score direction; the test set behaves like our holdout. | `submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv` (unchanged). If there is time before 22:00, revisit the A3 candidate (see below). |
| **0.20–0.45** | Our direction; the test set is more like In-the-Wild or out of family. | The same file. "Most FA-robust" means lowest ITW brief-cost minDCF, which is the false-alarm-weighted, threshold-swept metric the frozen rule was chosen on. It already has the best ITW brief (0.260) among ratified rules. |
| **0.45–0.90** | Ambiguous. Do **not** flip. | `submissions/20260926-0928_BRANCH_C_M1b_alone_our_direction.tsv` (M1b v3 alone, same 0.001 + 0.999·p map). |
| **0.95–1.00** | NSA's code is reading our scores inverted. | `submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_FLIPPED_only_if_NSA_scores_inverted.tsv`. |

**If NSA returns EER instead:** 2–5% means our direction, so use the first row. 95–98% means their code's polarity, so use the last row.

**Also pre-computed, not tied to a band:**
- `…BRANCH_C_M1b_alone_FLIPPED.tsv`, for the case where NSA's number points to inversion and M1b alone is still preferred.
- `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_*.tsv`: the A3 candidate (positive result, not shipped).

All of these files are archived at `/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/submissions/`.

**Docker / runner:** unchanged. It points at `models/fusion_v1/constants.json` (E on A α 0.2). The flip branches only change the TSV; if NSA confirms inversion, the runner would need its own `--flip` handling.
