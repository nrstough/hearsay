
## One-time draft review

**Approved by Nathan, Sat Sep 26, ~08:25.**

- **Payload:** `submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv`, copied to `~/Downloads/HEARSAY_predictions.tsv`. The two copies are byte-identical.
- **Rule:** M1b v3 80% + handcrafted v5 20% (rank), with M3 used only for false-alarm suppression. Pre-declared in `docs/reports/2026-09-26_fusion-sweep-predeclared.md`.
- **Validation:**
  - holdout minDCF 0.014
  - In-the-Wild 0.260 (brief cost) / 0.267 (sponsor-code cost)
  - test share above 0.5: 27.4%
- **Ask NSA for:** minDCF, plus P_FA, P_miss and EER.
- **Pre-flipped copy:** `..._FLIPPED_only_if_NSA_scores_inverted.tsv`.

**Decoding the returned minDCF** (per the fusion consult response):

| Returned minDCF | Action |
|---|---|
| 0.00–0.20 | Ship unchanged. |
| 0.20–0.45 | Switch to the most FA-robust candidate. |
| 0.45–0.90 | Don't flip; fall back to M1b v3 alone. |
| 0.95–1.00 | NSA's code is reading our scores inverted: submit the pre-flipped file. |

If NSA returns EER instead, 2–5% means our direction and 95–98% means theirs.

**Sent:** _pending; Nathan sends via the NSA x HexLabs Discord DM. Record the returned numbers here._
