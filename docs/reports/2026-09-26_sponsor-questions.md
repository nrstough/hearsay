
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

**Sent:** Sat 12:30 by Discord DM (team Cross Exam), superseding the 08:13 payload above: `CrossExam_predictions.tsv` = `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv` (rule A3 w0.2 + E, sha256 `fb783076…`, `submissions/log.csv` row 12:30:37).

**Returned (Sat, ~15:30, relayed by Nathan): minDCF 0.0733, EER 3.534%.** No P_FA, P_miss, threshold, cost weighting or direction convention was given.

- **Decoding:** 0.0733 is in the 0.00–0.20 band, so the shipped rule stays; EER 3.5% (not 96.5%) confirms 1.0 = synthetic is being read as intended. No flip, no fallback.
- **Against our proxies:** holdout 0.0065, In-the-Wild 0.228; the test set sits at 11× the holdout and a third of In-the-Wild, next to M1b v3 alone on the holdout (0.072).
- **Field (draft review, Sat ~16:15):** one team ahead at minDCF 0.0584 / EER 2.5%; under points = (1 − ours)/(1 − best) × 60 that is a 0.95-point gap on the 60-point detection score.
- **Status:** the number is the fixed baseline for any further work before the Sunday 05:00 freeze. The follow-up consult on what to build next is `docs/consults/2026-09-26_post-draft-review_CONSULTATION.md`.
