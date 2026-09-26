# M1b: adding ASVspoof 2019 real speakers to the frozen XLS-R probe (Sat Sep 26, ~03:50)

**Why:** the M5 consult response (VeriLM, two experts) called real-speaker diversity the highest-value lever under C_FA = 4. One false alarm costs four misses, and about 11 false alarms decide the score.

**What changed:** ASVspoof 2019 LA train+dev bona fide was added to M1's training data.
- 5,128 clips from **40** VCTK speakers. The consult's "~100 speakers" was wrong, and the consult itself flagged this.
- Plus 2,520 A01–A06 spoof clips (420 per attack) as a channel anchor, so "VCTK sound = real" can't become a shortcut.
- The extra rows went to **inner folds only**. The outer holdout is unchanged, so the holdout numbers compare directly.

**Unchanged:**
- The input path: segment mode, silence trim, test-length crops, no tiling, z-normalization.
- The model: layer 7 (re-selected by inner CV), logistic regression.

**Files:**
- Folds: `splits/nsa_folds_plus_asv19.csv` (`scripts/extend_folds.py`).
- Models: `models/m1_wav2vec2-xls-r-300m_L7_20260926-0347` (M1, retrained for the stress readout; same numbers as `…-0302`) and `…-0348` (M1b).

**Stress set.** 3,000 In-the-Wild clips: 2,000 real from 54 speakers, 1,000 spoof. **Evaluation only, never trained on.** The readout is minDCF, plus P_FA/P_miss at the threshold that minimizes inner-fold out-of-fold DCF.

Normalized minDCF at π = 0.3, C_FA = 4:

| | M1 (NSA only) | M1b (+ASV19) |
|---|---|---|
| Inner-fold CV minDCF (the selection metric; M1b's folds contain extra rows) | 0.250 | 0.248 |
| Holdout minDCF | 0.146 | 0.079 |
| Holdout EER / AUC | 2.5% / 0.997 | 1.7% / 0.999 |
| Holdout, playht | 0.032 | 0.033 |
| Holdout, wavegrad2 | 0.214 | 0.104 |
| Holdout, bona fide LibriSpeech | 0.159 | 0.101 |
| Holdout, bona fide LJ | 0.088 | 0.047 |
| Sponsor code as-is / flipped (holdout) | 1.00 / 0.090 | 1.00 / 0.055 |
| **In-the-Wild minDCF** | **0.374** | 0.405 |
| In-the-Wild EER | 8.4% | 9.3% |
| In-the-Wild P_FA / P_miss at the inner threshold | 1.2% / 27.4% | 5.7% / 13.4% |

**Reading**
- **Holdout.** The holdout improves on every slice. But the gap (0.067) is inside the holdout's one-sigma noise (about 0.07–0.10 per the consult), and inner CV is a tie.
- **In-the-Wild.** The rank metric gets slightly worse on In-the-Wild, and the inner-fold threshold carries over worse: P_FA rises from 1.2% to 5.7%. More VCTK read speech did **not** reduce false alarms on web-sourced celebrity speech, contrary to the consult's prediction.
- **The biggest finding** is the size of the In-the-Wild gap: both models sit at 0.37–0.40, against 0.08–0.15 on the NSA-domain holdout. If NSA's real test audio includes "smartphone, telephony and field recordings" (per the brief), the holdout is optimistic, and robustness to real-world recording conditions matters more than the fusion choice.

**Decision status:** open.
- M1 remains the logged TSV.
- M1b is a candidate for the one-time draft review.
- Next levers, which the M5 fine-tune and the fusion consult should weigh:
  - augmentation that reproduces field and telephony channels on **both** classes
  - more varied bona fide sources: In-the-Wild is eval-only, so look elsewhere
  - checking whether the In-the-Wild errors are false alarms or misses by source
