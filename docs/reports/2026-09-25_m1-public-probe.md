# M1 on public data: frozen XLS-R 300M + logistic probe (Fri Sep 25, 21:09)

This rung runs before the NSA data arrives, to shake down the M1 pipeline end to end (`scripts/extract_embeddings.py` → `scripts/train_probe.py`). **It is not an NSA number.**

**Data.** ASVspoof 2019 LA, class-balanced samples (`outputs/manifests/`, built from the protocol files):
- Train: `asv19_train_probe`, 5,160 clips. All 2,580 bona fide plus 430 per attack for A01–A06.
- Validation: `asv19_eval_probe`, 2,002 clips. 1,001 bona fide plus 77 per attack for A07–A19. These attacks are unseen in train (A16 and A19 reuse A04 and A06 by design).

**Features.** 4 s windows (50% hop, at most 4 per clip). Mean-pool each of the 25 hidden layers, then average over windows. Extraction takes 0.198 s per clip on the M3 Pro (MPS).

**Probe.** StandardScaler plus class-balanced logistic regression (C = 1).
- Layer chosen by generator-grouped 5-fold CV minDCF on train only (spoof grouped by attack, bona fide by speaker). Layer 7 won with CV minDCF 0.180 and EER 8.0%.
- Platt map fit on out-of-fold scores.

**Held-out result** (scored once; normalized minDCF, C_FA = 4, π = 0.5):

| | minDCF | EER | AUC |
|---|---|---|---|
| A07–A19 pooled | 0.023 | 0.6% | 0.9998 |
| hardest: A10 | 0.067 | 3.6% | |
| A15 | 0.028 | 1.0% | |
| A12 | 0.013 | 0.2% | |

All other attacks score 0.

**Caveats**
- **Silence shortcut.** ASVspoof 2019 has a known silence shortcut. The dry-run inventory gave leading-silence AUC 0.295 and voiced-fraction AUC 0.684, so the probe may partly key on silence. The NSA set probably lacks this shortcut; expect worse numbers there.
- **CV vs. held-out.** CV (leave-attack-out within A01–A06, with only 5 attacks in each training fold) is far harsher than the held-out result. Neither predicts NSA performance.
- **Next.** Rerun the same pipeline on the NSA training data with a generator- or speaker-grouped split.

**Model.** `models/m1_wav2vec2-xls-r-300m_L7_20260925-2109/` (gitignored; `meta.json` has per-layer CV and per-attack results).
