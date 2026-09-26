# Fusion sweep: pre-declared candidates and selection rule (written before any result, Sat Sep 26 ~08:15)

Per the fusion consult response (`docs/consults/2026-09-26_fusion-strategy_RESPONSE.md`).

## Candidates

Written before running:
- **A. α-sweep (rank domain).** score = (1 − α)·rank(M1b v3) + α·rank(handcrafted v5), for α ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}. Ranks are each file's ECDF position against that detector's own inner OOF distribution.
- **B. min rule and max rule** over the two ranks.
- **C. Cascade.** M1b rank everywhere, except in the ambiguous middle band (M1b rank between the inner 40th and 90th percentiles), where it is replaced by the mean of the M1b and handcrafted ranks.
- **D. Non-negative stacker shrunk toward equal weights**, over z(M1b) and z(handcrafted). Fit on inner OOF with LJ bona fide excluded; weights clipped at ≥ 0, then 50/50 shrinkage toward equal.
- **E. M3 as false-alarm suppression only**, applied on top of the winner of A–D. Where M3's margin is below −3 ("strongly bona fide") and the base rank is above 0.5, the base rank is multiplied by 0.5. M3 is never used to raise a score, and nothing is fitted on M3.

## Readouts, for every candidate

- **Inner OOF minDCF**, brief cost (π_synth = 0.3, C_FA = 4).
- **Holdout minDCF** under both costs, plus the raw (#FA, #miss) at each candidate's holdout argmin threshold.
- **Per real source:** LJ and LibriSpeech.
- **In-the-Wild minDCF** under both costs:
  - brief cost: 9.33·P_FA + P_miss
  - miss-averse cost (the sponsor code's semantics): P_FA + 4·P_miss at π 0.5
- **In-the-Wild P_FA and P_miss** at the inner-OOF threshold.

## Selection rule (fixed now)

1. **Admissible:** inner OOF within 0.03 of the best candidate's inner OOF, **and** In-the-Wild miss-averse minDCF ≤ 0.45.
2. **Among admissible candidates**, pick the lowest In-the-Wild brief-cost minDCF. Ties within 0.01 go to the smaller α, meaning more M1b.
3. **Apply E only if** it lowers In-the-Wild brief-cost minDCF by ≥ 0.01 **and** does not raise holdout or inner OOF by more than 0.01.
4. **No iteration after the results.** If nothing is admissible, ship M1b v3 alone.
