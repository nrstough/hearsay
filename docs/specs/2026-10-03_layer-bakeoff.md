# Layer bake-off: confirming XLS-R L7 + L20 (pre-declared)

Sat Oct 3, 2026. v2 chunks V2-20 and V2-21 (`docs/prod-chunks.md`).

- **Follows from:** `docs/reports/2026-10-03_layer-diversity.md`. There, an equal-weight rank blend of XLS-R layers 7 and 20 beat layer 7 alone on inner OOF (0.255 vs 0.265) and on In-the-Wild (0.267 vs 0.342).
- **Purpose:** this spec fixes the candidates, the tests and the pass rules before any of the new readouts exist.
- **When it was fixed:** it was committed before the script ran; the commit is the timestamp.
- **Script:** `scripts/bakeoff_layers.py`.
- **Outputs:** `outputs/v2/bakeoff/`.
- **Report:** `docs/reports/2026-10-03_layer-bakeoff.md`.

## Candidates

Every probe uses the M1b recipe on the XLS-R 300M v3 embeddings:
- StandardScaler, then class-balanced logistic regression (C = 1).
- Five inner folds of `splits/nsa_folds_plus_asv19.csv`, 23,790 rows.
- A Platt map on the pooled OOF.
- A full-inner fit scores the holdout, the test set, In-the-Wild, the perturbation cohort and MLAAD.

"Rank blend" means equal-weight ranks, each taken against that layer's own inner-OOF distribution.

| ID | Candidate | Why it's here |
|---|---|---|
| P0 | L7 | The shipped M1b probe; the baseline |
| **P1** | **L7 + L20 rank blend** | The lead under test |
| P2 | L7 + L21 rank blend | Neighbour check: is it "top layers" or just layer 20? |
| P3 | L7 + L18 rank blend | Neighbour check |
| P4 | L6 + L20 rank blend | Neighbour check on the mid side |
| P5 | L8 + L20 rank blend | Neighbour check on the mid side |
| P6 | L7 ‖ L20 concatenated, one logistic regression (2,048-d) | A different way to combine the same two layers |
| P7 | Stacked: logistic regression over all 25 per-layer OOF logits, nested by inner fold | "Use every layer" |

**Fusion level:**
- **F0** is the shipped rule `A3_w0.2_E`: 0.6·rank(M1b) + 0.2·rank(handcrafted v5) + 0.2·rank(M5), then the M3 suppression step.
- **F(Pk)** is the same rule with the M1b column replaced by candidate Pk's score. It is ranked against Pk's own inner OOF on the 16,142 NSA inner rows.

## Populations

| Population | Rows | Notes |
|---|---|---|
| Inner OOF | 23,790 (probe) / 16,142 NSA rows (fusion) | Pooled, plus each of the 5 folds (held-out generators) |
| Outer holdout | 3,858 | Read once by this run. Overall, per generator, per real source. |
| In-the-Wild | 3,000 (2,000 real, 1,000 fake, 54 speakers) | Plus a speaker-cluster bootstrap (1,000 resamples) of the paired gain over P0 |
| ITW perturbation cohort | 250 real + 250 fake ITW clips (seed 0) × {none, mp3 64k, noise 20 dB, speed ±2%, 1-sample shift} | The perturbation functions of `scripts/m3_probes.py`, applied to the decoded clip before `prepare_segment`. Same crops and seeds as `itw_stress_v3`, so "none" must reproduce the stored embeddings. |
| MLAAD | The 572 fakes of `outputs/channel/m3_mlaad.csv` | Miss rate at the inner threshold. Never trained on by any probe. |
| NSA test | 1,671, no labels | Share above the inner threshold; Spearman and flips against P0 |
| ITW by duration | ≤ 4 s vs > 4 s | Readout only |

The holdout perturbation cohort (`outputs/channel/m3_perturb.csv`) needs the external drive, which is not mounted. It is out of scope here, as is a noise-gated WavLM, which needs noise floors for the inner rows.

## Pass rules

All rules use normalized minDCF ("brief" cost, 9.33·P_FA + P_miss) unless marked "averse" (the sponsor code's convention, P_FA + 4·P_miss). "Inner threshold" means the candidate's own inner-OOF min-cost threshold.

A candidate **confirms** if every rule holds against P0:

| Rule | Test | Pass if |
|---|---|---|
| R1 | Inner OOF | Pooled ≤ P0 + 0.010, and no single fold worse than P0 + 0.030 |
| R2 | Outer holdout | ≤ P0 + 0.020 (the holdout's σ is about 0.07, so it is a guard, not a target) |
| R3 | In-the-Wild | Gain ≥ 0.030, and the bootstrap 5th percentile of the gain > 0 |
| R4 | ITW perturbations | For each of the 5 kinds, cohort cost ≤ P0's cost on the same kind + 0.030 |
| R5 | MLAAD | Miss rate at the inner threshold ≤ P0's + 0.030 |
| R6 | NSA test | Share above the inner threshold within ±0.05 of P0's |
| R7 | ITW, averse cost | ≤ P0 + 0.010 |

**Verdicts:**
- **"Idea holds":** P1 confirms, and at least two of P2–P5 also confirm. If P1 confirms alone, the verdict is "layer-specific, treat as luck until a second wild set says otherwise".
- **Fusion level:** F(P1) vs F0 is reported on inner, holdout, In-the-Wild (both costs) and the bootstrap. It is not adopted here. Adoption into a shipped v2 rule waits for the V2-01 benchmark and a second wild population (V2-30).

Nothing under `models/`, `submissions/` or `splits/` is written. v1 is unchanged.
