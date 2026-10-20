# The closed v2 verdicts restated under package-0.3

**Date:** 2026-10-20. **Why:** rule 5 changed. The answer key reproduced NSA's returned 0.0733 under the package reading at Pspoof 0.3 (`docs/reports/2026-10-20_v2-02-error-atlas.md`, §3), and Nathan made package-0.3 the primary selection reading the same day. Three verdicts had been gated on package-0.5. This note reads their existing outputs under 0.3; **nothing is re-run and nothing is re-selected.** The sealed key is not read.

The reading: `min_cost(y, s, 0.3, c_fa=1.0, c_miss=4.0)` = P_FA + 1.714·P_miss. Package-0.5 was P_FA + 4·P_miss, so 0.3 weighs misses 2.3× less relative to false alarms. Every candidate that lost under 0.5 by missing fakes gets a smaller penalty here; the question is whether any verdict flips.

## V2-60 neural fusion (`docs/reports/2026-10-12_v2-fusion-verdict.md`)

Pooled nested inner OOF with E, 16,142 rows. The pre-declared tie band for package-0.3 was 0.03 (`benchmarks/v2_60_candidates.json`, `tie_bands`), the LOSE rule "worse than control by more than the band on any reading", WIN "better than control by more than the primary band, no reading regresses, f4 guard holds".

| Candidate | package-0.5 (declared primary) | **package-0.3** | Δ vs C0 under 0.3 | Verdict as declared | Verdict if 0.3 were primary |
|---|---|---|---|---|---|
| C0 shipped rule | 0.3390 | **0.2156** | | control | control |
| C1 equal weights | 0.4520 | 0.2684 | +0.0528 > 0.03 | LOSE | LOSE |
| C2 learned, shrunk | 0.3764 | 0.2247 | +0.0091 (inside the band, not better) | TIE | TIE |
| C3 shipped + 0.1 WavLM + 0.1 L20 | 0.3584 | 0.2318 | +0.0162 (inside the band) | LOSE (on brief, +0.0227 > 0.02) | LOSE (brief still regresses) |

No flip. C1 still loses outright, C2 still ties, C3 still loses through the brief reading, which the LOSE rule applies regardless of which reading is primary. The verdict code keeps the pre-declared reading (`hearsay.fusion_v2.PRIMARY = "package05"`, pinned to `primary_reading` in the candidates file by `test_v2_60_verdict_reading_is_the_predeclared_one_not_the_benchmark_default`), so re-running V2-60 reproduces the closed verdict rather than silently re-gating it.

## V2-33 hard-negative mining, stage 0 (`docs/reports/2026-10-13_v2-33-hardneg-stage0.md`)

A diagnostic on the legacy split, exploratory inference, nothing adopted. Its bootstrap contrasts were computed on package-0.5 only; the per-cell package-0.3 readings exist in every cell file (`outputs/v2/hardneg/cells/*.json`, `primary.package03`, NSA-only basis). Below is the **descriptive** restatement: per-stratum delta of package-0.3 against the matching base cell, 24 strata (4 layers × 2 pools × 3 C), negative = better. The paired group-bootstrap P(better) was not re-run.

| Contrast | median Δ package-0.3 | strata better | by C (median; better/8) | package-0.5 for comparison |
|---|---|---|---|---|
| hardest − none | −0.0044 | 13 / 24 | C1 +0.0054 (3/8), C8 −0.0067 (5/8), C64 −0.0050 (5/8) | −0.0056, 13/24 |
| easiest − none | **−0.0157** | **22 / 24** | C1 −0.0077 (7/8), C8 −0.0179 (8/8), C64 −0.0162 (7/8) | −0.0269, 22/24 |
| hardest − random | −0.0023 | 13 / 24 | C1 −0.0044 (5/8), C8 −0.0018 (4/8), C64 +0.0026 (4/8) | |
| easiest − random | −0.0139 | 22 / 24 | C1 −0.0113 (7/8), C8 −0.0185 (8/8), C64 −0.0108 (7/8) | |
| random − none | −0.0007 | 13 / 24 | a wash at every C | |
| focal γ=2 − none | +0.0482 | 0 / 5 | worse at every C | worse everywhere |

Same shape, smaller magnitudes: the hardest arm is a coin flip, the easiest arm is consistently a little better than base and than random, focal loss is worse. The stage 0 conclusion stands: no survivor, nothing adopted, and the lead (easiest-side weighting, the C curve) still goes to a speaker-disjoint benchmark row before it means anything.

## V2-53 AntiDeepfake backbones, zero-shot In-the-Wild readout (`docs/specs/2026-10-11_v2-backbone.md`)

The one population those models were not trained on. `outputs/v2/backbone/<model>__itw.meta.json`, 3,000 rows, "ours" input view. v1 references from `outputs/v2/benchmark/v1_scorecard.json` (ITW, report-only).

| Column | package-0.5 | **package-0.3** | brief | EER |
|---|---|---|---|---|
| xls-r-2b-anti-deepfake | 0.0417 | **0.0223** | 0.0342 | 1.10% |
| xls-r-1b-anti-deepfake | 0.0466 | 0.0280 | 0.0438 | 1.23% |
| mms-1b-anti-deepfake | 0.0680 | 0.0488 | 0.0730 | 1.88% |
| mms-300m-anti-deepfake | 0.1100 | 0.0849 | 0.1257 | 3.40% |
| spectra_aasist (v1, suppressor) | 0.1330 | 0.0682 | 0.0650 | 3.20% |
| SHIP rule (v1) | 0.2385 | 0.1498 | 0.2280 | |
| m1b_v3 (v1 carrier) | 0.2955 | 0.1934 | 0.3420 | 7.55% |
| wavlm_l (unshipped) | 0.3825 | 0.2911 | 0.7523 | 11.57% |

The ordering is the same under every reading, so the lane's readout conclusion is unchanged: the 2B model's ITW number is about **7× better than the shipped rule's** and about 9× better than the carrier's, and the size trend holds (2B < 1B < MMS-1B < MMS-300M). This remains a readout: ITW is spent as a selection set (rule 7), these models' other populations are in-sample, and the test set itself may be (DiffSSD and LJ Speech are in their training data). It is also the reason V2-53's benchmark rows are the next lane.

## What changed in code

- `hearsay.benchmark.PRIMARY = "package03"`. Scorecards' inner-chosen thresholds (test share above threshold, flips vs control) now use it. The persisted v1 scorecard and every earlier card on disk were written under package-0.5 and say so in their `reading` fields.
- `hearsay.fusion_v2.PRIMARY = "package05"`: V2-60 stays on its pre-declaration.
- Pins: `test_scorecard_primary_is_package03`, `test_v2_60_verdict_reading_is_the_predeclared_one_not_the_benchmark_default`.
