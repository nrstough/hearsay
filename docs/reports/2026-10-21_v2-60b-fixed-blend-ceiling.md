# V2-60b: the fixed v1 + 2B rank-blend ceiling (readout)

**Date:** 2026-10-21. **Chunk:** V2-60 (Phase 6), the pre-declared readout that the conditions-gated design of record has to beat. **Pre-declaration:** `benchmarks/v2_60b_candidates.json`, committed as `cfa7699` **before any read**. **Benchmark:** registry v1.3 (the four AntiDeepfake columns pinned). **Reading:** package-0.3 primary (rule 5). **Not read:** the holdout (ledgered, not spent on a readout) and the sealed key.

**Status: a readout, not a selection.** The inner folds are in-sample for the 2B column (DiffSSD, LJ Speech, LibriTTS are in its training table), so inner numbers favour it by construction; In-the-Wild is held out for every column but is spent as a selection set (rule 7). Nothing is adopted here.

## 1. Candidates

Rank blends with the frozen E step (Spectra suppression below −3). B0 is the shipped v1 rule; B1–B3 scale v1's weights by (1−a) and give a to the 2B; B4 is the 2B alone (with E); B5 asks whether the handcrafted and M5 seats still earn their keep next to the 2B.

| | m1b_v3 | handcrafted_v5 | m5_xlsr_ft | xlsr_2b |
|---|---|---|---|---|
| B0 (control, shipped) | 0.60 | 0.20 | 0.20 | 0 |
| B1 | 0.45 | 0.15 | 0.15 | 0.25 |
| B2 | 0.30 | 0.10 | 0.10 | 0.50 |
| B3 | 0.15 | 0.05 | 0.05 | 0.75 |
| B4 | 0 | 0 | 0 | 1.00 |
| B5 | 0.50 | 0 | 0 | 0.50 |

Scored with `scripts/v2_benchmark.py rule` (nested rank references on inner; ITW with the per-column override files); cards `outputs/v2/benchmark/V2_60b_B*.json`.

## 2. The table (package-0.3 unless marked)

| | inner ⁱ | inner f4 ElevenLabs ⁱ | inner f0 grad_tts+unit_speech ⁱ | **ITW** | ITW 90% band | ITW EER | ITW without E | test share ≥ inner thr | Spearman vs shipped TSV |
|---|---|---|---|---|---|---|---|---|---|
| B0 shipped | 0.2156 | **0.1069** | 0.3114 | 0.1498 | 0.096–0.200 | 5.80% | 0.1618 | 28.7% | 1.000 |
| B1 (2B 0.25) | 0.1256 | 0.0669 | 0.1606 | 0.1183 | 0.070–0.162 | 4.65% | 0.1228 | 35.8% | 0.986 |
| B2 (2B 0.50) | 0.0533 | **0.0543** | 0.0406 | 0.0799 | 0.045–0.110 | 3.25% | 0.0834 | 38.8% | 0.949 |
| B3 (2B 0.75) | 0.0260 | 0.0617 | 0.0067 | 0.0410 | 0.020–0.056 | 1.60% | 0.0440 | 39.7% | 0.911 |
| B4 (2B alone) | 0.0507 | 0.1159 | 0.0006 | **0.0206** | 0.008–0.036 | 1.00% | 0.0221 | 38.2% | 0.649 |
| B5 (carrier + 2B) | 0.0537 | 0.1006 | 0.0550 | 0.1123 | 0.063–0.156 | 4.20% | 0.1208 | 32.9% | 0.914 |

ⁱ in-sample for the 2B column. Inner brief / package-0.5 and the per-fold tables are in the cards.

## 3. Reading it

- **On In-the-Wild, every gram of v1 you mix in costs.** ITW is monotone in the 2B weight: 2B alone 0.021, then 0.041, 0.080, 0.118, 0.150 for the shipped rule. The complementarity the sealed key showed (v1 catches the 14 fakes the 2B ranks most real) does not show on ITW: whatever the 2B misses there, v1 does not catch either. A fixed blend cannot beat the 2B alone on the one held-out population.
- **On the ElevenLabs fold, the blend wins and neither parent does.** Fold 4 is the one inner fold where the 2B is not near zero (0.116) and the shipped rule led it (0.107). B2, half and half, scores **0.054**, half of either parent; B1 and B3 are 0.067 and 0.062. This is the complementarity, on the one in-sample fold that behaves like a held-out one. Which parent to trust depends on the generator, which is the gating argument in one row.
- **The handcrafted and M5 seats still earn their keep.** B5 (carrier + 2B, nothing else) is worse than B2 everywhere that matters: ElevenLabs 0.101 vs 0.054, ITW 0.112 vs 0.080. Dropping v1's two minor columns to make room for the 2B is the wrong trade.
- **Without E, every row is worse by a little** (ITW: 0.150 → 0.162 for B0, 0.080 → 0.083 for B2, 0.021 → 0.022 for B4). The frozen suppression step keeps its place; it never matters much next to the 2B.
- **The test-set diagnostics move the way the Phase 1 read predicted.** The share of test clips above the inner-chosen threshold rises from 28.7% (B0) to 36–40% for anything with the 2B, and the rank correlation with the shipped TSV falls to 0.65 for the 2B alone. The 2B re-orders the test set substantially; the key said its errors there are nearly disjoint from v1's.

## 4. What it settles and what it leaves

- **Settled:** a fixed blend has no single best weight. ITW wants 2B alone; the ElevenLabs fold wants half and half; the sealed key (Phase 1) says the two parents fail on different clips. That is the case for the conditions-gated blend in the V2-60 design of record, measured rather than argued.
- **The ceiling for the gate:** on ITW it must at least match 2B alone (0.021 [0.008–0.036]); on the ElevenLabs fold it must beat B2 (0.054); and it must do both at once, which no fixed weight does.
- **Not settled, and not this readout's to settle:** adoption. The gate's features come from V2-24; it is fitted on inner only, read on ITW and the cohorts, and gets one sealed read at the Phase 6 close. This readout spent no holdout and no sealed read, so the budget is intact.
- **A caution on inner:** B3's inner 0.026 and B4's fold-0 0.0006 are in-sample numbers and mean nothing about new generators. The ElevenLabs fold is the only inner cell quoted above as evidence, and even it is in DiffSSD.

## 5. Files

| What | Where |
|---|---|
| Pre-declaration (committed before the reads) | `benchmarks/v2_60b_candidates.json`, rules `benchmarks/v2_60b/B*.json` |
| Cards | `outputs/v2/benchmark/V2_60b_B{0..5}.json` |
| Registry | `benchmarks/v1.yaml` v1.3 (adds the four AntiDeepfake columns and their ITW overrides) |
| Log | `experiments/log.csv`, rung `V2-60b` (six rows, inner package-0.3 as `validation_score`, in-sample caveat in `notes`) |
