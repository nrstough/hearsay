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

## Results (appended after the run, ~08:13; the rule above was not changed)

| Candidate | Inner (brief) | Holdout (brief) | Holdout (averse) | Holdout argmin [#FA, #miss] | ITW brief | ITW averse | ITW P_FA / P_miss at inner threshold |
|---|---|---|---|---|---|---|---|
| A α 0.0 (M1b alone) | 0.301 | 0.072 | 0.057 | [8, 63] | 0.343 | 0.296 | 0.8% / 27.8% |
| A α 0.1 | 0.260 | 0.044 | 0.034 | [5, 37] | 0.319 | 0.285 | 0.65% / 28.6% |
| **A α 0.2** | 0.236 | 0.030 | 0.029 | [3, 30] | 0.322 | 0.280 | 0.45% / 36.0% |
| A α 0.3 | 0.230 | 0.021 | 0.023 | [0, 41] | 0.322 | 0.280 | 0.25% / 39.5% |
| A α 0.4 | 0.232 | 0.015 | 0.020 | [1, 19] | 0.339 | 0.276 | 0.25% / 45.0% |
| A α 0.5 | 0.241 | 0.016 | 0.019 | [1, 21] | 0.356 | 0.299 | 0.2% / 52.8% |
| B min | 0.285 | 0.029 | 0.032 | [2, 38] | 0.495 | 0.308 | 0.2% / 76.5% |
| B max | 0.285 | 0.079 | 0.044 | [4, 118] | 0.459 | 0.498 | 0.3% / 46.9% |
| C cascade | 0.239 | 0.016 | 0.019 | [1, 21] | 0.355 | 0.296 | 0.2% / 52.6% |
| D non-neg shrunk | 0.230 | 0.025 | 0.020 | [0, 49] | 0.324 | 0.280 | 0.4% / 36.9% |
| **E on α 0.2 (FINAL)** | 0.140 | 0.014 | 0.009 | [1, 18] | **0.260** | **0.267** | 1.4% / 14.8% |

**Admissible:** α 0.2–0.5, C and D. The lowest ITW brief-cost score is α 0.2 and α 0.3, tied at 0.322 with D at 0.324 inside the 0.01 band. The tie goes to the smaller α, so the winner is **α 0.2**, inside the consult's predicted 0.2–0.3.

**E (M3 as false-alarm suppression only)** lowers ITW brief from 0.322 to 0.260 and does not raise the holdout or inner scores, so it is applied. **FINAL = E on α 0.2.**

**Caveat.** E's inner, holdout and ITW gains all involve M3, whose training data is undisclosed. E can only move a file toward "real", so if M3 is wrong about a file, the cost is a miss (weight 1), never a false alarm.

**Files:**
- Primary: `submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv`
- Flipped (use only if NSA's feedback decodes as their code's polarity): `..._FLIPPED_only_if_NSA_scores_inverted.tsv`
- Constants for the runner: `models/fusion_v1/constants.json`
- Speech gate: 0 of 1,671 test files gated. Determinate scores sit in [0.001, 1].

## Addendum: M5 candidates, pre-declared before any run (Sat ~08:35, requested by Nathan via the oversight chat)

M5 = `outputs/detector_scores/m5_xlsr_ft.csv`. It is the frozen-backbone, head-only arm; its inner rows are out-of-fold. Its In-the-Wild scores are the 2,000 real clips already scored plus the 1,000 spoof clips scored now with `scripts/m5_score.py`, using the same seed-200 crops.

**F: M5 as a second false-alarm suppressor, on top of the shipped E-on-A-α0.2 rule.**
- Order: after the M3 step. If `m5_logit < t_F` and `base > 0.5`, then `base *= 0.5`. M5 never promotes a file toward synthetic.
- Threshold: `t_F` is fixed from inner OOF only, before the holdout or In-the-Wild is looked at. It is the M5 logit value below which **1% of inner spoof rows** fall, so the suppression can fire wrongly on at most ~1% of known fakes.
- Why that reading: the request said "~1% inner-real coverage", which is ambiguous. A threshold that fires on only 1% of *real* rows would almost never act, so this is the conservative interpretation that keeps the step meaningful. The fraction of inner real rows it covers is reported alongside.

**A3: three-way rank blend** with weights (0.8 − w, 0.2, w) for (M1b, handcrafted, M5).
- w ∈ {0.1, 0.2}, each run **without** and **with** the M3 suppression step (same E rule as before), so four variants.

**Readouts:** identical to the 08:13 sweep.

**Decision rule (fixed now).** The current rule (E on A α 0.2) is replaced by a candidate only if **all** of these hold:
1. In-the-Wild brief-cost minDCF improves by ≥ 0.01.
2. In-the-Wild miss-averse minDCF gets worse by no more than 0.01.
3. Holdout brief-cost gets worse by no more than 0.05 (inside the holdout's noise band).
4. Inner OOF gets worse by no more than 0.03.

If several candidates qualify, the one with the best In-the-Wild brief-cost wins. If none qualifies, the rule stays frozen and the negative result is recorded here. Any change needs Nathan's ratification before it ships.

**Expectation, stated in advance:** F may buy a little on In-the-Wild false alarms. A3 will probably lose on the holdout's short clips and LibriSpeech real.

### Addendum results (appended after the runs, ~09:15; the rule was not changed)

**Deviation from the addendum, disclosed.** The addendum said M5's In-the-Wild spoofs would be scored "using the same seed-200 crops". `scripts/m5_score.py` has no crop option, so the first run scored all 3,000 M5 In-the-Wild clips whole, capped at 8 s, while the other detectors saw seed-200 test-length crops. The run was then repeated **crop-fair**: the identical `prepare_segment(x, crop_s, 200 + row)` crops were written to WAV and scored with `m5_score.py`. Both runs are reported. **The crop-fair run is the decision input.**

**`t_F`** = −3.90 (1% of inner spoof rows fall below it; it covers 41.8% of inner real rows).

| Candidate | Inner (brief) | Holdout (brief) | Holdout (averse) | Holdout [#FA, #miss] | ITW brief (whole-clip M5 → crop-fair) | ITW averse (whole-clip → crop-fair) | ITW P_FA / P_miss at inner threshold (crop-fair) |
|---|---|---|---|---|---|---|---|
| CURRENT: E on A α 0.2 | 0.140 | 0.014 | 0.009 | [1, 18] | 0.260 → 0.260 | 0.267 → 0.267 | 1.4% / 14.8% |
| F: M5 suppression | 0.136 | 0.014 | 0.009 | [1, 18] | 0.258 → 0.258 | 0.267 → 0.268 | 1.4% / 14.4% |
| A3 w 0.1 | 0.222 | 0.022 | 0.027 | [1, 33] | 0.282 → 0.290 | 0.264 → 0.265 | 0.4% / 33.3% |
| A3 w 0.1 + E | 0.136 | 0.0095 | 0.009 | [0, 19] | 0.241 → 0.244 | 0.251 → 0.252 | 1.45% / 13.0% |
| A3 w 0.2 | 0.213 | 0.020 | 0.025 | [1, 30] | 0.266 → 0.274 | 0.246 → 0.251 | 0.3% / 32.4% |
| **A3 w 0.2 + E** | **0.135** | **0.0065** | 0.009 | [0, 13] | 0.215 → **0.228** | 0.234 → **0.239** | 1.3% / 12.2% |

**Decision under the pre-declared rule:** A3 w 0.1 + E and A3 w 0.2 + E both qualify; the winner is **A3 w 0.2 + E**. The weights are 0.6 M1b v3 + 0.2 handcrafted v5 + 0.2 M5 in rank space, followed by the M3 false-alarm suppression step. F does not qualify: its ITW gain of 0.003 is under the 0.01 bar.

**The advance expectation was wrong.** A3 was expected to lose on the holdout's short clips and LibriSpeech real, but with the M3 step it improved both (LibriSpeech 0.0186 → 0.0065).

**Effect on the test set** (current rule vs candidate): Spearman 0.974, and only **3 of 1,671 files** cross 0.5.

**Status: candidate, pending Nathan's ratification. Not shipped.**
- TSVs, both polarities, are logged: `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_*.tsv`.
- Runner constants: `models/fusion_v2_candidate/constants.json` (moved at 10:45 to `outputs/fusion/fusion_v2_candidate/constants.json`, the record; the runner's file is `models/fusion_v2/constants.json`, see the 10:45 note below).
- Shipping it would require the pipeline and Docker to add the M5 scorer: a 657 MB checkpoint from the HF Hub, about 0.2 s per clip on CPU. _(10:45: the pipeline half is done, see below; the image half waits for the switch.)_

**Nathan's decision (09:25): positive result, not shipped pending draft review.**
- E on A α 0.2 stays frozen, and it is the file sent for the draft review.
- A3 w 0.2 + E is revisited only if NSA's draft-review number decodes to our direction and there is time left before 22:00.
- The candidate artifacts stay where they are. No pipeline changes. _(Superseded at ~10:20 by the note below: Nathan approved building the runner side now, gated, so the switch is one line if the number comes back in our direction.)_
- Reaction branches are in `docs/reports/2026-09-26_draft-review-branches.md`.

**Runner side built, not switched (M4 fusion chat, 10:45; run spec `docs/specs/2026-09-26_m4-fusion-v2-m5-scorer.md`).** Nathan (~10:20, "build now, switch later"): make the revisit a one-line switch without changing anything that ships. Done:
- `scripts/fuse_sweep_m5.py --write` (the flag now exists; without it nothing under `models/` is written) wrote `models/fusion_v2/constants.json`: weights 0.6 / 0.2 / 0.2, the three inner-OOF rank references, the same M3 step, the Platt map, a `how` line, and the M5 checkpoint's hashes (`m5_xlsr_ft_20260926-0741/model`). Against the 09:14 candidate file: weights and references identical, Platt a and b identical to the last digit (delta 0.0). The candidate constants moved to `outputs/fusion/fusion_v2_candidate/` so `docker/build.sh` stops staging them.
- The runner and the API execute it with `--fusion models/fusion_v2/constants.json` (M5 loads and runs only under a file that weights it; a checkpoint whose hashes differ from the file's is refused before anything is scored). Exports through the new file reproduce both 09:14 candidate TSVs on all 1,671 rows to 7e-16; live from audio on 50 template files: Spearman 1.0, max abs diff 2.6e-4 (the known CPU-vs-A100 M5 gap), 0 scorer errors. Full-set numbers in `docs/reports/2026-09-26_runner-docker.md` (v3).
- **Unchanged:** the default rule (`models/fusion_v1/constants.json`, `e_on_a`), every TSV under `submissions/` (the full-set live run added `submissions/20260926-1131_M4_runner_fusion_v2_M5_live_PARITY_our_direction.tsv` with a log row at 11:31: Spearman 1.0 and max abs diff 9.3e-4 against the candidate on all 1,671 rows, both polarities; a parity artifact, never a promotion), the draft-review payload, the Docker image. The switch (default → v2, the shipped TSV pair, the image with `models/m5_shipped`, README) is a separate commit on Nathan's word after NSA's number.
