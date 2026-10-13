# V2-60 fusion test: verdict

Run spec: `docs/specs/2026-10-11_v2-benchmark-fusion.md`. Pre-declaration: `benchmarks/v2_60_candidates.json` (committed f8e4ca6, before any code read data). Benchmark: v1 (`benchmarks/v1.yaml`). Code: a371d79. Readout: `outputs/v2/fusion/v2_60_readout.json` (gitignored). Rows in `experiments/log.csv`.

## Verdict

**No fusion candidate beats the shipped rule.** The small learned weighting (C2) ties, and both fixed option-B blends lose. The larger learned model is not triggered.

| Candidate | brief | package-0.5 | package-0.3 | EER | f4 package-0.5 | Verdict |
|---|---|---|---|---|---|---|
| C0, shipped rule (control) | 0.1558 | 0.3390 | 0.2156 | 0.0813 | 0.1649 | control |
| C1, equal weights over five columns | 0.1959 | 0.4520 | 0.2684 | 0.1041 | 0.2968 | LOSE |
| C2, learned non-negative weights, shrunk | 0.1620 | 0.3764 | 0.2247 | 0.0878 | 0.1939 | TIE |
| C3, shipped weights plus 0.1 WavLM and 0.1 layer 20 | 0.1785 | 0.3584 | 0.2318 | 0.0866 | 0.2243 | LOSE |

All numbers are pooled nested inner OOF (16,142 NSA rows) with the frozen E step. The verdict reads nothing else.

- **C1 loses on all three readings.** Package-0.5 is 0.1130 worse than C0 (band 0.05), driven by the ElevenLabs fold (0.2968 vs 0.1649).
- **C3 loses on brief** (0.0227 worse; band 0.02). It also fails the ElevenLabs guard (0.2243 > 0.1649 + 0.05).
- **C2 ties.** It is 0.0374 worse than C0 on package-0.5, inside the 0.05 band, and passes the guard. It is not better on any reading.

## Paired bootstrap against C0 (report-only)

Difference = candidate cost minus C0 cost (positive = candidate worse). Class-stratified, 2,000 draws, seed 0. Primary: LJ held fixed. Sensitivity: LJ resampled as one cluster. P(better) is the share of draws where the candidate is better.

| Candidate | Reading | median (p5, p95), LJ fixed | P(better), LJ fixed | P(better), LJ resampled |
|---|---|---|---|---|
| C1 | package-0.5 | 0.116 (−0.000, 0.145) | 0.086 | 0.083 |
| C2 | package-0.5 | 0.036 (0.000, 0.041) | 0.002 | 0.003 |
| C3 | package-0.5 | 0.020 (0.000, 0.054) | 0.021 | 0.079 |
| C1 | brief | 0.039 (−0.004, 0.089) | 0.277 | 0.299 |
| C2 | brief | 0.006 (−0.001, 0.009) | 0.107 | 0.162 |
| C3 | brief | 0.022 (−0.002, 0.050) | 0.208 | 0.236 |

C2 is worse than C0 in almost every resample. The intervals are dominated by the 8 spoof generators; per-fold intervals for f3 and f4 (one generator each) are degenerate and not reported.

## C2 in detail

- **α hit the top of the grid (0.75) in every fold**, as the early look predicted. C2 is therefore 75% the shipped weights plus 25% a fit. The grid was frozen; α at the edge is reported, not changed.
- **The fit barely uses the option-B columns.** Mean final weights: m1b_v3 0.607, m5_xlsr_ft 0.219, handcrafted_v5 0.171, WavLM 0.003, layer 20 0.000. The off-inner model gives both 0.
- **Near-separation.** Raw fit weights have an L1 norm of 15–37 and intercepts of −6 to −17. The optimizer converged in every fold.
- **Effective parameters:** 1.5 per fold (1.75 in f4).
- **Overfitting check (owner's P2 risk): C2 overfits.** The measure compares C2 with C0 on the same rows, so fold difficulty cancels. In-sample (training folds, train-fold references) C2 beats C0 in every fold; out of fold it ties or loses. Optimism = out-of-fold advantage minus in-sample advantage, package-0.5:

  | Outer fold | in-sample C2 − C0 | out-of-fold C2 − C0 | optimism |
  |---|---|---|---|
  | f0 | −0.0037 | +0.0147 | 0.0184 |
  | f1 | −0.0144 | 0.0000 | 0.0144 |
  | f2 | −0.0146 | 0.0000 | 0.0146 |
  | f3 | −0.0152 | 0.0000 | 0.0152 |
  | f4 | −0.0180 | +0.0290 | 0.0470 |

  Mean optimism: package-0.5 0.022, package-0.3 0.012, brief 0.008. The fit's edge exists only on the rows it was fitted on, and it is largest on the ElevenLabs fold. Holdout and ITW agree: C2 is no better than C0 there.
- **No Platt.** C2 ends at the E step (K1, Nathan, Oct 12). The early look had shown C2 at 0.376 without per-fold Platt and 0.514 with it; the ruling came after that look. Under either reading C2 does not win.

## Report-only populations

These do not enter the verdict. Holdout was read once for the candidate set (ledger key 2ff99dee). The holdout perturbation cohort (holdout clips and labels) is a separate ledgered read (985d6109). Lane total: 4 distinct holdout reads. ITW is spent on v1 selection.

| Candidate | holdout brief | holdout package-0.5 | ITW brief | ITW package-0.5 |
|---|---|---|---|---|
| C0 | 0.0065 | 0.0087 | 0.2280 | 0.2385 |
| C1 | 0.0005 | 0.0020 | 0.1853 | 0.2120 |
| C2 | 0.0070 | 0.0103 | 0.2303 | 0.2385 |
| C3 | 0.0030 | 0.0071 | 0.2077 | 0.2280 |

- **C1 and C3 look better off-inner.** They are the two that lose on inner, through the ElevenLabs fold. Holdout (PlayHT and WaveGrad2) and ITW contain no ElevenLabs-style generator, and both were already read for these columns in the bake-off. This is exactly the disagreement a sealed population has to settle; it is not a reason to re-pick.
- **Test set** (unlabelled): decision flips against C0 at each rule's inner package-0.5 threshold: C1 287, C3 74, C2 16 of 1,671. Spearman with the shipped TSV: C1 0.921, C3 0.980, C2 0.996.
- **Holdout perturbation cohort:** only C0 and C2 are scorable (C2's off-inner model gives zero weight to the unscored columns). 20 dB noise: C0 0.196, C2 0.192 package-0.5. C1 and C3 are not scored until the cohort job runs.
- **MLAAD** (in-sample for M5): detected share at the inner threshold, C0 0.970, C2 0.972. C1 and C3 not scored.
- **E** fires on 345 (C0), 332 (C1), 364 (C2) and 349 (C3) inner reals, and on no spoofs.

## What this means

Fusion of today's columns cannot beat the shipped rule on the selection metric. The learned weights move back toward the shipped ones, and the two option-B columns add nothing the fit can use. The expected and useful outcome holds: fusion needs new columns or new data (V2-31, V2-33, V2-53) before it is worth revisiting. V2-60 stays 🟡.

## Disclosures

- **Early look.** A planning agent computed inner readouts for the frozen candidates (C1 0.452, C3 0.358, C2 prototype 0.376) after the candidate table was committed and before this run. No holdout labels were touched. This run reproduces those numbers.
- **K1 after the look.** The no-Platt ruling was made after seeing that per-fold Platt would turn C2's TIE into a LOSE. It does not change the conclusion.
- **Second-order stacking leakage.** The base columns' OOF scores came from models that saw the held-out fusion fold when scoring training rows. This favours fitted candidates only; a TIE or LOSE is robust to it.
- **Cohort read outside the ledger before the fix.** The dry run and the first real run computed the holdout-cohort readout for C0 and C2 outside the ledger; the pre-audit critique caught it and the cohort now goes through the ledger. Nothing was selected on it.
- **Option-B columns were not fresh off-inner.** WavLM and layer 20 were read on holdout and ITW in the October 3 bake-off, and both were trained on the 23,790-row basis (with ASVspoof 2019 train/dev); the shipped columns were not.

## Limitations

- The folds are generator-held-out but not wholly speaker-disjoint (LJ chapters cross folds; generators share target voices).
- Bootstrap intervals are conditional on LJ (one speaker, 63% of inner reals).
- Not evaluated: telephony, reverb, codecs beyond mp3, short crops, physical replay, ASVspoof 5, MLAAD by family, and v1's duration, level, silence, container and single-speaker shortcut limits.
- The ITW perturbation cohort and the option-B holdout cohort are pending (ask-first compute).

## Decision needed: the sealed population

Adoption of anything in v2 waits for a fresh sealed population (rule 7). C1's off-inner advantage against its inner loss is the first concrete case that needs one. Options:

1. The team records new real and fake clips.
2. MLAAD models released after 2026-04-18 (new pulls).
3. Deepfake-Eval-2024 (evaluation-only licence).
4. NSA's per-clip cross-team results, if shared.
