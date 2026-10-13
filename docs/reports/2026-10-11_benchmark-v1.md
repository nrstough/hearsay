# Benchmark v1: v1's baseline scorecard (V2-01-lite)

Run spec: `docs/specs/2026-10-11_v2-benchmark-fusion.md`. Harness: `src/hearsay/benchmark.py` and `scripts/v2_benchmark.py`. Registry: `benchmarks/v1.yaml` (33 pinned inputs, including the model and backbone directories). Machine-readable scorecard: `outputs/v2/benchmark/v1_scorecard.json` (gitignored). Branch `v2-benchmark-fusion`.

## What this is

One scorer for every v2 number. It takes any per-clip score file or fused rule and scores it on every population we hold today, under three metric readings plus EER. A number from anywhere else is a note, not a result (C.1 rule 2).

- **package-0.5** (primary, NSA's shipped minDCF): P_FA + 4·P_miss.
- **package-0.3**: P_FA + 1.714·P_miss.
- **brief** (v1's selection reading): 9.33·P_FA + P_miss.

Every inner number is on the **16,142-row NSA basis**, out-of-fold by `splits/nsa_folds.csv`. Fusion rules use **nested** rank references on inner rows (each fold ranked against the other four folds only). Holdout, ITW, cohorts and test are ranked against all inner rows.

## Reference reproduction

All seven reference numbers reproduce within 5e-5 of their rounded values (tests `test_ref_*`). The shipped rule recomputed on the test set reproduces the shipped TSV to 1e-12 (`test_c0_reproduces_shipped_tsv_on_test`).

| What | Basis | brief | package-0.5 | package-0.3 |
|---|---|---|---|---|
| m1b_v3 export | inner 16,142 | 0.3013 | 0.3307 | 0.2280 |
| m1b_v3 export | holdout | 0.0717 | 0.0571 | 0.0356 |
| m1b_v3 export | ITW | 0.3420 | 0.2955 | 0.1934 |
| SHIP, transductive refs | inner 16,142 | 0.1351 | 0.2997 | 0.1845 |
| SHIP, transductive refs | holdout | 0.0065 | 0.0087 | 0.0053 |
| SHIP, transductive refs | ITW | 0.2280 | 0.2385 | 0.1498 |
| SHIP, nested refs | inner 16,142 | 0.1558 | 0.3390 | 0.2156 |

## v1 scorecard

Pooled. Inner is selectable; holdout is read once; ITW is report-only (spent on v1 selection, rule 7).

| What | Population | brief | package-0.5 | package-0.3 | EER |
|---|---|---|---|---|---|
| m1b_v3 | inner | 0.3013 | 0.3307 | 0.2280 | 0.0845 |
| m1b_v3 | holdout | 0.0717 | 0.0571 | 0.0356 | 0.0140 |
| m1b_v3 | ITW | 0.3420 | 0.2955 | 0.1934 | 0.0755 |
| handcrafted_v5 | inner | 0.4687 | 0.6268 | 0.3492 | 0.1312 |
| handcrafted_v5 | holdout | 0.1371 | 0.0962 | 0.0748 | 0.0314 |
| handcrafted_v5 | ITW | 0.9837 | 0.8160 | 0.6874 | 0.3023 |
| m5_xlsr_ft | inner | 0.3016 | 0.4957 | 0.2933 | 0.1097 |
| m5_xlsr_ft | holdout | 0.3625 | 0.1790 | 0.1384 | 0.0604 |
| m5_xlsr_ft | ITW | 0.3900 | 0.3870 | 0.2526 | 0.0970 |
| spectra_aasist | inner | 0.0045 | 0.0037 | 0.0027 | 0.0011 |
| spectra_aasist | holdout | 0.0120 | 0.0022 | 0.0022 | 0.0016 |
| spectra_aasist | ITW | 0.0650 | 0.1330 | 0.0682 | 0.0320 |
| wavlm_l | inner | 0.3108 | 0.6241 | 0.3888 | 0.1471 |
| wavlm_l | holdout | 0.0717 | 0.0449 | 0.0314 | 0.0145 |
| wavlm_l | ITW | 0.7523 | 0.3825 | 0.2911 | 0.1157 |
| xlsr_L20 | inner | 0.3292 | 0.6581 | 0.3843 | 0.1461 |
| xlsr_L20 | holdout | 0.0561 | 0.0374 | 0.0223 | 0.0086 |
| xlsr_L20 | ITW | 0.3267 | 0.3715 | 0.2391 | 0.0925 |
| SHIP, nested | inner | 0.1558 | 0.3390 | 0.2156 | 0.0813 |
| SHIP | holdout | 0.0065 | 0.0087 | 0.0053 | 0.0021 |
| SHIP | ITW | 0.2280 | 0.2385 | 0.1498 | 0.0580 |

Spectra's numbers are not comparable evidence: its training data is undisclosed, so every population may be in-sample for it. It is used only as the frozen E step.

**The shipped rule in more detail (nested inner):**

- Per fold, package-0.5: f0 0.4183, f1 0.0032, f2 0.0, f3 0.0, f4 0.1649. Macro 0.1173 (brief macro 0.1191).
- Without E: brief 0.2511, package-0.5 0.3814. E fires on 345 inner reals (f0 57, f1 56, f2 90, f3 111, f4 31) and on no spoofs.
- At the package-0.5 threshold (0.3744 on the blended rank) it makes 1,413 false alarms out of 8,142 reals and 331 misses out of 8,000 spoofs.
- Spearman with the fused score on inner real rows: m1b_v3 0.84, m5_xlsr_ft 0.58, handcrafted_v5 0.35. No dominance flag.
- Test set: 39.6% of the 1,671 files sit above that threshold. Spearman with the shipped TSV: 1.0000.

## Bootstrap intervals (inner and ITW)

Class-stratified cluster bootstrap, 2,000 draws, seed 0 (p5 to p95). Inner, primary row: LJ is one fixed stratum present once in every draw, so the intervals are conditional on LJ; LibriSpeech speakers and spoof generators are resampled. ITW: speakers resampled within class. Holdout has only 2 spoof generators, so no interval is reported: m1b_v3 and SHIP carry the degenerate marker in the JSON; the five baseline columns' holdout readouts come from the Oct 12 read, which predates the interval code.

| What | inner brief | inner package-0.5 | ITW brief | ITW package-0.5 |
|---|---|---|---|---|
| m1b_v3 | 0.097 to 0.504 | 0.060 to 0.454 | 0.229 to 0.441 | 0.186 to 0.404 |
| handcrafted_v5 | 0.264 to 0.660 | 0.175 to 0.980 | 0.955 to 0.996 | 0.674 to 0.937 |
| m5_xlsr_ft | 0.112 to 0.513 | 0.072 to 0.690 | 0.290 to 0.466 | 0.287 to 0.486 |
| spectra_aasist | 0.001 to 0.010 | 0.001 to 0.004 | 0.044 to 0.086 | 0.089 to 0.173 |
| wavlm_l | 0.079 to 0.553 | 0.047 to 0.776 | 0.526 to 0.995 | 0.286 to 0.488 |
| xlsr_L20 | 0.092 to 0.581 | 0.045 to 0.797 | 0.224 to 0.425 | 0.226 to 0.527 |
| SHIP, nested | 0.006 to 0.371 | 0.002 to 0.486 | 0.155 to 0.293 | 0.145 to 0.330 |

The inner intervals are wide because they resample only 8 spoof generators: whether grad_tts is drawn moves the shipped rule's cost from near 0 to about 0.5. The sensitivity row in the JSON resamples LJ as one cluster among the real clusters; 719 of its 2,000 draws contain no LJ at all, which is why LJ is held fixed in the primary row. Use paired differences, not these marginal intervals, to compare rules.

## Holdout perturbation cohort (report-only)

500 holdout clips × 5 kinds, the four shipped columns only (`outputs/channel/m3_perturb.csv`). Pooled package-0.5 / brief per kind:

| Rule | none | mp3 | noise20 | speed | shift1 |
|---|---|---|---|---|---|
| SHIP | 0.004 / 0.012 | 0.004 / 0.037 | 0.196 / 0.269 | 0.000 / 0.000 | 0.004 / 0.008 |
| m1b_v3 | 0.048 / 0.056 | 0.056 / 0.085 | 0.268 / 0.271 | 0.052 / 0.052 | 0.052 / 0.048 |
| handcrafted_v5 | 0.084 / 0.048 | 0.104 / 0.088 | 1.000 / 0.996 | 0.076 / 0.089 | 0.084 / 0.048 |
| m5_xlsr_ft | 0.264 / 0.433 | 0.280 / 0.432 | 0.380 / 0.487 | 0.280 / 0.425 | 0.260 / 0.428 |

20 dB white noise remains the shipped rule's hole, and handcrafted_v5 collapses completely under it. The cohort's precomputed fused columns are never read (they used transductive references). Its clips are holdout clips with holdout labels, so this readout is its own ledgered read.

## MLAAD (report-only, in-sample for M5)

572 English MLAAD fakes, all in M5's training manifest. Share detected at each rule's inner package-0.5 threshold: SHIP 0.970, m1b_v3 0.948, handcrafted_v5 0.872, m5_xlsr_ft 0.988 (in-sample).

## Holdout accounting

- m1b_v3 and the shipped rule (all-inner references) are the two published holdout references and are exempt, but only for the registry-pinned files: a different file under the same name is ledgered.
- The other five columns were read once as one ledgered set, "benchmark-v1 baseline", on Oct 12. The ledger key now includes each file's sha256; the new key is recorded as an alias of that read (same files), not a new read.
- The holdout perturbation cohort is one ledgered read.
- Ledger total for this lane: 4 distinct holdout reads (this baseline set, this cohort, and the V2-60 candidate set and its cohort). The tracked audit copy is `benchmarks/holdout_ledger.jsonl`; the authority files live in the shared `outputs/v2/benchmark/holdout_ledger/`.
- **This run's historical exception.** Before the pre-audit critique, the cohort readout ran outside the ledger: in the first scorecard run (Oct 12) and in the V2-60 dry run and first run. Nothing was selected on it. The ledger cannot certify those earlier reads after the fact.
- **What the ledger guarantees from commit 0d9a6e4 on** (plus the per-key lock added after the Codex audit): every holdout and holdout-cohort read goes through `read_holdout`; a rule set is read at most once, even by two concurrent runs; only the two registry-pinned references are exempt.

## Pending and deferred

- **Pending** (cohort scoring job, ask-first): the ITW perturbation cohort for every column, and WavLM and layer 20 on the holdout cohort. When they land the registry becomes v1.1 and this scorecard is regenerated without a new holdout read.
- **Deferred:** MLAAD held out by family, ASVspoof 5 eval, reverb / codec / telephony / short-crop cohorts, the sealed set, and actDCF (to V2-61).

## Limitations

- The folds are generator-held-out but not wholly speaker-disjoint: LJ is one speaker whose chapters cross folds, and generators share target voices.
- Bootstrap intervals keep LJ as one fixed stratum, so they are conditional on LJ. Holdout (2 spoof generators) and inner f3 and f4 (1 generator each) are too few clusters for an interval.
- Not evaluated: telephony, reverb, codecs beyond mp3, short crops, physical replay, ASVspoof 5, MLAAD by family, and v1's duration, level, silence, container and single-speaker shortcut limits.
- wavlm_l and xlsr_L20 were trained on the 23,790-row basis (with ASVspoof 2019 train/dev rows); the shipped columns were not. Their holdout and ITW were already read in the October 3 bake-off.
