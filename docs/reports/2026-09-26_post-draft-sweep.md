# Post-draft sweep: the locked table and the packet (Sat Sep 26, 2026, 19:08 EDT)

**Recommendation: KEEP.** None of the four candidates (T2, W4, P_wl, H_noise) passes its pre-declared test. `submissions/CrossExam_predictions.tsv` (sha256 `fb783076…`, the file NSA scored at minDCF 0.0733) stays final. H_noise was judged in Run 2 (19:17, final report `354502ca…`; section "Run 2" below). The "Pending" section below is the 19:08 state, kept for the record.

**Provenance:**

| | |
|---|---|
| Pre-declaration | `docs/reports/2026-09-26_post-draft-manifest.md`: `141b254` (17:39), clarifications `b877760`, the P_wl "with room" amendment `ae9c4c4`, the H_noise note `43916ee`. Every one was committed before the number it governs was computed. The refit-seat line in `43916ee` came about a minute after `wavlm_l.csv` existed but before it was read; it is a diagnostic only (see the manifest's erratum). |
| Script | `scripts/fuse_sweep_v3.py` at `43916ee`, run once at 19:08, a minute after `wavlm_l.csv` landed |
| Locked report | `outputs/fusion/sweep_v3_report.json`, sha256 `c2b6bc22d69f67acb037cb1f5e1c11e0762f2b66d34ce1a8a60139c376ed69ad`; also archived as `sweep_v3_report_20260926-190836.json`. `--write` requires this hash. |
| WavLM export | `outputs/detector_scores/wavlm_l.csv` (sha256 `cef08ea6…`; 16,142 / 3,858 / 1,671 / 3,000 rows) |
| WavLM probe | `models/m1_wavlm-large_L9_20260926-1906` (heads chat, commit `e611d1b`; report `docs/reports/2026-09-26_post-draft-heads.md`; its 57-of-158 count is at the ITW argmin, while this report's 41 of 122 is at the inner-OOF threshold) |
| Self-check | CURRENT reproduces the shipped numbers on every field; Spearman against the shipped file is 1.0000000 over all 1,671 rows (the pinned block is empty) |

## The locked table

Every cell is minDCF (lower is better). "Brief" is the brief's cost, 9.33·P_FA + P_miss; "averse" is the sponsor code's convention, P_FA + 4·P_miss. Spearman and 0.5-crossings are measured on the test set against the shipped file.

| Rule | Inner brief / averse | Holdout brief / averse | ITW brief / averse | Holdout argmin [FA, miss] | ITW P_FA / P_miss at inner thr | Test Spearman | 0.5-crossings | Verdict |
|---|---|---|---|---|---|---|---|---|
| **CURRENT** (shipped A3 w0.2 + E) | 0.1351 / 0.2997 | 0.0065 / 0.0087 | 0.2280 / 0.2385 | [0, 13] | 1.3% / 12.2% | 1.0000 | 0 | shipped |
| T2 (second M3 tier) | 0.1351 / 0.2997 | 0.0065 / 0.0087 | 0.2280 / 0.2385 | [0, 13] | 1.3% / 12.2% | 0.9857 | 0 | **FAIL** (gate) |
| W4 (M5 weight 0.4, post hoc) | 0.1373 / 0.3366 | 0.0065 / 0.0088 | 0.1997 / 0.2075 | [0, 13] | 1.0% / 11.7% | 0.9702 | 4 | **BAKEOFF FAIL** |
| P_wl (WavLM as a fourth column) | 0.1557 / 0.3330 | 0.0030 / 0.0071 | 0.2047 / 0.2300 | [0, 6] | 1.3% / 10.0% | 0.9667 | 9 | **FAIL** (gate and room) |
| H_noise (handcrafted v6) | 0.1366 / 0.3058 | 0.0070 / 0.0103 | 0.2283 / 0.2405 | [0, 14] | 1.25% / 11.9% | 0.9953 | 1 | **FAIL** (gate; run 2, 19:17) |

## Verdicts

**T2 fails the gate** (rules 2a, 3a and 5). Its six cells are identical to CURRENT's, as the Round 2 verification predicted. The mechanism diagnostic passes: the deeper tier fires on 0 spoof rows in every split and every perturbation kind. It fires only on bona fide rows (inner 133, holdout 34, ITW 2, perturbation 0–5), which are already below the threshold region. It reorders files without moving any metric, so it has nothing to offer.

**W4 fails its fresh-evidence bake-off** (Nathan's 17:40 ruling):
- Part 1, the standing rule: passes.
- Part 2, perturbation robustness: fails. Δ is CURRENT − W4 per kind, brief / averse; three cells fall below −0.010. The tripwire held at 1.1e-16.

  | Kind | Δ brief | Δ averse |
  |---|---|---|
  | none | −0.012 | 0.000 |
  | mp3 | −0.004 | −0.004 |
  | noise20 | +0.051 | +0.020 |
  | speed | −0.004 | −0.012 |
  | shift1 | −0.016 | −0.004 |

- Part 3, the ITW speaker-cluster bootstrap (54 speakers, 2,000 replicates, seed 0): fails. The 5th percentile is +0.0044 under brief and **−0.0023 under averse**.

Raising M5's weight buys noise robustness and some In-the-Wild misses. It pays for them on clean, speed-changed and shifted audio, and its In-the-Wild gain under the sponsor's convention cannot be told apart from zero. For the record, it would also fail the stricter gate (rules 2a and 3a doubled, 3b, 5).

**P_wl fails both the gate and the room.**
- **Gate, with thresholds doubled** because the test Spearman is 0.9667, below 0.974:
  - 2a: ITW brief Δ +0.0233, below the doubled 0.040. It clears the undoubled 0.020, but the doubled bar governs.
  - 3a: inner brief Δ **−0.0206**, worse; 3b: inner averse Δ **−0.0333**, worse.
  - Diagnostic: fails. The corrective share on inner OOF is 0.31, not above 0.5.
  - Rules 2b, 4a, 4b and 5 (both costs) pass: the holdout improves (Δ +0.0035 / +0.0016; argmin misses 13 → 6).
- **Room:** ITW Δ is +0.0233 brief and +0.0085 averse, against the 0.030 required under each.
- **Mechanism (diagnostic numbers):**
  - Catches: of CURRENT's 122 ITW misses at the inner-OOF threshold, `wavlm_l` alone flags **41**. That clears both the gate's 16 and the room's 19, against M1b 1, M5 19 and Spectra-AASIST 79.
  - Corrective share: 0.80 on ITW, but 0.31 on inner OOF.
  - Spearman against M1b: 0.86 on holdout, 0.79 on test.
  - Reading: WavLM carries real new information about In-the-Wild's miss tail. On the NSA-domain inner folds, though, it pushes more of the shipped rule's errors the wrong way than the right way.

**Refit seat, diagnostic only (not a candidate; Nathan, 19:10).** `wavlm_l` in M1b's seat (`scripts/fuse_sweep_m5.py --refit-m1b wavlm_l`):
- inner 0.1922, holdout 0.0035 / 0.0045, ITW 0.275 / 0.2665;
- ITW P_FA / P_miss at the inner threshold: **5.3%** / 6.1%.

Replacing M1b halves the ITW misses but quadruples the ITW false alarms, and a false alarm costs 9.33 times a miss. The fourth-column entry was the right one to pre-declare. Outputs: `outputs/fusion/sweep_refit_report_wavlm_l.json`. The earlier refit report was preserved as `sweep_refit_report_before_wavlm.json`.

## Decision rule applied

There are no passers, so rule 7 does not arise. The decision is **KEEP**: `submissions/CrossExam_predictions.tsv`, sha256 `fb7830762691d04d8be938f8e2615e349c998cdfd72c58e94d02385d57368607`, was final unless H_noise passed (the 19:08 state; H_noise failed in Run 2, below). Nothing was tuned after the table: no weight, threshold or seat changed after 19:08.

## What goes in the README (for the README chat)

- **Second backbone (WavLM Large).** In the blend at 0.2 it catches 41 of the shipped rule's 122 In-the-Wild misses. It still fails the pre-declared gate: inner CV worse by 0.021 (brief) and 0.033 (averse), with the ITW gain +0.023 / +0.009 short of the bar. In M1b's seat it quadruples In-the-Wild false alarms (5.3%). Not shipped.
- **M5 at weight 0.4** failed a pre-declared fresh-evidence bake-off: −0.016 on one-sample-shifted audio, and a sponsor-cost ITW gain whose 5th percentile is below zero.
- **Second M3 suppression tier:** inert. Identical metrics, and it never fires on a fake.

## Pending at 19:08 (superseded by Run 2 below)

- **H_noise:** judged by the frozen gate. Its diagnostic uses the CPU chat's noise-AUC numbers via `--hnoise-evidence`; without them it fails.
- **Nathan's ruling window:** the packet is ready now. With no passers there is nothing to ratify. His word confirms KEEP, or waits for H_noise.

## Run 2 (19:17): H_noise, and the final report

The CPU chat's v6 export arrived at 19:16, earlier than the expected 20:30. It came with its noise-AUC evidence in `outputs/channel/hc_noise.json` (sha256 `2d4b4455…`; bundle `models/hc_lgbm_20260926-191630`; CPU chat commits `0065641` and `04f5391`, report `docs/reports/2026-09-26_post-draft-hc-noise.md`). The run:

`uv run python scripts/fuse_sweep_v3.py --hnoise-evidence 0.7716,0.9931 --hnoise-evidence-source outputs/channel/hc_noise.json`

The **final report** has sha256 `354502ca559118fb3f785ce04986b80a53c3e73453636dbc8b3e7b57c4bbddff`, archived as `sweep_v3_report_20260926-191752.json`. This is the hash `--write` would require. The 19:08 report (`c2b6bc22…`) is kept as `sweep_v3_report_20260926-190836.json`. T2, W4 and P_wl are unchanged cell for cell between the two runs.

**H_noise fails the gate** on 2a, 3a, 3b, 5-brief, 5-averse and 5-diagnostic.
- **Metrics:** every cell is a little worse than CURRENT.
  - inner Δ −0.0015 / −0.0061
  - holdout −0.0005 / −0.0016
  - ITW −0.0003 / −0.0020
- **Diagnostic:** it fails. The v6 AUC under 20 dB white noise is **0.7716**, below the 0.90 bar (v5b: 0.6403). Its clean AUC is 0.9931, within 0.005 of 0.998.

The noise twins recover part of the 20 dB lane, but not enough to reach the bar. They also cost the clean holdout: the column alone goes from 0.137 to 0.227, per the CPU chat.

**Decision: KEEP**, with all four candidates judged. `submissions/CrossExam_predictions.tsv` (sha256 `fb783076…`) is final.

After the Codex audit (19:17), the gate code now requires the full five-kind perturbation cohort and all ten bake-off cells. A partial cohort is INVALID. Re-running with that code reproduces `354502ca…` byte for byte.

## Ruling (Nathan, ~19:35): KEEP

Nathan ratified KEEP. The `submissions/log.csv` row records it, with the final-name copy re-verified at sha256 `fb783076…`. `submissions/CrossExam_predictions.tsv` is the file for NSA. No `models/fusion_v3/` was written, and the lane is closed.
