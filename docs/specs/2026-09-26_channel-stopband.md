# Run spec: does the double low-pass stopband move any detector? (Sat Sep 26, 2026, 14:35)

**Go:** Nathan, in the channel-robustness chat ("go stop band"), after oversight scoped it as measurement only. Follows the open shortcut-ledger entry "Double low-pass stopband" (`docs/architecture.md` §11) and `docs/reports/2026-09-26_channel-robustness.md` (A). Short path (sub-hour, measurement only): this spec, one script, hermetic tests, one Claude critique. No TSV, no rule change, no refit.

## Question

Every detector applies `hearsay.handcrafted.band_limit` once inside its own path: M1b in `prepare_segment`, M3 in `prepare_input`, M5 in `deploy_transform`, handcrafted in `_crop`. NSA test files arrive already low-passed at ~7.2 kHz, so at test time they are filtered twice, while training clips are filtered once. Scoring `band_limit(x)` instead of `x` for the same clip reproduces the test condition. The difference is the detectors' response to the stopband depth alone.

## Design (fixed before any result)

- **Cohort:** the 500 outer-holdout clips of action E (`scripts/m3_probes.holdout_rows(500)`, same seeds and test-length crops): 250 real, 250 spoof. These are out of sample for M1b, M5 and handcrafted v5.
- **Conditions per clip:** `once` = the raw segment; `twice` = `band_limit(segment)`. Both are scored through the runner's own paths (`hearsay.pipeline.Models`, CPU) and fused with `models/fusion_v2/constants.json`, as in E. The fused probability is recorded as well.
- **Readouts:**
  - Per detector (M1b, M5, M3, handcrafted v5): Δlogit = twice − once; mean, median, 95th percentile of |Δ|, and signed mean by class.
  - The same for the fused score and the fused probability.
  - The M3 step's firing rate under each condition.
  - Verdict flips at P = 0.5, and at the brief-cost argmin threshold of the `once` fused scores (descriptive, computed on this cohort).
  - Holdout minDCF under each condition, per detector and fused.
- **Reading rule (pre-declared):**
  - A detector **responds** if its 95th-percentile |Δlogit| exceeds 10% of the IQR of its inner-OOF logits (from its export in `outputs/detector_scores/`).
  - The fused rule **responds** if its holdout minDCF moves by more than 0.037 (one holdout false alarm).
  - **Inconclusive** if fewer than 500 complete pairs, any readout input is non-finite, or an IQR is zero.
  - If nothing responds, the ledger entry closes as "measured, no response".
  - If something responds, the report states the refit cost (handcrafted ~30 min; M1b ~1–1.2 h; M5 not feasible today). Any refit goes to Nathan and through `scripts/fuse_sweep_m5.py`.

## Failure modes → tests (`tests/test_channel_stopband.py`)

| Failure mode | Test |
|---|---|
| The second pass changes the passband (a confound) | band levels below 6.5 kHz move < 0.5 dB; the 7.5–8 kHz level drops by > 20 dB |
| `once` is not the untouched segment | `once` returns the input unchanged |
| Unpaired or non-finite rows enter the readout | a NaN or ±inf in any column for either condition drops that clip |
| The reading rule's guards | each trigger alone flips it to "responds"; a zero IQR, a NaN or a short cohort gives "inconclusive" |
| Verdict flips miscounted | a known shift across 0.5 gives the exact flip count |

## Documentation

- Create: this spec; a section in `docs/reports/2026-09-26_channel-robustness.md`.
- Update: the §11 ledger entry in `docs/architecture.md` (its status), and the STATUS channel-robustness row.

## Results (14:41)

**Measured, no response** on all four detectors and the fused rule (`outputs/channel/stopband.json`): 500 of 500 clips paired, 250 real and 250 spoof.

| Detector | 95th pct \|Δlogit\| | Threshold | Share of threshold |
|---|---|---|---|
| M1b v3 | 0.139 | 0.992 | 14% |
| Handcrafted v5 | 0.466 | 0.886 | 53% |
| M5 | 0.099 | 0.732 | 13% |
| M3 | 0.301 | 1.684 | 18% |

- **Fused holdout minDCF:** 0.012 → 0.016 (limit 0.037).
- **Verdict flips:** 0 at P 0.5; 2 at the `once` argmin threshold.
- **M3 step:** fires on 2.0% of clips under both conditions.
- **Ledger:** the entry in `architecture.md` §11 is closed; no refit.
- **Tests:** `tests/test_channel_stopband.py`, 13 passing.
