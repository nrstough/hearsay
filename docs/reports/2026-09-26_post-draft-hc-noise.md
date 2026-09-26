# Handcrafted v6: noise twins did not make the handcrafted model noise-robust (Sat Sep 26, 2026, evening)

**Question.** The channel lane found that 20 dB white noise breaks the handcrafted model: AUC 0.998 → 0.64 on 500 holdout clips (`docs/reports/2026-09-26_channel-robustness.md`, §E). The handoff's fix was to train on noisy copies of both classes. Does that recover the noise lane without costing clean rows?

**Answer: no.** Noise AUC rises only to 0.77, against a pre-declared bar of 0.90. The clean holdout gets worse, 0.137 → 0.227. v6 is recorded as a measured negative, and `models/hc_selected` stays on v5b.

Run by the post-draft CPU lane on Nathan's 19:00 instruction, relayed by the orchestrator. The recipe and pass bar are from `docs/handoffs/2026-09-26_post-draft-cpu-handoff.md`, written before any v6 number existed.

## Recipe

v6 is v5b with one more augmentation. Nothing else changed.

**Extraction.** `scripts/extract_handcrafted.py` gained `--noise-frac` / `--noise-snr-db`:
- **The noise.** White Gaussian noise at exactly the stated SNR (`hearsay.handcrafted.white_noise`). It is the same recipe as the channel probe's `noise20`.
- **The draw.** It comes from the same per-row `default_rng(seed + row)` as launder and tilt. It is recorded in `hc_augment` as `noise:<snr>:<seed>` and applied before band match, trim and crop, to both classes.
- **Draws unchanged.** The noise draw comes after the launder and tilt draws, so every row keeps its v5 launder/tilt draw. This was checked on all 20,000 rows. A test pins it, together with the SNR (`tests/test_hc_augment.py::test_noise_draw_keeps_v5_draws_and_hits_the_snr`).

```bash
uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_train_sample.csv \
  --name nsa_train_sample_v6aug --crop test --seed 0 \
  --families lfcc,phase,cqcc,modulation,breath,jitter \
  --launder-frac 0.35 --tilt-frac 0.35 --noise-frac 0.35 --noise-snr-db 20 --workers 4
```

- 20,000 rows, 0 failures, 1,358 s.
- 14,436 rows augmented (v5: 11,550).
- 6,901 rows noised: 3,422 real, 3,479 fake.

**Training.** Exactly v5b: the clean v4 rows, plus the augmented twins of the inner rows as fit-only extras (11,614, against v5b's 9,300). Every readout and the export stay on clean rows.

```bash
uv run python scripts/train_handcrafted.py --train nsa_train_sample_v4 --folds splits/nsa_folds.csv \
  --test nsa_test_v4 --extra-rows-from outputs/handcrafted/nsa_train_sample_v6aug.csv --out-name handcrafted_v6
```

The bundle is `models/hc_lgbm_20260926-191630`. It is not pinned; the runner still loads v5b through `models/hc_selected`.

**In-the-Wild.**
- `_itw_handcrafted_v6.csv` scores the seed-200 v4 In-the-Wild features with the bundle. Applied to v5b, the same code reproduces `_itw_handcrafted_v5.csv` to 1.8e-15.
- `scripts/eval_bundle.py` gives the readout.

**Noise diagnostic.** `scripts/hc_noise_probe.py` imports the channel lane's own cohort and perturbation code from `scripts/m3_probes.py`: 500 outer-holdout clips, one seeded crop each, with `none` and `noise20`. Features go through the runtime path, and both bundles score the same rows.

Self-check: its v5b logits match the channel lane's cached `m3_perturb.csv` values on all 1,000 rows (max |Δ| = 0.0).

## Results

| | v5b (shipped) | v6 (noise twins) |
|---|---|---|
| **Noise cohort, 20 dB white noise: AUC** (bar ≥ 0.90) | 0.640 | **0.772: fails** |
| Noise cohort, 20 dB: minDCF brief / averse | 0.996 / 1.000 | 0.869 / 0.844 |
| **Noise cohort, clean: AUC** (bar: within 0.005 of 0.998) | 0.998 | 0.993: passes by 0.0001 |
| Noise cohort, clean: minDCF brief / averse | 0.048 / 0.084 | 0.163 / 0.140 |
| Inner OOF minDCF (EER) | 0.469 (13.1%) | 0.474 (14.0%) |
| LibriSpeech real, inner | 0.588 | 0.566 |
| Outer holdout minDCF (EER) | **0.137** (3.1%) | **0.227** (4.6%) |
| Holdout, LibriSpeech real | 0.155 | 0.298 |
| Holdout, augmented twins | 0.486 (n 2,250) | 0.762 (n 2,822) |
| In-the-Wild AUC | 0.768 | 0.722 |
| In-the-Wild P_FA / P_miss at the inner threshold | 0.40% / 96.3% | 0.30% / 96.8% |
| Test files above 0.5 | 45.1% | 47.6% |

Per generator, inner OOF (v5b → v6):
- Got worse: pro_diff 0.63 → 0.77, ElevenLabs 0.67 → 0.72.
- Got better: OpenVoice 0.43 → 0.35, unit_speech 0.087 → 0.053, your_tts 0.105 → 0.085.
- grad_tts stays at 1.00.

Sources: the two bundles' `meta.json` and `eval_itw_stress.json`, and `outputs/channel/hc_noise.json` (all gitignored; the numbers above are the record).

## Reading

- **The noise lane recovers only a little.** AUC goes from 0.64 to 0.77, and minDCF under noise stays near 0.87. The bar was 0.90 AUC.
- **Clean rows pay for it.** Clean-cohort minDCF more than triples (0.048 → 0.163), and the outer holdout goes from 0.137 to 0.227, mostly on LibriSpeech real speech (0.155 → 0.298). A model at 0.227 would not have been picked for fusion over v5b's 0.137.
- **Why.** The 234 columns are envelope and fine-structure statistics: spectral flatness and contrast, LFCC/CQCC variability, phase coherence, jitter, the between-speech frames. At 20 dB, white noise fills exactly the low-energy regions most of these cues are read from. The model can't find a noise-invariant version of those cues in these columns. It learns to weaken them instead, which blurs clean rows too. The holdout's own augmented twins show it: 0.486 → 0.762.
- **What it would take.** Noise robustness for this detector would need noise-robust features (e.g. voiced-frame-only statistics or denoising before extraction), not more training rows. That is future work.
- **Probably not a test-set problem.** In the channel lane's per-feature table (`docs/reports/2026-09-26_channel-robustness.md`, §A), the test set's lowest noise-floor percentiles (p2, p10) sit beyond the *clean* end of our two validation sets. Broadband additive noise at 20 dB would raise them. Its floor stationarity (1.08) and flatness (1.74) sit beyond the wild end, though, so the test floor is unusual, not simply clean. Both outside opinions read this as no additive noise on the test set, and we agree, with that caveat.

## Gate

H_noise (the shipped rule with `handcrafted_v6` in the handcrafted seat) goes to the gate chat's `scripts/fuse_sweep_v3.py` under the manifest's rule (`docs/reports/2026-09-26_post-draft-manifest.md`). The evidence sent was `--hnoise-evidence 0.7716,0.9931`. The noise AUC is below 0.90, so H_noise fails rule 5 regardless of its fused numbers.

**Result (gate run 2, 19:17): FAIL.** In the fused rule, every cell is slightly worse than the shipped rule. Δ against CURRENT, brief / averse:
- inner −0.0015 / −0.0061
- holdout −0.0005 / −0.0016
- In-the-Wild −0.0003 / −0.0020

It fails rules 2a, 3a, 3b and 5 (both costs and the diagnostic). Its test ranking has Spearman 0.9953 against the shipped file, with one file crossing 0.5. All four post-draft candidates failed, and the shipped file stays final (`docs/reports/2026-09-26_post-draft-sweep.md`, "Run 2").

## Files

- **Tracked:** `src/hearsay/handcrafted.py` (`white_noise`, the `noise` part in `draw_augment` / `apply_augment`), `scripts/extract_handcrafted.py` (the two options, recorded in the meta), `scripts/hc_noise_probe.py`, `tests/test_hc_augment.py` (one test), this report.
- **Gitignored:**
  - `outputs/handcrafted/nsa_train_sample_v6aug.csv` (~23 min to regenerate at 4 workers; not archived)
  - `models/hc_lgbm_20260926-191630/` (not archived; 17 s to retrain from the feature file)
  - `outputs/detector_scores/{handcrafted_v6,_itw_handcrafted_v6}.csv`
  - `outputs/channel/hc_noise{.json,_features.csv,_scores.csv}`
