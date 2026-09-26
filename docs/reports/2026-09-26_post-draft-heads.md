# Post-draft heads: a frozen WavLM Large probe on the M1b recipe (Sat Sep 26, 2026, 19:10)

**Run spec:** `docs/specs/2026-09-26_post-draft-heads.md` (checks and diagnostics pre-declared before any number was seen). **Handoff:** `docs/handoffs/2026-09-26_post-draft-heads-handoff.md`. **Export for the gate chat:** `outputs/detector_scores/wavlm_l.csv`. This report computes no fused numbers; the gate chat judges the column under its pre-declared entry (0.4·M1b + 0.2·WavLM + 0.2·handcrafted v5 + 0.2·M5, then the unchanged M3 step).

## Result in one paragraph

WavLM Large, frozen and probed exactly like M1b v3, is **level with M1b on the NSA holdout** (minDCF 0.0717 both; EER 1.45% vs 1.4%), **slightly worse on inner CV** (0.2936 vs 0.2649), and **much worse on In-the-Wild** (0.752 vs 0.342 under the brief's cost; 0.383 vs 0.296 under the sponsor-averse cost). At its inner threshold it calls 7.05% of In-the-Wild real speech synthetic, against M1b's 0.8%. It is **not a clone** of M1b (test-set Spearman 0.79), and it **reaches part of the miss tail M1b cannot**: of the shipped rule's 158 In-the-Wild misses, WavLM alone flags 57 at its inner threshold (M1b 1, handcrafted 2, M5 33, Spectra 107). So the mechanism the consult hoped for is real, but on web-sourced real speech it comes with roughly 141 false alarms per 2,000. Whether a 0.2 rank weight keeps the 57 and not the 141 is the gate's question.

## Gate verdict (from the gate chat, commit 57b9c90; not computed here)

The pre-declared entry failed both the pre-declared gate and the room. Details: `docs/reports/2026-09-26_post-draft-sweep.md`.

**The blend** is 0.4 M1b + 0.2 WavLM + 0.2 hc v5 + 0.2 M5, then E. Against the shipped rule:

| | Blend (brief / averse) | Shipped rule | Direction |
|---|---|---|---|
| Inner | 0.1557 / 0.3330 | 0.1351 / 0.2997 | worse |
| Holdout | 0.0030 / 0.0071 | 0.0065 / 0.0087 | better |
| In-the-Wild | 0.2047 / 0.2300 | 0.228 / 0.2385 | better, but short of the room's 0.03 under each cost |

**The gate's diagnostic** counts, at the inner-OOF threshold, the shipped rule's 122 In-the-Wild misses: WavLM catches 41 of them. Its corrective share is 0.80 on In-the-Wild but 0.31 on inner OOF, which fails the diagnostic. This report's 57 of 158 is the same kind of count against the misses at the In-the-Wild argmin.

**The refit seat** (WavLM replacing M1b, diagnostic only) reaches In-the-Wild 0.275 / 0.2665 with In-the-Wild P_FA 5.3%. That is the wild-speech false-alarm problem the Reading section below describes.

**WavLM does not ship.** The recorded negative result stands on its own: a second backbone reaches fakes M1b misses, but it brings more false alarms on wild real speech than a rank blend can absorb.

## Recipe (unchanged from M1b v3)

- **Backbone.** `weights/wavlm-large`: 24 layers, 25 hidden states, 1,024-d, loaded through `transformers.AutoModel`. Its feature-extractor config has `do_normalize: true`, the same zero-mean / unit-variance step `hearsay.embed` applies, so nothing in the input path differs.
- **Input path.** `hearsay.embed.prepare_segment`: band match (7.25 kHz), silence trim, test-length crop, 8 s cap. Then `embed_segment`: per-input normalization, time-mean per layer, fp16.
- **Extraction flags,** copied from each v3 twin's `extract_meta.json` and identical to them (check A1):

| Set | Rows | Crop | Seed |
|---|---|---|---|
| `nsa_train_sample_wl` | 20,000 | test | 0 |
| `asv19_addon_wl` | 7,648 | test | **100** |
| `nsa_test_wl` | 1,671 | none | — |
| `itw_stress_wl` | 3,000 | test | 200 |

  The seed-100 row is a correction: the handoff's command said 0, but `asv19_addon_v3` recorded 100.

- **Probe.** `scripts/train_probe.py --model wavlm-large --train nsa_train_sample_wl,asv19_addon_wl --folds splits/nsa_folds_plus_asv19.csv --stress itw_stress_wl`. The layer is the inner-CV argmin (ASV19 rows are inner-only). StandardScaler + class-balanced logistic (C = 1) + Platt, all fold-local. The holdout is read once and In-the-Wild is never fit on. Model: `models/m1_wavlm-large_L9_20260926-1906` (59 s).
- **Export.** `scripts/export_probe_scores.py --probe models/m1_wavlm-large_L9_20260926-1906 --name wavlm_l --train nsa_train_sample_wl,asv19_addon_wl --folds splits/nsa_folds_plus_asv19.csv --test nsa_test_wl --stress itw_stress_wl`.

## Inner-CV minDCF by layer (selection metric; folds include the ASV19 rows)

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9** | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| minDCF | 0.798 | 0.829 | 0.680 | 0.582 | 0.477 | 0.410 | 0.399 | 0.332 | 0.302 | **0.294** | 0.304 | 0.298 | 0.302 |
| EER | 0.252 | 0.253 | 0.253 | 0.216 | 0.195 | 0.147 | 0.142 | 0.132 | 0.134 | **0.135** | 0.143 | 0.147 | 0.147 |

| Layer | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| minDCF | 0.318 | 0.313 | 0.322 | 0.312 | 0.314 | 0.297 | 0.314 | 0.325 | 0.343 | 0.330 | 0.363 | 0.375 |
| EER | 0.134 | 0.121 | 0.114 | 0.105 | 0.095 | 0.093 | 0.098 | 0.096 | 0.105 | 0.098 | 0.106 | 0.108 |

The curve is flat from layer 8 to layer 18 (0.294–0.322); layer 9 wins narrowly over 18 (0.2974) and 11 (0.2979). M1b's XLS-R curve bottoms at layer 7 (0.2649).

## Readouts beside M1b v3

Normalized minDCF, π = 0.3, C_FA = 4 ("brief") unless marked. M1b v3 from `models/m1_wav2vec2-xls-r-300m_L7_20260926-0521/meta.json`; WavLM from `models/m1_wavlm-large_L9_20260926-1906/meta.json`; the "export" rows from `scripts/heads_diagnostics.py wavlm_l`.

| | M1b v3 (XLS-R L7) | WavLM L9 |
|---|---|---|
| Inner-CV minDCF (trainer; NSA + ASV19 rows) | 0.2649 | 0.2936 |
| Inner-OOF brief / averse (export; NSA rows only) | 0.3013 / 0.3307 | 0.3108 / 0.6241 |
| Holdout minDCF / EER / AUC | 0.0717 / 1.40% / 0.9986 | 0.0717 / 1.45% / 0.9991 |
| Holdout act DCF (Platt at π = 0.3) | 0.0742 | 0.0806 |
| Holdout averse (export) | 0.0571 | 0.0449 |
| Holdout, playht / wavegrad2 | 0.0371 / 0.0892 | 0.0902 / 0.0431 |
| Holdout, bona fide LibriSpeech / LJ | 0.102 / 0.027 | 0.0893 / 0.017 |
| Sponsor code as-is / flipped (holdout) | 1.00 / 0.0571 | 1.00 / 0.0449 |
| **In-the-Wild minDCF, brief** | **0.342** | **0.752** |
| In-the-Wild averse (export) | 0.2955 | 0.3825 |
| In-the-Wild EER | 7.55% | 11.57% |
| In-the-Wild P_FA / P_miss at the inner threshold | 0.8% / 27.8% | 7.05% / 19.9% |
| Inner threshold (LLR) | 2.325 | 1.650 |

## Diagnostics (pre-declared, D4; `scripts/heads_diagnostics.py wavlm_l`)

**Rank agreement with M1b (Spearman of logits):**

| | Holdout (3,858) | Test (1,671) | In-the-Wild (3,000) |
|---|---|---|---|
| WavLM vs M1b | 0.863 | **0.795** | 0.774 |

For reference, Spectra vs M1b on the test set is 0.620 by this script. The handoff quotes 0.57 from another computation; the difference was not chased.

**Miss tail.**
- **Reproduction first.** The shipped rule (A3 w0.2 + E) reproduces on In-the-Wild from the exports: brief 0.2280, **15 FA / 158 miss** at the In-the-Wild brief argmin (check A9).
- **Method.** Each column's inner threshold is the brief-cost argmin over its own inner-OOF logits. Each column's count below is how many of the 158 it flags on its own.

| Column | Flags of the 158 |
|---|---|
| M1b v3 | 1 |
| handcrafted v5 | 2 |
| M5 (XLS-R fine-tune) | 33 |
| **WavLM L9** | **57** |
| Spectra-AASIST (M3) | 107 |

The four reference counts reproduce the Fable consult's numbers exactly (`docs/consults/2026-09-26_post-draft-review_RESPONSE_fable.md`).

## Reading

- **A different backbone does see different fakes.** WavLM flags 57 of the misses, 57 times M1b's count and more than M5, while agreeing with M1b's ranking at only 0.79 on the test set. This is the new information the consult said only a new representation could bring.
- **On the NSA domain it ties M1b on the holdout and loses on inner OOF.** Holdout minDCF ties M1b (0.0717). The generator profile flips: WavLM is better on wavegrad2 and worse on playht. Real LibriSpeech and LJ are slightly better. Inner CV is worse by 0.03, and the inner-OOF averse cost nearly doubles (0.624 vs 0.331): at the averse operating point it misses far more of the inner generators than M1b does.
- **On wild real speech it false-alarms.** P_FA at the inner threshold is 7.05% against M1b's 0.8%, and the brief-cost minDCF doubles. Under C_FA = 4 that is the costly direction. M1b's own In-the-Wild weakness is misses; WavLM's is false alarms. Complementary errors are what a rank blend can exploit, but WavLM's false alarms sit where the blend's threshold operates.
- **Against the gate.** Before the verdict this report named the sponsor-averse In-the-Wild condition as the one to watch because of the false-alarm rate. That was the wrong one.
  - In the blend, In-the-Wild improved under both costs (+0.023 brief, +0.009 averse). It fell short of the room's 0.03 but was not worse.
  - The candidate failed on inner (worse by 0.021 brief and 0.033 averse; the column-alone inner-OOF averse gap above predicted this), on the gate's corrective-share diagnostic, and on thresholds doubled by the 0.967 test-set Spearman of the blend with the shipped rule.

## Incident: transient decode failures under parallel extraction (fixed; D2)

**What happened.**
- WavLM extraction at one clip per forward pass ran 0.17–0.25 s/clip. Serial, it would have finished near 19:15.
- At ~17:40 the serial chain was cut. The four sets then ran in three concurrent processes, and the training sample resumed from its five completed shards.
- Each process grew to ~5.5 GB (the MPS allocator caches buffers for variable-length inputs). The 18 GB Mac swapped (~16 GB of swap in use), throughput fell to 0.4–0.5 s/clip, and FFmpeg failed transiently on files that decode fine on retry.
- The extractor replaces an undecodable clip with 1 s of zeros and records `decode_error` in the shard and in `manifest.csv`, so the failure is recorded. But `hearsay.probe.load_embeddings` does not read the flag, so such a row would have entered training and scoring as a silent clip without warning.

**Counts.**
- 99 rows in total: 74 train sample, 10 ASV19, 4 test, 11 In-the-Wild.
- They include a burst of 51 consecutive train-sample rows (8925–8975) at the swap peak.
- 0 in any v3 set and 0 in the five shards written serially.
- 98 in shards written while three processes ran, 1 (row 10487) in a shard written with two, and 0 in the eight training-sample shards written after the 18:34 restart.

**Fix.**
- The process count dropped to two at ~18:15, and the training-sample process was restarted clean at 18:34. Resumes are bit-identical: crop lengths and per-row seeds are regenerated deterministically.
- New `scripts/repair_decode_errors.py` re-embeds only flagged rows through the extractor's own segment path (the stored crop length, seed = meta seed + row).
- It first re-embeds unflagged rows of the same set (`--verify`, 5 in every run here) and refuses to write unless they reproduce.
- It ran on the test set, In-the-Wild and ASV19 once each after their extraction finished. The fixed row IDs are listed below.
- The train sample's 74 rows were repaired by the finish chain (`outputs/logs/wavlm_finish_chain.log`), which also re-ran ASV19 (a no-op) before training.
- **Tightened after the pre-audit critique.**
  - The first version's identity bar was a whole-matrix max relative error, lax at WavLM's value range; it is now element-wise, |diff| ≤ 5e-3 + 5e-3·|stored|.
  - The tool now refuses an empty verify pool, a mismatched manifest and an unfinished set, and writes shards atomically.
- **Persisted evidence for A1–A5.** The tool's new `--verify-only` mode was run on all four sets after training. It changes nothing. It compared each set with its v3 twin (meta, paths, stored crop lengths), counted decode errors, checked finiteness, and re-embedded 10 random rows per shard plus every repaired row: 426 rows in all, every one matching exactly (max abs diff 0.0). Logs: `outputs/logs/wavlm_verify_{nsa_train_sample_wl, asv19_addon_wl, nsa_test_wl, itw_stress_wl}.json`.
- **Repaired rows.**
  - Test: 1141, 1143, 1164, 1233.
  - In-the-Wild: 1057, 1090, 1390, 2049, 2050, 2051, 2118, 2215, 2224, 2368, 2719.
  - ASV19: 1279, 1303, 1350, 1365, 1377, 2912, 2913, 3369, 3370, 3371.
  - Train sample: the 74 listed in the chain log.
- `tests/test_repair_decode_errors.py` covers it (6 hermetic tests, including a counterfactual: an identity mismatch writes nothing).

**Latent hazard for other lanes.** `extract_embeddings.py` falls back to zeros on a decode failure and only flags the row. Any future extraction under memory pressure should check `manifest.csv`'s `flag` column before training, or run the repair tool.

## Acceptance checks (run spec)

| # | Check | Result |
|---|---|---|
| A1 | `extract_meta.json` equal to v3 twins | pass (all four) |
| A2 | Rows 20,000 / 7,648 / 1,671 / 3,000, paths in v3 order | pass |
| A3 | 0 decode errors after repair | pass (99 repaired, 0 still failing) |
| A4 | Shape (N, 25, 1024), finite | pass |
| A5 | Re-embedding reproduces stored rows | pass: 426 rows re-embedded across all four sets (10 per shard plus all 99 repaired rows), max abs diff 0.0 (`outputs/logs/wavlm_verify_*.json`) |
| A6 | Layer by inner-CV argmin, table recorded | pass (layer 9) |
| A7 | Holdout AUC > 0.5 | pass (0.9991) |
| A8 | Export splits 16,142 / 3,858 / 1,671 / 3,000, path sets equal to `m1b_v3.csv`, no NaN or duplicates | pass |
| A9 | Shipped rule reproduces 15 FA / 158 miss | pass |
| A10 | `pytest -q` all pass; `ruff check .` clean | pass; see the run spec's Results for the counts at each commit |

## Timing

| Stage | When (Sat Sep 26) |
|---|---|
| Extraction | 17:19–19:05, including the parallel detour; a clean serial run would take ~1 h 55 min at 0.215 s/clip |
| Train-sample shards after the 18:34 restart | 0.10–0.41 s/clip per shard, 0.23 cumulative |
| Repair, validation, training, export | 19:05–19:07 (training 59 s) |

## Artifacts (gitignored; not archived)

- `outputs/embeddings/wavlm-large/{nsa_train_sample_wl, asv19_addon_wl, nsa_test_wl, itw_stress_wl}`: ~1.5 GB fp16, ~2 h to regenerate.
- `models/m1_wavlm-large_L9_20260926-1906/`: `probe.joblib` and `meta.json`; its numbers are copied above.
- `outputs/detector_scores/wavlm_l.csv`: regenerable from the probe in about a minute.
- Logs: `outputs/logs/wavlm_extract_*.log`, `outputs/logs/wavlm_finish_chain.log`.
