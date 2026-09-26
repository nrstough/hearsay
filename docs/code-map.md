# HEARSAY code map

HEARSAY scores audio clips for synthetic-speech likelihood: CPU contract detectors (rule-based and learned), frozen-SSL probes (M1), an off-the-shelf anti-spoof scorer (M3) and a fine-tune lane (M5) each export fold-validated scores, which are fused into one score per test file and written as a TSV submission judged by normalized minDCF (C_FA = 4, C_MISS = 1, pi_synth = 0.3).
Read first: docs/STATUS.md (one-page state), docs/architecture.md (component diagram, built / in-flight / planned tags), then this file for where things live and where to plug in. Every entry below cites file:line as reported by the finder pass on Sat Sep 26; re-check a line before editing.

## Repo layout

```
src/hearsay/
  audio.py          decode/resample (ffmpeg -> 16 kHz mono float32), windows(), trim_silence()
  embed.py          frozen SSL embedding: v1 window path, v2 prepare_segment + embed_segment
  probe.py          M1 logistic probe on one SSL layer (load_embeddings, oof_scores, Probe)
  handcrafted.py    v3 75-feature extractor, band_limit (NSA 7.25 kHz wall), _crop, ctx for v4 families
  hc_v4.py          v4 feature families registry FAMILIES (lfcc, phase, cqcc, modulation, breath, jitter)
  compression.py    compression-forensics features + launder() codec round-trip
  trees.py          pure-numpy LightGBM inference and contributions (no lightgbm import at inference)
  metrics.py        EER, minDCF, cost_at, report(); the NSA cost constants
  submission.py     score_files, write_submission (TSV, never overwrites), append_log, preflight
  spectra.py        Spectra-AASIST loader and scorer (M0-era version only)
  m5_bundle.py, m5_data.py, m5_model.py    M5 fine-tune lane library
  detectors/        contract (base.py), shared learned base (_learned.py), handcrafted, compression,
                    container, enf, splice, engineered.py (single import point)
scripts/            CLIs: extract_/train_handcrafted, eval_bundle, extract_embeddings, train_probe,
                    make_probe_csv, make_folds, extend_folds, score_detector, extract_compression,
                    m5_*, spectra_direction, manifest_*, pull_mlaad, hash_audio, bench_latency,
                    make_constant_csv, inventory*
scripts/cloud/      launch.sh, box_chain.sh, reaper.sh, r2_guard.sh, teardown_check.sh, box_codecs.py
splits/             nsa_folds.csv (20,000 rows), nsa_folds_plus_asv19.csv (27,648), nsa_test_durations.csv (1,671)
tests/              pytest suite, 301 collected
docs/               STATUS.md, architecture.md, plan.md, nsa-challenge.md, handoffs/, reports/, specs/, consults/
models/ outputs/ weights/ data/ submissions/    gitignored artifacts and logs
```

## Handcrafted features (D-track feature-engineering lane)

| Path | Purpose | Key entry points |
|---|---|---|
| src/hearsay/hc_v4.py | v4 feature families; each is a function of a ctx dict registered in FAMILIES | FAMILIES:300, lfcc:59, phase:165, cqcc:174, modulation:197, breath:221, jitter:247 |
| src/hearsay/handcrafted.py | v3 extractor: band_limit -> crop -> RMS-normalize, then spectral + prosody features; `families` kwarg appends hc_v4 columns | features:72, features_for_path:143, ctx dict:136, band_limit:43, _crop:55 |
| scripts/extract_handcrafted.py | manifest -> parallel feature extraction -> outputs/handcrafted/<name>.csv + .meta.json sidecar | --families flag:48, families tuple:51, executor map:64, meta.json families:78 |
| scripts/train_handcrafted.py | trains logreg/LightGBM on a feature .csv via PredefinedSplit folds, exports fusion scores, saves bundle | META_COLS:46, META_SUFFIXES:48, FAMILY_PREFIXES:51, is_meta/dropped:55, side_meta read:138, bundle construction:216 |
| scripts/eval_bundle.py | eval-only: score any labeled feature .csv with a saved bundle at its own inner-OOF threshold (ITW stress) | best_threshold:30, main:37 |
| src/hearsay/detectors/_learned.py | FeatureModelDetector base: loads bundle, recomputes features with the bundle's families, scores, builds evidence | FeatureModelDetector.run:88, families kw:92, _evidence:102, contributions:51 |
| src/hearsay/detectors/handcrafted.py | contract wrapper registering `handcrafted`; owns LABELS | LABELS:21, HandcraftedDetector:101 |

Where to add

- New feature family: src/hearsay/hc_v4.py:300-303, the FAMILIES dict literal. Write `fn(ctx) -> dict[str, float]` and add it under a new key (the key is the `--families` CLI name). ctx keys (built at src/hearsay/handcrafted.py:136-137): x (float32 audio), Z (complex STFT 512/160), S (power STFT), freqs, voiced (bool frame mask), rms_db, f0 (YIN track). Read Z/S from ctx; do not recompute the STFT.
- Column-name rules (scripts/train_handcrafted.py:46-48): unique, prefixed by family, not in META_COLS (path, label, generator, speaker, utt, source, filename, group, fold), not ending in _flag / _launder / _crop_s.
- Feature-content rules (tests/test_hc_v4.py): no content above 7000 Hz (FMAX), no absolute level, no duration or count features (means, stds, percentiles, rates only), offset-invariant (rel_tol 0.25 / abs_tol 0.05 on a shifted copy).
- scripts/train_handcrafted.py:51 FAMILY_PREFIXES: add `name -> (prefix, ...)`. Without it, `FAMILY_PREFIXES.get(fam, (fam,))` (used at lines 143-144) matches the family name as a literal column prefix, which is wrong for families like phase (gd_/pc_).
- src/hearsay/detectors/handcrafted.py LABELS (ends line 92): one plain-English entry per new column (or a dict-comprehension block like lfcc/cqcc/mfcc). Missing entries fall back to the raw column name at src/hearsay/detectors/_learned.py:109; the brief treats labels as required for judged evidence quality.
- src/hearsay/handcrafted.py:133-139 needs no change: the families tuple flows straight into `FAMILIES[name](ctx)`.
- Add the family's expected key set to the EXPECTED dict in tests/test_hc_v4.py:33-44.

Invariants you must keep

- features() maps every non-finite value to 0.0 at src/hearsay/handcrafted.py:140; families need not self-guard.
- Unknown family name raises KeyError (src/hearsay/handcrafted.py:139; tests/test_hc_v4.py test_unknown_family_raises).
- With families=() the extractor returns exactly the 75 v3 columns (tests/test_hc_v4.py, len(features(x)) == len(V3)).
- feat_cols = columns that are not is_meta and not dropped, in file order; that order is saved as bundle['features'] and is the order _learned.py builds the inference vector in.
- At inference kw['families'] = tuple(b['families']) (src/hearsay/detectors/_learned.py:92-93), so the bundle recomputes exactly the families it was trained with (tests/test_handcrafted_detector.py:141).
- scripts/eval_bundle.py:53 `assert not missing`: the feature file must already contain every bundle['features'] column; eval_bundle never recomputes features from audio.
- Output .csv of extract_handcrafted = manifest columns + hc_crop_s + one column per feature; failures are flagged in hc_flag (line 73), never dropped as rows.

Gotchas

- hc_v4 is imported lazily inside features() only when families is non-empty (src/hearsay/handcrafted.py:134); importing hearsay.handcrafted alone does not pull it in. group_delay and cqcc import librosa/scipy inside the function body.
- `--drop-columns` (exact name or `prefix*`) is a training-time gate: a variant trains without re-extracting, per the v4 report's corpus-cue blocklist.
- eval_bundle's threshold comes from the bundle's own detector_scores inner_oof rows joined to splits/nsa_folds.csv labels (scripts/eval_bundle.py:59-63), not from the --features file.
- band_limit is defined at src/hearsay/handcrafted.py:43 (one finder cited 44; 43 is correct).

## Engineered detectors (contract, rule-based, compression, tree evaluator)

| Path | Purpose | Key entry points |
|---|---|---|
| src/hearsay/detectors/base.py | the contract: DetectorResult, ClipContext, Detector protocol, safe_run, name-sorted Registry | NEUTRAL_SCORE:40, ClipContext:44, from_array:56, audio property:71, probe property:87, memo:93, DetectorResult:110, __post_init__:122, Detector:149, safe_run:163, Registry:195, singletons:220 |
| src/hearsay/detectors/enf.py | rule-based mains-hum consistency check; always applies; never learned | _frames:35, analyze:51, EnfDetector:92, run:100, register guard:121 |
| src/hearsay/detectors/splice.py | rule-based editing-seam detector (clicks, DC-offset jumps) | constants:30-33, analyze:46, SpliceDetector:84, run:92, register guard:112 |
| src/hearsay/detectors/container.py | rule-based container/metadata score plus routing facts (lossy, is_pcm_wav, ffmpeg_written) | LOSSY/AI_TOOL_TERMS/SYNTH_TERMS:30-41, probe_container:44, tts word-boundary regex:80, classify_tags:88, ContainerDetector:100, run:108, routing features:117-128, register guard:144 |
| src/hearsay/compression.py | compression-forensics features (bandwidth, spectral holes, floor depth, tilt) and launder() | LAUNDER_CODECS:41, features:46, launder:105, draw_laundering:132, features_for_path:150 |
| src/hearsay/detectors/compression.py | thin wrapper wiring hearsay.compression.features into FeatureModelDetector | LABELS:24, latest_model_dir:49, CompressionDetector:53, register guard:64 |
| src/hearsay/detectors/_learned.py | shared machinery for the two learned detectors: bundle discovery, predictor, contributions, generic run() | TOP_K:28, latest_model_dir:31, predictor:41, contributions:51, FeatureModelDetector:61, bundle property:75, applies:85, run:88, clip:96, _evidence:102 |
| src/hearsay/trees.py | numpy re-implementation of LightGBM binary inference + Saabas contributions from dump_model() | Trees.__init__:25, decision_type check:41, raw:53, predict_proba:72, contrib:76 |
| src/hearsay/detectors/engineered.py | single import point registering all five D-track detectors; LEARNED vs RULE_BASED lists | import line:16, LEARNED/RULE_BASED:18-19 |

Where to add

- New rule-based contract detector: copy the shape of src/hearsay/detectors/enf.py (123 lines): module docstring with rationale and score calibration, `applies()` returning True, `run()` memoizing one `analyze(ctx.audio)` via ctx.memo, DetectorResult built from named constants, then `DETECTOR = X(); if NAME not in REGISTRY.names(): register(DETECTOR)` at module bottom (enf.py:121-123, container.py:144-145, detectors/compression.py:65-66). Constraint: name must be unique or register raises ValueError (base.py:206). Without label coverage keep scores mild (enf uses 0.4/0.5/0.6, enf.py:9-11 and 32; splice uses 0.5/0.6, splice.py:33).
- New learned contract detector: subclass FeatureModelDetector (src/hearsay/detectors/_learned.py:61-114) exactly as src/hearsay/detectors/compression.py does: define feature_fn (audio -> dict[str, float]), a LABELS dict, set name/prefix/kind_label/memo_key/labels/feature_fn, register at import. Bundle must contain features, kind ('logreg' or 'lgbm'), feature_stats ({name: {real_mean, real_std, auc}}), crop_mode, band_match, and model or trees_dump (_learned.py docstring lines 7-12). prefix must match models/<prefix>_<kind>_<stamp>/ (_learned.py:31-38).
- Register the new module in src/hearsay/detectors/engineered.py:16-23 (import line plus LEARNED or RULE_BASED at lines 18-19); otherwise tooling that only imports engineered misses it.
- New AI-tool or synthesis vocabulary: src/hearsay/detectors/container.py:30-41 tuples; matching is substring except 'tts' (word-boundary regex, lines 80-81).
- Training-time laundering: src/hearsay/compression.py launder():105 (mp3 or aac only, ValueError otherwise at line 110), driven by scripts/extract_compression.py on a random half of both real and spoof rows.

Invariants you must keep

- run() is only ever invoked through safe_run (base.py:163); it skips when applies() is False, and any exception, wrong type, name mismatch or non-ok status becomes status='error' with score NEUTRAL_SCORE = 0.5 (base.py:40).
- features values must be finite (base.py:104-105); DetectorResult.__post_init__ (line 122) raises, never clips.
- safe_run wraps the result in dataclasses.replace (base.py:186), so a detector cannot keep mutating a returned features dict.
- Registry rejects duplicate names (base.py:205-206); names()/all_detectors() iterate sorted by name so fusion columns are stable.
- enf score is always in {0.4, 0.5, 0.6} (enf.py:32) and features include all analyze() keys (enf.py:118); splice score is always 0.5 or 0.6 (splice.py:33); classify_tags never raises and returns one of exactly three pairs.
- container routing features is_pcm_wav (container.py:119) and lossy (line 118) gate other detectors.
- hearsay.compression.features output is finite (compression.py:102); LAUNDER_CODECS weights MP3 3:1 over AAC (line 41).
- predict_proba is clipped away from 0/1 (_learned.py:96) before the logit.
- Trees.contrib + bias reproduces raw() exactly (tests/test_trees.py:32); raw()/predict_proba match real LightGBM within 1e-9 on tests/fixtures/lgbm_toy.json (tests/test_trees.py:22); no source or test file imports lightgbm (tests/test_trees.py:69).

Gotchas

- NEUTRAL_SCORE is a sentinel: fusion never feeds it in as a real score, it imputes the fold-local train mean plus a missing indicator (base.py:14-17).
- src/hearsay/trees.py exists because lightgbm and torch each ship libomp and loading both in one process fails (trees.py:4-9); do not reintroduce a direct lightgbm import at inference.
- splice floor_range_db is reported in features/evidence (splice.py:66-71, 108) but does not move the score (docstring lines 12-14).
- container scoring is rule-based on purpose: on training data the container is the label and the NSA test set is uniform PCM16/16 kHz (container.py:13-17).
- enf's mild scores are a stated limitation (no hum in training corpora, enf.py:6-11), not an oversight.
- ContainerDetector.run raises RuntimeError when ffprobe finds no audio stream; safe_run turns that into an error result.

## M1 deep path (frozen SSL embedding -> logistic probe -> submission)

| Path | Purpose | Key entry points |
|---|---|---|
| src/hearsay/audio.py | single decode path plus windowing and silence trimming shared by every rung | load_audio:22, ffmpeg cmd:27-32, DecodeError raise:36, NaN zeroing:41, windows:74, tile padding:82, trim_silence:91, min_keep guard:106 |
| src/hearsay/embed.py | frozen backbone embedding: v1 windows averaged, v2 single prepared segment | REPO:18, MAX_SEGMENT_S:19, TEST_DURATIONS:20, load_backbone:28, clip_windows:32, normalize_windows:36, embed_windows:44, embed_clip:55, test_duration_sampler:71, prepare_segment:79, embed_segment:107 |
| src/hearsay/probe.py | standardized logistic regression on one layer, grouped CV, Platt calibration | load_embeddings:26, cv_groups:38, make_clf:43, oof_scores:49, PredefinedSplit branch:58, Probe fields:65-73, llr:79, p_synthetic:86, p_decision:90, save:94, load:99 |
| scripts/extract_embeddings.py | bulk, sharded, resumable extraction to outputs/embeddings/<model>/<name>/ | main:43, crop draw:70, extract_meta.json:71-73, shard loop:78-132, skip-existing:81-86, segment branch:87-111, per-row seed:90, window branch:112-132, manifest write:134-136 |
| scripts/train_probe.py | per-layer grouped CV, Platt, refit, score val once, write models/m1_<model>_L<layer>_<stamp>/ | main:32, --folds branch:46-60, --max-train:56-58, layer loop:70-75, extract_meta read:81-85, val block:87-106, auc assert:89, --stress:108-121, meta val:126 |
| scripts/make_probe_csv.py | score every test file with a saved probe, write submission TSV + sidecar, log | main:40, shift:55-57, score closure:60, segment branch:61, raw append:66, fb:74, preflight:75, raw.clear:76, raw_col:85-86 |
| scripts/make_folds.py | base outer-holdout + k inner folds grouped by generator/speaker | group_key:28, main:36, holdout selection:50-58, generator assert:51, min-per-class assert:69, straddle assert:71, columns:74, known limits:77-78 |
| scripts/extend_folds.py | append training-only rows into inner folds without touching holdout | main:23, group key + fold assign:34-40, overlap assert:41, straddle assert:43, holdout count:48 |
| src/hearsay/handcrafted.py (shared) | band_limit and _crop used by prepare_segment and the CPU detectors | BAND_MATCH:39, band_limit:43, _FIR cache:40, _crop:55, segment call:61-64, RMS normalize:68 |
| src/hearsay/metrics.py | NSA cost metrics used for layer selection and reporting | constants:29-31, cost_at:45, min_cost:54, decision_logit:67 |

Where to add

- Decode args (resample rate etc.): src/hearsay/audio.py:27-32 only; every consumer goes through load_audio. Changing SR (src/hearsay/__init__.py:3) requires re-extracting all embeddings.
- A new embedding mode: add a field to the Probe dataclass (src/hearsay/probe.py:65-73) and a branch in scripts/make_probe_csv.py:61, which currently switches on probe.segment.
- prepare_segment band_match flag (src/hearsay/embed.py:80): callers that already band-limited (handcrafted._crop, src/hearsay/handcrafted.py:62) pass band_match=False. MAX_SEGMENT_S (embed.py:19) caps v2 segments; changing it invalidates probes trained under the old cap.
- Extra training corpora: scripts/extend_folds.py:34-40 builds group = source + ':' + (generator if spoof else speaker) and assigns fold ids '0'..'k-1' as strings, never 'holdout'. Verify a new corpus needs band matching or is already band-limited (src/hearsay/handcrafted.py:26-30) or the probe may learn the wall as a cue.
- Generalization check: scripts/train_probe.py:108-121 `--stress` loads a never-fit-on embedding set and reports minDCF/EER at its own threshold and at the train-OOF Bayes threshold (line 114); it never refits.
- Resumable extraction: scripts/extract_embeddings.py:81-86 skips existing shard_XXXXX.npz; rerun the same command to resume.

Invariants you must keep

- All train/test/live audio passes through load_audio (src/hearsay/audio.py:1-4); trim_silence never returns less than min_keep_s = 1.0 s (audio.py:106).
- embed_clip and the bulk window path produce identical features (tests/test_audio_and_submission.py:179).
- prepare_segment order is band -> trim -> crop -> cap (embed.py:82-85, 96-104); it never pads or tiles a short clip (tests/test_audio_and_submission.py:270). scripts/extract_embeddings.py and scripts/make_probe_csv.py must call it identically or train/test features diverge.
- load_embeddings asserts every manifest row is covered exactly once (src/hearsay/probe.py:34) and reorders by row (line 35); a partial extraction fails loudly.
- CV, layer selection and Platt fit touch only train (or non-holdout fold rows); val is scored once (scripts/train_probe.py:87-106); `assert auc > 0.5` at line 89.
- With a fold file, oof_scores uses PredefinedSplit (src/hearsay/probe.py:58-59), never a fresh StratifiedGroupKFold.
- make_folds: every fold has >= --min-per-class per class (scripts/make_folds.py:69), no group straddles folds (line 71), output columns are exactly path,label,generator,speaker,source,group,fold (line 74). extend_folds: no path overlap with base (scripts/extend_folds.py:41), no group straddles folds (line 43), holdout count unchanged (line 48).
- write_submission never overwrites (src/hearsay/submission.py:85-86) and validates range, uniqueness and length.
- C_FA = 4.0, C_MISS = 1.0, PI_SYNTH = 0.3 (src/hearsay/metrics.py:29-31); scripts default --pi-synth to 0.3.

Gotchas

- windows() tiles short clips (src/hearsay/audio.py:82-83); prepare_segment never does. The same manifest extracted with and without --segment yields structurally different statistics.
- Two different per-row randomness schemes: crop lengths come from one shared RNG stream in test_duration_sampler (embed.py:76) drawn once per manifest row in order (scripts/extract_embeddings.py:70), while the crop offset uses default_rng(args.seed + r) (embed.py:102, extract_embeddings.py:90). Manifest row order changes crops. Any rescoring script must reproduce both.
- Resuming after changing --seed, --crop, --win-s or --max-windows silently mixes settings across shards (scripts/extract_embeddings.py:81-86, 134-136); delete stale shards or use a new --name.
- If extract_meta.json is missing, train_probe silently defaults probe.segment=False (scripts/train_probe.py:81-85).
- cv_groups (src/hearsay/probe.py:38) is used only on the non-fold path of train_probe; the --folds path uses the fold file's group column.
- The sidecar llr column is NaN for rows flagged decode_error or score_error (scripts/make_probe_csv.py:85-86); fb (line 74) is sigmoid(shift), the prior-only posterior, not 0.5 in general.
- band_limit's _FIR cache (src/hearsay/handcrafted.py:40) is built once per process; changing BAND_MATCH (line 39) needs a restart.
- make_folds documents known leakage limits (DiffSSD multi-speaker generators share cloned targets; LJ-voice generators share the LJ voice) at scripts/make_folds.py:9-11 and 77-78; they are reported, not silently fixed.
- src/hearsay/metrics.py:13-15 flags an open sponsor-formula ambiguity (P_FA weighting), unresolved.

## M5 fine-tune lane (XLS-R 300M fine-tune)

| Path | Purpose | Key entry points |
|---|---|---|
| src/hearsay/m5_bundle.py | decode every clip once, trim as at test time, write 16 kHz PCM16 FLAC + shortcut features + PCM hash; tree-sha the bundle | MAX_CLIP_S:20, PEAK:21, prepare_clip:35, write_clip:44, clip_ok:75, tree_sha:86 |
| src/hearsay/m5_data.py | manifest construction with fold discipline, crop/batch pipeline, deployment transforms | FOLDS shas:28-29, FAMILY_EXCLUSIONS:36-40, MAX/MIN_CROP_S:42-43, DEPLOY_MAX_S:44, check_folds_file:55, group_key:63, build_manifest:81, ITW guard:150-151, training_rows:166, validation_rows:182, holdout_rows:189, shortcut_aucs:193, LengthMixer:223, crop:238, collate:247, deploy_transform:258, band_match:264, bundle_transform:272 |
| src/hearsay/m5_model.py | truncated XLS-R + layer-weighted sum + masked attentive stats pooling + linear head; hash-verified save/load | M5Config:22, keep_layers:23, train_top:24, pooling:26, bottleneck:27, mix:43-47, TINY config:53-57, AttnStatsPool:60, M5Net.forward:96, build_model:113, param_groups:138, save_m5:170, load_m5:199, hash refusal:206-212, score_batch:223 |
| scripts/m5_build_manifest.py | writes outputs/manifests/m5_manifest.csv from build_manifest | main:32, --extra-spoof-cap:39, fold sha checks:44-46 |
| scripts/m5_build_bundle.py | materializes the bundle: parallel decode, drop rules, ASV19 duration matching, cross-scope PCM dedup, code.tgz, tree sha | DEFAULT_OUT:39, make_code_tgz:68, main:84, decode pool:133-142, drop rules:165-217, core FATAL:172, error-frac abort:174-175, duration matching:180-194, collision dedup:196-217 |
| scripts/m5_train.py | the trainer, one fold or full model, wall-clock-fixed steps, exports OOF/holdout/test scores | main:60, band_match after augment:104, --cpu-smoke:218, CUDA check:230, build_model:312 |
| scripts/m5_score.py | CPU inference path: manifest or file list -> path/score/logit/flag .csv | score_paths:30, decode flag:36-38, main:49, --max-s:58 |
| scripts/m5_assemble.py | assembles pulled runs into exported scores + meta.json with the M1 gate decision | newest_m1_dir:41, find_runs:54, _m1b_comparison:68, --m1-itw-pfa:119, have_all_folds:133-137, oof assert:147, holdout/test asserts:151-152, itw comparison:190-204, gate block:205, holdout_passed:208, gate rule text:212-214 |
| scripts/cloud/launch.sh | rents one cloud instance, budget-guards, provisions, starts box_chain.sh; self-destroys on early failure | budget guard:33, NEED:39, COMMIT:40, refuse:41-42, cleanup trap:62 |
| scripts/cloud/box_chain.sh | runs on the instance: setup -> JOBS -> DONE, heartbeats STATUS to R2, polls for NEXT | fail():15, EXIT trap:16, run_jobs:30, NEXT poll:60-70 |
| scripts/cloud/reaper.sh | local watchdog: polls STATUS in R2, destroys on DONE/FAIL/stall/deadline, appends the ledger | caffeinate note:4, DEADLINE/STALL_MIN:11-12, main loop:21, ledger append:45, never-exit-while-active:50-55 |
| scripts/cloud/r2_guard.sh | every R2 path must be under r2:pa-source/hearsay/ | HEARSAY_R2_PREFIX:4, r2_path:6, r2_copy:15 |
| scripts/cloud/teardown_check.sh | manual backstop: destroys every instance on the account (or --list) | main:10, nonzero-if-remaining:18 |

Where to add

- Exclude an extra spoof generator family from a fold's training: src/hearsay/m5_data.py:36-40 FAMILY_EXCLUSIONS (fold -> substring tuples); must match that fold's validation generator families.
- Per-generator extra-spoof cap: scripts/m5_build_manifest.py:39 `--extra-spoof-cap` (default 1000).
- Bundle output location: scripts/m5_build_bundle.py:39 DEFAULT_OUT is a machine-specific path; override with --out.
- CPU test path: scripts/m5_train.py:218 `--cpu-smoke` runs the identical code on the TINY random-init config (src/hearsay/m5_model.py:53-57).
- Deployment cap: scripts/m5_score.py:58 `--max-s` (default DEPLOY_MAX_S = 8.0).
- ITW P_FA bar fallback: scripts/m5_assemble.py:119 `--m1-itw-pfa` (default 0.012); lines 187-188 prefer M1's own meta.json value.
- Watchdog timing: scripts/cloud/reaper.sh:11-12 DEADLINE / STALL_MIN env vars; NEXT wait: scripts/cloud/box_chain.sh:60-70 WAIT_MIN (default 45).

Invariants you must keep

- MAX_CLIP_S = 15.0 and PEAK = 0.999 (src/hearsay/m5_bundle.py:20-21).
- Fold-file shas pinned (src/hearsay/m5_data.py:28-29) and enforced by check_folds_file (line 55); In-the-Wild paths never enter the training manifest (lines 150-151); DEPLOY_MAX_S = 8.0 shared with M1 and every detector (line 44); crops in [3.0, 14.0] s (lines 42-43).
- keep_layers = 12, train_top = 12 by default (src/hearsay/m5_model.py:23-24); load_m5 refuses on any hash mismatch (lines 206-212).
- Bundle build: core and test rows are never dropped, a core decode error is FATAL (scripts/m5_build_bundle.py:172); extra decode-error fraction under --max-error-frac (default 0.005) or abort (lines 174-175); cross-scope PCM collisions drop the extra row and log dropped_extra_collisions.csv (lines 206-210); core/test overlap is reported in test_overlap.csv, never hidden (lines 211-213).
- Trainer: trains where CUDA is available and exits otherwise unless --cpu-smoke (scripts/m5_train.py:230); refuses on bundle tree-sha or XLS-R config-sha mismatch (lines 9-10); refuses if any trivial shortcut AUC exceeds --shortcut-max on its own training rows (shortcut_aucs, src/hearsay/m5_data.py:193); one crop length per batch (LengthMixer, m5_data.py:223-235); steps are wall-clock-fixed, never checkpoint-picked on validation.
- band_match is applied to every clip the model sees, after augmentation (src/hearsay/m5_data.py:264, scripts/m5_train.py:104).
- Assembler: len(oof) == len(inner) == 16142 (scripts/m5_assemble.py:147); holdout 3858 and test 1671 (lines 151-152); no partial stacking set: a missing fold model means no inner_oof rows and stackable=false (lines 133-137); the M1 bar is the newest models/m1_* whose meta.json `train` is a single NSA set with no comma (lines 41-48), M1b reported beside it and never gated on.
- Cloud: no unprovisioned instance is left running (scripts/cloud/launch.sh:62-63); all R2 paths are under HEARSAY_R2_PREFIX (scripts/cloud/r2_guard.sh:4); status and log are pushed to R2 on EXIT (scripts/cloud/box_chain.sh:16); the reaper never exits while a job is active (scripts/cloud/reaper.sh:50-55).

Gotchas

- scripts/m5_score.py flags decode errors per row (lines 36-38) rather than failing.
- deploy_transform (src/hearsay/m5_data.py:258) and bundle_transform (line 272) differ only in trim / band-limit order (tested to shift the boundary by at most 1 frame).
- The gate rule text (scripts/m5_assemble.py:212-214) names three conditions (holdout, pooled OOF, ITW P_FA) but only holdout_passed (line 208) is a computed boolean; the other two are separate fields a caller must AND itself.
- reaper.sh must run under caffeinate; Ctrl-C does not destroy instances (scripts/cloud/reaper.sh:4). teardown_check.sh destroys every instance on the account, not only M5 ones.
- m5_data.group_key (src/hearsay/m5_data.py:63) must stay identical to scripts/make_folds.py:group_key per its docstring; not diffed in the finder pass.

## M3 Spectra-AASIST stream and shared validation/submission machinery

| Path | Purpose | Key entry points |
|---|---|---|
| src/hearsay/spectra.py | loads the Spectra-AASIST checkpoint from weights/Spectra-AASIST and produces synth_logit per clip; M0-era version only (63 lines) | WIN/HOP:26-27, load_spectra:30, spectra_logits:54, synth_logit:62 |
| src/hearsay/metrics.py | EER, minDCF (oracle and sponsor-code variants), act_dcf, report() | C_FA/C_MISS/PI_SYNTH:29-31, eer:34, cost_at:45, min_cost:54, bayes_llr_threshold:62, decision_logit:67, sponsor_min_dcf:78, report:99 |
| src/hearsay/submission.py | score every test file in fixed order, write a validated never-overwritten TSV, append the log | LOG_COLUMNS:29-31, score_files:44, bare except:60, write_submission:73, score_max:80, FileExistsError:86, read-back asserts:107-109, append_log:113, AUDIO_EXT:136, list_test_files:139, preflight:148 |
| scripts/spectra_direction.py | frozen direction guard: AUC >= 0.5 and spoof median > bonafide median; optional diagnostic Platt | main:45, direction fail:88-89, Platt fit:92-108, slope assert:95 |
| scripts/score_spectra.py | NOT PRESENT; spec D8 (docs/specs/2026-09-26_m3-spectra-aasist.md:54-55) describes its CLI | none |
| scripts/make_constant_csv.py | M0 rollback: constant-score submission through the real loader | main:33, decision:44, log row:62 |
| scripts/manifest_nsa.py | NSA training manifest from DiffSSD + LJ Speech + LibriSpeech | diffssd_rows:29, lj_rows:43, libri_rows:51, main:58 |
| scripts/manifest_asvspoof19.py | ASVspoof2019 LA train/dev/eval manifests | PROTO files:18-22, main:25 |
| scripts/pull_mlaad.py | generator-sampled English MLAAD subset via the HF Hub API | main:41 |
| scripts/hash_audio.py | file-bytes + decoded-PCM hash index for leakage detection | hash_one:31, main:40 |
| scripts/bench_latency.py | one SSL front-end forward pass per 4 s window against a 750 ms p95 gate | GATE_P95_MS:23, main:33 |
| tests/test_docs_consistency.py | tripwire suite pinning living docs to code (A1-A8) | RUBRICS:22, LIVING:23, BASE_SHA:24, BANNER_PREFIX:25, MAX_CUT_LINES:27, A1:72, A2:83, A3 constants:107, A3 no conflict:116, A4:124, A5:133, A7:156, A8:178 |
| splits/nsa_folds.csv | 20,000 rows; holdout 3858, inner folds 0-4 sum to 16,142; 10,000 bonafide / 10,000 spoof | columns path,label,generator,speaker,source,group,fold |
| splits/nsa_test_durations.csv | 1,671 rows, columns filename, duration_s | basis for the short-clip claim in the M3 spec |

Where to add

- M3 extensions land in src/hearsay/spectra.py spectra_logits (lines 54-59): spec D4 asks for a pad_mode parameter; spec D2/D3 requires the input to go through hearsay.embed.prepare_segment before windowing, which the current file does not call.
- hearsay.metrics.by_group (spec D9) is to be copied verbatim from scripts/train_handcrafted.py into src/hearsay/metrics.py after report() (past line 106); not present yet.
- write_submission score_max (src/hearsay/submission.py:80) is the only knob if the score ceiling ever changes from [0, 1].
- Any new .claude/skills/*/SKILL.md or codex-*.md rubric is pulled into every living-doc check automatically (tests/test_docs_consistency.py:22-23); it must state the pinned constants correctly and avoid banned words.

Invariants you must keep

- WIN = 64,600 samples, HOP = WIN // 2 (src/hearsay/spectra.py:26-27); synth_logit = logit_spoof - logit_bonafide and must increase with synthetic likelihood (line 62-63).
- Direction gate: auc < 0.5 or median_synth_spoof <= median_synth_bonafide is a hard failure (scripts/spectra_direction.py:88). Its Platt fit is diagnostic, marked provisional, and must never be fit on the NSA validation split or In-the-Wild (docstring lines 11-12).
- Submission format: TSV, header `filename<TAB>cm-score`, score in [0, 1], 0.0 = bona fide, 1.0 = synthetic (src/hearsay/submission.py:3-5). write_submission never overwrites (line 86) and re-reads the file to assert header, order and values (lines 107-109). LOG_COLUMNS order is fixed: timestamp, rung, validation_score, validation_score_clean_only, csv_path, notes (lines 29-31).
- score_files records `fallback` plus a flag ('decode_error', 'score_error:<Type>', 'nonfinite_score') for every failing file, guaranteeing one row per input (src/hearsay/submission.py:44).
- min_dcf at pi_synth = 0.3 is the sponsor-judged number (src/hearsay/metrics.py:19); reports lead with it, not EER. C_FA = 4.0, C_MISS = 1.0, PI_SYNTH = 0.3 (lines 29-31) are pinned by tests/test_docs_consistency.py:107, which also requires CLAUDE.md and docs/plan.md to contain the literal strings 'C_FA = 4' and 'π_synth = 0.3'.
- Living docs (CLAUDE.md, README.md, docs/plan.md, every .claude/skills/*/SKILL.md, the two codex-*.md rubrics) may not spell out the comma-separated-values word (the TSV-format rule) outside the allowlist (tests/test_docs_consistency.py:124), may not state a conflicting prior (line 116), and may not contain dropped-scope vocabulary except on lines marked '(cut)', capped at MAX_CUT_LINES = 6 (lines 27, 72).
- docs/scoping.md and docs/master-doc.md must equal their content at git sha e1a1e88 plus exactly one banner line; the first handoff must be byte-identical (tests/test_docs_consistency.py:156).

Gotchas

- scripts/score_spectra.py and tests/test_spectra.py do not exist; git log on src/hearsay/spectra.py shows one real commit (0e2df3d, M0). The M3 spec describes planned work that has not landed.
- The m3val map left it unverified whether hearsay.audio.windows() repeat-pads short clips; the deep map confirms it tiles them (src/hearsay/audio.py:82-83).
- sponsor_min_dcf (src/hearsay/metrics.py:78) uses the sponsor's own constants and polarity (higher score = bona fide); it exists to catch a polarity mismatch with the sponsor's script, not as the primary metric.
- score_files swallows all scorer exceptions (src/hearsay/submission.py:60); check ScoredFiles.flags / n_flagged after scoring.
- bench_latency times only the backbone forward pass; pooling and head are excluded (docstring line 5).
- splits/nsa_folds_plus_asv19.csv (27,648 rows, 15,128 bonafide / 12,520 spoof) also exists; do not conflate it with the 20,000-row base file.
- MLAAD is gated with a non-commercial notice (scripts/pull_mlaad.py:3).

## Documentation (docs/, CLAUDE.md, README.md)

| Path | Status | Purpose |
|---|---|---|
| README.md | living | public intro, setup, Docker target, layout; points to docs/STATUS.md and CLAUDE.md; to be rewritten before submission (README.md:17) |
| CLAUDE.md | living | project constitution: detector contract, model ladder, conventions, sources-of-truth precedence, frozen pre-event work log (100-111), AI-use disclosure kept current (113-155), skills table; team table (17-22) |
| docs/STATUS.md | living snapshot | what is done / in flight / planned, data locations, findings log, how to help; stamped Sat Sep 26 ~04:40 (STATUS.md:1) |
| docs/architecture.md | living | Mermaid diagrams, components tagged built / in-flight / planned; plan wins on scope, submissions/log.csv and meta.json win on numbers (architecture.md:1) |
| docs/plan.md | living | software-only working plan: scoring, deliverables, score direction, ladder gates, timeline; the scope source of truth |
| docs/nsa-challenge.md, .pdf | spec | sponsor brief text (pdftotext) and the original PDF; rules/scoring authority |
| docs/scoping.md, docs/master-doc.md | historical, frozen | pre-rescope memos, pinned by tests/test_docs_consistency.py |
| docs/handoffs/2026-09-26_feature-engineering-map.md | living for the lane | file:line "where do you plug in" map for hc_v4 families, ranked feature ideas, gate procedure |
| docs/handoffs/2026-09-26_handcrafted-v4-brief.md | brief | math, metric and gate criteria for handcrafted v4 |
| docs/handoffs/2026-09-25_first-two-hours-handoff.md, 2026-09-26_cpu-detectors-handoff.md, _m3-spectra-aasist-handoff.md, _m5-finetune-handoff.md | historical | lane spin-up runbooks |
| docs/reports/2026-09-26_cpu-detectors.md | report | D-track results: shortcuts found and fixed, container-is-the-label, 7.2 kHz band-wall finding |
| docs/reports/2026-09-26_handcrafted-v4.md | report, untracked | v4 per-family gate results; stops before training-variant results |
| docs/reports/2026-09-26_m1b-asv19-bonafide.md | report | M1b: better holdout, worse In-the-Wild false alarms |
| docs/reports/2026-09-26_m5-xlsr-finetune.md | live skeleton | most cells TBD, filled as M5 pilot runs land |
| docs/reports/2026-09-26_m3-spectra-aasist-plan.md, docs/specs/2026-09-26_m3-spectra-aasist.md | untracked | M3 reviewed plan and run spec |
| docs/reports/2026-09-26_m5-xlsr-finetune-plan.md, -plan-review.md, docs/specs/2026-09-26_m5-xlsr-finetune.md | frozen | M5 plan, Codex review, run spec (D1-D8) |
| docs/reports/2026-09-25_*.md | frozen | M1 public-probe shakedown, rescope plan and review, sponsor questions |
| docs/reports/cloud-expense-ledger.md | living ledger | one row per cloud rental / destroy / credit event, appended by scripts/cloud/*.sh |
| docs/consults/2026-09-26_*.md | consult records | fusion-strategy and M5 data-mix prompts and responses (summary and verbatim) |
| docs/specs/2026-09-25_software-only-rescope.md, -audit.md | frozen | rescope run spec and audit |

Where to add

- A new lane's "where do I plug in" map: docs/handoffs/ (pattern: docs/handoffs/2026-09-26_feature-engineering-map.md). Results: docs/reports/. Run specs: docs/specs/. External consults: docs/consults/.
- Any completed AI-assisted work: CLAUDE.md:113-155 AI-use disclosure, which the rules require to be kept current through the weekend.
- Team ownership: CLAUDE.md:17-22.
- Constraints: CLAUDE.md, README.md, docs/plan.md and every SKILL.md are scanned by tests/test_docs_consistency.py (no spelled-out comma-separated-values word, no conflicting prior, no dropped-scope vocabulary, exact 'C_FA = 4' and 'π_synth = 0.3' strings).

Invariants you must keep

- Sources of truth: docs/plan.md for scope, submissions/log.csv and models/*/meta.json for numbers (architecture.md:1), docs/nsa-challenge.md for rules.
- docs/scoping.md, docs/master-doc.md and docs/handoffs/2026-09-25_first-two-hours-handoff.md are frozen at git sha e1a1e88 (tests/test_docs_consistency.py:24, 156).

Gotchas

- docs/STATUS.md and docs/architecture.md are snapshots, not auto-updated; commits ea65f73, 982aa11 and 8445095 (04:25-04:34) land at or after their timestamp.
- docs/reports/2026-09-26_m5-xlsr-finetune.md is intentionally a skeleton.

## How-to recipes

### (a) Add a feature family to the handcrafted detector

1. Write `myfam(ctx) -> dict[str, float]` in src/hearsay/hc_v4.py reading x/Z/S/freqs/voiced/rms_db/f0 from ctx; prefix every key with the family name; obey the naming and content rules above.
2. Register it in FAMILIES (src/hearsay/hc_v4.py:300-303).
3. Add its key set to EXPECTED in tests/test_hc_v4.py:33-44 and run `uv run pytest tests/test_hc_v4.py -q` (key names, finiteness, offset invariance, disjointness from v3).
4. Add `'myfam': ('myfam_',)` to FAMILY_PREFIXES (scripts/train_handcrafted.py:51).
5. Add plain-English LABELS entries in src/hearsay/detectors/handcrafted.py (LABELS ends line 92); run `uv run pytest tests/test_handcrafted_detector.py -q`.
6. Extract: `uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/<m>.csv --name <name> --families lfcc,phase,cqcc,modulation,breath,jitter,myfam --workers 6 [--crop test --seed 0]` (writes outputs/handcrafted/<name>.csv + .meta.json).
7. Train and gate: `uv run python scripts/train_handcrafted.py --train <name> --folds splits/nsa_folds.csv --test <test_name> --out-name handcrafted_v4 [--drop-columns 'contrast0_mean,mfcc2_std*']`; selection by inner CV only, holdout read once.
8. Stress: `uv run python scripts/eval_bundle.py --bundle models/hc_lgbm_<stamp> --features outputs/handcrafted/<file>.csv --oof outputs/detector_scores/<out-name>.csv`.
9. Record the gate result in docs/reports/2026-09-26_handcrafted-v4.md.

### (b) Add a new contract detector and export it for fusion

1. Rule-based: copy src/hearsay/detectors/enf.py's shape (docstring with rationale and calibration, `applies()` True, `run()` memoizing `analyze()` via ctx.memo, named score constants, register guard at module bottom). Learned: subclass FeatureModelDetector as src/hearsay/detectors/compression.py does (feature_fn, LABELS, name/prefix/kind_label/memo_key), and train the bundle with scripts/train_handcrafted.py so it lands in models/<prefix>_<kind>_<stamp>/.
2. Import it in src/hearsay/detectors/engineered.py:16 and add it to LEARNED or RULE_BASED (lines 18-19).
3. Write a hermetic test in the pattern of tests/test_handcrafted_detector.py (toy bundles, ClipContext.from_array, contract B1-B9) and add it to tests/test_engineered_registry.py's expected set.
4. Run `uv run pytest tests/test_detector_contract.py tests/test_engineered_registry.py tests/<new test> -q`.
5. Export fold-validated scores: `uv run python scripts/score_detector.py --detector <name> --folds splits/nsa_folds.csv --test-manifest outputs/manifests/nsa_test.csv --workers 6` (inner_oof / holdout / test splits under outputs/detector_scores/).

### (c) Run the fold-validated gate for any detector

1. Fold file: splits/nsa_folds.csv (or splits/nsa_folds_plus_asv19.csv when the row set includes ASV19); never build fresh folds. Extend with training-only rows via `scripts/extend_folds.py --base splits/nsa_folds.csv --extra <rows> --out <new file>`.
2. Rule-based or any registered detector: `uv run python scripts/score_detector.py --detector <name> --folds splits/nsa_folds.csv --test-manifest outputs/manifests/nsa_test.csv --workers 6`.
3. Handcrafted / compression bundles: `scripts/train_handcrafted.py --train <name> --folds splits/nsa_folds.csv --test <test_name> --out-name <out>`; compression features first via `uv run python scripts/extract_compression.py --manifest outputs/manifests/nsa_train_sample.csv --name nsa_train_sample --crop test --seed 0 --launder-frac 0.5 --workers 6`.
4. M1 probe: `scripts/train_probe.py --folds splits/nsa_folds.csv --train <embedding sets> [--stress <itw set>]` (layer picked by argmin CV (min_dcf, eer), holdout scored once, `assert auc > 0.5`).
5. M5: scripts/m5_assemble.py checks 16142 / 3858 / 1671 row counts and compares against the newest NSA-only M1 bar.
6. Read minDCF at pi_synth = 0.3 first (hearsay.metrics.report), then EER; stress sets are read at the inner-OOF threshold (scripts/eval_bundle.py, scripts/train_probe.py --stress).

### (d) Produce a TSV and log it

1. Rollback baseline: `scripts/make_constant_csv.py --test-dir <dir> --manifest <m> --pi-synth 0.3 --out <path> --notes <text>`; it runs preflight, score_files with a constant, write_submission and append_log with validation_score 1.0 (scripts/make_constant_csv.py:33-62).
2. M1 probe: `scripts/make_probe_csv.py --probe models/m1_<...> --test-dir <dir> --manifest <m> --pi-synth 0.3 [--cost-shift] --val-mindcf <x> --val-eer <y> --notes <text>`; preflight (submission.py:148) scores silence and a chord twice and asserts reproducibility before any real file.
3. write_submission (src/hearsay/submission.py:73) refuses to overwrite; pick a new path per attempt. It validates equal lengths, unique ids, finite scores in [0, 1] and re-reads the file.
4. append_log (src/hearsay/submission.py:113) appends one row to submissions/log.csv with the fixed LOG_COLUMNS; csv_path is stored repo-relative.
5. Check ScoredFiles.flags / n_flagged and the sidecar's flag column before trusting the TSV.

## Tests: what pins what

| Test | Pins |
|---|---|
| tests/test_hc_v4.py | per-family expected key sets (EXPECTED, lines 33-44), finite floats, no collision with the 75 v3 names, offset invariance (rel_tol 0.25 / abs_tol 0.05), families=() gives len(V3) columns, unknown family raises KeyError, directional checks (phase coherence, lfcc1 tilt, modulation fractions sum to 1, jitter small on a steady tone) |
| tests/test_handcrafted_detector.py | HandcraftedDetector against contract B1-B9 with toy logreg/lgbm bundles: result shape, score in [0, 1], features = bundle features + hc_logit, direction follows the model, evidence has an SD figure and a LABELS phrase, deterministic and memoized, missing bundle -> error result, latest_model_dir picks newest stamp, REGISTRY registration, families recomputed at inference (line 141) |
| tests/test_detector_contract.py | DetectorResult validation and safe_run never-raises / skip / error conversion (base.py:163-192) |
| tests/test_trees.py | Trees matches LightGBM bit-for-bit (line 22), contrib sums to raw (line 32), categorical splits rejected (line 61), no lightgbm import anywhere in src/ or tests/ (line 69) |
| tests/test_enf_detector.py, test_splice_detector.py, test_container_detector.py, test_compression_detector.py | each detector's scoring/evidence on synthetic inputs via ClipContext.from_array; fixed score sets; tag rule matching; compression wiring to a saved bundle |
| tests/test_engineered_registry.py | LEARNED/RULE_BASED/NAMES partition; importing engineered registers exactly five detectors |
| tests/test_audio_and_submission.py | embed_clip == bulk path within 1e-3 (line 179, needs weights); prepare_segment never tiles, deterministic, 8 s cap (line 270); band_limit reproduces the 7.25 kHz wall (line 294); trim_silence level-independent and safe (line 256); normalize_windows removes level and offset (line 222); test_duration_sampler in [3.0, 14.0] s with median in (3.2, 3.7) (line 284, skipped without splits/nsa_test_durations.csv). No separate test_embed.py or test_probe.py exists. |
| tests/test_m5_model.py | M5Net forward on a fixed tiny model; save_m5/load_m5 hash round-trip |
| tests/test_m5_bundle.py | write_clip peak clamp, pcm_sha256 length, prepare_clip peak scaling |
| tests/test_m5_manifest.py, test_m5_crops.py, test_m5_codecs.py, test_m5_scores.py | manifest fold/exclusion rules; crop/collate/LengthMixer; codec round-trip; assemble report shape incl. score_duration_spearman_bonafide (test_m5_scores.py:106). Not read in the finder pass. |
| tests/test_docs_consistency.py | C_FA = 4, C_MISS = 1, PI_SYNTH = 0.3 in code and as literal strings in CLAUDE.md and docs/plan.md; no spelled-out comma-separated-values word, no conflicting prior, no dropped-scope words in living docs; historical docs frozen at e1a1e88 plus one banner |
| tests/test_spectra.py | does not exist; the M3 spec lists ~40 planned tests |
| whole suite | `uv run pytest -q` passes 301/301; with `-m "not needs_data and not needs_weights and not slow"` 294 pass, 7 deselected |

## Known gaps and stale docs

- docs/reports/2026-09-26_handcrafted-v4.md ends at line 34 with the placeholder "_(filled in below from the full 20,000-clip run; four variants, all selected by inner CV only, holdout read once each.)_" and nothing follows; the training-variant section is not written.
- No v4-trained bundle exists under models/: only hc_lgbm_20260926-0232 and three hc_logreg_* dirs, all with train='nsa_train_sample_v3'; outputs/detector_scores/ has handcrafted.csv, _v1, _v2, _v3 but no handcrafted_v4.csv. outputs/handcrafted/ does contain nsa_test_v4.csv, nsa_train_sample_v4.csv and the hc_gate_*.csv per-family gate files with sidecars.
- No speaker-embedding-drift detector exists anywhere in src/hearsay/detectors/: no stub, TODO or partial implementation.
- No Dockerfile exists yet for the M5/Docker deployment path; the inferred image file list is src/hearsay/m5_model.py, m5_data.py, audio.py, handcrafted.py, embed.py, metrics.py, scripts/m5_score.py, a saved model dir (backbone/, head.safetensors, m5_config.json, hashes.json) and pyproject.toml / uv.lock.
- The CPU-parity check in docs/specs/2026-09-26_m5-xlsr-finetune.md:106 (Spearman > 0.99, max |logit diff| < 0.05 on 50 clips) was not found implemented in scripts/m5_train.py, scripts/m5_score.py or any tests/test_m5*.py.
- scripts/m5_assemble.py describes a 3-condition gate (lines 212-214) but exposes holdout_passed, the pooled-OOF comparison and itw['passed'] as separate fields, not one combined boolean.
- scripts/score_spectra.py does not exist; src/hearsay/spectra.py lacks the M3 extensions (pad_mode, prepare_segment integration, direction check, crop plan, export helpers); hearsay.metrics.by_group is absent; the M3 spec's Results, Codex plan review and Audits sections are placeholders.
- docs/STATUS.md:44 claims "297 passing, 301 collected"; `uv run pytest -q` reports 301 passed, 301 collected.
- docs/STATUS.md:41 and docs/architecture.md (both stamped Sat Sep 26 ~04:40) describe M5 as in flight with no pilot; commits ea65f73, 982aa11 and 8445095 show the pipeline built, a pilot launched, an MLAAD probe A (cap 12%), an In-the-Wild P_FA gate in the assembler, and the gate bar changed to NSA-only M1 by content with M1b beside it.
- CLAUDE.md:117-126 AI-use disclosure lists work only through the D-track CPU chat; the M1 TSV submission and M1b addition (dba960c, 582ecee), the 7.2 kHz band-match fix (133e536) and consult findings (0ad7a30), the M3 plan/spec/handoff, the M5 pipeline and pilot (ea65f73, 982aa11, 8445095), the v4 scaffolding (4b644c9 through 096754d), the architecture diagrams (9288d73) and docs/consults/ have no entry.
- CLAUDE.md:17-22 team table rows for Teammate A/B/C are still "_fill in_".
- docs/handoffs/2026-09-26_feature-engineering-map.md:3 says a repo-wide code map is being generated at docs/code-map.md; this file fulfils that pointer.
- src/hearsay/metrics.py:13-15 flags an unresolved sponsor-formula ambiguity (P_FA weighting by P(attack)).
- m5_data.group_key (src/hearsay/m5_data.py:63) vs scripts/make_folds.py:group_key identity was not diffed.

## Reading order

Feature-engineering teammate

1. docs/STATUS.md
2. docs/handoffs/2026-09-26_feature-engineering-map.md, then docs/handoffs/2026-09-26_handcrafted-v4-brief.md (section 3 "Where to plug in" is the recipe)
3. src/hearsay/handcrafted.py (features:72, ctx:136, preprocessing 43-68)
4. src/hearsay/hc_v4.py (FAMILIES:300 and the six family functions)
5. tests/test_hc_v4.py (the contract every family must satisfy)
6. scripts/extract_handcrafted.py (--families -> .csv + .meta.json)
7. scripts/train_handcrafted.py (META_COLS/META_SUFFIXES, FAMILY_PREFIXES, --drop-columns, bundle contents)
8. src/hearsay/detectors/_learned.py (run:88, _evidence:102) and src/hearsay/detectors/handcrafted.py (LABELS:21)
9. tests/test_handcrafted_detector.py (hermetic contract-test pattern)
10. docs/reports/2026-09-26_cpu-detectors.md, then docs/reports/2026-09-26_handcrafted-v4.md
11. scripts/eval_bundle.py once a family is real

Docker teammate

1. README.md for the target CLI contract (`docker run --network none -v <test_dir>:/data:ro -v <out_dir>:/out hearsay`)
2. CLAUDE.md Conventions (offline-only, 16 kHz mono, no hosted API)
3. docs/plan.md Docker section; docs/STATUS.md:43 currently lists Docker as not started and unowned, verify before starting
4. docs/architecture.md for loader -> orchestrator -> detectors -> fusion -> TSV writer
5. src/hearsay/detectors/base.py (Detector:149, safe_run:163, Registry:195) and src/hearsay/submission.py (score_files:44, write_submission:73, append_log:113)
6. src/hearsay/audio.py (load_audio:22) and src/hearsay/m5_data.py (deploy_transform:258, DEPLOY_MAX_S:44), scripts/m5_score.py (score_paths:30) for the CPU inference path
7. src/hearsay/trees.py (why lightgbm is never imported at inference)
8. docs/STATUS.md Data table for what is mounted read-only, baked in (weights) or excluded (train-only data)
