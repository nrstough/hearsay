# HEARSAY architecture

Living document. Status as of **Sat Sep 26, 2026, ~04:00**. Where this file and `docs/plan.md` disagree on scope, the plan wins; on numbers, `submissions/log.csv` and the `models/*/meta.json` files win. Each component says whether it is **built**, **in flight** or **planned**, and which lane owns it (lanes are the parallel chats listed in `docs/handoffs/`).

## 1. What shapes the design

| Constraint | Source | Architectural consequence |
|---|---|---|
| 60% of the grade is normalized minDCF on 1,671 unlabeled test files; a false alarm costs 4× a miss; about 70% of files are real | NSA instructions | Every learned piece is selected on `9.33·P_FA + P_miss`; the real-speech side gets the most attention; ranking is all that matters, calibration is cosmetic |
| 20% for distinct techniques "actually leveraged", plus a bonus for choosing analyses per file | NSA brief rubric | One detector per rubric technique behind one contract; a rule-based orchestrator with a routing log |
| 20% for documentation in our own words, including what did not work | NSA brief | Every rung leaves a report; failed ideas are recorded, not deleted |
| Output is `filename<TAB>cm-score`, 1.0 = synthetic, one row per test file in template order | NSA instructions | A validating writer that never overwrites and always logs; a constant rollback TSV exists from hour one |
| Inference must run offline on CPU in a Docker image | NSA instructions | No hosted API or LLM on the scoring path; LightGBM is never imported at inference; one heavy model in memory at a time |
| The sponsor's shipped scoring code treats a higher score as bona fide, the opposite of the instructions | `data/nsa/HackGTMinDCF` | Both readings are logged for every model; direction is decided once (1.0 = synthetic, never flipped) and checked at the draft review |

## 2. Data flow

```
audio file (any container)
   │
   ├── hearsay.audio.probe_audio ──► container facts (format, codec, rate, tags)      [built]
   └── hearsay.audio.load_audio  ──► 16 kHz mono float32, one FFmpeg path, decoded once  [built]
                │
                ▼
        ClipContext (path, read-only audio, probe, per-clip memo cache)               [built]
                │
                ▼
        Orchestrator: rule-based routing + routing log                                 [planned, M4]
                │
   ┌────────────┼──────────────────────────────────────────────────┐
   ▼            ▼                                                  ▼
 deep detectors                          engineered detectors (CPU)
   M1  frozen XLS-R L7 + logistic probe   [built]      handcrafted  spectral+prosody → trees   [built]
   M3  Spectra-AASIST score stream        [in flight]  compression  codec traces → trees       [built]
   M5  fine-tuned XLS-R (12 layers)       [in flight]  container    header facts, rules        [built]
                                                       enf          mains hum, rules            [built]
                                                       splice       clicks / DC jumps, rules    [built]
                                                       speaker drift (ECAPA windows)            [planned]
   │            │                                                  │
   └────────────┴──────────── DetectorResult (score, evidence, features, status) ──┘   [built]
                │
                ▼
        Fusion: logistic stacker on logit(score) columns, fold-local                  [planned, M4]
                │
                ├──► teamName_predictions.tsv  via hearsay.submission.write_submission   [built]
                ├──► submissions/log.csv row (rung, holdout minDCF, EER, notes)          [built]
                └──► per-file explanation report (top detectors + evidence, routing log) [planned, M4]
```

Today's logged TSV bypasses the orchestrator and fusion: `scripts/make_probe_csv.py` runs M1 alone (load_audio → prepare_segment → embed_segment → probe LLR → posterior at π = 0.3). That is the rollback path if fusion is cut.

## 3. Components

| Component | Where | Status | Lane |
|---|---|---|---|
| Loader, probe, silence trim, windows | `src/hearsay/audio.py` | built | main |
| Detector contract, `safe_run`, registry | `src/hearsay/detectors/base.py` | built | main (changes need every detector owner) |
| Metric (minDCF, EER, both sponsor-code readings) | `src/hearsay/metrics.py` | built | main |
| Fold file | `splits/nsa_folds.csv` (`scripts/make_folds.py`); training-only extension `splits/nsa_folds_plus_asv19.csv` (`scripts/extend_folds.py`) | built, frozen | main |
| Segment preparation for deep detectors | `src/hearsay/embed.py` (`prepare_segment`, `normalize_windows`, `test_duration_sampler`) | built | main |
| M1 embeddings + probe + TSV | `scripts/extract_embeddings.py`, `train_probe.py`, `make_probe_csv.py`, `src/hearsay/probe.py` | built | main |
| M3 Spectra-AASIST stream | `src/hearsay/spectra.py`, `scripts/spectra_direction.py`; `scripts/score_spectra.py` to come | in flight | M3 |
| M5 fine-tune (bundle, trainer, cloud scripts) | `src/hearsay/m5_bundle.py`, `scripts/m5_*.py`, `scripts/cloud/` to come | in flight | M5 (vast.ai only) |
| Handcrafted features + trainer + export | `src/hearsay/handcrafted.py`, `scripts/extract_handcrafted.py`, `scripts/train_handcrafted.py` | built (v3); v4 brief in `docs/handoffs/` | CPU / teammate |
| Compression features + laundering | `src/hearsay/compression.py`, `scripts/extract_compression.py` | built | CPU |
| Rule-based detectors + exporter | `src/hearsay/detectors/{container,enf,splice}.py`, `scripts/score_detector.py` | built | CPU |
| Learned-detector wrapper, numpy trees | `src/hearsay/detectors/_learned.py`, `src/hearsay/trees.py` | built | CPU |
| Orchestrator, fusion, explanation report | not yet in `src/` | planned (M4, after detector scores land Sat 19:00; frozen Sat 22:00) | main |
| Docker image | no `Dockerfile` yet; design in `docs/plan.md` "Docker" | planned (first amd64 image Sat 14:00) | unassigned (plan says Teammate C) |
| Submission writer + log | `src/hearsay/submission.py` | built | main |

## 4. Invariants (the rules the code enforces)

1. **One audio path.** Everything decodes through `load_audio`: FFmpeg, first audio stream, downmix, 16 kHz, float32, non-finite samples zeroed. Nothing downstream resamples or denoises. This also erases the container shortcut (training real speech is FLAC/22 kHz WAV, training fakes include MP3, every test file is 16 kHz PCM WAV).
2. **Same preparation for train and test, per detector.** Deep detectors: `prepare_segment` = `band_limit` (same filter) → silence trim (35 dB relative) → random crop to a test-duration length for training rows only → 8 s cap → per-input zero-mean/unit-variance. Engineered detectors: `hearsay.handcrafted._crop` = `band_limit` (Kaiser low-pass matching the test set's 7.2 kHz roll-off) → `prepare_segment` → RMS-normalize. Test clips are never cropped or tiled. The deep path gained `band_limit` inside `prepare_segment` on Sat Sep 26, ~04:20; `handcrafted._crop` passes `band_match=False` to it because it filters first.
3. **Score direction.** Every detector's score increases with synthetic likelihood; `DetectorResult` rejects anything else; `spectra_direction.py` asserts it on labeled data for M3; `train_handcrafted.py` defines the label so the export cannot invert.
4. **Never lose a row.** `safe_run` turns a crashing detector into `status="error"`, score 0.5, which fusion treats as missing, never as evidence. `score_files` gives a decode failure the fallback score plus a flag. `write_submission` refuses duplicates, non-finite values, out-of-range scores and existing paths.
5. **Fold discipline.** Spoof grouped by generator, real by speaker (LJ by chapter). Outer holdout = `playht` + `wavegrad2` + 26 real groups (3,858 rows), never trained on, never used for selection, read once per model. Inner folds 0–4 (16,142 rows) give out-of-fold scores; every learned piece, including scalers, Platt maps and the stacker, is fit fold-locally. Extra training data (ASVspoof 2019 real speech for M1b, the M5 extra pool) enters inner folds only, so holdout numbers stay comparable across models.
6. **One export format.** `outputs/detector_scores/<name>.csv` with `path, fold, split ∈ {inner_oof, holdout, test}, score, logit`, 21,671 rows, unique paths. Fusion is trained on the `inner_oof` rows, read out on `holdout`, applied to `test`.
7. **No LightGBM at inference.** It and torch each bundle an OpenMP runtime and crash together on macOS. Bundles store the dumped trees; `hearsay.trees` evaluates them in numpy and gives per-feature contributions for the evidence sentence.
8. **Offline.** Weights are local (`weights/`), `HF_HUB_OFFLINE=1`; no LLM or hosted service is on the scoring path.
9. **Every TSV is logged** with its outer-holdout minDCF and EER; a submitted TSV is never overwritten.

## 5. The detector contract

`ClipContext` → `DetectorResult(name, score ∈ [0, 1], evidence: str, features: {str: float}, status ∈ {ok, skipped, error})`. Detectors load heavy models lazily inside `run`, share expensive intermediates through `ctx.memo`, and never read filenames or filesystem timestamps. Fusion input per detector: the exported `logit` column from `outputs/detector_scores/<name>.csv`, standardized on inner rows, when `ok`. That column holds the raw decision value (e.g. Spectra-AASIST's synth_logit, whose margins run ±10–17), so nothing saturates. For live `DetectorResult`s without an export it falls back to `logit(clip(score, 1e-4, 1 − 1e-4))`. When a detector isn't `ok`, the value is imputed with the fold-local train mean plus a `<name>.missing` indicator. The registry iterates in sorted name order so fusion columns are stable.

## 6. Deep detectors

**M1, frozen XLS-R 300M + logistic probe (built).** Segment-mode embeddings: per-layer time-mean of the whole prepared clip, all 25 hidden states kept (`(N, 25, 1024)` shards under `outputs/embeddings/`). Layer chosen by inner-fold CV (layer 7 wins; 6 and 8 close, 20+ poor). Standardized class-balanced logistic regression, Platt-mapped to a prior-neutral LLR on out-of-fold scores. Current: holdout minDCF 0.146, EER 2.5%. **M1b** adds 5,128 ASVspoof 2019 real clips (40 speakers) plus 2,520 spoof anchors to the inner folds: holdout 0.079, but In-the-Wild stress slightly worse (0.405 vs 0.374) and P_FA at the inner threshold up from 1.2% to 5.7%. Both sit at 0.37–0.40 on In-the-Wild against 0.08–0.15 on the NSA-domain holdout; that gap, not the fusion choice, is the headline risk.

**M3, Spectra-AASIST (in flight).** Off-the-shelf detector wrapping its own XLS-R; logits `(spoof, bonafide)`, `synth_logit = logit_spoof − logit_bonafide`. Scored on windows of 64,600 samples; the handoff recommends the trimmed `prepare_segment` clip with zero padding instead of the repeat-padding the checkpoint was trained with, so tiling cannot become a test-only seam. No training. Value: a second deep score stream trained on different data, for the stacker. Cost is comparable to M1 extraction, so it competes with the main chat for the MPS GPU.

**M5, fine-tuned XLS-R (in flight, vast.ai A100s).** Backbone truncated to 12 transformer layers, CNN encoder frozen, learned softmax-weighted layer sum → attentive statistics pooling → linear logit. Trained on the NSA sample plus a training-only extra pool (rest of DiffSSD capped per generator, ASVspoof 2019 real + a spoof channel anchor, gated MLAAD), with class-blind augmentation (noise, band-limits, reverb, codec round-trips, RawBoost convolutive, trim jitter). Protocol: pilot → NSA-only vs NSA+extra ablation on inner folds → 5 fold models (out-of-fold scores for stacking) + 1 full model (holdout and test). Gate: beat M1 on inner OOF and holdout by Sat 12:00 or M1 stays. Ships one checkpoint that must run on CPU with Spearman > 0.99 against the box's bf16 scores.

## 7. Engineered detectors

**Handcrafted spectral + prosody (built, v3).** 75 features on the band-matched, trimmed, test-length-cropped, RMS-normalized clip: high-band energy ratios, roll-offs, centroid, bandwidth, flatness, 6-band spectral contrast, flux, 20 MFCC means + stds, YIN pitch statistics, energy dynamics, pauses, zero-crossing rate. Logistic vs LightGBM chosen by inner OOF; LightGBM wins since band matching. Holdout minDCF 0.254, EER 4.3%; blind to `grad_tts`, `pro_diff` (1.00) and `elevenlabs` (0.80); weak on LibriSpeech real (0.77 inner). The v4 brief (`docs/handoffs/2026-09-26_handcrafted-v4-brief.md`) adds LFCC, phase/group-delay, CQCC, modulation, jitter/shimmer and breath families behind a 30-minute per-family gate.

**Compression forensics (built).** 19 features in 3–7 kHz: spectral holes (deep and local), hole flicker and runs, floor depth, effective bandwidth, high-band tilt. Trained with laundering equalization (a random half of both classes re-encoded through MP3/AAC) so "lossy = ElevenLabs/PlayHT" cannot be learned. It detects codec history well (AUC 0.90–0.94) and the class barely (holdout 0.90). The test set reads as laundered throughout, so this is routing and evidence, not a fusion column, unless leave-one-out says otherwise.

**Container, ENF, splice (built, rule-based).** Container returns header facts and a constant 0.5 on this test set by design (the container is the label on the training data; every test file carries the same FFmpeg tag). ENF tracks 50/60 Hz hum stability (present in 24 test files; in 36% of LJ clips, so a learned version would learn "hum = real"). Splice flags clicks and DC jumps (58 test files). All three carry evidence and routing facts for the rubric; none moves minDCF.

## 8. Validation and selection

- **Selection metric:** normalized minDCF at π_synth = 0.3, C_FA = 4, on inner out-of-fold scores. EER and the sponsor-code readings are reported alongside, never selected on.
- **Holdout:** read once per model; two generators and 26 real groups, so its noise is about ±0.07–0.10 and a gap under 0.15 is not evidence.
- **Stress sets (eval only, never trained on):** In-the-Wild (2,000 real from 54 speakers, 1,000 spoof) for false alarms on unfamiliar real speech; MLAAD (143 generators) available as a miss-rate readout for models that did not train on it.
- **Test-set distribution check:** share of test files above 0.5 against the stated ~30% synthetic. Handcrafted went from 90% to 42% with band matching; M1 sits at 40%.
- **Preflight before every TSV:** silence and a synthetic chord through the scorer; finite, reproducible, and the scores are logged (M1 currently scores both near 1.0, which a non-speech gate must address).
- **The one labeled readout on the real test set** is NSA's draft review on Saturday afternoon. It settles the score-direction question and compares at most two candidates.

## 9. Open architecture decisions

1. **Band matching in the deep path.** The engineered detectors apply `band_limit` to every clip; the deep path does not, and M5's plan has band-limiting only as a random augmentation. The fix is one call in `prepare_segment` (main chat owns it; M3 and M5 import it), then re-extraction. Cost: about one GPU hour for M1's four embedding sets; M5 inherits it in the bundle build if it lands first. Decide before M5 trains and before M3 scores.
2. **Fusion rule.** Inner OOF says M1 alone (0.250) beats the M1 + handcrafted mean (0.258); the holdout says the mean is far better (0.066 vs 0.146); the logistic stacker zeroes the handcrafted column. Both gaps are inside holdout noise. Candidates: constrained stacker (non-negative weights, shrink toward equal), rank averaging, nested CV of the fusion rule. LJ is one speaker on both sides of every fold, so the stacker should be fit on LibriSpeech and ASVspoof out-of-fold real rows, or treat LJ as weakly grouped. Awaiting the fusion consult's answer (`docs/consults/2026-09-26_fusion-strategy_CONSULTATION.md`).
3. **Which columns fuse.** M1 (or M5), handcrafted, M3 if it lands: yes. Compression: only if leave-one-out shows a gain. Container, ENF, splice: evidence and routing only. M5 exports `stackable:false` if any fold model is missing; fusion must handle a test/holdout-only column.
4. **Default answer for undetermined files.** Decode failures, non-speech (silence, music), detector disagreement: with a false alarm at 9.33× a miss, uncertain files belong at the real end of the ranking, deliberately tie-broken. A voiced-fraction gate is the first candidate (test median voiced fraction 0.8). Depends on the sponsor's scoring direction; the draft review is the tripwire.
5. **Orchestrator scope.** Routing facts already exist as detector features: container `lossy`/`is_pcm_wav`/`ffmpeg_written`, compression `bw_hz` (69 test files lack the 7.2 kHz wall; a few are telephony-band), ENF presence, splice seams, voiced fraction. v1 is static rules over these plus a per-file routing log in the explanation report. Escalation (run the expensive deep model only where cheap detectors disagree) is possible but unnecessary at 1,671 files; see 6.
6. **Test-time compute.** The test set is about 95 minutes of audio, so inference cost is not a constraint; validation is. Options ranked by expected value: average M5's five fold models with the full model at test time (free once trained, reduces variance, which is what drives false alarms); average M1 probes on layers 6–8 (embeddings exist; validate on inner folds); a second backbone (WavLM Large weights are downloaded, never run); multi-crop averaging only for the few test clips over 8 s (75% are under 3.8 s). Anything adopted must be validated on inner folds and must keep the test-set share above 0.5 near 30%.
7. **Docker ownership.** First amd64 image is due Sat 14:00 with the M1 payload; nobody has started. If M5 lands, the image needs its checkpoint from the HF Hub or a GitHub release plus the CPU-parity check; if speaker drift lands, it needs SpeechBrain and the ECAPA weights.

## 10. Shortcut ledger

Every train-vs-test difference found so far, and where it is closed. Any new detector or dataset is checked against this list.

| Shortcut | Evidence | Closed by |
|---|---|---|
| Peak level | test peaks near 1.0, training real ~0.5; AUC 0.66 alone | per-input z-normalization (deep), RMS normalization (engineered); level is never a feature |
| Leading silence | LibriSpeech 0.37 s vs test 0.06 s | `trim_silence` on train and test |
| Duration and tiling | training 5–9 s vs test ~3.4 s; repeat-padding put a seam only in test data | segment mode: no tiling, training crops drawn from the test-duration distribution |
| Container / codec | real = FLAC or 22 kHz WAV, fakes include MP3, test = 16 kHz PCM with one FFmpeg tag (also on PlayHT) | `load_audio` for everything; container detector never learned; laundering-equalized compression training |
| 7.2 kHz low-pass on the test set | 1,602 of 1,671 test files drop 44 dB between 6.5 and 7.5 kHz; no training corpus does | `band_limit` in both paths (`prepare_segment` since Sat ~04:20); v2 deep embeddings predate it and are being re-extracted |
| Mains hum | stable hum in 36% of LJ, 21% of LibriSpeech, <0.5% of nine generators | ENF never learned; low-frequency features checked LJ-vs-LibriSpeech |
| Crop edges | training crops start mid-word, test clips are whole utterances | no onset/offset features |
| LJ single speaker | one voice on both sides of every fold | folds group LJ by chapter; stacker to be fit on non-LJ real rows |
| Non-speech | silence and a chord score ~1.0 with M1 | gate pending (decision 4) |
