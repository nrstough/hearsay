# Plan: M5, XLS-R 300M fine-tune on vast.ai (Sat Sep 26, 2026, ~03:20; revised 03:25 after Codex + critique + the consult answer)

Run spec: `docs/specs/2026-09-26_m5-xlsr-finetune.md` (D1–D8, acceptance criteria, appendix with the P2 failure-mode tables). Deep mode: three exploration agents (architecture, file impact, risk), a Codex plan review (`…-plan-review.md`, 15 findings) and a Claude critique (17 findings) fed this revision. Section 12 maps every review finding to what changed; section 13 maps the consult answer (`docs/consults/…_RESPONSE.md`). Worktree `~/Projects/hearsay`, branch `main`, shared with two other chats: stage only the files in section 9.

**The bar:** M1 v2 (`models/m1_wav2vec2-xls-r-300m_L7_20260926-0302/meta.json`) outer-holdout **minDCF 0.1458** at π = 0.3 (EER 2.5%, AUC 0.997; wavegrad2 0.21, playht 0.03; LibriSpeech real 0.16, LJ real 0.09). M5 passes the gate if its whole-clip holdout minDCF is below 0.1458.

## 0. Findings that shaped the plan

| Finding | Effect |
|---|---|
| Suite is at **215 passed**, ruff clean. | Acceptance: 215 + new. |
| **A100 SXM4 at $0.67–0.80/h** (32 cores, CUDA ≥ 12.8) vs H100 $2.20 (2 offers; the cheap one has driver 555). | A100s, several at once. Filter `cuda_max_good >= 12.8`, `cpu_cores_effective >= 16`. |
| Home disk 14 GiB free (a 20 GB redundant `DiffSSD.tar` sits in Downloads); external drive 76 GiB free. | Bundle under `/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/m5_bundle/v1/`. |
| `weights/wav2vec2-xls-r-300m/` holds `pytorch_model.bin`, not safetensors. | The box pulls the HF repo at a pinned revision and verifies `config.json` sha against the bundled copy; fallback = upload a locally converted safetensors copy. |
| `test_duration_sampler` reads `REPO/splits/nsa_test_durations.csv`. | `code.tgz` ships `splits/nsa_test_durations.csv` and `splits/nsa_folds.csv`; a test unpacks the tarball into `tmp_path` and runs `--cpu-smoke` from there. |
| Every extra `nsa_train_full` row maps to an existing fold-file group; 8,000 extra playht/wavegrad2 and 1,340 holdout-group bona fide must be dropped. | Pure group → fold lookup, holdout-fold rows dropped. |
| ASV19 trims to bona fide median 1.9 s / spoof 2.1 s (p90 2.9 / 4.6). | ≥ 2.0 s cut, spoof pre-sampled 3:1 then histogram-matched to 1:1 per 0.25 s bin. |
| MLAAD unique = 6,180. | Dedupe; fix `CLAUDE.md:140`. |
| Mask verified (2.6e-6 masked vs 0.63 unmasked); layerdrop 0.1 + SpecAugment on by default. | Always mask; masked mean; `layerdrop=0`; SpecAugment on. |
| CPU inference 0.24 s per 3.4 s clip (Mac); 16 of 24 layers cuts 35–50%. | `keep_layers=16`; the shipped checkpoint is the full truncated model (~850 MB safetensors). |
| Image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` ships torch 2.8.0+cu128; the cu128 index has no 2.14.0. | Use the image's torch; install `transformers==5.17.0` (needs torch ≥ 2.5), `safetensors`, `soundfile`, `scipy`, `pandas`, `scikit-learn`, `tqdm`. Mac/Docker load the safetensors under torch 2.14.0; parity test covers the gap. |
| Codec round-trips through `load_audio`: MP3/Opus/μ-law 0 lead; AAC +128 tail; AMR-NB +320 tail, 159 lead. | Fine after re-trim. |
| Nathan is awake now (messaging at 03:50). | Ask for a **spend cap** now, so the pilot can launch the minute the bundle is up. |

## 1. Timeline and cutoffs (EDT)

| When | What | Cutoff rule |
|---|---|---|
| now–03:50 | §2 manifest builder + tests; start the bundle build (§3) in the background at ~03:50 | |
| 03:50–05:30 | §4 crops/augment/model, §6 trainer/scorer/assembler, §7 cloud scripts, all tests | |
| 05:30–06:00 | §5 upload (500 MB probe first), full suite, commit WIP | Bundle not in R2 by **06:45** → drop `extra/` from the upload (NSA-only arm) |
| 06:00–06:45 | Pilot box: setup, HF pull, codec pass (30% subset + holdout), 600-step pilot, one resume | Pilot not green by **07:45** → skip the ablation |
| 06:45–07:30 | Ablation, only if on time: `nsa` vs `nsa_extra` on folds 0 and 4 (2 boxes) | |
| 07:30–10:00 | Final 6 models on 3 boxes (2 jobs each, ≤ 45 min per model); full-model box scores holdout/test in fp32 | Any fold model unfinished by **11:00** → fallback contract (§8.5) |
| 10:00–11:40 | Pull, assemble, parity, `meta.json`, HF upload, report, commit, critique, Codex audit | |
| **11:45** | Absolute rental deadline: the reaper destroys every instance | |
| 12:00 | Gate | |

Spend: pilot 0.75 h + ablation 2 × 0.75 h + finals 3 × 2.5 h ≈ 9.75 A100-hours × ~$0.70 ≈ $7, plus storage/bandwidth ≈ $1. **Ask: approval for up to $12 on A100-class boxes, and a $10 top-up.**

## 2. Manifest builder: `src/hearsay/m5_data.py` (part 1) + `scripts/m5_build_manifest.py`

- `FOLDS_SHA256` constant; `check_folds_file(path)` raises on mismatch (A8).
- `group_key(row)`: copy of `scripts/make_folds.py:28-33`.
- `HOLDOUT_GENERATORS = {"playht", "wavegrad2"}`, `HOLDOUT_ALIASES = ("playht", "play.ht", "wavegrad")`, `FAMILY_EXCLUSIONS = {"1": ("openvoice",), "2": ("xtts", "vixtts"), "4": ("elevenlabs",)}` (substring, lower-cased `generator`/`model_name`).
- `build_manifest(...) -> DataFrame[id, path, label, generator, speaker, source, group, fold, train_scope, model_name]`:
  - core = fold file, `train_scope="core"`.
  - extra NSA = full − sample paths; group via `group_key`; fold from the fold file's group → fold map (assert all known); **drop fold == holdout**; cap extra spoof at **1,000 per generator** (seeded; ~2k total per inner generator, consult §3), all extra bona fide kept; `train_scope="extra_nsa"`.
  - ASV19 (train + dev, 40 bona fide speakers): `fold="extra"`, `train_scope="extra_asv19"`, **all bona fide; spoof pre-sampled to 7,500 stratified over A01–A06** (seeded), which the bundle's ≥ 2.0 s cut and duration matching reduce to a ~2.5k channel anchor (consult §2); `group="asv:<speaker>"` / `"gen:asv19_<attack>"`.
  - MLAAD: union, dedupe by path, `fold="extra"`, `train_scope="extra_mlaad"`, `generator=model_name`.
  - refuse on holdout alias in any extra name, or `in_the_wild` in any path.
- `training_rows(m, fold)`: fold model k → core rows with fold ∉ {k, holdout} + extra_nsa with fold ≠ k + extra_asv19 + extra_mlaad − family exclusions for k; `fold=None` (full) → everything but holdout, no exclusions. Asserts both labels and no holdout row. `validation_rows(m, k)` → core fold k.
- `tests/test_m5_manifest.py`: A1 (incl. holdout-group bona fide from `full`), A2, A3 + counterfactual, A4, A5, A6, A7, A8, A9, A10.

## 3. Bundle writer: `scripts/m5_build_bundle.py` + `m5_data.py` (part 2)

- Output `m5_bundle/v1/{core,extra,test}/<id>.flac`, `manifest.csv`, `test/manifest.csv`, `errors.csv`, `diagnostics.json`, `code.tgz`, `config_sha.txt`, `TREE_SHA`, `bundle_meta.json`.
- `write_clip(path, out, max_s=15, min_s=None)`: `load_audio` → `trim_silence` → cap → peak-scale to 0.999 if above → PCM_16 FLAC. Returns `duration, peak, lead_s, pcm_sha256` (sha of the trimmed int16 samples, mirroring `scripts/hash_audio.py`).
  - **Core and test rows are never dropped**: no `min_s`, and a `DecodeError` on a core/test row aborts the build (B4/B6 revised; Codex 5). Extras: `min_s=0.5` (ASV19: 2.0), decode failures listed, abort if > 0.5% of extras.
- 6 workers, `nice`, resumable: an existing file is reused only if `soundfile.info` opens it and its frame count matches the manifest (Codex 9).
- After durations: ASV19 ≥ 2.0 s cut on both classes, then spoof subsampled per 0.25 s duration bin to at most **0.5×** the bona fide count in that bin (target ≈ 2.5k spoof, consult §2).
- **Cross-scope dedup** (Codex 2): refuse if any `pcm_sha256` appears in more than one of {core/test, extra_*}. Speaker overlap documented in the report (MLAAD = M-AILABS speakers, ASV19 = VCTK, LJ, LibriSpeech: disjoint by construction).
- `diagnostics.json`: per `train_scope`, for each of duration/peak/lead_s, `max(AUC, 1−AUC)` of that scope's rows against the **opposite class of the whole training pool** (so MLAAD, spoof-only, is measured against all bona fide). Informational here; the gate is per model in §6.
- `code.tgz` = `src/hearsay`, `scripts/m5_*.py`, `scripts/cloud/box_*`, `splits/nsa_test_durations.csv`, `splits/nsa_folds.csv`, `pyproject.toml`, `uv.lock`, `weights/wav2vec2-xls-r-300m/{config.json,preprocessor_config.json}`.
- `TREE_SHA` = sha256 over sorted `(relpath, sha256)` of **every** file in the version dir (FLACs, manifests, code.tgz); `bundle_meta.json` carries it plus the manifest row counts and the HF revision to pull.
- `tests/test_m5_bundle.py`: B1 (formats via `ffmpeg_sine`, incl. MP4/AAC), B2, B3, B4 (17 s → 15 s; a 0.3 s **extra** dropped and counted; a 0.3 s **core** kept), B6 (corrupt extra → errors; corrupt core → abort), B7 (tree-sha changes on any file change; manifest change included), B8, dedup refusal, smoke-from-tarball (`code.tgz` unpacked into `tmp_path`, `m5_train.py --cpu-smoke` runs from it).

## 4. Crops, augmentation, model

**Crops (`m5_data.py` part 3)**: `LengthMixer(seed, p_test=0.7, lo=3.0, hi=14.0)` (no data-derived cap: the pooled gate in §6 replaces it); `crop(x, crop_s, rng)` never pads or tiles; `BatchLengthSampler` yields one length per batch; `collate` normalizes over valid samples only (`normalize_windows(x[None])[0]`), zero-pads, masks; `M5Dataset` re-seeds per epoch; `BalancedSourceSampler(batch, real_mix={lj:0.25, libri:0.35, asv19:0.40}, spoof_mix={diffssd:0.62, mlaad:0.28, asv19:0.10}, mlaad_cap_per_model=120)` (constant mix, no curriculum; consult §3); `test_transform(x) = normalize_windows(prepare_segment(x, max_s=14.0)[None])[0]`.
`tests/test_m5_crops.py`: C1, **C2** (2 s clip under a 5 s crop → 2 s of audio then zeros, mask 0, autocorrelation shows no repeat), **C3** (10k draws: 70% ± 3% from test durations, all in [3, 14]), C5, C6 (bit-exact vs the training transform with no crop/augment), C7.

**Augmentation (`augment.py`)**: `add_noise` (white + pink, exact SNR), `band_limit` (order-8 Butterworth, `sosfiltfilt`), `reverb` (exponential-decay RIR, `h[0]=1`), `gain_clip`; codec op = swap in the clip's variant file if it has one. plus `rawboost_conv` (RawBoost's linear + nonlinear convolutive algorithm only, numpy) and `trim_jitter` (training-only: re-trim at a random 30–40 dB threshold, keep 0–200 ms of edge at random; applied before cropping). `Augmenter(p_aug=0.65, seed)` picks 1–2 ops class-blind with band-limit upweighted (×2) and `rawboost_conv` at p = 0.25; realized rates are logged per source × class; `.clean()` identity; `holdout_slice(rows, seed)` fixed. `tests/test_augment.py`: D1–D9 (D6 with the counterfactual).

**Model (`m5_model.py`)**: `M5Config` (`keep_layers=12, train_top=12, layer_weighted_sum=True, pooling="attn_stats", lr_backbone=1e-5, llrd=0.85, lr_head=5e-4, weight_decay=0.01, warmup=0.08, schedule="cosine", label_smoothing=0.05, batch_size=32, grad_clip=1.0, layerdrop=0.0, spec_augment=True, p_aug=0.65, p_test_len=0.7, seed=0, mix`; fallback `keep_layers=16` if the learned layer weights concentrate on the top layer in the pilot), `.hash()`. `build_model(cfg, weights_dir, tiny=False)`: load, truncate `encoder.layers[:keep_layers]`, `freeze_feature_encoder()`, freeze lower layers if `train_top < keep_layers`; head = softmax-weighted sum over the retained layers' hidden states (init uniform) → masked attentive statistics pooling (mean + std, 128-d bottleneck) → `nn.Linear(2·hidden, 1)`. Forward uses `output_hidden_states=True` and the frame mask from `_get_feature_vector_attention_mask`; the learned layer weights are logged every eval; **high logit = synthetic**. Save = `backbone.save_pretrained(dir, safe_serialization=True)` (truncated config) + `head.safetensors` + `m5_config.json` + `hashes.json`. Masked-mean pooling stays as the `pooling="mean"` fallback. `tests/test_m5_model.py`: E1 (+ counterfactual `train_top=0`), E2, E3, E5, E8, E6 (loader refuses on hash mismatch).

## 5. Upload

- `rclone copy --files-from <list> "<bundle>/v1" r2:pa-source/hearsay/bundle/v1/` in the order `code.tgz, bundle_meta.json, manifest.csv, test/, core/, extra/`; a 500 MB probe first. Uplink too slow for `extra/` by 06:45 → the NSA-only arm runs first and `extra/` follows for the finals if it lands.
- `scripts/cloud/r2_guard.sh`: `r2_path()` refuses anything outside `r2:pa-source/hearsay/`; the same prefix check lives in `m5_train.py` (`assert prefix.startswith(...)`). Static test G1 covers `scripts/cloud/*.sh` **and** `scripts/m5_*.py`.
- HF revision: `huggingface_hub.model_info("facebook/wav2vec2-xls-r-300m").sha` recorded in `bundle_meta.json`; the box downloads that revision and asserts `sha256(config.json)` equals the bundled `config_sha.txt`.

## 6. Trainer, scorer, assembler

**`scripts/m5_train.py`** (box; `--cpu-smoke` uses `tiny=True` and a 12-clip synthetic bundle)
- Args: `--bundle --fold {0..4,full} --arm {nsa,nsa_extra} --steps --steps-per-epoch 1000 --eval-every 250 --out --r2-prefix --resume --deadline <unix ts>`.
- Refuses: no CUDA (unless smoke); `TREE_SHA` mismatch (full recompute at setup, once per box, ~1 min); weights config sha mismatch; **shortcut gate**: on this model's `training_rows` only, `max(AUC,1−AUC)` per scope (vs the opposite class of that training set) and pooled AUC of `min(duration, 14)` under the sampler's expected weights, each must be ≤ 0.85; logged to `run_meta.json`. Never reads holdout durations or labels (Codex 1; tested: perturbing holdout rows leaves the gate values unchanged).
- Loop: AdamW with layer-wise LR decay (γ = 0.85 from the top retained layer; head 5e-4 without weight decay; weight decay 0.01 off biases/LayerNorm), 8% warmup + cosine over `--steps`, bf16 autocast, grad clip 1.0, BCEWithLogits with label smoothing 0.05. Every `eval_every` steps: validation on fold k's fixed test-length-crop slice → `report()`, **abort if AUC ≤ 0.5 after 500 steps**; abort on non-finite loss; checkpoint (full trainable state + rng) pushed to `r2:…/runs/<job>/ckpt/` every eval and at exit; `--resume` pulls the latest. `--deadline` stops training and proceeds to export.
- Export (all fold models and the full model) uses the **deployment transform in fp32** (whole clip ≤ 14 s, no augmentation, `torch.autocast` off): fold models write `scores_<k>.csv` for `validation_rows` (`split=inner_oof`); the full model writes `scores_holdout.csv`, `scores_test.csv`. Diagnostics (not exported to fusion): `diag_holdout_testlen.csv` (test-length crops), `diag_holdout_aug.csv` (fixed augmented slice, per-op tag; codec op uses the holdout variants made in §7), `train_log.jsonl` (incl. realized augmentation rates per source × class and the layer weights), `run_meta.json` (config hash, tree-sha, weights sha, git sha, clips/s, GPU, seconds, gate values). Prints `DONE`.
- **Selection is wall-clock fixed**: `--steps` comes from the pilot's clips/s so a model takes ≤ 45 min; no checkpoint picking on validation folds (folds 3 and 4 hold one generator each). Ablation arm choice uses the end-of-run fold 0/4 minDCF (reported at π = 0.3, π = 0.5, both sponsor directions; a flip is flagged).

**`scripts/m5_score.py`** (CPU, the Docker path): loads the truncated safetensors model + head, refuses on hash mismatch, batches by length with masks, `test_transform`, writes `path, score, logit`. Parity (F6, revised): 50 test clips, box fp32 vs Mac CPU fp32, max |Δlogit| < 1e-2 and Spearman > 0.999; and (Codex 14) 20 original test WAVs through `load_audio` vs their bundled FLACs, max |Δlogit| < 0.05.

**`scripts/m5_assemble.py`** (Mac): pulls runs, builds `outputs/detector_scores/m5_xlsr_ft.csv` (F1–F4 asserted: 16,142 `inner_oof` + 3,858 `holdout` + 1,671 `test`, unique paths, joined against `nsa_folds.csv`/`nsa_test.csv`, finite [0,1]); writes `models/m5_xlsr_ft_<stamp>/meta.json` (keys per spec D7, `by_group` copied from `train_handcrafted.py:75-88`, `m1_bar={val_min_dcf: 0.1458, model_dir}`, `oof_pooled{report}` (5-fold OOF minDCF, the larger-sample reading), `gate={holdout_passed, oof_passed, margin, rule: "both"}`, `score_duration_spearman` (OOF bona fide, consult failure mode 1), `itw_bonafide_pfa` (In-the-Wild bona fide read-out at the inner-fold threshold, per arm; Probe C), `stackable: true|false`, spend); copies the checkpoint dir. F5, F7, F8 tests in `tests/test_m5_scores.py`.

## 7. Cloud: `scripts/cloud/`

- **`launch.sh <job> <offer-id...>`**: budget guard = credit − Σ(dph × remaining est. hours of running instances) ≥ est_hours × dph + 1.50 (critique 7); re-checks the offer's `cuda_max_good`/cores; `create instance --image pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime --disk 60 --ssh`; **launcher-side trap destroys the CID on any failure before the chain starts** (Codex 10); NET-OK check; R2 creds piped over ssh **stdin** into `rclone config create` (never argv); starts `box_chain.sh` with `JOB`, `JOBS`, `DEADLINE`; appends a ledger row. The vast key stays on the Mac (read into a variable from `.env` `VAST_API_KEY` if set, else `~/.config/vastai/vast_api_key`; never echoed; `set -x` forbidden).
- **No API key on the box** (critique 6): the box only writes `STATUS` (`SETUP`, `RUNNING <job>`, `DONE`, `FAIL <reason>`) to `r2:…/runs/<job>/`. **`reaper.sh`** on the Mac (under `caffeinate`) polls STATUS every 60 s and destroys an instance on `DONE`/`FAIL`, on no STATUS change for 40 min, or at 11:45 EDT; it appends the invoice total to the ledger and ends with `vastai show instances` = 0. `teardown_check.sh` is the manual backstop.
- **`box_setup.sh`**: apt `ffmpeg`; pip `transformers==5.17.0 safetensors soundfile scipy pandas scikit-learn tqdm huggingface_hub rclone`-less (rclone via .deb); `PYTHONPATH=/root/m5/src`; pull `bundle/v1` with `--files-from` per arm; recompute `TREE_SHA`; HF download at the pinned revision + config sha check; `python -c "import torch; assert torch.cuda.is_available()"`; `SETUP-DONE`.
- **`box_codecs.py`** (pilot box only): one random codec (mp3/opus/aac/amrnb/mulaw) for a seeded 30% of training clips and for **every holdout clip**, decoded back through `load_audio` + re-trim, 30 workers; writes `codec_manifest.csv`; pushed to `r2:…/bundle/v1/codecs/`; later boxes pull it. `tests/test_m5_codecs.py` runs it on 3 fixtures with the local ffmpeg (B5 tolerances; missing encoder → recorded, skipped).
- **`box_chain.sh`**: `trap 'status FAIL' ERR EXIT` (status only; the reaper destroys), runs setup then each job, pushes logs; `DONE`.
- `tests/test_cloud_scripts.py`: G1 (both dirs), G2 (reaper has the deadline and the DONE/FAIL branches; chain has the trap), G3 (guard arithmetic via a mocked `vastai` stub), G5, `bash -n` on all scripts, and a mocked-failure test for the launcher trap.

## 8. Runs (after Nathan's spend approval)

1. **Pilot** (1 box, fold 4, `nsa_extra`, 600 steps, eval every 250): clips/s, loss falling, minDCF finite, AUC > 0.5; kill after the step-250 checkpoint and `--resume`. Fix `--steps` for ≤ 45 min per model.
2. **Ablation** (only if the pilot is green by 07:45): `nsa` vs `nsa_extra` (the consult mix), folds 0 and 4, 2 boxes; pre-declared criterion: mean fold minDCF at π = 0.3. Else `nsa_extra` is the arm and NSA-only is reported as "not run".
   **MLAAD Probe A** (Mac CPU, in the background during the build, non-GPU): LightGBM on handcrafted features (`scripts/extract_handcrafted.py` outputs) separating MLAAD spoof from DiffSSD spoof on augmented copies; project LJ/LibriSpeech/ASV19 bona fide onto that axis. AUC > 0.95 with LibriSpeech on the MLAAD side → MLAAD share 12%; else 28%. Result recorded in `meta.json`. **Probe C** at assembly: In-the-Wild bona fide P_FA per arm at the inner-fold threshold; ΔP_FA > +1 point → MLAAD dropped from the finals if they haven't started, else reported.
3. **Finals** (3 boxes × 2 jobs): folds 0–4 + full, chosen arm, same `--steps`; the full-model box scores holdout and test in fp32.
4. `pull_runs.sh` → `m5_assemble.py` (writes the gate as **both** holdout and pooled 5-fold OOF vs M1; a holdout gap under 0.15 is reported as within noise, per the consult) → parity → `hf upload nrs124554433/hearsay-m5-xlsr --private` (fallback `gh release`), re-download + sha check → reaper confirms 0 instances.
5. **Fallback contract** (Codex 11): if any fold model is missing at 11:00, the export contains **no `inner_oof` rows at all** (never a partial set); `meta.json` says `stackable: false`; the main chat can use M5 standalone or in a mean fusion. The report names it as a blocker.
6. **Handoff to the main chat** (Codex 12): message naming the score file, `meta.json`, the checkpoint repo and `m5_score.py` as the Docker entry point. TSVs stay theirs.

## 9. Files

**Create:** `src/hearsay/{m5_data,augment,m5_model}.py`; `scripts/{m5_build_manifest,m5_build_bundle,m5_train,m5_score,m5_assemble}.py`; `scripts/cloud/{launch.sh,reaper.sh,box_setup.sh,box_chain.sh,box_codecs.py,r2_guard.sh,pull_runs.sh,teardown_check.sh}`; `tests/{test_m5_manifest,test_m5_bundle,test_m5_codecs,test_m5_crops,test_augment,test_m5_model,test_m5_scores,test_cloud_scripts}.py`; `docs/reports/cloud-expense-ledger.md`; `docs/reports/2026-09-26_m5-xlsr-finetune.md`; plus spec/plan/consult files.
**Modify:** `CLAUDE.md` disclosure (lines 131, 138, 140 → 6,180, 142, 148 → full LJ moved up; vast.ai under infrastructure) as a minimal diff. Conditional: `docs/STATUS.md:40`, `docs/plan.md:88/132/187`.
**Never touch:** `src/hearsay/embed.py`, the M1 scripts, `splits/`, `submissions/`, other chats' files, the local GPU. Secrets: the vast key never leaves the Mac; R2 creds reach the box over ssh stdin only (documented exception to `.env`-only, in the spec).

## 10. Verification

```bash
cd ~/Projects/hearsay
uv run pytest -q tests/test_m5_manifest.py tests/test_m5_bundle.py tests/test_m5_codecs.py tests/test_m5_crops.py tests/test_augment.py tests/test_m5_model.py tests/test_m5_scores.py tests/test_cloud_scripts.py
uv run pytest -q            # 215 + new
uv run ruff check .
git diff --stat -- src/hearsay/embed.py scripts/extract_embeddings.py scripts/train_probe.py scripts/make_probe_csv.py splits/ submissions/   # empty
uvx vastai show instances   # "No instances found." at the end
```

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A100 offers churn | `launch.sh` takes several ids; relaxed filter fallback; H100 NVL last. |
| Upload slow | probe; `extra/` optional; NSA-only arm first. |
| GPU starved by augmentation | 32-core boxes, 12 persistent workers, numpy ops ≤ 20 ms/crop, codec swap is a file read. |
| Instability / collapse | bf16 train, fp32 export, LR 1e-5 + warmup, clip 1.0, layerdrop 0, AUC tripwire, R2 checkpoints. |
| Length or corpus shortcut | ASV19 matching, per-model gate ≤ 0.85 on training rows only, pooled `min(dur,14)` gate, per-length-bucket readouts, cross-scope pcm dedup. |
| Wall clock | 3 boxes; ≤ 45 min per model; phase cutoffs in §1; deadline in the trainer; reaper at 11:45. |
| Boxes outliving a crash or an asleep Mac | launcher trap, reaper, `caffeinate`, `teardown_check.sh`. |
| Consult arrives late | defaults; config only. |

## 12. Review findings → changes

| # | Source | Change |
|---|---|---|
| Codex 1, critique 10 | Gates from training rows only; pooled `min(dur,14)` gate; no data-derived cap | §6, §4 |
| Codex 2 | `pcm_sha256` cross-scope dedup; speaker overlap documented | §3 |
| Codex 3, critique 2 | `splits/*.csv` in `code.tgz`; smoke-from-tarball test; `PYTHONPATH` | §3, §7 |
| Codex 4 | `max(AUC,1−AUC)`, opposite-class measurement handles MLAAD; MLAAD-vs-LibriSpeech handcrafted probe runs on the Mac in the background and is reported (non-blocking) | §3, §6 |
| Codex 5 | Core/test rows never dropped; decode error on core aborts | §3 |
| Codex 6 | All fusion exports use the deployment transform; crops/augmented are diagnostics | §6 |
| Codex 7, critique 12 | Steps-per-epoch defined; eval + checkpoint every 250 and at exit; pilot 600 steps; selection is wall-clock fixed | §6, §8 |
| Codex 8, critique 3 | Codec pass = 30% subset + holdout, once, on the pilot box; `--disk 60` | §7 |
| Codex 9 | Versioned `v1/`, tree-sha over every file, `--files-from`, resume verifies frames | §3, §5 |
| Codex 10, critique 6, 7 | Launcher trap; Mac-side reaper with deadline; no key on the box; aggregate budget guard | §7 |
| Codex 11 | Fallback = no `inner_oof` rows, `stackable:false` | §8.5 |
| Codex 12 | Named handoff to the main chat; TSVs stay theirs (handoff rule) | §8.6 |
| Codex 13 | Secrets exception documented; creds over stdin | §7, §9 |
| Codex 14, critique 5 | fp32 export; WAV-vs-FLAC parity | §6 |
| Codex 15, critique 8 | Cutoffs table; deadline flag; spend approval asked now | §1 |
| critique 1 | HF pull at pinned revision + config sha | §0, §5 |
| critique 4 | Image torch 2.8.0; no reinstall | §7 |
| critique 9 | ASV19 spoof 3:1 pre-sample | §2 |
| critique 11 | Holdout codec variants | §7 |
| critique 13 | Full truncated model saved | §4 |
| critique 14 | G1 covers `scripts/m5_*.py` | §5 |
| critique 15 | P2 tables appended to the spec; C2/C3 mapped | spec appendix, §4 |
| critique 16 | Ablation conditional; attn pooling cut; resume exercised once | §4, §8 |

## 13. Consult answer → changes (`docs/consults/2026-09-26_m5-extra-data-finetune_RESPONSE.md`)

| Item | Change |
|---|---|
| Kill the +50k DiffSSD spoof; ~2k per inner generator | extra spoof cap 1,000/generator (§2) |
| All ASV19 bona fide (40 speakers); spoof ~2.5k stratified A01–A06 | §2, §3 |
| MLAAD 28% of spoof side, ≤ 120/model, gated by Probe A; Probe C on In-the-Wild bona fide | sampler (§4), §8 |
| Real 25/35/40, spoof 62/28/10, constant mix | sampler (§4) |
| L = 12, all retained layers trainable, layer-weighted sum, attentive statistics pooling | `M5Config`, `build_model` (§4) |
| LLRD 0.85, head 5e-4, wd 0.01, 8% warmup + cosine, label smoothing 0.05, no OC-softmax | §6 |
| p_aug 0.65, convolutive RawBoost 0.25, band-limit upweighted, trim jitter, per-cell rate logging | §4 |
| Score–duration coupling, OOF-pooled gate, "gap < 0.15 is noise" | assembler (§6), §8.4 |
| Not adopted: half-epoch checkpoint selection (single-generator folds), frozen-backbone control (M1 is it), peak-norm before ZMUV (no-op) | recorded in the response synthesis |
| Passed to the main chat: LJ single-speaker OOF bias in the stacker; decide M5-vs-M1 on OOF + holdout | §8.6 |
