# Handoff: M5 XLS-R fine-tune on vast.ai (Sat Sep 26, 2026, ~01:30)

**Purpose of this chat:** plan (through `/plan-review`), build and run **M5**: fine-tune XLS-R 300M on a rented vast.ai H100, trained on multiple clip lengths with laundering augmentation. Deliver a checkpoint plus holdout and test scores in the fusion format.

**Gate:** M5 must **beat M1 on the outer holdout**, measured by normalized minDCF at π = 0.3. If it doesn't by **Sat midday**, freeze M1 (the `docs/plan.md` hard gate). **Ask Nathan before renting anything.**

## Context

HEARSAY is the HackGT 13 NSA HEARSAY entry, software only. Read `CLAUDE.md`, `docs/plan.md` and `docs/STATUS.md` first.

**Scoring**
- 60% MinDCF: false alarm ×4, about 70% of test files real. `hearsay.metrics` uses `C_FA = 4`, `π_synth = 0.3`, i.e. 9.33·P_FA + P_miss. Only ranking matters.
- Output is 1.0 = synthetic. Never flip.
- The final TSV is due before 8 AM Sunday. The Docker image must run M5 on CPU.

**Three chats run in parallel.** Stay in your lane.

| Chat | Owns |
|---|---|
| **Main** | Local GPU (MPS), M1, Spectra-AASIST, fusion, orchestrator, **all TSVs and `submissions/`** |
| **CPU detectors** (`docs/handoffs/2026-09-26_cpu-detectors-handoff.md`) | Handcrafted and engineered detectors, CPU only |
| **This one** | M5 on vast.ai. **Never run training on the local MPS GPU.** It's busy. |

**M1 status at handoff.** The v2 run is in flight: segment mode, meaning silence-trimmed, no tiling, 8 s cap, with training crops drawn from the NSA test-duration distribution.
- Embeddings: `outputs/embeddings/wav2vec2-xls-r-300m/nsa_{test,train_sample}_v2`
- Probe: `models/m1_wav2vec2-xls-r-300m_L*_*/meta.json`
- Its holdout minDCF is the **number M5 must beat**. Read it from that `meta.json` once it exists (`val_min_dcf`, plus `cv_by_layer` showing which frozen layer works best).

## Data

**Training**
- Sample: `outputs/manifests/nsa_train_sample.csv`, 20,000 rows: 1,000 per DiffSSD generator × 10, plus 6,000 LJ Speech real and 4,000 LibriSpeech real.
- Full manifest: `outputs/manifests/nsa_train_full.csv`, about 70k fakes plus 18.4k real. More is available if M5 needs it.
- Raw audio lives on the external drive through the symlinks `data/nsa/DiffSSD`, `data/ljspeech` and `data/librispeech`. It's 20 GB+, so **don't upload it raw.**

**Folds:** `splits/nsa_folds.csv`. Never regenerate it; it's shared.
- Spoof is grouped by generator; bona fide by speaker, or by chapter for LJ.
- **Outer holdout** = generators `playht` and `wavegrad2`, plus 26 bona fide groups.
- Inner folds are `0`–`4`.

**Test:** `data/nsa/HackGTHearsayTesting/`, 1,671 WAVs, 16 kHz PCM, 3.0–13.6 s (median 3.4 s). Row order comes from `data/nsa/HearsayScoreKey4TeamX.tsv`; the manifest is `outputs/manifests/nsa_test.csv`.

**Shortcuts already neutralized, and M5 must keep them neutralized:**
- **Level:** per-input zero-mean, unit-variance normalization (`hearsay.embed.normalize_windows`). XLS-R expects this anyway (`do_normalize: true`).
- **Leading silence:** LibriSpeech ~0.37 s vs test ~0.06 s. Handled by `hearsay.audio.trim_silence`, applied identically to train and test.
- **Length and tiling:** handled by `hearsay.embed.prepare_segment`.
- **Container:** DiffSSD has MP3 and 22 kHz files; the test set is all 16 kHz WAV. Decoding everything through `hearsay.audio.load_audio` into 16 kHz before training removes this.

## Reuse the Powerlifting-Analyzer vast tooling (already set up on this Mac)

- **Scripts to adapt:** `~/Powerlifting-Analyzer/scripts/cloud/`
  - `launch_robust.sh`: tries offers until one boots with working internet, auto-destroys bad hosts, then provisions.
  - `train_box_setup.sh`: venv plus torch cu121, verified imports.
  - `box_chain.sh`, `pull_and_analyze.sh`.
- **Skill:** `fleet-campaign` covers the budget guard, pilot before fan-out, and mandatory teardown plus a ledger. Ledger example: `~/Powerlifting-Analyzer/docs/reports/cloud-expense-ledger.md`.
- **Storage:** the rclone remote `r2:` (Cloudflare R2, bucket `pa-source`) works from this Mac. Creds are in `~/.config/cloudflare-r2-pa-source.txt`. **Use only the prefix `r2:pa-source/hearsay/`**, and never touch the powerlifting prefixes.
- **vast CLI:** `uvx vastai`, with the API key in `~/.config/vastai/vast_api_key`; don't print or copy it. The account's SSH key matches `~/.ssh/id_ed25519.pub`.
- **Credit $13.37 at handoff**, about 6 h of H100. Tell Nathan if runs will need more; he adds credit himself.
- **Offers at ~01:20:** H100 SXM `29019357` at $2.20/h (verified, Czechia, 4 Gbps down, R 99.8); `41555534` at $3.22/h. Re-search before renting:
  `uvx vastai search offers 'gpu_name=H100_SXM num_gpus=1 verified=true datacenter=true disk_space>=100 reliability>0.98 inet_down>500' -o 'dph_total'`

## What to do next

1. **`/plan-review` for M5**, using the project copy in `.claude/skills/plan-review/`, which carries the HEARSAY standing failure modes. The time box is set at P1: about 3 h of build plus 1–3 training runs. Design points to settle:
   - **Multiple clip lengths**
     - Each batch samples a crop length from `hearsay.embed.test_duration_sampler`, with length-bucketed batches, padding and `attention_mask`.
     - Each epoch re-crops at a random offset.
     - Evaluate the holdout both with test-length crops (fixed seed) and on whole clips up to 8 s.
     - Test-time is `prepare_segment(x)`, the whole clip up to 8 s, **identical to M1 v2's scoring path**. Optionally average several crops for clips over 8 s.
   - **Augmentation** (both classes, keeping a **clean-only** holdout score beside the augmented one):
     - additive noise at 5–30 dB SNR
     - codec round-trips (MP3, Opus, AAC, AMR-NB, μ-law) via ffmpeg
     - band-limiting (3.4, 4, 7 kHz)
     - simulated RIR/reverb
     - random gain before normalization

     Consider precomputing augmented copies into the bundle, so the box's dataloader doesn't depend on ffmpeg.
   - **Fold discipline**
     - Train only on inner rows, never holdout rows.
     - Fusion needs stacking features that weren't fit on themselves. Out-of-fold predictions for all 5 folds cost 5 fine-tunes, which is too expensive. The cheaper route is to train on folds 0–3, predict fold 4 and the holdout, and use fold 4 as the stacking set. State the tradeoff with the main chat.
   - **Fine-tune scope:** partial first (top N transformer layers plus head, feature encoder frozen), and full only if stable. Class-balanced sampling. Choose the checkpoint on fold 4 minDCF, never on the holdout.
2. **Data bundle.**
   - Decode each training-sample row (and the test set) with `hearsay.audio.load_audio`, which gives 16 kHz mono and uniform format. Store the full clip up to 12 s as FLAC, so crops can vary per epoch.
   - Include a manifest with path id, label, generator, speaker, source, group and fold.
   - Expect about 1.5–2 GB. Upload to `r2:pa-source/hearsay/bundle/`.
   - Also bundle the code (`src/hearsay`, the training script) and pin the XLS-R revision, or ship the local weights (1.27 GB).
3. **Pilot before fan-out** (fleet-campaign). One short run of a few hundred steps to check throughput, loss and the minDCF readout. Then the real run. Checkpoint to R2 regularly, and have the box **self-destruct at DONE**. Record the spend in a ledger.
4. **Deliver:**
   - `outputs/detector_scores/m5_xlsr_ft.csv` with columns `path, fold, split (stack|holdout|test), score, logit`.
   - `models/m5_*/meta.json` with the holdout readouts via `hearsay.metrics.report`, including `sponsor_code_asis` and `sponsor_code_flipped`, per generator and per bona fide source, clean vs augmented.
   - The checkpoint on a private HF Hub repo or GitHub release (not git LFS; the Docker image needs it).
   - A short report in `docs/reports/` on what worked and what didn't.
   - **Do not write TSVs or touch `submissions/`.** The main chat turns M5 scores into submissions.

**Do not touch:**
- `submissions/`
- `splits/nsa_folds.csv`
- the M1 scripts: `scripts/extract_embeddings.py`, `train_probe.py`, `make_probe_csv.py`
- `src/hearsay/embed.py`: import from it, don't edit it (the main chat owns it)
- the CPU chat's detector files
- the local GPU

Stage only your own files (`git add <paths>`; never `-A`). Don't push without asking Nathan.

## IMPORTANT: tests & at-risk artifacts

- Test: `cd ~/Projects/hearsay && uv run pytest -q` → **152 passed** (at `3a6fed9`/`90e9f0b`). `uv run ruff check .` → clean. Add tests for the crop and augmentation code, e.g. deterministic crops given a seed, no tiling, and augmentation that preserves length and label.
- **In flight** (main chat): v2 embeddings, then M1 training. Check with `pgrep -fl "extract_embeddings|train_probe"` and `ls models/`.
- **At risk:**
  - Any vast box: data dies with the instance. Checkpoint to R2 and keep the ledger.
  - `outputs/`, `models/`: gitignored and not archived.
  - The raw training corpora exist only on the external drive `/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/`, which must be mounted to build the bundle.
- **Spend:** the vast account balance is tracked through `uvx vastai show user --raw` (the `credit` field). Destroy instances at DONE, and verify with `uvx vastai show instances`.

## Analytical notes

- **Local measurements (MPS):** XLS-R forward is 62 ms p95 per 4 s window (fp32), so a fine-tune on the Mac would take roughly 3–4 h for 3 epochs × 20k clips, and would block the GPU. That's why we use vast.
- **The public-data frozen probe** found layer 7 best on ASVspoof. Fine-tuning usually helps most by adapting the middle and upper layers.
- **The sponsor scoring code** (`data/nsa/HackGTMinDCF/`) treats higher = bona fide. `hearsay.metrics.sponsor_min_dcf` reproduces it; report both directions.
- **Physical replay is not covered** (no hardware); only simulated RIR is. State that in the report.

## Pointers

`CLAUDE.md` → `docs/plan.md` (Ladder, Data, Timeline, Docker, Cut order: M5 is capped at 1 run if behind) → `docs/STATUS.md` → `src/hearsay/embed.py` (`prepare_segment`, `normalize_windows`, `test_duration_sampler`) → `src/hearsay/metrics.py` → `scripts/make_folds.py` → the `~/Powerlifting-Analyzer/scripts/cloud/*` scripts and `fleet-campaign` skill → `.claude/skills/plan-review/SKILL.md` (HEARSAY standing failure modes).
