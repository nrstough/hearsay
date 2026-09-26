# Handoff: CPU detector track (Sat Sep 26, 2026, ~01:10)

**Purpose of this chat:** run HEARSAY's **CPU-only** detector work in parallel with the main chat, which owns the GPU, M1 and TSVs. Deliver interpretable engineered detectors (handcrafted spectral + prosody first, then metadata/container, compression, splice, ENF). Each one needs out-of-fold and holdout scores on the shared fold file and a scored test set, in a fixed format that the main chat's fusion (M4) can consume. Record what worked and what didn't. **Never use the GPU (MPS).**

## Context

HEARSAY is the HackGT 13 NSA HEARSAY entry, rescoped to **software only** (commit `a523112`; audited Acceptable by Claude and Codex).

**Scoring**
- 60% MinDCF. False alarms (real called synthetic) cost 4×, and about 70% of test files are real.
- `hearsay.metrics`: `C_FA = 4`, `π_synth = 0.3`, i.e. normalized minDCF = 9.33·P_FA + P_miss. Only ranking matters.
- 20% forensic diversity: "points for each distinct technique your system actually leverages". This track is where most of those points come from.
- 20% docs, including "what worked / what had no effect".

**Output format.** 1.0 = synthetic. Never flip (decided by Nathan; see `docs/plan.md`, "Score direction").

**Data**
- **Test set:** `data/nsa/HackGTHearsayTesting/`, 1,671 WAVs, all PCM16 16 kHz mono, 3.0–13.6 s (median 3.4 s), peaks near 1.0, almost no leading silence.
- **Training:**
  - DiffSSD synthetic, 70k clips from 10 generators, 22 kHz WAV plus MP3 for ElevenLabs and PlayHT. Symlink `data/nsa/DiffSSD` points to the external drive.
  - LJ Speech real, 13,100 clips (`data/ljspeech`).
  - LibriSpeech dev+test real, 5,323 clips (`data/librispeech`).
  - Both symlinks point to `/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/`, and **the drive must be mounted**.
- **Training sample:** `outputs/manifests/nsa_train_sample.csv`, 20,000 rows. That's 1,000 per generator, plus 6,000 LJ and 4,000 LibriSpeech.
- **Fold file:** `splits/nsa_folds.csv`.
  - Groups: spoof by generator; bona fide by speaker, or by chapter for LJ.
  - **Outer holdout** = generators `playht` and `wavegrad2`, plus 26 bona fide groups. Inner folds are `0`–`4`.
- **Detector contract:** `src/hearsay/detectors/base.py`.
  - Returns `DetectorResult(name, score∈[0,1] increasing with synthetic, evidence, features, status)`.
  - `safe_run` never loses a row.
  - Fusion uses `logit(clip(score, 1e-4, 1−1e-4))` when `status == "ok"`.

**Shortcut findings.** These come from `outputs/inventory/nsa_train_vs_test_shortcuts.csv`.
- Peak level alone separates the classes (AUC 0.66), and LibriSpeech has 0.37 s of leading silence against 0.06 s in the test set.
- Durations differ too: training 5–9 s, test about 3.4 s.
- `hearsay.handcrafted` already trims silence, RMS-normalizes, and excludes duration, peak and absolute level.
- **Container format is a trap.** Training fakes include MP3s and 22 kHz WAVs, but every test file is 16 kHz PCM WAV. Any metadata or compression feature that sees the original container will learn "mp3/22 kHz = fake" and fall apart on the test set. Measure that before trusting it.

## Working branch / worktree

`main` in `~/Projects/hearsay`, clean at `f394717` when this was written. **The main chat commits to the same worktree at the same time**, so:
- Stage only your own files: `git add <paths>`. **Never `git add -A`.**
- Don't push without asking Nathan.

## Environment / setup

```bash
cd ~/Projects/hearsay
uv sync
ls data/nsa/DiffSSD data/ljspeech data/librispeech   # external drive must be mounted
uv run pytest -q                                       # expect 152 passed
```

Run everything with `uv run …` from the repo root. Use at most about 6 CPU workers: the main chat's GPU extraction also needs CPU for decoding.

## What to do next

1. **Handcrafted features for the training sample** (in flight from the previous chat).
   - Check that `outputs/handcrafted/nsa_train_sample.csv` exists with 20,000 rows. `outputs/handcrafted/nsa_test.csv` is already done: 1,671 rows, 75 features, 0 failures.
   - If it's missing (the job died with the old chat), re-run:
     `uv run python scripts/extract_handcrafted.py --manifest outputs/manifests/nsa_train_sample.csv --name nsa_train_sample --workers 6`
     It takes about 0.63 s per clip, about 35 min with 6 workers.
2. **Train and read out the handcrafted detector.**
   - Run `uv run python scripts/train_handcrafted.py --train nsa_train_sample --folds splits/nsa_folds.csv`.
   - Record CV minDCF (logreg vs LightGBM), and holdout minDCF, EER and AUC under every reading (including `sponsor_code_asis` and `sponsor_code_flipped`).
   - Record per-generator and per-bona-fide-source results, and the top-15 feature AUCs.
   - Write a short report in `docs/reports/` covering what separated the classes and what didn't.
3. **Export scores for fusion.** This is the format the main chat's M4 expects; add it to `scripts/train_handcrafted.py`.
   - Write `outputs/detector_scores/handcrafted.csv` with columns `path, fold, split, score, logit`:
     - `split` is `inner_oof` for inner rows (OOF predictions from the predefined folds), `holdout` for outer-holdout rows (model fit on all inner rows), and `test` for the 1,671 test files (keyed by path; the manifest is `outputs/manifests/nsa_test.csv`).
     - `score` is P(synthetic), and `logit` is the decision value.
   - Every learned piece is fit fold-locally, including the scaler.
4. **Length parity with the deep detector (v2).** The main chat's embeddings now use `hearsay.embed.prepare_segment`: trim, then a test-duration random crop with `seed + row` for training, no tiling, 8 s cap. `hearsay.handcrafted._crop` still takes the first 4 s after the trim. For a fair fusion:
   - Switch `_crop` to `prepare_segment(x, crop_s, seed)`, with crop lengths drawn exactly as `scripts/extract_embeddings.py --segment --crop test --seed 0` draws them (`test_duration_sampler(0)`, one draw per manifest row, in order). Test-time uses `prepare_segment(x)` with no crop.
   - Re-extract and compare holdout minDCF against the first version, and log both.
5. **Wrap as a contract detector.** Add `src/hearsay/detectors/handcrafted.py`, a `Detector` that loads the saved model and returns a `DetectorResult`. Its evidence names the top contributing features, e.g. "high-band energy ratio 3.1 SD above real speech". Add tests under `tests/`, following `tests/test_detector_contract.py`.
6. **Next CPU detectors**, time-boxed per `docs/plan.md` "Forensic techniques". The same score export applies to each.
   - **Container/metadata**, 1.5 h. Use ffprobe/ExifTool embedded fields only, never file dates or filenames. First check that the test set is uniform (all PCM16 16 kHz), then report honestly that the container carries almost no signal on it. It still counts as a technique if used, so gate it via the orchestrator.
   - **Compression forensics**, 1.5 h. Look for MP3-origin traces that survive transcoding to WAV (spectral holes, a ~16 kHz lowpass at 44.1 kHz origin). Because training fakes include raw MP3s, **transcode every training clip to 16 kHz PCM WAV first**, so train matches test.
   - **Splice** (1 h) and **ENF** (1 h) come last. They are the first to be cut if time runs short.

**Do not touch** (the main chat owns them):
- `src/hearsay/embed.py`, `src/hearsay/probe.py`, `src/hearsay/submission.py`
- `scripts/extract_embeddings.py`, `scripts/train_probe.py`, `scripts/make_probe_csv.py`
- `submissions/` (the TSVs and `log.csv`)
- `splits/nsa_folds.csv`: never regenerate it; everyone shares it.

## IMPORTANT: tests & at-risk artifacts (make sure these survive)

- Test: `uv run pytest -q` → **152 passed** at `f394717`. `uv run ruff check .` → clean.
- Test: `uv run pytest -q tests/test_docs_consistency.py` → must stay green. It pins the metric constants and the no-hardware/no-CSV wording in living docs, and checks the plan's technique list.
- **In flight:** handcrafted extraction for the training sample, started by the previous chat. Check with `ls -la outputs/handcrafted/` and `pgrep -fl extract_handcrafted`. If the old chat was closed, the job may have died; re-run step 1.
- **In flight, main chat's job, don't interfere:** GPU v2 embeddings `nsa_test_v2` and `nsa_train_sample_v2`, then M1 training. Check with `pgrep -fl extract_embeddings`.
- **At risk:**
  - `outputs/` (handcrafted CSVs, manifests, inventory) and `models/`: gitignored and **NOT archived**. They can be regenerated from the scripts, but slowly.
  - `data/nsa/` (test set, LJ subset, template, scoring code) is archived at `/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/nsa/`.
  - DiffSSD, LJ Speech and LibriSpeech **exist only on the external drive**; the originals were downloaded to `~/Downloads/DiffSSD.tar` (21.5 GB), keithito.com and openslr.org.
- **Tracked:** `submissions/log.csv`. It holds the M0 row. The M0 constant TSV (`submissions/20260926-0036_M0_constant.tsv`, all 0.0) is archived to the external drive's `hearsay/submissions/`.

## Analytical notes

- **Sponsor scoring code:** `data/nsa/HackGTMinDCF/asvspoof5/evaluation-package`. It's ASVspoof5 with `Pspoof = 0.5, Cmiss = 1, Cfa = 4`, and it treats a **higher score as bona fide**. `hearsay.metrics.sponsor_min_dcf(y, s, flip)` reproduces it exactly (a test pins that). Report both `flip=False` and `flip=True`, and select on our π = 0.3 minDCF.
- **Public-data M1 baseline, not an NSA number:** frozen XLS-R layer 7 plus logistic regression gets minDCF 0.023 on ASVspoof19 A07–A19. It scores pure silence as 0.99 synthetic. That's a shortcut warning, not an NSA number.
- **Handcrafted cost:** about 0.63 s per clip, dominated by YIN pitch tracking. The first call pays a roughly 30 s numba JIT warm-up.
- **Multiprocessing on macOS:** scripts must be files with an `if __name__ == "__main__":` guard. Inline `python - <<EOF` with a `ProcessPoolExecutor` fails.
- **LJ is single-speaker,** and 4 of the 10 DiffSSD generators speak in LJ's voice. That's good: those pairs isolate synthesis artifacts from speaker identity. The multi-speaker generators share 10 cloned target voices across folds, a stated limit in `scripts/make_folds.py`.

## Pointers

Read in this order:
1. `CLAUDE.md`
2. `docs/plan.md`: "Forensic techniques", "Architecture", "Data", "Timeline" hard cutoffs. **Teammate detector out-of-fold scores are due Sat 19:00; fusion freezes Sat 22:00.**
3. `docs/nsa-challenge.md`: the brief's rubric list.
4. `src/hearsay/detectors/base.py`
5. `src/hearsay/handcrafted.py`, `scripts/extract_handcrafted.py`, `scripts/train_handcrafted.py`
6. `scripts/make_folds.py`
7. `docs/reports/2026-09-25_sponsor-questions.md`: the rules, including Nathan's notes from the talk.
