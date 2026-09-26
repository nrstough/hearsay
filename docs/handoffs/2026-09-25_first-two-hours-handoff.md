# Handoff — first-two-hours (2026-09-25, 8:01 pm)

**Purpose of this chat:** hacking has started (8:00 pm Fri Sep 25). Birth the repo, then run the model owner's first-two-hours runbook from `docs/plan.md`: read the NSA rules, inventory the drop, build grouped splits, ship the **constant-prior CSV**, verify Spectra-AASIST score direction, and start the latency microbenchmark plus frozen-embedding extraction. No custom architecture work in this window.

## Context

HEARSAY is an audio deepfake detector for the HackGT 13 NSA challenge (clip in → 0–100% synthetic likelihood → CSV on a hidden test set), plus a live pan-tilt listening-head demo. I own the detector, validation split, CSV, and streaming scorer. Pre-event setup is done: uv env, skills, CLAUDE.md, Codex review scripts, weights and two datasets downloaded. **No application code exists yet**; `src/hearsay/` is empty.

`docs/plan.md` is the working plan and **supersedes the model ladder in CLAUDE.md** where they differ (CLAUDE.md: "the docs win on plan and scope"). Plan changes: fine-tune M5 comes right after the first learned CSV; M4 is logistic fusion + Platt scaling, not gradient boosting; M2 is one 30-minute gate; M6 is cut unless labels are explicit; 2 mics; no laser. The core front end is `facebook/wav2vec2-xls-r-300m`.

## Working branch / worktree

`~/Projects/hearsay`, **not a git repo yet**. `scripts/first-commit.sh` has not run. It is past 8:00 pm, so it will now run.

## Environment / setup

```bash
cd ~/Projects/hearsay
bash scripts/first-commit.sh      # FIRST: git init, private GitHub repo nrstough/hearsay, first commit, push
                                  # prompts about docs/master-doc.md (missing): answer y, or add it first
uv sync                           # env already built; Python 3.12 in .venv
uv run python -c "import torch; print(torch.backends.mps.is_available())"   # expect True
```

- Run everything with `uv run …` from the repo root.
- Load audio with `soundfile` / `librosa` / ffmpeg. `torchaudio.load()` fails here (needs torchcodec, not installed); `torchaudio.functional.resample` works.
- ffmpeg 7.1.1 has encoders for Opus, AMR-NB, AAC, MP3, and μ-law (the codec round-trip set).
- Hugging Face: **not logged in**. Run `hf auth login` (user types the token) before requesting MLAAD access.
- After the first push, the user adds collaborators: GitHub → repo Settings → Collaborators, or `gh api -X PUT repos/nrstough/hearsay/collaborators/<user> -f permission=push`.

## What to do next (plan.md runbook, clock starts when the NSA data is in hand)

1. **Birth the repo** (`scripts/first-commit.sh`, above).
2. **0:00–0:20: rules.** Read the metric, label semantics, outside-data policy, and submission schema. Send the sponsor three written questions: outside data allowed? which metric? are manipulation types labeled? Also ask whether pretrained anti-spoof checkpoints count as outside data, and whether pre-event weight downloads are OK.
3. **0:20–0:40: inventory the NSA drop** into `data/nsa/` (gitignored). Report N, formats, sample-rate and duration histograms **by class**, class balance, and decode failures on a stratified sample. Use `/validate-training-data`.
4. **0:40–1:00: split and rollback CSV.**
   - Build a generator- and speaker-disjoint split (or the best available grouping key, stated explicitly).
   - Write the **constant-prior CSV** through the real test loader: one path to 16 kHz mono float32, every file yields one finite score in exact row order.
   - Save it as a new file under `submissions/` and append a row to `submissions/log.csv`.
   - Hash-dedupe the NSA train/test against In-the-Wild and ASVspoof.
5. **1:00–1:30: Spectra-AASIST on a labeled sample.** Verify score direction, then fit a provisional Platt map. Model card: logits `(batch, 2)`, **index 0 = spoof, index 1 = bonafide**, default threshold −1.140625 on the bonafide logit. The exported value must *increase* with synthetic likelihood; assert that with known examples before any CSV ships.
6. **1:30–2:00: latency and extraction.** Run the 100-window XLS-R latency microbenchmark on MPS (4 s windows; gate is 750 ms at the 95th percentile). Start frozen XLS-R embedding extraction for M1 and the bake-off.
7. **Hard gate:** no learned CSV by Friday midnight → stop the bake-off and ship calibrated Spectra-AASIST.
8. **Parallel non-model tasks (plan.md, "Before 8 pm" leftovers):**
   - Request MLAAD access.
   - Set up a vast.ai datacenter-verified H100 with ≥100 GB disk. **Ask the user before renting**; spend cap ~$75.
   - Pull a ReplayDF subset (not downloaded; CC-BY-NC-SA-4.0, Git LFS).
   - Pick the demo phones.
   - Replay capture with the EE ~23:15–00:15: 480-clip matrix. Split by source utterance and speaker **before** replaying, and hold out one phone-position cell.

Any rung expected to take >1 h goes through `/plan-review`. Time-box every rung; when the box runs out, stop and report.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- **Tests:** none exist. `uv run pytest` collects 0 tests (exit 5); report that as "no tests", not a pass. Markers are ready: `needs_data`, `needs_weights`, `slow`.
- **Before every CSV** (plan.md): run one music file and one silence file through the loader; check schema, row order, finite range, and reproducibility.
- **In flight: ASVspoof 2019 LA download** (7.64 GB, ~50% at 8:01 pm, ETA ~8:20 pm). It is `aria2c` running as a background job of the *previous* chat and may die with it. Check with `ls -l data/asvspoof2019/`: done when `LA.zip.aria2` is gone. If it's interrupted, resume with the same command (the `.aria2` file makes it pick up where it stopped; the md5 is verified automatically at the end):
  ```bash
  cd ~/Projects/hearsay/data/asvspoof2019 && aria2c -x16 -s16 -k20M --file-allocation=none --continue=true --max-tries=20 --retry-wait=5 --console-log-level=warn --checksum=md5=30c98f11d8b2bc21f2c257bfd78bb5c5 -o LA.zip "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
  ```
  Then `unzip -q LA.zip` in that directory. Source: Edinburgh DataShare, handle 10283/3336, ODC-BY. It's slow per connection, so keep the parallel connections.
- **At-risk: `weights/`** (4.0 GB: XLS-R 300M, WavLM-large, WavLM-base, Spectra-AASIST). Only on the Mac and gitignored. Re-downloadable from Hugging Face, all verified to load. **NOT archived.**
- **At-risk: `data/in_the_wild/`** (16 GB incl. 8.2 GB zip). SHA-256 verified; 31,779 wavs (19,963 bona-fide / 11,816 spoof); labels in `release_in_the_wild/meta.csv`. **Held-out stress test only, never train on it.** NOT archived.
- **At-risk: submission CSVs.** `submissions/*.csv` is **gitignored**; only `log.csv` is tracked. Archive every submitted CSV to the external drive: `/Volumes/Crucial P3 NVME Gen 3 2TB` (104 GB free, Thunderbolt NVMe ~2.7 GB/s). NOT archived yet (none exist).
- **At-risk: the NSA dataset** once it arrives. It exists nowhere else; copy it to the external drive as soon as it lands.

## HEARSAY status

Rung: pre-M0 · Latest valid CSV: none · Val (held-out generators): clean — / replay —
Time left to freeze: ~34 h to Sunday ~06:00–08:00 freeze (hacking ends 8 am Sun) · Contract changes pending: none

## Analytical notes

- **Mac:** 65 GB free; M3 Pro, 18 GB unified memory, MPS works. M3 Pro timings in the plan are analytic (133 GFLOPs per 4 s XLS-R window), not measured; the microbenchmark replaces them.
- **Replay evidence:** EER 4.7% → 18.2% under physical replay; failures are asymmetric (fakes pass as real). Keep a clean-only validation score beside any replay-augmented one; the NSA test set is probably clean.
- **Class imbalance:** anti-spoof sets run ~90% spoof. Use a weighted sampler at ~50% bona fide per batch; never quote accuracy; never assume the test prior matches train.
- **Metric unknown:** keep raw logits, calibrated probabilities, and hard decisions separately. Submit calibrated probabilities, with a rank-preserving raw-score CSV ready.
- **Shortcuts to check in hour one:** clip duration correlating with label, and silence duration. VAD-gate the live scorer. Never denoise or loudness-normalize test audio.
- **License conflicts to settle before Devpost** (both recorded in the CLAUDE.md AI use disclosure):
  - Spectra-AASIST: repo header Apache-2.0 vs card text MIT.
  - In-the-Wild: Hugging Face CC-BY-SA-4.0 vs site Apache-2.0.
  - WavLM: no license on its Hugging Face card (released via unilm, MIT).
- **Other:** `docs/master-doc.md` is still missing. The external drive also holds unrelated personal data; only write into a new `hearsay/` folder there.

## Pointers

Read first: `CLAUDE.md` → `docs/plan.md` (working plan, first-two-hours table, hard gates, cut order) → `docs/scoping.md` (the VeriLM scoping memo; evidence, ~283 KB, search it rather than reading linearly). Skills: `/plan-review`, `/validate-training-data`, `/diagnose`, `/handoff`. Log every CSV in `submissions/log.csv`.
