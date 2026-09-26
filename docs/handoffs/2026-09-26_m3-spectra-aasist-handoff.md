# Handoff: M3, Spectra-AASIST as the second deep detector (Sat Sep 26, 2026, ~03:35)

**Purpose of this chat:** run the off-the-shelf Spectra-AASIST detector over the NSA fold file and test set, and export its scores in the fusion format so M4 (fusion) gets a second deep score stream that is decorrelated from M1. Time box: **45 min of build plus the scoring run**; `docs/plan.md` says "drop on friction". No training, no fine-tuning: this is a scoring rung.

**Deliver:**
- `outputs/detector_scores/spectra_aasist.csv` with columns `path, fold, split, score, logit`, `split` ∈ {`inner_oof`, `holdout`, `test`}, one row per fold-file row (20,000) plus one per test file (1,671), same shape as `outputs/detector_scores/handcrafted_v1.csv`.
- `models/m3_spectra_<stamp>/meta.json` via `hearsay.metrics.report` on the outer holdout: overall, per generator, per bona fide source, `sponsor_code_asis` and `sponsor_code_flipped`, plus the direction check.
- A short note in `docs/reports/2026-09-26_m3-spectra.md` (numbers, timing, what it's blind to), and an M3 line in `docs/STATUS.md`.
- **Do not write TSVs or touch `submissions/`.** The main chat turns detector scores into submissions.

## Context

HEARSAY is the HackGT 13 NSA HEARSAY entry (software only). Read `CLAUDE.md`, `docs/plan.md`, `docs/STATUS.md` first. Scoring is 60% normalized minDCF with C_FA = 4, π_synth = 0.3 (false alarms on real speech cost 9.33× a miss); output 1.0 = synthetic, never flip.

**Four chats run in parallel. Stay in your lane.**

| Chat | Owns |
|---|---|
| Main | local MPS GPU, M1, fusion (M4), orchestrator, **all TSVs and `submissions/`**, `src/hearsay/embed.py`, the M1 scripts |
| CPU detectors | engineered detectors, CPU only (`docs/handoffs/2026-09-26_cpu-detectors-handoff.md`) |
| M5 | XLS-R fine-tune on vast.ai only (`docs/handoffs/2026-09-26_m5-finetune-handoff.md`) |
| **This one (M3)** | Spectra-AASIST scoring. New files only: `scripts/score_spectra.py`, `tests/test_spectra.py`, the report. `src/hearsay/spectra.py` is yours to extend (nobody else edits it). |

**Device decision (make it first, with Nathan):** Spectra-AASIST wraps a full XLS-R 300M, so it costs about what M1's extraction cost: ~0.2 s per 4 s window on MPS, roughly 3–5× that on CPU. 21,671 clips at up to 16 windows each is **~1–2 h on MPS, 4–8 h on CPU**. The main chat is using the MPS GPU (an In-the-Wild extraction was running at 03:30: `pgrep -fl extract_embeddings`). Options:
1. **Coordinate MPS with the main chat**: run when their extraction finishes, or run at lower priority (`nice`) in segment mode with `max_windows=4`.
2. **CPU with `max_windows=2`** on the 6 spare cores (`nice -n 10`), which fits the 45-min box only for a subset; use `--limit` for a pilot, then let it run in the background.
Ask before taking the GPU. Never run on vast.ai (that's the M5 chat's budget).

**What already exists:**
- `src/hearsay/spectra.py`: `load_spectra(device)`, `spectra_logits(model, x, max_windows=16)` (repeat-pads short clips into 64,600-sample windows with 50% hop, pre-emphasis, mean of window logits), `synth_logit(logits) = logit_spoof − logit_bonafide` (increases with synthetic likelihood). Weights in `weights/Spectra-AASIST/` (`model.py` + `model.safetensors`, 1.26 GB; `load_spectra` patches the hub id to the local XLS-R copy).
- `scripts/spectra_direction.py`: asserts direction (AUC > 0.5 and spoof median above bona fide median) on a labeled manifest and fits an optional class-balanced Platt map. Already run on 494 ASVspoof 2019 eval clips: `outputs/spectra/asv19eval500_scores.csv` (raw logits per file; spoof synth_logit ≈ +11, bona fide ≈ −7 on the first rows). Result on that ASVspoof sample: AUC 1.0, EER 0.0, spoof median synth_logit +9.7 vs bona fide −7.6, **0.169 s per clip on MPS**; provisional Platt map (a = 4.42, b = 10.51) in `outputs/calibration/spectra_platt_asv19eval500.json`. That is public data with a known silence shortcut; expect worse on NSA.
- `scripts/score_detector.py` shows the fusion-format export for non-learned detectors (`split=inner_oof` for inner rows, `holdout`, `test`; NaN score for failed rows; `logit = log(s/(1−s))` on a clipped score). Mirror its export block. `scripts/train_handcrafted.py:75-88` has `by_group()` for the per-generator / per-source readouts; copy it.
- Fold file: `splits/nsa_folds.csv` (path, label, generator, speaker, source, group, fold; holdout = playht + wavegrad2 + 26 bona fide groups; inner folds "0"–"4"). **Never regenerate it.** Test manifest: `outputs/manifests/nsa_test.csv` (`filename, path`, relative paths; row order is the submission order).

## Working branch / worktree

`main` in `~/Projects/hearsay` (shared worktree). Uncommitted at handoff: the M5 chat's untracked spec/plan/consult files under `docs/` (not yours; never stage them). Stage only your own files with `git add <paths>`; never `-A`. Don't push without asking Nathan.

## Environment / setup

```bash
cd ~/Projects/hearsay && uv sync
uv run pytest -q          # 215 passed at 3c170d5
uv run ruff check .       # clean
uv run python -c "from hearsay.spectra import load_spectra; m = load_spectra('cpu'); print(type(m).__name__)"
```

## What to do next

1. **Device decision** with Nathan (above). Check `pgrep -fl extract_embeddings` and the main chat before touching MPS.
2. **Input path parity.** Spectra was trained on ~4 s crops. Decide and record whether to score (a) `spectra_logits` on the raw `load_audio` clip (current code: repeat-pads short clips, which is the tiling shortcut M1 v2 removed), or (b) the silence-trimmed clip via `hearsay.embed.prepare_segment(x)` (trim, cap 8 s) and then windows without repeat-padding for clips ≥ 64,600 samples, zero-padding shorter ones. Recommendation: (b), with the same treatment for train and test rows, so silence and tiling can't become shortcuts. Whatever you pick, the **same function scores every row** (inner, holdout, test).
3. **Direction check on NSA first**: run `scripts/spectra_direction.py` on ~300 inner-fold rows (150 per class, sampled from `splits/nsa_folds.csv` where fold != holdout) before any export. If AUC < 0.5 it exits non-zero; do not "fix" by flipping without telling Nathan.
4. **Write `scripts/score_spectra.py`**: loads the fold file + test manifest, scores every row with the chosen path, exports `outputs/detector_scores/spectra_aasist.csv` (score = sigmoid(synth_logit) or a Platt map fit **only on inner-fold rows**, never holdout; logit = synth_logit), writes `models/m3_spectra_<stamp>/meta.json` with `hearsay.metrics.report` on the holdout, `by_group` breakdowns, timing, device, `max_windows`, and the input-path choice. Support `--limit N` for a pilot and `--device`.
5. **Pilot** with `--limit 200`, check timing, extrapolate, then run the full set in the background (`nohup … &` or the terminal), logging to `outputs/spectra/score_spectra.log`.
6. **Tests** (`tests/test_spectra.py`, hermetic, no weights): direction contract (`synth_logit` increases with spoof logit), export schema and counts (20,000 + 1,671, unique paths, `split` values), NaN on decode error, no repeat-padding if you chose (b). Mark anything that needs weights `@pytest.mark.needs_weights @pytest.mark.slow` and skip when `weights/Spectra-AASIST` is missing (pattern in `tests/test_audio_and_submission.py:177-194`).
7. **Report + STATUS line**, disclosure line in `CLAUDE.md` (Spectra-AASIST "how used": M3 score stream; license still unclear, say so). Tell the main chat the file name so fusion picks it up.

## IMPORTANT: tests & at-risk artifacts

- Test: `uv run pytest -q` → **215 passed** at `3c170d5`; `uv run ruff check .` → clean. Add your tests; the docs-consistency test bans the bare word "CSV" in `CLAUDE.md`/`docs/plan.md` (write `.csv`) and any π other than 0.3.
- At risk (gitignored, NOT archived): `outputs/detector_scores/*.csv`, `outputs/spectra/`, `models/`. `weights/Spectra-AASIST` (1.26 GB) and `weights/wav2vec2-xls-r-300m` are local only.
- In flight (main chat, MPS): `extract_embeddings.py --manifest outputs/manifests/itw_stress.csv` at 03:30. Check `pgrep -fl "extract_embeddings|train_probe"`.
- The raw audio lives on the external drive (`/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/`, symlinked from `data/`); it must be mounted. ASVspoof/In-the-Wild/MLAAD were copied there at 03:12 and may be symlinked by the time you start.

## Analytical notes

- M1 v2 (frozen XLS-R probe, layer 7) is the bar for deep detectors: outer-holdout minDCF **0.1458**, EER 2.5% (`models/m1_wav2vec2-xls-r-300m_L7_20260926-0302/meta.json`); wavegrad2 0.21, LibriSpeech real 0.16. M3 does not need to beat it; its value is a second, differently trained deep score for the stacker. Report where it disagrees with M1 (per generator, per source).
- Handcrafted detector v3: holdout minDCF 0.25, blind to grad_tts, pro_diff and ElevenLabs. Compression forensics: near chance on decoded inputs.
- Known traps from the CPU chat's report (`docs/reports/2026-09-26_cpu-detectors.md`): the test set is band-limited differently from the training corpora, so anything keyed on the 7–8 kHz band false-alarms on real test audio. Check `frac_above_half` on the test set in `meta.json`: with a 70% real test set, a value far above 0.3 means false alarms.
- Sponsor scoring code treats higher = bona fide; `hearsay.metrics.report` gives both readings. Submit 1.0 = synthetic regardless.
- Spectra's output convention: index 0 = spoof, index 1 = bonafide (model card); `synth_logit` already handles it. The direction script is the guard.

## Pointers

`CLAUDE.md` → `docs/plan.md` (ladder row M3, cut order) → `docs/STATUS.md` → `src/hearsay/spectra.py` → `scripts/spectra_direction.py` → `scripts/score_detector.py` (export format) → `scripts/train_handcrafted.py:75-88` (`by_group`) → `src/hearsay/metrics.py` → `docs/reports/2026-09-26_cpu-detectors.md` (band-limit trap).
