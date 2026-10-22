# Handoff — V2-31 shared symmetric augmentation library (2026-10-22)

**Purpose of this chat:** build the library half of V2-31: one augmentation module every detector trains through, symmetric by construction, with a pre-declarable chain spec and tests. **No retraining in this chat** (MPS is held by other lanes and the retrain is the chunk's second half).

## Context

HEARSAY (`docs/prod-chunks.md` Part C is the plan of record; chunk **V2-31**, Phase 3) is an audio deepfake detector from HackGT 13, now in v2. v1 built augmentation three separate times and never augmented the carrier (M1b, a frozen XLS-R layer-7 probe):
- `src/hearsay/augment.py` (M5's trainer): `add_noise` (pink), `band_limit`, `reverb` (parametric RT60), `gain_clip`, `rawboost_conv`, `trim_jitter`, an `Augmenter` class and `holdout_slice`. Used by `hearsay.m5_data` / `scripts/m5_train.py`.
- the handcrafted v5 "laundering" rows (codec round-trips, random tilt + low-pass) via `scripts/extract_handcrafted.py --extra-rows-from` and `hearsay.handcrafted`.
- the compression detector's laundering equalisation in `hearsay.compression` / `scripts/extract_compression.py`.
v1's biggest measured hole is 20 dB noise (holdout 0.012 → 0.269, `docs/reports/2026-09-26_channel-robustness.md`). The Oct 21 amendment to V2-31 says the RL/adversarial search over augmentation *chains* lives here later and targets channel robustness; it is not a fix for the V2-02 miss style (which is wide-band and clean, `docs/reports/2026-10-20_v2-02-error-atlas.md`).

What happened Oct 20–21 that bears on this: the NSA answer key is the sealed set (rule 3; never opened by lanes); rule 5's primary reading is package-0.3; the 2B AntiDeepfake column is now a benchmark column (registry v1.3) and the design of record for fusion is a conditions-gated blend (V2-60); V2-30's profile pass found no available corpus carries the miss style, so augmentation and the gate are the active levers for robustness. V2-24 (the mechanistic probe) is running in its own session on `claude/priceless-payne-4a4d4e` and holds MPS when it needs it.

## Working branch / worktree

Start from `v2` (pushed, `origin/v2` at `66e63d9`), clean apart from five long-standing uncommitted files that are not this lane's (`.claude/codex-audit-prompt.md`, `.claude/launch.json`, `docs/specs/2026-09-26_m5-xlsr-finetune-audit.md`, `next-env.d.ts`, `tsconfig.json`): leave them alone. Work on a branch cut from `v2` (e.g. `v2-31-augment`), in the main checkout or a worktree; commit, do not push unless Nathan says so.

## Environment / setup

```bash
cd /Users/nathanstough/Projects/hearsay
uv sync            # once
uv run pytest -q   # suite: 1,268 passed, 3 skipped on Oct 21 (about 6 min)
uv run ruff check .
```
ffmpeg is installed (codec round-trips). Audio corpora are under `data/` (gitignored) and the external Crucial drive at `/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/` (it has dropped off the Mac twice this week; check the mount before any run that reads it). **No MUSAN noise bank and no measured RIRs are on disk** (`ls data | grep -i "musan|rir|noise"` is empty).

## What to do next

1. Read: `docs/prod-chunks.md` V2-31 (and V2-30, V2-33 for who consumes this), `src/hearsay/augment.py`, `hearsay.m5_data`, `scripts/extract_handcrafted.py --extra-rows-from`, `hearsay.compression` laundering, `hearsay.embed.prepare_segment` (the 7.25 kHz band match), `docs/code-map.md`.
2. Promote `src/hearsay/augment.py` into the shared module (keep the import path or add a compatibility import; **M5's trainer must keep working**, pinned by a test that imports `hearsay.m5_data` and builds an `Augmenter`).
3. Unify the three code paths behind it (handcrafted laundering, compression laundering) without changing their outputs for existing bundles; a parity test per path (same seed → same samples as before the refactor, on a synthetic clip).
4. Add, each seeded and deterministic, with a unit test:
   - additive noise at a target SNR from a **noise bank** interface; ship a small synthetic bank (pink, brown, babble-like, hum) and a loader for a real one (MUSAN layout), and say in the docs which is which;
   - room impulse responses: a parametric generator (exponential decay, early reflections) labelled synthetic, plus a loader for measured RIRs;
   - speed and pitch perturbation;
   - telephony: 8 kHz band-limit + G.711 round-trip; AMR-NB via ffmpeg if the encoder exists (detect, skip with a reason otherwise);
   - MP3 and AAC round-trips at a few bitrates via ffmpeg (reuse what `scripts/channel_codec.py` already does);
   - the test set's 7.2 kHz resampler wall, via `prepare_segment`'s band match.
5. **Symmetry structurally:** the API takes a batch or a manifest and applies the same chain distribution regardless of label; a test proves a label-conditioned call is impossible or refused. Every chain is a JSON-serialisable spec that hashes (so it can be pre-declared), and the module records the chain applied per clip.
6. Write `docs/reports/2026-10-22_v2-31-augmentation-library.md` (API, what is real vs synthetic, the follow-up: retrain M1b through it on MPS and benchmark clean vs augmented on the noise cohort). Update `docs/prod-chunks.md` V2-31 status and `docs/code-map.md`.
7. ruff clean, full suite passing, commit. Report.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test: `uv run pytest -q` → expected **1,268 passed, 3 skipped** before your changes (Oct 21); add yours on top. `uv run ruff check .` → "All checks passed!".
- Parity tests you must add: M5's `Augmenter` path, handcrafted laundering, compression laundering (same seed → same output as pre-refactor).
- At-risk (not this lane's, do not touch): `benchmarks/holdout_ledger.jsonl`, `benchmarks/sealed_ledger.jsonl`, `outputs/v2/benchmark/` (ledgered reads), `data/nsa/HackGTAnswerKey.tsv` (sealed key, never opened).
- At-risk, archived: `models/`, `outputs/`, `data/`, `weights/` are gitignored and live on the Mac and the Crucial drive; nothing this lane makes should live only in a job-tmp dir.
- In flight elsewhere: V2-24 (`claude/priceless-payne-4a4d4e`) may hold MPS; check `pgrep -fl "python.*v2_|extract_embeddings|bakeoff"` before any torch job. This lane needs none.

## Analytical notes

- Rule 5 reading for any number you quote: `hearsay.metrics.min_cost(y, s, 0.3, c_fa=1.0, c_miss=4.0)` (package-0.3), with package-0.5 and the brief beside it. Every augmented score is reported next to a clean-only score (model rule).
- The noise hole: holdout 0.012 → 0.269 at 20 dB white noise for the shipped rule; M1b P_miss 0.07 → 0.88 at the inner threshold (`docs/reports/2026-09-26_channel-robustness.md`).
- The test set's wall is a smooth resampler low-pass at ~7.2 kHz, not MP3/AAC (`scripts/channel_codec.py`, 34 settings tried); the band match in `prepare_segment` reproduces it at 7.25 kHz.
- Shortcut risk the module must not create: a transform applied to one class only (the container-is-the-label finding, README null results). Hence structural symmetry.

## Pointers

- `docs/prod-chunks.md` V2-31, V2-30 (status Oct 21: no corpus passed the style profile), V2-60 design of record (Oct 21), C.1 rules.
- `docs/reports/2026-09-26_channel-robustness.md`, `docs/reports/2026-09-26_handcrafted-v4.md` (v5 augmentation), `docs/reports/2026-09-26_m5-xlsr-finetune.md` (M5's augmentation).
- `docs/handoffs/2026-10-20_v2-24-what-sees-the-misses-handoff.md` for the sibling lane's rules on MPS and the key.
- Memory for the next chat: the sealed key is `data/nsa/HackGTAnswerKey.tsv`, testing only; rule 5 = package-0.3.
