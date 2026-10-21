# V2-53: benchmark rows for the AntiDeepfake backbones and WavLM Large

**Date:** 2026-10-20. **Lane:** V2-53 (Phase 5). **Branch:** `v2`. **Benchmark:** registry v1.2, reading per rule 5 (package-0.3 primary; package-0.5 and brief beside it). **Sealed key: not read.**

## 1. What was run

The four `nii-yamagishilab/*-anti-deepfake` checkpoints (CC BY-NC-SA 4.0; identity-bound loads, `outputs/v2/backbone/*.identity.json`), **zero-shot, nothing trained**, through the lane's scorer (`scripts/v2_backbone_score.py`, "ours" input view: band match at 7.25 kHz, trim, 8 s cap, the same path as v1's deep columns) over the inner folds (16,142), the outer holdout (3,858) and the test set (1,671, unlabelled), on the Mac's MPS. In-the-Wild (3,000) had been scored on Oct 13. The per-population exports were assembled into one Interface v1 score file per model (`scripts/v2_backbone_to_benchmark.py`, counts and flags asserted) and scored with `scripts/v2_benchmark.py score --holdout --lane v2-53`: **one ledgered holdout read per column**, their first.

A zero-shot model's inner rows are out-of-fold by construction; the fold column is carried so the fold table and the nested rank references work.

| Model | holdout + inner wall time | clips/s | holdout ledger key |
|---|---|---|---|
| mms-300m | 12:42–13:53 | ~4.5 | `58ac4647960b` |
| xls-r-1b | 13:53–15:50 | ~2.9 | `018576dc2c3f` |
| mms-1b | 15:50–17:42 | ~3.0 | `b2ced457d80b` |
| xls-r-2b | 17:42–21:59 (the session restart at ~19:20 killed the chain at 3,200 inner rows; resumed from the partial at 19:26) | ~1.2–1.7 (paging: swap was 13.3 of 14.3 GB before the restart) | `2837507bf7d6` |

WavLM Large needs no new run: `wavlm_l` (frozen layer-9 probe on the M1b recipe) has had a benchmark card since Oct 12, holdout read in the "benchmark-v1 baseline set" (`098654b3b458`).

## 2. The table

package-0.3 unless marked; 90% cluster-bootstrap intervals in brackets where the card has them (2,000 draws; inner by generator and speaker, ITW by speaker). **In-sample cells are struck through in meaning, not in value:** the AntiDeepfake training table (`outputs/v2/backbone/cards/*.card.md`) lists DiffSSD, LJSpeech and LibriTTS, so for those four models every inner and holdout cell is a number on data they were trained on, and only In-the-Wild is held out. Reference rows from `outputs/v2/benchmark/v1_scorecard.json`.

| Column | inner | holdout | **ITW** (held out for all) | ITW package-0.5 | ITW brief | ITW EER |
|---|---|---|---|---|---|---|
| **xls-r-2b-anti-deepfake** | 0.0261 ⁱ [0.0002–0.0573] | 0.0009 ⁱ | **0.0221** [0.0088–0.0392] | 0.0415 | 0.0340 | 1.10% |
| xls-r-1b-anti-deepfake | 0.0301 ⁱ [0.0002–0.0708] | 0.0000 ⁱ | **0.0279** [0.0162–0.0419] | 0.0465 | 0.0437 | 1.23% |
| mms-1b-anti-deepfake | 0.0522 ⁱ [0.0002–0.1151] | 0.0028 ⁱ | **0.0488** [0.0299–0.0654] | 0.0680 | 0.0730 | 1.88% |
| mms-300m-anti-deepfake | 0.0511 ⁱ [0.0020–0.0927] | 0.0025 ⁱ | **0.0849** [0.0584–0.1077] | 0.1100 | 0.1257 | 3.40% |
| spectra_aasist (v1, suppressor only; training data undisclosed) | 0.0027 ᵘ | 0.0022 ᵘ | 0.0682 [0.0471–0.0907] | 0.1330 | 0.0650 | 3.20% |
| **SHIP rule (v1)** | 0.2156 | 0.0053 | **0.1498** | 0.2385 | 0.2280 | 5.80% |
| m1b_v3 (v1 carrier) | 0.2280 [0.0412–0.3373] | 0.0356 | 0.1934 [0.1271–0.2554] | 0.2955 | 0.3420 | 7.55% |
| wavlm_l (built, not shipped) | 0.3888 [0.0336–0.6023] | 0.0314 | 0.2911 [0.2008–0.3690] | 0.3825 | 0.7523 | 11.57% |

ⁱ in-sample for this model (its training data contains the population's corpora). ᵘ unknown exposure.

Per-fold package-0.3 on inner (fold → generators: 0 grad_tts + unit_speech, 1 diffgan_tts + openvoicev2, 2 pro_diff + xtts_v2, 3 your_tts, 4 elevenlabs):

| Column | f0 | f1 | f2 | f3 | **f4 ElevenLabs** |
|---|---|---|---|---|---|
| xls-r-2b | 0.0012 | 0.0000 | 0.0000 | 0.0000 | **0.1154** |
| xls-r-1b | 0.0039 | 0.0000 | 0.0000 | 0.0000 | **0.1536** |
| mms-1b | 0.0138 | 0.0009 | 0.0000 | 0.0000 | **0.2409** |
| mms-300m | 0.0348 | 0.0032 | 0.0000 | 0.0000 | **0.1672** |
| SHIP rule | 0.3114 | 0.0021 | 0.0000 | 0.0000 | **0.1069** |
| m1b_v3 | 0.3575 | 0.0383 | 0.0056 | 0.0012 | 0.2819 |
| wavlm_l | 0.4920 | 0.0455 | 0.0043 | 0.0023 | 0.7141 |

## 3. Reading it

- **On the only clean population, every AntiDeepfake model beats the shipped rule, and the ordering is by size.** ITW package-0.3: 2B 0.022, 1B 0.028, MMS-1B 0.049, 300M 0.085, against SHIP 0.150 and the carrier 0.193. The 2B and 1B intervals ([0.009–0.039], [0.016–0.042]) do not touch SHIP's point. The Oct 13 ITW readout said this already (`docs/reports/2026-10-20_v2-verdicts-under-package03.md` §V2-53); the benchmark rows add the intervals, the inner fold table and the ledgered holdout read.
- **Four of five inner folds are near zero for them, which is what training data looks like.** Folds 1–3 (diffgan, openvoice, pro_diff, xtts, your_tts) sit at 0.000–0.014; the holdout (PlayHT, WaveGrad2) at 0.000–0.003. These cells say the models have seen DiffSSD; they do not say how the models would do on a generator they have not seen.
- **ElevenLabs (fold 4) is the one inner fold where they are not perfect, and there the shipped rule is still better.** 2B 0.115, 1B 0.154, 300M 0.167, MMS-1B 0.241 against SHIP 0.107 (m1b alone 0.282). The 2B closes most of the gap; none of them crosses it. ElevenLabs is in DiffSSD, so this is not a clean held-out fold either, but it is the fold most like the modern commercial voices the test set is suspected to carry, and it is a warning against reading "7× better on ITW" as "7× better everywhere".
- **WavLM Large is the worst column on every population here**, including ITW (0.291), despite ranking 16 of the 24 test misses synthetic (V2-02). Seeing the misses and ranking the population well are different things; WavLM's false alarms on ITW reals sink it. It is a candidate for V2-24's probe question, not for a column.
- **The 2B is the Phase 1 candidate on every cell.** Best ITW (0.0221), best ElevenLabs fold (0.115), and the same in-sample zeros elsewhere. Its inner run had two transient decode failures (ffmpeg rc=-5 under memory pressure, the CLAUDE.md gotcha); the assembler refused the export, `--repair` re-scored the two rows, and the assembled file carries no flags.
- **An unlabelled warning from the test set.** At each model's own inner-chosen package-0.3 threshold, the share of the 1,671 test clips called synthetic is 64% (300M), 87% (1B), 99% (MMS-1B) and 78% (2B), against a 28.7% prior and 34% for the carrier at its threshold. Rank-based fusion does not care about a threshold, but the shift says the test set does not look like the inner folds to these models either: their inner thresholds sit where the in-sample DiffSSD margins are huge, and the test clips land far inside that margin. It is the same shape as v1's handcrafted-column domain shift (B.5) and one more reason the holdout and inner cells say nothing about the test.

## 4. What this does and does not settle

- **Settled:** each candidate has a benchmark card with holdout row (chunk "done when" item 3), licence and training-data record (item 1), and the contamination argument is now measured rather than argued (item 2: the in-sample folds are at zero, the held-out population is not).
- **Not settled: whether any of these can be a column.** Three things stand between a good ITW row and adoption. (1) Rule 7: ITW is spent as a selection set, so an ITW win is a readout, not a selection. (2) The test set itself may be in-sample: DiffSSD and LJ Speech are in the AntiDeepfake training data, and the sealed key is drawn from those. A sealed read that comes back near zero would be as uninformative as the holdout cells above. (3) The licence is CC BY-NC-SA: non-commercial and share-alike, a decision owed before anything built on these weights ships.
- **The Phase 1 candidate for a sealed read**, if Nathan wants one taken: the 2B model (or the 1B if 2B's rows disappoint), read once, with the in-sample caveat written into the ledger reason. The read would tell us whether the test set behaves like ITW (held out) or like the holdout (memorised) for these models, which is itself the most useful fact about them.
- **For V2-24:** the AntiDeepfake models and Spectra see what the carrier misses; WavLM sees some of it and pays for it with false alarms. The probe question is what layer and direction carry that, in a backbone we can train on our own data without the licence and contamination problems.

## 5. Files

| What | Where |
|---|---|
| Per-population exports | `outputs/v2/backbone/<model>__{holdout,inner,itw,test}.csv` (+ `.meta.json`, `.raw.csv`) |
| Assembled Interface v1 files | `outputs/v2/benchmark/scores/<model>.csv`, `_itw_<model>.csv` |
| Benchmark cards | `outputs/v2/benchmark/{mms_300m,xlsr_1b,mms_1b,xlsr_2b}_antideepfake.json` |
| Holdout ledger | `benchmarks/holdout_ledger.jsonl` (four new reads, lane v2-53: `58ac4647960b`, `018576dc2c3f`, `b2ced457d80b`, `2837507bf7d6`) |
| Chain log | `outputs/v2/backbone/logs/chain.log`, per-run logs beside it |
| Assembler + tests | `scripts/v2_backbone_to_benchmark.py`, `tests/test_v2_backbone_to_benchmark.py` |
| Log rows | `experiments/log.csv`, rung `V2-53-rows` |
