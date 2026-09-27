# Independent second opinion: a Claude Fable 5.1 agent on the post-draft consult (Sat Sep 26, 2026, 17:00–17:15 EDT)

Requested by Nathan as a comparison to the VeriLM memo. The agent read `2026-09-26_post-draft-review_CONSULTATION.md` and the repository (exports, reports, scripts) and was told not to read the VeriLM RESPONSE files. It ran computations on the exported logits with the sweep's own loaders; the consult chat re-verified the M5-weight and second-tier claims (see the addendum in `…_RESPONSE.md`). Verbatim below.

---

## Independent opinion: after the draft review, what (if anything) moves the test number

Sat Sep 26, 17:00 EDT. Freeze Sun 05:00. Everything below is computed from the repo's exported logits (`outputs/detector_scores/*.csv`, In-the-Wild files, `outputs/channel/m3_mlaad.csv`) with the same loaders and cost functions as `scripts/fuse_sweep.py`; the shipped rule reproduces exactly (inner 0.1351, holdout 0.0065 [0 FA, 13 miss], ITW 0.228 [15 FA, 158 miss], averse 0.2385). I did not read the RESPONSE files.

## 1. Verdict on the returned number

**0.0733 / EER 3.53% is in our direction (an unflipped file scores ~1.0 under the sponsor's code), and it is consistent with the profile our own rule produces everywhere else: near-zero false alarms and a miss tail. Which cost NSA applied is undeterminable, and the number cannot support any of the changes below; from here on you are optimizing proxies only. Confidence: high on direction, moderate on profile, none on cost.**

Pushback on the premises:

- **The cost is not "the brief's cost" with certainty, and your own repo says so.** The HackGT scorer (`data/nsa/HackGTMinDCF/asvspoof5/evaluation-package/calculate_metrics.py`) was deliberately edited from ASVspoof5's `Pspoof = 0.05, Cfa = 10` to `Pspoof = 0.5, Cmiss = 1, Cfa = 4`, and in that code `Cfa` multiplies `far` = *spoof accepted as bona fide* = fake-called-real. The kickoff notes (`docs/reports/2026-09-25_sponsor-questions.md` lines 16–18) define false alarm as real-called-synthetic, ×4. The package contains no score inversion and its keys are plain `bonafide`/`spoof`, so NSA adapted it outside the package: if they negated our scores, the 4× sits on misses (`P_FA + 4·P_miss`); if they swapped the keys, it sits on false alarms as stated. The pair (0.0733, 3.53%) fits both: brief-cost e.g. 3 FA + 25 misses; averse e.g. 5 misses + 40 FA. Per-file granularity on ~1,170/500: brief FA 0.008, miss 0.002; averse FA 0.00085, miss 0.008. Your "improve under both costs" rule is the right hedge; keep it and stop calling one of them the brief's.
- **Error profile.** Under the brief cost the likeliest reading is a miss tail of 20–35 fakes with 0–4 false alarms, because that is what the rule does on every labeled set (0/13 holdout, 15/158 ITW at the brief argmin; on ITW only 1 of the 158 misses is a file the M3 step touched). The EER (59 files wrong at the equal-error point) says the middle of the ranking is soft; minDCF is scored at the low-FA end, which held up better, so misses dominate. To match the leader's 0.0584 you would need to remove ~2 FAs or ~7 misses (brief) or ~2 misses (averse).
- **The misses are not reachable from the XLS-R probe family.** Of the 158 ITW misses under the shipped rule, M1b alone would flag 1 and handcrafted 2 at their inner thresholds; M5 would flag 33; M3 would flag 107. Any "more heads on the same frozen backbone" idea inherits M1b's blind spot; only a different backbone or M3 can see the tail.
- **The holdout is not a measurement of a change.** It is 0 FA / 13 misses (12 wavegrad2, 1 playht) on 3,858 rows; one FA is worth 0.005. Anything below ~0.02 on it is noise. Its "11×" gap to the test is granularity, not a regime.
- **Handcrafted is not 20% of the decision.** The admissible non-negative stacker fit on inner non-LJ rows gives it coefficient −0.04 (M1b 0.66, M5 0.37). On the test set its Spearman with M1b is 0.21 (holdout 0.79): it is reading the envelope shift. It earns its place as the readable column, not as score.
- **The MoE premise is backwards.** M5 is the *weak* detector on short clips (holdout ≤ 4 s: M5 0.344, M1b 0.066); more M5 weight on short clips helps ITW because M5 has fewer wild-domain false alarms, not because of clip length.

## 2. Ranked options (asks 2–4)

| # | Approach | Mechanism that could move the test number | Expected test minDCF effect | Effort | Risk | Verdict |
|---|---|---|---|---|---|---|
| 1 | **A3 (0.4, 0.2, 0.4) + E** (M5 weight 0.2 → 0.4, same rule) | The pre-declared sweep stopped at w = 0.2 on an expectation that proved wrong; more M5 cuts wild-domain FAs. ITW 0.228 → 0.200, averse 0.239 → 0.208, holdout 0.0065 → 0.0065 [0, 13], inner 0.135 → 0.137; passes all four conditions; **4 test files cross 0.5**, Spearman 0.970 | −0.005 ± 0.01, sign not reliable | 1.5 h (edit the `w` grid in `fuse_sweep_m5.py`, `--write`, full-set runner parity ~40 min under load, TSV, log row, README numbers) | Post hoc (these numbers are now seen); M5's CPU-vs-A100 logit gap (5 rows > 0.02) scales with its weight | **Maybe** — the only candidate worth putting to Nathan |
| 2 | WavLM Large probe on the M1b recipe (`extract_embeddings.py --model wavlm-large --segment --crop test` on nsa_train_sample, asv19_addon, itw_stress, nsa_test; `train_probe.py --model wavlm-large --folds splits/nsa_folds_plus_asv19.csv --stress ...`; then `fuse_sweep_m5.py --refit-m1b <export>`) | The only build that changes the backbone, hence the only one that can see the miss tail M1b cannot | 0 ± 0.01 | ~1.5–2 h of shared CPU plus the refit sweep; finishes by ~19:30 | Extraction time under load; refit seat changes the ratified stream | **Maybe, as a bake-off for the README; ratify only if it passes the four conditions and crosses ≤ 4 test files** |
| 3 | M3 suppression threshold −3 → −1.5 | Loosens the one deliberately conservative parameter. Inner 0.123, holdout 0.003 [0, 6], ITW 0.188, averse 0.233 all improve; 6 test files; MLAAD spoof below −1.5 is 2.10% vs 1.75% at −3 | 0 to +0.01 | 1 h | Evidence leans on ITW, which M3's authors evaluated on; post hoc | Skip |
| 4 | Length-gated mixture (≤ 4 s → 0.4/0.2/0.4) | ITW 0.207, averse 0.208, holdout 0.006; 4 files | Same as #1 | 2 h (new pipeline branch) | Stated mechanism is wrong; dominated by #1 | Skip |
| 5 | Non-negative stacker over the three admissible logits (inner non-LJ) | Fitted weights (0.66, −0.04, 0.37); ITW 0.222 (under the 0.01 bar), holdout 0.029 (worse) | ≤ 0 | minutes | Fails the bar | Skip |
| 6 | Per-detector isotonic calibration | None: the rank-ECDF is already a monotone map; minDCF is rank-only | 0 | — | — | Skip |
| 7 | Cascade (M1b alone outside its ambiguous band) | Already lost the pre-declared sweep (ITW 0.355) | ≤ 0 | — | — | Skip |
| 8 | Second suppression tier (< −6 → ×0.25) | **Inert.** Halving already puts every touched file (base ≤ 0.5) below every untouched > 0.5 file; quartering reorders only within the suppressed set. Identical readouts on every split; 0 extra test files | 0 exactly | — | — | Skip |
| 9 | Capped Bayesian use of M3's margin / M3 promotion | The one fusion change with a large mechanism: promoting `m3 > 6.9 & base < 0.5` recovers 76 of 158 ITW misses with 0 added FAs (ITW 0.152, averse 0.152), but the holdout shows 2 reals promoted [2, 11]; touches 10 test files (2 with M5's agreement) | brief +0.01, averse up to +0.08 if the 10 are fakes; −0.008 per real among them | 1 h | Forbidden by your rule; the returned number says nothing about M3 | **Skip.** The number does not inform the rule; the rule is a documented principle worth more in the 40 non-DCF points than the expected gain |
| 10 | Multi-layer XLS-R probe (ran it: layer-7 baseline reproduces M1b 0.2649 / 0.0717 / ITW 0.342 exactly) | mean L5–9: holdout 0.052 but ITW 0.368 and ITW P_FA 0.8% → 1.55%; concat L6–8 / L5,7,9 / L5–9: ITW 0.353 / 0.430 / 0.399 | ≤ 0 | done | — | **Dead; write up** |
| 11 | M5 fold-model averaging | Fold checkpoints are not on disk (`outputs/m5_runs/*/fold*_frozen` hold only run_meta/scores/log; R2 pull of 5 × 657 MB, then ~28 min CPU per model per split); fold heads have different logit scales | 0 | 3 h+ | Wall clock | Skip |
| 12 | Attentive pooling on frozen L7 frames | Frames are not stored (embeddings are per-layer time means); M5 already is this and lost the holdout | 0 | 3 h+ | — | Skip |
| 13 | Noise-augmented probe / noise twins for handcrafted | The test set shows no additive noise (floor p2/p10 beyond the clean end); re-extraction of both classes ≈ 1 h | 0 on test | 1–2 h | — | README only |
| 14 | Delta-only handcrafted (`train_handcrafted.py --drop-columns` on existing features) | Pruning lost every time; the column carries ~0 weight | 0 | 20 min | — | README only |
| 15 | New handcrafted families; generator-family routing head | Inherit the envelope shift (train-vs-test AUC 0.99); routing has no minDCF mechanism | 0 | 2 h each | — | Skip |

**Direct verdict on the fusion layer.** No change to the fusion layer can be expected to move the test minDCF beyond noise, and the reason is structural, not a matter of finding the right stacker. The three admissible streams agree on the test set (M1b–M5 Spearman 0.78; 117 files called synthetic by all four detectors, 976 real by all four); the fused ranking's uncertainty lives in ~187 files with base in (0.45, 0.65); and the ceiling for *any* reweighting of the three admissible ranks, computed by cheating (weights chosen on ITW itself, (0.3, 0.2, 0.5)), is ITW 0.186, a gain of 0.04 that costs the holdout 2 false alarms. An in-sample logistic stacker on the three logits fit *on* ITW scores 0.257, worse than the shipped rank blend. The test set is 3× easier than ITW, so the transferable share of that 0.04 is ~0.01, with 4–13 files deciding the sign. The only stream that would resolve the ambiguous band is M3, and every use of M3 beyond suppression is the one thing you have ruled out. Stackers, isotonic maps, cascades and second tiers are either inert (#8), already lost (#5, #7), or have no minDCF mechanism (#6).

## 3. Procedures for the two "maybe"s

**#1, the M5 weight.** Add `w ∈ {0.3, 0.4}` to the grid in `scripts/fuse_sweep_m5.py` (it currently loops `(0.1, 0.2)`); run without `--write`; it prints the same four-condition decision the addendum used, CURRENT = shipped. Fold discipline is unchanged: the three ranks are against inner-OOF references, the Platt map is fit on inner OOF, nothing touches holdout, ITW or test. Acceptance, declared now and disclosed as post hoc since I have seen the numbers: replace only if all four conditions hold **and** averse improves **and** ≤ 4 test files cross 0.5 **and** the full-set runner parity to the new export is Spearman ≥ 0.999 with max |Δp| ≤ 1e-3. Then `--write`, new TSV, new log row, both polarities. Stop at 21:30 regardless of state; if any step is unfinished, the shipped file stands.

**#2, WavLM Large.** Four extractions with the v3 flags (`--segment --crop test --seed 0`), then `train_probe.py` with the M1b arguments and `--model wavlm-large`, then export in the standard `path, fold, split, score, logit` format with an `_itw_<name>.csv` on the seed-200 crops, then `fuse_sweep_m5.py --refit-m1b <name>`. Same four conditions; additionally the clean holdout must be within 0.03 of M1b's 0.072 (your addendum-2 floor). Pre-declared entry: the refit seat only (weight 0.6, or 0.4 if #1 ships), never a fourth column tonight. If it fails, it is a bake-off row in the README.

## 4. Time split

- **17:00–18:30.** Session A: #1 sweep (30 min) and its candidate TSV, nothing under `models/`. Session B: WavLM extraction and probe (#2). Session C: README negative results from this memo (multi-layer probe, admissible stacker, inert second tier, the M3-promotion ceiling and why it stays forbidden, the cost-direction ambiguity). Session D: copy the shipped TSV to `CrossExam_predictions.tsv` in its final location tonight, so that sleeping equals shipping.
- **18:30–19:30.** Nathan decides on at most one candidate. GO: `--write`, parity, TSV, log, README numbers (1–2 h of doc churn is the real cost). NO-GO: nothing changes.
- **19:30–21:30.** Documentation only. **21:30: hands off.**

Single highest-value change: A3 (0.4, 0.2, 0.4) + E. Expected ≤ +0.01, not reliably above zero.

## 5. Are you over-engineering; should you stop touching the file?

Yes, and yes. The upside of matching the leader is 0.95 of 100 points; the downside of a bad switch is at least as large; the granularity is 4–6 files whose labels you cannot know; you do not know which error the 4× lands on; and the only stream with a real mechanism is the one your principles exclude. No hedging: run #1 and #2 as bounded, documented experiments because the README grades negative results, and ship the 12:03 file unless #1 clears every gate above by 19:30 with room to spare.

## 6. Failure modes you have not named

1. **The 4× may be on misses.** Every candidate must improve under both costs; #3 and #9 change sign between them.
2. **Sleep must equal ship.** The final-named file should exist before anyone sleeps, with its sha256 in the log.
3. **M5 parity scales with M5's weight.** Re-run the full 1,671-row parity if w changes; the 5 rows already over the 0.02 mark will widen.
4. **Proxy contamination.** M3's authors evaluated on ITW; any change that "passes the bar" by moving M3's threshold is validated on evidence M3 may have seen.
5. **Do not listen to the crossing files to decide.** Choosing a rule by inspecting the 4 test files that cross is selection on the test set, which your hard line forbids.
6. **Documentation drift.** A 20:00 switch invalidates every cited number in README, docs site, disclosure and `STATUS.md`; the doc-consistency test pins only constants, not results.
7. **The holdout cannot see your change.** Do not let a "0.0065 → 0.0060" appear anywhere as evidence.

**Final call.** The returned number tells you the file is right-side-up and behaves like a low-false-alarm system with a modest miss tail; it does not tell you which files are wrong, which cost was used, or anything about the one detector that could shrink the tail. Every admissible fusion change is worth at most about a percent of the score with an unknown sign, every new head on the same backbone is dead on arrival (the layer variants I ran tonight prove it), and the one build with a mechanism, a second backbone, is a bake-off for the write-up rather than a Saturday-night switch. Run the M5-weight sweep and the WavLM bake-off for the record, decide by 19:30, and otherwise stop touching the file: the 40 points that reward technique and honest documentation are where the remaining hours pay, and the shipped file is already the best-validated thing you own.
