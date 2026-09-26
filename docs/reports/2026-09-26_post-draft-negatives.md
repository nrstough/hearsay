# What did not work after the draft review (Sat Sep 26, 2026, evening)

NSA returned minDCF 0.0733, EER 3.53% on our draft file. We then asked what could still move that number. Two outside opinions and our own checks gave mostly negative answers. This report writes them down, each with its number, its mechanism and its source, so the README can cite them.

**Sources.** The consult prompt and its addenda: `docs/consults/2026-09-26_post-draft-review_CONSULTATION.md`. The VeriLM memo and the Round 2 comparison: `…_RESPONSE.md` (verbatim: `…_RESPONSE_verbatim.md`). The Fable agent's memo: `…_RESPONSE_fable.md`. The Fable agent computed on our exported scores with the sweep's own loaders. The consult chat then re-checked two of its claims from the exports: the M5-weight rows and the second-tier row (Round 2 table in `…_RESPONSE.md`). It did **not** re-run the layer probes, the stackers or the M3 promotion; those numbers are the Fable agent's alone, and each one is marked below.

**Costs.** "Brief" is our selection cost, 9.33·P_FA + P_miss (π_synth = 0.3, C_FA = 4). "Averse" is the sponsor script's cost, P_FA + 4·P_miss (item 5 explains why it differs). The shipped rule is A3 w0.2 + E: inner 0.1351 / 0.2997, holdout 0.0065 / 0.0087, In-the-Wild 0.228 / 0.2385 (brief / averse).

---

## 1. More heads on the same backbone are dead on arrival

**What we tried.** Probes on other XLS-R layers, trained with M1b's recipe on the embeddings we already had. *(Fable memo, option 10; not re-run by the consult chat.)*

**The numbers.** The layer-7 baseline reproduces M1b exactly: inner 0.2649, holdout 0.0717, In-the-Wild 0.342. Averaging layers 5–9 improves the holdout to 0.052. But In-the-Wild gets worse, 0.368, and its false-alarm rate doubles, from 0.8% to 1.55%. Concatenating layers 6–8, layers 5/7/9 and layers 5–9 gives In-the-Wild 0.353, 0.430 and 0.399.

**Why.** At the threshold that minimizes its In-the-Wild cost, the shipped rule misses 158 fakes there. Each detector alone, at its own inner threshold, would catch these many of them: M1b 1, the handcrafted model 2, M5 33 and Spectra-AASIST 107. At the rule's inner-OOF threshold, the one the gate reads, it misses 122, of which M5 would catch 19 and Spectra-AASIST 79 (correction in `…_RESPONSE.md`, commit 589e9f2). So M1b's backbone does not see these fakes at any layer, and a new head on that backbone has the same blind spot. Only a different backbone can see them. That is why the one head experiment we ran tonight is a WavLM Large probe.

**The different backbone, measured.** WavLM Large, probed exactly like M1b, is level with M1b on the holdout (0.0717 both). It is not a clone: its test-set Spearman with M1b is 0.79. And it does reach the miss tail. Alone, it flags 41 of the shipped rule's 122 In-the-Wild misses at the inner-OOF threshold, against M1b's 1 and M5's 19. But it pays with false alarms on wild real speech (7.05% against M1b's 0.8%). As a fourth column at weight 0.2 it failed the pre-declared gate:
- inner CV got worse by 0.021 (brief) and 0.033 (averse)
- its In-the-Wild gain (+0.023 / +0.009) was short of the bar

In M1b's seat it quadruples In-the-Wild false alarms. Not shipped (`docs/reports/2026-09-26_post-draft-heads.md`; `docs/reports/2026-09-26_post-draft-sweep.md`, commit 57b9c90).

## 2. Re-weighting the same three scores has a measured ceiling

**What we tried.** Every way of re-slicing the three scores we trust: M1b, handcrafted and M5.

**The numbers.**
- **Cheating on purpose.** We chose weights on In-the-Wild itself, (0.3, 0.2, 0.5). That reaches In-the-Wild 0.186, a gain of 0.04. It costs the holdout two false alarms. No honest rule can beat a rule tuned on the test set it is scored on, so 0.04 is the ceiling. *(Fable memo, §2.)*
- **A stacker fit on In-the-Wild.** A logistic stacker on the three logits, fit on In-the-Wild, scores 0.257 there. That is worse than the shipped rank blend. *(Fable memo, §2.)*
- **An admissible stacker.** A non-negative stacker fit on inner rows without LJ Speech gives weights (0.66, −0.04, 0.37) for M1b, handcrafted and M5. It scores In-the-Wild 0.222, under our 0.01 bar, and holdout 0.029, which is worse. *(Fable memo, option 5.)*
- **A second M3 suppression tier.** Quartering files with Spectra margin below −6, on top of halving below −3, gives identical numbers on every split: inner 0.1351 / 0.2997, holdout 0.0065 / 0.0087, In-the-Wild 0.228 / 0.2385 *(verified, Round 2 table)*. Halving already puts every touched file below every untouched file above 0.5, so quartering only reorders files inside the suppressed set.
- **Calibration.** Platt or isotonic maps keep the order of scores. minDCF depends only on the order, so calibration cannot move it at all.
- **A length-gated mixture.** More M5 weight on clips of 4 s or less scores In-the-Wild 0.207. It rested on a wrong premise: M5 is the *weaker* detector on short clips (holdout, ≤ 4 s: M5 0.344, M1b 0.066). Its gain comes from M5's lower wild-domain false-alarm rate, not from clip length, and raising M5's weight everywhere does the same (item 7). *(Fable memo, §1 and option 4.)*

**Why.** The three scores agree on most of the test set. Our analysis of the unlabeled test scores found 117 files called synthetic by all four detectors and 976 called real by all four; the uncertain files are about 187 with a base score between 0.45 and 0.65 *(Fable memo, §2)*. Re-weighting only reorders that middle band, and the ceiling above says how much that can buy on In-the-Wild. The test set is about three times easier than In-the-Wild, so maybe 0.01 of that 0.04 would carry over, decided by a handful of files.

## 3. The handcrafted score is the readable column, not a vote

**The numbers.** The admissible stacker in item 2 gives the handcrafted score a coefficient of −0.04. Its rank correlation (Spearman) with M1b is 0.79 on the holdout but only 0.21 on the test set *(Fable memo, §1)*.

**Why.** The README already describes the test-domain offset: even band-matched, the test recordings have a darker, codec-like spectral envelope, and the 234 handcrafted columns tell train from test at AUC 0.99. On the test set the handcrafted model is reading that shift more than it is reading the class. That is why its correlation with M1b collapses there. Its 20% seat in the shipped rule stays as it is. Its value is that it explains a file in physical terms. It is not an independent vote.

## 4. The excluded detector could shrink the miss tail, and stays excluded

**The numbers** *(Fable memo, options 3 and 9; not re-run by the consult chat)*.
- **Promotion.** Raising files with Spectra-AASIST margin above 6.9 and base score below 0.5 recovers 76 of the 158 In-the-Wild misses (counted at the In-the-Wild argmin; see item 1) with no added false alarms (In-the-Wild 0.152 under both costs). On the holdout it promotes two real files, [2 FA, 11 misses] against the shipped [0, 13].
- **Looser suppression.** Moving the suppression threshold from −3 to −1.5 improves every proxy: inner 0.123, holdout 0.003 [0, 6], In-the-Wild 0.188, averse 0.233. The cost is that more MLAAD fakes fall below the threshold: 2.10% against 1.75% at −3.

**Why we skip both.** Spectra-AASIST's authors report results on In-the-Wild and name no training set. The evidence for both changes is In-the-Wild, a set the model's authors may have tuned on. We set the suppression-only rule before any of these numbers existed. Changing it now because the numbers look good is the kind of post hoc choice the rule exists to prevent. So this is the measured ceiling of what the excluded detector could do, not a change.

## 5. The cost convention can't be read from the returned pair

**The code and the spoken rule disagree.** The sponsor's scorer (`data/nsa/HackGTMinDCF/asvspoof5/evaluation-package/calculate_metrics.py`) was edited from ASVspoof5's defaults (Pspoof 0.05, Cfa 10) to Pspoof 0.5, Cmiss 1, Cfa 4. In that code, Cfa multiplies the rate at which *spoofs are accepted as bona fide*, i.e. a fake called real. At the kickoff NSA said a *false alarm* (a real voice called synthetic) costs 4× a miss. NSA must have adapted the scorer outside the package, and how they did it decides where the 4× lands: negating our scores puts it on misses, swapping the labels puts it on false alarms.

**The EER limits both readings.** At any single threshold, the larger of P_FA and P_miss is at least the EER, 3.53% (VeriLM memo, "What the number means" item 3). With about 1,170 real and 500 synthetic files:
- **Brief cost:** P_FA must be at most 0.41% and P_miss 3.5–7.3%. That means 0–4 false alarms plus 18–37 misses, for example 3 false alarms and 25 misses.
- **Averse cost:** it inverts. P_FA is 3.5–7.3% and P_miss at most 0.95%. That means 42–86 false alarms plus 0–4 misses, for example 67 false alarms and 2 misses.

The two profiles cannot both hold. The pair fits either one, so it can't tell us which convention NSA used. Two corrections followed:
- The decomposition we wrote in the consult prompt (1.3% false alarms plus 1.5% misses under the averse cost) was impossible, because both rates sit below the EER.
- The Fable memo's averse example of "5 misses + 40 false alarms" is also just outside the feasible range: 40 of about 1,170 is 3.4%, and 5 of about 500 is 1.0%.

**What we do about it.** We work on the reading that the error is a miss tail, because NSA said "false alarm" out loud. We hedge by requiring every candidate to improve under **both** costs.

## 6. How far our proxies were from the one measurement

The shipped rule, recomputed from the exported scores (CONSULTATION, "Added to 'anything else' (Sat ~16:05)"):

| Split | Rows (real / synthetic) | minDCF (brief) | EER | P_FA / P_miss at the minDCF threshold |
|---|---|---|---|---|
| Outer holdout | 1,858 / 2,000 | 0.0065 | 0.21% | 0.00% / 0.65% (0 FA, 13 misses) |
| Inner OOF | 8,142 / 8,000 | 0.135 | 6.83% | 0.25% / 11.2% |
| In-the-Wild | 2,000 / 1,000 | 0.228 | 5.80% | 0.75% / 15.8% |
| **NSA test (returned)** | ~1,170 / ~500 | **0.0733** | **3.53%** | not given |

**What the gaps say.**
- The holdout was optimistic by 11× on minDCF and 17× on EER.
- In-the-Wild was pessimistic by 3× on minDCF but only 1.6× on EER.
- Inner OOF was pessimistic by 2× on both.

On a log scale between the holdout and In-the-Wild, the test sits at 0.68 for minDCF and 0.85 for EER, further toward the wild end than our λ̂ = 0.51 suggested. The test's minDCF held up better than its EER. That fits a system with few false alarms and a tail of missed fakes, but it is an inference, not a measurement.

**Lessons.**
- Our holdout (two generators, clean read speech) cannot measure a change: one false alarm on it is worth 0.005, and its noise is ±0.07–0.10.
- A "0.0065 → 0.0060" is not evidence.
- On the brief's cost, one false alarm on the test set is worth about 0.008 and one miss about 0.002. So gains under about 0.008 are single-file effects.

## 7. A heavier M5 weight did not survive fresh evidence

**What we tried.** The pre-declared M5 sweep stopped at weight 0.2 on the expectation that more M5 would hurt. After the draft review, the Fable agent extended it. The consult chat re-checked the numbers from the exports (Round 2 table):

| Rule | Inner (brief / averse) | Holdout | In-the-Wild | Test Spearman vs shipped |
|---|---|---|---|---|
| Shipped A3 w0.2 + E | 0.1351 / 0.2997 | 0.0065 / 0.0087 | 0.228 / 0.2385 | 1.000 |
| A3 w0.3 + E | 0.1374 / 0.3139 | 0.0060 / 0.0091 | 0.2113 / 0.2310 | 0.993 |
| A3 w0.4 + E | 0.1373 / 0.3366 | 0.0065 / 0.0088 | 0.1997 / 0.2075 | 0.970 |

**What it shows.** Weight 0.4 improves In-the-Wild by 0.028 (brief) and 0.031 (averse) with the holdout unchanged, and makes inner worse by 0.002 (brief) and 0.037 (averse). The mechanism is M5's lower false-alarm rate on wild audio. It passes our standing four-condition rule. It fails the stricter gate adopted tonight: rule 3 requires inner to improve by 0.010 with no averse regression. It is post hoc, because its numbers were seen before it became a candidate.

**The bake-off.** Nathan ruled at about 17:40 that, because its proxy numbers were already seen, W4 must pass a bake-off on fresh evidence, declared before any of it was computed (`docs/reports/2026-09-26_post-draft-manifest.md`, "The W4 bake-off", commit 141b254). To be ratifiable it has to pass all three tests:
- **The standing rule.** It already passes.
- **Perturbation robustness.** The cohort is the channel lane's 500 holdout clips under five kinds of change: none, MP3 64 kbps, 20 dB white noise, ±2% speed, and a one-sample shift. In each of the 10 kind × cost cells, W4 may be no more than 0.010 worse than the shipped rule.
- **A gain that is not noise.** We resample In-the-Wild by speaker, 2,000 times with seed 0. The 5th percentile of W4's gain must be above zero under each cost separately.

If it fails any of the three, it becomes a measured negative: post hoc, and it did not survive fresh evidence.

**Result: it failed two of the three** (`scripts/fuse_sweep_v3.py`, commit 20a0d04; re-run here with identical output).
- **Standing rule:** passes.
- **Perturbation robustness:** fails. Below is the shipped rule's minDCF minus W4's, per kind of change, brief / averse. A negative number means W4 is worse.
  - no change: −0.012 / 0.000
  - MP3: −0.004 / −0.004
  - 20 dB noise: +0.051 / +0.020
  - speed: −0.004 / −0.012
  - one-sample shift: −0.016 / −0.004

  Three cells are below the −0.010 bar. W4 helps a lot under noise, where M5 holds up better than M1b, but it costs on the clean, speed and shift cells.
- **Bootstrap:** fails. Across 54 In-the-Wild speakers and 2,000 replicates, the 5th percentile of W4's gain is +0.0044 under the brief cost but −0.0023 under the averse cost. Its In-the-Wild gain under the sponsor's cost can't be told apart from zero once clips of the same speaker are treated as one unit.

**Why it matters.** A gain of 0.028 on In-the-Wild looked like the one cheap win left. It came from about 54 speakers, and it reversed on clean audio the rule had not been tuned on. Seeing the numbers first and then testing on new evidence is what caught it. W4 is not in the ratifiable set, and the shipped rule stays.

---

## 8. Noise twins did not make the handcrafted model noise-robust

**What we tried.** The shipped rule's known weak spot is additive noise: at 20 dB white noise the handcrafted AUC falls from 0.998 to 0.64 (README). We retrained it exactly like v5b, adding a 20 dB white-noise copy of 35% of the training rows in both classes.

**The numbers** (`docs/reports/2026-09-26_post-draft-hc-noise.md`), on the channel lane's 500 holdout clips:
- **Under noise:** AUC 0.64 → 0.77. The pre-declared bar was 0.90.
- **On clean clips:** minDCF 0.048 → 0.163.
- **Outer holdout:** 0.137 → 0.227, mostly on LibriSpeech real speech (0.155 → 0.298).
- **In-the-Wild AUC:** 0.768 → 0.722.

**Why.** The handcrafted columns read fine spectral and phase structure in the quiet parts of the signal, and white noise fills exactly those parts. The model does not find a noise-proof version of the cues; it learns to trust them less, and that blurs clean clips too. Noise robustness for this detector would need different features (e.g. voiced-frame-only statistics), not more rows. In the fused rule (H_noise) every split got slightly worse under both costs (e.g. In-the-Wild −0.0003 / −0.0020), and it failed the gate (`docs/reports/2026-09-26_post-draft-sweep.md`, "Run 2"). `models/hc_selected` stays on v5b.

**Outcome of the evening.** Four candidates were pre-declared and judged by a frozen gate: the second M3 tier, the M5 weight, WavLM as a fourth column and the noise twins. All four failed. The file NSA scored at 0.0733 stays final.

## Not run tonight, and why

- **Delta-only handcrafted features.** The earlier column-pruning experiments failed every time (README), and this is the same experiment in different clothes.
- **Averaging M5's fold models.** Naive averaging leaks each out-of-fold row into its own prediction (VeriLM memo). The fold checkpoints are also not on disk (Fable memo, option 11). And M5's problem is bias, not variance.
- **New feature families** (LTAS residuals, vocoder harmonics, formants, sub-band modulation). Each is a research lane. Any of them would also have to survive the train-vs-test AUC-0.99 envelope shift (item 3).
