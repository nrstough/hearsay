# V2 fusion and hard-negative research (Oct 10, 2026)

Eight researchers worked in parallel, and an adversarial verifier checked each one. This report uses only findings the verifier confirmed or corrected, with the corrections applied. `research/` means `outputs/v2/research/ (copied from the session scratchpad; gitignored)`. It is gitignored, so it is not committed.

**Terms used below.**
- **Inner OOF:** out-of-fold scores on the 16,142 NSA inner rows. Each row is scored by a model that never trained on its fold.
- **Nested LOFO:** leave one inner fold out. Everything fitted, rank references included, uses only the other four folds.
- **Costs.** There are three readings:
  - brief: 9.33·P_FA + P_miss (`src/hearsay/metrics.py:29-31`)
  - averse: the sponsor-code reading at Pspoof 0.5, P_FA + 4·P_miss
  - package-0.3: the ASVspoof5 package at Pspoof 0.3, P_FA + 1.714·P_miss
- **E:** v1's Spectra suppression. If Spectra's margin is below −3 and the base score is above 0.5, the base is halved (`models/fusion_v2/constants.json` e_rule).
- **Folds:** f0 = grad_tts + unit_speech, f1 = diffgan + openvoice, f2 = pro_diff + xtts, f3 = your_tts, f4 = ElevenLabs (the binding fold).

## Bottom line

- **Build V2-60 on option A** (m1b_v3, handcrafted_v5, m5_xlsr_ft), with Spectra only as the frozen E post-step. The option-B columns are blind on f4: WavLM 0.991 and XLS-R L20 0.985 on NSA-only inner rows. WavLM is also the worst column under package-0.3 (0.389). Sources: `research/repo-coverage`, `research/other-teams/cost_readings.py`.
- **The reported inner 0.1351 is a convention artefact.** It ranks each column against all inner rows, including the fold being scored. With train-fold references the shipped rule ("SHIP") scores 0.1558 brief / 0.3390 averse. That gap of about 0.02 is as large as the differences between candidates (`research/inner-screening/cond_shift.json` rank_reference_check, independently recomputed).
- **No learned fusion beats the fixed blends on inner data.** The best learned candidate scores 0.1625–0.1634 against SHIP's 0.1558. The bootstrap probability of beating SHIP is at most 0.42 on brief and at most 0.28 on averse. Learned fits do help f4 (0.035–0.048 vs 0.109) but lose pooled. V2-60's done-when ("beat SHIP, equal-weight and a stacker under both costs") is probably out of reach until V2-31/V2-33 add columns or data.
- **A condition gate cannot learn anything from inner rows.** Inner holds no noisy real speech, and WavLM's inner false-alarm rate is flat across noise bins. Under nested selection the gate collapses to a constant.
- **Spectra must touch no fitted parameter.** Free fits hand it 45–88% of the weight. Its training data is still undisclosed (HF API sha eb65c266 = local copy).
- **The cost convention is the largest open risk.** The three readings rank candidates differently. Several teams report that NSA scored at package-0.3, while NSA's own tech talk says false alarms cost 4×. Fix the cost before fitting anything.
- **Hard negatives:** inner holds few of them (61 L7 false alarms, on 17 speakers), and inner false-alarm rate does not predict wild false-alarm rate. Our measured noise hole is a miss problem. A CPU stage-0 loss bake-off is runnable today. In our setting, "RL" should mean a bandit over augmentation choices.
- **In-the-Wild is spent as a selection domain**, and F0's ITW 0.228 is a selection-biased control. V2-60 needs a fresh sealed population.

## Answers to the open P1 questions

### (1) Option A vs option B

**Coverage on the clean populations is identical.** All six columns join cleanly on inner 16,142, holdout 3,858, test 1,671 and ITW 3,000, with 0 duplicates and 0 non-finite values (verified).

**The cohorts separate them:**
- **Holdout perturbation cohort** (2,500 rows): only the four shipped columns exist. All 500 source clips sit on the unmounted Crucial drive (`outputs/channel/m3_perturb.meta.json` m1_truncated_to_layer 7).
- **MLAAD** (572 fakes): all 572 are in M5's training manifest, so any fusion containing M5 is in-sample there (`outputs/manifests/m5_manifest.csv`).
- **ITW perturbation cohort:** only XLS-R embeddings exist. Even option A and F0 cannot be evaluated there until handcrafted_v5, M5 and Spectra are scored, an estimated ~35 min of CPU.

**Cost to make B fully evaluable.** These are estimates extrapolated from `outputs/v2/bakeoff/extract.log` (0.252 s/clip on MPS), not measurements:
- Mount the drive.
- Run full-depth XLS-R plus WavLM on the holdout cohort: about 22–25 min MPS.
- Run WavLM on the ITW cohort (12–13 min) and on MLAAD (2.5 min).
- Re-extract XLS-R on MLAAD with matched crops (2.6 min).
- Total: about 40 min MPS, plus the 35 min CPU job that A needs anyway.

**Why B loses on inner data.** NNLR (non-negative logistic regression) gives L20 a weight of about 0 in every fold. WavLM gets 0.10–0.15 only when f4 is held out. Adding both at equal weight worsens f4 from 0.109 to 0.311.

**Recommendation:** choose A. Allow B later only as one fixed-weight candidate with an f4 guard.

### (2) Can a condition gate learn from inner rows? No.

**No support for the failure in inner rows.**
- WavLM false-alarms on 0.39% of inner reals (32 of 8,142), flat at 0.25–0.49% across energy_db_p10 quintiles. On ITW the same rate goes 3.45% → 4.95% → 12.7% across noise tertiles (`research/condition-features/c_out.json` C3; `outputs/v2/layer_diversity/noise.json`).
- With the exact ITW noise-floor definition, 0 of 900 inner ASVspoof 2019 reals exceed −29.5 dB (p99 −38.8). This is a proxy: the LJ, LibriSpeech and DiffSSD inner audio cannot be measured while the drive is unmounted.

**The gate collapses.** Nested λ picks the strongest penalty in every fold, leaving gate SD ≤ 0.003. The gate's mean weights are the same on inner, ITW and test (about [0.58, 0.17, 0.25]).

**The candidate condition features are shortcuts, not channel descriptors.**
- Their per-fold AUCs flip sign (energy_db_p10: 0.756 on f1 vs 0.265 on f3).
- hc_crop_s is NaN on every test row, by design.
- Whole-file gate features separate inner from test at AUC 0.97–0.99. Every test file is FFmpeg-written and no inner file is (`container_results.csv`).
- The p10–p95 noise floor is confounded by silence trimming. Trimming clean VCTK alone moves it from about −59 to −46 dBFS.

**Minimum fix:**
- Noisy copies of inner rows of both classes, rescored by every column under the same folds (needs the drive plus extraction).
- A wild real corpus in the inner folds, with condition-matched fakes.
- A validated, trim-invariant SNR estimator (WADA-SNR, or Brouhaha at arXiv 2210.13248).
- A pilot showing that synthetic noise reproduces WavLM's wild false alarms. This is unmeasured: no WavLM score exists on any noise cohort.

The ASVspoof 2019 bona fide rows trained m1b, L20, WavLM and M5, so a pilot on them needs per-fold refits.

### (3) How to treat Spectra

**Inner data cannot price it.**
- Pooled inner 0.0045, at most 0.017 on any fold, d′ 12.63 against 2.26–2.71 for the other columns.
- Within-class Spearman with M1b is only 0.058 (real) and 0.032 (spoof), so a learner sees it as independent, near-perfect evidence.
- E fires on 327 of 8,142 inner reals and 0 of 8,000 spoofs.
- Every threshold from −6 to −1.5 vetoes zero spoofs, so inner tuning always picks the loosest value (thr 0 in 5 of 5 nested folds).

**Free fits collapse onto it.** It gets 54–75% of the weight in LR, 74–85% in a full-inner NNLR, and 0.88 of the MoE gate. The LR coefficient is 13.2 against at most 2.7 for the others, with m5's coefficient going negative.

**One-sided learned terms still let it dominate.** Even a lower-only hinge reorders the real side (Spearman 0.95; that hinge fit was not re-verified).

**Rule:** keep E frozen (thr −3, ×0.5, AND base > 0.5) outside any fitted layer, and report every readout with and without E. v1's record shows −1.5 also improved holdout and ITW (`docs/reports/2026-09-26_post-draft-negatives.md:49`). Those sets may be in-sample for Spectra, so freezing at −3 rests on the untrusted-evaluation argument, not on observed harm.

**Loosening gate:** compute false-veto and correct-veto rates over all files of each class that the veto moves across the threshold. To first order, break-even is 9.33 (brief) or 0.25 (averse). E goes quiet under noise on its own: the ambiguous-band share is 3.0% clean vs 30.2% at 20 dB. It reaches about 53% of ITW reals.

**Membership tests:**
- **T1:** the DiffSSD official split file (8.6 MB, ask before downloading) gives an exact permutation test on the existing scores.
- **T2:** a positive control on ASVspoof 2019. It needs Spectra inference (torch), so it needs a go-ahead.
- **T3:** a temporal holdout. Per-model MLAAD release dates are unverified.

### (4) What inner-OOF screening says

Nested LOFO, 16,142 rows, learned outputs ranked against the training folds (`research/inner-screening/screen_results.json`, `extra.json`, `analysis.json`). The verifier recomputed the SHIP, EQ3, LR and equal-z rows. The bootstrap P(better) values were not re-run.

| Candidate | Pooled brief | Pooled averse | f0 | f4 | Macro | P(better) brief / averse |
|---|---|---|---|---|---|---|
| SHIP, all-inner ref (reported) | 0.1351 | 0.2997 | 0.4365 | 0.0647 | 0.1011 | (transductive) |
| **SHIP+E, nested (control)** | **0.1558** | **0.3390** | 0.4798 | 0.1094 | 0.1191 | — |
| SHIP, no E | 0.2511 | 0.3814 | 0.5119 | 0.2299 | 0.1533 | — |
| EQ3+E (equal rank mean) | 0.1541 | 0.4593 | 0.4928 | 0.0790 | 0.1158 | 0.67 / 0.23 |
| W4+E (0.4/0.2/0.4) | 0.1561 | 0.4006 | 0.4858 | 0.0852 | 0.1148 | 0.60 / 0.22 |
| Equal z-logit weights +E | 0.1633 | 0.4138 | 0.4755 | 0.0302 | n/a | n/a |
| NNLR z-logits, nested shrink +E | 0.1625 | 0.3736 | 0.4625 | 0.0352 | 0.1005 | 0.34 / 0.27 |
| NNLR ranks, nested shrink +E | 0.1629 | 0.3599 | 0.4505 | 0.0477 | 0.1006 | 0.42 / 0.28 |
| Gated, 2 conditions, nested λ +E | 0.1633 | 0.3649 | 0.4610 | 0.0497 | 0.1027 | 0.36 / 0.25 |
| Unconstrained LR, 3 columns +E | 0.1699 | 0.3714 | 0.4625 | 0.1376 | 0.1211 | 0.06 / 0.003 |
| LR + 7 conditions + interactions +E | 0.2036 | 0.5413 | 0.4955 | 0.3213 | 0.1647 | 0.035 / 0.003 |
| Option B, equal weight 5 columns +E | 0.1959 | 0.4520 | 0.5010 | 0.3112 | 0.1627 | 0.28 / 0.07 |
| Spectra as a free column (contaminated) | 0.0775 | 0.0518 | 0.0077 | 0.0130 | 0.0041 | not selectable |

Single columns on NSA-only inner rows, brief cost per fold, plus package-0.3 (`research/other-teams/cost_readings.py`):

| Column | Pooled | f0 | f1 | f2 | f3 | f4 | Package-0.3 pooled / f4 |
|---|---|---|---|---|---|---|---|
| m1b_v3 | 0.3013 | 0.543 | 0.110 | 0.004 | 0.001 | 0.542 | 0.228 / 0.282 |
| handcrafted_v5 | 0.4687 | 0.616 | 0.204 | 0.482 | 0.129 | 0.575 | 0.349 / 0.238 |
| m5_xlsr_ft | 0.3016 | 0.520 | 0.263 | 0.026 | 0.000 | 0.308 | 0.293 / 0.149 |
| spectra_aasist | 0.0045 | 0.000 | 0.000 | 0.001 | 0.000 | 0.017 | 0.003 / 0.003 |
| wavlm_l | 0.3108 | 0.541 | 0.085 | 0.002 | 0.002 | 0.991 | 0.389 / 0.714 |
| xlsr_L20 | 0.3292 | 0.612 | 0.101 | 0.009 | 0.020 | 0.985 | not computed |

**How to read these tables:**
- **Pooled inner is coarse.** At SHIP's optimum, grad_tts misses make up about 75% of the cost (931 of 1,000 missed with E, 997 without).
- **The costs disagree on direction.** Under package-0.3 with no fitting, shipped weights 0.2246 beat equal-rank 0.2348 and zmean 0.2439. On f4 the order flips (zmean 0.0728 vs shipped 0.1265).
- **The low-capacity families are statistical ties.** Only high-capacity learners (MLP, MoE, unconstrained LR, condition interactions) are clearly worse.

## Recommended V2-60 candidate list to pre-declare

1. **C0, control: SHIP (A3 w0.2 + E)**, read with nested references. Risk: none; it is the bar to beat.
2. **C1, fixed equal weights over A, plus E.** No fitting. Pick one score domain (rank or z) in advance; on inner data they are ties. Rank matches v1 and the explanation JSON; z is better on f4 (0.030 vs 0.079). Risk: v1 showed equal weighting doubled ITW misses (28% → 59%), because handcrafted collapses on wild audio (`docs/consults/2026-09-26_fusion-strategy_RESPONSE.md:13`).
3. **C2, one learned candidate:**
   - non-negative LR on train-fold z-logits
   - cost-weighted loss at the pre-declared cost's effective prior
   - shrunk toward a pre-declared target, with α chosen by inner-inner LOFO from {0, .25, .5, .75}
   - then E, then Platt calibration only

   Its free parameters total about 1–2 scalars. Inner data has about 10 independent spoof generators (ICC 0.78–0.84), which supports roughly one free parameter. Risk: high chance it does not beat C0. Its value is in testing that.
4. **Optional C3, option B at a fixed small L20 weight**, with a hard guard (within-fold f4 brief ≤ SHIP + 0.03). Better: defer it.

**Constraints that apply to every candidate:**
- Spectra stays out of all weights, gates and thresholds.
- Monotone, non-negative weights only. In classification this buys interpretability, not accuracy.
- Each candidate reports its parameter count and its in-sample gap.
- A dominance check reports each column's weight share and Spearman with the fused score on real rows.
- New exports write NSA paths in absolute form and assert full coverage after joining. `fuse_sweep.norm()` follows the data symlinks, so a mixed form silently drops rows (`scripts/fuse_sweep.py:31-33`).

**Excluded:** gated mixtures and MoE, MLPs, lattice and monotone networks, unconstrained LR, the 28-column stacker, condition interactions, and anything fitted with Spectra. The evidence is in the tables above and in `research/fusion-literature/per_generator.json`, where MLP overfit is +0.034 to +0.082 and f4 runs 0.51–0.79.

**Protocol to pre-declare:**
- Train-fold rank references.
- Every readout with and without E.
- Pooled brief plus f0, f4 and macro (the average of within-fold minDCF).
- A cluster bootstrap over speakers and generators.
- Differences under 0.02 brief or 0.05 averse count as ties.
- All three costs, with no regression allowed under any of them.
- Where E's "base > 0.5" is measured for learned outputs.
- At most 4 candidates.
- Final selection on the V2-01 benchmark plus a fresh sealed set. F0's α, E and M5 seat were all chosen on ITW (`docs/reports/2026-09-26_fusion-sweep-predeclared.md:26-28, 47-49, 100`).

**Open decision: handcrafted_v5.** Shrinking toward the shipped 0.2 conflicts with a cap that allows zero. On test, hc_v5's Spearman with the deep columns falls to 0.10–0.25, against 0.68–0.76 on inner. Other teams' feature-model fusions hurt or did not help.

## Hard-negative mining and RL (V2-33)

**What it means.**
- **Hard negatives:** real clips the detector flags as synthetic.
- **Hard positives:** fakes it misses.
- **Mining:** finding these and up-weighting them or adding more like them.
- **"RL"** in a hackathon setting most plausibly means a score → mine → retrain loop, or a search over augmentation and codec chains that maximizes detector error. For us that is a bandit (EXP3 or Thompson sampling) over augmentation "arms". No audio-spoofing paper doing RL augmentation search was found (unverified absence).
- **CRDA**, the nearest work, is for faces. Its adversarial reward is miss-side: it rewards synthesized fakes that fool the detector. A false-alarm reward on augmented reals would be our own design.

**Evidence** (`research/hard-negative-rl/`):
- Inner holds few hard negatives. L7 makes 61 false alarms out of 13,270 reals, on 17 of 103 speakers, and the top 10 speakers give 84% of them. F0 makes 20 false alarms, on 8 speakers.
- Inner false-alarm rate does not predict wild false-alarm rate: Spearman 0.30 (p = 0.06) across 40 probes. XLS-R and WavLM have the same inner rate (0.41% vs 0.44%) but very different ITW rates (1.2% vs 8.2%).
- Inner reals separate from ITW at AUC 0.9999, and the hard inner reals are no more ITW-like (p = 0.81).
- The 20 dB noise hole is a miss problem: the shipped holdout cohort goes 0.012 → 0.269 (`docs/reports/2026-09-26_channel-robustness.md:90-97`). It was measured with white noise, which may not resemble the test channel.

**Staged plan:**

| Stage | Needs | What |
|---|---|---|
| 0 (now) | CPU, saved embeddings, about 30 min | A pre-declared bake-off on L6, L7 and L8. Pilot results (23,790-row basis): base 0.2649 / f4 0.4643; focal γ=2 0.2502 / 0.4241 (false alarms 61 → 75); OHEM 0.2505 / 0.4228; nested mine-and-retrain 0.2590 / 0.4101 (f0 worse). Selection by per-fold mean, with f4 not worse. Score under all three costs. Here "f4" includes ASVspoof 2019 A04 and 1,015 ASVspoof 2019 reals. |
| 0.5 | MPS extraction, needs your go-ahead | A worst-op sweep on local ASVspoof 2019 train/dev audio: noise, band-limit, reverb, codecs (local ffmpeg has AMR-NB, Opus, MP3, AAC and G.726), plus speed (an NSA-stated test condition). Needs per-fold refits, balanced augmentation of both classes, and a shortcut AUC gate. LA eval is barred. |
| 1 | Drive mounted | Worst-of-K augmentation, mine and retrain, EXP3 updates, severity caps (SNR ≥ 5 dB). Copy-synthesis fakes need a permissively licensed vocoder; `weights/` has none. |
| 2 | GPU, ask before renting | An M5-style head with OHEM or focal loss plus the bandit; a QAMO-style head with clean and degraded real-speech centroids. About $1–2 per configuration. |

**Expected failure modes:**
- Inner hard reals do not transfer to wild false alarms.
- The model memorizes the few speakers the false alarms concentrate on.
- OHEM fits 2019-era artifacts.
- Mining reals cannot fix the noise misses.
- The model learns the shortcut "degraded audio means real".
- The search drifts to unrealistic transforms.
- The clean score regresses.
- decode_error zero-rows get up-weighted as "hard".
- Selecting on inner data alone repeats the L7+WavLM trap.

**Literature prior is weak:**
- Losses that emphasize spoofs hurt on LA19: CCE 1.39% → focal 1.67% (arXiv 2406.17246).
- In our pilot, OC-softmax was worse on f4 (0.5065), and a Gaussian model of real speech scored 0.908.
- Do not self-train on test pseudo-labels.

## Data to get for v2 (V2-30/31), ranked

1. **MUSAN** (CC BY 4.0, 11 GB, openslr.org/17) and **RIRS_NOISES** (Apache-2.0, 1.3 GB, openslr.org/28). Both fit on today's 29 GiB free. Telephony comes from local ffmpeg, so it needs no download.
2. **Common Voice Spontaneous EN v3.0** (CC0, 18.38 h, 596 speakers, 459 MB; only 7.09 h validated) as the wild-real pilot. Scripted EN (about 87 GB, size not re-checked) waits for the drive.
3. **The rest of MLAAD English** (CC BY-NC 4.0, already accepted). 143,000 clips are listed and 6,181 are local; the full set is estimated at ~49 GB. Every local clip is in M5's manifest, so a held-out MLAAD population needs new pulls.
4. **ASVspoof 5** (ODC-By, bona fide CC BY 4.0, 142.3 GB, Zenodo 14498691). Its codec conditions apply to both classes. Join MLS and LibriSpeech reader IDs before folding.
5. **VoxCeleb1 dev** (CC BY 4.0). The VGG pages say the audio is "no longer available from this website", so it goes through a request form. Drop the 10 ITW identities (id10012, id10463, id10524, id10710, id11193, id02005, id03087, id04538, id06484, id08714). VoxPopuli EN (CC0, 543 h, speaker IDs) is the open alternative.
6. **SpoofCeleb** (CC BY 4.0, 134 GB). Gated; Gmail addresses are refused. Use its fakes only; its bona fide audio is speech-enhanced.
7. **Deepfake-Eval-2024**, evaluation only: its terms forbid training. A candidate sealed population.

**Avoid:**
- AUDETER's Celebrity subset, which is In-the-Wild itself.
- CodecFake+ CoRS, which labels codec-resynthesized real speech as spoof.
- Emilia and SpoofCeleb bona fide (processed audio).
- People's Speech and YODAS, which have no speaker IDs.
- DNS clean speech, which contains VoxCeleb2.

**Rules for any new real data:** augment the spoof side to match it (same codec, noise and room responses), and pre-declare a shortcut check (codec and noise features alone must give AUC < 0.6). Before any web-sourced real corpus enters a fold, drop clips close to ITW speakers with an ECAPA filter.

## What other HEARSAY teams did

All sources are public; scores are self-reported.

- **1st, SideQuests** (0.0317 / EER 1.44%; github.com/kevinharvey2025/HactGT13AudioAuthentication):
  - three fine-tuned AntiDeepfake backbones (XLS-R-2B, XLS-R-1B, MMS-1B), equal-weight z-mean plus Platt calibration, a test-matched input view, and 11,900 copy-synthesis fakes
  - their own table shows the untuned 2B backbone already scored ITW clean 0.038
  - copy-synthesis mostly repaired fine-tuning damage (0.060 → 0.040)
  - the backbone's post-training data includes LA19, likely the eval partition (arXiv 2506.21090 §IV-A), plus DiffSSD and MLAAD
- **2nd, Athena / danmano411** (interim 0.0584):
  - fine-tuned XLS-R-300M with 11 corpora and 5,204 copy-synthesis fakes
  - LR fusion with group cross-fitting over XLS-R and LightGBM
  - their final E7 score is unpublished
- **Feature-model fusion did not help anyone who reported it.** Team Gemini's flipped 134 confidently-fake test clips to real; ProofVoice's did not beat the neural detector out-of-fold.
- **dh squad's condition-aware fusion** (0.178 → 0.156) worked only because they fitted it on noise-matched data.
- **Our position:** our 0.0733 earns 57.42 of 60 points against 0.0317.
- **"RL hard negative mining":** no public recipe. Athena's attacker-defender co-evolution is for keystroke audio, not deepfakes.
- **The test set looks VCTK-like.** Two teams say so; the transcript claim is unverified. We verified that 98.0% of test lengths fall on a 512-sample grid at 22.05 kHz. Do not hash-match test clips against our local LA eval copy without your decision.

## Claims that did not survive verification

- **repo-coverage:** "a learned fusion would upweight B and fail f4". Wrong: NNLR gives L20 about 0 weight, and the f4 regression comes from fixed equal weights.
- **inner-screening:** "93% of grad_tts undetected without Spectra". The 93.1% is with E; without E it is 997 of 1,000.
- **spectra-untrusted:**
  - The veto break-even used conditional rates. It holds only for rates over all files of each class that cross the threshold, and only to first order.
  - "Stackers lose by about 0.07" is about 0.03–0.04 under a common convention.
  - "Shipped weights chosen on inner" is wrong; they were chosen on ITW.
  - Only one unvetoed real exceeds +6.917, not two.
- **fusion-literature:**
  - "zmean won" is within noise; every family hits the grad_tts floor.
  - The real-side n_eff ≈ 6 is meaningless (only 2 corpora); speaker-level n_eff is about 180–300.
  - In classification, non-negativity buys interpretability, not accuracy.
  - The fit prior π_eff 0.097 assumes the brief cost.
- **hard-negative-rl:**
  - "One ASVspoof 2019 attack per fold" is wrong: f0 = A03 + A05, f1 = A06, f2 = A02, f3 = A01, f4 = A04.
  - The CRDA reward was misread.
  - Duroselle's 2.53% is AASIST; the ResNet got 1.56%.
  - The ShiftySpeech quote was not found in the paper.
- **other-teams:**
  - "The test is 17 dB noisier than VCTK" is not established. Trimming alone explains most of the gap, and the test's quiet frames are not broadband noise.
  - "Data design is the bigger lever" is overstated; the backbone is the main driver.
- **condition-features:**
  - The inner-vs-holdout control AUC of about 0.5 holds only with grouped CV (0.615 / 0.795 without grouping).
  - The HNR "test is near clean" claim is unvalidated.
- **wild-data-sources:** SpoofCeleb's "2.69M" was not seen; the paper says over 2.5M.
- **Not re-verified:**
  - the WavLM re-derivation to 1.8e-15, kNN novelty, white-noise twins, bandwidth coverage, and raw-vs-rank pooling deltas
  - the hinge-variant fits and the bootstrap P(better) values
  - all extraction-time projections
  - EchoFake's license, and MLAAD release dates

## Completeness check: still open before P1 freezes

- **Cost convention.** Ask NSA which error takes the 4×. Add package-0.3 as a new metrics function, leaving logged numbers intact.
- **No sealed population yet.** Options: team recordings made after 2026-06-25, MLAAD models released after 2026-04-18, or Deepfake-Eval-2024.
- **ASVspoof 2019 policy.** Decide whether train/dev may be used for training and mining. Bar LA eval everywhere. The shipped columns have no ASVspoof 2019 OOF scores; producing them means scoring handcrafted, M5 (GPU) and Spectra.
- **Second-order stacking leakage.** Base models trained on the held-out fold; the size of this leakage is unmeasured.
- **Condition measure.** No trim-invariant measure is defined, and whether synthetic noise reproduces WavLM's wild false alarms is untested.
- **Calibration** (actDCF, V2-61) was not studied.
- **M5's inner OOF** is mildly optimistic: recipe selection was biased on folds 0 and 4 (6,267 rows), so it is unclear how much weight it deserves.
- **ITW perturbation cohort** needs handcrafted_v5, M5 and Spectra scored.
- **Actions needing your approval:**
  - downloading the DiffSSD split file
  - emailing lab260 about Spectra's training data
  - mounting the drive and running MPS extraction
  - any GPU rental
- **Licences.** Spectra's licence is contradictory (Apache-2.0 header, MIT body). AASIST3 is CC BY-NC-ND. MLAAD is non-commercial. These matter if v2 becomes a product.
- **Process.** One researcher killed processes by pattern (`pkill` on "LokyProcess") on the shared Mac, which could have hit other agents' workers. Future agents should stop processes by explicit PID only.