# Response: after the draft review — what to build in the last hours (received Sat Sep 26, 2026, ~16:50)

**Source:** VeriLM "Deep Think" memo, Frontier tier: two experts (GPT-5.6 Sol at extra-high effort, Gemini 3.1 Pro at high) with a synthesis; confidence stated as medium. Verbatim copy with figures: `2026-09-26_post-draft-review_RESPONSE_verbatim.md`. Prompt and our scope answers: `2026-09-26_post-draft-review_CONSULTATION.md`. (The panel's third seat, a Claude model, did not appear in the delivered memo.)

**Read this first.** Two things the memo did not have:
- **The field.** NSA's feedback puts one team ahead at minDCF 0.0584 / EER 2.5%. Under points = (1 − ours)/(1 − best) × 60 the gap is **0.95 of 60 points** today (2.7 if the leader reaches 0.03). Matching 0.0584 means removing 0.015 minDCF: about 2 false alarms or 7 misses under the brief's cost. The memo's "stop by 20:15" advice only strengthens with this: the largest expected gain it names (WavLM, EV −0.002) is worth about 0.13 points.
- **The clock.** The memo's schedule starts at 15:40; it arrived at 16:50. Every time in its table shifts by about 75 minutes (see the adopted plan below).

## Verdict (the memo's, one line)

The (minDCF, EER) pair cannot identify NSA's cost convention, but under the brief's cost the optimum is forced to ≤ 5 false alarms plus 18–37 misses; NSA's spoken statement of the analyst scenario, not the numbers, makes the few-false-alarm / miss-tail read the working hypothesis. One aggregate readout cannot validate any fusion change; stop all fusion redesign. Keep the shipped TSV as presumptive final and run only three or four bounded, pre-declared experiments under a gate stricter than the standing rule; if nothing clears by the decision window, freeze and sleep.

## What the number means (adopted)

1. **Direction:** the file was read with a polarity compatible with our ranking; the 0.00–0.20 branch of the decoding table is correctly triggered; no flip, no fallback. "Direction is ours" is slightly too strong (NSA may have inverted into the ASVspoof convention themselves), and operationally moot.
2. **Regime:** not "holdout-like". At 11× the holdout's minDCF and 17× its EER, the test set is its own moderately shifted domain whose low-false-alarm tail held up far better than In-the-Wild's.
3. **Error profile, forced by the EER** (Gemini's feasibility argument, kept; its "definitive" claim rejected): at any single threshold, max(P_FA, P_miss) ≥ EER. So under the brief's cost the minimum must have P_FA ≤ 0.41% (0–5 real files) and P_miss 3.5–7.3% (18–37 fakes); under the sponsor's cost it inverts to P_FA 3.5–7.3% (42–86 real files) and P_miss ≤ 0.95%. The two profiles are mutually exclusive; which one holds depends on the convention, which the numbers cannot settle. Working read: miss tail, at roughly 80–85% confidence, because NSA said "false alarm 4× a miss" in person.
4. **Our own decomposition in the prompt was wrong:** the sponsor-cost example (1.3% FA, 1.5% miss) is impossible because both are below the EER.
5. **Evidentiary weight:** improvements below about 0.008 are one expensive-file quantum; fixed-threshold standard errors run 0.008–0.024 before correlation and oracle-threshold effects; effective sample size is well below 1,671 (shared speakers, generators, one recording pipeline). **We are optimizing proxies only.** The only defensible changes inject new discriminative information (a new representation or new feature physics), never a re-slicing of the same three score columns.
6. **Three corrections to our premises** (GPT-5.6): M5 is not a short-clip expert (holdout 0.60 on clips ≤ 4 s, worse than M1b), so the mixture-of-experts idea was backwards; order-preserving calibration (Platt, isotonic) cannot move an oracle-threshold minDCF, so calibration work is worth zero; M3 restricted to suppression can only move files toward real and cannot repair a miss tail no matter how good it is at unseen generators.

## What to do (adopted)

**Gate, frozen before any candidate's numbers are seen** (GPT-5.6's, adopted whole; strictly tighter than the standing rule, whose "holdout may worsen by 0.05" both experts called indefensible for one shot):
1. Each candidate's definition, weights and preprocessing go in a manifest before its proxy table exists; candidates are never combined tonight.
2. In-the-Wild brief-cost improvement ≥ 0.020; sponsor-cost regression ≤ 0.002.
3. Inner-OOF brief-cost improvement ≥ 0.010; no sponsor-cost regression at printed precision.
4. Holdout degradation ≤ 0.010 under either convention (a safety constraint, not evidence).
5. Both costs must improve on at least two of the three proxy populations, and the candidate's mechanism diagnostic must pass (noise twins must recover the 20 dB lane; a new probe must correct baseline errors rather than clone M1b's ranks).
6. If the candidate's test-ranking Spearman against the shipped file falls below 0.974, the required gains double; 0.5-crossings are counted for disclosure only.
7. If two candidates pass, ship neither unless one Pareto-dominates across all six split × cost cells. Nathan ratifies exactly one candidate from one consolidated packet; the runner must then reproduce the exact ordering and pinned block. Nothing auto-ships.

**Candidates, ranked** (the memo's table, with our repo facts added):

| # | Candidate | Verdict | Expected test ΔminDCF | Repo status |
|---|---|---|---|---|
| 1 | **WavLM Large frozen probe**, M1b recipe exactly (band match, segment mode, test crops, time-mean, fold-local scaler + logistic + Platt, ASV19 rows inner-only) | do first | EV −0.002; range −0.015 to +0.020 | `scripts/extract_embeddings.py --model wavlm-large` loads through `AutoModel` with `output_hidden_states`, so no code change; four extractions on MPS (train sample 20k ≈ 22 min, ASV19 add-on 7.6k ≈ 8 min, test 1.7k, In-the-Wild 3k) ≈ 40 min; `train_probe.py --folds splits/nsa_folds_plus_asv19.csv` picks the layer by inner CV as for M1b |
| 2 | **XLS-R layers 5–9 concatenated probe** | do second | EV −0.001; range −0.008 to +0.010 | all 25 layers of the v3 embeddings exist for every row set (`outputs/embeddings/wav2vec2-xls-r-300m/*_v3`, `asv19_addon_v3`, `itw_stress_v3`); `train_probe.py` needs a `--layers 5,6,7,8,9` option (concat → 5,120-d); CPU, under an hour. Gemini's layer-5 literature claim failed source verification: hypothesis, not prior |
| 3 | **Second M3 suppression tier** (halve below −3 as now, quarter below −6) | do as a free audit | EV ≈ 0; range ±0.006 | exported scores only, one variant of `scripts/fuse_sweep_m5.py`; ten minutes. Absolute diagnostic: the deeper tier must fire on zero known spoof rows across all proxies and perturbations |
| 4 | **Noise-augmented handcrafted twins** (one 20 dB white-noise twin per class, fit-side rows only, clean validation rows, unchanged LightGBM recipe, direct replacement of the handcrafted slot) | do in a parallel session only | EV ≈ 0; range −0.006 to +0.012 | `hearsay.augment.add_noise` exists; `scripts/extract_handcrafted.py` has `--launder-frac` / `--tilt-frac` and needs a `--noise-frac`; re-extraction ≈ 17 min at 4 workers plus training via `--extra-rows-from`; about an hour. Caveat: additive noise is not an established test-set mechanism (the shift is a smooth low-pass) |
| 5 | Delta/dynamics-only handcrafted | maybe, one bounded run if a session is idle | EV +0.001 | disputed (Gemini do, GPT-5.6 skip); the column-pruning failure points against it |
| 6 | M5 fold-model averaging | skip | 0 | naïve averaging leaks each OOF row into its own prediction; M5's problem is bias, not variance |
| 7 | Noise-augmented M1b embeddings | maybe only if WavLM finishes early | EV −0.001 | duplicates the WavLM slot |
| 8 | New feature families (LTAS residuals, vocoder harmonics, formants, sub-band modulation, bispectral) | skip tonight; README future work | 0 to +0.004 | each is a research lane; LTAS residuals would learn the AUC-0.99 train/test channel split |
| 9 | Generator-family router, property-gated mixture of experts, stackers, isotonic-before-blend, cascade | **skip all** | +0.002 to +0.005, wide downside | re-slicing the same three columns against thrice-reused proxies; the short-clip premise is contradicted by M5's numbers |
| 10 | Capped Bayesian M3 likelihood | forbidden | unbounded asymmetric downside | M3's margin is not a calibrated likelihood; violates our suppression-only rule |

**Pre-declared fusion entry for a new probe (1 or 2):** the new column takes 0.20 from M1b's 0.60; handcrafted and M5 stay at 0.20; the M3 step is unchanged; no weight sweeps tonight. One candidate per slot.

**Clock (memo times shifted +75 min to the memo's arrival; Nathan's bandwidth, not compute, is the binding constraint):**

| Time (EDT) | Activity | Nathan |
|---|---|---|
| 17:00–17:20 | Hash the shipped TSV; write the manifest (four candidates, one fusion entry each) and the gate above; run the M3 second tier | approve the manifest only |
| 17:20–18:50 | Parallel: WavLM probe (MPS), XLS-R 5–9 probe (CPU), noise twins (CPU); optional delta-only run | none |
| 18:50–19:30 | One locked table: inner / holdout / In-the-Wild under both costs, mechanism diagnostics, rank displacement | none |
| 19:30–20:00 | Review at most one ready-to-ratify candidate | 30 min hard cap |
| 20:00–21:30 | Only if ratified: regenerate all 1,671 scores through the runner, reproduce ordering and pinned block, update numbers | final ratification |
| 21:30–05:00 | Stop touching the scoring path and the TSV; sleep; documentation lanes continue; README polish 05:00–07:30 | none |

**Stop/go (the memo's direct call):** continue, but only the frozen bounded jobs above, for about three hours; change the TSV now, no; ship a marginal proxy win, no; if nothing clears by the decision window, stop entirely rather than filling the clock. Gemini's full-13-hour schedule is rejected.

## What to ignore (and why)

- All fusion redesign: constrained stackers, isotonic calibration, property-gated experts, cascades, Bayesian M3, a second learned layer. Same three columns, thrice-reused proxies, no new information; and calibration cannot move minDCF at all.
- M5 fold averaging (leaks OOF rows). New feature families tonight (feature lottery). Gemini's "mathematically definitive" brief-cost claim and its high confidence.
- Two literature claims failed VeriLM's own source check: WavLM-fusion precedent (GPT-5.6) and layer-5 discriminability in ASVspoof 5 (Gemini). Both candidates stay on mechanism plausibility only.

## Where the experts disagreed

Gemini: the cost convention is certain from the EER feasibility argument, and the layer 5–9 probe fixes a representation gap. GPT-5.6: the pair is non-identifying (sponsor-feasible points exist, e.g. 67 false alarms and 2 misses); the miss-tail read rests on NSA's spoken statement. Synthesis adopted GPT-5.6's framing, kept Gemini's constraint as a sharpening, demoted the layer idea to a cheap second experiment. Delta-only handcrafted: Gemini do, GPT-5.6 skip; left as a maybe.

## Failure modes the memo named that we had not

1. **Winner's curse from proxy reuse:** fold discipline prevents row leakage, not selection-fitting from many free sweeps on the same three tables; cap the candidate count in the manifest.
2. **Sponsor-code sweep bug:** any sweep that silently optimizes P_FA + 4·P_miss instead of 9.33·P_FA + P_miss drives the system into the high-false-alarm region the brief punishes 9× per file. Check the cost function in every new script.
3. **Effective sample size below 1,671** makes every file-count equivalent look more certain than it is.
4. **Reproducibility near-ties:** Spearman 1.0 with max drift 9.3e-4 does not guarantee identical ordering among near-tied files; compare exact sorted file IDs before and after any regeneration.
5. **Fatigue-driven ratification error:** one packet at the decision window and an early immutable freeze are worth more than any experiment; a 03:00 ratification is the likeliest way to lose the 0.0733 in hand.

## Unmeasured

NSA's cost convention, P_FA, P_miss and threshold; every candidate's effect on the test set (unobservable before the freeze); whether the presumed miss tail is unseen generators (inference from MLAAD and M3 probes, not measurement).

## Our own reading, added

- **Rubric arithmetic.** The detection gap to first place is 0.95 of 60 points. The four experiments together have an expected gain of a fraction of a point, and none adds a new rubric technique (a second SSL probe is still "deep anti-spoofing"). They are worth running because they are cheap, bounded, and produce graded negative results; they are not worth a late night. The 40 points for diversity and documentation remain the larger lever, and those lanes must not be starved for this.
- **Delta-only handcrafted** is the one disputed item we lean toward skipping: it is the pruning experiment in different clothes, and the CPU session's hour is better spent on the noise twins the channel lane's measurement actually motivates.

---

## Round 2: an independent Claude Fable 5.1 opinion, and what it changes (Sat ~17:20)

Nathan asked for a second opinion from a Fable agent with the same question and repo access, blind to the VeriLM memo. Verbatim: `2026-09-26_post-draft-review_RESPONSE_fable.md`. Unlike the panel, it computed on the exports. The consult chat re-verified its two load-bearing claims from the exports with the sweep's own loaders:

| Rule | Inner (brief / averse) | Holdout | In-the-Wild | Test Spearman vs shipped |
|---|---|---|---|---|
| Shipped A3 w0.2 + E | 0.1351 / 0.2997 | 0.0065 / 0.0087 | 0.228 / 0.2385 | 1.000 |
| A3 w0.3 + E | 0.1374 / 0.3139 | 0.0060 / 0.0091 | 0.2113 / 0.2310 | 0.993 |
| **A3 w0.4 + E** | 0.1373 / 0.3366 | 0.0065 / 0.0088 | **0.1997 / 0.2075** | 0.970 |
| Second M3 tier on shipped | 0.1351 / 0.2997 | 0.0065 / 0.0087 | 0.228 / 0.2385 | 0.986 |

Both claims hold: the M5 weight at 0.4 improves In-the-Wild by 0.028 under both costs with the holdout unchanged and inner worse by 0.002 (brief) and 0.037 (averse); the second tier is inert at the metric on every split (it reorders files only below the threshold region).

**Where the two opinions agree:** direction is ours; the cost convention is unknowable from the pair and every candidate must improve under both costs; the likely error is a miss tail; one readout licenses nothing; calibration is worth zero; the mixture-of-experts premise is backwards; M5 fold averaging is dead; stop by ~21:30 and let sleep equal shipping.

**Where they differ, and what we adopt:**
1. **XLS-R layers 5–9 (VeriLM's #2 "do"): dropped.** The Fable agent ran it: with the layer-7 baseline reproducing M1b exactly (inner 0.2649, holdout 0.0717, In-the-Wild 0.342), the mean of layers 5–9 improves the holdout (0.052) but worsens In-the-Wild (0.368) and doubles its false-alarm rate (0.8% → 1.55%); the concatenations of 6–8, of 5/7/9 and of 5–9 give In-the-Wild 0.353 / 0.430 / 0.399. Every head on the same backbone inherits M1b's blind spot: of the 158 In-the-Wild misses under the shipped rule, M1b would flag 1, M5 33, M3 107. Recorded as a negative result for the README; the heads chat runs WavLM only.
2. **Second M3 tier (VeriLM's #3 "free audit"): inert**, verified. Still run by the gate chat as a ten-second audit so the record shows it, expected outcome "identical".
3. **Noise twins (VeriLM's #4): README only.** Both opinions put its test effect at zero; the Fable agent adds that the test set shows no additive noise (its noise-floor percentiles sit beyond the clean end in the channel lane's per-feature table). The CPU chat is re-scoped to negative-result write-ups; the twins run only if that chat has idle time.
4. **New candidate: A3 (0.4, 0.2, 0.4) + E, post hoc.** The pre-declared M5 sweep stopped at w = 0.2 on an expectation that proved wrong. Under the standing four-condition rule it qualifies; under the stricter gate adopted above it **fails rule 3** (inner must improve by ≥ 0.010 with no sponsor-cost regression; it worsens by 0.002 and 0.037). It is added to the manifest as the one fusion candidate, labelled post hoc, with the disclosure that its numbers were seen before the manifest was written. Which gate governs it is Nathan's call to make before the 19:30 packet, not after; the consult chat's recommendation is the stricter gate, which means KEEP unless he decides otherwise.
5. **WavLM's fusion entry:** VeriLM says a fourth column at 0.2 taken from M1b; the Fable agent says the refit seat (replace M1b) via the existing `--refit-m1b` mode. We keep the fourth-column entry as the single pre-declared candidate (it adds a backbone without removing the best-validated stream) and do not run the refit variant tonight (one candidate per column).
6. **M3 threshold −3 → −1.5 and M3 promotion:** both improve every proxy (promotion recovers 76 of 158 In-the-Wild misses at zero added false alarms) and both stay skipped: the evidence is In-the-Wild, which M3's authors evaluated on, and the suppression-only rule is a documented principle. Recorded as the measured ceiling of what the excluded detector could do.

**Adopted additions to the plan:** the gate chat copies the shipped TSV to its final name with its sha256 in the log before anyone sleeps ("sleep equals ship"); the full 1,671-row parity is re-run if the M5 weight changes (its CPU-vs-A100 logit gap scales with the weight); no one inspects the files that cross 0.5 on the test set to decide anything (that is selection on the test set); the handcrafted column's role is documented honestly (admissible stacker coefficient −0.04; test-set Spearman with M1b 0.21 against 0.79 on the holdout, so it reads the envelope shift; it is the readable column, not a vote).

**Correction (CPU lane, ~19:10):** the Fable memo's sponsor-cost example "5 misses + 40 FA" is infeasible: 40 of ~1,170 real files is 3.4%, below the 3.53% EER, and under the sponsor's cost the feasible region is P_FA in [3.53%, 7.33%] (41–86 false alarms) with P_miss ≤ 0.95%. Cite the feasible example instead: 67 false alarms + 2 misses (`docs/reports/2026-09-26_post-draft-negatives.md`, item 5). The memo's brief-cost example (3 FA + 25 misses) is feasible.

**Correction (gate lane, ~19:05; recorded at `b877760`):** the miss-tail counts quoted in Round 2 ("158 In-the-Wild misses; M1b flags 1, M5 33, Spectra 107") were taken at the In-the-Wild argmin threshold, not at the inner-OOF threshold the rule names. At CURRENT's inner-OOF threshold the shipped rule misses 122 In-the-Wild fakes, of which M5 alone flags 19 and Spectra 79. The mechanism reading is unchanged (heads on M1b's backbone cannot see the tail; only a different backbone or Spectra can); the numbers the gate's P_wl diagnostic uses are the inner-threshold ones. Also: Nathan had already ruled on W4 at ~17:40 (a fresh-evidence bake-off: standing rule, within −0.010 on all ten perturbation cells, speaker-cluster bootstrap 5th percentile of the In-the-Wild gain above 0 under both costs; manifest `141b254`), so the Round 2 line "which gate governs it is Nathan's call to make" was already settled.

**Correction and outcome (CPU lane, ~19:30):** Round 2 item 3 said "the test set shows no additive noise"; that is too strong. In the channel lane's per-feature table the noise-floor percentiles (p2, p10) sit beyond the clean end, but floor stationarity (1.08) and flatness (1.74) sit beyond the wild end, so the reading is "probably no additive noise", with that caveat. The noise twins were run anyway on Nathan's word (handcrafted v6, `docs/reports/2026-09-26_post-draft-hc-noise.md`): noise-cohort AUC 0.640 → 0.772 against the 0.90 bar, clean AUC 0.998 → 0.993, clean holdout 0.137 → 0.227, In-the-Wild AUC 0.768 → 0.722. A measured negative; `models/hc_selected` untouched. With that, all four post-draft candidates (T2, W4, P_wl, H_noise) failed their pre-declared rules (`docs/reports/2026-09-26_post-draft-sweep.md`); the decision is KEEP and the shipped file stands.
