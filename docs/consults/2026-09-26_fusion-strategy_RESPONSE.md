# Response: fusion, selection, false alarms and the "default answer" (received Sat Sep 26, 2026, 08:05)

**Source:** VeriLM "Deep Think" memo, two experts (Claude and Gemini) with a synthesis; confidence medium. Verbatim copy: `2026-09-26_fusion-strategy_RESPONSE_verbatim.md` (500 KB, includes figures). Prompt: `2026-09-26_fusion-strategy_CONSULTATION.md`.

**Read this first.** The memo was written against the numbers as of ~05:50. Three things have moved since: M3's holdout landed at 0.012 (below the memo's 0.05 threshold for admitting it as a capped vote); the non-speech gate is built and trips 0 of 1,671 test files (the memo's "30 minutes lost, documented negative" branch); Docker is owned, built and verified. Everything else below stands.

## Verdict (the memo's, one line)

Ship an M1b-dominant ranking, not a 50/50 rank or z-score fusion; spend the single draft shot on the exact artifact we will ship, to read the score direction; pin indeterminate and non-speech files strictly below every determinate score and never flip that block; admit M3 only as a capped vote or a false-alarm-suppressing gate, never inside a fitted stacker; document the dead engineered detectors honestly.

## What to do (adopted)

1. **Do not ship the equal-weight rules (zmean, rankmean).** Both experts rejected them on the same evidence: they buy about 0.03 holdout DCF (roughly six holdout files, inside the holdout's resolution floor) while doubling In-the-Wild misses from 28% to 59%, a ~20σ effect. Under the sponsor code's inverted cost, where the 4× lands on spoof-accepted, the equal-weight family collapses (ITW 0.59–0.60) and the M1b-dominant family holds (0.31–0.34). **This reverses the main chat's 06:02 recommendation of zmean.**
2. **Ship `stack_nonlj` over M1b v3 + handcrafted, with M1b v3 alone as the fallback.** They are the two ends of one α-sweep. Run the sweep this morning in rank domain, α ∈ {0, 0.1, …, 0.5}, plus the min rule, max rule, cascade veto and a non-negative stacker shrunk toward equal weights; report each on inner OOF, holdout, per-real-source, ITW under **both** cost weightings, and the mixture estimate π̂; pre-declare the objective; do not iterate. Expect the optimum near α ≈ 0.2–0.3. Ship the stacker if the sweep confirms a non-zero optimum, M1b alone if not.
3. **Never let a fitted stacker see M3.** The 06:02 `stack_nonlj` TSV includes Spectra in the fit; that is the configuration the memo forbids. With M3's holdout now 0.012, admit it only as a capped vote in an unfitted rank rule (median-of-three, or weight 0.25), and preferably as **FA suppression only**: damp a file toward real when M1b says synthetic and M3 says strongly bona fide; never let M3 promote a file toward synthetic. Rationale: its one solid claim (0 of 2,000 ITW false alarms) is on the real side, and memorising real speech is far less dangerous than memorising generators. Run the three leakage probes (about an hour): perturbation sensitivity on 500 inner clips (MP3 round-trip, +20 dB noise, ±2% stretch, one-sample shift; compare ΔAUC of M3 vs M1b); Spearman of M1b vs M3 ranks on the holdout versus on the test set (a large drop means holdout agreement is memorisation); MLAAD spoof scored by M3.
4. **Draft review at 14:00: send the actual shipped artifact**, gate and pinned block included, exact TSV format. One number then yields the direction, the true test minDCF of what we will ship, how far the holdout overstates, and a format check. Ask NSA for P_FA, P_miss and EER alongside minDCF. **Pre-compute both polarity TSVs before 14:00** so the post-feedback action is a file copy. Decoding table:

   | Returned minDCF | Inference | Action |
   |---|---|---|
   | 0.00–0.20 | our direction; test behaves like the holdout | ship unchanged; remaining hours to README and diversity |
   | 0.20–0.45 | our direction; test is wild-like or out-of-family | switch to the most FA-robust candidate; keep the pinned block |
   | 0.45–0.90 | ambiguous | do not flip (a flipped good detector lands at 1.00, not 0.6); fall back to M1b v3 alone, widen the block |
   | 0.95–1.00 | the shipped code's polarity (1.00 is the trivial-system ceiling) | flip the final TSV, keep the block pinned at the minimum, re-select the rule under the miss-averse cost |

   If EER comes back instead: 2–5% means our direction, 95–98% means theirs. Reject the two-level "benchmark-maxing" TSV: the 1.00 cap saturates it, it decodes nothing and burns the shot.
5. **Default answer: the pinned block is the one direction-robust hedge.** For a block with internal spoof rate q, placing it at the bottom is optimal under the brief if q < 0.800 and under the shipped code if q > 0.097; indeterminate files sit near q ≈ 0.30, inside the window with ~3× margin on both sides. Implementation is placement, not value: map determinate scores into [0.001, 1.000], put the block in [0.000, 0.001) strictly below everything, order within the block by whatever weak signal remains (raw M1b score, voiced fraction, duration) and hash-jitter the rest so no two files share a value (a superset of thresholds can only help, by ~0.001–0.002 DCF, never hurt). Decode failures and non-speech go to the bottom of the block; high-disagreement files stay in the determinate range, compressed toward its low end; mid-confidence files stay put. **Our current `apply_default_answer` (0.02 + 1e-4 × fused) does not do this:** determinate real files score as low as 0.0009, so a gated file at 0.02 would rank above them. Fix the placement even though 0 test files are gated today.
6. **Add the miss-averse cost as a second reported column in fuse.py** (P_miss + 4·P_FA, the shipped code's semantics). Ten minutes; protects every selection against the cost inversion.
7. **Estimate λ, the test set's wild-domain weight.** Label-free channel statistics (95% roll-off, long-term tilt, noise-floor level and stationarity, a reverberation proxy), a tiny classifier separating holdout-real from ITW-real, averaged over the 1,671 test files. Crossover 0.32: below it rankmean would win, above it the stacker. The memo's prior: a uniform 7.2 kHz wall on every file looks like one curated re-encode pipeline, not heterogeneous field audio (real telephony would show a 3.4 kHz mode). Thirty minutes, the most decision-relevant unmeasured quantity.
8. **Diagnose the 7.2 kHz edge as a codec and, if it is, replicate the round-trip on BOTH classes.** Encode 50 LJ clips at several bitrates, compare roll-off slope and high-band flatness to the test files, pick the closest, apply to all training clips, re-extract, refit. Symmetric or not at all: augmenting one class recreates the 7.2 kHz shortcut in reverse. Keep the band-match low-pass; avoid aggressive high-pass augmentation on an SSL front-end.
9. **One good wild, non-read bona fide corpus, speaker-grouped into inner folds only**, if it downloads within 30 minutes; otherwise abandon. The memo corrects our own premise: post-band-fix, VCTK helped on both axes (ITW 0.380 → 0.342, holdout 0.159 → 0.072), so the diversity hypothesis is alive. One corpus, not three: piling on data was worse than a two-corpus setup in the cited SSL results.
10. **README and diversity write-up are worth ~40 points** and dwarf every remaining minDCF move. Document the dead engineered detectors with their mechanisms (uniform decode destroys container evidence; 16 kHz PCM puts ENF below the useful band; re-encoding destroys splice discontinuities). For orchestration credit, show that routing changed decisions: a router on measured file properties, an abstention path, a per-file JSON trace, 5–10 worked examples, and a router-on vs router-off ablation. A measured null delta is itself rubric-positive.

## What to ignore (and why)

- **"Test share above 0.5" as a selection signal.** minDCF is a minimum over a threshold sweep and depends only on ranking; any monotone map (the 0.5 cut, z-scoring, Platt, test-ECDF) changes the share and changes minDCF by exactly zero. The relevant Bayes threshold is 0.903 under the brief, not 0.5. Keep π̂ from a two-component mixture fit as a smoke alarm only.
- **Nested CV and weight-stability machinery for fusion selection.** The outer loop still contains only the eight in-family generators; a better estimate of the wrong quantity.
- **More handcrafted features, compression, ENF or splice development.** No further points there.
- **Nearest-neighbour-to-LJ labelling.** A trap: four DiffSSD generators clone LJ's voice, so LJ-proximity is a spoof indicator as much as a real one. README-only.
- **One-class or centre-loss objectives.** Contraindicated with a narrow, multi-cluster bona fide manifold.
- **Further holdout tuning below 0.05.** Below the resolution floor.
- **Gemini's RawBoost-on-the-fine-tune recommendation.** Out of scope (M5 is done and did not clear its gate); the augmentation principle is retained for the frozen path only, and the cited SSL results warn it can hurt.
- **Inference-time speech enhancement.** A model the Docker image does not need today.

## Where the experts disagreed

Gemini said no direction-agnostic hedge exists; Claude showed the shipped ASVspoof5 code also inverts which error carries the 4×, so the expensive claim sits at the high end of the file under both conventions and the minimum-pinned block is robust in both. Adopted. Gemini wanted M1b v3 alone; Claude wanted the stacker; resolved by the α-sweep. Gemini repeated our stale "VCTK did not help" line; Claude caught that the v3 numbers contradict it. Adopted.

## Unmeasured, and how to measure it today

M3's provenance (three probes above); λ (item 7); whether the 7.2 kHz edge is a codec (item 8); the polarity NSA grades with (the draft review); the size of any wild-corpus gain in a frozen-embedding setup (item 9; do not expect full-training figures).

## Failure modes the memo named that we had not

The cost asymmetry inverts inside the sponsor's own script. Rules had been partly selected on the test share, a calibration statistic. Every intuition anchored on the 0.5 cut is off by an order of magnitude in likelihood ratio. A handful of non-speech files at the top of the ranking would cost more than the entire fusion decision (five real files there ≈ 2.5 points; twenty ≈ 10 points). Class-asymmetric augmentation recreates the band bug in reverse. There is no clean evaluation set for M3 anywhere in our possession.
