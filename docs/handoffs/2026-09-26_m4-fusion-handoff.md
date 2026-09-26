# Handoff: M4 fusion (Sat Sep 26, 2026, ~09:10)

**Purpose of this chat:** own the fusion rule from now to the 22:00 freeze and the Sunday 05:00 final TSV. Three jobs, in order: (1) get Nathan's decision on the M5 sweep result, which by the pre-declared rule replaces the frozen rule; (2) decode NSA's draft-review number when it arrives and act on the decoding table; (3) produce the final TSV, its preflight and its direction check. Every change to what ships is ratified by Nathan; this chat proposes and executes, it does not decide.

Read first: `docs/reports/2026-09-26_fusion-sweep-predeclared.md` (the rule, the sweep, the M5 addendum), `docs/consults/2026-09-26_fusion-strategy_RESPONSE.md` (why the rule is shaped this way; items 1–6 are adopted), `models/fusion_v1/constants.json` (the `how` field is the rule in one line), then `outputs/fusion/sweep_m5_report.json`.

## Context

**The shipped rule, `e_on_a`, frozen 08:13.** Rank each of M1b v3 and handcrafted v5 against its own inner-OOF logits (ECDF); `base = 0.8·rank_m1b + 0.2·rank_hc`; if Spectra-AASIST's margin is below −3 and `base` is above 0.5, `base *= 0.5` (M3 suppresses false alarms, never promotes); Platt map at the 0.3 prior into [0.001, 1]; files that are not speech or fail to decode sit in a pinned block below 0.001. Readout: inner 0.140, holdout 0.014, In-the-Wild 0.260 brief / 0.267 miss-averse. Chosen by a selection rule written before the results; the candidates, rule and results are all in the pre-declared report and were not changed after the run.

**The draft-review payload** is `submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv`, copied to `~/Downloads/HEARSAY_predictions.tsv` and approved by Nathan. A pre-flipped twin sits beside it. Nathan DMs it to NSA by 14:00 and asks for P_FA, P_miss and EER along with minDCF.

**The runner and the Docker image implement `e_on_a`.** `hearsay.pipeline.FusionConstants.fuse` reads the arithmetic from the constants file (no fitting at inference); `scripts/run_pipeline.py` reproduces the 08:13 TSV from exported logits to 4e-16 on all 1,671 rows. The image `hearsay:20260926-0753` predates the rule and is being rebuilt on runner v2 (commit `9614c18`) by the Docker chat. Any rule change means: new constants file, a pipeline change if the inputs change, a 22-minute Mac rerun, a Docker rebuild (~8 min) and the 50-file parity check again.

**The M5 sweep has run and the frozen rule lost.** `scripts/fuse_sweep_m5.py` (08:58, uncommitted, main chat's file) evaluated the two pre-declared M5 candidates against the current rule on identical row sets (16,142 inner, 3,858 holdout, 3,000 In-the-Wild; verified). From `outputs/fusion/sweep_m5_report.json`:

| Candidate | Inner | Holdout brief | Holdout averse | Holdout [#FA, #miss] | LibriSpeech real | ITW brief | ITW averse | ITW P_FA / P_miss at inner thr |
|---|---|---|---|---|---|---|---|---|
| CURRENT E on A α0.2 | 0.1395 | 0.014 | 0.0088 | [1, 18] | 0.0186 | 0.2603 | 0.267 | 1.4% / 14.8% |
| F: M5 as second suppressor | 0.1364 | 0.014 | 0.0088 | [1, 18] | 0.0186 | 0.2577 | 0.267 | 1.35% / 14.4% |
| A3 w0.1 (no M3 step) | 0.2222 | 0.0215 | 0.0272 | [1, 33] | 0.0261 | 0.282 | 0.2635 | 0.35% / 33.3% |
| A3 w0.1 + E | 0.1364 | 0.0095 | 0.0092 | [0, 19] | 0.0095 | 0.2407 | 0.251 | 1.4% / 13.0% |
| A3 w0.2 (no M3 step) | 0.2128 | 0.020 | 0.0249 | [1, 30] | 0.0246 | 0.266 | 0.246 | 0.25% / 32.6% |
| **A3 w0.2 + E** | **0.1351** | **0.0065** | **0.0087** | **[0, 13]** | **0.0065** | **0.2147** | **0.234** | 1.2% / 12.1% |

A3 w0.2 + E is ranks (0.6 M1b, 0.2 handcrafted, 0.2 M5) with the same M3 suppression. It clears all four pre-declared conditions (ITW brief −0.046 against a required −0.01; the other three improve rather than degrade) and is the report's `winner`. F improves ITW brief by only 0.0026 and does not qualify. The M5 suppression threshold `t_F` = −3.90 covered 41.8% of inner real rows, so F was not a dead rule; it just adds nothing on top of M3.

**That table was the whole-clip run; the crop-fair run supersedes it (main chat, 09:15).** The addendum promised M5's In-the-Wild scores on "the same seed-200 crops" as the other columns. `scripts/m5_score.py` has no crop option, so the first run scored all 3,000 M5 ITW rows whole, capped at 8 s, while M1b, handcrafted and M3 saw test-length crops. The main chat disclosed the deviation, wrote the identical `prepare_segment(x, crop_s, 200 + row)` crops to WAV, re-scored M5 and re-ran the sweep. Crop-fair, A3 w0.2 + E moves to ITW brief **0.228** (from 0.215 whole-clip; current 0.260) and ITW averse **0.239** (from 0.234; current 0.267); inner and holdout are unchanged. It still clears all four conditions (ITW brief −0.032 against a required −0.01). F stays under the bar. Both runs are recorded in the addendum. Verified by the oversight chat at 09:15: the candidate TSVs have 1,671 rows in template order, scores in [0.0014, 0.9996], no NaN; Spearman 0.974 against the current TSV; 3 of 1,671 files cross 0.5; share above 0.5 is 27.4% either way.

Candidate artifacts, none shipped: `models/fusion_v2_candidate/constants.json` (weights 0.6 / 0.2 / 0.2 in rank space, the same M3 `e_rule`, Platt, the map applied once; its `status` field says CANDIDATE), `submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_{our_direction,FLIPPED_only_if_NSA_scores_inverted}.tsv`, and two log rows marked "pending ratification".

**Main chat's recommendation (09:15):** ratify A3 for the FINAL only if the Docker chat can add the M5 scorer and re-verify smoke and parity comfortably before 22:00; the metric gain is real but modest and the test ranking barely moves, so Docker risk is the deciding factor. Otherwise keep `e_on_a` frozen. The draft review stays the approved `e_on_a` file either way. **Docker chat's estimate (09:20):** about 45 min wall-clock after the runner commit with the M5 scorer lands, 1 h with one retry (staging line and test 10 min; rebuild with cached deps 5–8 min, +0.63 GB layer; smoke 5; 50-file parity 5–6 min at ~5.5 s/file under Rosetta; PCM 1; spec and commit 10). Fits the 20 GB VM disk only with one image tag kept. Memory: M5 cannot share M1b's backbone (fine-tuned, hash-checked weights of its own, 12 layers), so resident memory goes from 3.3 GiB to about 4.2 GiB in a 5.8 GiB VM; fits for a scoring run alone, with little headroom; Colima can be restarted at 8 GB in two minutes without a rebuild. Its own state: the runner-v2 rebuild failed at 08:4x on VM disk (three 7.5 GB tags under the 20 GB cap; fix is deleting the superseded tag first) and is being restarted now; the full in-image run was stopped at 664/1,671 and is not resumed. So the pipeline change (1–2 h, this lane) is the long pole, not the image.

**Nathan's decision (09:25): keep `e_on_a` frozen.** The candidate is not shipped and nothing is re-pointed; the draft review goes out as the approved 08:13 file. The candidate is revisited only after NSA's draft-review number is back (after 14:00), and only if it decodes to our direction with time left before 22:00 for the pipeline change (1–2 h) plus the image (~1 h). Until then, this lane's work is the decoding-table branches, not A3. Both chats have been told. The addendum says a change needs his ratification, and adopting A3 has costs the sweep does not price:
- M5 is not a pipeline scorer today. `hearsay.pipeline.DETECTOR_ORDER` is `("m1b_v3", "handcrafted_v5", "spectra_aasist")` and `RANKED` is the two-detector tuple. Adding M5 means a new scorer in `pipeline.py` around `hearsay.m5_model.load_m5` / `score_batch` with the deploy transform from `hearsay.m5_data` (trim, cap 8 s, normalize; the exported order is trim then band-limit), a three-way `RANKED`, a `fusion_v2/constants.json` with M5's inner-OOF rank reference, and tests in `tests/test_pipeline.py`. The pipeline is frozen at `9614c18`; the Docker chat rebuilds on every change and must be told first.
- The image grows by the M5 model (`models/m5_xlsr_ft_20260926-0741/model`, 631 MB) and M5 loads the first 12 XLS-R layers where M1b truncates at 7; ~0.2 s per clip on the Mac, several times that under Rosetta.
- M5's known weak spots are the test set's shape: clips ≤ 4 s score 0.60 alone (test median 3.4 s), LibriSpeech real 0.43 alone. Inside the blend those slices improved (LibriSpeech real 0.0186 → 0.0065, holdout misses 18 → 13), which is the argument for it, but the holdout is eight in-family generators and the 0.014 → 0.0065 move is inside the resolution floor the consult warned about. The In-the-Wild gain (0.260 → 0.215) is the real evidence, and ITW is 3,000 clips scored once, never trained on.
- Under the sponsor code's inverted cost (miss-averse column) A3 + E also improves (0.267 → 0.234), so the change is not a bet on the direction.
- Timeline: pipeline change 1–2 h, Mac rerun 22 min, rebuild and parity ~30 min. Feasible before 22:00 if started by early afternoon; not feasible after the draft-review number arrives if that number itself forces a change.

A defensible middle path, if Nathan wants the gain without the rebuild risk: ship A3 + E as the TSV (computed from the exports, exactly as the 08:13 TSV was) and keep the Docker image on `e_on_a`, with the difference disclosed in the README. That trades a parity claim ("the image reproduces the TSV") for minDCF. Say so plainly to Nathan; do not choose for him.

## Working branch / worktree

`main` in `~/Projects/hearsay`, shared by several chats. Fusion files (`scripts/fuse.py`, `fuse_sweep.py`, `fuse_sweep_m5.py`, `models/fusion_v*`, `submissions/`) belong to this lane. `scripts/fuse_sweep_m5.py` is uncommitted; commit it with the addendum results before anything else. Stage by path; never `git add -A`. The main chat pushes; ask Nathan before pushing.

## Environment / setup

```bash
cd ~/Projects/hearsay && uv sync
```

Re-run the frozen sweep (reads the exports, ~1 min; `--write` re-emits the winner's TSVs in both polarities and the constants):

```bash
cd ~/Projects/hearsay && uv run python scripts/fuse_sweep.py
```

Re-run the M5 sweep (reads `outputs/detector_scores/{m1b_v3,handcrafted_v5,m5_xlsr_ft}.csv`, `_itw_*.csv`, `outputs/spectra/itw_stress/scores.csv`):

```bash
cd ~/Projects/hearsay && uv run python scripts/fuse_sweep_m5.py
```

Reproduce a TSV live from audio with the runner (22 min on the Mac; resumable):

```bash
cd ~/Projects/hearsay && uv run python scripts/run_pipeline.py --data data/nsa/HackGTHearsayTesting --out outputs/runner/<name> --template data/nsa/HearsayScoreKey4TeamX.tsv
```

## What to do next

1. **Finish and commit the M5 sweep**: the crop-fair M5 ITW re-score, the re-run of `scripts/fuse_sweep_m5.py`, and both result tables (whole-clip and crop-fair) appended under the addendum in `docs/reports/2026-09-26_fusion-sweep-predeclared.md`, with the deviation named and the verdict in one sentence ("qualifies on crop-fair ITW; awaiting ratification" or "does not qualify; rule stays frozen"). A pre-declared rule whose result is not recorded next to it is worth nothing. Do not start step 2 from the whole-clip table.
2. **Put the decision to Nathan** in five lines: the table row for A3 w0.2 + E against CURRENT, the four conditions met, the three costs (pipeline change, image size and rebuild, M5's short-clip weakness), and the middle path. Ask for one of: ratify and rebuild; ratify TSV-only; keep frozen. Record the answer in the addendum.
3. **If ratified (either form):** `uv run python scripts/fuse_sweep_m5.py --write` (add the flag if it does not exist yet, mirroring `fuse_sweep.py`) to emit `models/fusion_v2/constants.json` and both polarity TSVs under a new timestamp, and append the log row. The TSV must have 1,671 rows in the template's order, determinate scores in [0.001, 1], the pinned block below, and min score reported in the notes. If the runner is to follow, message the Docker chat before touching `pipeline.py`.
4. **Pre-compute what the draft-review number can trigger.** Per the decoding table (consult RESPONSE item 4): 0.00–0.20 ship unchanged; 0.20–0.45 switch to the most false-alarm-robust admissible candidate (from the 08:13 table that is α 0.3 or D, and A3 w0.2 + E if ratified); 0.45–0.90 fall back to M1b v3 alone and widen the block; 0.95–1.00 flip. Have the TSV for each branch written before 14:00 so the reaction is a file copy. If EER comes back instead: 2–5% is our direction, 95–98% theirs.
5. **Watch the channel-robustness chat's 16:00 report.** If it ships a symmetric codec refit of M1b or handcrafted, the exports change, and both sweeps must be re-run from scratch under the same pre-declared rules; the rank references in the constants are tied to the exports they were built from.
6. **Freeze at 22:00.** After it, only a decode of NSA's number or a broken TSV reopens the rule. Sunday 05:00: final TSV, `preflight` (row count, order, header, range, no NaN, min score), direction check against the log, hand to Nathan.

## IMPORTANT — tests & at-risk artifacts (make sure these survive)

- Test: `uv run pytest -q tests/test_pipeline.py tests/test_api.py` → passes at `9614c18` (part of the 444-test suite). These pin the rule's arithmetic, the pinned-block placement and the `[0.001, 1]` map; run them after any constants or pipeline change.
- Test: `uv run pytest -q tests/test_docs_consistency.py` → 52 passed; run after editing the report, STATUS or README (no prior but 0.3, no bare "CSV").
- Reproduction number: `scripts/fuse_sweep.py` must print inner 0.1395 / holdout 0.014 / ITW brief 0.2603 for E on A α0.2; `fuse_sweep_m5.py` must print the table above. If either drifts, an export changed underneath you; find out which before doing anything else.
- No test covers `fuse_sweep.py` or `fuse_sweep_m5.py` directly; the reproduction numbers above are the test.
- At-risk, NOT archived, all regenerable but slowly: `outputs/detector_scores/*.csv` (the fused columns' exports: `m1b_v3` 3.3 MB, `handcrafted_v5`, `spectra_aasist`, `m5_xlsr_ft`, and the `_itw_*` ITW columns; M1b's export needs the 22-min embedding run, M5's needs the 631 MB model, Spectra's the weights). `outputs/spectra/itw_stress/scores.csv`. `outputs/fusion/sweep_report.json`, `sweep_m5_report.json`, `sweep_final_test.csv`. `models/fusion_v1/constants.json` and `models/m5_xlsr_ft_20260926-0741/` (gitignored; the M5 checkpoint also lives on the private HF Hub repo `nrs124554433/hearsay-m5-xlsr`). Nothing under `outputs/` or `models/` is committed; copy to the external drive only if it has room (it was at 100% this morning; check `df -h` first) and never delete, move.
- In flight, other lanes: Docker chat rebuilding on runner v2 and resuming the full in-image run under `outputs/docker/k-full/` (664 of 1,671 at 09:07); channel-robustness chat (report 16:00); README chat (20:00). The oversight chat relays; message it rather than the lanes directly for anything cross-lane.

## Analytical notes

- Score direction: 1.0 = synthetic, never flipped unless NSA's number decodes to 0.95–1.00. The pinned block is the one hedge that is right under both conventions (consult item 5).
- The equal-weight rules (zmean, rankmean) are dead: they double In-the-Wild misses. The 06:02 `stack_nonlj` TSV had Spectra inside a fitted stacker, which is forbidden (undisclosed training data). Any rule that fits on M3 is out regardless of its numbers.
- "Test share above 0.5" is not a selection signal (monotone maps cannot change minDCF); 27.4% of the test set is above 0.5 under the frozen rule, 458 files, as a smoke alarm only.
- The holdout is optimistic (eight in-family generators); In-the-Wild under both costs is what the selection rule keys on, and it is 2,000 real + 1,000 spoof clips scored once.
- The consult's three M3 leakage probes (perturbation sensitivity, holdout-vs-test Spearman drop, MLAAD spoof through M3) sit with the channel-robustness chat; M3 stays suppression-only whatever they say.
- λ̂ (the test set's wild-domain weight; crossover 0.32) is also the channel chat's; if it comes back high, the FA-robust branch of the decoding table matters more.

## Pointers

- `docs/reports/2026-09-26_fusion-sweep-predeclared.md`: candidates, rule, results, M5 addendum.
- `outputs/fusion/sweep_m5_report.json`: the M5 result, `t_F`, qualifying list, winner.
- `docs/consults/2026-09-26_fusion-strategy_RESPONSE.md`: adopted items and the decoding table; verbatim memo beside it (500 KB; the copy in `~/Downloads` is byte-identical).
- `src/hearsay/pipeline.py`: `FusionConstants` (lines ~180–270) and `fusion_block`; `RANKED`, `DETECTOR_ORDER` at the top.
- `scripts/m5_score.py`, `src/hearsay/m5_model.py`, `src/hearsay/m5_data.py`: the M5 CPU path, if M5 joins the pipeline; parity vs A100 max |Δlogit| 0.012 (`outputs/m5_runs/parity_f6.json`).
- `docs/reports/2026-09-26_m5-xlsr-finetune.md`: M5's slices (short clips, LibriSpeech real, codecs).
- `submissions/log.csv`: every TSV with its numbers; the source of truth on numbers.
- `docs/handoffs/2026-09-26_oversight-handoff.md`: the lanes, their addresses and the day's deadlines.
