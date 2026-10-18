# V2-33 stage 0: hard-example bake-off (run spec)

**Lane:** hard negatives (V2-33). **Branch:** `v2`. **Date opened:** Mon Oct 13, 2026.
**Handoff:** `docs/handoffs/2026-10-10_v2-hard-negatives-handoff.md`.
**Pipeline:** `/plan-review`, deep mode. P1 frozen Oct 13; P2 frozen Oct 13 ("freeze deep").

**Stakes:** HIGH — it decides how a scoring model is trained, and its recommendation feeds a
benchmark row. The handoff classified it HIGH; Nathan did not downgrade.

---

## Owner's statement

Verbatim, as typed. Never edited, paraphrased or tidied.

### P1

**What it does:**

> "This step decides. How we use RL and hard negatives, and how we are benchmarking everything so that we can notice progress from step to step."

**Most likely way it is wrong:**

> "Most likely why this goes wrong is a silent bug in the RL loop."

### P2

**What the tests prove:**

> "These tests make sure that the benchmark is correct and works right generally. Make sure the folds match up correctly and then all the data gets loaded in right because like if data gets loaded in like one row off it can kind of screw everything over. The loop makes sure that the weights are changed in the right way. RL is great because, okay, I'm voice typing and it's not working super well. RL is great because its batch efficiency is so high, so we can get some good results out of even the small number of clips. Also looks at other somewhat boilerplate stuff, but it makes sure that the right things are selected and the outputs come out correctly."

**A failure the tests could miss:**

> "One thing I don't see in here that I feel like should be there in a test is we're looking for hard negatives to data mine, but I don't think we talk about how to find those hard negatives in the first place. Like, is there some interpretability or something that we need to do, or is it as simple as looking at whether a clip is real or fake and finding ones that it's confidently wrong on, or even hedging on. Those are the two things that we really don't want. I feel like each is equally bad. And I guess the overall worst thing is a confidently wrong false positive."

### Where the risks went

- **P1 risk (silent bug in the loop)** → failure-mode group **C** (recipes and mining loop,
  8 tests) for stage 0, and group **G1** (pre-declared bandit-loop test contract) for stage 1.
- **P2 risk (how hard negatives are found in the first place)** → failure-mode group **H**
  (6 tests), a new pre-declared **mining-definition axis** with three arms, and the
  **test break check** slot (H2).

---

## Phase 0: resolved conventions

| Thing | Resolved | Source |
|---|---|---|
| Run spec dir | `docs/specs/` | CLAUDE.md "Doc layout" |
| Feature spec dir | **none** — this project has no feature-spec layer | CLAUDE.md, explicit |
| Plan / report dir | `docs/reports/` | CLAUDE.md |
| Super docs | `docs/prod-chunks.md` (V2-33 section only; shared file), `CLAUDE.md` | handoff ownership table |
| Bug log | **none** (no `BUGS.md`); historical sweep used reports, specs and the research report's failure lists | checked Oct 11 |
| Full test command | `uv run pytest -q`; lint `uv run ruff check .` | CLAUDE.md |
| Codex scripts | `.claude/review-plan.sh`, `.claude/review-audit.sh` (project copies, HEARSAY rubrics) | present, executable |

---

## P1 (frozen)

### Problem

v1 trained its scoring model treating every training clip as equally important. Other teams
reportedly gained from focusing training on the hardest clips, including an RL-driven variant
(organizer feedback, Oct 2026). v1 never did it. The Oct 10 research pilot said it helps, but was
read on a cost convention we no longer trust; re-read on rule 5's reading the pilot's ranking
inverts. We have no trustworthy answer to whether hard-example training helps, or in which
direction to lean it.

### Approach

Pre-declare and commit every recipe, the selection rule and the tie bands before reading any
number. Execute stage 0 on saved embeddings, CPU only, four layers, both row populations, all
three cost readings. Then pre-declare the design of stages 0.5, 1 and 2 — including the RL
(bandit) loop — each gated on resources only Nathan releases (MPS go-ahead, the external drive,
GPU rental). The output is a **recommendation**; adoption waits for a V2-01 benchmark row beside
an unmined twin and a sealed population (rule 7).

### Design decisions

- **D1 — ASVspoof 2019 train/dev policy: measure, don't rule.** The handoff's named P1 question
  ("may these 7,648 rows be mined and scored?") is answered empirically: run **both training
  pools** (with and without them) and read both on the NSA-only basis the benchmark uses.
  Fold assignments agree across the two split files on all 20,000 shared paths (verified Oct 11).
- **D2 — Selection statistic: pooled, one threshold.** Primary statistic is **pooled** package-0.5
  minDCF on inner OOF, because NSA scores one pile with one threshold. The per-fold mean is
  reported as a stability readout only.
  - Evidence for the choice (verified Oct 11, pilot OOFs): the per-fold mean is the only one of
    four statistics under which focal γ=2 ranks first, while pooled, z-pooled and rank-pooled all
    rank it below base. The per-fold mean also ranks OC-softmax mid-field (0.2354) where pooled
    reads 0.5763. Fold-mean grants five private thresholds; deployment and NSA grant one.
- **D3 — The read-once holdout is not spent in stage 0.** Stage 0 produces a recommendation, not
  an adoption decision; the one-shot holdout is reserved for V2-01. ITW and the local MLAAD cohort
  stay report-only (rules 3 and 7). Nathan's note recorded: "spent" is about the decisions made on
  a population, not the bytes — a clean v2 fork that inherits none of v1's choices would largely
  restore In-the-Wild's value, and that is the cheapest route back to a clean evaluation set.
- **D4 — Four layers now; a recipe must hold across layers.** Nathan overrode the original
  recommendation (L7 primary, L6/L8 confirmatory) on the grounds that the test is needed eventually
  and time is not the constraint. Layers **6, 7, 8 and 20** all run. The multiple-comparison
  protection is kept as rule R5: a recipe must pass on at least 3 of 4 layers. Any *layer*
  preference emerging here is recorded as a lead for separate confirmation, never as a selection
  (this is the L7+WavLM trap, `docs/reports/2026-10-03_layer-diversity.md`).
- **D5 — The cost convention is unresolved; stage 0 is built to survive either answer.**
  NSA's document says a false alarm costs 4× a miss (confirmed by Nathan via NSA, Sat Sep 26
  ~11:50, `docs/reports/2026-09-25_sponsor-questions.md:103`). NSA's shipped code puts the 4 on
  `Cfa`, defined in its own comment as *falsely accepting a nontarget speaker* — our miss
  (`data/nsa/HackGTMinDCF/asvspoof5/evaluation-package/calculate_metrics.py:16-21`,
  `calculate_modules.py:111-125`). These are opposite. Rule 5 chose the code. Verified Oct 11: run
  unmodified on our synthetic-high scores the sponsor code returns **1.0**, the worst possible
  value, so the direction was handled somewhere before our 0.0733 came back; that tells us nothing
  about which cost convention was applied.
  - **Consequence for stage 0:** all three readings are reported for every cell, and rule R3 refuses
    any candidate whose advantage does not survive the brief reading. **Adoption is blocked until
    NSA answers.** The question belongs on the list already open in
    `docs/reports/2026-10-10_v2-fusion-research.md`, "Completeness check".
  - Nathan's 70/30 observation is correct and lands on `Pspoof`, a declared weight rather than a
    measurement: `frr` and `far` are per-class rates, so the test set's actual class balance
    cancels out of the arithmetic. Verified Oct 11: moving the prior from 0.5 to 0.3 does **not**
    change the pilot's winner (reals ×4 wins both).
- **D6 — Benchmark ownership.** Nathan's P1 sentence includes "how we are benchmarking everything".
  The benchmark is V2-01, owned by the benchmark + fusion lane, and does not exist yet (no
  `outputs/v2/benchmark/`). This lane defines its outputs to be benchmark-ready and pre-declares
  what it reports; it does not build the benchmark. Default taken, Nathan informed, open to change.

---

## P2 (frozen)

### Failure-mode inventory

Groups A–H below; every row has at least one test. Sources are this project's own incident record.

**A. Load and row alignment** (6 tests) — A1 row misalignment, caught by a *monotone-invariant*
rebuild check against the shipped probe (Spearman == 1.0 and minDCF equal to <1e-12); the pilot's
`max|diff|` check was mis-specified and read 57.4 on an identical ranking because the shipped OOF is
Platt-rescaled (`scripts/bakeoff_layers.py:132-138`; relation verified Oct 11 as
`shipped = 0.34480497 x pilot + 1.99776021`, residual 1e-14). A2 absolute-vs-relative path forms
through the `data/` symlinks silently dropping rows (`scripts/fuse_sweep.py:31-33`). A3 duplicate
paths silently deduped. A4 `decode_error` zero-rows trained and up-weighted as "hard" (the real
99/32,319 WavLM incident). A5 an empty `flag` cell read as NaN so the comparison never matches.
A6 off-by-one on the 25-layer stack.

**B. Fold machinery** (5 tests) — B1 generator or speaker crossing folds, with a clip-level-split
counterfactual. **B2 nested inner-inner folds leaking so a fold mines its own hard examples**, with
an un-nested counterfactual. B3 fold ids disagreeing with the split file. B4 holdout rows in the
inner population. B5 a barred population entering any path (LA eval `asv19_eval_probe`, ITW,
MLAAD, test), enforced by an allow-list.

**C. Recipes and the mining loop** (8 tests; owner's P1 risk) — **C1 weights computed and never
applied**, caught by requiring each recipe at its identity setting (k=1, gamma=0, top-0%) to
reproduce base exactly *and* each active recipe's coefficients to differ from base. **C2 mining the
wrong class**, caught by a directional test. C3 the hand-derived focal gradient wrong, caught by
finite differences to 1e-6. C4 focal gamma=0 not reducing to weighted logistic. C5 OHEM selecting
off the wrong margin sign or pooling classes. C6 the optimizer silently not converging on 1024-dim
near-separable data. C7 a fold model scoring a row it trained on. C8 non-determinism.

**D. Readings and thresholds** (6 tests) — D1 a count taken at the wrong threshold: the same L7
scores give **2,348 FA / 351 miss** at the package-0.5 threshold and **61 / 2,335** at the brief
threshold (both verified Oct 11). D2 `min_cost` drifting from `sponsor_min_dcf`. D3 a reading
mislabeled, caught by weight recovery (1:4, 1:1.714, 9.33:1). D4 a basis mislabeled. D5 pooled and
per-fold columns swapped. D6 a NaN reading comparing as a pass.

**E. The selection gate** (8 tests, one counterfactual per rule) — E1 a rule nothing can fail.
E2 boundary ambiguity. E3 rounding before comparison. E4 a rule satisfiable by one lucky layer.
E5 the convention-robustness rule unenforced. E6 the gate reading a barred population.

**F. Outputs and provenance** (4 tests) — F1 overwriting the pilot in `outputs/v2/research/`.
F2 a stale artifact silently reused, caught by a config hash. **F3 a test writing a tracked file** —
this has happened here: six fabricated rows once landed in the expense ledger. F4 a log row missing
its basis or readings.

**G. Stages 0.5 / 1 / 2 — contracts pre-declared now, built later** (4) — **G1 a silent bug in the
bandit loop**: a synthetic bandit with a known-best arm must converge to it, a constant-reward arm
must never be preferred, and the reward sign is pinned by a directional test. G2 one-sided
augmentation teaching "degraded means real", with a shortcut gate (codec+noise features alone must
give AUC < 0.6). G3 augmented ASVspoof reals scored by models trained on their clean twins.
G4 severity drifting past the SNR >= 5 dB cap.

**H. The definition of "hard"** (6 tests; owner's P2 risk) — H1 the definition baked in, answering
only one version of the question. **H2 the sort running the wrong way so the *easiest* examples are
mined** — one token, and the table still looks plausible. H3 two arms silently collapsing to the
same clips. H4 a mined set that is label noise or one speaker rather than a phenomenon, caught by a
required characterization readout. **H5 confidently-wrong false positives — the case Nathan called
worst — not separately visible**, so they are counted and named in every readout at every threshold.
H6 hedging mining concentrating on one voice.

Measured Oct 11 on the shipped L7 OOF, top 5% per class at the package-0.5 threshold, which is why
H exists: **confidently-wrong and hedging reals share zero clips** (Jaccard 0.000, 0 of 663), and
their source composition differs (confidently wrong: LibriSpeech 252 / ASVspoof 213 / LJSpeech 198
over 57 speakers; hedging: LJSpeech **383** / ASVspoof 179 / LibriSpeech 101 over 70 speakers, and
LJSpeech is a single speaker). On fakes the two largely coincide (Jaccard 0.527, 363 of 526).

### Test commands

```bash
uv run pytest tests/test_hardneg.py -q
uv run pytest -q && uv run ruff check .
```

Baseline suite: **859 tests collected** (Oct 11). The new file contributes ~70; expected ~930 with
zero failures and clean lint.

### Acceptance criteria

1. Every pre-declared cell present: 12 recipes x 4 layers (6, 7, 8, 20) x 2 training pools x
   3 readings x 2 bases, with any N/A cell declared in advance rather than discovered.
2. The base recipe reproduces the shipped L7 probe under the monotone-invariant check (A1).
3. Zero `decode_error` rows, zero all-zero rows, full join coverage asserted (A2-A5).
4. The gate verdict computed, survivors named, and each survivor's status stated under all three
   readings (R1-R8).
5. Every mining arm reports its characterization readout, and confidently-wrong false positives are
   counted separately at each threshold (H4, H5).
6. Run spec and report written; one `experiments/log.csv` row whose `notes` names the basis and all
   three readings.
7. `outputs/v2/research/` byte-identical to before the run.

### Regression

Any of: a previously-passing test fails; `ruff check .` regresses; the research directory changes;
any tracked file is modified by the test suite; the base recipe stops reproducing the shipped probe.

### Test break check

**Candidate: H2** — the mined set is the hardest, not the easiest. Mutation: flip the sort direction
so the selection takes the easiest examples. One token, entirely realistic, and it would still
produce a believable table, which is exactly the owner's stated risk. Alternate if implementation
makes it the better fit: **C2** (wrong class up-weighted).

**Record (Sat Oct 17, post-commit `7ba7faf`).** The design of record moved the check from H2 to
**C2** (FINAL DESIGN, "Test-design changes": flipping the sort *improves* the table under the primary
reading, so that mutation corrupts nothing visible). The frozen paragraph above stands as written.

- **Test:** `tests/test_hardneg.py::test_c2_a_shape_applied_to_one_class_moves_that_classs_error_rate`,
  three seeds. It did not exist at run 1 (the inventory named it; the file had H2, C1, C3, C4, C7, C9)
  and was written at close-out, **before** this check ran. It fits two Gaussian classes with the real
  arms and asserts that `hardest_reals` cuts the false-alarm rate (trading misses) and
  `hardest_fakes` cuts the miss rate (trading false alarms), at the decision threshold.
- **Mutation:** one line in `select_picks`, the class mapping
  `{"both": (0, 1), "reals": (0,), "fakes": (1,)}` → `{"both": (0, 1), "reals": (1,), "fakes": (0,)}`,
  so a one-sided arm up-weights the other class. Realistic and silent: the grid would still run
  to completion and print a plausible table in which `hardest_reals` is a hardest-fakes recipe.
  This is the owner's P1 risk ("a silent bug in the loop") in its literal form.
- **Where:** a detached scratch worktree of `7ba7faf` under the session scratchpad, outside the repo
  tree (the fusion lane's in-tree worktree once broke the docs-site tests). Never in the checkout.
- **Result:** baseline 3 passed; against the mutation **3 failed on the first attempt (CAUGHT)**:
  on every seed the mutated `hardest_reals` failed to cut false alarms and the mutated
  `hardest_fakes` failed to cut misses. The whole lane file against the mutation: 5 failed, 85
  passed, 4 `needs_data` skipped. The other two failures: `test_h2_side_selects_only_that_class`
  (expected, the per-class pick counts land on the wrong class) and
  `test_f2_fingerprint_changes_when_a_score_changing_input_changes`, which fails in *any* checkout
  without the gitignored embedding manifests the fingerprint hashes, so it is unrelated to the
  mutation; it is now guarded with the file's `HAVE_EMB` probe.
- **Fix-and-recheck rounds:** none needed.
- **Cleanup:** `git worktree remove --force`; `git worktree list` shows only the main checkout and the
  two other lanes' worktrees; `git log -S'"reals": (1,), "fakes": (0,)' --all -- scripts/v2_hardneg_bakeoff.py`
  returns nothing, so the mutation is in no commit.

**Round 2, the gate layer (Sat Oct 17, post-commit `ce60219`).** The Claude critique (finding 13)
observed that all three close-out defects sat in the gate and readout layer, where no mutation had
been tried, and that a mutation making `R1_indeterminate` irrelevant would have been caught by
nothing. After the audit-round-1 fixes:

- **Test:** `tests/test_hardneg.py::test_e_r1_indeterminate_is_not_a_pass_and_not_r5_eligible`, the
  test most tied to the gate change (it asserts the *verdict*, where the earlier test asserted only
  the flag).
- **Mutation:** one token in `gate_cell`: `elif r["R1_indeterminate"]:` → `elif False:`, so the
  indeterminacy clause never fires and a P(better) of 0.904 passes again. Realistic and silent: the
  grid, the report and every other rule are unaffected; the headline would read 9 passes.
- **Where:** a detached scratch worktree of `ce60219` under the session scratchpad, outside the repo.
- **Result:** baseline 1 passed; against the mutation **1 failed, first attempt (CAUGHT)**. Whole lane
  file against the mutation: 2 failed, 101 passed, 5 skipped; the second failure,
  `test_f3_fingerprint_covers_shard_contents_not_just_the_manifest`, fails in any checkout without
  the gitignored `outputs/v2/bakeoff/rows.json` the fingerprint hashes, unrelated to the mutation,
  and now carries the file's `HAVE_BAKEOFF` skip guard (same class as the round-1 `test_f2` finding).
- **Fix-and-recheck rounds:** none needed.
- **Cleanup:** worktree removed; `git worktree list` shows only the main checkout and the two other
  lanes' worktrees; `git log -S'elif False:  # mutated' --all -- scripts/v2_hardneg_bakeoff.py`
  returns nothing.

---

## Technical spec

### The grid

**Recipes (12).** base; cost-weighted reals xk; cost-weighted fakes xk; focal gamma=2;
OHEM x 3 mining definitions; nested mine-and-retrain x 3 mining definitions; source-balanced;
speaker-balanced.

**Mining definitions (3), the new pre-declared axis.** `conf_wrong` (worst signed margin);
`hedging` (smallest absolute distance to the operating threshold); `wrong_at_thr` (misclassified at
the pre-declared threshold). Applied per class and reported per class.

**Layers (4).** 6, 7, 8, 20 — all four probes exist under `outputs/v2/bakeoff/probes/`.

**Training pools (2).** `pool_full` = 23,790 rows (NSA + ASVspoof 2019, `splits/nsa_folds_plus_asv19.csv`);
`pool_nsa` = 16,142 NSA-only rows (`splits/nsa_folds.csv`). Verified Oct 11: the NSA-only subset of
the inner population is exactly 16,142 rows, and f4 remains the ElevenLabs fold on both
(pool_full f4: 2,647 bona fide + 1,000 ElevenLabs + 420 A04; pool_nsa f4: 1,632 + 1,000).

**Readings (3).** package-0.5 `min_cost(y, s, 0.5, c_fa=1.0, c_miss=4.0)` — primary;
package-0.3 `min_cost(y, s, 0.3, c_fa=1.0, c_miss=4.0)`; brief `min_cost(y, s)`. EER beside them.

**Bases for reading (2).** The full 23,790 rows and the NSA-only 16,142. Primary basis for
selection: **NSA-only**, matching what the V2-01 benchmark will use (rule 2).

### The gate (pre-declared, with bands)

A candidate is a (recipe, mining-definition) pair, evaluated per layer and per training pool.

| Rule | Statement |
|---|---|
| **R1** | Pooled package-0.5 on the primary basis beats base, with speaker-cluster bootstrap P(better) >= 0.90 |
| **R2** | Pooled package-0.3 also beats base. Both package readings must agree in direction |
| **R3** | Brief does not regress beyond **0.010** vs base (convention robustness, D5) |
| **R4** | Fold-4 (ElevenLabs) package-0.5 not worse than base's f4 by more than **0.010** |
| **R5** | R1-R4 hold on at least **3 of the 4** layers. Passing on 1-2 layers is a lead, not a survivor (D4) |
| **R6** | Inner false alarms at the pooled package-0.5-optimal threshold <= base + **10% relative** (base = 2,348 on the full basis) |
| **R7** | Any mined set: no single speaker > **25%** of the set; characterization readout present; confidently-wrong false alarms counted (H4-H6) |
| **R8** | All integrity asserts pass: zero `decode_error`, zero all-zero rows, full join coverage, nested-fold disjointness, no NaN reading (A, B, D6) |

**Primary question:** does any recipe survive R1-R8, and in which direction (real-side or
fake-side emphasis). **Secondary question, reported in full but not permitted to generate the
headline claim:** which definition of "hard" works best.

Bands are declared here and are not to be revised after any number is read. The bootstrap in R1
follows the existing speaker-cluster pattern (`frac_better` in
`outputs/v2/research/hard-negative-rl/oneclass.json`).

### Files

**Created by this lane:**

| Path | What |
|---|---|
| `scripts/v2_hardneg_bakeoff.py` | The bake-off: load, fit, score, gate, write |
| `tests/test_hardneg.py` | ~70 tests, groups A-H |
| `outputs/v2/hardneg/` | Results table, per-candidate OOFs, characterization readouts, config hash |
| `docs/specs/2026-10-13_v2-33-hardneg-stage0.md` | This run spec |
| `docs/reports/2026-10-13_v2-33-hardneg-stage0-plan.md` | Implementation plan (Codex-reviewed) |
| `docs/reports/2026-10-13_v2-33-hardneg-stage0.md` | Results and readouts |

**Edited:** `docs/prod-chunks.md` V2-33 section only, committed only when the diff shows only this
lane's hunks. `experiments/log.csv`, one appended row. `CLAUDE.md` AI-use disclosure — conditional,
confirmed at execution.

**Read-only, never written:** `models/`, `submissions/`, `splits/`, `outputs/detector_scores/`,
`outputs/embeddings/`, `outputs/v2/research/`, `outputs/v2/bakeoff/`, `outputs/v2/layer_diversity/`,
`src/hearsay/metrics.py`, `src/hearsay/augment.py`, `scripts/extract_embeddings.py`, the hackathon
scripts, and the other two lanes' files. `scripts/train_probe.py` and
`scripts/export_probe_scores.py` are never run. The five uncommitted files belonging to other lanes
(`.claude/codex-audit-prompt.md`, `.claude/launch.json`,
`docs/specs/2026-09-26_m5-xlsr-finetune-audit.md`, `next-env.d.ts`, `tsconfig.json`) are left alone.

**The pilot is never re-run in place.** `outputs/v2/research/hard-negative-rl/hn_*.py` write into
their own directory and would overwrite the pilot. Their logic is copied into this lane's script.
Backup confirmed present: `~/hearsay-v2-research-2026-10-10.tgz`, 33 MB, Oct 11 01:16.

### Compute and constraints

CPU only. n_jobs <= 4 (18 GB Mac, shared with two other lanes). No MPS, no torch, no GPU, no
downloads, no new audio. No process is stopped except by explicit PID. Pilot timing: 5 recipes x
1 layer x 5 folds took 74 s, so the full grid is estimated at 45-90 min. Time box 3 h, then stop
and report. Verified Oct 11: no MPS or extraction jobs running; ~16 GiB free on the home volume,
down from the 27-29 GiB the handoff records for Oct 10 — relevant to stage 1's 12.3 GB of downloads,
not to stage 0.

---

## Results

Full write-up: `docs/reports/2026-10-13_v2-33-hardneg-stage0.md`. Everything below is the
execution record; the numbers are in the report, and `outputs/` is gitignored.

### Execution record

- **Run 1, Mon Oct 13, 17:11 to 19:38 EDT.** `fit --jobs 4` (132 of 132 declared cells in 18 min,
  131 OK, 1 INVALID: `focal_g2` L7 `pool_full` C=64, gradient norm 0.0112 > 0.01 on fold 2), then
  `csweep` (64 rows, complete), `gate`, `report`. Self-check passed: Spearman exactly 1.0 against
  the shipped L7 probe, package-0.5 gap 0.0. Persisted 3 h deadline not reached. Outputs preserved
  unchanged at `outputs/v2/hardneg_run1_20261013/` before run 2.
- **Close-out, Sat Oct 17.** Writing the report from run 1's outputs found three defects in the
  readout and gate code (none in a fit, a score or a band). Fixed, tested, and the deterministic
  grid re-run as **run 2** with the committed script; the gate re-run; the C sweep not re-run
  (nothing in the fixes touches it). The byte-identity of run 2's OOF scores against run 1 is the
  report's "Run record" row (all 131 OK cells byte-identical in `oof` and `mask` and identical in every
  primary reading; the one INVALID cell, which writes no score file, INVALID for the same reason; 132 of
  132 agree).
- **Run 3, Sat Oct 17 21:36 EDT, interrupted** after 64 cells at 21:40 (stopped by PID) when the Claude critique
  arrived with fixes that touch the script, which the whole-file fingerprint would have turned into
  "stale" checkpoints mid-run. Log kept as `fit_run3_interrupted.log`.
- **Run 4, Sat Oct 17, 21:43 to 22:12 EDT, final.** After the audit-round-1 fixes, under the
  source-scoped, shard-hashed fingerprint: 131 computed, 0 reused, 1 INVALID (same cell, same reason),
  868 s; self-check Spearman 1.0, gap 0.0; the validated gate 21:58 to 22:12 (14.5 min); persisted 3 h
  deadline not reached. Identity against run 1 as above: 132 of 132 agree. Verdicts: 24 BASE, 93 FAIL,
  **8 PASS, 6 INDETERMINATE**, 1 INVALID; no R5 survivor.
- **Run 4b, Sat Oct 17 22:40 to 22:54 EDT, gate only.** After the Codex round-2 fixes to the gate's validation
  layer (outside `FINGERPRINT_SOURCES`, so every run-4 checkpoint stays valid): gate 22:40 to 22:54 EDT; `report.json` identical to run 4's in every key (gate, contrasts, H3, H5, base reference, invalid reasons; run 4's copy preserved as `report_run4_20261017.json` beside run 1), verdicts 24 BASE, 93 FAIL, 8 PASS, 6 INDETERMINATE, 1 INVALID. This is the report the results describe.

- **Pre-run commit record: missing** (Codex audit round 1, #6). The declaration was frozen in files on disk
  on Oct 13 but not committed until Oct 17; recorded as a limitation, see the report's disclosures.

### Defects found at close-out (details and tests in the report's defects table)

| # | Defect | Run-1 consequence | Disposition |
|---|---|---|---|
| 1 | R7 / H4 computed group concentration over the basis population, not the mined set; no mined-set characterisation or H3 overlap existed | R7 could never fail (E1); every cell read 0.0316 / `lj:LJ028` | Fixed: picks stored per fold, `characterise()` on the mined rows, R7 reads it, H3 Jaccard added; re-run |
| 2 | R4a/R4b looked up `int` fold keys on JSON-loaded cells (keys are strings after `clean()` + `json`) | R4a and R4b failed in **all 107** non-base cells, 13 on R4 alone; run 1's verdict column is wrong wherever R4 decided | Fixed: keys normalised; round-trip test; gate re-run |
| 3 | H5 "confidently wrong" as frozen saturates whenever false alarms cover the real decile | 815 in every NSA-basis cell, 1,327 on the full basis; no information | **Not redefined** (frozen); reported as saturated; `report.json` records the check; stage 0.5 item |

Also found: the designated break-check test **C2 did not exist** in the committed test file
(the inventory named it; the file had H2, C1, C3, C4, C7, C9). Written at close-out as
`test_c2_a_shape_applied_to_one_class_moves_that_classs_error_rate` (three seeds) before the
break check ran. The characterisation's promised **duration** field is unavailable: the embedding
manifests carry no duration column and this lane may not read audio.

### Acceptance criteria, checked

| # | Criterion | State |
|---|---|---|
| 1 | Every pre-declared cell present; N/A declared in advance | 132 / 132; `pool_nsa` x full basis N/A by declaration |
| 2 | Base reproduces the shipped L7 probe under the monotone-invariant check, and the check **blocks** | Spearman 1.0, gap 0.0, both runs; `self_check` raises `SystemExit` |
| 3 | Zero `decode_error`, zero all-zero rows, full join coverage | Enforced by `load_layer` / `load_world` guards; both runs loaded 23,790 inner rows in `bakeoff/rows.json` order |
| 4 | Gate verdict computed, survivors named, status under all three readings | Computed for all 132 cells (`outputs/v2/hardneg/report.json`, run 4): 24 BASE, 93 FAIL, **8 PASS, 6 INDETERMINATE**, 1 INVALID. No (arm, pool, C) passes R5 (each pass is on one layer, never L20), so **no survivor**; the eight single-layer passes, one of them the random control, and the six indeterminate cells are named in the report's gate section with all three readings, fold 0 / fold 4, false alarms and P(better) ± SE. The primary answer is the per-C shared-resample aggregate of the four pre-declared contrasts (report, "The primary answer") |
| 5 | Every mining arm reports its characterisation; confidently-wrong FA counted per cell | Characterisation present for all 102 cells that mine rows after the fix (the five OK `focal_g2` cells mine nothing by construction); H5 counted but saturated (defect 3) |
| 6 | Run spec and report written; one `experiments/log.csv` row naming basis and readings | This section; the report; rows appended by `v2_hardneg_bakeoff.py log`: 2026-10-18T01:13:15+00:00 (the first commit's gate: 9 passes, C-pooled means) and 2026-10-18T02:13:15+00:00 (the final gate: 8 passes, per-C aggregate P(better) for both primary contrasts), the second superseding the first; both carry `validation_score` 0.240507 (the reference cell `none` L7 `pool_nsa` C=1 on the NSA-only basis), all three readings, EER, cell counts and `benchmark_version` "none (pre-V2-01)". The log is append-only, so the first row stays |
| 7 | `outputs/v2/research/` byte-identical | 48 MB, no file newer than run 1's start; the test suite's autouse guard covers the directory |

`CLAUDE.md` AI-use disclosure: **conditional not triggered.** The lane uses no new pretrained
model, dataset or tool, and the other v2 lanes have not added by-lane bullets, so the disclosure is
unchanged.

### Verification before commit (plan Step 9)

Round 1 (commit `7ba7faf`, Oct 17 ~21:20 EDT):

```text
uv run pytest tests/test_hardneg.py -q      -> 94 passed (82 at run 1; +12 written at close-out)
uv run pytest -q                            -> 1234 passed, 3 skipped (needs_weights / MPS, not this lane's)
uv run ruff check .                         -> All checks passed
grep -nE "9\.3333|min_cost\(|c_fa=4|PI_SYNTH|class_weight=.balanced.*sample_weight" scripts/v2_hardneg_bakeoff.py
                                            -> hits only in the comment and the three named wrappers
                                               brief / package05 / package03, and the W1 docstring
git diff --stat docs/prod-chunks.md         -> one hunk, lines 518-523, the V2-33 status block only
shasum -a 256 splits/*.csv                  -> nsa_folds a607d3ec…, nsa_folds_plus_asv19 7bcd377d…, unchanged
du -sh outputs/v2/research                  -> 48M; no file newer than run 1's start (find -newer)
```

Round 2 (after the audit-round-1 fixes and run 4):

```text
uv run pytest tests/test_hardneg.py -q      -> 108 passed (+14: the audit dispositions' tests)
uv run pytest -q                            -> 1248 passed, 3 skipped, 2 warnings, 6 min 32 s
uv run ruff check .                         -> All checks passed
the min_cost grep gate                      -> now a test: test_a1_min_cost_is_called_only_inside_the_three_named_wrappers
git diff docs/prod-chunks.md                -> the V2-33 status block only (lines 518-523)
outputs/v2/research                         -> 48M, untouched through runs 2 to 4
```

Round 3 (after the Codex round-2 and critique round-2 fixes):

```text
uv run pytest tests/test_hardneg.py -q      -> 114 passed (+6: the round-2 dispositions' tests)
uv run pytest -q                            -> 1254 passed, 3 skipped, 2 warnings, 6 min 40 s
uv run ruff check .                         -> All checks passed
```

The docs-site test `tests/test_build_site.py::test_every_page_has_shell_invariants` had been failing
on this lane's uncommitted plan and spec (more than one `#` heading each, rendered as two `<h1>`); the
extra headings are now `##` and the test passes. The five other-lane modified files
(`.claude/codex-audit-prompt.md`, `.claude/launch.json`, `docs/specs/2026-09-26_m5-xlsr-finetune-audit.md`,
`next-env.d.ts`, `tsconfig.json`) are left uncommitted, as the Files section says.

## Review scorecards

### Codex audit, round 1 (Sat Oct 17, on `5820143`): Overall **Fail**

Artifact: `docs/specs/2026-10-13_v2-33-hardneg-stage0-audit.md` (round 1 copy preserved as
`…-audit-round1.md`).

| Dimension | Grade | Codex's note |
|---|---|---|
| Plan adherence | Fail | Primary analysis and gate differ materially from the final design |
| Scope discipline | Excellent | Changes stay within lane-owned files |
| Test coverage | Fail | Passing tests miss required gate behaviour and aggregate inference |
| Review compliance | Fail | Accepted findings #2, #3 and #8 remain incompletely implemented |
| Freeze integrity | Acceptable | No freeze hashes; owner statements present; pre-run frozen text cannot be independently verified |
| Regression check | Acceptable | 1,234 passed recorded; 26 targeted tests and lint verified independently |
| Documentation | Fail | Report and status propagate an incorrect pass count and unsupported aggregate conclusions |
| Test sensitivity | Excellent | C2 mutation caught first time across three seeds; mutation absent from history |

**All six findings accepted.** Dispositions, each with its test:

| # | Finding | Fix | Test |
|---|---|---|---|
| 1 | The primary contrast averaged all 24 strata, C included, while claiming "C never averaged over"; no replicate-level aggregate or aggregate uncertainty | `aggregate_contrast()`: per C, the (layer, pool) strata are averaged **inside** each shared group-resample (same draw plan and seed as the per-stratum bootstraps), giving P(better), its SE, median, p5/p95 per C. The all-strata mean is kept as `mean_delta_package05_all_strata`, labelled descriptive. Report, status and log row re-stated on the per-C aggregates | `test_gate_contrast_is_aggregated_per_c_inside_shared_resamples` (on one stratum the aggregate must coincide with that stratum's own bootstrap) |
| 2 | An R1-indeterminate cell could still PASS (`easiest_both` L7 `pool_nsa` C=8 at 0.904) | `gate_cell` returns **INDETERMINATE** when R1 is within 2 SE of the band and no other rule fails; R5 counts PASS only. **Interpretation stated:** the clause is applied symmetrically, so a cell below 0.90 but within 2 SE is INDETERMINATE rather than FAIL (five such cells; FAIL 98 → 93, PASS 9 → 8) | `test_e_r1_indeterminate_is_not_a_pass_and_not_r5_eligible` |
| 3 | Fingerprint hashed manifests but not shard contents; npz writes non-atomic; the gate never validated fingerprints nor refused an incomplete grid; a truncated checkpoint JSON aborted the whole fit at resume (Claude critique #5) | `shard_digest()` hashes manifest **and** every shard (cached per process, keyed on size and mtime); the script's own contribution to the fingerprint is now the **source of the fitting-path functions** (`FINGERPRINT_SOURCES`, taken from the file's AST) rather than the whole file, so a gate or report edit no longer forces a refit while a fitting change still does; `atomic_savez()`; `reusable_checkpoint()` treats a stale, truncated or npz-less checkpoint as "recompute", never as an abort; `validate_cells()` at gate time refuses stale (fingerprint mismatch), unreadable / truncated and incomplete-coverage checkpoints as INVALID with the reason; `stage_gate` refuses a partial grid outright (no verdicts, `refused` recorded, exit 2) | `test_f3_fingerprint_covers_shard_contents_not_just_the_manifest`, `test_f3_fingerprint_hashes_the_fitting_path_source`, `test_f3_resume_rule_recomputes_a_stale_or_truncated_checkpoint`, `test_f3_atomic_savez_writes_whole_or_not_at_all`, `test_gate_refuses_a_stale_checkpoint`, `test_gate_refuses_a_truncated_checkpoint`, `test_gate_refuses_an_incomplete_grid_instead_of_gating_it` |
| 4 | R9 degeneracy was an ordinary FAIL with `status = OK`, still eligible for contrasts; no constant-score refusal; saturation not checked on every reading; a null reading after the JSON round-trip raised `TypeError` in the gate (Claude critique #14) | `degenerate_reason()` centralises R9 (non-finite or null, **constant**, AUC < 0.5, any of the three readings saturated, on every basis); applied at fit (cell written INVALID) and again at gate from the checkpoint; INVALID cells never enter a contrast; `gate_cell` reads every number through a null-safe accessor | `test_r9_degenerate_reason_covers_constant_scores_and_every_reading`, `test_gate_marks_a_degenerate_cell_invalid_and_excludes_it_from_contrasts`, `test_e_r9_null_reading_after_the_round_trip_is_refused_not_a_crash` |
| 5 | The gate never consulted the persisted deadline | `stage_gate` ticks the deadline before every bootstrap; on expiry it keeps what it has, records `incomplete` and claims no completeness; `main` exits 2 | `test_gate_honours_the_persisted_deadline` |
| 6 | No pre-run commit of the declaration: the script and spec were first committed Oct 17, after both runs | **Recorded as a limitation**, here and in the report's disclosures. The pre-declaration's integrity rests on the file history on disk and on the preserved run-1 outputs, not on a commit. Any confirmation run (stage 0.5, V2-01 row) commits its declaration before execution | none (process rule) |
| (Codex #10 / critique #7) | The `min_cost(` grep gate was a manual check, not a test | Pinned as an AST test: every `min_cost` call must sit inside `brief`, `package05` or `package03` | `test_a1_min_cost_is_called_only_inside_the_three_named_wrappers` |

The fixes to #1 to #4 change what the gate and the contrasts *compute*, not any fit, so the grid was
re-run as **run 4** under the new fingerprint (which the shard digests change; run 3 was interrupted,
see the execution record) and the OOF scores checked against run 1 again; the re-gated numbers are in
the Results section and the report.

### Claude critique, round 1 (Sat Oct 17, on `5820143`, same rubric): Overall **Fail**

Run as a background agent with the rubric, the commits and the outputs, instructed to recompute
rather than trust. It verified every number it could recompute in the report against `report.json`
and the cells (Appendix A, C, the stratum table, the nine-pass table, the contrasts, the rule-fail
counts, the H3 and mined-set ranges, the C sweep, the base reference, the H5 saturation, the log
row), reproduced the break check on a scratch copy (3/3 pass, 3/3 fail under the mutation), and
confirmed the three run-1 defects and their fixes.

| Dimension | Grade | Critique's note |
|---|---|---|
| Plan adherence | Fail | R1 "within 2 SE is indeterminate, not a pass" not applied to the verdict; no aggregate shared-resample uncertainty for the primary answer, stratum counts used instead |
| Scope discipline | Excellent | Write list respected; `prod-chunks.md` one hunk; research dir unchanged |
| Test coverage | Acceptable | 94 passed; no test that an indeterminate cell is withheld from PASS; no stale / truncated resume test |
| Review compliance | Acceptable | #3, #8, #10 partially delivered |
| Freeze integrity | Acceptable | Owner's statement in the owner's voice; pre-commit text not independently verifiable |
| Regression check | Acceptable | Lane tests, lint, docs-site test pass; one well-formed log row |
| Documentation | Acceptable | Every recomputed number matches; cosmetic defects (findings 8-12) |
| Test sensitivity | Excellent | Reproduced; caveat that the gate layer itself had no mutation check |

Findings and dispositions (its numbering):

| # | Finding | Disposition |
|---|---|---|
| 1 | Indeterminate R1 still PASSes (`easiest_both` L7 `pool_nsa` C=8 at 0.904); headline is 8 + 1 indeterminate, not 9 | Same as Codex #2; fixed (`INDETERMINATE` verdict, R5 counts PASS only); counts restated after the re-gate |
| 2 | No aggregate bootstrap with uncertainty; stratum counting substituted. Its own shared-resample aggregate over all 24 strata: easiest − none P(better) 0.993, 95% CI of the mean Δ [−0.038, −0.0005]; hardest − none 0.242 | Same as Codex #1; fixed per C (`aggregate_contrast`), which is the pre-declared unit; the all-strata figure is kept as descriptive. The critique's point that the easiest-side interval nearly touches zero is carried into the report's reading of the result |
| 3 | "132 of 132 byte-identical" overclaims: the INVALID cell writes no score file, so it is 131 npz plus one INVALID with the same reason | Wording fixed in the report; the commit message of `7ba7faf` cannot be changed and is noted here |
| 4 | H5 saturation was knowable at freeze time from the spec's own reference FA rates (17.7%, 20.2% > 10%) | Acknowledged in the report's H5 section |
| 5 | A truncated checkpoint JSON aborted the whole fit at resume; no stale / truncated tests | `reusable_checkpoint()` + tests (see Codex #3 row) |
| 6 | Deadline not consulted by the gate; partial grid still gated | Same as Codex #5 and #3; fixed |
| 7 | `min_cost` grep gate not pinned as a test | AST test added |
| 8 | Report test count 82 → 91 stale | Fixed (94, with the C2 and F4 tests named) |
| 9 | "107 mined cells" is 102 (five `focal_g2` cells mine nothing) | Fixed in report and spec |
| 10 | Appendix D one-sided rows mis-rendered | Checked after regeneration from run 4 |
| 11 | SE range 0.006-0.016 is 0.0054-0.0158 | Fixed (0.005 to 0.016) |
| 12 | Disclosure conditional and log row confirmed correct | none |
| 13 | No mutation check on the gate layer; a mutation making `R1_indeterminate` irrelevant was caught by nothing | The new verdict test asserts the verdict, not the flag; a gate-layer mutation check is run and recorded under "Test break check, round 2" |
| 14 | A null reading after the JSON round-trip raises `TypeError` in `gate_cell` | Null-safe accessor; `degenerate_reason` refuses null readings; test added |
| 15 | Serial grid instead of the plan's `Parallel` pool; INVALID strata "counted" only implicitly | Justified deviation (one LR fit is 0.05 s); `invalid_reasons` now lists every excluded cell in `report.json` |

### Codex audit, round 2 (Sat Oct 17, on `ba19d54`): Overall **Fail**, two findings

Artifact preserved as `…-audit-round2.md`.

| Dimension | Grade | Codex's note |
|---|---|---|
| Plan adherence | Fail | Checkpoint integrity requirements remain incomplete |
| Scope discipline | Excellent | Lane-owned files only |
| Test coverage | Fail | Tests miss incomplete score masks and truncated checkpoint JSON at gate time |
| Review compliance | Fail | Accepted finding #3 only partially resolved |
| Freeze integrity | Acceptable | No hashes; pre-run declaration cannot be verified from git |
| Regression check | Acceptable | 31 targeted tests and lint passed independently; 132 cells match current fingerprints; 131 OOF/mask arrays unchanged vs run 1; verdict counts match |
| Documentation | Acceptable | Results match saved artifacts; limitations disclosed |
| Test sensitivity | Excellent | Both mutation catches reproduced; neither mutation in the change's commits |

Codex's own summary: "Remaining defects concern checkpoint refusal, not evidence that the saved
results are wrong." Both accepted and fixed:

| # | Finding | Fix | Test |
|---|---|---|---|
| 1 | `validate_cells` trusted the checkpoint's own mask: dropping one row from the mask (score NaN) still read OK, so the gate could bootstrap a different population from the cached readings | The authoritative pool masks are rebuilt from `rows.json` (`pool_masks()`), and a checkpoint is INVALID unless its mask is boolean, one-dimensional, the right shape and **exactly equal** to the pool's | `test_gate_refuses_a_checkpoint_whose_mask_is_not_the_pools` (one row dropped; and a non-boolean mask) |
| 2 | `load_cells` parsed every JSON before validation, so a truncated cell JSON aborted the gate with no refusal report | `load_cells` turns an unparseable checkpoint into an INVALID cell keyed from its filename (`key_from_name`) with the reason; the gate then refuses it like any INVALID cell | `test_gate_turns_a_truncated_checkpoint_json_into_an_invalid_cell`, `test_key_from_name_round_trips_every_declared_cell` |

These touch only the gate's validation layer, which is outside `FINGERPRINT_SOURCES`, so the
run-4 checkpoints stay valid and only the gate was re-run (run 4b); every number is unchanged.

### Claude critique, round 2 (Sat Oct 17, on `ba19d54`): Overall **Acceptable**

Same adversarial brief as round 1, asked to check every round-1 disposition. It recomputed the per-C
aggregates from the npz scores with the shared resample plan (identical to `report.json` to 6 dp),
the per-stratum bootstrap for the formerly-passing cell (0.903904 ± 0.009325), every verdict,
R5, the run-1 identity (131/131 OK arrays), both break checks on scratch copies, and every number
in the Verdict, primary-answer, gate and L7 sections, Appendices A/B/D, prod-chunks and both log rows.

| Dimension | Grade | Note |
|---|---|---|
| Plan adherence | Acceptable | All six round-1 dispositions implemented and verified; pre-run commit still missing (recorded); symmetric indeterminacy reading is a post-read interpretation |
| Scope discipline | Excellent | |
| Test coverage | Acceptable | Two disposition tests could not fail on the defect they name; truncated npz at resume untested |
| Review compliance | Acceptable | All dispositions delivered as described except the "truncated" clause of #3 (JSON only) |
| Freeze integrity | Acceptable | |
| Regression check | Acceptable | |
| Documentation | Acceptable | Cosmetic errors only |
| Test sensitivity | Excellent | Both mutations reproduced, caught first time, absent from history |

Findings and dispositions (its numbering):

| # | Finding | Disposition |
|---|---|---|
| 1 | "four at L6, four at L7" is five and three | Fixed |
| 2 | Run 3 started 21:36 (not 21:58 / 21:43); run 2's deadline file is 20:46; spec said "re-run as run 3" | Fixed in report and spec |
| 3 | SE range over the 96 strata is 0.0041-0.0158; test count stale | Fixed |
| 4 | The per-C aggregation test used one C and could not catch a reversion to pooling | Two-C fixture: `test_gate_aggregates_per_c_and_never_pools_c` (hardest beats base at C=1, loses at C=8; the two aggregates must differ) |
| 5 | The fitting-path digest test asserted membership only | `test_f3_fitting_path_digest_reacts_to_a_fitting_edit_and_ignores_a_gate_edit` (a one-token edit in `shaped_weights` changes it; a comment in `gate_cell` does not) |
| 6 | `reusable_checkpoint` reused a truncated npz; only validation caught it later | It now loads the npz and recomputes on any failure; `test_f3_resume_rule_recomputes_a_truncated_score_file` |
| 7 | No pre-run commit (already recorded) | none |
| 8 | The symmetric reading of R1 indeterminacy is a post-read interpretation and the FAIL count change (98 → 93) was not stated | Stated in the report's gate section and in the Codex #2 disposition row |
| 9 | Both log rows verified | none |
| 10 | Speaker-disjointness not met; conclusions correctly labelled diagnostic | none |
| 11 | Both break checks reproduced | none |
| 12 | Every regenerated table matches | none |

### Codex audit, round 3

_Run after the round-2 fixes were committed; recorded below._

---

## Phase 3 additions and corrections (Oct 13, deep-mode exploration)

Added after P2 froze, by the three exploration agents plus local verification. **No candidate has
been fitted and no candidate number has been read**, so the bands below are still pre-declared; the
corrections are to reference values and rule construction, not to results. The frozen P1/P2 text
above is unedited. Where a rule here conflicts with the earlier gate table, **this section
supersedes it.**

## Verified reference values (base = L7, shipped probe, package-0.5)

| | n | reals | fakes | pkg-0.5 | threshold | FA | miss | FA rate |
|---|---|---|---|---|---|---|---|---|
| `pool_full` (23,790) | 23,790 | 13,270 | 10,520 | 0.3104 | −1.4414 | 2,348 | 351 | 17.69% |
| `pool_nsa` (16,142) | 16,142 | 8,142 | 8,000 | **0.3307** | −1.4414 | **1,642** | 258 | 20.17% |

`pool_nsa` 0.3307 matches the handoff's independently recorded export figure exactly.

Per-fold package-0.5 on `pool_nsa`: f0 **0.4735**, f1 0.0503, f2 0.0067, f3 0.0012, f4 **0.3950**.
**Fold 0 is harder than fold 4 under the primary reading** (the brief reading has them near-tied at
0.5432 / 0.5424). Folds 2 and 3 are at the floor on both readings, which is the quantitative reason
D2 rejects the per-fold mean: that statistic is nearly a linear function of f0 and f4 alone, and
would double-count the ElevenLabs fold while making f1-f3 movement invisible.

Base f0/f4 package-0.5: `pool_full` 0.4696 / 0.3450; `pool_nsa` 0.4735 / 0.3950.

## Verified structural facts that change rule construction

- **Class balance differs by pool.** `pool_full` 13,270 real / 10,520 fake (ratio 1.2614);
  `pool_nsa` 8,142 / 8,000 (ratio 1.0177, nearly balanced). `make_clf` hard-codes
  `class_weight="balanced"`, and sklearn **multiplies** that by `sample_weight`, so the same nominal
  k is a materially different intervention on each pool. Declared, not corrected.
- **The group-weighting arms change definition between pools.** Real sources 3 → 2; real speakers
  103 → 63; LJSpeech's share of reals 38.55% → **62.82%**, all under the single `speaker` token
  `lj`. So "source-balanced" gives LJ 1/3 of real weight on one pool and 1/2 on the other, and
  "speaker-balanced" crushes 62.8% of reals to 1/63 on `pool_nsa`. This is the shortcut-ledger entry
  "LJ single speaker — one voice on both sides of every fold" (`docs/architecture.md:267`) and it is
  why speaker-balanced was the worst pilot variant (f4 0.5059).
- **Bootstrap units must be the fold file's `group` column, not `speaker`.** 158 inner groups
  (`spk:` / `gen:` prefixed). LJSpeech is **one** speaker token but **42** groups spread across all
  five folds. Fold 4's spoof side is 40 speaker tokens but **2 groups** on `pool_full` and
  **1 group** on `pool_nsa`. Clustering by speaker would resample fake clips as ~20x more
  independent units than the fold design admits, and make LJ a single all-or-nothing real cluster.
- **`cv_groups()` does not reproduce the fold file's `group` column** — 117 groups vs 158. It splits
  no fold-file group (a strict coarsening), so it is safe for leakage but is a *different*
  partition. A fold-integrity test written against `cv_groups()` would not be testing the fold file.
  **B1 checks the `group` column.**
- **`min_cost(y, s, 0.5)` with default kwargs is a fourth, undeclared reading** (4·P_FA + P_miss),
  numerically distinct from package-0.5. The donor code hard-codes the brief cost at eight sites,
  including **inside the mining selection step** (`hn_reweight.py:74`), so the pilot selected its
  hard examples under the brief cost. Verified locally.
- **`min_cost` saturates at exactly 1.0.** A sign-inverted detector and a zero-information detector
  both read `1.0000`, so a delta of 0 between saturated cells is not evidence of equality.
- **`cost_at` returns NaN (not an exception) on an empty class slice**, and `not (nan > band)` is
  `True` — a silent pass. Verified locally.
- **`PredefinedSplit` with `test_fold == -1`** puts those rows in **every** training split and
  leaves their score at the array's initial value. Verified: `folds=[0,0,1,1,-1,-1]` gives
  `n_splits=2` and `oof=[99,99,99,99,0.0,0.0]`. With `np.zeros` the bug is invisible because 0.0 is
  a legal decision value.
- **`LogisticRegression` convergence is not the binding numerical risk** (~3 s per fit on
  23,790x1024 at `max_iter=3000`, terminating on tolerance). The real risk is the hand-rolled focal
  optimizer, whose `minimize(...).success` is never checked in the donor code. The focal **gradient
  is correct** — verified against finite differences to 5e-7 relative at gamma = 0, 1, 2 — so C3
  will pass and cannot catch non-convergence; C6 must assert on `success` and the gradient norm.
- **`StandardScaler` ignores `sample_weight`**, so a weighted arm standardises on unweighted
  moments. Harmless, but it means a weighted arm is not "the same recipe at a different prior".

## Additional failure modes (each gets at least one test)

| # | What could go wrong | Test |
|---|---|---|
| **B6** | A sentinel fold id (`-1`) trains on a row in every fold and leaves its score silently unset | Score arrays initialise to **NaN**, never `zeros`; assert no NaN survives; counterfactual asserts a `-1` row is detected |
| **B7** | Fold-integrity tested against `cv_groups()` rather than the fold file | B1 asserts on the `group` column; a test pins that the two partitions differ (158 vs 117) so the substitution can't pass silently |
| **C9** | A fold is never written, leaving exact 0.0 that still scores | Per-candidate coverage assert mirroring `probe.py:34` / `hn_common.py:9-12`. **Closest test to the owner's literal risk** |
| **C10** | The focal optimizer stops early and returns a plausible score column | Assert `result.success` and a gradient-norm bound; a non-converged fit is INVALID, not a number |
| **D7** | A saturated reading (exactly 1.0) compared as if measurable | Explicit saturation flag; two saturated cells may not be called equal |
| **D8** | A reading computed with default kwargs, or the mining step selecting under the wrong cost | Every reading goes through one named helper; assert the three helpers are pairwise distinct on real data; assert `package05 == sponsor_min_dcf(flip=True)`; assert the mining threshold uses the declared cost |
| **E7** | **The whole gate is decorative** | A synthetic candidate built to be overfit (label signal leaked into f0 and f4 only, chance elsewhere) **must be rejected** by R1-R10. This single test exercises D7, D8, C9, R6's basis and R10 at once |
| **E8** | A broken candidate passes instead of being refused | INVALID checklist per candidate: inner AUC < 0.5 (direction inverted), duplicate paths, fold disagreement, non-finite scores, incomplete coverage. INVALID never passes |
| **E9** | Two passers and no tie-break | Pareto rule: recommend none unless one passer dominates every other across all declared cells |
| **H7** | A margin-based arm silently means a different band *width* per candidate, because cost-weighting and focal change the logit scale | All margin-based selection on a per-candidate scale-normalised score; one declared scale for every R6/R7/H5 count. (The Oct-11 Jaccard 0.000 measurement is scale-dependent and was taken on the shipped Platt scale) |

## Declared free parameters (previously undeclared)

Fixed now, before any fit. Not swept; a sweep would break the freeze.

| Parameter | Value | Source |
|---|---|---|
| Cost-weight multiplier `k` | **4** | matches the pilot |
| OHEM selection fraction / weight | top **5%** per class, weight **x5** | matches the pilot |
| OHEM margin | **in-sample** (the fold model's own margin on its training rows), named `ohem_insample` | matches the pilot; in-sample and nested are different recipes |
| Mine-and-retrain fraction / weight | top **5%** per class by the declared definition, weight **x5**, nested inner-inner | matches the pilot's weight |
| Focal gamma | **2** | matches the pilot |
| Bootstrap replicates | **1,000**, paired, shared RNG, seed 0, resampling `group` units | the pre-declared precedent (`docs/specs/2026-10-03_layer-bakeoff.md:43,61`) |

**Recipes: 13** (was 12). The added one is `mine_nested_pilot` — reals above the 98th percentile,
fakes in the bottom 5%, weight x5 — which reproduces the pilot's `mine_nested` row. None of the
three declared mining definitions matches that quantile rule, so without this arm no cell in the
grid would be comparable to the pilot's published number. **It is labelled pilot-comparable and is
not eligible for recommendation.**

## Gate, superseding the earlier table

| Rule | Statement |
|---|---|
| **R1** | Pooled package-0.5 on the primary basis beats base, with a paired `group`-cluster bootstrap (1,000 replicates, seed 0) giving P(better) >= 0.90 |
| **R2** | Pooled package-0.3 also beats base; both package readings agree in direction |
| **R3** | Brief does not regress beyond **0.010** vs base (convention robustness, D5) |
| **R4a** | Fold **0** package-0.5 not worse than base by more than **0.010** (`pool_nsa` base 0.4735, `pool_full` 0.4696) |
| **R4b** | Fold **4** package-0.5 not worse than base by more than **0.010** (`pool_nsa` base 0.3950, `pool_full` 0.3450). **A tripwire, not a statistical test**: f4's spoof side is 1 group on `pool_nsa` and 2 on `pool_full`, so a group bootstrap is not available there. Stated, not hidden |
| **R5** | R1-R4b hold on at least **3 of the 4** layers, **and the 3 must include at least one of {L7, L20}**. L6/L7/L8 are near-duplicates (ITW error Jaccard 0.6-0.7, `docs/reports/2026-10-03_layer-diversity.md:102-110`); L20 is the only genuinely independent cell, so "3 of 4" would otherwise be satisfiable by three copies of one probe |
| **R6** | False alarms at the candidate's **own** pooled package-0.5-optimal threshold <= base + **10% relative**. Base: `pool_nsa` **1,642**, `pool_full` **2,348**. The band is relative, so it is basis-portable; the threshold is found by an exact re-sweep over `unique(s) + inf`, never from `roc_curve`'s thresholds (incomplete under `drop_intermediate=True`) |
| **R7** | Any mined set: no single `group` > **25%**; characterization readout present (source, generator, `group`, duration, speaker concentration); confidently-wrong false alarms counted |
| **R8** | Integrity: zero `decode_error`, zero all-zero rows, full join coverage, full per-candidate score coverage (no NaN, no unwritten fold), nested-fold disjointness, no NaN reading, no saturated reading treated as measurable |
| **R9** | **Degenerate-candidate refusal:** constant scores, non-finite scores, inner AUC <= 0.5, a reading saturated at 1.0, or a focal fit with `success == False` makes the candidate **INVALID**, never a pass |
| **R10** | **Multiplicity declared:** N recorded in the report, inference stated as **exploratory**, and the Pareto tie-break (E9) applied before anything is called a winner |

## Multiplicity: the recommendation is a direction, not a cell

13 recipes x 4 layers x 2 pools = **104 gate evaluations**, against a documented cap of
*"four candidates against one gate ... do not add a fifth after seeing results"*
(`docs/handoffs/2026-09-26_post-draft-gate-handoff.md:49`) and
*"cap the candidate count in the manifest"* (`docs/consults/2026-09-26_post-draft-review_RESPONSE.md:75`).

Resolution, pre-declared:

- The **primary question** — does hard-example training help, and in which direction — is answered
  by a pre-declared **aggregate** comparison: real-side arms versus fake-side arms, averaged across
  layers and pools. That is **one** comparison, not 104.
- **No single cell is ever the recommendation.** A cell that passes R1-R10 is reported as a **lead
  requiring confirmation**, in the same language the layer bake-off used for its pair blends.
- The secondary question (which definition of "hard") is reported in full and may not generate the
  headline claim.
- This is consistent with the owner's P1 statement, which describes a decision about *how* to use
  hard negatives — a direction — not a model selection.

## Self-check must block, not print

Acceptance criterion 2 is strengthened: the base-reproduction check **raises** and halts the run
before any candidate is read. Precedent and the reason: the pilot's equivalent check was a `print`,
it emitted `max|diff| 57.4`, and the run continued to produce the table that has been quoted since
(`hn_reweight.py:87`, `reweight.log:7`). The rule it should have followed:
*"If the self-check fails, no candidate is read until it is fixed"*
(`docs/reports/2026-09-26_post-draft-manifest.md:89`).

## Engineering constraints added

- Slice the four layers **at load**, per set, never materialising the 25-layer stack (2.83 GB for
  both training sets vs ~450 MB for four layers). The 18 GB Mac is shared with two active lanes, and
  the documented memory incident put it into ~16 GB of swap.
- `n_jobs` a module constant, **<= 4**, with `pre_dispatch="n_jobs"`; `OMP_NUM_THREADS=1` in workers
  to avoid process x BLAS oversubscription (nothing in the repo sets it for CPU probe scripts).
- Per-cell `try/except` marking that cell INVALID so one divergent fit cannot kill the grid, and
  per-cell `.npz` resume so the run is restartable inside the time box.
- Stop processes by explicit PID only. joblib's workers are Loky processes, and a pattern-kill on
  this shared Mac has already hit another agent's workers once
  (`docs/reports/2026-10-10_v2-fusion-research.md:288`).
- The config hash covers the **realized** candidate list, so a partial table can never be read as
  the full grid.
- **Time box: a wall-clock cutoff that checkpoints, stops and reports** — not an estimate. 3 h.

---

## Phase 3 round 2 — FINAL DESIGN (supersedes every earlier grid and gate statement)

Reframed on Nathan's instruction, Oct 13: **"yeah reframe it, hard vs easy vs random"**.

An adversarial critique pass plus local verification found that the earlier design could not answer
its own question. This section is self-contained and is the design of record. The frozen P1/P2 text
and the earlier Phase 3 section remain above as the audit trail.

## Why the earlier design was withdrawn (all verified locally, Oct 13)

| # | Finding | Evidence |
|---|---|---|
| **W1** | **The two arms that were to answer the direction question were the same model at two regularization strengths.** `class_weight="balanced"` is computed by sklearn on the **sample-weighted** class counts, which exactly cancels a per-class `sample_weight` and leaves a single global scale; a global scale on `sample_weight` is exactly equivalent to scaling `C`. | Effective-weight ratio is one constant for every row, = (k·n_r + n_f)/n: **2.6734** (`pool_full` reals x4), 2.3266 (fakes x4), **2.5132 / 2.4868** on `pool_nsa` — a **1% difference on the primary basis**. `reals x4` at C=1 equals base at **C=2.8** to **5.6e-16** on coefficients. So the pilot's `cost4` row (0.2779) measures a looser prior, not cost weighting. |
| **W2** | **The in-sample mining family is degenerate.** The L7 probe is perfectly separable in-sample: **0 in-sample errors in all five folds** (min margin 0.76-1.95). So `conf_wrong` and `hedging` select identical clips and `wrong_at_thr` mines the empty set, i.e. two of three declared arms collapse and the third equals base. | Measured per fold. |
| **W3** | **R7's 25% group cap was unsatisfiable on the fake side by construction.** There are only **14** spoof groups (vs 144 real). The hardest fakes are **61-70% `gen:grad_tts`**. Every mining recipe would have failed R7 before any score was read, while the non-mining recipes satisfied it vacuously — so it could never fail independently either. | top group share: reals 3.3-6.9% (pass), fakes 61.3-69.6% (fail). |
| **W4** | **R5's cross-layer clause could not fail.** All four 3-subsets of {6,7,8,20} contain L7 or L20. | Enumerated. |
| **W5** | **The reference table pinned *basis* values under *pool* names, and `base` was never declared per layer.** The shipped L7 probe trains on all 23,790 rows; 0.3307 is that model read on the 16,142 NSA rows — a basis, not a pool. Per-layer bases are far apart (L20 pooled 0.5943), so a +/-0.010 band against L7's values would have failed every L6/L8/L20 candidate for reasons unrelated to the candidate. | L06 0.3517 / L07 0.3104 / L08 0.3812 / L20 0.5943 (`pool_full`). |
| **W6** | **83% of inner rows sit in a speaker that crosses folds**, so B1's speaker half was untestable. **41 of 113** speakers span >1 fold, covering **19,699 of 23,790 rows**. Groups cross 0 folds; one generator crosses. This is structural to ASVspoof 2019 LA (its spoofs are built from its bona fide speakers: `LA_0069` is bona fide in fold 1 and the source of A01-A06 spread across all five folds). | Measured. |
| **W7** | **The dominant effect is the regularization constant, not which examples are up-weighted.** | See below. |

## W7, the finding that sets the design

Base probe, L7, `pool_full`, 5 folds, pooled package-0.5:

| C | package-0.5 | package-0.3 | brief |
|---|---|---|---|
| 0.25 | 0.3852 | 0.2353 | 0.2597 |
| **1.0** (v1's inherited value) | **0.3104** | 0.2082 | **0.2649** |
| 8.0 | 0.2514 | 0.1770 | 0.2728 |
| 64.0 | 0.2333 | 0.1651 | 0.2768 |
| 256.0 | **0.2304** | **0.1634** | 0.2775 |

Monotone to C=256 under both package readings (**-0.080**), and **slightly worse** under brief
(+0.013). Per-fold at C=1 -> 64: f0 0.4696 -> 0.4332, f4 0.3450 -> 0.3354, f1 0.1990 -> 0.2032,
f2 0.0057 -> 0.0108 — the gain is in the two error-carrying folds, not one fold.

Weighting shape with `sum(sample_weight)` held at n (pure shape, no scale change):

| | base | easiest | hardest | random |
|---|---|---|---|---|
| at C = 1 | 0.3104 | **0.2814 (-0.029)** | 0.3204 (+0.010) | 0.3337 (+0.023) |
| at C = 64 | 0.2333 | 0.2381 (+0.005) | 0.2305 (-0.003) | — |

**The easy-mining gain is an artifact of over-regularization.** At a tuned C every shape moves the
score by < 0.005 and the signs flip. v1 fixed C = 1.0 (`src/hearsay/probe.py:43-47`) under the brief
reading, where C barely matters; under rule 5's reading that choice is the largest single lever in
this lane.

**Consequence for the design:** a weighting-shape comparison at a single C answers the wrong
question, and any comparison not at matched C is confounded with regularization.

## The reframed question

**Primary:** at **matched regularization**, does up-weighting a 5% subset of training examples change
the score, and does the selection rule matter — **hardest vs easiest vs random**?
`random` is the control that makes the other two interpretable.

**Secondary, descriptive, explicitly not a selection:** how does the score move with C? Selecting C
on inner OOF is the L7+WavLM trap (`docs/reports/2026-10-03_layer-diversity.md:64-80`:
*"Inner CV contains no wild real speech, so it cannot see WavLM's false alarms"*). The C curve is
reported as a **lead for the V2-01 benchmark lane**, never adopted here.

## Final grid

**All shape selection uses nested inner-inner out-of-fold margins**, never in-sample (W2). A fold
never selects its own hard examples, and out-of-fold errors actually exist (2,348 FA / 351 miss at
the package-0.5 threshold).

**All arms hold `sum(sample_weight) = n`**, so every comparison is at matched effective
regularization. Class balancing is computed once on **unweighted** counts and multiplied by the
arm's shape, then renormalised (W1).

| Arm | Layers | Note |
|---|---|---|
| `none` (base) | 6, 7, 8, 20 | reference, per layer x pool x C |
| `hardest_both`, `easiest_both`, `random_both` | 6, 7, 8, 20 | the primary three-way question |
| `hardest_reals`, `hardest_fakes`, `easiest_reals`, `easiest_fakes` | 7 | direction probe |
| `focal_g2` | 7 | continuous hard-example emphasis; the one arm with literature behind it |
| `pilot_quantile` | 7 | reals above the 98th pct, fakes bottom 5%; reproduces the pilot's `mine_nested` row for comparability. **Not eligible for recommendation** |

**C: {1.0, 8.0, 64.0}** — v1's inherited value, a middle, and the near-asymptote.
**Pools: `pool_full` (23,790 fitted), `pool_nsa` (16,142 fitted).**
**Bases for reading: both, every number labelled with its basis.** `pool_nsa` x full-basis does not
exist (a `pool_nsa` fit has no OOF for the 7,648 ASV19 rows) and is **declared N/A in advance**.
**C descriptive sweep: C in {0.25, 1, 2, 4, 8, 16, 64, 256} x 4 layers x 2 pools**, base arm only.

Cells: 4 arms x 4 layers x 2 pools x 3 C = 96, plus 6 arms x 1 layer x 2 pools x 3 C = 36, plus the
64-cell descriptive C sweep. Each cell is ~5 cheap fits (verified: an LR fit on 18,256x1024 is
~0.05 s; the focal fit is ~9 s), so the grid is minutes, not hours.

## Final gate

A candidate is an (arm, layer, pool, C) cell. **`base` always means the `none` arm at the same
layer, pool, basis and C** (fixes W5). Every comparison is within a matched (layer, pool, C) stratum.

| Rule | Statement |
|---|---|
| **R1** | Pooled package-0.5 on the primary basis beats matched `base`, with a paired `group`-cluster bootstrap (1,000 replicates, seed 0, identical resamples for arm and base) giving P(better) >= 0.90. The Monte-Carlo SE of P(better) at 1,000 replicates is ~0.0095 and **is reported beside it**; a verdict within 2 SE of the 0.90 boundary is reported as **indeterminate**, not as a pass |
| **R2** | Pooled package-0.3 agrees in direction with R1 |
| **R3** | Brief does not regress beyond **0.010** vs matched base |
| **R3b** | **Direction agreement (new):** the sign of the arm-vs-base delta must agree across all three readings. A sign flip is reported as **convention-dependent** and cannot be a recommendation — this lane's cost convention is unresolved (D5), and the verified example is `reals x4` vs `fakes x4`, which disagree between package-0.5 and brief |
| **R4a / R4b** | Fold 0 and fold 4 package-0.5 not worse than matched base by more than **0.010**. **Tripwires, not statistical tests**: f4's spoof side is 1 group on `pool_nsa` and 2 on `pool_full`, so no group bootstrap exists there. Stated, not hidden |
| **R5** | R1-R4b hold on at least 3 of the 4 layers **and must include L20**. L6/L7/L8 are near-duplicates (ITW error Jaccard 0.6-0.7); L20 is the only independent cell, so requiring "L7 or L20" could not fail (W4) |
| **R6** | False alarms at the candidate's **own** pooled package-0.5-optimal threshold <= matched base + **10% relative**. Per-layer/pool/C base counts are computed with the base arm in the same run, never pinned from a prior run. Threshold by exact re-sweep over `unique(s) + inf`, with ties resolved to the **highest** threshold (declared; `min()` over a sorted candidate list would take the lowest and maximise FA among tied optima) |
| **R7** | **Real side:** no single `group` > 25% of the mined set. **Fake side: reported, not gated** — there are only 14 spoof groups and whole generators are groups, so any cap is unsatisfiable by construction (W3). The fake-side concentration and its top group are reported for every mined set |
| **R8** | Run preconditions, checked once, not per candidate: zero `decode_error`, zero all-zero rows, full join coverage, nested-fold disjointness, speaker-crossing documented (W6). A failure halts the run |
| **R9** | Per-candidate refusal -> **INVALID**, never a pass: non-finite or constant scores, inner AUC **< 0.5** (one convention, matching E8), any reading saturated at 1.0, incomplete score coverage, a mined set below a minimum size of **20 per class**, or a focal fit whose final gradient norm exceeds **1e-2** (`result.success` alone is insufficient — L-BFGS-B returns success on the f-reduction criterion with gradient norms of 4e-3 to 9.5e-4, verified) |
| **R10** | Multiplicity declared: the realized cell count, the N/A cells, and the statement that inference is **exploratory** |

**What the gate is for, and what it is not.** R1-R10 decide, per stratum, whether a weighting shape
beats no weighting. The **answer to the primary question** is the three-way comparison of
`hardest` / `easiest` / `random` against `none`, aggregated across layers and pools with the paired
group bootstrap, and reported with its uncertainty. **No single cell is a recommendation and no cell
is adopted**; a passing cell is a lead. The Pareto tie-break is withdrawn — it selected a cell,
which contradicted that.

## Test-design changes

- **Break check moves from H2 to C2.** Verified: flipping the mining sort direction produces a
  *better* table under the primary reading (0.2798 vs 0.3139), so that mutation would be caught only
  by a test that mirrors the mutated line, and it improves the deliverable rather than corrupting
  it. **C2** (the directional test: a shape applied to one class must move that class's error rate)
  is the mutation whose failure is both realistic and damaging. E7 stays as the whole-gate
  counterfactual.
- **B1's speaker clause is replaced, not deleted.** Groups and generators are fold-disjoint and are
  asserted. Speakers are **not** (W6), so the test asserts the measured, documented state — 41
  speakers across >1 fold covering 19,699 rows — and fails if it changes. CLAUDE.md's
  "validate by generator **and** speaker" is not satisfied by this fold file; that is now recorded
  rather than assumed.
- **Rules that are not per-candidate are not given per-candidate counterfactuals.** R8 is a run
  precondition and R10 is a reporting checklist; E1's per-rule counterfactual applies to
  R1-R7 and R9 only. The earlier plan promised one-per-rule for all ten, which is undeliverable.
- **D7 is withdrawn** as unreachable: R9 makes any saturated candidate INVALID, so "two saturated
  cells compared as equal" cannot arise.
- **Baseline test count is 906**, not 859 (commit `735d0a3` added 47).
- Embeddings load with a **per-shard single-layer loader**, not `load_embeddings` + slice:
  `probe.load_embeddings` materialises the full 25-layer fp32 array and a second copy inside the
  call, so slicing afterwards does not avoid the peak. Verified: the per-shard loader reads L7 for
  27,648 rows in **0.5 s**.
- `sample_weight` cannot pass through `make_clf`'s Pipeline (verified: `Pipeline.fit` rejects it).
  Arms scale explicitly and fit a bare `LogisticRegression`, so no arm double-scales.

---

## Codex plan review round 1 — disposition

Review: `docs/reports/2026-10-13_v2-33-hardneg-stage0-plan-review.md` (7 Critical, 4 Suggestions).
**All eleven accepted**; resolutions are in the plan's "Codex plan review round 1" section. The three
that change this spec:

1. **Stage 0 is a diagnostic on the legacy split, not a validation** (Codex #1). The fold file has
   known speaker overlap (41 of 113 speakers across >1 fold, 19,699 of 23,790 rows) and is read-only
   to this lane. Recording that is not the same as fixing it, so no validation conclusion is drawn
   here. **V2-33 stays open** pending a V2-01 benchmark row on generator- **and** speaker-disjoint
   folds, and that requirement is raised with the benchmark lane.
2. **The primary comparison is now fully specified** (Codex #2): basis NSA-only; strata
   (layer, pool, C) with C never averaged over; equal stratum weights; INVALID cells excluded and
   counted; **four pre-declared paired contrasts** — `hardest−none`, `easiest−none`,
   `hardest−random`, `easiest−random` — with one shared group resample reused across strata so
   overlapping pools and near-duplicate layers are not counted as independent evidence.
3. **R5 eligibility is declared** (Codex #7): it applies only to the four-layer primary family
   (`none`, `hardest_both`, `easiest_both`, `random_both`). The L7-only arms are descriptive with
   cross-layer eligibility N/A, so their failure is never predetermined by a missing experiment.

Also accepted and specified in the plan: per-checkpoint input/config fingerprints with atomic writes
and grid-derived completeness (#3); the join takes only `fold` and `group` and asserts the
overlapping manifest columns agree (#4, verified to raise otherwise); the loader keeps
`probe.load_embeddings`' exact-permutation guarantee (#5, verified that `seen.all()` admits a
negative index); frozen definitions for margin normalisation and for H5 "confidently wrong" (#6); one
persisted run deadline covering every subcommand (#8); a semantic hardest-vs-easiest fixture test
beside C2 (#9); a corrected verification grep (#10); and an explicit "what this does not measure"
section in the report (#11).
