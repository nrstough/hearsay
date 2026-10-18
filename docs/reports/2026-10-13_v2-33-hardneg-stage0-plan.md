# Implementation plan — V2-33 stage 0: example-weighting bake-off (hard vs easy vs random)

Run spec: `docs/specs/2026-10-13_v2-33-hardneg-stage0.md`. **The design of record is that spec's
"Phase 3 round 2 — FINAL DESIGN" section**; the earlier sections are the audit trail. Branch `v2`.

Reframed on Nathan's instruction, Oct 13 ("yeah reframe it, hard vs easy vs random") after a
critique pass showed the earlier design could not answer its own question (spec W1-W7).

**Writes only:** `scripts/v2_hardneg_bakeoff.py`, `tests/test_hardneg.py`, `outputs/v2/hardneg/`,
this lane's docs, the V2-33 section of `docs/prod-chunks.md` (**lines 511-537**), one row in
`experiments/log.csv`.

---

## Step 0 — Pre-flight (no edits)

1. `git branch --show-current` == `v2`; `pwd` == `/Users/nathanstough/Projects/hearsay`.
2. `git status --short`: the five other-lane modified files, plus the backbone lane's untracked docs.
   **Stage explicit paths only; never `git add -A`.** A sibling worktree is live at
   `.claude/worktrees/v2-benchmark-fusion/` — do not touch it.
3. `pgrep -fl "extract_embeddings|repair_decode_errors|score_spectra|m5_score|m3_probes|run_pipeline|bakeoff_layers|python.*v2_"` — if anything shows, stop and ask.
4. Baseline: `uv run pytest -q` and `uv run ruff check .`. **Expect 906 collected** (not 859; commit
   `735d0a3` added 47). A `test_docs_consistency` failure is likely the uncommitted
   `.claude/codex-audit-prompt.md`, which is in that test's gated set — another lane's edit; record
   and continue.
5. Confirm `~/hearsay-v2-research-2026-10-10.tgz` exists (33 MB).

## Step 1 — Script skeleton

Model on `scripts/explore_layer_diversity.py`. Module docstring naming this spec and its usage
lines; `from __future__ import annotations`; stdlib / third-party / `hearsay.*` groups; `# fmt: skip`
on hand-aligned literals. Ruff default rules only (E4/E7/E9/F); E501 not enabled; formatter width 100.

Module globals — one source of truth each, so a single `monkeypatch.setattr` makes tests hermetic:

```python
OUT     = REPO / "outputs" / "v2" / "hardneg"      # the ONLY write root
MODEL   = "wav2vec2-xls-r-300m"
TRAIN_SETS = ("nsa_train_sample_v3", "asv19_addon_v3")
FOLDS   = REPO / "splits" / "nsa_folds_plus_asv19.csv"
LAYERS  = (6, 7, 8, 20)
C_GRID  = (1.0, 8.0, 64.0)                          # the three gated values
C_SWEEP = (0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0)   # descriptive only, base arm
FRAC, MULT, MIN_MINED = 0.05, 5.0, 20
N_JOBS, BOOT_N, SEED = 4, 1000, 0
FOCAL_GAMMA, FOCAL_GRAD_TOL = 2.0, 1e-2
TIME_BOX_S = 3 * 3600
LOG_COLUMNS = ("timestamp","rung","validation_score","validation_score_clean_only",
               "csv_path","notes","benchmark_version")  # fmt: skip
```

Module level is constants only — no reads, no `mkdir` — because every test imports the file by path
(`scripts/` is not importable; `pyproject.toml:42` puts only `src` on the path).
Subcommands: `fit`, `csweep`, `gate`, `report`; flags `--jobs`, `--layers`, `--resume`.

## Step 2 — Named cost helpers

`min_cost`'s defaults are `c_fa=4.0, c_miss=1.0` (`src/hearsay/metrics.py:29-30`), so
`min_cost(y, s, 0.5)` is a **fourth** reading (4·P_FA + P_miss), verified numerically distinct from
package-0.5. Nothing in this script calls `min_cost` directly:

```python
def brief(y, s):     return min_cost(y, s)
def package05(y, s): return min_cost(y, s, 0.5, c_fa=1.0, c_miss=4.0)   # == bakeoff_layers.averse
def package03(y, s): return min_cost(y, s, 0.3, c_fa=1.0, c_miss=4.0)
READINGS = {"package05": package05, "package03": package03, "brief": brief}
```

Threshold (R6). None of the repo's three `inner_threshold` variants uses the package convention, and
`roc_curve`'s `thresholds` is incomplete under `drop_intermediate=True`. **Ties resolve to the
highest threshold** — declared, because the lowest maximises FA among tied optima and 8 of the
23,790 L7 scores are tied:

```python
def thr_package05(y, s):
    cand = np.r_[np.unique(s), np.inf]
    costs = np.array([cost_at(y, s, t, 0.5, c_fa=1.0, c_miss=4.0) for t in cand])
    return float(cand[np.flatnonzero(costs == costs.min())[-1]])      # highest tied optimum
```

`cost_at`'s positional order is `(y, s, thr, pi_synth, c_fa, c_miss)` — `thr` **third**.
Measured cost of the sweep: 2.4 s at 23,790 scores.

## Step 3 — Load (per-shard, single layer)

`probe.load_embeddings` materialises the full 25-layer fp32 array **and a second copy** inside the
call (`src/hearsay/probe.py:30-35`), so slicing afterwards does not avoid the peak. Use the
per-shard loader instead — verified at **0.5 s** for 27,648 rows, one layer:

```python
def load_layer(name, layer):
    d = REPO / "outputs" / "embeddings" / MODEL / name
    m = pd.read_csv(d / "manifest.csv")
    X, seen = np.zeros((len(m), 1024), np.float32), np.zeros(len(m), bool)
    for p in sorted(d.glob("shard_*.npz")):
        z = np.load(p); r = z["row"]
        X[r] = z["emb"][:, layer, :].astype(np.float32); seen[r] = True
    assert seen.all(), f"{name}: shard coverage incomplete"
    assert "flag" in m.columns, f"{name}: no flag column"          # absent column must not pass
    assert m["flag"].fillna("").eq("").all(), f"{name}: decode_error rows"
    assert not (X == 0).all(axis=1).any(), f"{name}: all-zero rows"
    return X, m
```

Then, once:

- concatenate the two sets, `reset_index(drop=True)` (positional alignment — `meta[mask]` must not
  carry a stale pandas index);
- join `FOLDS` on `path` taking **`fold`, `source`, `speaker`, `group`**; assert `fold.notna().all()`;
- **drop the holdout rows immediately and unread** (the `explore_layer_diversity.py:62` pattern) —
  D3/B4 enforced structurally;
- reindex to `outputs/v2/bakeoff/rows.json`'s `inner_path` order so the self-check in Step 5 can
  compare positionally;
- pools from **one** join: `pool_full = ones(n, bool)`, `pool_nsa = (source != "asvspoof2019")`
  (16,142 verified). Joining against `splits/nsa_folds.csv` instead would leave the 7,648 ASV19 rows
  unmatched and trip the coverage assert;
- mask the holdout **before** `.astype(int)` — `PredefinedSplit` raises on `'holdout'` (verified).
  **Never encode a pool by setting a row's fold to `-1`**: verified, such rows join every training
  split and keep the array's initial value;
- write `OUT/"rows.json"` with `inner_path, inner_y, inner_fold, inner_source, inner_speaker,
  inner_group` — the bake-off's own `rows.json` has no source and no speaker.

## Step 4 — Arms, at matched regularization

The W1 fix. Class balance is computed on **unweighted** counts, multiplied by the arm's shape, then
renormalised so `sum(sw) == n` for every arm — otherwise the arm is confounded with `C`:

```python
def weights(y_tr, shape_idx, mult=MULT):
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)   # unweighted counts
    w = np.where(y_tr == 0, cw[0], cw[1])
    w[shape_idx] *= mult
    return w * (len(y_tr) / w.sum())        # matched effective regularization
```

Never `class_weight="balanced"` together with `sample_weight`: sklearn computes "balanced" on the
sample-weighted counts, which cancels the class differential and leaves a global scale — verified
exactly equivalent to scaling `C` (5.6e-16 on coefficients).

`sample_weight` cannot pass through `make_clf`'s Pipeline (verified: `Pipeline.fit` rejects it), so
arms scale explicitly with a fold-local `StandardScaler` and fit a bare
`LogisticRegression(C=C, max_iter=3000)`. No arm double-scales, which is what keeps Step 5 exact.

**Shape selection is on nested inner-inner OOF margins, never in-sample** (W2: the probe is
perfectly separable in-sample, 0 errors in all five folds, which collapses the in-sample family).
For outer fold k, run a 4-fold inner-inner OOF over the training rows, then:

| shape | selection |
|---|---|
| `hardest` | lowest signed margin `where(y==1, d, -d)` |
| `easiest` | highest signed margin |
| `random` | `default_rng(SEED)` choice — the control |
| `pilot_quantile` | reals above the 98th pct of their OOF, fakes in the bottom 5% (reproduces the pilot's `mine_nested`; asymmetric 2%/5%, unlike every other arm — stated in the report) |

`side in {both, reals, fakes}`; a mined set smaller than `MIN_MINED = 20` per class makes the cell
INVALID (R9) rather than silently empty — folds 2 and 3 have 0 out-of-fold fake errors.

Coverage, the C9 fix:

```python
oof = np.full(len(y), np.nan)      # NaN, never zeros — 0.0 is a legal decision value
...
if not np.isfinite(oof).all():
    raise ValueError(f"score coverage incomplete: {int((~np.isfinite(oof)).sum())} rows unwritten")
```

`focal_g2` keeps the pilot's objective and gradient (verified correct to 5e-7 relative at
gamma 0/1/2, and gamma=0 reproduces sklearn's balanced LR to 2e-7). It is INVALID unless the final
gradient norm is below `FOCAL_GRAD_TOL = 1e-2`; `result.success` alone is insufficient — L-BFGS-B
returns success on the f-reduction criterion at gradient norms of 4e-3 to 9.5e-4 (verified).

Per-cell isolation and resume, the `fuse_sweep_v3.run_candidates:475-483` pattern:

```python
try:    cell = fit_cell(...)
except Exception as e:  # noqa: BLE001 - one bad fit must not kill the grid
    cell = {"status": "INVALID", "reason": f"{type(e).__name__}: {e}"}
```

plus `np.savez(OUT/"cells"/f"L{layer:02d}_{pool}_C{C}_{arm}.npz", ...)`, skip-if-exists resume, and a
wall-clock check between cells that **checkpoints, stops and reports** at `TIME_BOX_S`.
`Parallel(n_jobs=N_JOBS, pre_dispatch="n_jobs")` with `OMP_NUM_THREADS=1` in workers — nothing in the
repo sets it for CPU probe scripts, and the donor files hard-code `n_jobs=6`.

## Step 5 — Self-check that BLOCKS

Before any candidate is read, the `none` arm at L7 / `pool_full` / C=1 must reproduce the shipped
probe:

```python
ref = np.load(REPO/"outputs/v2/bakeoff/probes/L07.npz")["oof"]            # read-only
if spearmanr(mine, ref).statistic != 1.0 or abs(package05(y, mine) - package05(y, ref)) > 1e-12:
    raise SystemExit("self-check failed: base does not reproduce the shipped L7 probe")
```

Monotone-invariant by construction; the shipped OOF is Platt-rescaled
(`bakeoff_layers.py:132-138`), relation verified as `shipped = 0.34480497·mine + 1.99776021`,
residual 1e-14, Spearman **exactly** 1.0, package-0.5 difference **exactly** 0.0. A raw `max|diff|`
reads **57.4** on that identical ranking — the pilot printed exactly that and continued
(`hn_reweight.py:87`, `reweight.log:7`). This **raises**.

Note the threshold value −1.4414 is on the *shipped Platt scale*; the lane's own base produces
−9.9743. Every readout recomputes its threshold; no threshold value is ever pinned.

## Step 6 — Readings, bootstrap, gate

- Each cell: three readings x {pooled, per-fold, f0, f4} x both bases, EER, the R6 FA count at its
  **own** threshold, the H5 confidently-wrong-FA count, and the R7 real-side and fake-side group
  concentrations. `pool_nsa` x full-basis is **N/A, declared in advance**.
- **Matched comparison only:** every arm is compared with the `none` arm at the same layer, pool,
  basis and C. No cross-C or cross-layer comparison enters the gate.
- **Bootstrap:** paired, 1,000 replicates, seed 0, resampling the fold file's **`group`** units (158
  inner groups), identical resamples for arm and base, single-class draws skipped and counted.
  Report `P(better)`, its Monte-Carlo SE (~0.0095 at 1,000), and the p5. A verdict within 2 SE of
  the 0.90 boundary is **indeterminate** — measured on the withdrawn arms, P(better) landed at 0.907
  and 0.897 against an SE of 0.017 at 300 replicates, so the band alone decides nothing.
  Sign convention: `delta = base_cost - arm_cost`, positive = arm better, **asserted by a test** —
  the three repo precedents use three different conventions.
- Bands: round to 4 dp before subtracting, compare with `EPS = 1e-12`; bootstrap percentiles on
  **unrounded** replicates.
- `gate()` returns the full `dict[str, bool]` of R1-R10 plus `verdict`. `clean()` recursion before
  `json.dumps(..., allow_nan=False)`.
- **Cell completeness:** the realized cell set must equal the declared set exactly (the
  `check_cohort:415-421` pattern) — a partial grid is INVALID, never a vacuous pass. The config hash
  covers the **realized** arm list.
- **The primary answer** is the three-way `hardest` / `easiest` / `random` vs `none` comparison,
  aggregated across layers and pools with the paired bootstrap and reported with its uncertainty.
  No cell is a recommendation; a passing cell is a lead.

## Step 7 — `tests/test_hardneg.py` (~70 tests)

No `conftest.py` exists in this repo, so the module is self-contained. Idioms copied:
`_load("v2_hardneg_bakeoff")` (`test_channel_robustness.py:19-26`); a synthetic world modelled on
`test_fuse_sweep_v3.py:481-514` with `fold`, `source` (incl. an `asvspoof2019` subset so both pools
are exercised), `speaker`, `group`; `_pass_d()` + per-rule `(rule, change)` parametrisation
(`:126-140`) for **R1-R7 and R9 only** — R8 is a run precondition and R10 a reporting checklist, so
neither gets a per-candidate counterfactual; the autouse byte-identity guard
(`test_cloud_scripts.py:16-25`) over `experiments/log.csv`, `splits/*.csv`,
`outputs/v2/bakeoff/` **and `outputs/v2/research/`** (the directory F1 exists to protect); the
import-hygiene `subprocess` test (`test_fuse_sweep_v3.py:436-443`) asserting `lightgbm` and `torch`
never enter `sys.modules`; `pytestmark = filterwarnings("ignore::RuntimeWarning")` for `cost_at`'s
empty slices; `needs_data` + a `HAVE_*` probe + a **module-scoped** fixture for the real-embedding
tests.

Tests carrying the most weight:

- **C2 — the break-check candidate.** A shape applied to one class must move that class's error rate
  in the stated direction. (H2 is demoted: flipping the mining sort produces a *better* table under
  the primary reading — 0.2798 vs 0.3139 — so that mutation improves the deliverable and is caught
  only by a verbatim mirror of the mutated line.)
- **C9** unwritten-fold detection; **B6** the `-1` sentinel counterfactual.
- **W1 regression test:** `class_weight="balanced"` + `sample_weight` equals base at scaled `C`, and
  the lane's own `weights()` does **not** — this pins the bug the design was rebuilt around.
- **E7** whole-gate counterfactual: a synthetic candidate with label signal leaked into f0 and f4
  only must be rejected by R1-R10.
- **B1 as measured:** groups and generators fold-disjoint (asserted); **41 speakers across >1 fold
  covering 19,699 rows** asserted as the documented state, failing if it changes. CLAUDE.md's
  "validate by generator **and** speaker" is not satisfied by this fold file — recorded, not assumed.
- **H3** arm-collapse detection (Jaccard between arms reported; the in-sample family was withdrawn
  because this would have fired on it).

## Step 8 — Run and report

1. `OMP_NUM_THREADS=1 uv run python scripts/v2_hardneg_bakeoff.py fit --jobs 4`, then `csweep`,
   `gate`, `report`. Estimated minutes; hard cutoff 3 h.
2. `docs/reports/2026-10-13_v2-33-hardneg-stage0.md`: the three-way answer with uncertainty, the C
   curve as a **lead for the V2-01 benchmark lane** (never adopted here — selecting C on inner OOF is
   the L7+WavLM trap), the full cell table, N, the N/A cells, the exploratory-inference statement,
   and every number with its reading, basis and threshold. **`outputs/` is gitignored
   (`.gitignore:7`), so the numbers must live in this report.**
3. One `experiments/log.csv` row via a lane-local appender (`hearsay.submission.append_log:113`
   writes 6 columns to `submissions/log.csv` and is v1 code this lane may not edit; nothing on `v2`
   appends to `experiments/log.csv`, which is header-only). `rung = "V2-33-stage0"`,
   `benchmark_version = "none (pre-V2-01)"`, `notes` = basis + all three readings + EER. Expect a
   union merge conflict here later.
4. `docs/prod-chunks.md` **lines 511-537** only; commit only when `git diff` shows only our hunks.
5. `CLAUDE.md` AI-use disclosure — conditional, confirm at execution.

## Step 9 — Verify before commit

```bash
uv run pytest tests/test_hardneg.py -q
uv run pytest -q && uv run ruff check .
grep -nE "9\.3333|min_cost\(|c_fa=4|PI_SYNTH|class_weight=.balanced.*sample_weight" scripts/v2_hardneg_bakeoff.py
git diff --stat docs/prod-chunks.md
shasum -a 256 experiments/log.csv splits/*.csv
du -sh outputs/v2/research   # must be unchanged
```

Expect 906 -> ~976 passing, ruff clean, the grep empty, `prod-chunks.md` showing only 511-537.

## Step 10 — Post-commit

1. **Break check** — C2, mutated in a detached scratch worktree outside the repo; expect CAUGHT;
   remove the worktree; record in the spec.
2. **Claude critique** against `.claude/codex-audit-prompt.md` (currently uncommitted and now
   eight-dimensional, including **Test sensitivity** and an **Owner's statement** check under Freeze
   integrity — the spec has that section), looped until Acceptable.
3. **Codex audit:** `bash .claude/review-audit.sh docs/specs/2026-10-13_v2-33-hardneg-stage0.md`.
4. Fix every finding, commit, re-audit, record both scorecards.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| An arm is confounded with regularization (what killed the first design) | `sum(sw) = n` for every arm; matched-C comparison only; a regression test pins the sklearn behaviour |
| "Hardest" undefined because the probe is separable in-sample | selection on nested out-of-fold margins only |
| An empty or tiny mined set scores anyway | `MIN_MINED = 20` per class -> INVALID |
| A fold never written reads as 0.0 | NaN init + coverage raise |
| A `-1` sentinel trains everywhere and scores nothing | never encode a pool as a fold id; counterfactual test |
| Mining under the wrong cost | named helpers only; grep gate |
| A gate rule that cannot fail | per-rule counterfactual for R1-R7, R9; R5 requires L20; R7's fake side is a readout, not a rule |
| The whole gate decorative | E7 synthetic overfit candidate |
| A bootstrap verdict decided by Monte-Carlo noise | SE reported; within 2 SE of the band -> indeterminate |
| Sign convention wrong in the bootstrap | convention declared and asserted by a test |
| Selecting C on inner OOF (the L7+WavLM trap) | the C curve is descriptive, reported as a lead, never adopted |
| Memory on a 3-lane 18 GB Mac | per-shard single-layer loader (0.5 s, ~110 MB/layer) |
| Partial grid read as complete | completeness refusal; config hash over the realized arm list |
| A test writes a tracked file | autouse byte-identity guard, including `outputs/v2/research/` |

## Open items for Nathan at STOP 3

1. **The C finding is bigger than the question we set out to answer.** Tuning the regularization
   constant moves inner pooled package-0.5 from 0.3104 to 0.2304; every weighting shape moves it by
   < 0.005 once C is tuned. It is reported as a lead for the benchmark lane, not adopted here,
   because inner-only selection is exactly what the L7+WavLM trap punished. Confirm that is the right
   disposition, or raise it with the benchmark lane now.
2. **132 gated cells plus a 64-cell descriptive sweep**, against the documented "four candidates
   against one gate" cap. The primary answer is one three-way comparison with a control, aggregated
   with a paired bootstrap; no cell is adopted. Still a departure from the letter of that rule.
3. **The cost convention remains unresolved** (document says false alarm x4, shipped code says
   miss x4), and it now also disagrees about C: package-0.5 improves monotonically with C while brief
   degrades. R3b refuses any arm whose sign flips across readings. **Adoption stays blocked** until
   NSA answers.

---

## Codex plan review round 1 — findings and resolutions

Review: `docs/reports/2026-10-13_v2-33-hardneg-stage0-plan-review.md`. 7 Critical, 4 Suggestions.
**All eleven accepted.** This section supersedes the corresponding step text above.

### 1. Critical — the fold policy still violates the required speaker holdout

Accepted as stated. Recording the leakage does not make validation speaker-disjoint, and resampling
`group` units does not account for speakers shared across groups. Rebuilding the fold file is not
available to this lane (it is read-only, and a new partition would desynchronise every saved score
vector and probe).

**Resolution — stage 0 is relabelled.** It is a **diagnostic on the legacy split**, not a validation.
- No validation conclusion is drawn here. Every verdict is explicitly conditional on the legacy
  split's known speaker overlap (41 of 113 speakers across >1 fold, 19,699 of 23,790 rows).
- The report states this in its first paragraph and in every table caption.
- **V2-33 stays open** pending a V2-01 benchmark row on generator-**and**-speaker-disjoint folds.
- The requirement for such folds is raised with the benchmark lane as a V2-01 input.

### 2. Critical — the primary statistical comparison was undefined

**Resolution — frozen now:**
- **Basis:** NSA-only (16,142) for every primary number; `pool_full` reported beside it.
- **Strata:** (layer, pool, C). **C is a stratum, never averaged over** — the arm effect is
  C-dependent (verified: easiest is −0.029 at C=1 and +0.005 at C=64).
- **Weights:** equal per stratum. INVALID cells are **excluded and counted**, and a stratum with any
  INVALID cell is reported separately rather than silently averaged.
- **Contrasts, all three pre-declared and paired:** `hardest − none`, `easiest − none`, **and
  `hardest − random`, `easiest − random`**. The arm-vs-`none` contrast alone cannot show that
  hardness matters; only the arm-vs-`random` contrast can, which is why the control exists.
- **One shared group resample** is drawn per replicate and reused across every contributing stratum,
  so overlapping pools (`pool_nsa` ⊂ `pool_full`) and near-duplicate layers are not counted as
  independent evidence.

### 3. Critical — resume could silently mix incompatible experiments

**Resolution:** every checkpoint carries a `fingerprint` field = sha256 over the fold-file sha, this
lane's `rows.json` sha, the four embedding manifests' shas, the arm's full parameter dict, `C`, the
layer, the pool mask's sha, and the sha of `scripts/v2_hardneg_bakeoff.py` itself. Skip-if-exists
applies **only** when the stored fingerprint matches the current one; otherwise the cell is recomputed.
Checkpoints are written `.tmp` then `os.replace` (atomic). **Completeness is derived from the frozen
requested grid**, never from the files on disk. Tests cover a stale fingerprint and a truncated file.

### 4. Critical — the metadata join is not executable as described

Verified: `manifest.join(folds[['fold','source','speaker','group']], on='path')` raises
`ValueError: columns overlap but no suffix specified: Index(['speaker', 'source'])`. The manifests
already carry `label, generator, speaker, source`.

**Resolution:** take **only `fold` and `group`** from the fold file (verified to join cleanly), and
turn the overlap into a free consistency check — assert the manifest's `label`, `generator`,
`speaker` and `source` agree with the fold file's on every joined row. Also assert the path mapping
is one-to-one (no duplicates on either side) and that the path **sets** agree before reordering to
`inner_path`.

### 5. Critical — the shard loader weakened an existing integrity check

Verified: `seen.all()` returns **True** for the index set `[0,1,2,3,4,-1]`, which is not a
permutation — a silently overwritten row. It also admits duplicates.

**Resolution:** keep `probe.load_embeddings`' guarantee verbatim —
`assert np.array_equal(np.sort(rows), np.arange(len(m)))` — and add integer-dtype, bounds, shape and
finiteness checks. Tests feed duplicate, negative, missing and non-finite shard data; each must fail
**R8 before any fitting**.

### 6. Critical — mining and confidence readouts had no defined score scale

The nested OOF margins come from four separately fitted inner models, and Step 4 pooled their raw
values to rank examples. The spec's H7 required normalisation; the plan did not implement it.

**Resolution — both definitions frozen:**
- **Normalisation:** each inner model's margins are divided by the IQR of that model's margins **on
  its own inner training rows only**, so the scale is fitted inside the training fold and never uses
  the rows being ranked. Ranking is then on normalised margins pooled across inner folds.
- **H5 "confidently wrong":** a false alarm whose normalised margin falls in the **worst decile** of
  the real-side normalised margin distribution for that cell. A numerical definition, reported with
  its count and its decile cut.
- Both are tested for the intended scale behaviour: multiplying one inner model's raw margins by a
  constant must not change the selected set.

### 7. Critical — R5 could not pass for the L7-only arms

**Resolution — gate eligibility declared explicitly:**
- **R5 applies only to the four-layer primary family:** `none`, `hardest_both`, `easiest_both`,
  `random_both`.
- The direction probes, `focal_g2` and `pilot_quantile` run at L7 only, are **descriptive**, and
  carry `cross_layer_eligibility: "N/A — single layer by design"`. They are never scored against R5
  and never reported as gate failures.

### 8. Suggestion — the time limit did not cover the declared workflow

**Resolution:** one **persisted run deadline** written at start-up, honoured by `fit`, `csweep`,
`gate` and `report` alike. Work stops being scheduled once it expires (including queued parallel
work — `pre_dispatch="n_jobs"` bounds the queue). Partial results produce an **incomplete report**
that cannot pass the complete-grid gate.

### 9. Suggestion — restore a semantic test for hardest versus easiest

Accepted; C2 alone does not detect a sort substitution reliably.

**Resolution:** add a fixture test with independently specified margins and the expected selected
index sets, asserting hardest/easiest ordering, per-class selection counts, and deterministic tie
handling. **C2 remains the break-check candidate** (its failure is damaging rather than
score-improving); this test is the semantic companion.

### 10. Suggestion — the verification grep failed its own expectation

Correct: the three approved wrappers themselves contain `min_cost(`. **Resolution:**

```bash
grep -nE "min_cost\(|9\.3333|PI_SYNTH|c_fa=4" scripts/v2_hardneg_bakeoff.py \
  | grep -vE "^(23|24|25):" || true      # the three wrapper definition lines, by number
```

i.e. the check permits `min_cost(` inside the named wrapper definitions and rejects it anywhere else,
with the permitted line numbers pinned in the test rather than in a comment.

### 11. Suggestion — make the limits of the stage-0 report explicit

**Resolution:** the report carries a "What this does not measure" section naming **laundering,
telephony, physical replay and wild-real false alarms** as untested, states that **V2-33 stays open**
pending its V2-01 benchmark row beside an unmined twin, and names the release blocker. The
`experiments/log.csv` row records it as a diagnostic; `submissions/log.csv` and v1's submitted TSV
are untouched.
