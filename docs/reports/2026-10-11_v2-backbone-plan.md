# Implementation plan: V2-53 bigger pretrained anti-deepfake backbones

**Run spec:** `docs/specs/2026-10-11_v2-backbone.md` (P1 and P2 frozen; amendments listed below).
**Branch:** `v2`, shared checkout. **Stakes:** HIGH.
**Planning:** deep mode — three parallel exploration agents, then an adversarial critique pass, then Codex plan review. Revision 2, after the critique found two unsatisfiable gates, nine orphaned tests and fifteen missing steps.
**Every number below is measured on this machine. Claims without a measurement or a `file:line` are marked as assumptions.**

---

## 1. What exploration and critique changed

### Corrections to the frozen P2

| # | P2 said | Measured reality | Fix |
|---|---|---|---|
| 1 | Join normalized with `fuse_sweep.norm` | **The trap is asymmetry, not the function.** Applied symmetrically, `norm` is lossless (20,000/20,000). Applied asymmetrically, **20,000 of 24,671 rows (81.1%) vanish**, because `data/librispeech`, `data/ljspeech` and `data/nsa/DiffSSD` are all symlinks and `norm` resolves relative paths but passes absolute ones through. ITW survives 3,000/3,000, so the damage is invisible | `E1` rewritten: **repo-relative keys**, not resolved ones (step 8) |
| 2 | fp16 primary (`D7`) | MPS working set is `recommended_max_memory()` = **13.32 GiB**, not 18. Naive fp16 drives allocation to 13.05 GiB and poisons the process for life: **74.1 s/forward vs 0.58 s**, and `empty_cache()` does not recover it. 2B at fp32 is **8.15 GiB** with 5.2 GiB headroom | **`D7` → fp32** (step 6) |
| 3 | `K7` tolerance 1e-4, cosine acceptable | fp16's numerical floor is `max abs d` = 0.127, only 5x below the q/k mis-map signature of 0.646. **Cosine is worse: the q/k swap passes at cos 0.99083** | `K7`: fp32 on CPU, gate `max abs d / absmax(ref)` at rel 1e-3, **never cosine** (step 5) |
| 4 | The pos_conv rename is the top risk | **It is not a risk.** torch's `_load_state_dict_pre_hook` remaps `weight_g`/`weight_v` even at `strict=True`. The real blind spot is **layer-index errors**: all 24/48 encoder layers are shape-identical, so a reversed or off-by-one order loads with zero missing and zero unexpected keys | `K8` downgraded; **`K18`** layer-order control added (step 5) |
| 5 | ~6–7 h, two stages | ITW is 2.5–4x faster than assumed (all four: **38.6 min**). The in-sample arm was unestimated: 8.1 h at full size | **three stages, stage 3 conditional** (§2) |
| 6 | `K15` validates the whole map offline | **Local `weights/wav2vec2-xls-r-300m` has zero `m_ssl` keys** — 429 tensors, all transformers-named with a `wav2vec2.` prefix, and its 7 pretraining extras are *differently named* (`quantizer.codevectors`, `project_hid.*`) than the AntiDeepfake 7. So the fairseq half of the map has nothing offline to consume, and **`K12`'s name assertion cannot run offline** | `K15` and **AC2** re-scoped to four achievable claims (step 5) |
| 7 | **`P3`: no verdict until a published number is reproduced** | **Deadlock.** There is no published MMS-300M ITW number at all, so stage 1 could never clear its own gate; and since the verdict is the lane's only deliverable, a failed `P3` forbade the output. There is also **no overlapping clip set** — their ITW is a 2,500-clip subsample of a 4,000-clip draw whose list was never published | `P3` split into an **in-sample ceiling gate** plus **external corroboration that cannot block** (step 7) |
| 8 | Pooling is "a free choice no test can catch" — ranked the top residual risk | **Overstated, against evidence we already held.** `APPROACH.md:279`: AntiDeepfake adds "mean pooling and a linear head with logits [fake, real]"; `:77` and `:537`: "time-mean of the last layer (the input of its linear head)". That is spec `D6` verbatim | `P1` becomes a documentation assertion citing those lines; the risk drops from top to moderate (step 7) |
| 9 | "one checkout, no worktrees" | A second worktree exists at `.claude/worktrees/v2-benchmark-fusion`, **its `outputs/` is a symlink to ours**, and its HEAD moved during planning — the fusion lane is active | `outputs/` is shared; MPS guard broadened (step 2) |

### Two asks withdrawn

The critique closed both items I was going to put to the user, correctly:

- **`.dockerignore`** is a non-issue. `D8` puts the weights on the external drive via a redirected HF cache, so nothing enters the repo or the build context. No decision needed.
- **`CLAUDE.md:108`** is not broken. It reads "Downloaded pretrained weights ... Nothing was trained or run on them **before the event**" — a claim about the pre-event window, which stays true. The real obligation is the AI-use disclosure that `A5`/`L4` already require, now step 13.

### The P1-level finding the user must rule on

**The size trend is below this population's noise floor.** Measured here, 1,000-resample speaker-cluster bootstrap over ITW's 54 speakers, paired Δ at package-0.5 (M1b L7 vs `_itw_m5`, 3,000 joined rows, 0 single-class draws):

```
p5 -0.1799   median -0.0932   p95 +0.0274   90% width 0.2073
=> smallest establishable gain ~0.104 at package-0.5
```

The winner's zero-shot ladder spans **2B 0.038 / 1B 0.050 / MMS-1B 0.065 = 0.027 total**, roughly a quarter of the floor. So `H7` ("monotone beyond the bootstrap interval") can essentially never confirm, and 54 speakers is the binding n — the In-the-Wild release has no more clusters to buy.

This contradicts the user's P1 decision: *"All four sizes, not one. The size trend is itself the finding; one model cannot show it."*

**What survives, and why scoring all four is still right.** The *primary* question is comfortably answerable: a candidate near 0.038 against our baseline's measured 0.193 is a ~0.155 gain at package-0.3, well clear of the floor. So the user's own P1 sentence — "does a larger backbone work better or not" — is answerable with room to spare. What is not answerable is *which size is best*. Four candidates still earn their place: they give the best shot at the primary question, a descriptive ladder with honest intervals, and no prejudgement of which size wins. The deliverable changes from "the trend is the finding" to "the ladder is descriptive; the primary comparison is candidate against baseline."

`H7` is therefore pre-declared to return "not established", and the report states the ~0.104 detectable floor so no reader mistakes the null for evidence of no trend.

---

## 2. Staging

Stage 3 is conditional. Scoring four models over the full index costs 8.1 h and three may be dropped; answer the question on samples, then spend the full pass only on a candidate that earns it.

| Stage | Work | Measured cost | Gate |
|---|---|---|---|
| **1** | Port; 81 tests; four offline claims; download MMS-300M (1.27 GB); archive cards; inventory; contamination record; **in-sample ceiling check (`P3a`)** | ~2 h incl. download | `K5`–`K18`, `P1`–`P2`, `W1`–`W4` green; `P3a` passes on ≥1 candidate |
| **2** | Download the other three (16.4 GB). Score four candidates × {ITW 3,000; NSA-inner sample 3,000; ASV19 train/dev sample 3,000; 5 perturbation cohorts 500 each} = 46,000 forwards | **~2.9 h forward** + ~25 min download + ~12 min IO + bootstraps/metrics ≈ **3.75 h** | differential, intervals and overfit gates computed; verdict written |
| **3** | **Conditional.** Full 24,671-row column, best candidate only | 300M 1.1 h · 1B 2.1 h · **2B 4.2 h** | only on a `column` verdict and if the benchmark lane wants the stream |

Forward arithmetic for stage 2: warm 8 s fp32 forwards sum to 0.110 + 0.290 + 0.290 + 0.615 = **1.305 s/clip across all four**. ITW's clips are short (only 11.3% exceed 8 s), measured at 0.772 s/clip summed → 38.6 min. The other three arms hit the 8 s cap, so ~1.305 s/clip → ~65 min each for in-sample and ASV19, ~35 min for the cohorts.

Sampling is right, for a reason the first draft did not give: measured, a 3,000-row stratified in-sample sample has a 90% width of **0.0607** against ~0.024 at full size — but the **ITW arm's width is 0.2073**, 3.4x wider, and it is the binding constraint on the differential. Sampling the in-sample arm costs almost nothing against the real bottleneck.

---

## 3. Execution steps

### Step 1 — Baseline snapshot (first, because AC12 is otherwise unverifiable)

The tree is **already dirty** with five modified tracked files outside this lane (`.claude/codex-audit-prompt.md`, `.claude/launch.json`, `docs/specs/2026-09-26_m5-xlsr-finetune-audit.md`, `next-env.d.ts`, `tsconfig.json`). Without a baseline the lane cannot distinguish its own changes from pre-existing ones.

```bash
git status --porcelain > outputs/v2/backbone/git_baseline.txt && shasum -a 256 submissions/log.csv >> outputs/v2/backbone/git_baseline.txt
```

### Step 2 — Environment guards

```bash
pgrep -fl "[p]ython" | grep -v "pgrep" || echo "no python processes"
```

Broadened deliberately: the MPS ceiling is a **device** property shared machine-wide, and the fusion lane runs from its own `.venv` in a sibling worktree. The P2 pattern would have missed `fuse_sweep_v4.py`, a training run, or a bare `uv run pytest` executing `needs_weights` MPS tests. Two concurrent 2B jobs = 18.4 GiB against a 13.32 GiB ceiling and **both** fall off the 120x cliff.

```bash
export HF_HOME="/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/hf"
```

Before the first download. `/` has **19 GiB** free against 16.4 GiB of weights; the external drive has **341 GiB**. This repo already lost three Docker builds to disk exhaustion (`docs/specs/2026-09-26_k-docker-image.md`).

`.python-version` pins 3.12 and pyenv lacks it, so bare `python` fails here. **Every documented invocation is `uv run`.**

Keep-awake for the unattended runs (stage 2 is ~3.75 h, stage 3 up to 4.2 h).

### Step 3 — Download and identity (`A1`–`A6`)

The step the first draft omitted entirely, which four tests depend on.

Per candidate: resolve the Hub revision sha and record it; download to `$HF_HOME`; verify each file's sha256 against the Hub LFS oid **before first use**; record the realpath; write `outputs/v2/backbone/<name>.identity.json` (model id, revision sha, file sha256, realpath, dtype, param count, download bytes).

**Archive the model cards** to `outputs/v2/backbone/cards/<name>.md` and the training-corpora table to `cards/<name>.training_corpora.json`. The spec's entire contamination table rests on cards read once from the web; nothing is saved in the repo today, so `M4`'s "membership evidence" and `A5`/`L4`'s licence disclosure would cite a page with no local artifact. Cheapest high-value fix in the plan.

### Step 4 — `src/hearsay/antideepfake.py`

New file. No new dependencies.

```
KEY_MAP: tuple[tuple[str, str], ...]        # the 14 patterns in the run spec
UNUSED: frozenset[str]                      # the 7 AntiDeepfake pretraining-only names
POOLING = "time_mean_last_layer"            # D6; cites APPROACH.md:279, :77, :537 + HF card
def map_keys(ckpt_names, n_layers, n_conv) -> (mapped, unused, unclassified)
def load_antideepfake(dir, device=None, dtype=torch.float32) -> AntiDeepfakeDetector
class AntiDeepfakeDetector(nn.Module)       # Wav2Vec2Model + Linear(hidden, 2) from proj_fc
def synth_logit(logits) -> float            # logits[0] - logits[1], VERIFIED not assumed
def reference_forward(ckpt, cfg, x)         # independent impl from fairseq-named tensors (K7)
```

Asserted on every load, never assumed:

- `len(mapped) + len(UNUSED) + 2 == len(ckpt_names)` — 431 at 24 layers, 815 at 48.
- All 422 (or 806) target parameters receive a tensor. **`load_state_dict(..., strict=True)`** — never `strict=False`, which returns silently and makes the caller read `.missing_keys`.
- Flags from the shipped `config.json` (`do_stable_layer_norm: true`, `feat_extract_norm: "layer"`, `conv_bias: true`). Never defaulted.
- **`.eval()` and `@torch.inference_mode()`.** The config carries `layerdrop: 0.1`, `apply_spec_augment: true`, `mask_time_prob: 0.075`, `hidden_dropout: 0.1`. In train mode the model is nondeterministic and silently degraded. House precedent: `src/hearsay/embed.py:31`, `:46`, `:105`; `src/hearsay/spectra.py:67`, `:116`, `:128`. The first draft omitted both.
- `pos_conv`: emit legacy `weight_g`/`weight_v` and let torch's pre-hook remap. Verified at `strict=True`.

### Step 5 — `tests/test_backbone_port.py` (Stages A, B, P, W — 29 tests)

**The four claims `K15` can actually make offline**, against local `weights/wav2vec2-xls-r-300m` (`pytorch_model.bin`, not safetensors — load through transformers):

1. **429 = 422 + 7** — the 422 target budget confirmed against a real file.
2. The map's target set equals the parameter set `Wav2Vec2Model(config)` exposes, built from the real shipped `config.json`, **prefix-stripped** (the local file prefixes everything `wav2vec2.`).
3. **The pos_conv legacy remap, proven on real data before any download** — this file carries `weight_g`/`weight_v` and zero `parametrizations.*`, so torch's pre-hook is testable here.
4. `K7`'s reference-forward equivalence and `K18`'s layer-order control on real 24-layer values, fp32 on CPU.

The fairseq-naming half of the map is first exercised on the MMS-300M download in stage 1. **AC2 is amended to say exactly that** rather than claiming the whole map validates offline.

`K7`, revised: fp32 on CPU; gate `max abs d / absmax(ref)` at **rel 1e-3**; never cosine. Negative controls with measured signatures — q/k swap in layer 0 (0.646), **layers 0/1 swapped (0.765)**, all layers reversed (1.09), against an fp32 floor of 1.42e-04.

`K9` rewritten: under the pre-hook approach there is no mapping to drop, so the counterfactual becomes — a state dict missing pos_conv raises under `strict=True`, and under `strict=False` `K6`'s coverage check names the three missing parameters.

**48-layer coverage.** `K7`/`K18` also run on the **1B** (3.86 GB × 2 ≈ 8 GiB, fits in 18 GiB). The **2B cannot** take a CPU reference forward — 8.65 GB checkpoint plus an independent second copy exceeds RAM — so it gets a single-encoder-layer equivalence plus `K18`'s order control, and the report states it is not end-to-end verified. `F6` is sequenced to release the MPS allocation before the CPU forward.

### Step 6 — Device and dtype (`D7` → fp32)

| Candidate | fp32 current / driver | warm 8 s forward | Fits under 13.32 GiB? |
|---|---|---|---|
| MMS-300M | 1.18 / 2.08 GiB | 0.110 s | yes |
| XLS-R-1B, MMS-1B | 3.59 / 4.19 GiB | 0.290 s | yes |
| XLS-R-2B | **8.15 / 9.18 GiB** | **0.615 s** | **yes, 5.2 GiB headroom** |

Measured on randomly-initialized models under `inference_mode`. fp16 buys ~12% and costs the cliff plus a `K7`-blinding noise floor. If ever needed: cast on **CPU** then `.to("mps")` — 5.07 GiB and 0.549 s, versus 13.05 GiB and 74.1 s naive.

**No padding, no batching.** Variable real ITW lengths are *cheaper* than padding to 8 s (0.103 vs 0.158 s/clip).

`F7` amended: fp16-vs-fp32 drift is measured on the **CPU-cast path, MMS-300M only**. Running it as originally specified would trigger the exact cliff this plan exists to avoid.

**Assumption, not measured:** that every op in the Wav2Vec2 graph has a native MPS kernel at both dtypes. What *is* measured: `PYTORCH_ENABLE_MPS_FALLBACK` is unset and nothing in the repo sets it, so an unimplemented op raises rather than silently falling back — which is what `F5` relies on.

### Step 7 — Stage P: pooling and usage (5 tests)

| # | Test |
|---|---|
| `P1` | `POOLING` is declared as a named constant citing `APPROACH.md:279`, `:77`, `:537`, the HF card and spec `D6`; asserted against the declared value |
| `P2` | All candidate poolings (time-mean, last-frame, max-over-time) computed on the **in-sample arm**; time-mean must dominate, and the margin is recorded. Replaces the near-vacuous "assert they differ" |
| `P3a` | **The real gate.** These are trained detectors whose post-training data we own. Pre-declared before the first read: **package-0.3 ≤ 0.05 and EER ≤ 2% on the in-sample arm for at least one candidate.** A miss is a usage bug, not a model finding. Costs nothing extra (that arm is already scored for `M1`), runs on MMS-300M in stage 1, needs no external number, and — being pre-declared on an in-sample population — does not touch rule 7 or `M6` |
| `P3b` | **Corroboration, never a bar.** Pre-declared: on the **native** view, XLS-R-2B ITW package-0.3 in **[0.019, 0.098]** (their 0.038, CI widened 2x for sample, crop and view) and ITW EER < 5% against their 1.44%. Lead with EER, which is prior-free. On a miss the report says "usage not corroborated" and the verdict carries `usage_unverified: true`. **The verdict still ships** |
| `P4` | Counterfactual for `P3b`: a candidate outside the band must set `usage_unverified: true`, matching the spec's `M4`/`M5`, `K8`/`K9`, `K13`/`K14`, `I1`/`I2` pattern |

`P3b` runs on the **native** view, not ours. `APPROACH.md` §3.2 states their input is "standardized per clip, as in the original recipe" — so running the anchor on our band-matched view would miss even with correct pooling. Their crop is irreproducible by design ("rebuild every evaluation clip's crop with the training RNG") and crop length dominates their ITW error, which is why this corroborates rather than gates.

### Step 8 — Input path, export and join (Stages C, D, E — step 7 of the first draft, merged here)

Merged deliberately: the first draft wrote the scoring script before deciding the path rules, so the script would have been rewritten.

Our view is `prepare_segment` (`src/hearsay/embed.py:80`): `band_limit` → `trim_silence` → optional crop → 8 s cap. `I3` pins that order byte-identically — M5's band-limit-before-trim bug showed as a 0.279 logit error that a probability-level check would have missed. ITW crop seed is **200** (`outputs/embeddings/wav2vec2-xls-r-300m/itw_stress_v3/extract_meta.json`).

**`E1`, three defences:**

1. **Emit a repo-relative key** — strip the `REPO/` prefix, do **not** resolve. The critique's refinement: a resolving normalizer makes every export key depend on the external volume's mount name, and macOS renames it to `... 1` after an unclean unmount. With 17.64 GB of weights on that drive across hours of runs, verbatim-relative keys survive where resolved ones do not.
2. **Normalize both sides identically.** Measured: symmetric `norm` is lossless (20,000/20,000); the 81.1% loss needs one side pre-resolved.
3. **Assert the exact expected intersection size after every join**, failing with the shortfall named, and validate **before** any dedup — `fuse_sweep.load`'s `.drop_duplicates("path")` (`scripts/fuse_sweep.py:43`) is what hides the trap, which is why `validate_export` (`scripts/fuse_sweep_v3.py:307`) exists.

Why three: a *total* mismatch crashes loudly in the metric, but a *partial* one returns real numbers from a biased subset, and `min_cost` on one class returns `nan` rather than raising. Ranking `nan`s picks a winner.

Source paths as the canonical writer does: inner/holdout from the fold file filtered to `splits/nsa_folds.csv` membership (`scripts/export_probe_scores.py:60`), test and itw from their own manifests with `fold=""`.

`E6` is new: `D10` had no test — assert `score == sigmoid(logit)` and that no calibration is applied to the export.

### Step 9 — `scripts/v2_backbone_score.py`

Mirror `scripts/score_spectra.py` (632 lines). Lifted: `_die` (`:135`), `_under` (`:141`), `_atomic_write_text`/`_atomic_copy` (`:145`), `_stamp` (`:157`), `config_hash` (`:215`), `_record` (`:241`), the `score_rows` decode-threads shape with `checkpoint()` (`:276`), `failure_table` (`:345`), `splits_over_gate` (`:362`, integer counts never the rounded rate), `_y` (`:341`).

**`run(argv=None, loader=load_antideepfake)`** plus `--repo-root` on every path (`score_spectra.py:477`) — the injection that lets all 77 hermetic tests run with a fake model and no weights, as `tests/test_spectra.py:75`'s `FakeSpectra` + `fail_calls` does.

Exit codes: `0` ok, `2` usage/config, `3` failure gate, `4` direction gate, `5` max-chunks.

Not copied: the pad-mode axis and `WIN`/`HOP`, `m1_disagreement` (`:450`), the hardcoded card threshold (`:386`), `rung: "M3"`, and publishing to `outputs/detector_scores/` (read-only for this lane). Writes `split="itw"` directly, not Spectra's `"stress"`.

Emits a `flag` column; the comparison step asserts flags empty (`scripts/bakeoff_layers.py:128`). Without it a mid-run decode failure on the external drive becomes a 0.0 score indistinguishable from a confident bonafide.

**No lightgbm.** `src/hearsay/trees.py:4-7` records that lightgbm-then-XLS-R crashes and the reverse hangs, and the guard test `tests/test_trees.py:69` scans only `src/` and `tests/` — nothing would catch it in a script.

### Step 10 — Populations and the pre-declared sample

**The in-sample arm is NSA inner at the 16,142 basis.** The first draft said "21,671 rows", which is 16,142 inner + 3,858 holdout + **1,671 unlabelled test** — a minDCF on an arm containing a label-free set is impossible, and folding holdout into inner breaks `M3`'s basis rule. Corrected: inner at a named basis; holdout reported separately; test is rank agreement only.

The spec's alternative "23,790" basis matches no file (`nsa_folds.csv` has 20,000; `nsa_folds_plus_asv19.csv` has 27,648). Amended: name the 16,142 basis, drop 23,790 unless its source is found.

**Pre-declare the sample as an artifact** before any read, per `H1`: seed, strata keys and the row list to `outputs/v2/backbone/sample_predeclared.json`. Stratify by **generator**, not "source and label" — those two keys are collinear in `nsa_folds.csv` (all 10,000 spoofs are diffssd; all bona fide are ljspeech/librispeech), so stratifying by source guarantees nothing per generator.

**Build the ASV19 train/dev manifest** (`D9`). No such manifest exists: `nsa_folds_plus_asv19.csv` holds only 7,648 asv19 rows folded into training folds 0–4, which is neither "train/dev only" nor the counts `D9` names. The audio is present (`LA_T` 25,380 flac; `LA_D` **24,986** on disk against the protocol's 24,844 — state the basis, which is exactly what `M3` exists to prevent).

### Step 11 — Metrics (Stage F — 11 tests)

Measured traps:

- **`report()['min_dcf_pi05']` is not package-0.5.** It is `min_cost(y,s,0.5)` with default costs = 4·P_FA + P_miss. Measured 0.5448 against 0.5229.
- **`report()['sponsor_code_flipped']` is package-0.5 but rounded to 4 dp** (0.5229 vs 0.522857142857143) — unusable for bootstraps or the differential. **Call `min_cost` directly.**
- `min_cost` **clamps at 1.0**: inverted and random detectors both read exactly 1.0. That means "no better than the constant decision", not "break-even".
- **No package-0.3 helper exists.** Write it. Verified it equals the winner's stated FPR + 1.714·FNR formula to 5.6e-17, so there is no convention gap with the external anchor.
- **`by_group` hardcodes the brief costs** (`src/hearsay/metrics.py:113`); `pi` is the only knob, and it needs a fresh `RangeIndex` (`src/hearsay/spectra.py:244`). `H6`'s package-0.5 per-generator table needs its own implementation.

`M2` pins `min_cost(y,s,0.5,c_fa=1.0,c_miss=4.0)` identical to `sponsor_min_dcf(y,s,flip=True)` — measured to 1.11e-16.

Two new tests the critique found missing:

- **`M10`** — AC10a says "every headline number carries a bootstrap interval", but `H3` bootstraps a *gain*, not each absolute. `M10` asserts an interval on every headline number.
- **`M11`** — rule 5 requires **EER beside every reading**, and the first draft's metrics step never mentioned EER. `M11` asserts it.

### Step 12 — Overfit stage (Stages H, G — 16 tests)

**`H3` needs its own bootstrap.** `boot_gain` (`scripts/bakeoff_layers.py:219`) has the right shape — injectable `cost=`, `BOOT_N` 1000 — but lacks two guards `cluster_bootstrap` (`scripts/fuse_sweep_v3.py:239`) has, and rounds to 4 dp: NaN/blank group values must raise (`:243`), and single-class draws must be skipped **and counted** (`:255`) because `roc_curve` returns `nan` rather than raising. Combine both, full precision.

**`H11`** is new: a stratified bootstrap on the **in-sample arm**, propagated with the ITW arm into the Δ interval. The first draft claimed "`H3` reports by how much sampling widens the differential" — false, since `H3` only bootstraps ITW's 54 speakers. Nothing put an interval on the in-sample arm.

**`H6` is degenerate on ITW and biased on the sample.** Measured: ITW's `generator` has exactly two values (`bonafide`, `itw_spoof`) and `source` exactly one — **the worst group *is* the mean**. It only has content on the in-sample arm, where a 3,000-row sample leaves 116–144 spoof clips per generator against 1,000 at full size; worst-of-ten is a max statistic, so it is biased upward and more so at n≈130. Fix: stratify by generator (step 10) **and** compare worst-group against a baseline recomputed on the *same* sampled rows.

**`H7`** pre-declared "not established", with the ~0.104 floor stated in the report (§1).

**`H5`** re-derives cohorts with `scripts/m3_probes.py:68 perturb()` — five kinds, 250/class, selection reproduced from `bakeoff_layers.py:75-80`. The stored `.npz` files are XLS-R-300M embeddings, useless here, but their `row` arrays give the identical 500 clips. Reproduce the `bakeoff_layers.py:270` sanity check: the `none` cohort must match full-ITW scores on those rows to ~1e-6, which is what proves the perturbation and main paths share an input pipeline. **Ordering:** `H5` therefore runs after the main ITW pass, not alongside it.

**`H9`** is the counterfactual that makes the stage real: a synthetic candidate near-perfect in-sample and at chance on ITW must be flagged by `H1`–`H8`.

Do not copy `bakeoff_layers.py`'s `n_jobs=6` (`:158`) — this lane's cap is 4.

**`L5`** is new: assert the working tree's changes against step 1's baseline, which is what makes AC12 checkable on an already-dirty tree.

### Step 13 — Logging, docs, commit

No v2 appender exists; this lane is first to write `experiments/log.csv` (header-only, 7 columns). `hearsay.submission.append_log:113` writes 6 and lives in v1 code this lane may not edit, so write a lane-local appender. **Define `validation_score` for a provisional readout**: package-0.5 on the named population, with the basis in `notes`. `benchmark_version` records that V2-01 does not exist.

`L1` is a module-scoped autouse fixture pointing tests at a temp log and asserting tracked `submissions/log.csv` is **byte-identical** — the remedy adopted after M5 round 11 appended six fabricated rows to a tracked ledger. No `conftest.py` exists and `pyproject.toml` is outside this lane, so the fixture lives in the test module; markers are the three registered (`needs_data`, `needs_weights`, `slow`) with an inner `pytest.skip` on the checkpoint path (`tests/test_speaker_drift.py:91-95`).

**Documents** (the promised five, all now assigned):

- `docs/reports/2026-10-11_v2-backbone.md` — inventory, contamination record citing the archived cards, three readings per population with intervals, the differential with both arms' uncertainty, the overfit audit, worst-group beside the mean, the shortcut audit, the detectable floor, and a verdict carrying N, `provisional: true` and `usage_unverified` if `P3b` missed.
- `docs/prod-chunks.md` **lines 610–623 only**. Status → 🟡, not ✅: the legend at `:24` defines ✅ as "done (v1)", no V2 chunk is ✅, and `:24` is outside this lane's range. **The chunk's own "Done when" requires a V2-01 benchmark row that does not exist**, so the same hunk records that the chunk stays open pending V2-01, and the report says so.
- **`CLAUDE.md`** — the AI-use disclosure entries (`A5`/`L4`), appended after line 147, following the `CLAUDE.md:147` format, with CC BY-NC-SA 4.0 phrased as the existing MLAAD non-commercial precedent at `:153`. The first draft omitted this promised document.
- The run spec's amendment section and break-check record.
- `README.md` only if a verdict changes shipped status (unlikely; provisional).

`outputs/` is gitignored, so committed evidence is the report with every number citing its file (`docs/prod-chunks.md:6`).

**Commit** explicit paths only. `docs/prod-chunks.md` goes in only if `git diff` shows only this lane's hunks — and `outputs/` is shared with an active sibling worktree, so check rather than assume.

### Step 14 — Codex plan review

```bash
bash .claude/review-plan.sh docs/reports/2026-10-11_v2-backbone-plan.md
```

Writes `docs/reports/2026-10-11_v2-backbone-plan-review.md`. `codex` is installed at `/usr/local/bin/codex`, version 0.154.0 ≥ the 0.153.1 the script's pin requires. Exit 2 means empty output.

### Step 15 — Test break check (post-commit)

Per `~/.claude/ownership-checks.md` §3: throwaway detached worktree **outside** the repo, mutate `q_proj`/`k_proj` in the key map, run only `K7`, expect CAUGHT, remove the worktree, confirm the main checkout is unchanged. Record under the spec's `## Test break check`. Never commit the mutation.

### Step 16 — Post-commit audit

Claude critique loop to Acceptable, then `bash .claude/review-audit.sh docs/specs/2026-10-11_v2-backbone.md`. Note `.claude/codex-audit-prompt.md` is uncommitted in the working tree — the rubric is another lane's pending change, so read it rather than assume it.

---

## 4. Test inventory — 81 tests, all assigned

| Stage | IDs | n | File |
|---|---|---|---|
| A acquisition and identity | A1–A6 | 6 | `test_backbone_port.py` |
| B strict key map | K5–K18 | 14 | `test_backbone_port.py` |
| P pooling and usage | P1, P2, P3a, P3b, P4 | 5 | `test_backbone_port.py` |
| W real weights | W1–W4 | 4 | `test_backbone_port.py` |
| C input path | I1–I8 | 8 | `test_backbone_score.py` |
| D forward and scoring | F1–F11 | 11 | `test_backbone_score.py` |
| E export and join | E1–E6 | 6 | `test_backbone_score.py` |
| F metrics and readout | M1–M11 | 11 | `test_backbone_score.py` |
| H overfit and generalization | H1–H11 | 11 | `test_backbone_overfit.py` |
| G logging and ownership | L1–L5 | 5 | `test_backbone_overfit.py` |

29 + 36 + 16 = **81 total, 77 hermetic, 4 weights-gated.** The first draft orphaned nine (E1–E5, L1–L4) and W1–W4 with no file or step.

```bash
uv run pytest tests/test_backbone_port.py tests/test_backbone_score.py tests/test_backbone_overfit.py -q
```

```bash
uv run pytest -q && uv run ruff check .
```

Then: 856 + 81 pass, 3 skipped (plus 4 without weights), ruff clean, the diff touches only this lane's paths against step 1's baseline, and `submissions/log.csv` is byte-identical by sha.

---

## 5. Risks and mitigations

| Risk | Mitigation | Residual |
|---|---|---|
| 81.1% silent row loss on the join | repo-relative keys + symmetric normalization + asserted intersection (step 8) | none for the loss; mount-name exposure removed by not resolving |
| 120x MPS cliff | fp32 (8.15 GiB, 5.2 GiB headroom) | **a concurrent MPS job in the sibling worktree still breaches the shared 13.32 GiB ceiling.** Step 2's broadened guard is the only protection, and the fusion lane is active |
| Layer-order mis-map loads clean | `K18` control; `K7` at fp32, rel-1e-3 on `max abs d` | **the 2B is not end-to-end verified** — a CPU reference forward exceeds 18 GiB RAM. Single-layer equivalence only, stated in the report |
| Wrong pooling | `P1` cites the documented recipe; `P2` shows time-mean dominates; `P3a` gates on the in-sample ceiling | moderate, down from "top". `P3b` corroborates but cannot gate, since their crop is irreproducible |
| Size trend unanswerable | pre-declared "not established"; floor stated; ladder descriptive | **accepted limitation; needs the user's ruling** (§1) |
| `H6` degenerate on ITW, biased on the sample | stratify by generator; baseline recomputed on the same rows | ITW worst-group stays uninformative by construction |
| 8.1 h in-sample scoring | stage 2 samples 3,000; full column conditional | `H11` puts an interval on the sampled arm |
| Decode failure becomes a 0.0 score | `flag` column + flags-empty assertion | none |
| Disk exhaustion | `HF_HOME` on the external drive | none |
| `outputs/` shared with an active lane | step 1 baseline + `E5` + `L3`/`L5` | we cannot protect our files from the other lane; stated in the report |
| V2-53's "Done when" needs a non-existent V2-01 | chunk stays 🟡 and open; recorded in the hunk and the report | open by design, not by omission |
| lightgbm in a script would segfault | no import; `hearsay.trees` if ever needed | the guard test still does not scan `scripts/` |

---

## Revision 3 — Codex plan review, findings addressed

`bash .claude/review-plan.sh` returned 13 findings (9 critical) into
`docs/reports/2026-10-11_v2-backbone-plan-review.md`. Each is addressed below; where a finding
changes a step above, the change here supersedes it.

### C1 — The detection floor was invalid (and my reporting of it was wrong)

**Accepted; my error.** The 0.2073 width belongs to the M1b-vs-M5 pair at package-0.5. I then compared it
against the winner's ladder spread of 0.027, which is a **package-0.3** figure. A cross-reading comparison.

Re-measured, paired 90% widths on ITW's 54 clusters:

| pair | package-0.5 | package-0.3 |
|---|---|---|
| M1b vs M5 | 0.2073 | 0.1009 |
| M1b vs handcrafted v5 | 0.3364 | 0.2498 |
| M1b vs WavLM L9 | 0.1577 | 0.1277 |

Widths vary by more than 2x **by pair**, because paired uncertainty depends on which speakers each model
gets wrong. **No universal floor exists.**

**Change:** `H7` stays **executable**. Compute candidate-specific paired intervals and report the measured
width for each comparison. **Do not pre-declare the outcome.** The honest statement is an expectation, not a
result: given a published ladder spread of ~0.027 and observed widths of 0.10-0.34 on this population, the
trend may well come out "not established" — but that is for the data to say.

**This substantially restores the user's P1 decision.** The earlier claim that measurement had undermined
"the size trend is itself the finding" rested on the unit error. Scoring all four is sound, and whether the
ladder resolves is an open empirical question.

### C2 — Stage 1's gate was unreachable

**Accepted.** Stage 1's gate required `K5`-`K18` green, but `K7`/`K18` on the 1B need a stage-2 download,
and `P3a` needs scoring plus a sample that later steps build.

**Change:** gates become stage-specific. **Stage 1:** the 24-layer map (`K5`-`K17` offline + MMS-300M),
`P1`, `W1`-`W4`. **Stage 2:** `K7`/`K18` on the 1B, `P2`, `P3a`, `P3b`. The sample pre-declaration
(step 10), the ASV19 manifest (step 10) and a minimal scorer move **before** stage 1's evaluation.

### C3 — `P2` and `P3a` turned expectations into correctness assertions

**Accepted, and this is the most important correction.** Documented mean pooling need not beat alternatives
on every sample, and corpus membership does not establish that *these clips* were training examples or
guarantee EER <= 2% after **our** preprocessing. A correct port could fail both gates.

**Change:** both become **diagnostics with a bounded stop-and-report**, not pass/fail correctness gates.
`P3a` reports the in-sample ceiling and, if a candidate misses it, the lane **stops and reports** for
investigation rather than declaring a usage bug. `P2` records the margin between poolings without
asserting dominance. Implementation correctness rests on `K7`'s independent reference forward, which is
the only test that actually establishes it.

### C4 — The input recipe contradicted the frozen candidate definition

**Accepted; I missed a frozen-spec contradiction.** The spec declares `repeat` primary with `zero`/`whole`
as diagnostics and `I5`/`W2` testing all three; the plan removed the pad-mode axis.

**Change, recorded as a spec amendment:** Spectra's pad modes exist because AASIST has a fixed
64,600-sample window; a wav2vec2 encoder has no such constraint, so the declared policy is
**whole clip, no padding, 8 s cap**. `I5`/`W2` are amended to test that policy's short-clip behaviour
instead — including clips shorter than the convolutional receptive field, which is a real boundary the
pad-mode framing hid. The native view's crop, normalization and minimum length are specified explicitly
rather than inherited.

### C5 — In-the-Wild still drove selection, breaching rule 7

**Accepted; the sharpest finding.** Stage 3 exported "the best candidate" chosen on ITW verdicts, and `H5`
excluded candidates from a column recommendation using ITW perturbations. That is selection through
aggregate metrics even without fitting on ITW rows, and `provisional: true` does not fix it.

**Change:** **this lane makes no selection.** It reports per-candidate numbers on every population and
ranks nothing on ITW. Stage 3 exports whichever candidate the **benchmark lane** requests against its
sealed population — not "the best" by our reading. `H5`'s perturbation result becomes a **reported flag**,
never an exclusion. Where a selection basis is unavoidable it is **inner OOF only**, carrying its in-sample
caveat, per the evaluation discipline.

### C6 — Verdict rules were incomplete

**Accepted.** `H5`'s "declared margin" was never supplied; there was no mapping to column/suppressor/drop,
no tie-breaking, and no non-regression bands across all three readings as rule 5 requires.

**Change:** freeze before scoring, in the run spec: the perturbation margin, the full verdict mapping,
the tie band, and a non-regression band per reading. Boundary cases get tests. Inference is declared
**exploratory, not simultaneous** — recording N does not correct for selecting the strongest of four
intervals, and the report says so.

### C7 — `H11` did not define group-aware uncertainty

**Accepted.** "Stratified bootstrap" could mean resampling clips within generators, ignoring
within-generator and within-speaker dependence.

**Change:** resampling units are the fold file's own `group` column (`gen:`, `lj:`, `spk:`,
`asvspoof2019:` prefixes). The **single LJ speaker** is named as a degenerate cluster and its treatment
stated. Candidate and baseline use **identical resamples**. The report distinguishes uncertainty
conditional on these corpora from uncertainty about unseen generators and speakers.

### C8 — The runtime budget omitted a required view and had no enforceable cutoff

**Accepted.** `D5` and `I8` require **both** the native and band-matched views, so 46,000 forwards is half
the work. Holdout/test diagnostics and real-weight verification were also missing.

**Change:** stage 2's forward budget **doubles to ~5.8 h** before IO, downloads, bootstraps and metrics.
Combined with stage 1 and a conditional stage 3, the honest figure is **a working day, not an afternoon**.
Each stage gets an **elapsed-time deadline that checkpoints, stops and reports** — an estimate is not a
time box, which is the project's own standing rule.

### C9 — The MPS guard could not enforce exclusivity

**Accepted.** A one-time `pgrep` neither blocks execution nor stops the active sibling lane starting
afterwards.

**Change:** a lockfile reservation under the shared `outputs/` tree, acquired for the whole span of
scoring and weight-dependent tests, with an explicit **wait-or-abort** path, plus a note to the fusion
lane. And **measure peak memory while loading the real 2B checkpoint** — the step-6 table used
randomly-initialized models, which does not capture checkpoint tensors and model allocations coexisting.
The 5.2 GiB headroom figure is provisional until that measurement exists.

### C10 — The differential is not an identified inflation estimate

**Accepted.** An NSA-versus-ITW gap mixes training exposure with corpus difficulty, channel, speakers and
generator composition; subtracting M1b's gap does not isolate memorization.

**Change:** reported as a **descriptive generalization-gap difference**, with the confounds named. The
stronger "inflation estimate" reading is used only if a matched-condition membership control supports it.
The run spec's `M1` wording is amended accordingly. This qualifies but does not remove the answer to the
owner's P1 risk — the gap is still the right thing to measure, it is just not a clean causal estimate.

### C11 — The "77 hermetic" claim conflicted with the test definitions

**Accepted.** `K15` needs local gitignored weights, `K7`/`K18` need real checkpoints, and `P2`/`P3a`/`P3b`
need labelled audio and measured performance. Loader injection does not make those hermetic.

**Change:** reclassified into **synthetic unit tests** (fake model, no assets) and **marked
data/weight integration tests** (`needs_weights` / `needs_data` plus an inner skip). The break-check
worktree must reach its fixtures without downloading or re-running the evaluation, so the break-check
test is chosen from the synthetic set — `K7` qualifies only if its reference forward can run on the local
300M, which step 5 claim 4 already requires.

### C12 — The 23,790 basis does exist; my amendment was wrong

**Accepted; my error.** Verified: `splits/nsa_folds_plus_asv19.csv` has 27,648 rows and
**23,790 non-holdout**, exactly the handoff's figure. The spec amendment claiming it "matches no file" is
corrected. Keeping the NSA-only 16,142 basis remains the right choice, and `M3` still requires every inner
number to name its basis — but the alternative is real, not phantom.

### C13 — Robustness verification left named failure modes uncovered

**Accepted.** `perturb()` covers MP3, noise, speed and shift; it does not cover telephony or physical
replay, and the input tests omitted sample-rate, channel, PCM-scaling and decoder-padding boundaries.

**Change:** boundary fixtures added through `hearsay.audio.load_audio`; a telephony cohort added; physical
replay **stated as untested** in the report rather than left implicit. `H8`'s shortcut audit extends beyond
its four continuous features to **container and single-speaker dependence**.

---

## Revised test inventory after the review

Stage-specific gates (C2), pad-mode policy change (C4), reclassification (C11) and the additions in C13
move the count. Final: **81 IDs** as tabulated above, plus boundary and telephony cases folded into `I5`,
`W2` and `H5` rather than new IDs — and the hermetic/integration split restated:

- **synthetic unit tests (fake model, no assets):** the A, E, F, M, L and H stages, and the key-map
  structure tests in B
- **integration tests (marked, with inner skips):** `K7`, `K15`, `K18`, `P2`, `P3a`, `P3b`, `W1`-`W4`

The earlier "77 hermetic" figure is withdrawn.
