# Plan: M4 fusion v2, M5 as a live scorer in the runner (Sat Sep 26, 2026, written 10:10, revised 10:20 after the critique agent)

Run spec: `docs/specs/2026-09-26_m4-fusion-v2-m5-scorer.md` (decisions D1–D10 with the 10:15 amendments, tests T1–T25, acceptance A1–A11). Worktree `main` at `28d26fe` (later commits by other lanes do not touch the cited files; the critique agent re-verified every line number at `244064d`). Baseline: 452 passed (full suite, 09:45). Deep mode: three exploration agents (architecture, file impact, risk) and a critique agent; the critique's sixteen findings are folded in below and listed in the spec's amendments.

## Ground truth the plan rests on (measured by the agents, read-only)

- The v2 candidate rule reproduces both 09:14 candidate TSVs from the exports with max |Δ| 6.7e-16 (our direction) and 7.2e-16 (flipped) when the blend is accumulated in weights order as `base = 0.0; for d in ranked: base += w[d] * rank[d]`. For v1 that loop is provably bit-identical to today's `(1 - a) * r1 + a * rh` (`0.0 + y == y`, `1.0 - 0.2 == 0.8`; 0 mismatches over all 16,143 × 9 v1 rank pairs). Never `np.dot` or `sum()`. The v2 rule is one ulp off the sweep script's `(0.8 - w)` (0.6000000000000001 vs the file's rounded 0.6), which is the 6.7e-16.
- Perturbing every M5 test logit by the recorded live-vs-export gap (±0.012): max |Δp| 0.0017, 0 verdict crossings, 0 E-rule flips; 5 rows within 0.003 of the base boundary, 4 within 0.05 of the M3 margin boundary. The A4 gate (max 0.005, mean 5e-4, Spearman 0.9999, no flips) is strict but passable; the M5-logit gate (≤ 0.02) catches a transform-order regression (the pre-fix bug read 0.279).
- `deploy_transform` then `collate` normalizes twice; the export did the same (`scripts/m5_train.py:186-197, 434-435`); skipping `collate` moves logits by up to 2.5e-5. Zero clips, sub-second clips and clips over 8 s are finite; an input under 400 samples raises inside the XLS-R CNN and `analyze_clip`'s per-scorer `except Exception` (`pipeline.py:584-585`) turns that into an error entry, imputed at 0.5 (T24 pins it; D6's loudness clause makes a systematic failure visible).
- `load_m5` verifies three hashes on every load (~11 s cold for 657 MB); eval mode, `layerdrop=0.0`, `apply_spec_augment=False`; fp32 throughout; the export is batch 16 on the A100 (the `full` model, `m5_assemble.py:176`).
- `open_cache` (`run_pipeline.py:254`) compares `header.get(k) == v` for every identity key except `created`, so an old v1 header without `m5` still matches a v1 identity with `m5=None`; any non-None value moves the file aside.
- `hearsay.pipeline` imports no torch at module top; `from hearsay.pipeline import M5_DIR` in the sweep script is side-effect-free. `.gitignore` covers both ends of the candidate move. Nothing in `docker/` or `tests/test_docker_image.py` loads every staged fusion file; a staged `fusion_v2` in an image without the checkpoint is inert (`--fusion` to it fails loudly on the missing `hashes.json`).
- `speaker_drift.py:67` halves torch's thread count globally when ECAPA first loads. Not this lane's file; M5's per-clip time is reported as measured.

## Schedule and stops (10:20 → 15:00)

| Step | Budget | Lands by |
|---|---|---|
| 0 pre-flight + Docker heads-up | 10 | 10:30 |
| 1 sweep script gate, `--write`, compare, move | 25 | 10:55 |
| 2 loader | 35 | 11:30 |
| 3 M5 scorer, gate, entries, `analyze_clip` | 40 | 12:10 |
| 4 runner | 45 | 12:55 |
| 5 `tests/test_pipeline.py` | 50 | 13:45 |
| 6 runner-subset suite + ruff | 10 | 13:55 (**stop 1: 14:00**) |
| 7 live 50-file v2 and v1; full run started in the background | 20 | 14:15 (**stop 2: A4 by 14:30**) |
| 8 API + `tests/test_api.py` | 25 | 14:40 |
| 9 full suite + ruff | 10 | 14:50 |
| 10 docs, commit, messages | 25 | 15:15 |
| 11 critique loop, Codex audit | 30 | 15:45 |

**Final deadline 15:45 for everything**: the full-set run (started at step 7, ~35 min) must have landed with A5 met by 15:00; the critique loop is at most two rounds and the Codex audit one round plus one fix commit. At 15:45 whatever gate is still open is reported as open in the run spec, with the fallback stated: option 2 (ship the candidate TSV, image on `e_on_a`) needs none of this, and the branch stays uncommitted if stop 1 or stop 2 was missed.

## Step 0. Pre-flight

1. `git branch --show-current` = `main`; `pwd` = `/Users/nathanstough/Projects/hearsay`; `git status --short` shows only the other lanes' known uncommitted files (`scripts/cloud/launch.sh`, `scripts/cloud/reaper.sh`, `tests/test_cloud_scripts.py`, the two `m5-xlsr-finetune` spec files) plus this lane's spec and plan. Never stage theirs.
2. Tell the Docker chat (SendMessage, "Docker handoff plan review"): the runner is about to change (M5 as an optional fourth scorer, `fusion_v2` executable via `--fusion`, default unchanged), the commit is backward compatible, nothing to stage until the switch, their next build will list `fusion_v2` in BUILD_INFO and the file is inert without `models/m5_shipped`. Not their approval, a notification.
3. `cp models/fusion_v2_candidate/constants.json <scratchpad>/fusion_v2_candidate_0914.json` (the 09:14 original; step 1 compares against it). Do **not** run the sweep script yet (at HEAD it rewrites that file).
4. `uv run python scripts/fuse_sweep.py` (read-only for `models/`; it writes `outputs/fusion/sweep_report.json`) must print inner 0.1395 / holdout 0.014 / ITW brief 0.2603 for `E_on_A_alpha0.2`: the export-drift tripwire.

## Step 1. `scripts/fuse_sweep_m5.py`: gate first, then write

Current: no argparse (L1-22 docstring, imports, dynamic import of `fuse_sweep.py` as `fs`); `main()` L23; winner block L101-126 writes `outputs/fusion/sweep_m5_final_test.csv` (L106-110) and `models/fusion_v2_candidate/constants.json` (L112-126) unconditionally; report JSON L127-129.

1. Add `import argparse`; in `main()` parse `--write` (`action="store_true"`, help "write models/fusion_v2/constants.json (the runner's file); without it nothing under models/ is written"). Docstring L1-6: usage `[--write]`; the report and `sweep_m5_final_test.csv` are always written, the constants only with `--write`.
2. Keep L101-110 unconditional. Wrap L112-126 in `if args.write:`; target `cdir = REPO / "models" / "fusion_v2"`; drop `"status"`; add `"how": "rank_d = searchsorted(rank_ref[d], logit_d)/len; base = sum(weights[d] * rank_d) over m1b_v3, handcrafted_v5, m5_xlsr_ft, accumulated in that order; if m3_logit < -3 and base > 0.5: base *= 0.5; p = determinate_map(base)"` and `"m5_checkpoint": {"dir": str(M5_DIR.relative_to(REPO)), **{k: hashes[k] for k in ("backbone_sha256", "head_sha256", "config_hash")}}` with `hashes = json.loads((M5_DIR / "hashes.json").read_text())`, `from hearsay.pipeline import M5_DIR` (M5_DIR is added in step 2.2; do step 2.2 first, or define the path locally and switch to the import at step 2). `weights` stays `{"m1b_v3": round(0.8 - w, 2), "handcrafted_v5": 0.2, "m5_xlsr_ft": w}`; the print at L126 names the new path.
3. Read-only run: `uv run python scripts/fuse_sweep_m5.py` must print `{'inner_oof': 16142, 'holdout': 3858, 'test': 1671, 'itw': 3000}`, `t_F = -3.8985`, A3_w0.2_E inner 0.1351 / holdout 0.0065 / ITW brief 0.228 / averse 0.2385, winner `A3_w0.2_E`; afterwards `ls models/` shows no `fusion_v2` and `cmp models/fusion_v2_candidate/constants.json <scratchpad>/fusion_v2_candidate_0914.json` is silent (A9).
4. `uv run python scripts/fuse_sweep_m5.py --write`, then a scratch comparison (read-only) of `models/fusion_v2/constants.json` against the 09:14 original: weights equal; each reference `np.array_equal`; `platt.a`/`platt.b` deltas printed and ≤ 1e-9 (A2); `how`, `m5_checkpoint`, `pi_synth` 0.3 present. Record the deltas in the run spec.
5. `mkdir -p outputs/fusion && mv models/fusion_v2_candidate outputs/fusion/fusion_v2_candidate` (the original; both paths gitignored). Record the move.

## Step 2. `src/hearsay/pipeline.py`: constants and the loader

1. Module docstring L2 "the two deep scorers" → "the deep scorers"; L18-29: add a `fusion_v2` bullet (`scripts/fuse_sweep_m5.py --write`, `A3_w0.2_E`: `weights` over m1b_v3, handcrafted_v5, m5_xlsr_ft; same M3 step; the default file is still `fusion_v1`) and an `m5_xlsr_ft` score-path bullet (`hearsay.m5_data.deploy_transform` → `collate` → `hearsay.m5_model.score_batch`, the `scripts/m5_score.py` sequence; trim before band-limit, cap 8 s, no fp16, no truncation).
2. Paths L62-68: add `CONSTANTS_V2_PATH = REPO / "models" / "fusion_v2" / "constants.json"`; **`DEFAULT_CONSTANTS_PATH = CONSTANTS_V1_PATH` unchanged** (T19; `tests/test_docker_image.py:112,114` pin it). After `HC_DIR`: `M5_DIR = REPO / "models" / "m5_xlsr_ft_20260926-0741" / "model"` with a comment naming the export it produced (byte-identical to that dir's `scores.csv`).
3. L75 `DETECTOR_ORDER = ("m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "spectra_aasist")` ("every fused column, canonical order; a constants file names the subset it uses"). L76 delete `RANKED`. L77 `SCORER_FLAGS` add `"m5": "m5_xlsr_ft"`. L79 `DEEP = ("m1b_v3", "m5_xlsr_ft", "spectra_aasist")` (`collect_logits` L622 keys on it). L87-91 `ROLE` add `"m5_xlsr_ft": "fused"` (else an errored M5 entry is labelled "evidence" via `_error_entry` L418-420 → L411).
4. `FusionConstants` fields L110-128: add `ranked: tuple[str, ...] = ()`, `rank_weights: dict[str, float] | None = None`, `m5_checkpoint: dict[str, str] | None = None`. Keep `alpha`.
5. `_from_v1` L136-160, the middle becomes:
   ```python
   refs = c["rank_ref_inner_oof_sorted"]
   alpha, weights = c.get("alpha_handcrafted"), c.get("weights")
   if (alpha is None) == (weights is None):
       raise ValueError("constants: give exactly one of alpha_handcrafted (fusion_v1) or weights (fusion_v2)")
   if weights is None:
       if not 0.0 <= float(alpha) <= 1.0:
           raise ValueError(f"constants: alpha_handcrafted must be in [0, 1], got {alpha!r}")   # test L153-156 matches "alpha"
       weights, alpha = {"m1b_v3": 1.0 - float(alpha), "handcrafted_v5": float(alpha)}, float(alpha)
   else:
       weights = {d: float(v) for d, v in weights.items()}
       bad = [d for d in weights if d not in DETECTOR_ORDER or d == "spectra_aasist"]
       if bad or not weights:
           raise ValueError(f"constants: weights must name fused columns other than spectra_aasist, got {sorted(weights)}")
       if not all(math.isfinite(v) and v > 0 for v in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-9:
           raise ValueError(f"constants: weights must be finite, > 0 and sum to 1, got {weights}")
   ranked = tuple(d for d in DETECTOR_ORDER if d in weights)
   rank_ref = {}
   for d in ranked:
       if d not in refs:
           raise ValueError(f"constants: no rank reference for {d!r}")
       ref = np.asarray(refs[d], dtype=np.float64)
       if ref.ndim != 1 or ref.size == 0 or not np.all(np.isfinite(ref)):
           raise ValueError(f"constants: rank reference for {d!r} must be a non-empty 1-D array of finite values")
       if not np.all(np.diff(ref) >= 0):
           raise ValueError(f"constants: rank reference for {d!r} is empty or not sorted")        # test L149-152 matches "sorted"
       rank_ref[d] = ref
   ck = c.get("m5_checkpoint")
   if "m5_xlsr_ft" in ranked and (not isinstance(ck, dict)
                                  or any(k not in ck for k in ("dir", "backbone_sha256", "head_sha256", "config_hash"))):
       raise ValueError("constants: a rule that weights m5_xlsr_ft must record m5_checkpoint {dir, backbone_sha256, head_sha256, config_hash}")
   ```
   then the existing `platt`/`e_rule` checks (L150-155) and the constructor with `detectors=(*ranked, "spectra_aasist")`, `alpha=alpha` (None for v2), `ranked=ranked`, `rank_weights=weights`, `m5_checkpoint=dict(ck) if ck else None`, `how=c.get("how")`.
6. `weights()` L211-219: rank branch returns `{**self.rank_weights, "spectra_aasist": 0.0}` (v1 → exactly the three keys of test L191-192).
7. `fuse()` L236-273: L243 `for d in self.ranked:`; L249-250 →
   ```python
   base = 0.0
   for d in self.ranked:            # weights order; bit-identical to (1 - a) * r1 + a * rh for a v1 file
       base += self.rank_weights[d] * terms[d]
   ```
   Imputation, the M3 strict `<`/`>`, `detail`, Platt unchanged.
8. `fusion_line()` L447-451 →
   ```python
   terms = " + ".join(f"{w[k]:.1f} x rank({k})" for k in fo.terms)   # rank rule: fo.terms holds exactly the ranked detectors, in order
   line = f"fusion: {fo.rule} ({d.get('final')}): {terms} = {d.get('base', fo.fused):.3f}"
   ```
   v1 output character-identical (test L355; README L24/L42 and the worked examples quote it). M3 clauses L452-460 unchanged.

## Step 3. `src/hearsay/pipeline.py`: the M5 scorer, identity gate, entries, `analyze_clip`

1. `Models.__init__` L316-319: add `m5_dir: str | Path = M5_DIR, load_m5: bool = False`; L326 `self.with_m5 = load_m5` (never `self.load_m5`, which would shadow the imported function name); `self.m5_dir = Path(m5_dir)`; L336 `self.probe = self.backbone = self.spectra = self.m5 = None`; `self.m5_hashes: dict[str, str] | None = None`. Docstring L312-314 lists M5.
2. Split `load_deep` L340-357 into `load_m1()` (L341-352 as is, truncation included), `load_spectra_model()` (L354-357), and
   ```python
   def load_m5_model(self) -> None:
       from hearsay.m5_model import load_m5 as _load_m5     # lazy: m5_model imports torch + transformers at module top
       t = time.time()
       self.m5_hashes = json.loads((self.m5_dir / "hashes.json").read_text())
       self.m5 = _load_m5(self.m5_dir, self.device)          # sha-verified, eval mode, layerdrop 0, no spec-augment
       self.load_seconds["m5"] = round(time.time() - t, 2)
   ```
   `load_deep()` = `self.load_m1(); if self.with_spectra: self.load_spectra_model(); if self.with_m5: self.load_m5_model()`. Truncation never touches M5 (saved at 12 layers; `M5Net.layer_logits` has 13 entries, `m5_model.py:85`).
3. After `spectra_logit` (L372-378):
   ```python
   def m5_logit(self, x: np.ndarray) -> float:
       """scripts/m5_score.py's sequence on one clip: deploy_transform (trim -> band-limit -> cap 8 s ->
       normalize) -> collate (normalizes again, as the export did) -> score_batch; fp32, batch 1."""
       from hearsay.m5_data import collate, deploy_transform
       from hearsay.m5_model import score_batch
       if self.m5 is None:
           raise RuntimeError("M5 was not loaded (load_m5=False)")
       xs, mask = collate([deploy_transform(x)])
       return float(score_batch(self.m5, xs, mask, self.device)[0])
   ```
4. `version()` L387-391: add `"m5": self.m5_name()` (the parent directory's name when the dir is called `model`, else its own name; `""` when M5 is not loaded), `"m5_backbone_sha"` and `"m5_head_sha"` (12 hex chars from `self.m5_hashes`, `""` when not loaded).
5. New module functions after `m1_only` (L285):
   ```python
   def m5_identity_from_dir(m5_dir: str | Path) -> dict[str, str]:
       """hashes.json of a checkpoint dir; the cheap pre-load gate reads this, never the 657 MB file."""
       p = Path(m5_dir) / "hashes.json"
       if not p.exists():
           raise RuntimeError(f"M5 checkpoint {m5_dir} has no hashes.json")
       return json.loads(p.read_text())

   def check_m5_identity(hashes: Mapping[str, str] | None, consts: FusionConstants | None) -> None:
       """Refuse a rule that weights M5 when M5 is absent or is not the checkpoint the rank reference
       was built from (D6). Called before the load with hashes.json and after it with the loaded net's."""
       if consts is None or "m5_xlsr_ft" not in consts.ranked:
           return
       if not hashes:
           raise RuntimeError("the fusion rule weights m5_xlsr_ft but M5 is not loaded (load_m5=False)")
       ck = consts.m5_checkpoint or {}
       bad = [k for k in ("backbone_sha256", "head_sha256", "config_hash") if hashes.get(k) != ck.get(k)]
       if bad:
           raise RuntimeError(f"M5 checkpoint does not match the fusion constants on {', '.join(bad)} "
                              f"(constants expect {ck.get('dir')})")
   ```
   Callers pass `models.m5_hashes` (post-load) or `m5_identity_from_dir(m5_dir)` (pre-load).
6. `_deep_entry` L423-432: `elif name == "m5_xlsr_ft": why = f"XLS-R fine-tuned head (M5): logit {logit:+.2f} ({'synthetic' if logit > 0 else 'real'}-like, P={p:.2f})"`; Spectra stays the `else`.
7. `analyze_clip` L546-565: signature `scorers: Sequence[str] | None = None`; first line of the body `scorers = tuple(scorers) if scorers is not None else (consts.detectors if consts is not None else ("m1b_v3",))` (with the four-column default every existing caller that omits `scorers` would otherwise request M5); L564-565 →
   ```python
   deep = [(n, getattr(models, attr)) for n, attr in (("m1b_v3", "m1_logit"), ("m5_xlsr_ft", "m5_logit"),
                                                        ("spectra_aasist", "spectra_logit")) if n in scorers]
   ```
   (lazy: fakes without `m5_logit` keep working under v0/v1; order = `DEEP` order).
8. L595-597: `ver.setdefault("fusion", "none" if consts is None else ("/".join(Path(consts.source).parts[-2:]) if consts.source != "<dict>" else consts.source))` → `fusion_v1/constants.json`; `toy` stays `toy`.

## Step 4. `scripts/run_pipeline.py`

1. Docstring L17-21: both files (`fusion_v1` default; `fusion_v2` via `--fusion`, four scorers; `--m5`); `--fusion` help L96-99 and `--app-root` help L118-119 likewise. Imports L55-75: add `DEEP`, `M5_DIR`, `check_m5_identity`, `m5_identity_from_dir`.
2. After `--hc` (L123): `ap.add_argument("--m5", type=Path, default=_env_path("HEARSAY_M5"), help="M5 checkpoint dir ($HEARSAY_M5; default <app-root>/models/m5_shipped or the pinned 0741 checkpoint); loaded only when the fusion file weights m5_xlsr_ft")`.
3. After `default_probe` (L148-153):
   ```python
   def default_m5(app_root: Path) -> Path:
       shipped = app_root / "models" / "m5_shipped"          # docker/build.sh's staging name (Docker lane, at the switch)
       if (shipped / "hashes.json").exists():
           return shipped
       pinned = app_root / M5_DIR.relative_to(REPO)
       return pinned if (pinned / "hashes.json").exists() or app_root != REPO else M5_DIR

   def scorers_for(consts: FusionConstants | None) -> tuple[str, ...]:
       """The fused columns a run computes: the constants file's detectors (v1: three, v2: four), M1 alone otherwise."""
       return consts.detectors if consts is not None else ("m1b_v3",)
   ```
4. `resolve_scorers` L131-145: delete the note (L142-144); keep the return `DETECTOR_ORDER, fusion` (tests L522-531). `cache_identity` L215-222: add `m5: str | None = None` and key `"m5": m5` after `"hc"`.
5. `main` L372-426, reordered: L372 `resolve_scorers` unchanged in return value, plus `requested = {SCORER_FLAGS[f] for f in args.detectors.split(",") if f.strip()}` (the flags actually typed, not the maximal set the resolver returns); L384-385 probe/hc; **then** the constants load (today L395-402) with the `sys.exit` messages as they are; `scorers = scorers_for(consts)`; `missing = set(scorers) - requested; if consts is not None and missing: print(f"note: the fusion file needs {', '.join(sorted(missing))} beyond --detectors {args.detectors!r}; running them too")` (a default `--detectors` run under v2 prints it once, naming M5; under v1 it is silent); `m5_dir = args.m5 or default_m5(args.app_root)`; `m5_hashes = None`; if `"m5_xlsr_ft" in scorers`: `try: m5_hashes = m5_identity_from_dir(m5_dir); check_m5_identity(m5_hashes, consts) except RuntimeError as e: sys.exit(f"error: {e}")` (the pre-load gate, before the cache and the 11 s load); then `cache_identity(..., m5=(m5_hashes or {}).get("head_sha256", "")[:12] or None)` and `open_cache` (today L392-393). `kw` (L405) and the print (L406-408) use `scorers`.
6. `version` L410: `"fusion": "/".join(fusion.parts[-2:]) if fusion else "none"`.
7. `Models(...)` L424-426: add `m5_dir=m5_dir, load_m5="m5_xlsr_ft" in scorers`; after `version["models"] = models.version()` (L427): `try: check_m5_identity(models.m5_hashes, consts) except RuntimeError as e: sys.exit(f"error: {e}")`. Print L430-431 adds `m5 {m5_dir}` when loaded.
8. `preflight` L284-311 (D6 loudness): after the `preflight_consistent(pa, pb)` check add `if not preflight_consistent(a["fusion"]["p_fused"], b["fusion"]["p_fused"]): raise RuntimeError(f"preflight {name}: fused probability not reproducible: ...")` and `bad = [n for n in kw.get("scorers", ()) if not any(d["name"] == n and d["status"] == "ok" for d in a["detectors"])]` → `raise RuntimeError(f"preflight {name}: scorer(s) not ok: {', '.join(bad)}")` (for `m1b_v3`, `m5_xlsr_ft`, `spectra_aasist` the entry name is the scorer; `handcrafted_v5`'s entry is `handcrafted`: map through `FUSED_FROM_DETECTOR`).
9. Timings: L436 `new_timings = not timings_path.exists()` → read the first line when the file exists; if it differs from the current header, rename `timings.csv.stale-<stamp>` and start fresh. Header L442-443: `["filename", "seconds", "engineered_s", *(f"{n}_s" for n in DEEP), "probability_synthetic", "is_speech", "flag"]`; L473 `eng = sum(v for n, v in by.items() if n not in DEEP)`; L474 row `*(by.get(n, "") for n in DEEP)`.
10. Scorer-error tally (D6), computed over the **completed `docs` collection after the loop** (cached rows take the early `continue` at L448-457 and must count too), with two pure helpers so the logic is testable without models:
    ```python
    def scorer_error_counts(docs: Mapping[str, dict], scorers: Sequence[str]) -> dict[str, int]:
        """Errored entries per fused column over every row (cached or fresh); the handcrafted column's
        entry is named `handcrafted` and is mapped back through FUSED_FROM_DETECTOR."""
        entry_of = {col: det for col, (det, _key) in FUSED_FROM_DETECTOR.items()}
        counts = {n: 0 for n in scorers}
        for doc in docs.values():
            status = {d["name"]: d["status"] for d in doc.get("detectors", [])}
            for n in scorers:
                if status.get(entry_of.get(n, n)) == "error":
                    counts[n] += 1
        return counts

    def scorer_error_exit(counts: Mapping[str, int], scorers: Sequence[str], n_rows: int, strict: bool) -> int:
        """4 when a weighted scorer errored on any row (strict, i.e. a --compare-tsv run) or on more than
        1% of rows; else 0. spectra_aasist only suppresses and is excluded."""
        limit = 0 if strict else 0.01 * n_rows
        return 4 if any(counts.get(n, 0) > limit for n in scorers if n != "spectra_aasist") else 0
    ```
    In `main`, after the loop: `errs = scorer_error_counts(docs, scorers)`; `meta["n_scorer_errors"] = errs` beside `meta["flags"]` (L491-493); print it in the `done:` line; after the TSV and `run_meta.json` are written, `code = scorer_error_exit(errs, scorers, len(items), bool(args.compare_tsv))`; if `code`: print the counts to stderr and `return code` (the TSV exists; the exit code says not to trust it; the entrypoint passes it through). T26 pins the helpers on synthetic docs: a fresh failure, a cached failure (a doc with no `seconds` field, as cached rows look), a `handcrafted` error counted under `handcrafted_v5`, the 1% boundary, and strict mode.

## Step 5. `tests/test_pipeline.py`

Fixtures:
- L20-32 imports: add `CONSTANTS_V2_PATH`, `DEFAULT_CONSTANTS_PATH`, `check_m5_identity`.
- L36-38: `CONSTANTS_V2 = REPO/"models"/"fusion_v2"/"constants.json"`, `CANDIDATE_V2 = REPO/"outputs"/"fusion"/"fusion_v2_candidate"/"constants.json"`; split `EXPORTS` into `EXPORTS_V1` (three columns, so the existing v0/v1 data-gated tests at L701/L718 gain no new dependency) and `EXPORTS_V2` (four); `TSV_V2 = {False: submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv, True: ..._CANDIDATE_FLIPPED_only_if_NSA_scores_inverted.tsv}`.
- After `TOY_V1` (L83-95): `TOY_V2` (refs m1b `[-10,-5,0,5,10]`, hc `[-4,-2,0,2,4]`, m5 `[-3,-1,0,1,3]`; `weights {0.6, 0.2, 0.2}`; same `e_rule`; `platt` a 17.7, b −8.15, prior_shift log(0.3/0.7); `final "A3_w0.2_E"`; `m5_checkpoint {"dir": "models/m5_toy/model", "backbone_sha256": "b"*64, "head_sha256": "h"*64, "config_hash": "c"*64}`; a `how` line), `LOGITS_V2 = {m1b_v3: 1.0, handcrafted_v5: 3.0, m5_xlsr_ft: 0.5, spectra_aasist: -7.0}` (ranks 0.6 / 0.8 / 0.6, base 0.64 exactly, suppressed → 0.32; verified with numpy), `TOY_V2_4PT = {**TOY_V2, "rank_ref_inner_oof_sorted": {d: [-2.0, -1.0, 1.0, 2.0] for d in three}}` for T5. Fixture `consts_v2`.
- `FakeModels` L281-300: `m5=None`, `m5_fail=False`, `m5_hashes=<toy hashes>`; `m5_logit(x)` appends `x` to `self.m5_inputs`, raises on `m5_fail`, returns `self._m5`; `version()` gains `"m5": "m5_toy"`. `fake_models_v2()` beside `fake_models_v1`.
- The nine assertions the 4-tuple breaks: L118, L121, L166-167 (v0: compare with `TOY["detectors"]`), L190 (v1: the three names), L329 (`[n for n in DEEP if n in scorers]`), L436 (`set(consts.detectors)`), L691 (needs_data v0: three), L526/L528 unchanged.

Tests (hermetic unless marked): T1, T2 (nine broken dicts incl. a zero weight), T3 (equal `FusionOutput`s on the three logit sets **and** `fo.detail["base"] == (1 - 0.2) * 0.6 + 0.2 * 0.8` exactly), T4, T5 (the 4-point fixture: logits 0.0 → base 0.5 not suppressed; m5 1.5 → 0.55 suppressed; margin −3.0 vs −3.0000001 on `TOY_V2`), T6 (`"fusion: e_on_a (A3_w0.2_E): 0.6 x rank(m1b_v3) + 0.2 x rank(handcrafted_v5) + 0.2 x rank(m5_xlsr_ft) = 0.640"`), T7 (`tmp_path/"models"/"fusion_v1"/"constants.json"` → `"fusion_v1/constants.json"`; `source="toy"` → `"toy"`), T8 (fake: `fm.m5_inputs[0] is ctx.audio`; real `Models(load_deep=False)` with a tiny net and a spy on `score_batch`: array equals `collate([deploy_transform(x)])[0]`), T9, T10 (`build_model(M5Config(keep_layers=2, spec_augment=False), tiny=True).eval()` as `tests/test_m5_model.py:75-76`; 0.3 s, zeros, 10 s → finite), T11 (`hashes.json` in `tmp_path/"m5"`; monkeypatch `hearsay.m5_model.load_m5`; `Models(load_deep=False, m5_dir=...).load_m5_model()`: hashes read, layer count unchanged, `version()["m5"] == "m5"`, both sha prefixes, `load_seconds["m5"]`), T12 (`check_m5_identity(toy hashes, consts_v2)` passes; a changed `head_sha256` → `RuntimeError` matching "head_sha256"; `None` with `consts_v2` → "not loaded"; anything with `consts_v1` → no-op), T13 (`rp.scorers_for`: v1 → three, v2 → four, `None` → `("m1b_v3",)`; `rp.resolve_scorers` unchanged), T14 (`cache_identity` v1 vs v2 differ; `open_cache` with a v1 header and a v2 identity moves the file aside, pattern of L579-600; a header without `m5` matches an identity with `m5=None`), T15, T16, T17, T18, T19, T24, T25 (`rp.preflight(fake_models_v2(m5_fail=True), consts_v2, "e_on_a", tmp_path, scorers=consts_v2.detectors)` raises matching `m5_xlsr_ft`; the same fake under `consts_v1` with its three scorers passes), T20 (`needs_data`, skip unless `CONSTANTS_V2` and `CANDIDATE_V2` exist), T21 (`slow`, `needs_data`, the twin of L716-732 with `EXPORTS_V2`, max |Δ| ≤ 1e-9 for both polarities).

## Step 6. Runner-subset suite and lint (stop 1)

```
uv run pytest -q tests/test_pipeline.py tests/test_docker_image.py tests/test_docs_consistency.py
uv run pytest -q -m "slow or needs_data" tests/test_pipeline.py -k "v2 or m5"
uv run ruff check .
```
Every existing test passes, the new ones pass, ruff clean. `tests/test_docker_image.py` is the Docker lane's file: if B6 or B8 fails, stop and report rather than editing it. (`tests/test_api.py` runs in step 9, after step 8.)

## Step 7. Live parity (stop 2)

1. 50 files, v2 (A4, A11):
   ```
   uv run python scripts/run_pipeline.py --data data/nsa/HackGTHearsayTesting --out outputs/runner/v2_50 --template data/nsa/HearsayScoreKey4TeamX.tsv --fusion models/fusion_v2/constants.json --limit 50 --compare-tsv submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv
   ```
   under `/usr/bin/time -l` (resident memory). Read `run_meta.json`: `compare`, `preflight`, `n_scorer_errors`, `version.models.m5*`, `model_load_seconds`, `load_seconds.m5`, exit code 0. Scratch join (read-only) of `results.jsonl` with the exports on those files: M5 max |Δlogit| ≤ 0.02; E-rule flag and verdict identical row by row; batch-1 vs batch-8 delta by scoring the same 50 decoded clips through `scripts/m5_score.py`'s `score_paths` against the runner's logits (recorded, not gated).
2. 50 files, v1 default (A6): the same command without `--fusion`, `--out outputs/runner/v1_50_after`, `--compare-tsv submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv`: Spearman 1.0, max ≤ 3e-5; `run_meta.json` shows three scorers and no `m5` load time.
3. Full set, v2, in the background (A5): first **without** `--flip`, `--out outputs/runner/v2_full --compare-tsv <the our-direction candidate>` (the runner writes one polarity per invocation); record wall time, mean s/file, mean `m5_xlsr_ft_s`, the compare block, `n_scorer_errors`, exit code. Then rerun the same `--out` with `--flip --compare-tsv submissions/20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_FLIPPED_only_if_NSA_scores_inverted.tsv` (re-fuses the cache in ~0.1 s per file; the compare is against the flipped reference). Both compares: 1,671 shared rows, the A4 marks, and the scratch join's row-by-row verdict and E-rule check on all rows.
4. Log the parity artifact (CLAUDE.md: every experiment gets a log row; the K spec's "log a container TSV once parity passes" rule): copy the full-set our-direction TSV to `submissions/20260926-HHMM_M4_runner_fusion_v2_M5_live_PARITY_our_direction.tsv` (a new file, never an overwrite) and `append_log("M4", <path>, validation_score=0.0065, notes="runner-made live TSV, fusion_v2 (A3_w0.2_E), PARITY artifact not a submission; vs 20260926-0914 candidate: spearman S, max abs diff D, n>0.01 N; M5 live vs export max |dlogit| L; not shipped; default remains fusion_v1")`. If A5 fails, no copy and no row; the failure is the named blocker in the run spec.

## Step 8. `src/hearsay/api.py` and `tests/test_api.py`

1. Docstring L13-19: `HEARSAY_M5`; first-load time "about 5 s" → "about 16 s under fusion_v2 (the M5 hash check and load)". `settings()` L56-76: `"m5": Path(os.environ.get("HEARSAY_M5", M5_DIR))`; `"scorers"`: when not m1-only, `file_scorers(fusion_path)` = `FusionConstants.load(path).detectors` (the file has no `detectors` field; the loader derives it from the layout, v1 → three, v2 → four), memoized on `(str(path), st_mtime_ns)` (one parse of ~1 MB per path, ~10 ms), falling back to the env-derived list (`DETECTOR_ORDER`, four names) only when the file does not exist, which `/health` then labels `"scorers_from": "env (fusion file missing)"` vs `"constants"`.
2. `get_models()` L79-90: `scorers = consts.detectors if consts is not None else ("m1b_v3",)`; pre-load gate `check_m5_identity(m5_identity_from_dir(s["m5"]), consts)` when M5 is in `scorers`; `Models(..., m5_dir=s["m5"], load_m5="m5_xlsr_ft" in scorers)`; post-load `check_m5_identity(models.m5_hashes, consts)`; `RuntimeError` → `HTTPException(500, str(e))` with the failure cached in `_state["error"]` so later requests return it without reloading; `_state["scorers"] = scorers`.
3. `_analyze_path` L147-155: `scorers=_state["scorers"]`. `health()` L119: `_state["scorers"]` when loaded else `s["scorers"]`; L122 `"fusion": "/".join(s["fusion"].parts[-2:]) if s["fusion"] else "none"`.
4. `tests/test_api.py`: L31 `_state` gains `"scorers": None, "error": None`; L47 → conditional on `DEFAULT_CONSTANTS_PATH.exists()` (the real v1 file's three detectors and `scorers_from == "constants"` when present; `list(DETECTOR_ORDER)`, four names, and `"env (fusion file missing)"` when absent); T22/T23 use writer-shaped JSON (a v1 dict with `alpha_handcrafted`, a v2 dict with `weights`), never a hand-added `detectors` key; L48 `"fusion_v1/constants.json"`; T22 (`HEARSAY_FUSION` → `tmp_path/"fusion_v2"/"constants.json"` holding `TOY_V2`, `HEARSAY_M5` → a tmp dir with a matching `hashes.json`, `api_mod.Models` monkeypatched with a recorder returning `fake_models_v2()`: `load_m5=True`; with `tmp_path/"fusion_v1"/"constants.json"` holding `TOY_V1`: `load_m5=False`); T23 (`/health` before load: three and `"fusion_v1/constants.json"` under the v1 toy, four and `"fusion_v2/constants.json"` under the v2 toy).

## Step 9. Full suite and lint

```
uv run pytest -q && uv run ruff check .
```
A1: ≥ 452 passed plus the new tests, ruff clean.

## Step 10. Docs, commit, messages

1. Run spec: Results (steps 0, 1, 6, 7, 9 numbers), Deviations, the moved candidate dir.
2. `docs/reports/2026-09-26_fusion-sweep-predeclared.md`: one dated line under "Nathan's decision (09:25)" (v2 file written, Platt deltas, runner executes it with `--fusion`, default unchanged, candidate moved); L106 gets the new path in place; L107 and L112 get a dated "superseded, see below" note.
3. `docs/reports/2026-09-26_runner-docker.md`: a "v3" section (parity table, timing, ECAPA thread note, what did not change) and the synopsis lines L102, L110-114, L127, L240; the pre-existing test-count staleness at L65 and L140-146 is left with a one-line note (not this change's).
4. `docs/handoffs/2026-09-26_m4-fusion-handoff.md`: a dated note under "What to do next" step 3 (runner part done, sha, `--fusion`, `--write`, `RANKED` gone, candidate moved).
5. Commit, staged by path: `scripts/fuse_sweep_m5.py src/hearsay/pipeline.py src/hearsay/api.py scripts/run_pipeline.py tests/test_pipeline.py tests/test_api.py docs/specs/2026-09-26_m4-fusion-v2-m5-scorer.md docs/reports/2026-09-26_m4-fusion-v2-m5-scorer-plan.md docs/reports/2026-09-26_m4-fusion-v2-m5-scorer-plan-review.md docs/reports/2026-09-26_fusion-sweep-predeclared.md docs/reports/2026-09-26_runner-docker.md docs/handoffs/2026-09-26_m4-fusion-handoff.md`. Message: `feat(M4): runner executes fusion_v2 (M5 as a fourth ranked scorer, weights dict); default stays fusion_v1`.
6. Messages. **Docker chat** (directly; sent at step 0 and again with the sha): backward compatible, default `fusion_v1`, nothing to stage until the switch; at the switch stage `M5_DIR` as `models/m5_shipped` (+0.63 GB, ~4.2 GiB resident), move the B6 pin, BUILD_INFO will list `fusion_v2` (inert without the checkpoint); their spec L96/L106 and `build.sh` L2-6 are stale in passing. **Oversight chat**: STATUS L41-44 (M5 "column only", "suite 444", the rebuild sentence), architecture L24/L42/L218 and `docs/img/architecture-flow.mmd:9`, the oversight handoff L16/L29, `docs/handoffs/2026-09-26_draft-review-handoff.md:13` and STATUS L42 (old candidate path), code map (no `pipeline.py` entry; `load_m5:199` → 202), CLAUDE.md disclosure at the switch. **README chat**: L89, L101, L141 (the placeholder), L275; at the switch L19-20, L131, L283, L306, L317. **Frontend (Hrushi, via oversight)**: `frontend-wiring-handoff.md:71-75` keys rule text on the rule name; under v2 the name is unchanged and the weights are 0.6/0.2/0.2, so read `fusion.weights` or `fusion.detail.final`; `frontend-contract.md:31-32,71` gain `m5_xlsr_ft` and `version.models.m5*`; nothing removed. Never treat a reply as Nathan's approval.

## Step 11. Post-commit gates

Claude critique agent against `.claude/codex-audit-prompt.md` (at most two rounds), then `bash .claude/review-audit.sh docs/specs/2026-09-26_m4-fusion-v2-m5-scorer.md` (one round plus one fix commit and re-audit); record both in the run spec; anything still open at 15:45 is recorded as open.

## Verification map

| Criterion | Proven at |
|---|---|
| A1 | steps 6, 9 |
| A2 | step 1.4 + T20 |
| A3 | T21 |
| A4 | step 7.1 |
| A5 | step 7.3 |
| A6 | step 7.2 |
| A7 | T12, T14 |
| A8 | T19; `git status` shows nothing under `submissions/`; `shasum -a 256 submissions/20260926-0813_M4_sweep_E_on_A_alpha0.2_our_direction.tsv` starts `096f3f0c` |
| A9 | step 1.3 |
| A10 | T18 |
| A11 | T25; step 7.1 and 7.3 `n_scorer_errors` |

## Risks and mitigations

- **v1 bit-identity after the loop rewrite:** proven by the critique agent's exhaustive check; T3's exact `==` and the existing L716-732 parity test (4.4e-16) pin it.
- **Fakes without `m5_logit`:** lazy binding; `analyze_clip`'s default scorers come from the constants, so v0/v1 callers never request M5.
- **Silent M5 failure:** D6 loudness (preflight `ok` requirement, `n_scorer_errors`, non-zero exit); T25 is the counterfactual.
- **The stale candidate shipping in an image:** moved out of `models/` in step 1.5; the Docker chat is told twice.
- **Platt last-digit drift:** measured in step 1.4 against the 09:14 original; A2/A3 tolerances absorb it; the candidate TSVs are not rewritten.
- **M5 slower than 0.2 s/clip (ECAPA thread halving):** measured in step 7, reported, not fixed (D-track file).
- **Memory:** measured with `/usr/bin/time -l` in step 7.1.
- **Time:** the schedule table and two stops; the API is severable and lands last.

## Codex plan review (10:17, `docs/reports/2026-09-26_m4-fusion-v2-m5-scorer-plan-review.md`) and how each finding was addressed

1. Error tally missed cached rows and misnamed the handcrafted column → step 4.10 now counts over the completed `docs` collection with `FUSED_FROM_DETECTOR` normalization, as two pure helpers pinned by T26.
2. `--flip` emits one polarity, so the full-set check compared the wrong file → step 7.3 runs our direction first, then `--flip` against the FLIPPED reference on the same cache.
3. No submission-log row → step 7.4 logs the full-set parity TSV as a new labelled file with its row; A8 amended in the spec; failure is the named blocker.
4. The constants have no `detectors` field → the API's `file_scorers` uses `FusionConstants.load(...).detectors`; `/health` says where the list came from; tests use writer-shaped JSON.
5. Outer holdout used in the pre-declared selection → recorded in the spec (standing failure mode 2): this change reproduces the frozen candidate, the holdout condition was a veto not the objective, and NSA's draft-review number on the test set is the pending independent evaluation; no re-selection here.
6. Unbounded schedule → a single 15:45 deadline, A5 by 15:00, the critique loop and audit bounded.
7. NaN/malformed constants → finite, 1-D, non-empty checks in step 2.5; T2 gains NaN weight and NaN reference cases.
8. `requested` was the maximal set → the typed flags are parsed separately (step 4.5), so the note fires exactly when `--detectors` omitted a column the file needs.
