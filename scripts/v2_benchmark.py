"""V2-01-lite benchmark CLI (Interface v1, docs/specs/2026-10-11_v2-benchmark-fusion.md).

  uv run python scripts/v2_benchmark.py adapt-l20
  uv run python scripts/v2_benchmark.py registry --build | --check [--no-trees]
  uv run python scripts/v2_benchmark.py score --file X.csv --name NAME [--itw ITW.csv] [--holdout] [--force]
  uv run python scripts/v2_benchmark.py rule --rule RULE.json [--holdout] [--force]
  uv run python scripts/v2_benchmark.py v1-scorecard [--force]

Outputs go to outputs/v2/benchmark/ (shared by every checkout through the outputs/ link).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(min(int(os.environ.get(_v, "4") or 4), 4))

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay import benchmark as B

REPO = B.REPO
OUT = REPO / "outputs/v2/benchmark"
REGISTRY = REPO / "benchmarks/v1.yaml"
L20_CSV = OUT / "scores/xlsr_L20.csv"

COLUMNS = {
    "m1b_v3": {"file": "outputs/detector_scores/m1b_v3.csv", "itw": None},
    "handcrafted_v5": {"file": "outputs/detector_scores/handcrafted_v5.csv",
                       "itw": "outputs/detector_scores/_itw_handcrafted_v5.csv"},
    "m5_xlsr_ft": {"file": "outputs/detector_scores/m5_xlsr_ft.csv", "itw": "outputs/detector_scores/_itw_m5.csv"},
    "spectra_aasist": {"file": "outputs/detector_scores/spectra_aasist.csv",
                       "itw": "outputs/spectra/itw_stress/scores.csv"},
    "wavlm_l": {"file": "outputs/detector_scores/wavlm_l.csv", "itw": None},
    "xlsr_L20": {"file": "outputs/v2/benchmark/scores/xlsr_L20.csv", "itw": None},
}
DATA_INPUTS = [
    "splits/nsa_folds.csv", "splits/nsa_test_durations.csv", "outputs/manifests/itw_stress.csv",
    "outputs/manifests/nsa_test.csv", "outputs/channel/m3_perturb.csv", "outputs/channel/m3_mlaad.csv",
    "models/fusion_v2/constants.json", "outputs/v2/bakeoff/probes/L20.npz", "outputs/v2/bakeoff/probes/L20.joblib",
    "outputs/v2/bakeoff/rows.json", "outputs/v2/bakeoff/emb_mlaad.npz",
    "outputs/embeddings/wav2vec2-xls-r-300m/itw_stress_v3/manifest.csv",
] + [f"outputs/v2/bakeoff/emb_itw_{k}.npz" for k in B.COHORT_KINDS]
TREE_INPUTS = [
    "models/m1_wav2vec2-xls-r-300m_L7_20260926-0521", "models/hc_selected", "models/m5_xlsr_ft_20260926-0741/model",
    "models/m1_wavlm-large_L9_20260926-1906", "weights/Spectra-AASIST", "weights/wav2vec2-xls-r-300m",
    "weights/wavlm-large",
]
POPULATIONS = {
    "inner": {"role": "selectable", "rows": B.N_INNER, "source": "splits/nsa_folds.csv fold 0-4 (nested OOF)"},
    "holdout": {"role": "read_once", "rows": B.N_HOLDOUT, "source": "splits/nsa_folds.csv fold holdout"},
    "itw": {"role": "report_only", "rows": B.N_ITW, "source": "outputs/manifests/itw_stress.csv"},
    "holdout_cohort": {"role": "report_only", "rows": 2500, "source": "outputs/channel/m3_perturb.csv (+ option-B cohort files)"},
    "itw_cohort": {"role": "report_only", "rows": 2500, "source": "outputs/v2/benchmark/cohorts/itw/<column>.csv"},
    "mlaad": {"role": "report_only", "rows": 572, "source": "outputs/channel/m3_mlaad.csv (spoof only; in-sample for M5)"},
    "test": {"role": "report_only", "rows": B.N_TEST, "source": "outputs/manifests/nsa_test.csv (unlabelled)"},
}
DEFERRED = {
    "mlaad_by_family": "needs new MLAAD pulls (models released after 2026-04-18)",
    "asvspoof5_eval": "not downloaded (V2-30 picks a modern public set)",
    "reverb_codec_telephony_shortcrop_cohorts": "not built yet",
    "sealed_set": "decision pending with Nathan (V2-01)",
    "actDCF": "moved to V2-61",
}


def _code_sha() -> str:
    """HEAD plus a hash of uncommitted tracked changes (a dirty tree is visible)."""
    return B.code_sha(REPO)


NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def safe_name(name: str) -> str:
    if not NAME_RE.fullmatch(name) or ".." in name:
        raise SystemExit(f"invalid --name {name!r}: letters, digits, '_', '.', '-' only")
    return name


def require_registry(*, trees: bool = False) -> dict:
    """Every command that reads pinned inputs refuses unless the registry checks clean."""
    if not REGISTRY.exists():
        raise SystemExit(f"no registry at {REGISTRY}; run `registry --build` first")
    reg = B.load_registry(REGISTRY)
    problems = check(reg, trees=trees)
    if problems:
        raise SystemExit("registry check failed:\n" + "\n".join(problems))
    return reg


def main_sha(sf: B.ScoreFile) -> str:
    """sha256 of the score file itself (not its ITW override)."""
    return next(iter(sf.shas.values()))


def write_json(path: Path, obj, *, force: bool) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite {path} (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(B.dumps(obj))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# adapt-l20
# ---------------------------------------------------------------------------


def adapt_l20(npz_path=REPO / "outputs/v2/bakeoff/probes/L20.npz", rows_path=REPO / "outputs/v2/bakeoff/rows.json",
              out=L20_CSV, pops=None) -> pd.DataFrame:
    """Turn the bake-off L20 probe scores into an Interface v1 score file.
    Inner rows are filtered by path to the 16,142 NSA basis (folds checked);
    test rows are keyed by full path, never by basename."""
    pops = pops or B.load_populations()
    z = np.load(npz_path)
    rows = json.loads(Path(rows_path).read_text())
    if len(rows["inner_path"]) != len(z["oof"]):
        raise B.BenchmarkError("rows.json inner_path and L20 oof differ in length")
    inner_keys = np.array([B.canonical_key(p) for p in rows["inner_path"]])
    pos = pd.Series(np.arange(len(inner_keys)), index=inner_keys)
    if pos.index.duplicated().any():
        raise B.BenchmarkError("duplicate inner paths in rows.json")
    ip = pops["inner"]
    missing = set(ip.keys) - set(pos.index)
    if missing:
        raise B.BenchmarkError(f"L20: {len(missing)} NSA inner rows missing from rows.json")
    sel = pos.reindex(ip.keys).to_numpy()
    folds_rows = np.array([str(int(f)) for f in rows["inner_fold"]])[sel]
    if not np.array_equal(folds_rows, ip.fold):
        raise B.BenchmarkError("L20 inner folds disagree with splits/nsa_folds.csv")
    parts = [pd.DataFrame({"path": ip.keys, "fold": ip.fold, "split": "inner_oof", "logit": z["oof"][sel]})]
    hk = np.array([B.canonical_key(p) for p in rows["holdout_path"]])
    parts.append(pd.DataFrame({"path": hk, "fold": "holdout", "split": "holdout", "logit": z["holdout"]}))
    tk = np.array([f"data/nsa/HackGTHearsayTesting/{n}" for n in rows["test_name"]])
    parts.append(pd.DataFrame({"path": tk, "fold": "", "split": "test", "logit": z["test"]}))
    wk = np.array([B.canonical_key(p) for p in rows["itw_path"]])
    parts.append(pd.DataFrame({"path": wk, "fold": "", "split": "itw", "logit": z["itw"]}))
    df = pd.concat(parts, ignore_index=True)
    df["logit"] = df["logit"].astype(float)
    df["score"] = 1.0 / (1.0 + np.exp(-df["logit"]))
    df = df[["path", "fold", "split", "score", "logit"]]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    meta = {"source_npz": str(npz_path), "source_rows": str(rows_path),
            "note": "logits carry a Platt map fitted on all 23,790 bake-off OOF rows (transductive scale); "
                    "use ranks only, never the z/logit domain",
            "basis": "inner filtered by path to the 16,142 NSA rows; OOF by nsa_folds_plus_asv19 folds, "
                     "identical to nsa_folds on NSA rows",
            "rows": {s: int((df["split"] == s).sum()) for s in B.SPLITS}}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return df


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def build_registry(version: str = B.BENCHMARK_VERSION, pending=None) -> dict:
    inputs = {}
    for c in COLUMNS.values():
        for rel in (c["file"], c["itw"]):
            if rel:
                inputs[rel] = {"kind": "file", "sha256": B.sha256_file(REPO / rel)}
    for rel in DATA_INPUTS:
        inputs[rel] = {"kind": "file", "sha256": B.sha256_file(REPO / rel)}
    for rel in TREE_INPUTS:
        inputs[rel] = {"kind": "tree", "sha256": B.sha256_tree(REPO / rel)}
    for f in sorted(COHORT_DIR.glob("*/*.csv")) if COHORT_DIR.exists() else []:
        inputs[f.relative_to(REPO).as_posix()] = {"kind": "file", "sha256": B.sha256_file(f)}
    tsv = B.shipped_tsv_path()
    return {
        "format": "JSON content in a .yaml file (valid YAML 1.2); parse with json",
        "version": version,
        "spec": "docs/specs/2026-10-11_v2-benchmark-fusion.md",
        "populations": POPULATIONS,
        "columns": COLUMNS,
        "inputs": inputs,
        "shipped_tsv": {"path": "submissions/CrossExam_predictions.tsv (main checkout)", "sha256": B.sha256_file(tsv)},
        "pending": pending if pending is not None else {
            f"{cohort}_cohort.{col}": "cohort scoring job (scripts/v2_cohort_scores.py), ask-first"
            for cohort, cols in EXPECTED_COHORT_FILES.items() for col in cols
            if not (COHORT_DIR / cohort / f"{col}.csv").exists()
        },
        "deferred": DEFERRED,
    }


def check(reg: dict, *, trees: bool = True) -> list[str]:
    problems = B.check_registry(reg, REPO, trees=trees)
    if B.sha256_file(B.shipped_tsv_path()) != reg["shipped_tsv"]["sha256"]:
        problems.append("shipped TSV sha256 mismatch")
    pops = B.load_populations()
    for k in ("inner", "holdout", "itw", "test"):
        if pops[k].n != reg["populations"][k]["rows"]:
            problems.append(f"population {k}: {pops[k].n} rows, registry says {reg['populations'][k]['rows']}")
    return problems


# ---------------------------------------------------------------------------
# Loading and scorecards
# ---------------------------------------------------------------------------


def load_bench(columns=None, pops=None) -> B.Bench:
    pops = pops or B.load_populations()
    bench = B.Bench(pops)
    for name in columns or COLUMNS:
        c = COLUMNS[name]
        sf = B.read_score_file(REPO / c["file"], name=name, itw=(REPO / c["itw"]) if c["itw"] else None)
        bench.add(sf)
    return bench


def holdout_sha(pops) -> str:
    h = pops["holdout"]
    return B.sha256_json({"keys": list(h.keys), "y": h.y.tolist(), "fold": list(h.fold)})


def bootstrap_block(pop, s, boot_n: int) -> dict:
    """K7 intervals for one population: inner gets the primary (LJ fixed) and the
    sensitivity (LJ resampled) rows; holdout and ITW get one stratified row
    (holdout is marked degenerate: 2 spoof generators)."""
    if boot_n <= 0 or pop.cluster is None:
        return {}
    if pop.name == "inner":
        return {"lj_fixed": B.bootstrap_ci(pop.y, s, pop.cluster, pop.stratum, n=boot_n, lj_mode="fixed"),
                "lj_resampled": B.bootstrap_ci(pop.y, s, pop.cluster, pop.stratum, n=boot_n, lj_mode="cluster")}
    return {"stratified": B.bootstrap_ci(pop.y, s, pop.cluster, pop.stratum, n=boot_n)}


def readout_scores(pops, scores_by_pop: dict, *, inner_counts=True, boot_n: int = 0) -> dict:
    """Readings per population given {pop: scores}. Inner gets the fold table and
    thresholded counts (each naming its reading and threshold); boot_n > 0 adds
    the K7 bootstrap intervals."""
    out = {}
    for p, s in scores_by_pop.items():
        if s is None:
            out[p] = {"scored": False}
            continue
        pop = pops[p]
        if pop.y is None:
            continue
        if p == "inner":
            out[p] = B.fold_table(pop.y, s, pop.fold)
            if inner_counts:
                out[p]["counts"] = {k: B.counts_at(pop.y, s, k) for k in B.READINGS}
        else:
            out[p] = {"pooled": B.readings(pop.y, s),
                      "counts": {k: B.counts_at(pop.y, s, k) for k in B.READINGS}}
        if boot_n:
            out[p]["bootstrap"] = bootstrap_block(pop, s, boot_n)
    return out


def cohort_holdout_frame(columns, pops):
    """The legacy holdout perturbation cohort (four shipped columns, wide): 500
    holdout clips x 5 kinds, 250 real + 250 spoof per kind. Its labels are
    holdout labels, so every readout of it goes through the ledger."""
    path = REPO / "outputs/channel/m3_perturb.csv"
    frames = {c: B.read_cohort_file(path, column=c) for c in columns}
    keys = frames[columns[0]]
    yv = pd.Series(pops["holdout"].y, index=pops["holdout"].keys)
    if not set(keys["key"]) <= set(yv.index):
        raise B.BenchmarkError("holdout cohort paths are not all holdout rows")
    for kind in B.COHORT_KINDS:
        y = yv.reindex(keys[keys["kind"] == kind]["key"]).to_numpy()
        if int((y == 0).sum()) != 250 or int((y == 1).sum()) != 250:
            raise B.BenchmarkError(f"holdout cohort kind {kind}: expected 250 real + 250 spoof")
    return frames, yv


LEGACY_HOLDOUT_COHORT = ("m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "spectra_aasist")
COHORT_DIR = OUT / "cohorts"
EXPECTED_COHORT_FILES = {"itw": ("m1b_v3", "handcrafted_v5", "m5_xlsr_ft", "spectra_aasist", "wavlm_l", "xlsr_L20"),
                         "holdout": ("wavlm_l", "xlsr_L20")}


def load_cohort_frames(cohort: str, pops, *, pinned: dict, cohort_dir: Path = COHORT_DIR, legacy: bool = True,
                       per_class: int = 250, root: Path = REPO) -> tuple[dict, pd.Series]:
    """All cohort scores for one cohort: the legacy wide holdout file (four shipped columns) plus every
    per-column file written by scripts/v2_cohort_scores.py under cohorts/<cohort>/<column>.csv. Every
    per-column file must be pinned in the registry (`pinned`: registry inputs) with a matching sha256,
    so changed cohort logits can never slip in unrecorded. Every column must cover the same (path,
    kind) set, inside the population, `per_class` real + spoof per kind."""
    pop = pops["holdout" if cohort == "holdout" else "itw"]
    labels = pd.Series(pop.y, index=pop.keys)
    frames = {}
    if cohort == "holdout" and legacy:
        frames, _ = cohort_holdout_frame(list(LEGACY_HOLDOUT_COHORT), pops)
    d = Path(cohort_dir) / cohort
    for f in sorted(d.glob("*.csv")) if d.exists() else []:
        if f.stem in frames:
            raise B.BenchmarkError(f"cohort {cohort}: {f.stem} is scored twice")
        rel = Path(os.path.relpath(f, root)).as_posix()  # lexical: outputs/ is a symlink, never resolve
        pin = pinned.get(rel)
        if pin is None:
            raise B.BenchmarkError(f"cohort file {rel} is not pinned in the registry: rebuild the registry first")
        if pin["sha256"] != B.sha256_file(f):
            raise B.BenchmarkError(f"cohort file {rel} changed since it was pinned in the registry")
        fr = B.read_cohort_file(f)
        if not set(fr["key"]) <= set(labels.index):
            raise B.BenchmarkError(f"cohort {cohort}: {f.name} has paths outside the {pop.name} population")
        for kind in B.COHORT_KINDS:
            y = labels.reindex(fr.loc[fr["kind"] == kind, "key"]).to_numpy()
            if int((y == 0).sum()) != per_class or int((y == 1).sum()) != per_class:
                raise B.BenchmarkError(f"cohort {cohort}: {f.name} kind {kind} is not {per_class} + {per_class}")
        frames[f.stem] = fr
    keysets = {c: set(zip(fr["key"], fr["kind"])) for c, fr in frames.items()}
    if len({frozenset(v) for v in keysets.values()}) > 1:
        raise B.BenchmarkError(f"cohort {cohort}: columns cover different (path, kind) sets")
    return frames, labels


def cohort_file_shas(cohort: str, frames: dict, pinned: dict) -> dict:
    """sha256 of every per-column cohort file in use (from the registry pins; legacy columns excluded)."""
    return {c: pinned[f"{COHORT_DIR.relative_to(REPO).as_posix()}/{cohort}/{c}.csv"]["sha256"]
            for c in sorted(frames) if c not in LEGACY_HOLDOUT_COHORT or cohort != "holdout"}


def cohort_key_extra(frames: dict, file_shas: dict | None = None) -> dict:
    """Ledger-definition extra for a holdout-cohort read: empty for the legacy four columns (so the
    already-completed reads keep their keys); otherwise the column set AND each new file's sha256, so
    a new column or a changed cohort file is a new, counted read."""
    if set(frames) == set(LEGACY_HOLDOUT_COHORT):
        return {}
    return {"cohort_columns": sorted(frames), "cohort_files": dict(sorted((file_shas or {}).items()))}


def cohort_sha(frames: dict, labels: pd.Series) -> str:
    keys = sorted(set(next(iter(frames.values()))["key"]))
    return B.sha256_json({"cohort_keys": keys, "y": labels.reindex(keys).astype(int).tolist()})


def cohort_readout(bench: B.Bench, rules: dict, frames: dict, labels: pd.Series) -> dict:
    """Per kind: readings for each rule (ranks vs all inner rows; E with the
    cohort's own Spectra logit), plus deltas vs the cohort's own 'none' kind."""
    out = {}
    for rname, rule in rules.items():
        need = [c for c, w in rule.weights.items() if w > 0] + ([B.E_COLUMN] if rule.e else [])
        miss = [c for c in need if c not in frames]
        if miss:
            out[rname] = {"scored": False, "reason": f"no cohort scores for {miss}"}
            continue
        per = {}
        for kind in B.COHORT_KINDS:
            d = {c: frames[c][frames[c]["kind"] == kind].set_index("key")["logit"] for c in need}
            keys = d[need[0]].index
            for c in need:
                if set(d[c].index) != set(keys):
                    raise B.BenchmarkError(f"cohort {kind}: column {c} covers different rows")
            r = {c: B.full_ranks(bench.cols[c]["inner"], d[c].reindex(keys).to_numpy())
                 for c in need if c != B.E_COLUMN}
            base = B.blend(r, rule.weights)
            fused = B.apply_e(base, d[B.E_COLUMN].reindex(keys).to_numpy())[0] if rule.e else base
            per[kind] = B.readings(labels.reindex(keys).to_numpy(), fused)
        for kind in B.COHORT_KINDS:
            per[kind]["delta_vs_none"] = {k: (None if per[kind][k] is None or per["none"][k] is None
                                              else per[kind][k] - per["none"][k]) for k in B.READINGS}
        out[rname] = per
    return out


def mlaad_readout(bench: B.Bench, rules: dict, thresholds: dict) -> dict:
    """MLAAD (572 fakes, single class): detection rate at the inner primary
    threshold. In-sample for any rule containing M5: readout only."""
    path = REPO / "outputs/channel/m3_mlaad.csv"
    df = pd.read_csv(path)
    keys = [B.canonical_key(p) for p in df["path"]]
    if len(df) != 572 or len(set(keys)) != len(keys):
        raise B.BenchmarkError(f"MLAAD cohort: expected 572 unique paths, got {len(df)} rows / {len(set(keys))} unique")
    if set(df["label"].astype(str)) != {"spoof"}:
        raise B.BenchmarkError("MLAAD cohort must be spoof only")
    out = {}
    for rname, rule in rules.items():
        need = [c for c, w in rule.weights.items() if w > 0]
        cols = need + ([B.E_COLUMN] if rule.e else [])
        if any(c not in df.columns for c in cols):
            out[rname] = {"scored": False, "reason": "column(s) not scored on MLAAD"}
            continue
        r = {c: B.full_ranks(bench.cols[c]["inner"], df[c].to_numpy(float)) for c in need}
        base = B.blend(r, rule.weights)
        fused = B.apply_e(base, df[B.E_COLUMN].to_numpy(float))[0] if rule.e else base
        thr = thresholds[rname]
        t = np.inf if thr == "inf" else float(thr)
        out[rname] = {"detected_share": float((fused >= t).mean()), "threshold": thr,
                      "reading": B.PRIMARY, "in_sample_for_m5": "m5_xlsr_ft" in need}
    return out


def column_scorecard(bench: B.Bench, name: str, *, with_holdout: bool, boot_n: int = 2000) -> dict:
    pops = bench.pops
    cols = bench.cols[name]
    scores = {p: cols[p] for p in ("inner", "itw")}
    if with_holdout:
        scores["holdout"] = cols["holdout"]
    card = {"column": name, "inputs": bench.files[name].shas, "itw_source": bench.files[name].itw_source,
            "direction_auc_inner": B.direction_guard(pops["inner"].y, cols["inner"], name),
            "populations": readout_scores(pops, scores, boot_n=boot_n)}
    if cols["test"] is not None:
        thr = card["populations"]["inner"]["counts"][B.PRIMARY]["threshold"]
        t = np.inf if thr == "inf" else float(thr)
        card["populations"]["test"] = {"share_above_inner_threshold": float((cols["test"] >= t).mean()),
                                       "threshold": thr, "reading": B.PRIMARY}
    return card


def rule_scorecard(bench: B.Bench, rule: B.Rule, *, refs="nested", with_holdout: bool,
                   control: B.Rule | None = None, boot_n: int = 2000) -> dict:
    pops = bench.pops
    res = {p: bench.score_rule(rule, p, refs) for p in ("inner", "holdout", "itw", "test")}
    noe = B.Rule(rule.name + " (no E)", rule.weights, e=False)
    res_noe = {p: bench.score_rule(noe, p, refs) for p in ("inner", "itw")}

    def fused(r):
        return r["fused"] if r["scored"] else None

    scores = {"inner": fused(res["inner"]), "itw": fused(res["itw"])}
    if with_holdout:
        scores["holdout"] = fused(res["holdout"])
    card = {"rule": rule.name, "definition": rule.definition(), "inner_refs": refs,
            "populations": readout_scores(pops, scores, boot_n=boot_n),
            "without_e": readout_scores(pops, {p: fused(r) for p, r in res_noe.items()}, inner_counts=False)}
    if res["inner"]["scored"]:
        inner = pops["inner"]
        m = res["inner"]["e_mask"]
        card["e_fires_inner"] = {"real": int((m & (inner.y == 0)).sum()), "spoof": int((m & (inner.y == 1)).sum()),
                                 "per_fold_real": {f: int((m & (inner.y == 0) & (inner.fold == f)).sum())
                                                   for f in sorted(set(inner.fold))}}
        real = inner.y == 0
        rk = {c: bench.ranks(c, "inner", refs)[real] for c, w in rule.weights.items() if w > 0}
        card["dominance"] = B.dominance(rk, res["inner"]["fused"][real], rule.weights)
        if res["test"]["scored"]:
            thr = card["populations"]["inner"]["counts"][B.PRIMARY]["threshold"]
            ctrl = None
            if control is not None:
                cr_in = bench.score_rule(control, "inner", refs)
                ctrl_thr = B.counts_at(inner.y, cr_in["fused"], B.PRIMARY)["threshold"]
                ctrl = (bench.score_rule(control, "test", refs)["fused"], ctrl_thr)
            card["populations"]["test"] = B.agreement_with_shipped(
                pops["test"].keys, res["test"]["fused"], thr,
                control_fused=ctrl[0] if ctrl else None, control_threshold=ctrl[1] if ctrl else None)
    return card


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_score(a) -> None:
    reg = require_registry(trees=False)
    name = safe_name(a.name)
    pops = B.load_populations()
    bench = B.Bench(pops)
    sf = B.read_score_file(a.file, name=name, itw=a.itw)
    bench.add(sf)
    card = column_scorecard(bench, name, with_holdout=False, boot_n=a.boot_n)
    if a.holdout:
        res = B.read_holdout(
            [B.column_def(name, main_sha(sf))],
            lambda: readout_scores(pops, {"holdout": bench.cols[name]["holdout"]}, boot_n=a.boot_n)["holdout"],
            ledger=B.HoldoutLedger.default(), holdout_sha=holdout_sha(pops), pinned=reg["inputs"],
            meta={"code_sha": _code_sha(), "input_shas": sf.shas, "lane": a.lane})
        card["populations"]["holdout"] = res["readout"]
        card["holdout_ledger"] = res["ledger"]
    card["benchmark_version"] = reg["version"]
    card["registry_sha256"] = B.sha256_file(REGISTRY)
    card["lane"] = a.lane
    write_json(OUT / f"{name}.json", card, force=a.force)
    print(score_markdown_row(name, card))


def score_markdown_row(name: str, card: dict) -> str:
    """The Interface v1 markdown scorecard row (pooled readings per population)."""
    lines = ["| What | Population | brief | package-0.5 | package-0.3 | EER |", "|---|---|---|---|---|---|"]
    for p in ("inner", "holdout", "itw"):
        r = card["populations"].get(p, {}).get("pooled")
        if r:
            lines.append(f"| {name} | {p} | {_fmt(r['brief'])} | {_fmt(r['package05'])} | "
                         f"{_fmt(r['package03'])} | {_fmt(r['eer'])} |")
    return "\n".join(lines)


def cmd_rule(a) -> None:
    reg = require_registry(trees=False)
    spec = json.loads(Path(a.rule).read_text())
    rule = B.Rule(safe_name(spec["name"]), spec["weights"], bool(spec.get("e", True)))
    need = [c for c, w in rule.weights.items() if w > 0] + ([B.E_COLUMN] if rule.e else [])
    bench = load_bench(need)
    card = rule_scorecard(bench, rule, refs=spec.get("refs", "nested"), with_holdout=False, boot_n=a.boot_n)
    if a.holdout:
        defn = {**rule.definition(), "refs": "full", "inputs": {n: main_sha(f) for n, f in bench.files.items()}}
        res = B.read_holdout([defn], lambda: readout_scores(
            bench.pops, {"holdout": bench.score_rule(rule, "holdout")["fused"]}, boot_n=a.boot_n)["holdout"],
            ledger=B.HoldoutLedger.default(), holdout_sha=holdout_sha(bench.pops), pinned=reg["inputs"],
            meta={"code_sha": _code_sha(), "input_shas": {n: f.shas for n, f in bench.files.items()}})
        card["populations"]["holdout"] = res["readout"]
        card["holdout_ledger"] = res["ledger"]
    write_json(OUT / f"{rule.name}.json", card, force=a.force)
    print(json.dumps(B.clean(card["populations"]["inner"]["pooled"]), sort_keys=True))


LEGACY_BASELINE_NOTE = ("the v1 baseline set was first read on 2026-10-12 under a name-only ledger key; the "
                        "sha-bound key is recorded as an alias of that read (not a new read) when the files are unchanged")


def baseline_holdout(bench, others, ledger, hsha, meta, pinned, boot_n):
    """The 'benchmark-v1 baseline' holdout read for the non-reference columns,
    keyed by name + file sha. A completed read under the earlier name-only key
    is aliased (same read, same files) instead of read again."""
    pops = bench.pops
    defs = [B.column_def(c, main_sha(bench.files[c])) for c in others]
    key = B.ledger_key(defs, hsha)
    legacy = B.ledger_key([{"kind": "column", "name": c} for c in others], hsha)
    note = None
    if ledger.persisted(key) is None and ledger.persisted(legacy) is not None:
        old = ledger.persisted(legacy)
        old_shas = old.get("meta", {}).get("input_shas", {})
        if all(old_shas.get(c) == bench.files[c].shas for c in others):
            ledger.alias(key, legacy, meta)
            note = LEGACY_BASELINE_NOTE
    res = B.read_holdout(defs, lambda: {c: readout_scores(pops, {"holdout": bench.cols[c]["holdout"]},
                                                         boot_n=boot_n)["holdout"] for c in others},
                         ledger=ledger, holdout_sha=hsha, meta=meta, pinned=pinned)
    entry = ledger.persisted(key) or {}
    if entry.get("alias_of") == legacy:
        note = LEGACY_BASELINE_NOTE
    return res, note


def v1_scorecard(boot_n: int = 2000) -> dict:
    """v1's full baseline scorecard: each column and the shipped rule, on every
    population we hold. Holdout: m1b_v3 and SHIP under the registry-pinned
    reference exemption; the other columns as one ledgered set ('benchmark-v1
    baseline'); the holdout perturbation cohort as its own ledgered read."""
    reg = require_registry(trees=True)
    pinned = reg["inputs"]
    bench = load_bench()
    pops = bench.pops
    ship = B.Rule("SHIP", dict(B.SHIP_WEIGHTS))
    out = {"benchmark_version": reg["version"], "registry_sha256": B.sha256_file(REGISTRY),
           "code_sha": _code_sha(), "columns": {}, "rules": {}}
    for name in COLUMNS:
        out["columns"][name] = column_scorecard(bench, name, with_holdout=False, boot_n=boot_n)
    ledger = B.HoldoutLedger.default()
    hsha = holdout_sha(pops)
    meta = {"code_sha": _code_sha(), "input_shas": {n: f.shas for n, f in bench.files.items()}}
    shas = {n: main_sha(f) for n, f in bench.files.items()}

    def hold(scores):
        return readout_scores(pops, {"holdout": scores}, boot_n=boot_n)["holdout"]

    ref = B.read_holdout([B.column_def("m1b_v3", shas["m1b_v3"])], lambda: hold(bench.cols["m1b_v3"]["holdout"]),
                         ledger=ledger, holdout_sha=hsha, meta=meta, pinned=pinned)
    out["columns"]["m1b_v3"]["populations"]["holdout"] = ref["readout"]
    out["columns"]["m1b_v3"]["holdout_ledger"] = ref["ledger"]
    others = [c for c in COLUMNS if c != "m1b_v3"]
    base, note = baseline_holdout(bench, others, ledger, hsha, meta, pinned, boot_n)
    for c in others:
        out["columns"][c]["populations"]["holdout"] = base["readout"][c]
        out["columns"][c]["holdout_ledger"] = base["ledger"] + " (benchmark-v1 baseline set)"
    if note:
        out["ledger_note"] = note
    for refs in ("transductive", "nested"):
        out["rules"][f"SHIP_{refs}"] = rule_scorecard(bench, ship, refs=refs, with_holdout=False, boot_n=boot_n)
    ship_inputs = {c: shas[c] for c in B.REFERENCE_COLUMN_FILES}
    sh = B.read_holdout([B.ship_def(ship_inputs)], lambda: hold(bench.score_rule(ship, "holdout")["fused"]),
                        ledger=ledger, holdout_sha=hsha, meta=meta, pinned=pinned)
    for refs in ("transductive", "nested"):
        out["rules"][f"SHIP_{refs}"]["populations"]["holdout"] = sh["readout"]
        out["rules"][f"SHIP_{refs}"]["holdout_ledger"] = sh["ledger"] + " (holdout uses all-inner refs)"
    # holdout perturbation cohort: holdout labels, so ledgered
    shipped4 = list(LEGACY_HOLDOUT_COHORT)
    frames, labels = load_cohort_frames("holdout", pops, pinned=pinned)
    single = {c: B.Rule(c, {c: 1.0}, e=False) for c in frames if c != B.E_COLUMN}
    cohort_rules = {"SHIP": ship, **single}
    extra = cohort_key_extra(frames, cohort_file_shas("holdout", frames, pinned))
    cdefs = [{"population": "holdout_cohort", "name": n, **r.definition(),
              "inputs": {c: shas[c] for c in (shipped4 if not extra else sorted(frames))}, **extra}
             for n, r in cohort_rules.items()]
    co = B.read_holdout(cdefs, lambda: cohort_readout(bench, cohort_rules, frames, labels), ledger=ledger,
                        holdout_sha=cohort_sha(frames, labels), meta=meta, pinned=pinned)
    out["holdout_cohort"] = co["readout"]
    out["holdout_cohort_ledger"] = co["ledger"]
    itw_frames, itw_labels = load_cohort_frames("itw", pops, pinned=pinned)
    out["itw_cohort_files"] = cohort_file_shas("itw", itw_frames, pinned)
    itw_rules = {"SHIP": ship, **{c: B.Rule(c, {c: 1.0}, e=False) for c in itw_frames if c != B.E_COLUMN}}
    out["itw_cohort"] = (cohort_readout(bench, itw_rules, itw_frames, itw_labels) if itw_frames else
                         {"scored": False, "reason": "pending: no ITW cohort files yet (Step 6, ask-first)"})
    thr = {"SHIP": out["rules"]["SHIP_nested"]["populations"]["inner"]["counts"][B.PRIMARY]["threshold"]}
    for c in single:
        r = bench.score_rule(single[c], "inner", "nested")["fused"]
        thr[c] = B.counts_at(pops["inner"].y, r, B.PRIMARY)["threshold"]
    out["mlaad"] = mlaad_readout(bench, cohort_rules, thr)
    out["ledger_count"] = ledger.count()
    out["pending"] = reg.get("pending", {})
    out["deferred"] = reg.get("deferred", {})
    return out


def _fmt(v):
    return "NA" if v is None else f"{v:.4f}"


def markdown_rows(card: dict) -> str:
    lines = ["| What | Population | brief | package-0.5 | package-0.3 | EER |", "|---|---|---|---|---|---|"]

    def row(label, pop, r):
        lines.append(f"| {label} | {pop} | {_fmt(r['brief'])} | {_fmt(r['package05'])} | "
                     f"{_fmt(r['package03'])} | {_fmt(r['eer'])} |")

    for name, c in card["columns"].items():
        for p in ("inner", "holdout", "itw"):
            if p in c["populations"] and c["populations"][p].get("pooled"):
                row(name, p, c["populations"][p]["pooled"])
    for name, c in card["rules"].items():
        for p in ("inner", "holdout", "itw"):
            if p in c["populations"] and c["populations"][p].get("pooled"):
                row(name, p, c["populations"][p]["pooled"])
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("adapt-l20")
    r = sub.add_parser("registry")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--check", action="store_true")
    r.add_argument("--no-trees", action="store_true", help="skip hashing model/weight directories")
    r.add_argument("--version", default=B.BENCHMARK_VERSION)
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--file", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--itw")
    s.add_argument("--lane", default="unknown")
    s.add_argument("--holdout", action="store_true")
    s.add_argument("--force", action="store_true")
    s.add_argument("--boot-n", type=int, default=2000)
    ru = sub.add_parser("rule")
    ru.add_argument("--rule", required=True)
    ru.add_argument("--holdout", action="store_true")
    ru.add_argument("--force", action="store_true")
    ru.add_argument("--boot-n", type=int, default=2000)
    v = sub.add_parser("v1-scorecard")
    v.add_argument("--force", action="store_true")
    v.add_argument("--boot-n", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "adapt-l20":
        df = adapt_l20()
        print(df["split"].value_counts().to_dict())
    elif a.cmd == "registry" and a.build:
        if REGISTRY.exists() and not a.force:
            raise SystemExit(f"refusing to overwrite {REGISTRY} (use --force)")
        REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY.write_text(json.dumps(build_registry(a.version), indent=1, sort_keys=True) + "\n")
        print(f"wrote {REGISTRY}")
    elif a.cmd == "registry":
        problems = check(B.load_registry(REGISTRY), trees=not a.no_trees)
        print("\n".join(problems) if problems else "registry clean")
        return 1 if problems else 0
    elif a.cmd == "score":
        cmd_score(a)
    elif a.cmd == "rule":
        cmd_rule(a)
    elif a.cmd == "v1-scorecard":
        card = v1_scorecard(a.boot_n)
        write_json(OUT / "v1_scorecard.json", card, force=a.force)
        print(markdown_rows(card))
    return 0


if __name__ == "__main__":
    sys.exit(main())
