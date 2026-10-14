"""V2-60: run the pre-declared fusion test (benchmarks/v2_60_candidates.json).

  uv run python scripts/v2_fusion.py run [--force] [--boot-n 2000]

Refuses unless the candidates file is committed and matches the frozen spec
region, and the registry checks clean. Selection and verdicts read nested inner
OOF only; holdout is read once for the candidate set (ledger); ITW, cohorts,
MLAAD and test are report-only. Writes outputs/v2/fusion/v2_60_readout.json and
one experiments/log.csv row per candidate.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(min(int(os.environ.get(_v, "4") or 4), 4))

import argparse
import csv
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from hearsay import benchmark as B
from hearsay import fusion_v2 as F

REPO = B.REPO
CANDIDATES = REPO / "benchmarks/v2_60_candidates.json"
SPEC = REPO / "docs/specs/2026-10-11_v2-benchmark-fusion.md"
OUT = REPO / "outputs/v2/fusion/v2_60_readout.json"
LOG = REPO / "experiments/log.csv"
LOG_COLUMNS = ["timestamp", "rung", "validation_score", "validation_score_clean_only", "csv_path", "notes",
               "benchmark_version"]

_spec = importlib.util.spec_from_file_location("v2_benchmark", REPO / "scripts/v2_benchmark.py")
VB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VB)


def fixed_rule(name: str, cand: dict) -> B.Rule:
    c = cand["candidates"][name]
    if c["kind"] != "fixed":
        raise B.BenchmarkError(f"{name} is not a fixed rule")
    return B.Rule(name, {k: float(v) for k, v in c["weights"].items()}, bool(c["e"]))


def holdout_defs(cand: dict, names) -> list:
    """Rule definitions for the holdout read, taken only from the committed JSON."""
    out = []
    for n in names:
        if n not in cand["candidates"]:
            raise B.BenchmarkError(f"{n} is not in the committed candidates file: no holdout read")
        out.append({"name": n, **cand["candidates"][n]})
    return out


def population_gap(inner_pooled: dict, pop_pooled: dict) -> dict:
    """K12: population cost - nested inner cost, per reading."""
    return {k: (None if pop_pooled.get(k) is None or inner_pooled.get(k) is None else pop_pooled[k] - inner_pooled[k])
            for k in B.READINGS}


def log_rows(readout: dict, version: str, csv_path: str) -> list[dict]:
    """K13: one experiments/log.csv row per candidate."""
    rows = []
    ts = readout["timestamp"]
    for name, c in readout["candidates"].items():
        p = c["inner"]["pooled"]
        v = readout["verdicts"].get(name, {"verdict": "control"})["verdict"]
        rows.append({"timestamp": ts, "rung": f"V2-60 {name}", "validation_score": f"{p['package05']:.4f}",
                     "validation_score_clean_only": f"{p['package05']:.4f}", "csv_path": csv_path,
                     "notes": f"brief={p['brief']:.4f}; package03={p['package03']:.4f}; eer={p['eer']:.4f}; "
                              f"basis=inner 16,142 nested OOF with E; verdict={v}",
                     "benchmark_version": version})
    return rows


def append_log(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    with open(path) as f:
        header = next(csv.reader(f))
    if header != LOG_COLUMNS:
        raise B.BenchmarkError(f"{path} header {header} != {LOG_COLUMNS}")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        for r in rows:
            if set(r) != set(LOG_COLUMNS):
                raise B.BenchmarkError(f"log row fields {sorted(r)} != {sorted(LOG_COLUMNS)}")
            w.writerow(r)


def c2_offinner_weights(bench: B.Bench, cols, target) -> dict:
    """K5 model, fitted from inner arrays only."""
    inner = bench.pops["inner"]
    L = np.column_stack([bench.cols[c]["inner"] for c in cols])
    return F.fit_c2_offinner(L, inner.y, inner.fold, cols, target)


def itw_block(bench: B.Bench, rule: B.Rule, boot_n: int) -> dict:
    """ITW readout for one rule: pooled readings and counts, the stratified speaker-cluster bootstrap,
    and the same without E. Report-only (ITW is spent on v1 selection)."""
    r = bench.score_rule(rule, "itw")
    if not r["scored"]:
        return {"scored": False, "reason": r["reason"]}
    out = VB.readout_scores(bench.pops, {"itw": r["fused"]}, boot_n=boot_n)["itw"]
    out["without_e"] = {"pooled": B.readings(bench.pops["itw"].y, r["base"])}
    return out


def inner_readout(bench: B.Bench, name: str, fused, base, mask, weights: dict, refs_cols) -> dict:
    inner = bench.pops["inner"]
    real = inner.y == 0
    rk = {c: bench.ranks(c, "inner", "nested")[real] for c in refs_cols if weights.get(c, 0) > 0}
    return {
        **B.fold_table(inner.y, fused, inner.fold),
        "basis": F.BASIS,
        "counts": {k: B.counts_at(inner.y, fused, k) for k in B.READINGS},
        "without_e": B.fold_table(inner.y, base, inner.fold),
        "e_fires": {"real": int((mask & real).sum()), "spoof": int((mask & ~real).sum()),
                    "per_fold_real": {f: int((mask & real & (inner.fold == f)).sum()) for f in sorted(set(inner.fold))}},
        "dominance": B.dominance(rk, fused[real], weights),
    }


def run(boot_n: int = 2000, *, dry_run: bool = False) -> dict:
    cand = F.check_candidates(CANDIDATES, SPEC)
    blob = F.committed_blob_sha(CANDIDATES)
    reg = B.load_registry(VB.REGISTRY)
    problems = VB.check(reg, trees=True)
    if problems:
        raise B.BenchmarkError(f"registry check failed: {problems}")
    cols = cand["columns"]
    F.refuse_fit_feature(cols)
    e = cand["e_rule"]
    if (e["spectra_logit_below"], e["base_above"], e["multiply_by"]) != (
            B.E_RULE["spectra_logit_below"], B.E_RULE["base_above"], B.E_RULE["multiply_by"]):
        raise B.BenchmarkError(f"candidates e_rule {e} differs from the frozen E step {B.E_RULE}")
    bench = VB.load_bench(cols + [cand["e_column"]])
    inner = bench.pops["inner"]
    target = np.array([cand["candidates"]["C0"]["weights"][c] for c in cols])
    scores, readout = {}, {"candidates": {}, "verdicts": {}}

    # nested inner
    for name in ("C0", "C1", "C3"):
        rule = fixed_rule(name, cand)
        r = bench.score_rule(rule, "inner", "nested")
        scores[name] = r["fused"]
        readout["candidates"][name] = {"definition": cand["candidates"][name],
                                       "inner": inner_readout(bench, name, r["fused"], r["base"], r["e_mask"],
                                                              rule.weights, cols)}
    L = np.column_stack([bench.cols[c]["inner"] for c in cols])
    c2 = F.run_c2_nested(L, inner.y, inner.fold, bench.cols[B.E_COLUMN]["inner"], cols, target, population=inner)
    scores["C2"] = c2["fused"]
    c2_diag = {f: {k: v for k, v in d.items()} for f, d in c2["folds"].items()}
    readout["candidates"]["C2"] = {
        "definition": cand["candidates"]["C2"],
        "inner": inner_readout(bench, "C2", c2["fused"], c2["base"], c2["e_mask"], c2["mean_weights"], cols),
        "c2": {"folds": c2_diag, "alpha_spread": c2["alpha_spread"], "alpha_at_grid_max_all": c2["alpha_at_grid_max_all"],
               "optimism_mean": c2["optimism_mean"], "mean_weights": c2["mean_weights"], "platt": None},
    }

    # verdicts (nested inner only)
    def vin(name):
        p = readout["candidates"][name]["inner"]
        return {"basis": p["basis"], "pooled": p["pooled"], "f4_package05": p["per_fold"]["4"]["package05"]}

    for name in ("C1", "C2", "C3"):
        readout["verdicts"][name] = F.verdict(vin("C0"), vin(name), cand["tie_bands"],
                                              cand["f4_guard"]["max_above_control"],
                                              label_provisional=(name == "C2"))

    # paired bootstrap vs C0 (report-only)
    for name in ("C1", "C2", "C3"):
        readout["candidates"][name]["bootstrap_vs_C0"] = {
            mode: B.cluster_bootstrap(inner.y, scores["C0"], scores[name], inner.cluster, inner.stratum, n=boot_n,
                                      seed=0, lj_mode=mode)
            for mode in ("fixed", "cluster")}
        readout["candidates"][name]["bootstrap_per_fold_degenerate"] = {
            f: B.cluster_bootstrap(inner.y[inner.fold == f], scores["C0"][inner.fold == f],
                                   scores[name][inner.fold == f], inner.cluster[inner.fold == f],
                                   inner.stratum[inner.fold == f], n=1).get("degenerate")
            for f in ("3", "4")}

    # off-inner rules (C2 from inner only)
    off = c2_offinner_weights(bench, cols, target)
    readout["c2_offinner"] = {k: v for k, v in off.items() if k != "weights"}
    rules = {n: fixed_rule(n, cand) for n in ("C0", "C1", "C3")}
    rules["C2"] = B.Rule("C2", {c: float(w) for c, w in off["weights_by_col"].items()}, True)

    # holdout: one ledgered read for the whole candidate set
    def compute_holdout():
        h = bench.pops["holdout"]
        out = {}
        for n, rule in rules.items():
            r = bench.score_rule(rule, "holdout")
            out[n] = {"scored": False, "reason": r["reason"]} if not r["scored"] else {
                "pooled": B.readings(h.y, r["fused"]), "counts": {k: B.counts_at(h.y, r["fused"], k) for k in B.READINGS}}
        return out

    defs = holdout_defs(cand, ["C0", "C1", "C2", "C3"])
    if dry_run:
        compute_holdout = None
    meta = {"code_sha": VB._code_sha(), "input_shas": {n: f.shas for n, f in bench.files.items()},
            "candidates_blob_sha": blob}
    ledger = B.HoldoutLedger.default()
    if dry_run:
        hres = {"readout": {n: {"scored": False, "reason": "dry run: holdout not read"} for n in rules},
                "ledger": "dry run (no holdout read)"}
    else:
        hres = B.read_holdout(defs, compute_holdout, ledger=ledger, holdout_sha=VB.holdout_sha(bench.pops), meta=meta)
    readout["holdout"] = hres["readout"]
    readout["holdout_ledger"] = hres["ledger"]
    readout["ledger_count"] = ledger.count()

    # ITW, test, cohorts, MLAAD: report-only
    readout["itw"], readout["test"], readout["gaps"] = {}, {}, {}
    c0_thr_inner = readout["candidates"]["C0"]["inner"]["counts"][B.PRIMARY]["threshold"]
    c0_test = bench.score_rule(rules["C0"], "test")["fused"]
    for n, rule in rules.items():
        readout["itw"][n] = itw_block(bench, rule, boot_n)
        t = bench.score_rule(rule, "test")
        thr = readout["candidates"][n]["inner"]["counts"][B.PRIMARY]["threshold"]
        readout["test"][n] = B.agreement_with_shipped(bench.pops["test"].keys, t["fused"], thr, control_fused=c0_test,
                                                      control_threshold=c0_thr_inner) if t["scored"] else {
            "scored": False, "reason": t["reason"]}
        inner_p = readout["candidates"][n]["inner"]["pooled"]
        readout["gaps"][n] = {
            "holdout": population_gap(inner_p, readout["holdout"][n].get("pooled", {})),
            "itw": population_gap(inner_p, readout["itw"][n].get("pooled", {}))}
    frames, labels = VB.load_cohort_frames("holdout", bench.pops, pinned=reg["inputs"])
    holdout_cohort_files = VB.cohort_file_shas("holdout", frames, reg["inputs"])
    if dry_run:
        readout["holdout_cohort"] = {"scored": False, "reason": "dry run: holdout cohort not read (holdout labels)"}
    else:  # holdout labels: its own ledgered read for the candidate set
        cdefs = [{"population": "holdout_cohort", **d, **VB.cohort_key_extra(frames, holdout_cohort_files)}
                 for d in defs]
        co = B.read_holdout(cdefs, lambda: VB.cohort_readout(bench, rules, frames, labels), ledger=ledger,
                            holdout_sha=VB.cohort_sha(frames, labels),
                            meta={**meta, "cohort_files": holdout_cohort_files})
        readout["holdout_cohort"] = co["readout"]
        readout["holdout_cohort_ledger"] = co["ledger"]
        readout["ledger_count"] = ledger.count()
    itw_frames, itw_labels = VB.load_cohort_frames("itw", bench.pops, pinned=reg["inputs"])
    readout["cohort_files"] = {"holdout": holdout_cohort_files,
                               "itw": VB.cohort_file_shas("itw", itw_frames, reg["inputs"])}
    readout["itw_cohort"] = (VB.cohort_readout(bench, rules, itw_frames, itw_labels) if itw_frames else
                             {"scored": False, "reason": "pending: no ITW cohort files yet (Step 6, ask-first)"})
    thr = {n: readout["candidates"][n]["inner"]["counts"][B.PRIMARY]["threshold"] for n in rules}
    readout["mlaad"] = VB.mlaad_readout(bench, rules, thr)

    readout["dry_run"] = dry_run
    readout["followon"] = {"allowed": F.followon_allowed(readout["verdicts"]["C2"]["verdict"]),
                           "rule": cand["followon"]["larger_model"]}
    readout.update({"timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "candidates_blob_sha": blob,
                    "registry_sha256": B.sha256_file(VB.REGISTRY), "benchmark_version": reg["version"],
                    "code_sha": VB._code_sha(), "early_look_disclosure": cand["early_look_disclosure"],
                    "k1": cand["clarifications"]["K1"]})
    return readout


def report(path: Path = OUT) -> str:
    """Print the verdict table from a stored readout, refusing a stale one."""
    readout = json.loads(Path(path).read_text())
    F.check_readout_fresh(readout, candidates_blob=F.committed_blob_sha(CANDIDATES),
                          registry_sha=B.sha256_file(VB.REGISTRY))
    return summary(readout)


def summary(readout: dict) -> str:
    lines = ["| Candidate | brief | package-0.5 | package-0.3 | EER | f4 package-0.5 | Verdict |",
             "|---|---|---|---|---|---|---|"]
    for n, c in readout["candidates"].items():
        p = c["inner"]["pooled"]
        v = readout["verdicts"].get(n, {"verdict": "control"})["verdict"]
        lines.append(f"| {n} | {p['brief']:.4f} | {p['package05']:.4f} | {p['package03']:.4f} | {p['eer']:.4f} | "
                     f"{c['inner']['per_fold']['4']['package05']:.4f} | {v} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--force", action="store_true")
    r.add_argument("--boot-n", type=int, default=2000)
    r.add_argument("--no-log", action="store_true")
    r.add_argument("--dry-run", action="store_true", help="no holdout read, no log row, writes *_dryrun.json")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        print(report())
        return 0
    out = OUT.with_name("v2_60_readout_dryrun.json") if a.dry_run else OUT
    if out.exists() and not a.force:
        raise SystemExit(f"refusing to overwrite {out} (use --force)")
    readout = run(a.boot_n, dry_run=a.dry_run)
    VB.write_json(out, readout, force=a.force)
    if not a.no_log and not a.dry_run:
        append_log(LOG, log_rows(readout, readout["benchmark_version"], "outputs/v2/fusion/v2_60_readout.json"))
    print(summary(readout))
    print(json.dumps(B.clean({"verdicts": readout["verdicts"], "followon": readout["followon"],
                              "holdout_ledger": readout["holdout_ledger"]}), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
