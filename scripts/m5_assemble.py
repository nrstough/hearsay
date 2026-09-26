"""Assemble the M5 deliverables from pulled run directories (outputs/m5_runs/<job>/<run>/).

Writes outputs/detector_scores/m5_xlsr_ft.csv (path, fold, split, score, logit) with exactly
the fold file's inner rows as inner_oof (from the five fold models), its holdout rows and the
test rows (from the full model), and models/m5_xlsr_ft_<stamp>/meta.json with the holdout
readouts (clean, augmented slice, test-length crops, per generator / bona fide source / length
bucket / augmentation op), the pooled out-of-fold readout, the M1 bar and the gate.

If any fold model is missing, NO inner_oof rows are written (never a partial stacking set);
meta.json then says stackable=false.

Usage: uv run python scripts/m5_assemble.py --runs outputs/m5_runs --arm nsa_extra
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hearsay.m5_data import FOLDS, INNER_FOLDS, check_folds_file
from hearsay.metrics import eer, min_cost, report

REPO = Path(__file__).resolve().parents[1]
def _m1_dirs() -> list[tuple[Path, dict]]:
    out = []
    for d in sorted((REPO / "models").glob("m1_*"), key=lambda p: p.stat().st_mtime):
        mp = d / "meta.json"
        if mp.exists():
            out.append((d, json.loads(mp.read_text())))
    return out


def newest_m1_dir(nsa_only: bool = True) -> Path:
    """The bar is M1, the NSA-only probe (oversight chat, 04:40): the newest models/m1_* whose
    `train` is a single NSA sample set (no comma-separated extra sets). M1b (with ASV19) is
    reported beside it, never gated on. --m1-dir overrides."""
    cands = [(d, m) for d, m in _m1_dirs()
             if (m.get("train", "").startswith("nsa_train_sample") and "," not in m.get("train", ""))
             == nsa_only]
    return cands[-1][0] if cands else REPO / "models" / "m1_missing"


M1_DIR = newest_m1_dir()


def find_runs(root: Path, arm: str) -> dict[str, Path]:
    """{fold or 'full': run dir} for the newest DONE run per fold of the given arm."""
    out: dict[str, Path] = {}
    for rm in sorted(root.rglob("run_meta.json")):
        meta = json.loads(rm.read_text())
        if meta.get("arm") != arm:
            continue
        d = rm.parent
        if not (d / "model" / "hashes.json").exists():
            continue
        out[str(meta["fold"])] = d
    return out


def _m1b_comparison(val_clean: dict, oof_pooled: dict | None, itw: dict | None) -> dict | None:
    """M1b (NSA + ASV19 bona fide) beside the gate, for the fusion decision; never gated on."""
    d = newest_m1_dir(nsa_only=False)
    if not (d / "meta.json").exists():
        return None
    m = json.loads((d / "meta.json").read_text())
    return {"model_dir": d.name, "train": m.get("train"), "m1b_holdout_min_dcf": m.get("val_min_dcf"),
            "m5_holdout_min_dcf": val_clean["min_dcf"],
            "m1b_itw_pfa": (m.get("stress") or {}).get("p_fa_at_inner_thr"),
            "m5_itw_pfa": None if itw is None else itw["pfa"],
            "m5_oof_pooled_min_dcf": None if oof_pooled is None else oof_pooled["min_dcf"]}


def by_group(dv: pd.DataFrame, yv: np.ndarray, s: np.ndarray, pi: float = 0.3) -> tuple[dict, dict]:
    """Copied from scripts/train_handcrafted.py: per generator (its spoofs vs all bona fide)
    and per bona fide source (its bona fide vs all spoofs)."""
    bona, spoof = s[yv == 0], s[yv == 1]
    per_gen, per_src = {}, {}
    for g in sorted(set(dv.generator) - {"bonafide"}):
        m = (dv.generator == g).to_numpy()
        yy, ss = np.r_[np.zeros(bona.size), np.ones(m.sum())], np.r_[bona, s[m]]
        per_gen[g] = {"min_dcf": round(min_cost(yy, ss, pi), 4), "eer": round(eer(yy, ss), 4),
                      "n": int(m.sum())}
    for src in sorted(set(dv.loc[yv == 0, "source"])):
        m = ((dv.source == src) & (yv == 0)).to_numpy()
        yy, ss = np.r_[np.zeros(m.sum()), np.ones(spoof.size)], np.r_[s[m], spoof]
        per_src[src] = {"min_dcf": round(min_cost(yy, ss, pi), 4), "eer": round(eer(yy, ss), 4),
                        "n": int(m.sum())}
    return per_gen, per_src


def by_length(dv: pd.DataFrame, yv: np.ndarray, s: np.ndarray, durations: np.ndarray) -> dict:
    out = {}
    for name, lo, hi in (("le4s", 0, 4), ("4to6s", 4, 6), ("gt6s", 6, 1e9)):
        m = (durations > lo) & (durations <= hi)
        if m.sum() >= 20 and len(set(yv[m])) == 2:
            out[name] = {**report(yv[m], s[m]), "n": int(m.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=REPO / "outputs" / "m5_runs")
    ap.add_argument("--arm", default="nsa_extra")
    ap.add_argument("--bundle-manifest", type=Path,
                    default=Path("/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/m5_bundle/v1/manifest.csv"))
    ap.add_argument("--out-name", default="m5_xlsr_ft")
    ap.add_argument("--m1-dir", type=Path, default=M1_DIR)
    ap.add_argument("--spend-usd", type=float, default=None)
    ap.add_argument("--itw-scores", type=Path, default=None,
                    help="m5_score.py output on In-the-Wild BONA FIDE rows (eval-only read-out)")
    ap.add_argument("--m1-itw-pfa", type=float, default=0.012,
                    help="M1's ITW real P_FA at its inner-fold threshold (main chat)")
    args = ap.parse_args()
    t0 = time.time()

    check_folds_file()
    folds = pd.read_csv(FOLDS)
    folds["fold"] = folds.fold.astype(str)
    test = pd.read_csv(REPO / "outputs/manifests/nsa_test.csv")
    runs = find_runs(args.runs, args.arm)
    print("runs:", {k: str(v) for k, v in runs.items()})
    if "full" not in runs:
        raise SystemExit("no full-model run for this arm")
    full = runs["full"]
    have_all_folds = all(k in runs for k in INNER_FOLDS)

    parts = []
    cv = {}
    if have_all_folds:
        for k in INNER_FOLDS:
            s = pd.read_csv(runs[k] / f"scores_{k}.csv")
            assert (s.split == "inner_oof").all() and (s.fold.astype(str) == k).all()
            parts.append(s)
            lab = folds.set_index("path").loc[s.path, "label"]
            cv[k] = report((lab == "spoof").to_numpy(int), s.logit.to_numpy())
        oof = pd.concat(parts, ignore_index=True)
        inner = folds[folds.fold != "holdout"]
        assert set(oof.path) == set(inner.path), "inner_oof rows must equal the fold file's inner rows"
        assert len(oof) == len(inner) == 16142
    hold = pd.read_csv(full / "scores_holdout.csv")
    tst = pd.read_csv(full / "scores_test.csv")
    hold_rows = folds[folds.fold == "holdout"]
    assert set(hold.path) == set(hold_rows.path) and len(hold) == 3858
    assert list(tst.path) == list(test.path) and len(tst) == 1671
    export = pd.concat(parts + [hold, tst], ignore_index=True)
    assert export.path.is_unique
    assert set(export.split) <= {"inner_oof", "holdout", "test"}
    assert np.isfinite(export.logit).all() and export.score.between(0, 1).all()

    # --- readouts ---
    hj = hold.set_index("path").join(hold_rows.set_index("path")[["label", "generator", "source"]])
    hy = (hj.label == "spoof").to_numpy(int)
    hs = hj.logit.to_numpy()
    val_clean = report(hy, hs)
    per_gen, per_src = by_group(hj.reset_index(), hy, hs)
    bm = pd.read_csv(args.bundle_manifest)
    dur = bm.set_index("path").duration.reindex(hj.index).to_numpy()
    val_len = by_length(hj, hy, hs, dur)
    bona = hy == 0
    rho = spearmanr(hs[bona], np.log(dur[bona] + 1e-6)).correlation if bona.sum() > 10 else None

    diag = {}
    for name in ("diag_holdout_testlen", "diag_holdout_aug"):
        p = full / f"{name}.csv"
        if p.exists():
            d = pd.read_csv(p).set_index("path").reindex(hj.index)
            diag[name] = report(hy, d.logit.to_numpy())
            if "op" in d:
                diag[name + "_by_op"] = {
                    op: report(hy[(d.op == op).to_numpy()], d.logit.to_numpy()[(d.op == op).to_numpy()])
                    for op in sorted(set(d.op.dropna()))
                    if len(set(hy[(d.op == op).to_numpy()])) == 2}
    oof_pooled = report((folds.set_index("path").loc[oof.path, "label"] == "spoof").to_numpy(int),
                        oof.logit.to_numpy()) if have_all_folds else None
    m1 = json.loads((args.m1_dir / "meta.json").read_text()) if args.m1_dir.exists() else {}
    m1_bar = m1.get("val_min_dcf")
    # M1's ITW real P_FA comes from the same meta.json (train_probe --stress block); the flag is
    # only the fallback, so the gate never compares against a stale bar (oversight chat, 04:35)
    m1_itw_pfa = (m1.get("stress") or {}).get("p_fa_at_inner_thr", args.m1_itw_pfa)
    m1_itw_src = "meta.json stress block" if (m1.get("stress") or {}).get("p_fa_at_inner_thr") is not None else "--m1-itw-pfa flag"
    # In-the-Wild real-speech false-alarm rate at the OOF minDCF threshold (eval-only)
    itw = None
    if have_all_folds and args.itw_scores and args.itw_scores.exists():
        from hearsay.metrics import C_FA, C_MISS, PI_SYNTH, cost_at

        oy = (folds.set_index("path").loc[oof.path, "label"] == "spoof").to_numpy(int)
        os_ = oof.logit.to_numpy()
        cand = np.unique(os_)
        cand = cand[:: max(1, len(cand) // 4000)]
        thr = float(cand[int(np.argmin([cost_at(oy, os_, t, PI_SYNTH, C_FA, C_MISS) for t in cand]))])
        it = pd.read_csv(args.itw_scores)
        it = it[it.flag.fillna("") == ""]
        pfa = float((it.logit.to_numpy() > thr).mean())
        itw = {"n_bonafide": len(it), "oof_threshold_logit": round(thr, 4),
               "pfa": round(pfa, 4), "m1_pfa": m1_itw_pfa, "m1_pfa_source": m1_itw_src,
               "passed": pfa <= m1_itw_pfa}
    gate = {
        "m1_holdout_min_dcf": m1_bar,
        "m5_holdout_min_dcf": val_clean["min_dcf"],
        "holdout_passed": (m1_bar is not None and val_clean["min_dcf"] < m1_bar),
        "holdout_margin": None if m1_bar is None else round(m1_bar - val_clean["min_dcf"], 4),
        "m5_oof_pooled_min_dcf": None if oof_pooled is None else oof_pooled["min_dcf"],
        "itw_bonafide": itw,
        "rule": "replace M1 only if M5 beats it on the holdout AND on the pooled 5-fold OOF AND "
                "its In-the-Wild real P_FA at the OOF minDCF threshold is <= M1's (main chat, "
                "04:00); a holdout gap under 0.15 is within noise (consult)",
        "noise_note": "holdout sigma(minDCF) ~ 0.07-0.10 with 2 generators and 26 bona fide groups",
    }
    run_meta = json.loads((full / "run_meta.json").read_text())
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    out_dir = REPO / "models" / f"{args.out_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(full / "model", out_dir / "model", dirs_exist_ok=True)
    scores_path = REPO / "outputs" / "detector_scores" / f"{args.out_name}.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(scores_path, index=False)
    export.to_csv(out_dir / "scores.csv", index=False)
    meta = {
        "rung": "M5", "arm": args.arm, "stackable": bool(have_all_folds),
        "config": run_meta["config"], "config_hash": run_meta["config_hash"],
        "steps": run_meta["steps"], "bundle_tree": run_meta["bundle_tree"],
        "git_sha": run_meta.get("git_sha"), "gpu": run_meta.get("gpu"),
        "runs": {k: str(v.relative_to(REPO)) if v.is_relative_to(REPO) else str(v) for k, v in runs.items()},
        "cv": cv, "oof_pooled": oof_pooled, "n_inner": len(oof) if have_all_folds else 0,
        "n_holdout": len(hold),
        **{f"val_{k}": v for k, v in val_clean.items()},
        "val_by_generator": per_gen, "val_by_bonafide_source": per_src,
        "val_by_length_bucket": val_len,
        "score_duration_spearman_bonafide": None if rho is None else round(float(rho), 4),
        "val_diagnostics": diag,
        "test": {"n": len(tst), "score_mean": round(float(tst.score.mean()), 4),
                 "frac_above_half": round(float((tst.score > 0.5).mean()), 4)},
        "m1_bar": {"val_min_dcf": m1_bar, "model_dir": str(args.m1_dir.name),
                   "m1_train": m1.get("train"), "m1_itw_pfa": m1_itw_pfa},
        "m1b_comparison": _m1b_comparison(val_clean, oof_pooled, itw),
        "gate": gate, "aug_rates": run_meta.get("aug_rates"),
        "shortcut_gate": run_meta.get("shortcut_gate"), "hashes": run_meta.get("hashes"),
        "spend_usd": args.spend_usd, "scores": str(scores_path.relative_to(REPO)),
        "seconds": round(time.time() - t0),
    }  # fmt: skip
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("stackable", "val_min_dcf", "val_eer", "oof_pooled",
                                            "val_by_generator", "val_by_bonafide_source",
                                            "val_by_length_bucket", "gate", "test")}, indent=1))
    print(f"saved {out_dir}; scores -> {scores_path}")


if __name__ == "__main__":
    main()
