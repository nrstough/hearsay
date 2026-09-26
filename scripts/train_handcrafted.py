"""Learned engineered detector (handcrafted spectral + prosody by default; compression
forensics with --features-dir outputs/compression --model-prefix cmp) on the fold file.

1. Per-feature AUC on the inner (training) rows: which cues separate real from synthetic.
2. Logistic regression (standardized, class-balanced) and LightGBM, each scored by
   out-of-fold minDCF on the predefined inner folds; the better one is refit on all inner rows.
   Every learned piece (the scaler included) is fit fold-locally.
3. One readout on the outer holdout (held-out generators + bona fide groups), all metric
   readings, per generator and per bona fide source.
4. Score export for fusion (M4): outputs/detector_scores/<out-name>.csv with columns
   path, fold, split, score, logit. split = inner_oof (out-of-fold on the predefined inner
   folds), holdout (model fit on all inner rows), test (the NSA test manifest, keyed by path,
   manifest order). score = P(synthetic); logit = the decision value = logit(score).
   Rows whose features failed keep their path with NaN score/logit (fusion imputes them).

The feature CSVs come from scripts/extract_handcrafted.py / extract_compression.py; their
.meta.json sidecar records crop_mode and band_match, which the saved bundle carries so the
contract detector prepares test audio the same way.

Usage: uv run python scripts/train_handcrafted.py --train nsa_train_sample_v3 \
    --folds splits/nsa_folds.csv [--test nsa_test_v3] [--out-name handcrafted]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import PredefinedSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hearsay.metrics import PI_SYNTH, eer, min_cost, report, sigmoid
from hearsay.trees import Trees

REPO = Path(__file__).resolve().parents[1]
META_COLS = {"path", "label", "generator", "speaker", "utt", "source", "filename", "group",
             "fold"}  # fmt: skip
META_SUFFIXES = ("_flag", "_launder", "_crop_s")
N_JOBS = 6  # leave cores for the GPU chat's decoding
# v4 family -> column prefixes, so a bundle only asks for the families its columns need
FAMILY_PREFIXES = {"lfcc": ("lfcc",), "phase": ("gd_", "pc_"), "cqcc": ("cqcc",),
                   "modulation": ("mod_",), "breath": ("breath_",), "jitter": ("jit_", "shim_")}  # fmt: skip


def is_meta(col: str) -> bool:
    return col in META_COLS or col.endswith(META_SUFFIXES)


def dropped(col: str, patterns: list[str]) -> bool:
    """`--drop-columns` entries are exact names or `prefix*` globs."""
    return any((pat.endswith("*") and col.startswith(pat[:-1])) or col == pat for pat in patterns)


def models() -> dict:
    return {
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
        ),
        "lgbm": lambda: LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
            verbose=-1, random_state=0, n_jobs=N_JOBS,
        ),
    }


def decision(model, X):
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def by_group(dv: pd.DataFrame, yv: np.ndarray, s: np.ndarray, pi: float) -> tuple[dict, dict]:
    """minDCF/EER per generator (its spoofs vs every bona fide clip) and per bona fide source
    (its bona fide clips vs every spoof)."""
    bona, spoof = s[yv == 0], s[yv == 1]
    per_gen, per_src = {}, {}
    for g in sorted(set(dv.generator) - {"bonafide"}):
        m = (dv.generator == g).to_numpy()
        yy, ss = np.r_[np.zeros(bona.size), np.ones(m.sum())], np.r_[bona, s[m]]
        per_gen[g] = {"min_dcf": round(min_cost(yy, ss, pi), 4), "eer": round(eer(yy, ss), 4)}
    for src in sorted(set(dv.loc[yv == 0, "source"])):
        m = ((dv.source == src) & (yv == 0)).to_numpy()
        yy, ss = np.r_[np.zeros(m.sum()), np.ones(spoof.size)], np.r_[s[m], spoof]
        per_src[src] = {"min_dcf": round(min_cost(yy, ss, pi), 4), "eer": round(eer(yy, ss), 4)}
    return per_gen, per_src


def score_rows(paths, fold, split, logit) -> pd.DataFrame:
    logit = np.asarray(logit, dtype=np.float64)
    return pd.DataFrame({"path": list(paths), "fold": fold, "split": split,
                         "score": sigmoid(logit), "logit": logit})  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="<features-dir>/<train>.csv")
    ap.add_argument("--folds", type=Path, required=True)
    ap.add_argument("--test", default="nsa_test", help="<features-dir>/<test>.csv")
    ap.add_argument("--test-manifest", type=Path,
                    default=REPO / "outputs" / "manifests" / "nsa_test.csv")
    ap.add_argument("--features-dir", type=Path, default=REPO / "outputs" / "handcrafted")
    ap.add_argument("--model-prefix", default="hc", help="models/<prefix>_<kind>_<stamp>/")
    ap.add_argument("--rung", default="D-track (handcrafted spectral+prosody)")
    ap.add_argument("--out-name", default="handcrafted",
                    help="outputs/detector_scores/<out-name>.csv")
    ap.add_argument("--pi-synth", type=float, default=PI_SYNTH)
    ap.add_argument("--drop-columns", default="",
                    help="comma-separated feature names or prefix* globs to leave out (v4 gate)")
    args = ap.parse_args()
    drop = [c for c in args.drop_columns.split(",") if c]

    t0 = time.time()
    fdir = args.features_dir.resolve()
    d_all = pd.read_csv(fdir / f"{args.train}.csv")
    flag_col = next(c for c in d_all.columns if c.endswith("_flag"))
    f = pd.read_csv(args.folds).set_index("path")[["group", "fold"]]
    d_all = d_all.join(f, on="path")
    assert d_all.fold.notna().all(), "rows missing from the fold file"
    d_all[flag_col] = d_all[flag_col].fillna("")
    d = d_all[d_all[flag_col] == ""].reset_index(drop=True)
    feat_cols = [c for c in d.columns if not is_meta(c) and not dropped(c, drop)]
    n_dropped = sum(1 for c in d.columns if not is_meta(c)) - len(feat_cols)
    X = d[feat_cols].to_numpy(np.float32)
    y = (d.label == "spoof").to_numpy(int)
    tr, va = (d.fold != "holdout").to_numpy(), (d.fold == "holdout").to_numpy()
    folds = d.loc[tr, "fold"].astype(int).to_numpy()
    side = fdir / f"{args.train}.meta.json"
    side_meta = json.loads(side.read_text()) if side.exists() else {}
    crop_mode = side_meta.get("crop_mode", "first4s")
    band_match = bool(side_meta.get("band_match", False))
    families = [str(n) for n in side_meta.get("families", [])]
    families = [fam for fam in families
                if any(c.startswith(FAMILY_PREFIXES.get(fam, (fam,))) for c in feat_cols)]  # fmt: skip
    n_laundered = int((d_all.get("cmp_launder", pd.Series(dtype=str)).fillna("") != "").sum())
    print(f"{args.train}: {len(d)} rows ({(d_all[flag_col] != '').sum()} feature failures "
          f"dropped), {len(feat_cols)} features, crop_mode={crop_mode}, band_match={band_match}, "
          f"families={families}, dropped={n_dropped} columns, laundered={n_laundered}, "
          f"inner {tr.sum()} / holdout {va.sum()}")

    auc = {c: round(float(roc_auc_score(y[tr], d.loc[tr, c])), 4) for c in feat_cols}
    top = sorted(auc.items(), key=lambda kv: -abs(kv[1] - 0.5))[:15]
    print("top single-feature AUCs (inner rows; >0.5 = higher on synthetic):")
    for k, v in top:
        print(f"  {k:24s} {v:.3f}")
    bona_tr = d.loc[tr & (y == 0), feat_cols]
    stats = {c: {"real_mean": float(bona_tr[c].mean()), "real_std": float(bona_tr[c].std(ddof=0)),
                 "auc": auc[c]} for c in feat_cols}  # fmt: skip

    cv, oofs = {}, {}
    for name, mk in models().items():
        oof = np.zeros(tr.sum())
        for a, b in PredefinedSplit(folds).split():
            oof[b] = decision(mk().fit(X[tr][a], y[tr][a]), X[tr][b])
        oofs[name] = oof
        cv[name] = {"min_dcf": round(min_cost(y[tr], oof, args.pi_synth), 4),
                    "eer": round(eer(y[tr], oof), 4)}  # fmt: skip
        print(f"CV {name}: {json.dumps(cv[name])}", flush=True)
    best = min(cv, key=lambda k: (cv[k]["min_dcf"], cv[k]["eer"]))
    oof = oofs[best]
    cv_gen, cv_src = by_group(d[tr].reset_index(drop=True), y[tr], oof, args.pi_synth)
    model = models()[best]().fit(X[tr], y[tr])

    s = decision(model, X[va])
    yv, dv = y[va], d[va].reset_index(drop=True)
    val = {f"val_{k}": v for k, v in report(yv, s, args.pi_synth).items()}
    val["val_auc"] = round(float(roc_auc_score(yv, s)), 4)
    per_gen, per_src = by_group(dv, yv, s, args.pi_synth)
    extra = {}
    if "cmp_launder" in dv:
        for name, m in (("clean", dv.cmp_launder.fillna("") == ""),
                        ("laundered", dv.cmp_launder.fillna("") != "")):  # fmt: skip
            m = m.to_numpy()
            if m.sum() and 0 < yv[m].sum() < m.sum():
                extra[f"val_{name}"] = {"n": int(m.sum()),
                                        "min_dcf": round(min_cost(yv[m], s[m], args.pi_synth), 4),
                                        "eer": round(eer(yv[m], s[m]), 4)}  # fmt: skip

    # NSA test set: keyed by path, in manifest order; failed rows stay as NaN.
    t = pd.read_csv(fdir / f"{args.test}.csv")
    tm = pd.read_csv(args.test_manifest)
    assert list(t.path) == list(tm.path), "test feature rows must match the test manifest order"
    t[flag_col] = t[flag_col].fillna("")
    t_ok = (t[flag_col] == "").to_numpy()
    st = np.full(len(t), np.nan)
    st[t_ok] = decision(model, t.loc[t_ok, feat_cols].to_numpy(np.float32))
    test_summary = {"n": len(t), "failures": int((~t_ok).sum()),
                    "score_mean": round(float(np.nanmean(sigmoid(st))), 4),
                    "frac_above_half": round(float(np.nanmean(sigmoid(st) > 0.5)), 4)}  # fmt: skip

    failed = d_all[d_all[flag_col] != ""]
    export = pd.concat([
        score_rows(d.loc[tr, "path"], d.loc[tr, "fold"].to_numpy(), "inner_oof", oof),
        score_rows(d.loc[va, "path"], "holdout", "holdout", s),
        score_rows(failed.path, failed.fold.to_numpy(),
                   np.where(failed.fold == "holdout", "holdout", "inner_oof"),
                   np.full(len(failed), np.nan)),
        score_rows(t.path, "test", "test", st),
    ], ignore_index=True)  # fmt: skip
    assert export.path.is_unique, "duplicate paths in the score export"
    assert len(export) == len(d_all) + len(t)

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")  # seconds: runs a minute apart collided
    out = REPO / "models" / f"{args.model_prefix}_{best}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    bundle = {"features": feat_cols, "kind": best, "feature_stats": stats,
              "crop_mode": crop_mode, "band_match": band_match, "families": families,
              "prefix": args.model_prefix,
              "train": args.train, "folds": str(args.folds)}  # fmt: skip
    if best == "lgbm":
        # Store the dumped trees, not the lightgbm object: inference must not import lightgbm
        # next to torch (hearsay.trees explains). Check the numpy evaluator reproduces lightgbm.
        dump = model.booster_.dump_model()
        gap = np.abs(Trees(dump).predict_proba(X[va])[:, 1] - model.predict_proba(X[va])[:, 1])
        assert gap.max() < 1e-6, f"numpy trees disagree with lightgbm by {gap.max():.2e}"
        bundle["trees_dump"] = dump
    else:
        bundle["model"] = model
    joblib.dump(bundle, out / "model.joblib")
    scores_path = REPO / "outputs" / "detector_scores" / f"{args.out_name}.csv"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(scores_path, index=False)
    export.to_csv(out / "scores.csv", index=False)
    fdir_txt = str(fdir.relative_to(REPO)) if fdir.is_relative_to(REPO) else str(fdir)
    meta = {"rung": args.rung, "train": args.train, "features_dir": fdir_txt,
            "crop_mode": crop_mode, "band_match": band_match, "families": families,
            "dropped_columns": drop, "n_laundered": n_laundered,
            "folds": str(args.folds), "model": best, "cv": cv,
            "cv_by_generator": cv_gen, "cv_by_bonafide_source": cv_src,
            "n_features": len(feat_cols), "n_inner": int(tr.sum()), "n_holdout": int(va.sum()),
            **val, **extra, "val_by_generator": per_gen, "val_by_bonafide_source": per_src,
            "test": test_summary, "scores": str(scores_path.relative_to(REPO)),
            "feature_auc_top15": dict(top), "feature_auc": auc,
            "seconds": round(time.time() - t0)}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("feature_auc_top15",
                                                                  "feature_auc")}, indent=2))
    print(f"saved {out}; scores -> {scores_path}")


if __name__ == "__main__":
    main()
