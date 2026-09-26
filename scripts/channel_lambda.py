"""Action A of the channel-robustness rung: lambda-hat, the NSA test set's wild-domain weight.

Label-free channel statistics per clip (run spec docs/specs/2026-09-26_channel-robustness.md,
D1-D3): a small classifier separates holdout real speech (clean, read) from In-the-Wild real
speech, and is then asked about the 1,671 test files. Two estimates: the mean posterior
P(wild) (the fusion consult's proposal) and adjusted classify-and-count from grouped
out-of-fold rates. The consult's crossover is 0.32.

Every clip is prepared the same way (hearsay.handcrafted._crop: band_limit -> trim_silence ->
8 s cap -> RMS normalize), so lead silence, level and the >7 kHz band are not domain cues.
Features that read above 7 kHz (bw_hz, band_7_8k_db, hf_slope_db_per_khz over 2-8 kHz) are
dropped: the test files are filtered twice (their own wall, then the pipeline's), the others
once, so those columns would separate the test set from both classes for a reason unrelated
to the recording channel.

Usage: uv run python scripts/channel_lambda.py [--workers 6] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "channel"
SHIPPED_TSV = REPO / "submissions" / "20260926-0914_M4_sweep_A3_w0.2_E_CANDIDATE_our_direction.tsv"
CROSSOVER = 0.32
MIN_MARGIN = 0.2  # TPR - FPR below this: adjusted classify-and-count is refused
DROP = ("bw_hz", "band_7_8k_db", "hf_slope_db_per_khz")  # read above 7 kHz (module docstring)
FRAME, HOP = 400, 160  # 25 ms frames, 10 ms hop at 16 kHz
BLOCK = 50  # frames per 0.5 s block for noise-floor stationarity
DECAY = 12  # frames (120 ms) after an offset for the decay slope


# ---------------------------------------------------------------- per-clip statistics


def _frame_db(y: np.ndarray) -> np.ndarray:
    if y.size < FRAME:
        y = np.pad(y, (0, FRAME - y.size))
    n = 1 + (y.size - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(y[idx] ** 2, axis=1) + 1e-12)


def extra_stats(y: np.ndarray) -> dict[str, float]:
    """The six added channel statistics on an already prepared (trimmed, RMS-normalized) clip."""
    import librosa

    e = _frame_db(np.asarray(y, dtype=np.float64))
    p90 = float(np.percentile(e, 90))
    f: dict[str, float] = {}
    f["floor_level_db"] = float(np.percentile(e, 10) - p90)
    blocks = [e[i : i + BLOCK] for i in range(0, e.size - BLOCK + 1, BLOCK)]
    f["floor_stationarity_db"] = (
        float(np.std([np.percentile(b, 10) for b in blocks])) if len(blocks) >= 2 else 0.0
    )
    sm = np.convolve(e, np.ones(3) / 3, mode="same")
    thr = float(np.percentile(sm, 50))
    offs = np.where((sm[:-1] >= thr) & (sm[1:] < thr))[0]
    slopes = []
    t = np.arange(DECAY) * HOP / 16000.0
    for o in offs:
        seg = sm[o : o + DECAY]
        if seg.size == DECAY:
            slopes.append(np.polyfit(t, seg, 1)[0])
    f["decay_slope_db_per_s"] = float(np.median(slopes)) if slopes else 0.0
    f["n_offsets"] = float(len(slopes))
    flat = librosa.feature.spectral_flatness(y=np.asarray(y, dtype=np.float32) + 1e-9,
                                             n_fft=512, hop_length=HOP)[0]  # fmt: skip
    f["flatness"] = float(np.mean(flat))
    active = e[e > p90 - 40]
    f["loud_std_db"] = float(np.std(active)) if active.size else 0.0
    f["pause_frac"] = float(np.mean(e < p90 - 30))
    return {k: (v if np.isfinite(v) else 0.0) for k, v in f.items()}


def channel_stats(x: np.ndarray) -> dict[str, float]:
    """D1: hearsay.compression.features (no crop, band match on) + extra_stats on the same
    prepared signal. Deterministic, level-invariant (RMS-normalized first)."""
    from hearsay.compression import features
    from hearsay.handcrafted import _crop

    x = np.asarray(x, dtype=np.float32)
    f = features(x, None, None, "segment", True)
    f.update(extra_stats(_crop(x, None, None, "segment", True)))
    return f


def _row(key: str) -> dict | None:
    """key = path, or path|codec-kbpsk@sr for a laundered control clip."""
    from hearsay.audio import load_audio
    from hearsay.compression import launder, parse_laundering

    path, _, spec = key.partition("|")
    try:
        x = load_audio(path)
        if spec:
            x = launder(x, *parse_laundering(spec))
        return {"key": key, **channel_stats(x)}
    except Exception as e:  # noqa: BLE001 - reported by the caller
        return {"key": key, "error": repr(e)[:200]}


# ---------------------------------------------------------------- classifier and estimators


def grouped_folds(groups: np.ndarray, k: int = 5) -> np.ndarray:
    """Validation fold id per row; every group lands in exactly one fold (GroupKFold)."""
    from sklearn.model_selection import GroupKFold

    fold = np.full(len(groups), -1)
    for i, (_, va) in enumerate(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=groups)):
        fold[va] = i
    return fold


def folds_leak(groups: np.ndarray, fold: np.ndarray) -> bool:
    """True if any group appears in more than one validation fold."""
    return bool(pd.DataFrame({"g": groups, "f": fold}).groupby("g").f.nunique().max() > 1)


def make_clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced",
                                                              max_iter=5000))  # fmt: skip


def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5) -> np.ndarray:
    fold = grouped_folds(groups, k)
    assert not folds_leak(groups, fold)
    p = np.full(len(y), np.nan)
    for i in range(k):
        va = fold == i
        p[va] = make_clf().fit(X[~va], y[~va]).predict_proba(X[va])[:, 1]
    return p


def lambda_mean_posterior(p_test: np.ndarray) -> float:
    return float(np.mean(p_test))


def lambda_acc(q: float, tpr: float, fpr: float, min_margin: float = MIN_MARGIN) -> tuple[float, str]:
    """Adjusted classify-and-count: (q - FPR) / (TPR - FPR), clipped to [0, 1]. Refused (NaN
    plus the reason) when the classifier barely separates the domains."""
    if not (tpr - fpr >= min_margin):
        return float("nan"), f"TPR - FPR = {tpr - fpr:.3f} < {min_margin}: classifier too weak"
    return float(np.clip((q - fpr) / (tpr - fpr), 0.0, 1.0)), ""


def feature_position(med_test: float, med_clean: float, med_wild: float) -> float:
    """0 = the test median sits at the clean median, 1 = at the wild median; outside [0, 1]
    means beyond either. NaN when the two domains share a median."""
    d = med_wild - med_clean
    return float((med_test - med_clean) / d) if abs(d) > 1e-9 else float("nan")


def bootstrap_ci(stat, arrays: list[np.ndarray], n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat(*resampled arrays); each array resampled independently."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        vals.append(stat(*[a[rng.integers(0, len(a), len(a))] for a in arrays]))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cluster_index(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a bootstrap sample drawn by whole groups (speakers), with replacement."""
    u = np.unique(groups)
    pick = u[rng.integers(0, len(u), len(u))]
    by = {g: np.flatnonzero(groups == g) for g in u}
    return np.concatenate([by[g] for g in pick])


def acc_ci(p_test: np.ndarray, p_clean: np.ndarray, g_clean: np.ndarray, p_wild: np.ndarray,
           g_wild: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of the adjusted estimate: test files resampled as rows, the reference
    domains resampled by speaker (their OOF rows are not independent within a speaker)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        t = p_test[rng.integers(0, len(p_test), len(p_test))]
        c, w = p_clean[cluster_index(g_clean, rng)], p_wild[cluster_index(g_wild, rng)]
        vals.append(acc_stat(t, c, w))
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 0.9 * n:  # too many refused resamples: the interval is not defined
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def identifiable(auc: float, ctl: dict, lam: float, lam_no_lj: float) -> tuple[bool, list[str]]:
    """Pre-declared (run spec D3b): lambda-hat is usable against the crossover only if the
    classifier separates the domains, an unseen clean read corpus reads clean, synthetic speech
    does not read wild, and dropping LJ from the clean side moves the estimate by <= 0.15.
    Any missing input fails the check (never a silent pass)."""
    why = []
    if not auc >= 0.8:
        why.append(f"domain AUC {auc} < 0.8")
    v = ctl.get("ctl_vctk_clean_read", {}).get("share_above_0.5")
    if v is None or not v < 0.25:
        why.append(f"unseen clean read speech (VCTK) reads wild: share {v}")
    sp = ctl.get("ctl_diffssd_spoof", {}).get("share_above_0.5")
    if sp is None or not sp < 0.25:
        why.append(f"synthetic speech reads wild: share {sp}")
    if not abs(lam - lam_no_lj) <= 0.15:
        why.append(f"LJ-excluded estimate {lam_no_lj} differs from {lam} by > 0.15")
    return (not why), why


def acc_stat(p_test: np.ndarray, p_clean: np.ndarray, p_wild: np.ndarray) -> float:
    return lambda_acc(float(np.mean(p_test > 0.5)), float(np.mean(p_wild > 0.5)),
                      float(np.mean(p_clean > 0.5)))[0]  # fmt: skip


def novelty_share(X_train: np.ndarray, X_test: np.ndarray, pct: float = 99.0) -> float:
    """Share of test rows beyond the training rows' `pct` Mahalanobis distance (standardized,
    ridge-regularized covariance): how much of the test set is like neither domain."""
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-9
    Z, T = (X_train - mu) / sd, (X_test - mu) / sd
    inv = np.linalg.inv(np.cov(Z, rowvar=False) + 1e-3 * np.eye(Z.shape[1]))
    d_tr = np.einsum("ij,jk,ik->i", Z, inv, Z)
    d_te = np.einsum("ij,jk,ik->i", T, inv, T)
    return float(np.mean(d_te > np.percentile(d_tr, pct)))


# ---------------------------------------------------------------- driver


def _abs(p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else (REPO / q).resolve())


def rows() -> pd.DataFrame:
    f = pd.read_csv(REPO / "splits" / "nsa_folds.csv")
    clean = f[(f.fold == "holdout") & (f.label == "bonafide")]
    # Speaker groups, not the fold file's `group` (LJ is grouped by chapter there): LJ is one
    # speaker, so it is one group here and never spans a training and a validation fold.
    clean = pd.DataFrame({"path": clean.path.map(_abs), "domain": "clean",
                          "group": "spk_" + clean.speaker.astype(str), "source": clean.source})  # fmt: skip
    itw = pd.read_csv(REPO / "outputs" / "manifests" / "itw_stress.csv")
    itw = itw[itw.label == "bonafide"]
    wild = pd.DataFrame({"path": itw.path.map(_abs), "domain": "wild",
                         "group": "itw_" + itw.speaker.astype(str), "source": "in_the_wild"})  # fmt: skip
    t = pd.read_csv(REPO / "outputs" / "manifests" / "nsa_test.csv")
    test = pd.DataFrame({"path": t.path.map(_abs), "domain": "test", "group": "test",
                         "source": "nsa_test", "filename": t.filename})  # fmt: skip
    # Negative controls, never trained on (the report reads them to say what the classifier learned).
    rng = 0
    inner = f[(f.fold != "holdout")]
    vctk = pd.read_csv(REPO / "outputs" / "manifests" / "asv19_addon.csv")
    vctk = vctk[vctk.label == "bonafide"].sample(300, random_state=rng)
    spoof = inner[inner.label == "spoof"].groupby("generator", group_keys=False).apply(
        lambda d: d.sample(min(len(d), 30), random_state=rng))
    real_in = inner[inner.label == "bonafide"].sample(300, random_state=rng)
    ctl = [
        pd.DataFrame({"path": vctk.path.map(_abs), "domain": "ctl_vctk_clean_read", "launder": ""}),
        pd.DataFrame({"path": spoof.path.map(_abs), "domain": "ctl_diffssd_spoof", "launder": ""}),
        pd.DataFrame({"path": real_in.path.map(_abs), "domain": "ctl_inner_real", "launder": ""}),
        pd.DataFrame({"path": real_in.path.map(_abs), "domain": "ctl_inner_real_mp3_64k@16k",
                      "launder": "mp3-64k@16000"}),
    ]  # fmt: skip
    out = pd.concat([clean, wild, test, *ctl], ignore_index=True)
    out["launder"] = out.launder.fillna("") if "launder" in out else ""
    out["key"] = out.path + np.where(out.launder != "", "|" + out.launder, "")
    return out


def extract(r: pd.DataFrame, workers: int) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "lambda_features.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["key"])
    if "key" not in done:
        done = done.rename(columns={"path": "key"})
    todo = [k for k in r.key if k not in set(done.key)]
    print(f"features: {len(done)} cached, {len(todo)} to extract", flush=True)
    new = []
    if todo:
        with ProcessPoolExecutor(workers) as ex:
            for i, row in enumerate(ex.map(_row, todo, chunksize=8)):
                new.append(row)
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(todo)}", flush=True)
                    pd.concat([done, pd.DataFrame(new)]).to_csv(cache, index=False)
    feats = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
    feats.to_csv(cache, index=False)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="rows per domain (smoke runs)")
    args = ap.parse_args()
    t0 = time.time()

    r = rows()
    if args.limit:
        r = r.groupby("domain", group_keys=False).head(args.limit)
    feats = extract(r, args.workers)
    if "error" in feats:
        bad = feats[feats.error.notna()]
        if len(bad):
            print(f"{len(bad)} extraction failures, e.g. {bad.iloc[0].to_dict()}")
        if len(bad) > 0.01 * len(r):
            raise SystemExit("more than 1% of clips failed to extract; refusing to estimate")
        feats = feats[feats.error.isna()].drop(columns="error")
    d = r.merge(feats, on="key", how="inner")
    cols = [c for c in feats.columns if c not in ("key", "error", "n_offsets") and c not in DROP]

    tr = d[d.domain.isin(["clean", "wild"])].reset_index(drop=True)
    ctl = d[d.domain.str.startswith("ctl_")].reset_index(drop=True)
    te = d[d.domain == "test"].reset_index(drop=True)
    X, y, g = tr[cols].to_numpy(float), (tr.domain == "wild").to_numpy(int), tr.group.to_numpy()
    Xt = te[cols].to_numpy(float)

    from sklearn.metrics import roc_auc_score

    oof = grouped_oof(X, y, g)
    auc = float(roc_auc_score(y, oof))
    clf = make_clf().fit(X, y)
    p_test = clf.predict_proba(Xt)[:, 1]
    p_clean, p_wild = oof[y == 0], oof[y == 1]
    tpr, fpr = float(np.mean(p_wild > 0.5)), float(np.mean(p_clean > 0.5))
    q = float(np.mean(p_test > 0.5))
    lam_acc, why = lambda_acc(q, tpr, fpr)

    res = {
        "n": {"clean": int((y == 0).sum()), "wild": int((y == 1).sum()), "test": len(te)},
        "features": cols, "dropped": list(DROP),
        "domain_auc_grouped_oof": round(auc, 4), "tpr_oof": round(tpr, 4), "fpr_oof": round(fpr, 4),
        "lambda_mean_posterior": round(lambda_mean_posterior(p_test), 4),
        "lambda_mean_posterior_ci95": [round(v, 4) for v in bootstrap_ci(lambda_mean_posterior, [p_test])],
        "test_share_above_0.5": round(q, 4),
        "lambda_acc": None if np.isnan(lam_acc) else round(lam_acc, 4), "lambda_acc_refused": why,
        "lambda_acc_ci95": [round(v, 4) for v in acc_ci(p_test, p_clean, g[y == 0], p_wild, g[y == 1])],
        "oof_mean_posterior": {"clean": round(float(p_clean.mean()), 4), "wild": round(float(p_wild.mean()), 4)},
        "crossover": CROSSOVER,
        "novelty_share_p99": round(novelty_share(X, Xt), 4),
        "novelty_share_p99_train_baseline": 0.01,
    }  # fmt: skip

    if SHIPPED_TSV.exists():  # the files the shipped rule calls real: lowest 70% of its scores
        s = pd.read_csv(SHIPPED_TSV, sep="\t")
        s = s.rename(columns={s.columns[0]: "filename", s.columns[1]: "score"})
        low = set(s.nsmallest(round(0.7 * len(s)), "score").filename)
        m = te.filename.isin(low).to_numpy()
        res["likely_real_subset"] = {
            "n": int(m.sum()), "lambda_mean_posterior": round(float(p_test[m].mean()), 4),
            "ci95": [round(v, 4) for v in bootstrap_ci(lambda_mean_posterior, [p_test[m]])],
            "lambda_acc": round(acc_stat(p_test[m], p_clean, p_wild), 4),
            "source": str(SHIPPED_TSV.relative_to(REPO)),
        }  # fmt: skip

    pos = {}
    for c in cols:
        mc, mw, mt = (float(tr.loc[y == 0, c].median()), float(tr.loc[y == 1, c].median()),
                      float(te[c].median()))  # fmt: skip
        iqr = float(tr[c].quantile(0.75) - tr[c].quantile(0.25)) or 1.0
        pos[c] = {"clean": round(mc, 3), "wild": round(mw, 3), "test": round(mt, 3),
                  "t": round(feature_position(mt, mc, mw), 3),
                  "gap_in_iqr": round(abs(mw - mc) / iqr, 3)}  # fmt: skip
    res["per_feature"] = dict(sorted(pos.items(), key=lambda kv: -kv[1]["gap_in_iqr"]))
    coef = clf[-1].coef_[0]
    res["coef_std"] = {c: round(float(w), 3) for c, w in sorted(zip(cols, coef), key=lambda kv: -abs(kv[1]))}
    res["per_source_oof_mean"] = {s: round(float(oof[(tr.source == s).to_numpy()].mean()), 4)
                                  for s in sorted(tr.source.unique())}  # fmt: skip
    # Sensitivity: the clean side without LJ (one speaker, 885 of 1,858 clean clips).
    keep = (tr.source != "ljspeech").to_numpy()
    oof2 = grouped_oof(X[keep], y[keep], g[keep])
    p_test2 = make_clf().fit(X[keep], y[keep]).predict_proba(Xt)[:, 1]
    lam_no_lj = acc_stat(p_test2, oof2[y[keep] == 0], oof2[y[keep] == 1])
    res["lambda_acc_without_lj"] = None if np.isnan(lam_no_lj) else round(lam_no_lj, 4)
    res["lambda_mean_posterior_without_lj"] = round(float(p_test2.mean()), 4)

    # Sensitivity: without the single largest-|coefficient| feature, and without every
    # spectral-floor/hole column (the codec fingerprint family), so a one-feature answer shows.
    top = max(zip(cols, clf[-1].coef_[0]), key=lambda kv: abs(kv[1]))[0]
    for tag, drop in (("without_top_feature", {top}),
                      ("without_floor_and_hole_features",
                       {c for c in cols if "hole" in c or "floor" in c or c == "valley_depth_db"})):
        keep_c = [i for i, c in enumerate(cols) if c not in drop]
        o3 = grouped_oof(X[:, keep_c], y, g)
        p3 = make_clf().fit(X[:, keep_c], y).predict_proba(Xt[:, keep_c])[:, 1]
        res[f"lambda_acc_{tag}"] = {
            "dropped": sorted(drop), "domain_auc": round(float(roc_auc_score(y, o3)), 4),
            "lambda_acc": round(acc_stat(p3, o3[y == 0], o3[y == 1]), 4),
            "lambda_mean_posterior": round(float(p3.mean()), 4)}  # fmt: skip

    p_ctl = clf.predict_proba(ctl[cols].to_numpy(float))[:, 1] if len(ctl) else np.array([])
    res["controls_mean_pwild"] = {
        dom: {"n": int((ctl.domain == dom).sum()),
              "mean_p_wild": round(float(p_ctl[(ctl.domain == dom).to_numpy()].mean()), 4),
              "share_above_0.5": round(float((p_ctl[(ctl.domain == dom).to_numpy()] > 0.5).mean()), 4)}
        for dom in sorted(ctl.domain.unique())
    }  # fmt: skip
    ok, why = identifiable(auc, res["controls_mean_pwild"], lam_acc, lam_no_lj)
    res["identifiable"], res["identifiable_failures"] = ok, why
    res["lambda_hat"] = (round(lam_acc, 4) if ok and np.isfinite(lam_acc) else None)
    res["lambda_hat_reading"] = (
        "unidentifiable: " + "; ".join(why) if not ok else
        f"lambda-hat {lam_acc:.3f} vs crossover {CROSSOVER}: "
        + ("above (wild-like test set; In-the-Wild should weigh heavily)" if lam_acc > CROSSOVER
           else "below (holdout-like test set)"))  # fmt: skip
    res["seconds"] = round(time.time() - t0)
    (OUT / "lambda.json").write_text(json.dumps(res, indent=2))
    te.assign(p_wild=p_test)[["filename", "p_wild"]].to_csv(OUT / "lambda_test_pwild.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_feature", "coef_std", "features")},
                     indent=2))  # fmt: skip
    print(f"wrote {OUT / 'lambda.json'}")


if __name__ == "__main__":
    main()
