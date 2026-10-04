"""Exploration (v2 chunks V2-20 / V2-21): do different layers and backbones fail on different files?

Question (Nathan, Oct 3): would more layers help because they fail differently, and does WavLM
fail the opposite way to XLS-R?

What it does, on embeddings already on disk (no extraction, no GPU):
  1. For every layer of XLS-R 300M (v3 sets) and WavLM Large (wl sets), the M1b recipe exactly:
     StandardScaler + class-balanced logistic regression (C = 1), out-of-fold on the five inner
     folds of splits/nsa_folds_plus_asv19.csv, a Platt map on the pooled OOF, a full-inner fit
     scored on In-the-Wild. Cached per probe under outputs/v2/layer_diversity/probes/.
  2. Error overlap: each probe's errors at its own inner-OOF min-cost threshold, on inner OOF and
     on In-the-Wild; pairwise Jaccard of error sets; how many of XLS-R L7's In-the-Wild misses
     each probe catches, and how many new false alarms it brings.
  3. Pairs: equal-weight rank blend of every pair of probes (rank against each probe's own inner
     OOF), scored on inner OOF. In-the-Wild is a readout only.

Rules kept: the outer holdout rows are dropped before anything is fit or scored, so this run
never reads them. In-the-Wild is never fit on and never used to choose; it is reported. Pair
selection is by inner OOF only; with 1,225 pairs the top inner pair is optimistically biased,
so treat it as a lead, not a result.

    uv run python scripts/explore_layer_diversity.py            # stages 1-3
    uv run python scripts/explore_layer_diversity.py --noise    # after stages 1-3
    uv run --with matplotlib python scripts/explore_layer_diversity.py --figure
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from hearsay.metrics import PI_SYNTH, cost_at, eer, min_cost
from hearsay.probe import load_embeddings, make_clf, oof_scores

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "v2" / "layer_diversity"
FOLDS = REPO / "splits" / "nsa_folds_plus_asv19.csv"
BACKBONES = {
    "xlsr": ("wav2vec2-xls-r-300m", ["nsa_train_sample_v3", "asv19_addon_v3"], "itw_stress_v3"),
    "wavlm": ("wavlm-large", ["nsa_train_sample_wl", "asv19_addon_wl"], "itw_stress_wl"),
}
REF = "xlsr_L07"  # the shipped M1b probe's backbone and layer


def load_backbone(key: str):
    model, train_sets, itw_set = BACKBONES[key]
    parts = [load_embeddings(model, n) for n in train_sets]
    X = np.concatenate([p[0] for p in parts])
    m = pd.concat([p[1] for p in parts], ignore_index=True)
    f = pd.read_csv(FOLDS).set_index("path")
    m = m.join(f[["fold"]], on="path")
    assert m.fold.notna().all(), "embedding rows missing from the fold file"
    inner = (m.fold != "holdout").to_numpy()  # the outer holdout is dropped here, unread
    X, m = X[inner], m[inner].reset_index(drop=True)
    Xs, ms = load_embeddings(model, itw_set)
    for mm in (m, ms):  # load_embeddings ignores the flag (CLAUDE.md gotcha), so check here
        assert "flag" not in mm or mm["flag"].isna().all(), "decode_error rows present"
    return X, m, Xs, ms


def fit_probe(X: np.ndarray, y: np.ndarray, folds: np.ndarray, Xs: np.ndarray) -> dict:
    oof = oof_scores(X, y, None, 1.0, folds=folds)
    platt = LogisticRegression(class_weight="balanced").fit(oof[:, None], y)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    itw = make_clf(1.0).fit(X, y).decision_function(Xs)
    return {"oof": a * oof + b, "itw": a * itw + b}


def stage_probes() -> None:
    pdir = OUT / "probes"
    pdir.mkdir(parents=True, exist_ok=True)
    for key in BACKBONES:
        todo = [L for L in range(25) if not (pdir / f"{key}_L{L:02d}.npz").exists()]
        if not todo:
            continue
        X, m, Xs, ms = load_backbone(key)
        y = (m.label == "spoof").to_numpy(int)
        folds = m.fold.astype(int).to_numpy()
        print(f"{key}: inner {X.shape}, ITW {Xs.shape}; fitting layers {todo}", flush=True)
        res = Parallel(n_jobs=5, verbose=5)(
            delayed(fit_probe)(X[:, L].copy(), y, folds, Xs[:, L].copy()) for L in todo
        )
        for L, r in zip(todo, res):
            np.savez(pdir / f"{key}_L{L:02d}.npz", **r)
        (OUT / f"{key}_rows.json").write_text(json.dumps({
            "inner_paths": m.path.tolist(), "inner_y": y.tolist(), "inner_source": m.source.tolist(),
            "inner_generator": m.generator.astype(str).tolist(),
            "itw_paths": ms.path.tolist(), "itw_y": (ms.label == "spoof").astype(int).tolist(),
        }))


def threshold(y: np.ndarray, s: np.ndarray) -> float:
    """The inner-OOF min-cost threshold (score >= thr means synthetic)."""
    cand = np.unique(np.quantile(s, np.linspace(0, 1, 2001)))
    return float(min(cand, key=lambda t: cost_at(y, s, t, PI_SYNTH)))


def rank_vs(ref: np.ndarray, s: np.ndarray) -> np.ndarray:
    srt = np.sort(ref)
    return np.searchsorted(srt, s, side="right") / len(srt)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def stage_analysis() -> dict:
    rows = {k: json.loads((OUT / f"{k}_rows.json").read_text()) for k in BACKBONES}
    assert rows["xlsr"]["inner_paths"] == rows["wavlm"]["inner_paths"], "inner row order differs"
    assert rows["xlsr"]["itw_paths"] == rows["wavlm"]["itw_paths"], "ITW row order differs"
    y = np.array(rows["xlsr"]["inner_y"])
    ys = np.array(rows["xlsr"]["itw_y"])
    probes = {}
    for key in BACKBONES:
        for L in range(25):
            z = np.load(OUT / "probes" / f"{key}_L{L:02d}.npz")
            probes[f"{key}_L{L:02d}"] = (z["oof"], z["itw"])
    names = list(probes)

    per = {}
    err_in, err_itw, miss_itw, fa_itw = {}, {}, {}, {}
    for n, (oof, itw) in probes.items():
        t = threshold(y, oof)
        pred_in, pred_itw = oof >= t, itw >= t
        err_in[n] = pred_in != y.astype(bool)
        err_itw[n] = pred_itw != ys.astype(bool)
        miss_itw[n] = (~pred_itw) & (ys == 1)
        fa_itw[n] = pred_itw & (ys == 0)
        per[n] = {
            "inner_min_dcf": round(min_cost(y, oof), 4), "inner_eer": round(eer(y, oof), 4),
            "itw_min_dcf": round(min_cost(ys, itw), 4), "itw_eer": round(eer(ys, itw), 4),
            "itw_p_fa_at_inner_thr": round(float(fa_itw[n][ys == 0].mean()), 4),
            "itw_p_miss_at_inner_thr": round(float(miss_itw[n][ys == 1].mean()), 4),
        }

    ref_miss = miss_itw[REF]
    catch = {n: {"catches_of_ref_misses": int((ref_miss & ~miss_itw[n]).sum()),
                 "new_false_alarms_vs_ref": int((fa_itw[n] & ~fa_itw[REF]).sum())}
             for n in names}  # fmt: skip

    jac_in = np.array([[jaccard(err_in[a], err_in[b]) for b in names] for a in names])
    jac_itw = np.array([[jaccard(err_itw[a], err_itw[b]) for b in names] for a in names])
    rho_in = spearmanr(np.column_stack([probes[n][0] for n in names])).statistic

    ranks = {n: (rank_vs(o, o), rank_vs(o, i)) for n, (o, i) in probes.items()}
    pairs = []
    for a, b in itertools.combinations(names, 2):
        s_in = (ranks[a][0] + ranks[b][0]) / 2
        pairs.append((a, b, min_cost(y, s_in)))
    pairs.sort(key=lambda p: p[2])
    top = []
    named = [(REF, "wavlm_L09"), (REF, "xlsr_L12"), (REF, "xlsr_L20"), (REF, "wavlm_L18")]
    for a, b, c in pairs[:10] + [(a, b, None) for a, b in named]:
        s_in = (ranks[a][0] + ranks[b][0]) / 2
        s_itw = (ranks[a][1] + ranks[b][1]) / 2
        t = threshold(y, s_in)
        top.append({"pair": f"{a}+{b}", "inner_min_dcf": round(min_cost(y, s_in), 4),
                    "itw_min_dcf": round(min_cost(ys, s_itw), 4),
                    "itw_p_fa_at_inner_thr": round(float((s_itw >= t)[ys == 0].mean()), 4),
                    "itw_p_miss_at_inner_thr": round(float((s_itw < t)[ys == 1].mean()), 4),
                    "rho_inner": round(float(rho_in[names.index(a), names.index(b)]), 3),
                    "selected_by": "inner OOF rank" if c is not None else "named before the run"})

    out = {"per_probe": per, "catch_vs_ref": catch, "pairs": top, "names": names,
           "n_inner": len(y), "n_itw": len(ys), "ref": REF}
    np.savez(OUT / "overlap.npz", jac_in=jac_in, jac_itw=jac_itw, rho_in=rho_in, names=names)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    return out


def _noise_floor_db(path: str) -> float:
    """Relative background level: 10th-percentile 25 ms frame energy minus the 95th, in dB.
    Near -60 = clean gaps between words; near -15 = noise under everything."""
    from hearsay.audio import load_audio

    x = load_audio(REPO / path)
    n = len(x) // 400
    if n < 10:
        return float("nan")
    e = 10 * np.log10(np.mean(x[: n * 400].reshape(n, 400) ** 2, axis=1) + 1e-12)
    return float(np.percentile(e, 10) - np.percentile(e, 95))


def stage_noise() -> dict:
    """Is a probe's In-the-Wild false-alarm rate tied to background noise? Real clips only."""
    rows = json.loads((OUT / "xlsr_rows.json").read_text())
    ys = np.array(rows["itw_y"])
    paths = np.array(rows["itw_paths"])
    f = OUT / "itw_noise.csv"
    if not f.exists():
        floors = Parallel(n_jobs=6)(delayed(_noise_floor_db)(p) for p in paths)
        pd.DataFrame({"path": paths, "noise_floor_db": floors}).to_csv(f, index=False)
    nf = pd.read_csv(f).noise_floor_db.to_numpy()
    y = np.array(rows["inner_y"])
    real = (ys == 0) & np.isfinite(nf)
    edges = np.quantile(nf[real], [0, 1 / 3, 2 / 3, 1])
    tert = np.digitize(nf, edges[1:-1])
    out = {"tertile_edges_db": [round(float(e), 1) for e in edges], "fa_rate_by_tertile": {}}
    for n in (REF, "xlsr_L12", "wavlm_L09", "wavlm_L18"):
        z = np.load(OUT / "probes" / f"{n}.npz")
        t = threshold(y, z["oof"])
        fa = z["itw"] >= t
        out["fa_rate_by_tertile"][n] = [round(float(fa[real & (tert == k)].mean()), 4)
                                        for k in range(3)]  # fmt: skip
    (OUT / "noise.json").write_text(json.dumps(out, indent=1))
    return out


def stage_figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads((OUT / "summary.json").read_text())
    ov = np.load(OUT / "overlap.npz")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for key, label in (("xlsr", "XLS-R"), ("wavlm", "WavLM")):
        Ls = range(25)
        axes[0].plot(Ls, [s["per_probe"][f"{key}_L{L:02d}"]["inner_min_dcf"] for L in Ls],
                     marker="o", ms=3, label=f"{label}, inner OOF")
        axes[0].plot(Ls, [s["per_probe"][f"{key}_L{L:02d}"]["itw_min_dcf"] for L in Ls],
                     marker="s", ms=3, ls="--", label=f"{label}, In-the-Wild")
        axes[1].plot(Ls, [s["per_probe"][f"{key}_L{L:02d}"]["itw_p_fa_at_inner_thr"] * 100
                          for L in Ls], marker="o", ms=3, label=f"{label}: false alarms")
        axes[1].plot(Ls, [s["per_probe"][f"{key}_L{L:02d}"]["itw_p_miss_at_inner_thr"] * 100
                          for L in Ls], marker="s", ms=3, ls="--", label=f"{label}: misses")
    axes[0].set(xlabel="layer", ylabel="minDCF (lower is better)", title="Each layer alone")
    axes[1].set(xlabel="layer", ylabel="% of In-the-Wild clips",
                title="How each layer fails on In-the-Wild (inner threshold)")
    for ax in axes[:2]:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    im = axes[2].imshow(ov["jac_itw"], vmin=0, vmax=1, cmap="viridis")
    axes[2].set_title("Do they fail on the same In-the-Wild files?\n(error overlap, 1 = identical)")
    ticks = [0, 7, 12, 24, 25, 32, 37, 49]
    axes[2].set_xticks(ticks, [str(ov["names"][i]).replace("_L", " L") for i in ticks],
                       rotation=60, fontsize=8)
    axes[2].set_yticks(ticks, [str(ov["names"][i]).replace("_L", " L") for i in ticks], fontsize=8)
    axes[2].axhline(24.5, color="w", lw=1)
    axes[2].axvline(24.5, color="w", lw=1)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "layer_diversity.png", dpi=120)
    print("wrote", OUT / "layer_diversity.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", action="store_true", help="only draw the figure (needs matplotlib)")
    ap.add_argument("--noise", action="store_true", help="only the In-the-Wild noise-floor test")
    args = ap.parse_args()
    if args.noise:
        print(json.dumps(stage_noise(), indent=1))
        return
    if args.figure:
        stage_figure()
        return
    stage_probes()
    out = stage_analysis()
    print(json.dumps({"pairs": out["pairs"]}, indent=1))


if __name__ == "__main__":
    main()
