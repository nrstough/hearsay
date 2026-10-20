"""Layer bake-off (pre-declared in docs/specs/2026-10-03_layer-bakeoff.md): does XLS-R L7 + L20 hold?

Stages (extract and probes are independent and run in parallel):
    uv run python scripts/bakeoff_layers.py extract   # GPU: ITW perturbation cohort + MLAAD embeddings
    uv run python scripts/bakeoff_layers.py probes    # CPU: 25 per-layer probes + the L7||L20 concat probe
    uv run python scripts/bakeoff_layers.py report    # readouts R1-R7, fusion F0 vs F(Pk), bootstrap

Writes only under outputs/v2/bakeoff/. Never writes models/, submissions/ or splits/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from hearsay.metrics import cost_at, min_cost
from hearsay.probe import load_embeddings, make_clf, oof_scores

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "v2" / "bakeoff"
MODEL = "wav2vec2-xls-r-300m"
TRAIN_SETS = ["nsa_train_sample_v3", "asv19_addon_v3"]
FOLDS = REPO / "splits" / "nsa_folds_plus_asv19.csv"
KINDS = ("none", "mp3", "noise20", "speed", "shift1")
ITW_CROP_SEED = 200  # itw_stress_v3/extract_meta.json: crop "test", seed 200
COHORT_PER_CLASS = 250

CANDIDATES = {
    "P0": ("single", [7]),
    "P1": ("pair", [7, 20]),
    "P2": ("pair", [7, 21]),
    "P3": ("pair", [7, 18]),
    "P4": ("pair", [6, 20]),
    "P5": ("pair", [8, 20]),
    "P6": ("concat", [7, 20]),
    "P7": ("stack", list(range(25))),
}
RULES = {"R1_pooled": 0.010, "R1_fold": 0.030, "R2": 0.020, "R3_gain": 0.030, "R4": 0.030,
         "R5": 0.030, "R6": 0.05, "R7": 0.010}  # fmt: skip
BOOT_N = 1000


def _mod(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def averse(y, s) -> float:
    return min_cost(y, s, 0.5, c_fa=1.0, c_miss=4.0)


def inner_threshold(y, s) -> float:
    cand = np.unique(np.quantile(s, np.linspace(0, 1, 4001)))
    return float(min(cand, key=lambda t: cost_at(y, s, t, 0.3)))


def rank_vs(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(ref), x, side="right") / len(ref)


# ---------------------------------------------------------------- stage: extract (GPU)
def cohort_rows() -> pd.DataFrame:
    m = pd.read_csv(REPO / "outputs/embeddings" / MODEL / "itw_stress_v3" / "manifest.csv")
    rng = np.random.default_rng(0)
    idx = np.r_[rng.choice(np.flatnonzero(m.label == "bonafide"), COHORT_PER_CLASS, replace=False),
                rng.choice(np.flatnonzero(m.label == "spoof"), COHORT_PER_CLASS, replace=False)]
    return pd.DataFrame({"row": np.sort(idx)})


def stage_extract() -> None:
    from hearsay.audio import load_audio
    from hearsay.embed import embed_segment, load_backbone, prepare_segment, test_duration_sampler

    mp = _mod("m3_probes")
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(REPO / "outputs/embeddings" / MODEL / "itw_stress_v3" / "manifest.csv")
    draw = test_duration_sampler(ITW_CROP_SEED)
    crop = [draw() for _ in range(len(m))]  # the same draw sequence as the stored ITW embeddings
    rows = cohort_rows().row.to_numpy()
    model = load_backbone(MODEL)
    t0 = time.time()
    for kind in KINDS:
        f = OUT / f"emb_itw_{kind}.npz"
        if f.exists():
            continue
        embs = []
        for i, r in enumerate(rows):
            x = mp.perturb(load_audio(REPO / m.path.iloc[r]), kind, int(r))
            embs.append(embed_segment(model, prepare_segment(x, crop[r], ITW_CROP_SEED + int(r))))
            if i % 100 == 0:
                print(f"  {kind} {i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        np.savez(f, emb=np.stack(embs).astype(np.float32), row=rows)
    f = OUT / "emb_mlaad.npz"
    if not f.exists():
        ml = pd.read_csv(REPO / "outputs/channel/m3_mlaad.csv")
        embs = []
        for i, p in enumerate(ml.path):
            embs.append(embed_segment(model, prepare_segment(load_audio(p))))
            if i % 100 == 0:
                print(f"  mlaad {i}/{len(ml)}  {time.time() - t0:.0f}s", flush=True)
        np.savez(f, emb=np.stack(embs).astype(np.float32), path=ml.path.to_numpy())
    print(f"extract done in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- stage: probes (CPU)
def load_all():
    parts = [load_embeddings(MODEL, n) for n in TRAIN_SETS]
    X = np.concatenate([p[0] for p in parts])
    m = pd.concat([p[1] for p in parts], ignore_index=True)
    m = m.join(pd.read_csv(FOLDS).set_index("path")[["fold"]], on="path")
    assert m.fold.notna().all()
    Xt, mt = load_embeddings(MODEL, "nsa_test_v3")
    Xs, ms = load_embeddings(MODEL, "itw_stress_v3")
    for mm in (m, mt, ms):
        assert "flag" not in mm or mm["flag"].isna().all(), "decode_error rows present"
    return X, m, Xt, mt, Xs, ms


def fit_one(Xin, y, folds, others: dict, name: str) -> dict:
    oof = oof_scores(Xin, y, None, 1.0, folds=folds)
    platt = LogisticRegression(class_weight="balanced").fit(oof[:, None], y)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    clf = make_clf(1.0).fit(Xin, y)
    joblib.dump({"clf": clf, "a": a, "b": b}, OUT / "probes" / f"{name}.joblib")
    return {"oof": a * oof + b, **{k: a * clf.decision_function(v) + b for k, v in others.items()}}


def stage_probes() -> None:
    (OUT / "probes").mkdir(parents=True, exist_ok=True)
    X, m, Xt, mt, Xs, ms = load_all()
    inner, hold = (m.fold != "holdout").to_numpy(), (m.fold == "holdout").to_numpy()
    y = (m.label[inner] == "spoof").to_numpy(int)
    folds = m.fold[inner].astype(int).to_numpy()
    jobs = {f"L{L:02d}": [L] for L in range(25)}
    jobs["L07+L20cat"] = [7, 20]
    todo = [k for k in jobs if not (OUT / "probes" / f"{k}.npz").exists()]
    print(f"inner {inner.sum()}, holdout {hold.sum()}, test {len(mt)}, itw {len(ms)}; fitting {todo}", flush=True)

    Xin, Xh = X[inner], X[hold]
    del X

    def sl(A, k):
        return A[:, jobs[k]].reshape(len(A), -1)

    res = Parallel(n_jobs=6, verbose=5, pre_dispatch="n_jobs")(
        delayed(fit_one)(sl(Xin, k), y, folds, {"holdout": sl(Xh, k), "test": sl(Xt, k), "itw": sl(Xs, k)}, k)
        for k in todo
    )
    for k, r in zip(todo, res):
        np.savez(OUT / "probes" / f"{k}.npz", **r)
    meta = {
        "inner_path": m.path[inner].tolist(), "inner_y": y.tolist(), "inner_fold": folds.tolist(),
        "holdout_path": m.path[hold].tolist(), "holdout_y": (m.label[hold] == "spoof").astype(int).tolist(),
        "holdout_generator": m.generator[hold].astype(str).tolist(), "holdout_source": m.source[hold].tolist(),
        "test_name": mt.filename.tolist(), "itw_path": ms.path.tolist(),
        "itw_y": (ms.label == "spoof").astype(int).tolist(), "itw_speaker": ms.speaker.tolist(),
    }  # fmt: skip
    (OUT / "rows.json").write_text(json.dumps(meta))
    print("probes done")


# ---------------------------------------------------------------- stage: report
def score_extra(key: str, emb: np.ndarray) -> np.ndarray:
    """Score extracted (N, 25, 1024) embeddings with a saved full-inner probe."""
    p = joblib.load(OUT / "probes" / f"{key}.joblib")
    L = [7, 20] if key == "L07+L20cat" else [int(key[1:])]
    return p["a"] * p["clf"].decision_function(emb[:, L].reshape(len(emb), -1)) + p["b"]


def build_scores(meta: dict) -> dict:
    """{candidate: {pop: score array}} for pops inner, holdout, test, itw, mlaad, itw_<kind>."""
    layers = {f"L{L:02d}": dict(np.load(OUT / "probes" / f"L{L:02d}.npz")) for L in range(25)}
    layers["L07+L20cat"] = dict(np.load(OUT / "probes" / "L07+L20cat.npz"))
    extra = {f"itw_{k}": np.load(OUT / f"emb_itw_{k}.npz")["emb"] for k in KINDS}
    extra["mlaad"] = np.load(OUT / "emb_mlaad.npz")["emb"]
    for key, d in layers.items():
        for pop, emb in extra.items():
            d[pop] = score_extra(key, emb)
    pops = ["holdout", "test", "itw", "mlaad", *[f"itw_{k}" for k in KINDS]]
    y = np.array(meta["inner_y"])
    folds = np.array(meta["inner_fold"])
    out = {}
    for c, (kind, Ls) in CANDIDATES.items():
        if kind == "single":
            d = layers[f"L{Ls[0]:02d}"]
            out[c] = {"inner": d["oof"], **{p: d[p] for p in pops}}
        elif kind == "concat":
            d = layers["L07+L20cat"]
            out[c] = {"inner": d["oof"], **{p: d[p] for p in pops}}
        elif kind == "pair":
            a, b = (layers[f"L{L:02d}"] for L in Ls)
            out[c] = {"inner": (rank_vs(a["oof"], a["oof"]) + rank_vs(b["oof"], b["oof"])) / 2,
                      **{p: (rank_vs(a["oof"], a[p]) + rank_vs(b["oof"], b[p])) / 2 for p in pops}}
        else:  # stack: nested by inner fold on the 25 per-layer OOF logits
            Z = np.column_stack([layers[f"L{L:02d}"]["oof"] for L in Ls])
            st = np.zeros(len(y))
            for f in np.unique(folds):
                lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Z[folds != f], y[folds != f])
                st[folds == f] = lr.decision_function(Z[folds == f])
            full = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Z, y)
            out[c] = {"inner": st, **{p: full.decision_function(
                np.column_stack([layers[f"L{L:02d}"][p] for L in Ls])) for p in pops}}
    return out


def boot_gain(y, spk, s0, s1, cost=min_cost, n=BOOT_N, seed=0) -> dict:
    """Speaker-cluster bootstrap of cost(s0) - cost(s1) (positive = s1 better)."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(y))).groupby(np.asarray(spk)).apply(np.array).tolist()
    g = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        g.append(cost(y[idx], s0[idx]) - cost(y[idx], s1[idx]))
    g = np.array(g)
    return {"p5": round(float(np.percentile(g, 5)), 4), "median": round(float(np.median(g)), 4),
            "p95": round(float(np.percentile(g, 95)), 4)}  # fmt: skip


def probe_readouts(meta: dict, sc: dict) -> dict:
    y, folds = np.array(meta["inner_y"]), np.array(meta["inner_fold"])
    yh, ys = np.array(meta["holdout_y"]), np.array(meta["itw_y"])
    gen, src = np.array(meta["holdout_generator"]), np.array(meta["holdout_source"])
    spk = np.array(meta["itw_speaker"])
    rows = np.load(OUT / "emb_itw_none.npz")["row"]
    yc = ys[rows]
    durs = np.array([_dur(REPO / p) for p in meta["itw_path"]])
    r = {}
    for c, d in sc.items():
        t = inner_threshold(y, d["inner"])
        hb = d["holdout"]
        per_gen = {g: round(min_cost(np.r_[np.zeros((yh == 0).sum()), np.ones((gen == g).sum())],
                                     np.r_[hb[yh == 0], hb[gen == g]]), 4)
                   for g in sorted(set(gen[yh == 1]))}  # fmt: skip
        per_src = {s: round(min_cost(np.r_[np.zeros(((src == s) & (yh == 0)).sum()), np.ones((yh == 1).sum())],
                                     np.r_[hb[(src == s) & (yh == 0)], hb[yh == 1]]), 4)
                   for s in sorted(set(src[yh == 0]))}  # fmt: skip
        r[c] = {
            "inner": round(min_cost(y, d["inner"]), 4),
            "inner_by_fold": {int(f): round(min_cost(y[folds == f], d["inner"][folds == f]), 4)
                              for f in np.unique(folds)},
            "holdout": round(min_cost(yh, hb), 4), "holdout_by_generator": per_gen,
            "holdout_by_real_source": per_src,
            "itw": round(min_cost(ys, d["itw"]), 4), "itw_averse": round(averse(ys, d["itw"]), 4),
            "itw_p_fa": round(float((d["itw"] >= t)[ys == 0].mean()), 4),
            "itw_p_miss": round(float((d["itw"] < t)[ys == 1].mean()), 4),
            "itw_short_le4s": round(min_cost(ys[durs <= 4], d["itw"][durs <= 4]), 4),
            "itw_long_gt4s": round(min_cost(ys[durs > 4], d["itw"][durs > 4]), 4),
            "cohort": {k: round(min_cost(yc, d[f"itw_{k}"]), 4) for k in KINDS},
            "mlaad_miss": round(float((d["mlaad"] < t).mean()), 4),
            "test_share": round(float((d["test"] >= t).mean()), 4),
            "test_spearman_vs_P0": round(float(spearmanr(d["test"], sc["P0"]["test"]).statistic), 4),
        }
        if c != "P0":
            r[c]["itw_gain_boot"] = boot_gain(ys, spk, sc["P0"]["itw"], d["itw"])
            r[c]["test_flips_vs_P0"] = int(((d["test"] >= t) != (sc["P0"]["test"] >= inner_threshold(
                y, sc["P0"]["inner"]))).sum())
    # sanity: the "none" cohort must reproduce the stored ITW embeddings' scores
    r["_sanity_none_vs_stored_max_abs"] = round(float(np.max(np.abs(
        sc["P0"]["itw_none"] - sc["P0"]["itw"][rows]))), 6)
    return r


def _dur(p: Path) -> float:
    import soundfile as sf

    return sf.info(str(p)).duration


def verdicts(r: dict) -> dict:
    b, v = r["P0"], {}
    for c in CANDIDATES:
        if c == "P0":
            continue
        x = r[c]
        checks = {
            "R1": x["inner"] <= b["inner"] + RULES["R1_pooled"]
            and all(x["inner_by_fold"][f] <= b["inner_by_fold"][f] + RULES["R1_fold"] for f in b["inner_by_fold"]),
            "R2": x["holdout"] <= b["holdout"] + RULES["R2"],
            "R3": (b["itw"] - x["itw"]) >= RULES["R3_gain"] and x["itw_gain_boot"]["p5"] > 0,
            "R4": all(x["cohort"][k] <= b["cohort"][k] + RULES["R4"] for k in KINDS),
            "R5": x["mlaad_miss"] <= b["mlaad_miss"] + RULES["R5"],
            "R6": abs(x["test_share"] - b["test_share"]) <= RULES["R6"],
            "R7": x["itw_averse"] <= b["itw_averse"] + RULES["R7"],
        }
        v[c] = {"checks": checks, "confirms": all(checks.values())}
    nb = sum(v[c]["confirms"] for c in ("P2", "P3", "P4", "P5"))
    v["idea"] = ("idea holds" if v["P1"]["confirms"] and nb >= 2 else
                 "layer-specific: treat as luck" if v["P1"]["confirms"] else "P1 does not confirm")
    v["neighbours_confirming"] = nb
    return v


def fusion_readouts(meta: dict, sc: dict) -> dict:
    fs = _mod("fuse_sweep")
    S = REPO / "outputs/detector_scores"
    cols = {"m1b_v3": fs.load("m1b_v3"),
            "handcrafted_v5": fs.load("handcrafted_v5", str(S / "_itw_handcrafted_v5.csv")),
            "m5_xlsr_ft": fs.load("m5_xlsr_ft", str(S / "_itw_m5.csv")),
            "m3": fs.load("spectra_aasist", str(REPO / "outputs/spectra/itw_stress/scores.csv"))}
    ref = cols["m1b_v3"]
    labels = pd.read_csv(REPO / "splits/nsa_folds.csv")
    labels["path"] = labels.path.map(fs.norm)
    lab = labels.set_index("path").label
    itwm = pd.read_csv(REPO / "outputs/manifests/itw_stress.csv")
    itwm["path"] = itwm.path.map(fs.norm)
    idx = {"inner": ref.index[ref.split == "inner_oof"], "holdout": ref.index[ref.split == "holdout"],
           "itw": pd.Index(itwm.path)}  # fmt: skip
    yv = {"inner": (lab.loc[idx["inner"]] == "spoof").to_numpy(int),
          "holdout": (lab.loc[idx["holdout"]] == "spoof").to_numpy(int),
          "itw": (itwm.label == "spoof").to_numpy(int)}  # fmt: skip
    spk = itwm.speaker.to_numpy()
    # candidate scores keyed by path, per population
    ip = [fs.norm(p) for p in meta["inner_path"]]
    hp = [fs.norm(p) for p in meta["holdout_path"]]
    sp = [fs.norm(p) for p in meta["itw_path"]]

    def cand_series(d):
        return {"inner": pd.Series(d["inner"], index=ip), "holdout": pd.Series(d["holdout"], index=hp),
                "itw": pd.Series(d["itw"], index=sp)}

    def fused(m1_by_pop: dict) -> dict:
        ref_inner = m1_by_pop["inner"].loc[idx["inner"]].to_numpy()
        out = {}
        for pop in ("inner", "holdout", "itw"):
            ii = idx[pop]
            base = (0.6 * rank_vs(ref_inner, m1_by_pop[pop].loc[ii].to_numpy())
                    + 0.2 * rank_vs(cols["handcrafted_v5"].logit.loc[idx["inner"]].to_numpy(),
                                    cols["handcrafted_v5"].logit.loc[ii].to_numpy())
                    + 0.2 * rank_vs(cols["m5_xlsr_ft"].logit.loc[idx["inner"]].to_numpy(),
                                    cols["m5_xlsr_ft"].logit.loc[ii].to_numpy()))
            m3 = cols["m3"].logit.loc[ii].to_numpy()
            out[pop] = np.where((m3 < -3.0) & (base > 0.5), base * 0.5, base)
        return out

    shipped = fused({p: cols["m1b_v3"].logit for p in ("inner", "holdout", "itw")})
    res = {"F0_shipped_from_exports": {p: round(min_cost(yv[p], shipped[p]), 4) for p in shipped}}
    fz = {}
    for c, d in sc.items():
        fz[c] = fused(cand_series(d))
        res[f"F({c})"] = {**{p: round(min_cost(yv[p], fz[c][p]), 4) for p in fz[c]},
                          **{f"{p}_averse": round(averse(yv[p], fz[c][p]), 4) for p in fz[c]}}
    for c in sc:
        if c != "P0":
            res[f"F({c})"]["itw_gain_boot_vs_F(P0)"] = boot_gain(yv["itw"], spk, fz["P0"]["itw"], fz[c]["itw"])
    return res


def stage_report() -> None:
    meta = json.loads((OUT / "rows.json").read_text())
    sc = build_scores(meta)
    pr = probe_readouts(meta, sc)
    v = verdicts(pr)
    fu = fusion_readouts(meta, sc)
    rep = {"spec": "docs/specs/2026-10-03_layer-bakeoff.md", "rules": RULES, "probe": pr, "verdicts": v, "fusion": fu}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=bool))
    print(json.dumps({"verdicts": v, "fusion": fu}, indent=1, default=bool))


if __name__ == "__main__":
    {"extract": stage_extract, "probes": stage_probes, "report": stage_report}[sys.argv[1]]()
