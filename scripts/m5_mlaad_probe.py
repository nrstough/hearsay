"""MLAAD corpus-shortcut probe ("Probe A" from the M5 consult), CPU only.

Holds the label constant: can handcrafted features separate MLAAD spoof from DiffSSD spoof
after the same augmentation + band-match the model sees? If yes (AUC > 0.95), where do the
bona fide pools land on that axis? LibriSpeech projecting onto the MLAAD side means an MLAAD
cue points straight at the worst false-alarm source -> cap MLAAD at 12% instead of 28%.

Reads bundle FLACs, writes outputs/m5_probe_a.json. Usage:
  uv run python scripts/m5_mlaad_probe.py --bundle "<bundle>/v1" --n 500 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.augment import Augmenter
from hearsay.handcrafted import features
from hearsay.m5_bundle import read_clip
from hearsay.m5_data import LengthMixer, band_match, crop

REPO = Path(__file__).resolve().parents[1]


def _feat(args):
    path, seed, crop_s = args
    rng = np.random.default_rng(seed)
    x = read_clip(path)
    x = crop(x, crop_s, rng)
    x = Augmenter(p_aug=0.65, p_rawboost=0.25)(x, rng)
    x = band_match(x)
    try:
        return features(x, band_match=False)
    except Exception:  # noqa: BLE001 - one bad clip becomes a missing row
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--n", type=int, default=500, help="clips per pool")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "m5_probe_a.json")
    args = ap.parse_args()
    os.nice(10)
    rng = np.random.default_rng(args.seed)
    m = pd.read_csv(args.bundle / "manifest.csv")
    m = m[m.fold.astype(str) != "holdout"]
    pools = {
        "mlaad_spoof": m[(m.train_scope == "extra_mlaad")],
        "diffssd_spoof": m[(m.source == "diffssd") & (m.label == "spoof")],
        "lj_bona": m[(m.source == "ljspeech") & (m.label == "bonafide")],
        "libri_bona": m[(m.source == "librispeech") & (m.label == "bonafide")],
        "asv19_bona": m[(m.source == "asvspoof2019") & (m.label == "bonafide")],
    }
    mixer = LengthMixer(args.seed, p_test=0.7)
    jobs, tags = [], []
    for name, d in pools.items():
        take = d.sample(min(args.n, len(d)), random_state=args.seed)
        for i, r in enumerate(take.itertuples()):
            p = args.bundle / ("core" if r.train_scope == "core" else "extra") / r.file
            jobs.append((str(p), int(rng.integers(1 << 30)), mixer.draw()))
            tags.append(name)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        feats = list(ex.map(_feat, jobs, chunksize=8))
    ok = [i for i, f in enumerate(feats) if f is not None]
    X = pd.DataFrame([feats[i] for i in ok]).to_numpy(np.float32)
    X = np.nan_to_num(X)
    tag = np.array([tags[i] for i in ok])
    print(f"features for {len(ok)}/{len(jobs)} clips in {time.time() - t0:.0f}s")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    sp = np.isin(tag, ["mlaad_spoof", "diffssd_spoof"])
    y = (tag[sp] == "mlaad_spoof").astype(int)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)
    oof = cross_val_predict(clf, X[sp], y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, oof))
    clf.fit(X[sp], y)
    proj = {}
    for name in ("lj_bona", "libri_bona", "asv19_bona", "diffssd_spoof", "mlaad_spoof"):
        mk = tag == name
        p = clf.predict_proba(X[mk])[:, 1]
        proj[name] = {"n": int(mk.sum()), "mean_p_mlaad": round(float(p.mean()), 4),
                      "frac_mlaad_side": round(float((p > 0.5).mean()), 4)}
    real_on_mlaad_side = max(proj["libri_bona"]["frac_mlaad_side"],
                             proj["asv19_bona"]["frac_mlaad_side"]) > 0.5
    verdict = ("no exploitable fingerprint -> keep 28%" if auc < 0.85 else
               "fingerprint and real speech (LibriSpeech/VCTK) lands on the MLAAD side -> "
               "cap MLAAD at 12% and watch ITW P_FA" if real_on_mlaad_side else
               "fingerprint, bona fide pools on the DiffSSD side -> keep 28%")
    out = {"auc_mlaad_vs_diffssd_cv": round(auc, 4), "projection": proj, "verdict": verdict,
           "n_per_pool": args.n, "seconds": round(time.time() - t0)}
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
