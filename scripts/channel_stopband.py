"""Does the double low-pass stopband move any detector? (run spec
docs/specs/2026-09-26_channel-stopband.md)

NSA test files arrive already low-passed at ~7.2 kHz and every detector applies
hearsay.handcrafted.band_limit once more; training clips pass it once. Scoring band_limit(x)
instead of x for the same clip approximates the test condition: it matches the floor depth,
not the test wall's transition-band shape (see the run spec's Review). Same 500 outer-holdout clips,
crops and scoring path as action E (scripts/m3_probes.py), fused with fusion_v2.

Usage:
  uv run python scripts/channel_stopband.py [--threads 8] [--readout-only]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "channel"
KINDS = ("once", "twice")
FUSED = ("fused_base", "fused", "p_fused")
IQR_FRACTION = 0.10
MIN_DCF_MOVE = 0.037  # one holdout false alarm (9.33 / 250 real clips)
N_EXPECTED = 500


def _m3p():
    spec = importlib.util.spec_from_file_location("m3_probes", REPO / "scripts" / "m3_probes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m3p = _m3p()
DETECTORS = m3p.MODELS
COLS = (*DETECTORS, *FUSED, "e_applied")


def stopband(x: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """once: the segment unchanged; twice: one extra band_limit pass (the test condition)."""
    from hearsay.handcrafted import band_limit

    x = np.asarray(x, dtype=np.float32)
    if kind == "once":
        return x.copy()
    if kind == "twice":
        return band_limit(x)
    raise ValueError(f"unknown condition {kind!r}")


def score_with_p(models, consts, x: np.ndarray) -> dict:
    """E's score_all plus the fused probability from the same constants."""
    out = _SCORE_ALL(models, consts, x)
    fo = consts.fuse({d: out[d] for d in DETECTORS})
    out["p_fused"] = float(fo.p)
    return out


_SCORE_ALL = m3p.score_all


def paired(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """once / twice frames indexed by path, restricted to clips with a finite value in every
    column under both conditions; y = 1 for spoof."""
    d = m3p.with_score_columns(d)
    for c in ("p_fused",):
        if c not in d:
            d[c] = np.nan
    d = d[d.error.isna()] if "error" in d else d
    piv = {k: d[d.kind == k].drop_duplicates("path", keep="last").set_index("path") for k in KINDS}
    idx = piv["once"].index.intersection(piv["twice"].index)
    ok = [p for p in idx if all(np.isfinite(pd.to_numeric(piv[k].loc[p, list(COLS)], errors="coerce")
                                           .to_numpy(float)).all() for k in KINDS)]  # fmt: skip
    a, b = piv["once"].loc[ok], piv["twice"].loc[ok]
    return a, b, (a.label == "spoof").to_numpy(int)


def shift_stats(once: np.ndarray, twice: np.ndarray, y: np.ndarray) -> dict:
    d = twice.astype(float) - once.astype(float)
    return {"mean": round(float(d.mean()), 5), "median": round(float(np.median(d)), 5),
            "p95_abs": round(float(np.percentile(np.abs(d), 95)), 5),
            "max_abs": round(float(np.abs(d).max()), 5),
            "mean_real": round(float(d[y == 0].mean()), 5) if (y == 0).any() else None,
            "mean_spoof": round(float(d[y == 1].mean()), 5) if (y == 1).any() else None}  # fmt: skip


def flips(once: np.ndarray, twice: np.ndarray, thr: float) -> int:
    return int(np.sum((once >= thr) != (twice >= thr)))


def responds(p95_abs: float, iqr: float, fraction: float = IQR_FRACTION) -> str:
    """"responds" / "no response" / "inconclusive" for one detector (pre-declared rule)."""
    ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (p95_abs, iqr))
    if not ok or iqr <= 0:
        return "inconclusive"
    return "responds" if p95_abs > fraction * iqr else "no response"


def fused_responds(dcf_once: float, dcf_twice: float, move: float = MIN_DCF_MOVE) -> str:
    ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (dcf_once, dcf_twice))
    if not ok:
        return "inconclusive"
    return "responds" if abs(dcf_twice - dcf_once) > move else "no response"


def inner_iqr(det: str) -> float:
    d = pd.read_csv(REPO / "outputs" / "detector_scores" / f"{m3p.EXPORT[det]}.csv")
    s = d[d.split == "inner_oof"].logit.to_numpy(float)
    return float(np.percentile(s, 75) - np.percentile(s, 25))


def readout(d: pd.DataFrame, iqr_of=inner_iqr) -> dict:
    from hearsay.metrics import min_cost

    a, b, y = paired(d)
    res: dict = {"n_paired": len(a), "n_real": int((y == 0).sum()), "n_spoof": int((y == 1).sum())}
    two = len(np.unique(y)) == 2
    verdicts = {}
    for c in (*DETECTORS, *FUSED):
        o, t = a[c].to_numpy(float), b[c].to_numpy(float)
        r = shift_stats(o, t, y) if len(o) else {}
        r["min_dcf_once"] = round(min_cost(y, o), 4) if two else float("nan")
        r["min_dcf_twice"] = round(min_cost(y, t), 4) if two else float("nan")
        if c in DETECTORS:
            iqr = iqr_of(c)
            r["inner_oof_logit_iqr"] = round(iqr, 4)
            r["threshold_p95_abs"] = round(IQR_FRACTION * iqr, 4)
            verdicts[c] = responds(r.get("p95_abs", float("nan")), iqr)
        res[c] = r
    verdicts["fused_rule"] = fused_responds(res["p_fused"]["min_dcf_once"], res["p_fused"]["min_dcf_twice"])
    po, pt = a.p_fused.to_numpy(float), b.p_fused.to_numpy(float)
    res["flips_at_p_0.5"] = flips(po, pt, 0.5)
    if two:
        from hearsay.metrics import cost_at

        grid = np.unique(po)
        thr = float(min(grid, key=lambda t: cost_at(y, po, t)))
        res["once_argmin_threshold"] = round(thr, 6)
        res["flips_at_once_argmin"] = flips(po, pt, thr)
    res["e_applied_rate"] = {k: round(float(f.e_applied.mean()), 4) if len(f) else None
                             for k, f in (("once", a), ("twice", b))}  # fmt: skip
    if len(a) < N_EXPECTED:
        verdicts = dict.fromkeys(verdicts, "inconclusive")
        res["short_cohort"] = f"{len(a)} of {N_EXPECTED} paired clips"
    res["verdicts"] = verdicts
    res["ledger"] = ("inconclusive" if "inconclusive" in verdicts.values() else
                     "responds: " + ", ".join(k for k, v in verdicts.items() if v == "responds")
                     if "responds" in verdicts.values() else "measured, no response")  # fmt: skip
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--readout-only", action="store_true")
    a = ap.parse_args()
    rows = m3p.holdout_rows(N_EXPECTED)
    if a.readout_only:
        d = m3p.select_requested(pd.read_csv(OUT / "m3_stopband.csv"), rows, KINDS)
    else:
        m3p.perturb = stopband  # run_scoring resolves these module globals at call time
        m3p.score_all = score_with_p
        d = m3p.run_scoring(rows, KINDS, "stopband", a.threads)
    res = m3p._lambda_mod().json_safe(readout(d))
    (OUT / "stopband.json").write_text(json.dumps(res, indent=2, allow_nan=False))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
