"""Validation metrics for the NSA rule (kickoff slides, Fri Sep 25): judged 60% on MinDCF,
ASVspoof5-style, with false alarms (real called synthetic) penalized by a factor of 4.
Miss = synthetic not identified; label 1 = synthetic.

    DCF = C_FA * P_FA * pi_real + C_MISS * P_miss * pi_synth,   C_FA = 4, C_MISS = 1
    normalized by the best constant decision, min(C_FA * pi_real, C_MISS * pi_synth),
    so 1.0 = no better than a constant answer, 0.0 = perfect.

pi_synth = 0.3 (70/30, said aloud at kickoff; the slide read ~50/50): normalized DCF =
9.33 * P_FA + P_miss. The slide's formula weights the FA term by P(attack) as transcribed, which
would change the weights; ask the sponsor which form and prior they plug in.

min_dcf sweeps the threshold: this is the judged number and depends only on ranking.
act_dcf uses the Bayes threshold on calibrated LLRs (calibration check only).
Selection (layer, bake-off, fusion, CSV) uses normalized minDCF; EER is reported alongside.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_curve

if TYPE_CHECKING:
    import pandas as pd

C_FA = 4.0
C_MISS = 1.0
PI_SYNTH = 0.3  # 70/30 real/synthetic, said aloud at kickoff; slide read ~50/50


def eer(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def _default_cost(pi_synth: float, c_fa: float, c_miss: float) -> float:
    return min(c_fa * (1 - pi_synth), c_miss * pi_synth)


def cost_at(y, s, thr, pi_synth=PI_SYNTH, c_fa=C_FA, c_miss=C_MISS) -> float:
    """Normalized cost when s >= thr is called synthetic (y: 1 = synthetic)."""
    y, s = np.asarray(y), np.asarray(s)
    p_fa = float(np.mean(s[y == 0] >= thr))
    p_miss = float(np.mean(s[y == 1] < thr))
    c = c_fa * p_fa * (1 - pi_synth) + c_miss * p_miss * pi_synth
    return c / _default_cost(pi_synth, c_fa, c_miss)


def min_cost(y, s, pi_synth=PI_SYNTH, c_fa=C_FA, c_miss=C_MISS) -> float:
    """Normalized cost at the best threshold (oracle sweep)."""
    fpr, tpr, _ = roc_curve(y, s)
    c = c_fa * fpr * (1 - pi_synth) + c_miss * (1 - tpr) * pi_synth
    return float(min(c.min(), _default_cost(pi_synth, c_fa, c_miss)) /
                 _default_cost(pi_synth, c_fa, c_miss))  # fmt: skip


def bayes_llr_threshold(pi_synth=PI_SYNTH, c_fa=C_FA, c_miss=C_MISS) -> float:
    """Call synthetic iff LLR > ln(c_fa/c_miss) - logit(pi_synth)."""
    return math.log(c_fa / c_miss) - math.log(pi_synth / (1 - pi_synth))


def decision_logit(llr, pi_synth=PI_SYNTH, c_fa=C_FA, c_miss=C_MISS):
    """Shift a prior-neutral LLR so a fixed 50% cut-off makes the Bayes decision:
    logit(q) = LLR + logit(pi_synth) - ln(c_fa/c_miss). q > 0.5 <=> posterior P > 0.8 (at 4:1).
    Monotone, so ranking metrics are unchanged."""
    return np.asarray(llr) - bayes_llr_threshold(pi_synth, c_fa, c_miss)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=np.float64)))


def sponsor_min_dcf(y, s, flip: bool) -> float:
    """minDCF exactly as the sponsor's shipped code computes it (data/nsa/HackGTMinDCF,
    ASVspoof5 calculate_metrics.py): Pspoof = 0.5, Cmiss = 1, Cfa = 4, and a HIGHER score is
    treated as bona fide. flip=False scores our synthetic-high scores as-is (what happens if they
    run the code unmodified); flip=True scores 1 - s (what happens if they invert first)."""
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    s = -s if flip else s
    bona, spoof = s[y == 0], s[y == 1]
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(bona.size), np.zeros(spoof.size)])[
        np.argsort(scores, kind="mergesort")
    ]
    frr = np.concatenate([[0.0], np.cumsum(labels) / bona.size])
    far = np.concatenate(
        [[1.0], (spoof.size - (np.arange(1, labels.size + 1) - np.cumsum(labels))) / spoof.size]
    )
    p_target = 0.5
    c_det = 1 * frr * p_target + 4 * far * (1 - p_target)
    return float(c_det.min() / min(1 * p_target, 4 * (1 - p_target)))


def report(y, llr, pi_synth=PI_SYNTH) -> dict:
    """EER + normalized min/actual cost for prior-neutral LLR-like scores."""
    y, llr = np.asarray(y), np.asarray(llr)
    return {
        "eer": round(eer(y, llr), 4),
        "min_dcf": round(min_cost(y, llr, pi_synth), 4),
        "act_dcf": round(cost_at(y, llr, bayes_llr_threshold(pi_synth), pi_synth), 4),
        "pi_synth": pi_synth,
        "min_dcf_pi05": round(min_cost(y, llr, 0.5), 4),
        "sponsor_code_asis": round(sponsor_min_dcf(y, llr, flip=False), 4),
        "sponsor_code_flipped": round(sponsor_min_dcf(y, llr, flip=True), 4),
    }


def by_group(dv: pd.DataFrame, yv: np.ndarray, s: np.ndarray, pi: float = PI_SYNTH) -> tuple[dict, dict]:
    """minDCF/EER per generator (its spoofs vs every bona fide clip) and per bona fide source
    (its bona fide clips vs every spoof). Copied here from scripts/train_handcrafted.py for M3;
    the trainer keeps its own copy (it imports LightGBM, which must never load next to torch)."""
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
