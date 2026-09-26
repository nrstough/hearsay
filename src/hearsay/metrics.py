"""Validation metrics for the NSA rule (kickoff slides, Fri Sep 25): judged 60% on MinDCF,
ASVspoof5-style, with false alarms (real called synthetic) penalized by a factor of 4.
Miss = synthetic not identified; label 1 = synthetic.

    DCF = C_FA * P_FA * pi_real + C_MISS * P_miss * pi_synth,   C_FA = 4, C_MISS = 1
    normalized by the best constant decision, min(C_FA * pi_real, C_MISS * pi_synth),
    so 1.0 = no better than a constant answer, 0.0 = perfect.

The slide's formula weights the FA term by P(attack) as transcribed; at the stated ~50/50
balance both readings give the same number. Ask the sponsor which prior they plug in.

min_dcf sweeps the threshold: this is the judged number and depends only on ranking.
act_dcf uses the Bayes threshold on calibrated LLRs (calibration check only).
Selection (layer, bake-off, fusion, CSV) uses normalized minDCF; EER is reported alongside.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_curve

C_FA = 4.0
C_MISS = 1.0


def eer(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def _default_cost(pi_synth: float, c_fa: float, c_miss: float) -> float:
    return min(c_fa * (1 - pi_synth), c_miss * pi_synth)


def cost_at(y, s, thr, pi_synth=0.5, c_fa=C_FA, c_miss=C_MISS) -> float:
    """Normalized cost when s >= thr is called synthetic (y: 1 = synthetic)."""
    y, s = np.asarray(y), np.asarray(s)
    p_fa = float(np.mean(s[y == 0] >= thr))
    p_miss = float(np.mean(s[y == 1] < thr))
    c = c_fa * p_fa * (1 - pi_synth) + c_miss * p_miss * pi_synth
    return c / _default_cost(pi_synth, c_fa, c_miss)


def min_cost(y, s, pi_synth=0.5, c_fa=C_FA, c_miss=C_MISS) -> float:
    """Normalized cost at the best threshold (oracle sweep)."""
    fpr, tpr, _ = roc_curve(y, s)
    c = c_fa * fpr * (1 - pi_synth) + c_miss * (1 - tpr) * pi_synth
    return float(min(c.min(), _default_cost(pi_synth, c_fa, c_miss)) /
                 _default_cost(pi_synth, c_fa, c_miss))  # fmt: skip


def bayes_llr_threshold(pi_synth=0.5, c_fa=C_FA, c_miss=C_MISS) -> float:
    """Call synthetic iff LLR > ln(c_fa/c_miss) - logit(pi_synth)."""
    return math.log(c_fa / c_miss) - math.log(pi_synth / (1 - pi_synth))


def decision_logit(llr, pi_synth=0.5, c_fa=C_FA, c_miss=C_MISS):
    """Shift a prior-neutral LLR so a fixed 50% cut-off makes the Bayes decision:
    logit(q) = LLR + logit(pi_synth) - ln(c_fa/c_miss). q > 0.5 <=> posterior P > 0.8 (at 4:1).
    Monotone, so ranking metrics are unchanged."""
    return np.asarray(llr) - bayes_llr_threshold(pi_synth, c_fa, c_miss)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=np.float64)))


def report(y, llr, pi_synth=0.5) -> dict:
    """EER + normalized min/actual cost for prior-neutral LLR-like scores."""
    y, llr = np.asarray(y), np.asarray(llr)
    return {
        "eer": round(eer(y, llr), 4),
        "min_dcf": round(min_cost(y, llr, pi_synth), 4),
        "act_dcf": round(cost_at(y, llr, bayes_llr_threshold(pi_synth), pi_synth), 4),
        "pi_synth": pi_synth,
    }
