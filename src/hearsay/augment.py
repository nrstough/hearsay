"""Laundering augmentation for M5 (spec D4), numpy/scipy only so the box's dataloader workers
stay cheap. Every op keeps the clip length and never touches the label.

Class-blind by construction: `Augmenter.__call__` takes only the audio and a per-row rng, so
the augmentation distribution cannot depend on the label or the source. The realized op counts
are tallied per (source, label) by the trainer so a skew would show up in the log.

Ops: additive noise (white + pink, exact SNR), band-limit (zero-phase Butterworth), synthetic
reverb (exponential-decay RIR, direct path kept at lag 0), gain + hard clip (a pure gain is a
no-op after normalization; clipping is the actual crest-factor change), RawBoost's convolutive
linear + nonlinear algorithm (Tak et al. 2022, the one axis the rest doesn't cover), trim jitter
(re-trim at a random 30-40 dB threshold, keep 0-200 ms of edge; training only) and a codec swap,
which the dataset performs by reading the clip's precomputed codec variant instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from hearsay import SR

OPS = ("noise", "band", "reverb", "gain_clip", "rawboost_conv", "codec")
SNR_DB = (5.0, 30.0)
CUTOFFS_HZ = (3400.0, 4000.0, 7000.0)
RT60_S = (0.2, 0.8)
GAIN_DB = (-12.0, 12.0)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate 1/f noise: white noise shaped by a cumulative-sum filter bank."""
    w = rng.standard_normal(n).astype(np.float32)
    # Paul Kellet's economical pink filter
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    p = signal.lfilter(b, a, w).astype(np.float32)
    return p / (np.abs(p).max() + 1e-9)


def add_noise(x: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    n = _pink(x.size, rng) if rng.random() < 0.5 else rng.standard_normal(x.size).astype(np.float32)
    px = float(np.mean(x**2)) + 1e-12
    pn = float(np.mean(n**2)) + 1e-12
    scale = np.sqrt(px / (pn * 10 ** (snr_db / 10)))
    return (x + scale * n).astype(np.float32)


def band_limit(x: np.ndarray, rng: np.random.Generator, cutoff_hz: float) -> np.ndarray:
    sos = signal.butter(8, cutoff_hz, "low", fs=SR, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def reverb(x: np.ndarray, rng: np.random.Generator, rt60_s: float) -> np.ndarray:
    n = int(rt60_s * SR)
    t = np.arange(n) / SR
    h = (np.exp(-6.9 * t / rt60_s) * rng.standard_normal(n)).astype(np.float32)
    h[: SR // 200] = 0.0  # 5 ms gap after the direct path
    drr_db = rng.uniform(0.0, 10.0)  # direct-to-reverberant ratio
    h *= np.sqrt(np.sum(x**2) * 0 + 1.0) / (np.sqrt(np.sum(h**2)) + 1e-9) / 10 ** (drr_db / 20)
    h[0] = 1.0
    y = signal.fftconvolve(x, h, mode="full")[: x.size]
    return (y / (np.abs(y).max() + 1e-9) * (np.abs(x).max() + 1e-9)).astype(np.float32)


def gain_clip(x: np.ndarray, rng: np.random.Generator, gain_db: float) -> np.ndarray:
    return np.clip(x * 10 ** (gain_db / 20), -1.0, 1.0).astype(np.float32)


def rawboost_conv(x: np.ndarray, rng: np.random.Generator, n_bands: int = 5) -> np.ndarray:
    """RawBoost algorithm 1: a random multi-band notch filter (linear convolutive noise) followed
    by a random polynomial nonlinearity (nonlinear convolutive noise)."""
    y = x.astype(np.float64)
    for _ in range(int(rng.integers(1, n_bands + 1))):
        f0 = rng.uniform(80.0, SR / 2 - 400.0)
        q = rng.uniform(1.0, 10.0)
        b, a = signal.iirnotch(f0, q, fs=SR)
        y = signal.lfilter(b, a, y)
    # nonlinearity: y + sum_k g_k * y^k, small gains so the signal stays audible
    for k, g in ((2, rng.uniform(0.0, 0.2)), (3, rng.uniform(0.0, 0.1))):
        y = y + g * np.sign(y) * np.abs(y) ** k
    peak = np.abs(y).max() + 1e-9
    return (y / peak * (np.abs(x).max() + 1e-9)).astype(np.float32)


def trim_jitter(x: np.ndarray, rng: np.random.Generator, frame: int = SR // 50) -> np.ndarray:
    """Training-only: re-trim at a random relative threshold (30-40 dB) and keep a random
    0-200 ms of the silent edges, so the exact 35 dB boundary can't become a cue. The length
    may change here (it runs before cropping), so it is not in OPS."""
    n = x.size // frame
    if n < 2:
        return np.ascontiguousarray(x, dtype=np.float32)
    top_db = float(rng.uniform(30.0, 40.0))
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / rms.max())
    keep = np.where(db > -top_db)[0]
    if keep.size == 0:
        return np.ascontiguousarray(x, dtype=np.float32)
    edge = int(rng.uniform(0.0, 0.2) * SR)
    lo = max(0, keep[0] * frame - edge)
    hi = min(x.size, (keep[-1] + 1) * frame + edge)
    if hi - lo < SR:  # never below 1 s (same floor as trim_silence)
        return np.ascontiguousarray(x, dtype=np.float32)
    return np.ascontiguousarray(x[lo:hi], dtype=np.float32)


@dataclass
class Augmenter:
    """Draws ops and parameters from the per-row rng only; the label is never an input."""

    p_aug: float = 0.65
    p_rawboost: float = 0.25
    band_weight: float = 2.0
    ops: tuple[str, ...] = OPS
    last: list[str] = field(default_factory=list)

    def plan(self, rng: np.random.Generator, has_codec: bool) -> list[tuple[str, float]]:
        """The ops to apply and their parameters, or [] (clean). Deterministic given rng."""
        if rng.random() >= self.p_aug:
            return []
        weights = {"noise": 1.0, "band": self.band_weight, "reverb": 1.0, "gain_clip": 1.0,
                   "codec": 1.0 if has_codec else 0.0}  # fmt: skip
        names = [o for o in self.ops if o in weights]
        w = np.array([weights[o] for o in names])
        w = w / w.sum()
        k = 1 if rng.random() < 0.6 else 2
        chosen = list(rng.choice(names, size=min(k, len(names)), replace=False, p=w))
        if rng.random() < self.p_rawboost:
            chosen.append("rawboost_conv")
        params = []
        for op in chosen:
            if op == "noise":
                params.append((op, float(rng.uniform(*SNR_DB))))
            elif op == "band":
                params.append((op, float(rng.choice(CUTOFFS_HZ))))
            elif op == "reverb":
                params.append((op, float(rng.uniform(*RT60_S))))
            elif op == "gain_clip":
                params.append((op, float(rng.uniform(*GAIN_DB))))
            else:
                params.append((op, 0.0))
        return params

    def apply(self, x: np.ndarray, plan: list[tuple[str, float]],
              rng: np.random.Generator) -> np.ndarray:  # fmt: skip
        """Apply every op but `codec` (the dataset swaps the file for that one)."""
        self.last = [op for op, _ in plan]
        for op, p in plan:
            if op == "noise":
                x = add_noise(x, rng, p)
            elif op == "band":
                x = band_limit(x, rng, p)
            elif op == "reverb":
                x = reverb(x, rng, p)
            elif op == "gain_clip":
                x = gain_clip(x, rng, p)
            elif op == "rawboost_conv":
                x = rawboost_conv(x, rng)
        return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def __call__(self, x: np.ndarray, rng: np.random.Generator, has_codec: bool = False):
        return self.apply(x, self.plan(rng, has_codec), rng)

    @staticmethod
    def clean(x: np.ndarray) -> np.ndarray:
        return x


def holdout_slice(n_rows: int, seed: int = 1234) -> list[list[tuple[str, float]]]:
    """A fixed augmentation plan per holdout row (same for every model), one op each, cycling
    through the op list so every op gets a per-op readout."""
    aug = Augmenter(p_aug=1.0, p_rawboost=0.0)
    plans = []
    ops = [o for o in OPS if o != "codec"] + ["rawboost_conv"]
    for i in range(n_rows):
        rng = np.random.default_rng([seed, i])
        op = ops[i % len(ops)]
        full = aug.plan(rng, has_codec=False)
        p = next((q for q in full if q[0] == op), None)
        if p is None:
            # draw the parameter for the cycled op directly
            p = (op, {"noise": float(rng.uniform(*SNR_DB)), "band": float(rng.choice(CUTOFFS_HZ)),
                      "reverb": float(rng.uniform(*RT60_S)), "gain_clip": float(rng.uniform(*GAIN_DB)),
                      "rawboost_conv": 0.0}[op])  # fmt: skip
        plans.append([p])
    return plans
