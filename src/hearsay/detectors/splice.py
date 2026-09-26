"""Splice / discontinuity detector (brief rubric, technique 8). CPU only, rule-based.

Looks for editing seams inside a clip:
1. Clicks: one sample-to-sample step far larger than the surrounding waveform's steps (a hard
   cut or an inserted segment that lands mid-cycle). Measured against the RMS of the steps in
   the surrounding 20 ms, the sample itself excluded, so a plosive burst (many large steps)
   does not count.
2. DC-offset jumps: the mean of adjacent 100 ms windows shifts by more than 2% of full scale
   and by more than 6x the clip's typical window-to-window drift (two sources with different
   DC bias joined together; the adaptive part keeps low-frequency-rich material from
   counting).
3. Background-floor variability: the range of a running-minimum level tracker, reported as a
   feature only. In a 1 h box it could not be separated from ordinary pauses (a pause exposes
   the room floor, a speech-only stretch does not), so it does not move the score.

The sponsor states no clip is partially synthetic, so a seam is evidence of editing (the
brief's "scene manipulation" and "insert or replace individual words" cases), not of
synthesis by itself. The score is mild: 0.6 when a click or DC jump is found, else 0.5.
Never learned; the training corpora contain no edited clips to validate on.
"""

from __future__ import annotations

import numpy as np

from hearsay import SR
from hearsay.detectors.base import REGISTRY, ClipContext, DetectorResult, register

NAME = "splice"
CLICK_WIN, CLICK_RATIO, CLICK_ABS, CLICK_MERGE = SR // 50, 15.0, 0.15, SR // 200
DC_WIN, DC_JUMP, DC_RATIO = SR // 10, 0.02, 6.0
FRAME, FLOOR_WIN_FRAMES = SR // 50, 20  # 20 ms frames, 0.4 s running minimum
SCORE_SEAM, SCORE_NONE = 0.6, 0.5


def _moving_sum(v: np.ndarray, win: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([[0.0], v]))
    out = np.empty_like(v)
    half = win // 2
    lo = np.clip(np.arange(v.size) - half, 0, v.size)
    hi = np.clip(np.arange(v.size) + win - half, 0, v.size)
    out[:] = c[hi] - c[lo]
    return out, (hi - lo)


def analyze(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.abs(x).max() + 1e-8)
    if x.size < 2 * CLICK_WIN:
        x = np.pad(x, (0, 2 * CLICK_WIN - x.size))
    d = np.abs(np.diff(x))
    s2, n = _moving_sum(d**2, CLICK_WIN)
    local = np.sqrt(np.maximum(s2 - d**2, 0.0) / np.maximum(n - 1, 1))
    ratio = d / (local + 1e-6)
    hits = np.where((ratio > CLICK_RATIO) & (d > CLICK_ABS))[0]
    clicks = [h for i, h in enumerate(hits) if i == 0 or h - hits[i - 1] > CLICK_MERGE]
    nseg = x.size // DC_WIN
    dc = x[: nseg * DC_WIN].reshape(nseg, DC_WIN).mean(axis=1) if nseg >= 2 else np.zeros(2)
    dj = np.abs(np.diff(dc))
    dc_thr = max(DC_JUMP, DC_RATIO * float(np.median(dj))) if dj.size else DC_JUMP
    dc_jumps = np.where(dj > dc_thr)[0]
    nf = x.size // FRAME
    rms_db = 20 * np.log10(
        np.sqrt(np.mean(x[: nf * FRAME].reshape(nf, FRAME) ** 2, axis=1)) + 1e-6
    )
    if nf >= FLOOR_WIN_FRAMES:
        floor = np.array([rms_db[i : i + FLOOR_WIN_FRAMES].min()
                          for i in range(nf - FLOOR_WIN_FRAMES + 1)])  # fmt: skip
        floor_range = float(floor.max() - floor.min())
    else:
        floor_range = 0.0
    first = min([c / SR for c in clicks] + [(j + 1) * DC_WIN / SR for j in dc_jumps], default=-1.0)
    return {
        "n_clicks": float(len(clicks)),
        "max_click_ratio": float(ratio[hits].max()) if hits.size else 0.0,
        "n_dc_jumps": float(dc_jumps.size),
        "max_dc_jump": float(dj.max()) if dj.size else 0.0,
        "floor_range_db": floor_range,
        "n_seams": float(len(clicks) + dc_jumps.size),
        "first_seam_s": float(first),
    }


class SpliceDetector:
    """Clicks and DC jumps -> mild score + evidence; floor variability reported only."""

    name = NAME

    def applies(self, ctx: ClipContext) -> bool:
        return True

    def run(self, ctx: ClipContext) -> DetectorResult:
        m = ctx.memo("splice.analysis", lambda: analyze(ctx.audio))
        parts = []
        if m["n_clicks"]:
            parts.append(f"{m['n_clicks']:.0f} click(s) (largest step {m['max_click_ratio']:.0f}x "
                         f"the surrounding waveform)")  # fmt: skip
        if m["n_dc_jumps"]:
            parts.append(f"{m['n_dc_jumps']:.0f} DC-offset jump(s) (largest "
                         f"{100 * m['max_dc_jump']:.1f}% of full scale)")  # fmt: skip
        if parts:
            score = SCORE_SEAM
            why = (f"editing seam at {m['first_seam_s']:.2f} s: " + "; ".join(parts)
                   + "; consistent with a cut or inserted segment")  # fmt: skip
        else:
            score = SCORE_NONE
            why = "no clicks or DC-offset jumps: no sign of a cut or inserted segment"
        why += f" (background-floor range {m['floor_range_db']:.0f} dB, informational)"
        return DetectorResult(self.name, score, why, m)


DETECTOR = SpliceDetector()
if NAME not in REGISTRY.names():
    register(DETECTOR)
