"""Acoustic-environment consistency: electrical network frequency (ENF) mains hum (brief
rubric, technique 4). CPU only, rule-based, never learned.

Mains hum leaks into recordings made near powered equipment as a narrow tone at 50 or 60 Hz
whose frequency drifts slowly (tens of mHz) with the grid. A TTS or vocoder does not
synthesize it, so a stable hum is mild evidence of a genuine electrical recording
environment; a hum whose frequency jumps within the clip is mild evidence of a fabricated or
spliced background. Most clips (studio, TTS, phone) carry no hum at all, which is no evidence
either way. The scores are deliberately mild (0.4 / 0.6 / 0.5): the training corpora are
studio and audiobook recordings without hum, so the rule cannot be validated on labels and a
strong swing would be "confidently wrong on a subpopulation".

Method: silence-trimmed clip capped at 8 s, DC-removed and level-normalized; 2 s Hann windows
(0.49 Hz bins, quadratic peak interpolation) at a 0.5 s hop. For each candidate (50, 60 Hz)
track the strongest bin within +-1 Hz per frame and measure its SNR against the +-2..8 Hz
neighbourhood. Present = median SNR >= 12 dB and >= 60% of frames above 12 dB. Stable =
tracked-frequency std <= 0.25 Hz and a max-minus-min range <= 0.5 Hz over the present frames.
"""

from __future__ import annotations

import numpy as np

from hearsay import SR
from hearsay.audio import trim_silence
from hearsay.detectors.base import REGISTRY, ClipContext, DetectorResult, register

NAME = "enf"
WIN_S, HOP_S, MAX_S = 2.0, 0.5, 8.0
CANDIDATES = (50.0, 60.0)
SNR_DB, FRAC_PRESENT, STD_HZ, JUMP_HZ = 12.0, 0.6, 0.25, 0.5
SCORE_STABLE, SCORE_JUMPY, SCORE_NONE = 0.4, 0.6, 0.5


def _frames(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_frames, n_bins) power spectra of 2 s Hann windows, and the bin frequencies."""
    x = trim_silence(np.asarray(x, dtype=np.float32))[: int(MAX_S * SR)]
    x = x - x.mean()
    x = x / (np.sqrt(np.mean(x**2)) + 1e-8)
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    n_fft = 1 << (win - 1).bit_length()
    w = np.hanning(win).astype(np.float32)
    starts = range(0, x.size - win + 1, hop)
    fr = np.stack([x[s : s + win] * w for s in starts])
    spec = np.abs(np.fft.rfft(fr, n_fft, axis=1)) ** 2
    return spec, np.fft.rfftfreq(n_fft, 1 / SR)


def analyze(x: np.ndarray) -> dict[str, float]:
    """Hum measurements for the better of the 50/60 Hz candidates."""
    spec, freqs = _frames(x)
    df = float(freqs[1] - freqs[0])
    best: dict[str, float] | None = None
    for f0 in CANDIDATES:
        inb = np.where((freqs >= f0 - 1) & (freqs <= f0 + 1))[0]
        nb = ((freqs >= f0 - 8) & (freqs <= f0 - 2)) | ((freqs >= f0 + 2) & (freqs <= f0 + 8))
        pk = inb[np.argmax(spec[:, inb], axis=1)]
        rows = np.arange(spec.shape[0])
        a, b, c = spec[rows, pk - 1], spec[rows, pk], spec[rows, pk + 1]
        la, lb, lc = (np.log(v + 1e-30) for v in (a, b, c))
        denom = la - 2 * lb + lc
        delta = np.where(np.abs(denom) > 1e-12, 0.5 * (la - lc) / np.where(denom == 0, 1, denom), 0)
        f_track = freqs[pk] + np.clip(delta, -1, 1) * df
        snr = 10 * np.log10(b / (np.median(spec[:, nb], axis=1) + 1e-30))
        on = snr >= SNR_DB
        f_on = f_track[on]
        res = {
            "enf_candidate_hz": f0,
            "enf_snr_db": float(np.median(snr)),
            "enf_frac_present": float(on.mean()),
            "enf_freq_hz": float(np.median(f_on)) if f_on.size else 0.0,
            "enf_freq_std_hz": float(f_on.std()) if f_on.size > 1 else 0.0,
            "enf_freq_range_hz": float(f_on.max() - f_on.min()) if f_on.size > 1 else 0.0,
            "enf_n_frames": float(spec.shape[0]),
        }
        if best is None or res["enf_snr_db"] > best["enf_snr_db"]:
            best = res
    assert best is not None
    best["enf_present"] = float(
        best["enf_snr_db"] >= SNR_DB and best["enf_frac_present"] >= FRAC_PRESENT
    )
    best["enf_stable"] = float(
        best["enf_present"] == 1.0
        and best["enf_freq_std_hz"] <= STD_HZ
        and best["enf_freq_range_hz"] <= JUMP_HZ
    )
    return best


class EnfDetector:
    """Mains-hum presence and stability -> mild score + evidence; runs on every clip."""

    name = NAME

    def applies(self, ctx: ClipContext) -> bool:
        return True

    def run(self, ctx: ClipContext) -> DetectorResult:
        m = ctx.memo("enf.analysis", lambda: analyze(ctx.audio))
        f0, snr, frac = m["enf_candidate_hz"], m["enf_snr_db"], m["enf_frac_present"]
        if not m["enf_present"]:
            score = SCORE_NONE
            why = (f"no mains hum at 50 or 60 Hz (best {f0:.0f} Hz candidate: median SNR "
                   f"{snr:.1f} dB, above {SNR_DB:.0f} dB in {frac:.0%} of frames): no "
                   f"environment evidence either way")  # fmt: skip
        elif m["enf_stable"]:
            score = SCORE_STABLE
            why = (f"stable {m['enf_freq_hz']:.2f} Hz mains hum (SNR {snr:.1f} dB in {frac:.0%} "
                   f"of frames, drift {1000 * m['enf_freq_std_hz']:.0f} mHz): consistent with a "
                   f"genuine electrical recording environment")  # fmt: skip
        else:
            score = SCORE_JUMPY
            why = (f"mains hum near {f0:.0f} Hz present but discontinuous (frequency std "
                   f"{m['enf_freq_std_hz']:.2f} Hz, range {m['enf_freq_range_hz']:.2f} Hz): "
                   f"consistent with a fabricated or spliced background")  # fmt: skip
        return DetectorResult(self.name, score, why, {k: float(v) for k, v in m.items()})


DETECTOR = EnfDetector()
if NAME not in REGISTRY.names():
    register(DETECTOR)
