"""Handcrafted v4 feature families (docs/handoffs/2026-09-26_handcrafted-v4-brief.md): cheap
acoustic statistics aimed at the generators v3 cannot see (grad_tts, pro_diff, elevenlabs),
computed on the shared preprocessing (band-limited, trimmed, cropped, RMS-normalized 16 kHz
audio) from the same 512/160 STFT that `hearsay.handcrafted.features` already takes.

Each family is a function of a context dict (`x`, `Z` complex STFT, `S` power STFT, `freqs`,
`voiced` frame mask, `rms_db`, `f0`) returning a flat {name: float}; `FAMILIES` maps names to
functions and `features(..., families=(...))` appends them. Rules from the brief: nothing above
7 kHz (the test set is low-passed there), no level, no duration, no onset/offset cues; means,
stds, percentiles and rates only. Column prefixes: `lfcc`, `gd_`, `pc_`, `cqcc`, `mod_`, `breath_`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hearsay import SR

N_FFT, HOP = 512, 160
FMAX = 7000.0
N_LFCC_FILTERS, N_LFCC = 40, 20
PHASE_FMIN, PHASE_FMAX = 300.0, 7000.0
GD_CLIP = 256.0
GD_FLOOR_DB = 30.0  # per frame, bins within this of the frame's peak count as energetic


def _stats(v: np.ndarray, prefix: str, pct: bool = False) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        v = np.zeros(1)
    out = {f"{prefix}_mean": float(v.mean()), f"{prefix}_std": float(v.std())}
    if pct:
        out[f"{prefix}_p10"] = float(np.percentile(v, 10))
        out[f"{prefix}_p90"] = float(np.percentile(v, 90))
    return out


# --- 8.1 LFCC: linear-frequency cepstral coefficients, 0-7 kHz -------------------------------


def _lfcc_filterbank(freqs: np.ndarray) -> np.ndarray:
    """(n_filters, n_bins) triangular filters linearly spaced from 0 to FMAX."""
    edges = np.linspace(0.0, FMAX, N_LFCC_FILTERS + 2)
    fb = np.zeros((N_LFCC_FILTERS, freqs.size))
    for i in range(N_LFCC_FILTERS):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs >= lo) & (freqs <= mid)
        down = (freqs > mid) & (freqs <= hi)
        fb[i, up] = (freqs[up] - lo) / (mid - lo + 1e-12)
        fb[i, down] = (hi - freqs[down]) / (hi - mid + 1e-12)
    return fb


_FB_CACHE: dict[int, np.ndarray] = {}


def lfcc(ctx: dict) -> dict[str, float]:
    from scipy.fft import dct

    S, freqs = ctx["S"], ctx["freqs"]
    if freqs.size not in _FB_CACHE:
        _FB_CACHE[freqs.size] = _lfcc_filterbank(freqs)
    fb = _FB_CACHE[freqs.size]
    logE = np.log(fb @ S + 1e-10)  # (n_filters, n_frames)
    c = dct(logE, type=2, norm="ortho", axis=0)[:N_LFCC]  # (n_lfcc, n_frames)
    f: dict[str, float] = {}
    d = np.diff(c, axis=1) if c.shape[1] > 1 else np.zeros((N_LFCC, 1))
    for i in range(N_LFCC):
        f[f"lfcc{i}_mean"] = float(c[i].mean())
        f[f"lfcc{i}_std"] = float(c[i].std())
        f[f"lfcc{i}_dstd"] = float(d[i].std())
    return f


# --- 8.2 Phase: group delay and instantaneous-frequency deviation ----------------------------


def _band_and_voiced(ctx: dict) -> tuple[np.ndarray, np.ndarray]:
    freqs, Z = ctx["freqs"], ctx["Z"]
    band = (freqs >= PHASE_FMIN) & (freqs <= PHASE_FMAX)
    voiced = np.asarray(ctx["voiced"], dtype=bool)[: Z.shape[1]]
    if voiced.sum() < 3:  # fall back to the loudest third of frames
        e = np.log(ctx["S"].sum(axis=0) + 1e-12)
        voiced = e >= np.percentile(e, 67)
    return band, voiced


def group_delay(ctx: dict) -> dict[str, float]:
    """Spread of the group delay across the energetic bins of each voiced frame (no phase
    unwrapping). tau = Re(Y / Z) with Y the STFT of n * x[n]; the per-frame constant offset
    from the global ramp is removed by subtracting the per-frame median. Only bins within
    GD_FLOOR_DB of the frame's peak count, so the near-empty bins between harmonics (whose
    group delay is noise) do not swamp the harmonics themselves."""
    import librosa

    x, Z = ctx["x"], ctx["Z"]
    band, voiced = _band_and_voiced(ctx)
    Y = librosa.stft((x * np.arange(x.size, dtype=np.float64)).astype(np.float32),
                     n_fft=N_FFT, hop_length=HOP)  # fmt: skip
    Zb, Yb = Z[band][:, voiced], Y[band][:, voiced]
    den = (Zb.real**2 + Zb.imag**2) + 1e-12
    tau = (Zb.real * Yb.real + Zb.imag * Yb.imag) / den
    mag_db = 10 * np.log10(den)
    energetic = mag_db >= mag_db.max(axis=0, keepdims=True) - GD_FLOOR_DB
    spread, iqr = [], []
    for t in range(tau.shape[1]):
        v = tau[energetic[:, t], t]
        if v.size < 4:
            continue
        v = np.clip(v - np.median(v), -GD_CLIP, GD_CLIP)
        spread.append(v.std())
        iqr.append(np.percentile(v, 75) - np.percentile(v, 25))
    return _stats(spread, "gd_std", pct=True) | _stats(iqr, "gd_iqr", pct=True)


def peak_coherence(ctx: dict) -> dict[str, float]:
    """Phase-vs-magnitude consistency at spectral peaks, per voiced frame (pitch-movement
    proof, unlike a fixed-bin instantaneous-frequency test). For each magnitude peak within
    GD_FLOOR_DB of the frame's peak (300-7000 Hz), the frequency implied by the phase advance
    to the next frame (phase-vocoder estimate) is compared with the frequency implied by the
    quadratic-interpolated magnitude peak. A coherent partial gives a few Hz of disagreement;
    incoherent phase (noise, some neural vocoders) gives tens of Hz. Features: median, mean and
    90th percentile of the disagreement in Hz, and the share of peaks off by more than a
    quarter bin (about 8 Hz)."""
    Z = ctx["Z"]
    band, voiced = _band_and_voiced(ctx)
    lo = int(np.argmax(band))
    Zb = Z[band]
    mag = np.abs(Zb) + 1e-12
    logm = np.log(mag)
    n_frames = Zb.shape[1]
    hz_per_bin = float(ctx["freqs"][1] - ctx["freqs"][0])
    hz_per_rad = SR / (2 * np.pi * HOP)
    errs: list[float] = []
    for t in range(n_frames - 1):
        if not (voiced[t] and voiced[t + 1]):
            continue
        col = logm[:, t]
        peaks = np.where((col[1:-1] > col[:-2]) & (col[1:-1] >= col[2:]))[0] + 1
        if peaks.size == 0:
            continue
        peaks = peaks[col[peaks] >= col.max() - GD_FLOOR_DB * np.log(10) / 10]
        if peaks.size == 0:
            continue
        a, b, c = col[peaks - 1], col[peaks], col[peaks + 1]
        denom = a - 2 * b + c
        delta = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(denom == 0, 1, denom), 0)
        f_mag = (peaks + lo + np.clip(delta, -0.5, 0.5)) * hz_per_bin
        dphi = np.angle(Zb[peaks, t + 1] * np.conj(Zb[peaks, t]))
        dphi = dphi - 2 * np.pi * (peaks + lo) * HOP / N_FFT
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        f_phase = (peaks + lo) * hz_per_bin + dphi * hz_per_rad
        errs.extend(np.abs(f_phase - f_mag).tolist())
    e = np.asarray(errs) if errs else np.zeros(1)
    return {
        "pc_err_median": float(np.median(e)),
        "pc_err_mean": float(e.mean()),
        "pc_err_p90": float(np.percentile(e, 90)),
        "pc_frac_incoherent": float((e > 0.25 * hz_per_bin).mean()),
    }


def phase(ctx: dict) -> dict[str, float]:
    return group_delay(ctx) | peak_coherence(ctx)


# --- 8.3 CQCC: constant-Q cepstral coefficients, 65 Hz - 7 kHz -----------------------------

CQT_FMIN, CQT_BINS, CQT_BPO, CQT_HOP, N_CQCC = 65.0, 162, 24, 256, 24  # fmax ~ 6,994 Hz


def cqcc(ctx: dict) -> dict[str, float]:
    import librosa
    from scipy.fft import dct

    C = np.abs(librosa.cqt(ctx["x"], sr=SR, fmin=CQT_FMIN, n_bins=CQT_BINS,
                           bins_per_octave=CQT_BPO, hop_length=CQT_HOP)) ** 2  # fmt: skip
    c = dct(np.log(C + 1e-10), type=2, norm="ortho", axis=0)[:N_CQCC]
    d = np.diff(c, axis=1) if c.shape[1] > 1 else np.zeros((N_CQCC, 1))
    f: dict[str, float] = {}
    for i in range(N_CQCC):
        f[f"cqcc{i}_mean"] = float(c[i].mean())
        f[f"cqcc{i}_std"] = float(c[i].std())
        f[f"cqcc{i}_dstd"] = float(d[i].std())
    return f


# --- 8.4 Rhythm / modulation spectrum of the loudness envelope --------------------------------

MOD_BANDS = ((1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0))
FRAME_RATE = SR / HOP  # 100 envelope samples per second
MOD_NFFT = 2048  # 0.05 Hz bins; zero-padded so the band edges are fixed for any clip length


def modulation(ctx: dict) -> dict[str, float]:
    e = np.asarray(ctx["rms_db"], dtype=np.float64)
    e = e - e.mean()
    if e.size < 8:
        e = np.pad(e, (0, 8 - e.size))
    n_fft = max(MOD_NFFT, 1 << (e.size - 1).bit_length())  # fixed grid: band edges do not move
    P = np.abs(np.fft.rfft(e * np.hanning(e.size), n_fft)) ** 2
    fr = np.fft.rfftfreq(n_fft, d=1.0 / FRAME_RATE)
    inb = (fr >= 1.0) & (fr < 16.0)
    tot = float(P[inb].sum()) + 1e-12
    f: dict[str, float] = {}
    for lo, hi in MOD_BANDS:
        f[f"mod_{int(lo)}_{int(hi)}"] = float(P[(fr >= lo) & (fr < hi)].sum() / tot)
    f["mod_peak_hz"] = float(fr[inb][np.argmax(P[inb])]) if inb.any() else 0.0
    q = P[inb] / tot
    f["mod_entropy"] = float(-(q[q > 0] * np.log(q[q > 0])).sum() / np.log(max(q.size, 2)))
    return f


# --- 8.6 Breath: the between-speech material ------------------------------------------------

BREATH_LO_DB, BREATH_HI_DB = -35.0, -20.0


def breath(ctx: dict) -> dict[str, float]:
    S, freqs, rms_db = ctx["S"], ctx["freqs"], np.asarray(ctx["rms_db"])
    voiced = np.asarray(ctx["voiced"], dtype=bool)
    n = min(S.shape[1], rms_db.size, voiced.size)
    band = freqs <= FMAX
    Sb = S[band][:, :n] + 1e-12
    cent = (freqs[band][:, None] * Sb).sum(axis=0) / Sb.sum(axis=0)
    flat = np.exp(np.log(Sb).mean(axis=0)) / Sb.mean(axis=0)
    mid = (rms_db[:n] > BREATH_LO_DB) & (rms_db[:n] < BREATH_HI_DB) & ~voiced[:n]
    v = voiced[:n]
    f = {"breath_frac": float(mid.mean()) if n else 0.0}
    if mid.sum() >= 3 and v.sum() >= 3:
        f["breath_centroid_ratio"] = float(cent[mid].mean() / (cent[v].mean() + 1e-12))
        f["breath_flatness_ratio"] = float(flat[mid].mean() / (flat[v].mean() + 1e-12))
        f["breath_flatness"] = float(flat[mid].mean())
    else:
        f["breath_centroid_ratio"] = f["breath_flatness_ratio"] = 1.0
        f["breath_flatness"] = 0.0
    return f


FAMILIES: dict[str, Callable[[dict], dict[str, float]]] = {
    "lfcc": lfcc, "phase": phase, "cqcc": cqcc, "modulation": modulation, "breath": breath,
}
