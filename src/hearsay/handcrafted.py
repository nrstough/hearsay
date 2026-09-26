"""Interpretable forensic features (brief rubric: spectral + prosody), CPU only.

Per clip: load_audio -> segment -> RMS-normalize (level is a corpus shortcut, never a
feature) -> features. Duration, peak and absolute level are deliberately excluded.

Segment (v2, the default `crop_mode="segment"`): `hearsay.embed.prepare_segment`, the exact
trim / random-crop / 8 s cap the deep detector (M1 v2) sees, so fusion compares the two on the
same audio. Training clips are cropped to lengths drawn from the NSA test-duration
distribution (`crop_s`, offset drawn from `seed`); test clips are scored whole, with no crop.
v1 (commit 6ee5aa1, `crop_mode="first4s"`) took the first 4 s after the trim; it is kept so the
v1-vs-v2 comparison in docs/reports stays reproducible. v3 adds `band_match` (below): a
low-pass that reproduces the NSA test set's ~7.2 kHz roll-off on every clip.

Spectral: rolloff (85/95%), centroid, bandwidth, flatness, high-band energy ratios
(>4/6/7 kHz), spectral contrast, 20 MFCC means + stds, spectral-flux stats.
Prosody: F0 (YIN) median/IQR/std, voiced fraction, F0 slope std (monotonicity), frame-energy
dynamics, pause fraction and count, ZCR stats.
"""

from __future__ import annotations

import numpy as np

from hearsay import SR
from hearsay.audio import load_audio, trim_silence

N_FFT = 512
HOP = 160  # 10 ms
CROP_S = 4.0  # v1 ("first4s") only
CROP_MODES = ("segment", "first4s")

# NSA test-set band match (docs/reports/2026-09-26_cpu-detectors.md, "Band-limit"). 1,602 of
# the 1,671 test files carry a ~7.2 kHz low-pass (median -18 dB in the 7.25-7.5 kHz band and
# -44 dB at 7.5-7.75 kHz, relative to 6.5 kHz) that no training corpus has, not even the
# sponsor's own resampled LJ clips. This Kaiser FIR reproduces the roll-off (3 dB rms error
# over four bands on training clips) and is applied to every clip, train and test alike, so
# the 7-8 kHz band stops being a corpus fingerprint. A second pass over an already
# band-limited test clip changes almost nothing. v3 = v2 + band match.
BAND_MATCH = {"cutoff_hz": 7250.0, "width_hz": 600.0, "atten_db": 45.0}
_FIR: np.ndarray | None = None


def band_limit(x: np.ndarray) -> np.ndarray:
    """Zero-delay linear-phase low-pass matching the NSA test set (see BAND_MATCH)."""
    global _FIR
    import scipy.signal as sg

    if _FIR is None:
        n, beta = sg.kaiserord(BAND_MATCH["atten_db"], BAND_MATCH["width_hz"] / (SR / 2))
        _FIR = sg.firwin(n | 1, BAND_MATCH["cutoff_hz"], window=("kaiser", beta), fs=SR)
        _FIR = _FIR.astype(np.float32)
    return sg.fftconvolve(x, _FIR, mode="same").astype(np.float32)


def _crop(
    x: np.ndarray, crop_s: float | None = None, seed: int | None = None,
    crop_mode: str = "segment", band_match: bool = True,
) -> np.ndarray:  # fmt: skip
    if band_match:
        x = band_limit(x)
    if crop_mode == "segment":
        from hearsay.embed import prepare_segment  # imports torch; lazy so v1 stays light

        x = prepare_segment(x, crop_s, seed, band_match=False)  # band match handled above
    elif crop_mode == "first4s":
        x = trim_silence(x)[: int(CROP_S * SR)]
    else:
        raise ValueError(f"crop_mode must be one of {CROP_MODES}, got {crop_mode!r}")
    return x / (np.sqrt(np.mean(x**2)) + 1e-8) * 0.1


def features(
    x: np.ndarray, crop_s: float | None = None, seed: int | None = None,
    crop_mode: str = "segment", band_match: bool = True,
) -> dict[str, float]:  # fmt: skip
    import librosa

    x = _crop(x, crop_s, seed, crop_mode, band_match).astype(np.float32)
    f: dict[str, float] = {}
    s = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    tot = s.sum(axis=0) + 1e-12
    for hz in (4000, 6000, 7000):
        r = s[freqs >= hz].sum(axis=0) / tot
        f[f"hf_ratio_{hz // 1000}k_mean"] = float(np.mean(r))
        f[f"hf_ratio_{hz // 1000}k_std"] = float(np.std(r))
    for q in (0.85, 0.95):
        ro = librosa.feature.spectral_rolloff(S=s, sr=SR, roll_percent=q)[0]
        f[f"rolloff{int(q * 100)}_mean"] = float(np.mean(ro))
        f[f"rolloff{int(q * 100)}_std"] = float(np.std(ro))
    for name, v in (
        ("centroid", librosa.feature.spectral_centroid(S=s, sr=SR)[0]),
        ("bandwidth", librosa.feature.spectral_bandwidth(S=s, sr=SR)[0]),
        ("flatness", librosa.feature.spectral_flatness(S=s)[0]),
    ):
        f[f"{name}_mean"] = float(np.mean(v))
        f[f"{name}_std"] = float(np.std(v))
    contrast = librosa.feature.spectral_contrast(S=s, sr=SR, n_bands=5)
    for i, row in enumerate(contrast):
        f[f"contrast{i}_mean"] = float(np.mean(row))
    flux = np.sqrt(np.sum(np.diff(np.log(s + 1e-10), axis=1) ** 2, axis=0))
    f["flux_mean"], f["flux_std"] = float(np.mean(flux)), float(np.std(flux))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=s, sr=SR)),
                                n_mfcc=20)  # fmt: skip
    for i in range(20):
        f[f"mfcc{i}_mean"] = float(np.mean(mfcc[i]))
        f[f"mfcc{i}_std"] = float(np.std(mfcc[i]))

    # Prosody
    f0 = librosa.yin(x, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=x, frame_length=400, hop_length=HOP)[0]
    rms_db = 20 * np.log10(rms / (rms.max() + 1e-12) + 1e-12)
    voiced = (rms_db > -30) & (f0 < 390) & (f0 > 65)
    v = f0[voiced[: f0.size]] if voiced.size >= f0.size else f0[voiced]
    f["voiced_frac"] = float(np.mean(voiced))
    if v.size > 5:
        lv = np.log(v)
        f["f0_median"] = float(np.median(v))
        f["f0_iqr_log"] = float(np.subtract(*np.percentile(lv, [75, 25])))
        f["f0_std_log"] = float(np.std(lv))
        f["f0_slope_std"] = float(np.std(np.diff(lv)))
    else:
        f["f0_median"] = f["f0_iqr_log"] = f["f0_std_log"] = f["f0_slope_std"] = 0.0
    f["energy_db_std"] = float(np.std(rms_db))
    f["energy_db_p10"] = float(np.percentile(rms_db, 10))
    pause = rms_db < -35
    f["pause_frac"] = float(np.mean(pause))
    f["pause_count"] = float(np.sum(np.diff(pause.astype(int)) == 1))
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=400, hop_length=HOP)[0]
    f["zcr_mean"], f["zcr_std"] = float(np.mean(zcr)), float(np.std(zcr))
    return {k: (v if np.isfinite(v) else 0.0) for k, v in f.items()}


def features_for_path(
    path: str, crop_s: float | None = None, seed: int | None = None,
    crop_mode: str = "segment", band_match: bool = True,
) -> dict[str, float] | None:  # fmt: skip
    try:
        return features(load_audio(path), crop_s, seed, crop_mode, band_match)
    except Exception:  # noqa: BLE001 - one bad file becomes a missing row, reported by caller
        return None
