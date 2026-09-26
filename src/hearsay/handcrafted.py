"""Interpretable forensic features (brief rubric: spectral + prosody), CPU only.

Per clip: load_audio -> trim_silence -> first 4 s (the same crop the SSL detector sees) ->
RMS-normalize (level is a corpus shortcut, never a feature) -> features. Duration, peak and
absolute level are deliberately excluded.

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
CROP_S = 4.0


def _crop(x: np.ndarray) -> np.ndarray:
    x = trim_silence(x)[: int(CROP_S * SR)]
    return x / (np.sqrt(np.mean(x**2)) + 1e-8) * 0.1


def features(x: np.ndarray) -> dict[str, float]:
    import librosa

    x = _crop(x).astype(np.float32)
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


def features_for_path(path: str) -> dict[str, float] | None:
    try:
        return features(load_audio(path))
    except Exception:  # noqa: BLE001 - one bad file becomes a missing row, reported by caller
        return None
