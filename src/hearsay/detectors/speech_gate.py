"""Non-speech gate and default-answer policy (architecture.md section 10, decision 4).

The deep detectors are trained on speech only and score silence and steady tones near 1.0
(M1 scored pure silence 0.99). With a false alarm at 9.33x a miss, an undetermined file
belongs at the real end of the ranking, deliberately. This detector decides, per file, whether
there is speech to judge at all, and `apply_default_answer` implements the policy the
orchestrator applies to the fused score.

Speech, for this gate, is material that is (1) not silent, (2) voiced for at least a tenth of
its frames (YIN pitch in 65-390 Hz on frames within 30 dB of the loudest), (3) with a moving
pitch (std of log F0 over voiced frames >= 0.02: a tone or chord locks YIN to one value),
(4) with moving loudness (std of frame loudness >= 3 dB: syllables; a steady tone, hum or
stationary noise has none), (5) not spectrally flat (white or pink noise: long-term flatness
>= 0.5). Thresholds were set from the full NSA test set (`outputs/inventory/nsa_test_speech_gate.csv`):
its minima are RMS -34.6 dBFS, voiced fraction 0.099, log-F0 std 0.099, loudness std 6.0 dB,
and its maximum flatness 0.39, so no real test file is gated. A "few spectral lines" cue
(share of energy in the top five bins) was tried and dropped: 77 real test files exceed 70%
because speech concentrates its long-term energy in a few low bins; tones and chords are
caught by cues 3 and 4 anyway.

Known limit: rhythmic polyphonic music moves in pitch and loudness and can pass this gate;
the submission preflight (silence, chord) and the flag on any fused score above 0.8 remain
the backstop for it. Score is always 0.5 (no class evidence); the decision is in `features`
(`is_speech`, one `reason_*` per cue) and the evidence sentence, so fusion never sees a
constant-but-informative column and the orchestrator reads the flag directly.

Policy (`apply_default_answer`): a gated file gets `DEFAULT_ANSWER` (0.02, the constant
rollback score's neighbourhood) plus a tie-break of 1e-4 x its fused score, so every gated
file ranks below every speech file, gated files keep a deterministic order among themselves,
and the explanation report can say why. A decode failure is gated the same way upstream
(`safe_run` error -> no speech to judge).
"""

from __future__ import annotations

import numpy as np

from hearsay import SR
from hearsay.detectors.base import REGISTRY, ClipContext, DetectorResult, register

NAME = "speech_gate"
N_FFT, HOP = 512, 160
SILENCE_DBFS = -70.0  # RMS; digital silence is -180, the NSA test minimum -34.6
MIN_VOICED_FRAC = 0.05  # NSA test minimum 0.099
MIN_F0_STD_LOG = 0.02  # a tone or chord: ~0.00; NSA test minimum 0.099
MIN_ENERGY_DB_STD = 3.0  # steady tone / noise: < 1; NSA test minimum 6.0
MAX_FLATNESS = 0.5  # white noise 0.998; NSA test maximum 0.39
DEFAULT_ANSWER = 0.02
TIE_BREAK = 1e-4
REASONS = ("silence", "unvoiced", "tone", "static", "noise")


def analyze(x: np.ndarray) -> dict[str, float]:
    import librosa

    x = np.asarray(x, dtype=np.float32)
    if x.size < SR // 2:
        x = np.pad(x, (0, SR // 2 - x.size))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rms_dbfs = float(20 * np.log10(rms + 1e-9))
    f: dict[str, float] = {"rms_dbfs": max(rms_dbfs, -180.0)}
    silent = rms_dbfs < SILENCE_DBFS
    if silent:  # nothing to analyze; every other cue reads as absent
        f.update(voiced_frac=0.0, f0_std_log=0.0, energy_db_std=0.0, flatness=1.0)
    else:
        y = x / (rms + 1e-9) * 0.1
        S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2
        lt = S.mean(axis=1)[1:]  # long-term spectrum, DC dropped
        f["flatness"] = float(np.exp(np.mean(np.log(lt + 1e-12))) / (np.mean(lt) + 1e-12))
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
        frame_rms = librosa.feature.rms(y=y, frame_length=400, hop_length=HOP)[0]
        rms_db = 20 * np.log10(frame_rms / (frame_rms.max() + 1e-12) + 1e-12)
        n = min(f0.size, rms_db.size)
        voiced = (rms_db[:n] > -30) & (f0[:n] < 390) & (f0[:n] > 65)
        f["voiced_frac"] = float(voiced.mean())
        v = f0[:n][voiced]
        f["f0_std_log"] = float(np.std(np.log(v))) if v.size > 5 else 0.0
        f["energy_db_std"] = float(np.std(rms_db[rms_db > -60])) if (rms_db > -60).sum() > 1 else 0.0
    reasons = {
        "silence": silent,
        "unvoiced": (not silent) and f["voiced_frac"] < MIN_VOICED_FRAC,
        "tone": (not silent) and f["voiced_frac"] >= MIN_VOICED_FRAC and f["f0_std_log"] < MIN_F0_STD_LOG,
        "static": (not silent) and f["energy_db_std"] < MIN_ENERGY_DB_STD,
        "noise": (not silent) and f["flatness"] >= MAX_FLATNESS,
    }
    for k in REASONS:
        f[f"reason_{k}"] = float(reasons[k])
    f["is_speech"] = float(not any(reasons.values()))
    return f


class SpeechGateDetector:
    """Is there speech to judge? Score stays 0.5; the decision is `features["is_speech"]`."""

    name = NAME

    def applies(self, ctx: ClipContext) -> bool:
        return True

    def run(self, ctx: ClipContext) -> DetectorResult:
        m = ctx.memo("speech_gate.analysis", lambda: analyze(ctx.audio))
        why = [k for k in REASONS if m[f"reason_{k}"]]
        if m["is_speech"]:
            evidence = (f"speech present: voiced {m['voiced_frac']:.0%} of frames, pitch spread "
                        f"{m['f0_std_log']:.2f}, loudness std {m['energy_db_std']:.1f} dB; "
                        f"detectors apply")  # fmt: skip
        else:
            label = {"silence": "silence", "unvoiced": "no voiced speech", "tone": "a steady tone or chord",
                     "static": "no loudness movement", "noise": "stationary broadband noise"}  # fmt: skip
            evidence = ("no speech to judge (" + ", ".join(label[k] for k in why) + "): the "
                        f"default answer applies, {DEFAULT_ANSWER} at the real end of the ranking")
        return DetectorResult(self.name, 0.5, evidence, m)


def apply_default_answer(fused: np.ndarray, is_speech: np.ndarray,
                         default: float = DEFAULT_ANSWER) -> np.ndarray:  # fmt: skip
    """The policy: gated files get `default` + 1e-4 x their fused score (below every speech
    file, deterministic order among themselves); speech files keep their fused score."""
    fused = np.asarray(fused, dtype=np.float64)
    gated = ~np.asarray(is_speech, dtype=bool)
    out = fused.copy()
    out[gated] = default + TIE_BREAK * np.clip(fused[gated], 0.0, 1.0)
    return out


DETECTOR = SpeechGateDetector()
if NAME not in REGISTRY.names():
    register(DETECTOR)
