"""Speaker-embedding consistency (brief rubric, technique 6): does the voice drift across the
clip? ECAPA-TDNN speaker embeddings (SpeechBrain `spkrec-ecapa-voxceleb`, Apache-2.0, in
weights/) on 1.0 s windows at a 0.5 s hop over the silence-trimmed clip; the pairwise cosine
similarities between windows are the evidence. A single real speaker's windows agree closely;
a clip whose windows disagree (a spliced second voice, a cloning system that wanders between
target identities, a partial fake) scores lower. Rule-based and mild, like ENF: 0.6 when the
minimum pairwise cosine falls below DRIFT_COS, 0.5 otherwise.

Calibration (`outputs/inventory/speaker_drift_calibration.csv`, inner rows only; 180 real
single-speaker clips, 160 DiffSSD fakes, 80 synthetic two-speaker splices of LibriSpeech
halves, 80 same-speaker splices as control): at DRIFT_COS = 0.05 the rule flags 94% of the
two-speaker splices and 6% of single-voice clips (LibriSpeech real 9%, LJ 5%, fakes 1%).
Fakes are the *most* self-consistent voices here (cos_min mean 0.28 vs 0.20 for real), so a
fused column would learn "drift = real"; this detector is evidence and routing only, never a
fusion column. On 3 s clips (4-6 windows) scores are near-constant by construction.

CPU only; the encoder is loaded lazily on first use from weights/spkrec-ecapa-voxceleb (never
from the Hub at run time: HF_HUB_OFFLINE is fine).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hearsay import SR
from hearsay.audio import trim_silence
from hearsay.detectors.base import REGISTRY, ClipContext, DetectorResult, register

NAME = "speaker_drift"
REPO = Path(__file__).resolve().parents[3]
WEIGHTS = REPO / "weights" / "spkrec-ecapa-voxceleb"
WIN_S, HOP_S, MAX_S = 1.0, 0.5, 8.0
DRIFT_COS = 0.05  # min pairwise cosine; calibrated: 94% of 2-speaker splices, 6% of single voices
SCORE_DRIFT, SCORE_NONE = 0.6, 0.5
_ENCODER = None


def encoder(weights: Path = WEIGHTS):
    """The ECAPA encoder, loaded once per process, CPU."""
    global _ENCODER
    if _ENCODER is None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        torch.set_num_threads(max(1, torch.get_num_threads() // 2))
        _ENCODER = EncoderClassifier.from_hparams(source=str(weights), savedir=str(weights),
                                                  run_opts={"device": "cpu"})  # fmt: skip
    return _ENCODER


def windows(x: np.ndarray) -> np.ndarray:
    x = trim_silence(np.asarray(x, dtype=np.float32))[: int(MAX_S * SR)]
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    starts = list(range(0, x.size - win + 1, hop))
    if starts[-1] + win < x.size:
        starts.append(x.size - win)
    return np.stack([x[s : s + win] for s in starts])


def embed(x: np.ndarray) -> np.ndarray:
    """(n_windows, 192) unit-norm ECAPA embeddings."""
    import torch

    w = windows(x)
    with torch.inference_mode():
        e = encoder().encode_batch(torch.from_numpy(w)).squeeze(1).cpu().numpy()
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)


def analyze(x: np.ndarray) -> dict[str, float]:
    e = embed(x)
    n = e.shape[0]
    if n < 2:
        return {"n_windows": float(n), "cos_min": 1.0, "cos_mean": 1.0, "cos_std": 0.0,
                "cos_adjacent_min": 1.0, "drift": 0.0}  # fmt: skip
    c = e @ e.T
    iu = np.triu_indices(n, k=1)
    pair = c[iu]
    adjacent = np.array([c[i, i + 1] for i in range(n - 1)])
    return {
        "n_windows": float(n),
        "cos_min": float(pair.min()),
        "cos_mean": float(pair.mean()),
        "cos_std": float(pair.std()),
        "cos_adjacent_min": float(adjacent.min()),
        "drift": float(pair.min() < DRIFT_COS),
    }


class SpeakerDriftDetector:
    """ECAPA window embeddings -> pairwise cosine -> mild score + evidence."""

    name = NAME

    def applies(self, ctx: ClipContext) -> bool:
        return True

    def run(self, ctx: ClipContext) -> DetectorResult:
        m = ctx.memo("speaker_drift.analysis", lambda: analyze(ctx.audio))
        if m["drift"]:
            score = SCORE_DRIFT
            why = (f"voice drifts across the clip: minimum window-to-window speaker similarity "
                   f"{m['cos_min']:.2f} (mean {m['cos_mean']:.2f}) over {m['n_windows']:.0f} "
                   f"windows; consistent with a second voice or an unstable cloned identity")  # fmt: skip
        else:
            score = SCORE_NONE
            why = (f"one consistent voice: minimum window-to-window speaker similarity "
                   f"{m['cos_min']:.2f} (mean {m['cos_mean']:.2f}) over {m['n_windows']:.0f} "
                   f"windows")  # fmt: skip
        return DetectorResult(self.name, score, why, m)


DETECTOR = SpeakerDriftDetector()
if NAME not in REGISTRY.names():
    register(DETECTOR)
