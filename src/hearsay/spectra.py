"""Spectra-AASIST (lab260) score stream, M3 / hard-gate fallback.

Model card: logits (batch, 2), index 0 = spoof, index 1 = bonafide; trained on 64,600-sample
(~4 s) crops after pre-emphasis. We score deterministic windows (repeat-pad short clips, 50%-hop
windows over long ones), average logits over windows, and export

    synth_logit = logit_spoof - logit_bonafide      (increases with synthetic likelihood)

Direction is asserted on labeled data by scripts/spectra_direction.py before any CSV ships.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights" / "Spectra-AASIST"
WIN = 64_600
HOP = WIN // 2


def load_spectra(device: str | None = None) -> torch.nn.Module:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    spec = importlib.util.spec_from_file_location("spectra_model", WEIGHTS / "model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # model.py hardcodes the hub id for its XLS-R skeleton; point it at weights/ instead.
    # Every encoder weight is then overwritten by Spectra-AASIST's own safetensors.
    hub_cls = mod.Wav2Vec2Model

    class _LocalWav2Vec2:
        @staticmethod
        def from_pretrained(name, *a, **kw):
            if name == "facebook/wav2vec2-xls-r-300m":
                name = REPO / "weights" / "wav2vec2-xls-r-300m"
            return hub_cls.from_pretrained(name, *a, **kw)

    mod.Wav2Vec2Model = _LocalWav2Vec2
    model = mod.SpectraAASIST.from_pretrained(str(WEIGHTS))
    return model.eval().to(device)


def windows(x: np.ndarray) -> np.ndarray:
    """(n_windows, WIN) float32. Short clips are tiled, as in the model card's pad_random."""
    if x.size == 0:
        x = np.zeros(WIN, dtype=np.float32)
    if x.size < WIN:
        x = np.tile(x, WIN // x.size + 1)[:WIN]
    starts = list(range(0, x.size - WIN + 1, HOP))
    if starts[-1] + WIN < x.size:
        starts.append(x.size - WIN)
    return np.stack([x[s : s + WIN] for s in starts]).astype(np.float32)


@torch.inference_mode()
def spectra_logits(model: torch.nn.Module, x: np.ndarray, max_windows: int = 16) -> np.ndarray:
    """Window-averaged (logit_spoof, logit_bonafide) for one 16 kHz mono clip."""
    device = next(model.parameters()).device
    w = torch.from_numpy(windows(x)[:max_windows])
    w = torchaudio.functional.preemphasis(w)
    return model(w.to(device)).float().cpu().numpy().mean(axis=0)


def synth_logit(logits: np.ndarray) -> float:
    return float(logits[0] - logits[1])
