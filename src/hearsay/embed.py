"""Frozen SSL embeddings, shared by bulk extraction (scripts/extract_embeddings.py) and the
per-clip CSV/live scorer, so train and test features come from one code path.

Per clip: 4 s windows (50% hop, first `max_windows`) -> backbone with all hidden states ->
time-mean per layer -> mean over windows => (n_layers, dim).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from hearsay import SR
from hearsay.audio import windows

REPO = Path(__file__).resolve().parents[2]


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_backbone(name: str = "wav2vec2-xls-r-300m", device: str | None = None):
    return AutoModel.from_pretrained(REPO / "weights" / name).eval().to(device or default_device())


def clip_windows(x: np.ndarray, win_s: float = 4.0, max_windows: int = 4) -> np.ndarray:
    return windows(x, int(win_s * SR))[:max_windows]


@torch.inference_mode()
def embed_windows(model, wins, batch: int = 16) -> np.ndarray:
    """(n_windows, n_layers, dim) float32: per-layer time-mean of each window."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(wins), batch):
        x = torch.from_numpy(np.stack(wins[i : i + batch])).to(device)
        hs = model(x, output_hidden_states=True).hidden_states
        out.append(torch.stack([h.mean(dim=1) for h in hs], dim=1).float().cpu().numpy())
    return np.concatenate(out)


def embed_clip(model, x: np.ndarray, win_s: float = 4.0, max_windows: int = 4) -> np.ndarray:
    """(n_layers, dim) float32 for one 16 kHz mono clip."""
    return embed_windows(model, list(clip_windows(x, win_s, max_windows))).mean(axis=0)
