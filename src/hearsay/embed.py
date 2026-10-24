"""Frozen SSL embeddings, shared by bulk extraction (scripts/extract_embeddings.py) and the
per-clip CSV/live scorer, so train and test features come from one code path.

Per clip: 4 s windows (50% hop, first `max_windows`) -> per-window zero-mean/unit-variance
normalization (as Wav2Vec2FeatureExtractor's do_normalize=True; also removes the level
difference between corpora, e.g. LJ peaks ~0.54 vs NSA test ~1.0) -> backbone with all hidden
states -> time-mean per layer -> mean over windows => (n_layers, dim).

torch and transformers are imported inside the functions that use them, so importing this
module (or the numpy helpers it re-exports from hearsay.segment) never needs torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hearsay import segment as _segment
from hearsay.segment import (  # noqa: F401  (re-exported: existing imports keep working)
    MAX_SEGMENT_S,
    clip_windows,
    normalize_windows,
    prepare_segment,
)

REPO = Path(__file__).resolve().parents[2]
TEST_DURATIONS = REPO / "splits" / "nsa_test_durations.csv"


def default_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_backbone(name: str = "wav2vec2-xls-r-300m", device: str | None = None):
    from transformers import AutoModel

    return AutoModel.from_pretrained(REPO / "weights" / name).eval().to(device or default_device())


def embed_windows(model, wins, batch: int = 16) -> np.ndarray:
    """(n_windows, n_layers, dim) float32: per-layer time-mean of each normalized window."""
    import torch

    device = next(model.parameters()).device
    out = []
    with torch.inference_mode():
        for i in range(0, len(wins), batch):
            x = torch.from_numpy(normalize_windows(np.stack(wins[i : i + batch]))).to(device)
            hs = model(x, output_hidden_states=True).hidden_states
            out.append(torch.stack([h.mean(dim=1) for h in hs], dim=1).float().cpu().numpy())
    return np.concatenate(out)


def embed_clip(model, x: np.ndarray, win_s: float = 4.0, max_windows: int = 4) -> np.ndarray:
    """(n_layers, dim) float32 for one 16 kHz mono clip."""
    return embed_windows(model, list(clip_windows(x, win_s, max_windows))).mean(axis=0)


# --- segment mode (v2): see hearsay.segment for the input path -------------------------------


def test_duration_sampler(seed: int = 0):
    """hearsay.segment.test_duration_sampler on this module's TEST_DURATIONS, so patching
    hearsay.embed.TEST_DURATIONS still reaches every caller that imports the sampler from here."""
    return _segment.test_duration_sampler(seed, TEST_DURATIONS)


def embed_segment(model, x: np.ndarray) -> np.ndarray:
    """(n_layers, dim) float32: normalized whole-segment forward, time-mean per layer."""
    import torch

    device = next(model.parameters()).device
    with torch.inference_mode():
        t = torch.from_numpy(normalize_windows(x[None, :])).to(device)
        hs = model(t, output_hidden_states=True).hidden_states
        return torch.stack([h.mean(dim=1)[0] for h in hs]).float().cpu().numpy()
