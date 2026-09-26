"""Frozen SSL embeddings, shared by bulk extraction (scripts/extract_embeddings.py) and the
per-clip CSV/live scorer, so train and test features come from one code path.

Per clip: 4 s windows (50% hop, first `max_windows`) -> per-window zero-mean/unit-variance
normalization (as Wav2Vec2FeatureExtractor's do_normalize=True; also removes the level
difference between corpora, e.g. LJ peaks ~0.54 vs NSA test ~1.0) -> backbone with all hidden
states -> time-mean per layer -> mean over windows => (n_layers, dim).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from hearsay import SR
from hearsay.audio import trim_silence, windows

REPO = Path(__file__).resolve().parents[2]
TEST_DURATIONS = REPO / "splits" / "nsa_test_durations.csv"
MAX_SEGMENT_S = 8.0


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_backbone(name: str = "wav2vec2-xls-r-300m", device: str | None = None):
    return AutoModel.from_pretrained(REPO / "weights" / name).eval().to(device or default_device())


def clip_windows(x: np.ndarray, win_s: float = 4.0, max_windows: int = 4) -> np.ndarray:
    return windows(x, int(win_s * SR))[:max_windows]


def normalize_windows(w: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per window (Wav2Vec2FeatureExtractor, eps 1e-7)."""
    w = np.asarray(w, dtype=np.float32)
    mu = w.mean(axis=-1, keepdims=True)
    var = w.var(axis=-1, keepdims=True)
    return ((w - mu) / np.sqrt(var + 1e-7)).astype(np.float32)


@torch.inference_mode()
def embed_windows(model, wins, batch: int = 16) -> np.ndarray:
    """(n_windows, n_layers, dim) float32: per-layer time-mean of each normalized window."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(wins), batch):
        x = torch.from_numpy(normalize_windows(np.stack(wins[i : i + batch]))).to(device)
        hs = model(x, output_hidden_states=True).hidden_states
        out.append(torch.stack([h.mean(dim=1) for h in hs], dim=1).float().cpu().numpy())
    return np.concatenate(out)


def embed_clip(model, x: np.ndarray, win_s: float = 4.0, max_windows: int = 4) -> np.ndarray:
    """(n_layers, dim) float32 for one 16 kHz mono clip."""
    return embed_windows(model, list(clip_windows(x, win_s, max_windows))).mean(axis=0)


# --- segment mode (v2): no tiling, test-matched lengths --------------------------------------
#
# Fixed 4 s windows tile clips shorter than 4 s (most NSA test clips are 3.0-3.8 s) and never
# tile training clips (5-9 s): a seam artifact present only at test time. Segment mode embeds
# the silence-trimmed clip at its own length (capped at MAX_SEGMENT_S); training clips are
# randomly cropped to lengths drawn from the NSA test-duration distribution.


def test_duration_sampler(seed: int = 0):
    """Callable returning a crop length (s) drawn from the NSA test durations."""
    import pandas as pd

    durs = pd.read_csv(TEST_DURATIONS).duration_s.to_numpy()
    rng = np.random.default_rng(seed)
    return lambda: float(rng.choice(durs))


def prepare_segment(
    x: np.ndarray, crop_s: float | None = None, seed: int | None = None,
    max_s: float = MAX_SEGMENT_S, band_match: bool = True,
) -> np.ndarray:  # fmt: skip
    """band_limit (NSA 7.25 kHz wall) -> trim_silence -> optional random crop to `crop_s`
    (never pads or tiles: a shorter clip stays as is) -> cap at `max_s`. Test-time scoring
    calls it with no crop.

    Band match: every NSA test clip is low-passed at ~7.2 kHz (7.5-8 kHz energy share 1.6e-8
    vs 6e-4..6e-3 in LJ, LibriSpeech and DiffSSD), so an unfiltered training clip carries a
    band no test clip has. The filter is hearsay.handcrafted.band_limit (single
    implementation, shared with the engineered detectors); callers that already filtered
    pass band_match=False."""
    if band_match:
        from hearsay.handcrafted import band_limit

        x = band_limit(x)
    x = trim_silence(x)
    if crop_s is not None and x.size > int(crop_s * SR):
        n = int(crop_s * SR)
        off = int(np.random.default_rng(seed).integers(0, x.size - n + 1))
        x = x[off : off + n]
    return np.ascontiguousarray(x[: int(max_s * SR)], dtype=np.float32)


@torch.inference_mode()
def embed_segment(model, x: np.ndarray) -> np.ndarray:
    """(n_layers, dim) float32: normalized whole-segment forward, time-mean per layer."""
    device = next(model.parameters()).device
    t = torch.from_numpy(normalize_windows(x[None, :])).to(device)
    hs = model(t, output_hidden_states=True).hidden_states
    return torch.stack([h.mean(dim=1)[0] for h in hs]).float().cpu().numpy()
