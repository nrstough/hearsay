"""Torch-free input preparation shared by the deep path (hearsay.embed, hearsay.spectra,
hearsay.m5_data) and the CPU-only engineered detectors (hearsay.handcrafted and the
compression features), so every path sees the same audio without importing torch.

Windows: 4 s windows (50% hop, first `max_windows`) and the per-window zero-mean/unit-variance
normalization Wav2Vec2FeatureExtractor applies (do_normalize=True). Segment mode (v2): band
match -> trim -> optional test-duration crop -> 8 s cap, no tiling. hearsay.embed re-exports
these names (test_duration_sampler as a wrapper on its own TEST_DURATIONS), so imports such as
`from hearsay.embed import prepare_segment` keep working.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hearsay import SR
from hearsay.audio import trim_silence, windows

REPO = Path(__file__).resolve().parents[2]
TEST_DURATIONS = REPO / "splits" / "nsa_test_durations.csv"
MAX_SEGMENT_S = 8.0


def clip_windows(x: np.ndarray, win_s: float = 4.0, max_windows: int = 4) -> np.ndarray:
    return windows(x, int(win_s * SR))[:max_windows]


def normalize_windows(w: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance per window (Wav2Vec2FeatureExtractor, eps 1e-7)."""
    w = np.asarray(w, dtype=np.float32)
    mu = w.mean(axis=-1, keepdims=True)
    var = w.var(axis=-1, keepdims=True)
    return ((w - mu) / np.sqrt(var + 1e-7)).astype(np.float32)


# --- segment mode (v2): no tiling, test-matched lengths --------------------------------------
#
# Fixed 4 s windows tile clips shorter than 4 s (most NSA test clips are 3.0-3.8 s) and never
# tile training clips (5-9 s): a seam artifact present only at test time. Segment mode embeds
# the silence-trimmed clip at its own length (capped at MAX_SEGMENT_S); training clips are
# randomly cropped to lengths drawn from the NSA test-duration distribution.


def test_duration_sampler(seed: int = 0, durations_csv: str | Path | None = None):
    """Callable returning a crop length (s) drawn from the NSA test durations
    (`durations_csv`, default TEST_DURATIONS)."""
    import pandas as pd

    path = TEST_DURATIONS if durations_csv is None else durations_csv
    durs = pd.read_csv(path).duration_s.to_numpy()
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
