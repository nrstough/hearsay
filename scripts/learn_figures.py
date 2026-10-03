"""Figures for docs/learn/v1-explained.md (teaching aids, not results).

Reads only local assets: the NSA test WAVs, In-the-Wild clips, the model meta.json files and the
exported detector scores. Writes PNGs to docs/learn/img/. matplotlib is not a project dependency:

    uv run --with matplotlib python scripts/learn_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft

from hearsay.audio import load_audio

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "learn" / "img"
SR = 16_000


def _spec_db(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = load_audio(path)
    x = x[: 4 * SR]
    f, t, z = stft(x, fs=SR, nperseg=512, noverlap=384)
    s = 20 * np.log10(np.abs(z) + 1e-10)
    return f, t, s - s.max()


def spectrograms() -> None:
    itw = pd.read_csv(REPO / "outputs/manifests/itw_stress.csv")
    real = REPO / itw[itw.label == "bonafide"].path.iloc[0]
    fake = REPO / itw[itw.label == "spoof"].path.iloc[0]
    test = REPO / "data/nsa/HackGTHearsayTesting/HGT1013455.wav"
    panels = [
        (real, "In-the-Wild, real (label known)"),
        (fake, "In-the-Wild, fake (label known)"),
        (test, "NSA test file HGT1013455 (no label)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (p, title) in zip(axes, panels):
        f, t, s = _spec_db(p)
        ax.pcolormesh(t, f / 1000, s, vmin=-90, vmax=0, shading="auto", cmap="magma")
        ax.axhline(7.2, color="cyan", lw=1, ls="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (kHz)")
    axes[2].text(0.1, 7.45, "7.2 kHz wall", color="cyan", fontsize=9)
    fig.suptitle("Spectrograms: brightness = energy at that pitch and time (first 4 s)")
    fig.tight_layout()
    fig.savefig(OUT / "spectrograms.png", dpi=120)
    plt.close(fig)


def layer_curves() -> None:
    models = {
        "XLS-R 300M (M1b v3)": "models/m1_wav2vec2-xls-r-300m_L7_20260926-0521/meta.json",
        "WavLM Large": "models/m1_wavlm-large_L9_20260926-1906/meta.json",
    }
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label, p in models.items():
        cv = json.loads((REPO / p).read_text())["cv_by_layer"]
        layers = sorted(int(k) for k in cv)
        ax.plot(layers, [cv[str(k)]["min_dcf"] for k in layers], marker="o", ms=3, label=label)
    ax.set_xlabel("layer (0 = input to the transformer, 24 = top)")
    ax.set_ylabel("inner-CV minDCF (lower is better)")
    ax.set_title("Where in each backbone the real-vs-fake information lives")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "layer_curves.png", dpi=120)
    plt.close(fig)


def score_distributions() -> None:
    labels = pd.read_csv(REPO / "splits/nsa_folds.csv", usecols=["path", "label"])
    cols = {
        "M1b (XLS-R probe)": "m1b_v3",
        "Handcrafted v5": "handcrafted_v5",
        "M5 (XLS-R head)": "m5_xlsr_ft",
        "Spectra-AASIST": "spectra_aasist",
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (title, name) in zip(axes, cols.items()):
        d = pd.read_csv(REPO / f"outputs/detector_scores/{name}.csv")
        d = d[d.split == "holdout"].merge(labels, on="path")
        lo, hi = np.percentile(d.logit, [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        ax.hist(d[d.label == "bonafide"].logit, bins=bins, alpha=0.6, label="real", density=True)
        ax.hist(d[d.label == "spoof"].logit, bins=bins, alpha=0.6, label="fake", density=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("logit (right = more synthetic)")
        ax.set_yticks([])
    axes[0].legend()
    fig.suptitle("Outer holdout (3,858 clips): how each detector spreads real and fake")
    fig.tight_layout()
    fig.savefig(OUT / "score_distributions.png", dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spectrograms()
    layer_curves()
    score_distributions()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))}")


if __name__ == "__main__":
    main()
