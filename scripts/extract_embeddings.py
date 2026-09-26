"""Frozen SSL embeddings for M1 and the bake-off.

For each clip: load_audio -> 4 s windows (50% hop, first --max-windows) -> frozen backbone with
all hidden states -> mean-pool over time per layer -> average over windows. Stores every layer
(fp16) so layer choice is a probe-time decision, not a re-extraction.

Output: outputs/embeddings/<model>/<name>/shard_XXXXX.npz with
  emb (n, n_layers, dim) float16, row (n,) int  -- row indexes the input manifest
plus <name>/manifest.csv (the input manifest + n_windows + flag). Shards are skipped if present,
so an interrupted run resumes.

Usage:
  uv run python scripts/extract_embeddings.py --manifest outputs/manifests/asv19_train.csv \
      --name asv19_train [--model wav2vec2-xls-r-300m] [--limit 5000]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

from hearsay import SR
from hearsay.audio import DecodeError, load_audio, windows

REPO = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def embed_batch(model, wins: list[np.ndarray], device: str) -> np.ndarray:
    """(n_windows, n_layers, dim) float32: per-layer time-mean of each window."""
    x = torch.from_numpy(np.stack(wins)).to(device)
    hs = model(x, output_hidden_states=True).hidden_states
    return torch.stack([h.mean(dim=1) for h in hs], dim=1).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="CSV with a `path` column")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="wav2vec2-xls-r-300m", help="dir under weights/")
    ap.add_argument("--win-s", type=float, default=4.0)
    ap.add_argument("--max-windows", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16, help="windows per forward pass")
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    if args.limit:
        m = m.iloc[: args.limit]
    out = REPO / "outputs" / "embeddings" / args.model / args.name
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(REPO / "weights" / args.model).eval().to(args.device)
    win = int(args.win_s * SR)
    n_win_col, flag_col = np.zeros(len(m), int), [""] * len(m)
    t0, n_done = time.time(), 0

    for s0 in range(0, len(m), args.shard):
        shard_path = out / f"shard_{s0 // args.shard:05d}.npz"
        rows = list(range(s0, min(s0 + args.shard, len(m))))
        if shard_path.exists():
            z = np.load(shard_path)
            n_win_col[rows] = z["n_windows"]
            for r, f in zip(rows, z["flag"], strict=True):
                flag_col[r] = str(f)
            continue
        # Flatten all windows of the shard, forward in fixed-size batches, then re-group.
        owners, wins = [], []
        for r in rows:
            try:
                w = windows(load_audio(m.path.iloc[r]), win)[: args.max_windows]
            except DecodeError:
                flag_col[r] = "decode_error"
                w = np.zeros((1, win), dtype=np.float32)
            n_win_col[r] = len(w)
            owners += [r] * len(w)
            wins += list(w)
        feats = np.concatenate(
            [embed_batch(model, wins[i : i + args.batch], args.device)
             for i in range(0, len(wins), args.batch)]
        )  # fmt: skip
        owners = np.array(owners)
        emb = np.stack([feats[owners == r].mean(axis=0) for r in rows]).astype(np.float16)
        np.savez(
            shard_path, emb=emb, row=np.array(rows),
            n_windows=n_win_col[rows], flag=np.array([flag_col[r] for r in rows]),
        )  # fmt: skip
        n_done += len(rows)
        el = time.time() - t0
        print(f"  {rows[-1] + 1}/{len(m)}  {el:.0f}s  {el / n_done:.3f}s/clip", flush=True)

    m = m.assign(n_windows=n_win_col, flag=flag_col)
    m.to_csv(out / "manifest.csv", index=False)
    print(f"done: {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
