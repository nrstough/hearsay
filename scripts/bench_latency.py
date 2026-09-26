"""Live-path latency microbenchmark (plan.md: 100 windows, 4 s, gate 750 ms p95).

Times one SSL front-end forward pass per 4 s, 16 kHz mono window on the chosen device,
synchronizing after each window so the number is wall-clock latency, not queue depth.
The pooling + linear head is negligible next to the backbone and is left out.

Usage: uv run python scripts/bench_latency.py [--model wav2vec2-xls-r-300m] [--device mps]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[1]
SR = 16_000
GATE_P95_MS = 750.0


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wav2vec2-xls-r-300m", help="dir name under weights/")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    dtype = torch.float16 if args.fp16 else torch.float32
    model = AutoModel.from_pretrained(REPO / "weights" / args.model, torch_dtype=dtype)
    model.eval().to(args.device)

    rng = np.random.default_rng(0)
    n_samples = int(args.window_s * SR)
    # Speech-like amplitude noise; content does not change the FLOP count.
    windows = [
        torch.from_numpy((0.1 * rng.standard_normal(n_samples)).astype(np.float32))
        for _ in range(args.warmup + args.n)
    ]

    times_ms = []
    with torch.inference_mode():
        for i, w in enumerate(windows):
            x = w.unsqueeze(0).to(args.device, dtype=dtype)
            sync(args.device)
            t0 = time.perf_counter()
            out = model(x).last_hidden_state.mean(dim=1)
            _ = out.float().cpu()
            sync(args.device)
            dt = (time.perf_counter() - t0) * 1000
            if i >= args.warmup:
                times_ms.append(dt)

    t = np.array(times_ms)
    res = {
        "model": args.model,
        "device": args.device,
        "dtype": str(dtype).removeprefix("torch."),
        "window_s": args.window_s,
        "n": len(t),
        "p50_ms": round(float(np.percentile(t, 50)), 1),
        "p95_ms": round(float(np.percentile(t, 95)), 1),
        "max_ms": round(float(t.max()), 1),
        "mean_ms": round(float(t.mean()), 1),
        "gate_p95_ms": GATE_P95_MS,
        "pass": bool(np.percentile(t, 95) <= GATE_P95_MS),
    }
    print(json.dumps(res))


if __name__ == "__main__":
    main()
