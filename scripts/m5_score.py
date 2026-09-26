"""Score audio files with a saved M5 model on CPU: the Docker inference path.

Input: a manifest with a `path` column (or --files ...). Output: path, score, logit, flag.
Every file goes through hearsay.audio.load_audio -> hearsay.m5_data.deploy_transform (trim,
cap 8 s by default, normalize), fp32, batched by length with attention masks. Refuses a checkpoint whose
hashes.json does not match its files.

Usage: uv run python scripts/m5_score.py --model models/m5_xlsr_ft_<stamp>/model \
    --manifest outputs/manifests/nsa_test.csv --out outputs/m5_test_cpu.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hearsay.audio import DecodeError, load_audio
from hearsay.m5_data import DEPLOY_MAX_S, collate, deploy_transform
from hearsay.m5_model import load_m5, score_batch
from hearsay.metrics import sigmoid

REPO = Path(__file__).resolve().parents[1]


def score_paths(net, paths: list[str], batch: int = 8, device: str = "cpu",
                max_s: float = DEPLOY_MAX_S) -> tuple[np.ndarray, list[str]]:  # fmt: skip
    clips, flags = [], [""] * len(paths)
    for i, p in enumerate(paths):
        try:
            clips.append(deploy_transform(load_audio(p), max_s=max_s))
        except DecodeError as e:
            clips.append(None)
            flags[i] = f"decode_error:{str(e)[:60]}"
    logits = np.full(len(paths), np.nan)
    ok = [i for i, c in enumerate(clips) if c is not None]
    ok.sort(key=lambda i: -clips[i].size)
    for b in range(0, len(ok), batch):
        ii = ok[b : b + batch]
        xs, mask = collate([clips[i] for i in ii])
        logits[ii] = score_batch(net, xs, mask, device).numpy()
    return logits, flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-s", type=float, default=DEPLOY_MAX_S)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.manifest:
        m = pd.read_csv(args.manifest)
        paths = [p if Path(p).is_absolute() else str(REPO / p) for p in m.path]
    else:
        paths = list(args.files or [])
    if args.limit:
        paths = paths[: args.limit]
    t0 = time.time()
    net = load_m5(args.model)
    logits, flags = score_paths(net, paths, args.batch, max_s=args.max_s)
    df = pd.DataFrame({"path": paths, "score": sigmoid(logits), "logit": logits, "flag": flags})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(paths)} files in {time.time() - t0:.1f}s ({(time.time() - t0) / max(1, len(paths)):.3f} s/file); "
          f"failed {sum(1 for f in flags if f)}; mean score {np.nanmean(df.score):.3f}; -> {args.out}")


if __name__ == "__main__":
    main()
