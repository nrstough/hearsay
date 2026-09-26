"""Re-embed rows that an extraction flagged `decode_error` and rewrite them in place.

A segment-mode extraction replaces a clip that FFmpeg failed to decode with 1 s of zeros and
flags the row. Under heavy parallel load (several extractions sharing the Mac, Sat Sep 26 WavLM
run) FFmpeg failed transiently on files that decode fine on retry. This tool re-runs exactly
the extractor's segment path for the flagged rows only: the crop length stored in the shard,
seed = extract_meta seed + row, prepare_segment -> embed_segment, fp16.

Identity check first: `--verify` unflagged rows of the same set are re-embedded the same way
and must match their stored values (fp16 tolerance), else nothing is written. A row that still
fails to decode stays flagged. manifest.csv's flag column is rewritten from the shards.

Usage (only after the extraction of <name> has finished):
  uv run python scripts/repair_decode_errors.py --model wavlm-large --name nsa_test_wl \
      --manifest outputs/manifests/nsa_test.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.audio import DecodeError, load_audio
from hearsay.embed import embed_segment, load_backbone, prepare_segment

REPO = Path(__file__).resolve().parents[1]


def embed_row(model, path: str, crop_s: float, seed: int) -> np.ndarray:
    crop = None if np.isnan(crop_s) else float(crop_s)
    return embed_segment(model, prepare_segment(load_audio(path), crop, seed)).astype(np.float16)


def repair(model, out: Path, manifest: pd.DataFrame, n_verify: int = 3, embed=embed_row) -> dict:
    meta = json.loads((out / "extract_meta.json").read_text())
    assert meta["mode"] == "segment", "repair supports segment-mode extractions only"
    seed0 = int(meta["seed"])
    shards = sorted(out.glob("shard_*.npz"))
    loaded = {p: dict(np.load(p)) for p in shards}

    # Identity check on unflagged rows (from the flagged shards when there are any).
    flagged = [p for p, z in loaded.items() if (z["flag"] == "decode_error").any()]
    pool = flagged or shards
    checks = []
    for p in pool:
        z = loaded[p]
        ok = np.flatnonzero(z["flag"] != "decode_error")[:n_verify]
        for i in ok:
            r = int(z["row"][i])
            e = embed(model, manifest.path.iloc[r], float(z["crop_s"][i]), seed0 + r)
            err = float(np.max(np.abs(e.astype(np.float32) - z["emb"][i].astype(np.float32))))
            scale = float(np.max(np.abs(z["emb"][i].astype(np.float32))))
            checks.append(err / max(scale, 1e-6))
        if len(checks) >= n_verify:
            break
    worst = max(checks) if checks else 0.0
    assert worst < 2e-3, f"re-embedding does not reproduce stored rows (rel err {worst:.2e})"

    fixed, still = [], []
    for p in flagged:
        z = loaded[p]
        for i in np.flatnonzero(z["flag"] == "decode_error"):
            r = int(z["row"][i])
            try:
                z["emb"][i] = embed(model, manifest.path.iloc[r], float(z["crop_s"][i]), seed0 + r)
                z["flag"][i] = ""
                fixed.append(r)
            except DecodeError:
                still.append(r)
        np.savez(p, **z)

    m = pd.read_csv(out / "manifest.csv") if (out / "manifest.csv").exists() else manifest.copy()
    flags = np.empty(len(m), dtype=object)
    for p in shards:
        z = np.load(p)
        flags[z["row"]] = [str(f) for f in z["flag"]]
    m["flag"] = pd.Series(flags).fillna("").astype(str)
    m.to_csv(out / "manifest.csv", index=False)
    return {"fixed": fixed, "still_failing": still, "verify_rel_err": worst, "n_verify": len(checks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--verify", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = REPO / "outputs" / "embeddings" / args.model / args.name
    model = load_backbone(args.model, args.device)
    res = repair(model, out, pd.read_csv(args.manifest), args.verify)
    print(json.dumps({"set": args.name, "n_fixed": len(res["fixed"]), **res}))


if __name__ == "__main__":
    main()
