"""Re-embed rows that an extraction flagged `decode_error`, and verify a finished embedding set.

A segment-mode extraction replaces a clip that FFmpeg failed to decode with 1 s of zeros and
flags the row (shard `flag` and manifest.csv; `hearsay.probe.load_embeddings` does not read the
flag). Under heavy parallel load (three WavLM extractions sharing the 18 GB Mac, Sat Sep 26)
FFmpeg failed transiently on files that decode fine on retry. Both modes re-run exactly the
extractor's segment path (scripts/extract_embeddings.py): the crop length stored in the shard,
seed = extract_meta seed + row, prepare_segment (band match on) -> embed_segment -> fp16.

repair (default): refuses unless the set is finished (manifest.csv present, shards cover every
row) and --manifest lists the same paths. Identity check first: --verify unflagged rows (drawn
from the flagged shards first) are re-embedded and must match their stored values element-wise
(|diff| <= ATOL + RTOL*|stored|), else nothing is written. Flagged rows are then re-embedded; a
row that still fails to decode stays flagged. Shards are rewritten atomically; manifest.csv's
flag column is rebuilt from the shards.

--verify-only: writes nothing. Checks the set against its twin (the same manifest extracted by
another backbone, e.g. the XLS-R v3 set): equal extract_meta.json, paths in the same order,
equal stored crop lengths; 0 decode errors; finite embeddings; and re-embeds --per-shard random
rows of every shard plus any --rows, element-wise as above. Prints one JSON line.

Usage (only after the extraction of <name> has finished):
  uv run python scripts/repair_decode_errors.py --model wavlm-large --name nsa_test_wl \
      --manifest outputs/manifests/nsa_test.csv [--verify 5]
  uv run python scripts/repair_decode_errors.py --model wavlm-large --name nsa_test_wl \
      --manifest outputs/manifests/nsa_test.csv --verify-only \
      --twin outputs/embeddings/wav2vec2-xls-r-300m/nsa_test_v3 [--per-shard 10] [--rows 1141,1143]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay.audio import DecodeError, load_audio
from hearsay.embed import embed_segment, load_backbone, prepare_segment

REPO = Path(__file__).resolve().parents[1]
RTOL = ATOL = 5e-3  # per element; fp16 spacing is ~1e-3 relative, stored values reach ~500


def embed_row(model, path: str, crop_s: float, seed: int) -> np.ndarray:
    """The extractor's segment path for one row (crop_s NaN = no crop)."""
    crop = None if np.isnan(crop_s) else float(crop_s)
    return embed_segment(model, prepare_segment(load_audio(path), crop, seed)).astype(np.float16)


def excess(e: np.ndarray, stored: np.ndarray) -> float:
    """Worst per-element |diff| - (ATOL + RTOL*|stored|); <= 0 means within tolerance."""
    e32, s32 = e.astype(np.float32), stored.astype(np.float32)
    return float(np.max(np.abs(e32 - s32) - (ATOL + RTOL * np.abs(s32))))


def _finished_set(out: Path, manifest: pd.DataFrame) -> tuple[dict, list[Path]]:
    meta = json.loads((out / "extract_meta.json").read_text())
    assert meta["mode"] == "segment", "repair supports segment-mode extractions only"
    own = out / "manifest.csv"
    assert own.exists(), f"{own} missing: the extraction has not finished"
    assert list(pd.read_csv(own).path) == list(manifest.path), \
        "--manifest does not list the same paths as the set's manifest.csv"  # fmt: skip
    shards = sorted(out.glob("shard_*.npz"))
    rows = np.concatenate([np.load(p)["row"] for p in shards]) if shards else np.array([], int)
    assert np.array_equal(np.sort(rows), np.arange(len(manifest))), \
        "shards do not cover every manifest row exactly once: the extraction has not finished"  # fmt: skip
    return meta, shards


def _check(model, embed, manifest, seed0, z, i) -> dict:
    r = int(z["row"][i])
    e = embed(model, manifest.path.iloc[r], float(z["crop_s"][i]), seed0 + r)
    diff = np.abs(e.astype(np.float32) - z["emb"][i].astype(np.float32))
    return {"row": r, "excess": excess(e, z["emb"][i]), "max_abs_diff": float(diff.max())}


def _savez_atomic(path: Path, z: dict) -> None:
    # Named outside the shard_*.npz glob, so a crash never leaves a second copy of the rows.
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **z)
    os.replace(tmp, path)


def repair(model, out: Path, manifest: pd.DataFrame, n_verify: int = 5, embed=embed_row) -> dict:
    assert n_verify >= 1, "the identity check needs at least one row"
    meta, shards = _finished_set(out, manifest)
    seed0 = int(meta["seed"])
    flags = {p: np.load(p)["flag"] for p in shards}
    flagged = [p for p in shards if (flags[p] == "decode_error").any()]

    # Identity check on unflagged rows, drawn from the flagged shards first.
    rng = np.random.default_rng(0)
    cands = []
    for group in (flagged, [p for p in shards if p not in flagged]):
        pool = [(p, i) for p in group for i in np.flatnonzero(flags[p] != "decode_error")]
        take = min(n_verify - len(cands), len(pool))
        cands += (
            [pool[j] for j in sorted(rng.choice(len(pool), take, replace=False))] if take else []
        )
    assert len(cands) == n_verify, (
        f"only {len(cands)} unflagged rows to verify against, need {n_verify}"
    )
    loaded = {}
    checks = []
    for p, i in cands:
        if p not in loaded:
            loaded[p] = dict(np.load(p))
        z = loaded[p]
        checks.append(_check(model, embed, manifest, seed0, z, i))
    worst = max(c["excess"] for c in checks)
    assert worst <= 0, f"re-embedding does not reproduce stored rows (worst excess {worst:.3g})"

    fixed, still = [], []
    for p in flagged:
        if p not in loaded:
            loaded[p] = dict(np.load(p))
        z = loaded[p]
        for i in np.flatnonzero(z["flag"] == "decode_error"):
            r = int(z["row"][i])
            try:
                z["emb"][i] = embed(model, manifest.path.iloc[r], float(z["crop_s"][i]), seed0 + r)
                z["flag"][i] = ""
                fixed.append(r)
            except DecodeError:
                still.append(r)
        _savez_atomic(p, z)

    m = pd.read_csv(out / "manifest.csv")
    col = np.empty(len(m), dtype=object)
    for p in shards:
        z = np.load(p)
        col[z["row"]] = [str(f) for f in z["flag"]]
    m["flag"] = pd.Series(col).fillna("").astype(str)
    m.to_csv(out / "manifest.csv", index=False)
    return {"fixed": fixed, "still_failing": still, "verified_rows": [c["row"] for c in checks],
            "verify_worst_excess": worst, "verify_max_abs_diff": max(c["max_abs_diff"] for c in checks)}  # fmt: skip


def verify_set(model, out: Path, manifest: pd.DataFrame, twin: Path | None = None,
               per_shard: int = 10, rows: tuple[int, ...] = (), embed=embed_row) -> dict:  # fmt: skip
    """Read-only checks of a finished set (A1-A5 of docs/specs/2026-09-26_post-draft-heads.md)."""
    meta, shards = _finished_set(out, manifest)
    seed0 = int(meta["seed"])
    res: dict = {"n_rows": len(manifest), "n_shards": len(shards)}
    zs = [dict(np.load(p)) for p in shards]
    order = np.argsort(np.concatenate([z["row"] for z in zs]))
    crop = np.concatenate([z["crop_s"] for z in zs])[order]
    res["decode_errors_shards"] = int(sum((z["flag"] == "decode_error").sum() for z in zs))
    res["decode_errors_manifest"] = int(
        (pd.read_csv(out / "manifest.csv").flag.fillna("") == "decode_error").sum()
    )
    res["emb_shape"] = [len(manifest), *zs[0]["emb"].shape[1:]]
    res["finite"] = bool(all(np.isfinite(z["emb"].astype(np.float32)).all() for z in zs))
    if twin is not None:
        res["meta_equal_twin"] = meta == json.loads((twin / "extract_meta.json").read_text())
        res["paths_equal_twin"] = list(pd.read_csv(twin / "manifest.csv").path) == list(
            manifest.path
        )
        tz = [np.load(p) for p in sorted(twin.glob("shard_*.npz"))]
        tcrop = np.concatenate([z["crop_s"] for z in tz])[
            np.argsort(np.concatenate([z["row"] for z in tz]))
        ]
        res["crop_equal_twin"] = bool(
            len(tcrop) == len(crop) and np.array_equal(tcrop, crop, equal_nan=True)
        )
    rng = np.random.default_rng(0)
    where = {int(r): (k, i) for k, z in enumerate(zs) for i, r in enumerate(z["row"])}
    picks = {(k, int(i)) for k, z in enumerate(zs)
             for i in rng.choice(len(z["row"]), min(per_shard, len(z["row"])), replace=False)}  # fmt: skip
    picks |= {where[int(r)] for r in rows}
    checks = [_check(model, embed, manifest, seed0, zs[k], i) for k, i in sorted(picks)]
    assert checks, "nothing to re-embed: raise --per-shard or pass --rows"
    checked = {c["row"] for c in checks}
    res["n_identity_checked"] = len(checks)
    res["identity_worst_excess"] = max(c["excess"] for c in checks)
    res["identity_max_abs_diff"] = max(c["max_abs_diff"] for c in checks)
    res["explicit_rows_checked"] = sorted(int(r) for r in rows if int(r) in checked)
    res["ok"] = bool(res["decode_errors_shards"] == 0 and res["decode_errors_manifest"] == 0 and res["finite"]
                     and res["identity_worst_excess"] <= 0
                     and all(res.get(k, True) for k in ("meta_equal_twin", "paths_equal_twin", "crop_equal_twin")))  # fmt: skip
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument(
        "--verify", type=int, default=5, help="repair: unflagged rows re-embedded first"
    )
    ap.add_argument("--verify-only", action="store_true", help="check the set; write nothing")
    ap.add_argument(
        "--twin", type=Path, help="verify-only: the same manifest under another backbone"
    )
    ap.add_argument(
        "--per-shard", type=int, default=10, help="verify-only: rows re-embedded per shard"
    )
    ap.add_argument(
        "--rows", default="", help="verify-only: extra rows to re-embed, comma-separated"
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out = REPO / "outputs" / "embeddings" / args.model / args.name
    manifest = pd.read_csv(args.manifest)
    model = load_backbone(args.model, args.device)
    if args.verify_only:
        rows = tuple(int(r) for r in args.rows.split(",") if r)
        twin = args.twin if args.twin is None or args.twin.is_absolute() else REPO / args.twin
        res = verify_set(model, out, manifest, twin, args.per_shard, rows)
        print(json.dumps({"set": args.name, **res}))
        raise SystemExit(0 if res["ok"] else 1)
    res = repair(model, out, manifest, args.verify)
    print(json.dumps({"set": args.name, "n_fixed": len(res["fixed"]), **res}))


if __name__ == "__main__":
    main()
