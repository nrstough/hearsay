"""Score an AntiDeepfake backbone over our populations. V2-53 stage 2.

Run spec `docs/specs/2026-10-11_v2-backbone.md`. Structurally mirrors `scripts/score_spectra.py`
(M3), which is this project's precedent for running an off-the-shelf detector: same exit-code
map, same config-hash + partial + sidecar resume, same per-split failure gate on integer counts,
same direction gate before anything is published, and the same `run(argv, loader=...)` injection
that lets the whole test suite run with a fake model and no weights.

Deliberately **not** copied from it: the pad-mode axis and the 64,600-sample window (AASIST's
fixed input forces those; a wav2vec2 encoder has no such constraint, and measured, variable real
lengths are *cheaper* than padding -- 0.103 s vs 0.158 s per clip), `m1_disagreement`, the
hardcoded model-card threshold, and publishing to `outputs/detector_scores/`, which is read-only
for this lane.

Two input views, both recorded per clip (`D5`, `I8`):

* **ours** -- `hearsay.embed.prepare_segment`: band match at 7.25 kHz, trim, 8 s cap. The
  primary, for comparability with v1.
* **native** -- the model's own recipe: per-clip standardisation only, no band match. The view
  the published reference numbers were produced under, so `P3b` corroborates against this one.

`logit` is `synth_logit = logit_fake - logit_real` and increases with synthetic likelihood. That
ordering is **verified on labelled rows by the direction gate before any export is written**,
never assumed, and nothing is ever flipped silently.

Exit codes: 0 ok · 2 usage/config · 3 failure gate · 4 direction gate · 5 --max-chunks.

Usage:
  export HF_HOME="/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/hf"
  uv run python scripts/v2_backbone_score.py --model mms-300m-anti-deepfake \
      --population itw --limit 40          # pilot, publishes nothing
  uv run python scripts/v2_backbone_score.py --model mms-300m-anti-deepfake --population itw
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hearsay.antideepfake import (
    file_sha256,
    load_antideepfake,
    mps_reservation,
    synth_logit,
)
from hearsay.antideepfake_eval import readings, refuse_degenerate

OUT = REPO / "outputs" / "v2" / "backbone"
BASE_COLS = ["path", "fold", "split", "score", "logit"]
VIEWS = ("ours", "native")
FAIL_RATE = 0.05
#: Hashed into the config id, so a code change invalidates a resume rather than mixing settings.
SOURCES = ("src/hearsay/antideepfake.py", "src/hearsay/antideepfake_eval.py", "src/hearsay/embed.py")

#: population -> (manifest, split label, has_labels). Paths are emitted **verbatim** from the
#: manifest: the canonical export writes inner/holdout absolute (from the fold file) and
#: test/itw relative (from their own manifests), and a naive re-spelling drops 81% of rows on
#: the join because three `data/` directories are symlinks.
POPULATIONS = {
    "itw": ("outputs/manifests/itw_stress.csv", "itw", True),
    "inner": ("splits/nsa_folds.csv", "inner_oof", True),
    "holdout": ("splits/nsa_folds.csv", "holdout", True),
    "test": ("outputs/manifests/nsa_test.csv", "test", False),
    # built by this lane under its own dir: `outputs/manifests/` is read-only here
    "asv19_control": ("outputs/v2/backbone/asv19_control.csv", "inner_oof", True),
    # the pre-declared in-sample arm (H1): committed before any score was read
    "inner_sample": ("outputs/v2/backbone/inner_sample.csv", "inner_oof", True),
    "mlaad": ("outputs/manifests/mlaad_en.csv", "itw", True),
}


def lane_path(name: str) -> Path:
    """A path under this lane's output dir, or refuse (`E5`).

    `OUT / name` is not enough: an absolute or `..`-bearing `name` escapes silently. Every
    write goes through here so the containment is enforced rather than assumed.
    """
    p = (OUT / name).resolve()
    if not str(p).startswith(str(OUT.resolve()) + os.sep):
        raise ValueError(f"refusing to write outside the lane dir: {p}")
    return p


def die(msg: str, code: int = 2):
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="checkpoint dir name, or a path")
    ap.add_argument("--population", required=True, choices=sorted(POPULATIONS))
    ap.add_argument("--repo-root", type=Path, default=REPO)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--limit", type=int, default=0, help="pilot: score N rows and publish nothing")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--chunk", type=int, default=100)
    # 4 is this lane's CPU cap on the 18 GB Mac; decode is I/O plus a C resampler, so it
    # overlaps the MPS forward without competing for the device.
    ap.add_argument("--decode-workers", type=int, default=4)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--seed", type=int, default=200, help="crop seed; ITW's recorded seed is 200")
    ap.add_argument("--no-lock", action="store_true", help="tests only: skip the MPS reservation")
    return ap.parse_args(argv)


def resolve_weights(model: str) -> tuple[Path, dict | None]:
    """Locate a checkpoint through its identity record, falling back to a path.

    Going through the record is the point: it is what binds a score file to the checkpoint that
    produced it, and the loader re-hashes the bytes against it.
    """
    rec_path = OUT / f"{model}.identity.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        if not rec.get("dry_run"):
            return Path(rec["realpath"]), rec
    p = Path(model)
    if (p / "model.safetensors").exists():
        return p, None
    die(f"no identity record at {rec_path.name} and {p} is not a checkpoint dir")
    raise AssertionError  # unreachable; keeps type checkers quiet


def build_rows(args, root: Path) -> pd.DataFrame:
    """Rows for one population, with `path` exactly as the manifest spells it."""
    rel, split, labelled = POPULATIONS[args.population]
    man = root / rel
    if not man.exists():
        die(f"{man} absent")
    df = pd.read_csv(man)
    if args.population in ("inner", "holdout"):
        if "fold" not in df.columns:
            die(f"{man} has no fold column")
        df = df[df.fold == "holdout"] if args.population == "holdout" else df[df.fold != "holdout"]
    df = df.reset_index(drop=True)
    if "fold" not in df.columns:
        df["fold"] = ""
    if "label" not in df.columns:
        df["label"] = ""
    df["split"] = split
    if not df.path.is_unique:
        die(f"{man}: duplicate paths; the run is keyed by path")
    if labelled and df.label.eq("").all():
        die(f"{man}: population {args.population} is declared labelled but has no labels")
    return df


def config_hash(args, root: Path, weights: Path) -> str:
    """Identity of this run's settings and code. A resume whose hash differs is refused rather
    than silently mixing two configurations -- the `score_spectra.py` discipline."""
    h = hashlib.sha256()
    for k in ("model", "population", "views", "dtype", "seed", "chunk"):
        h.update(f"{k}={getattr(args, k)}\n".encode())
    rel, _, _ = POPULATIONS[args.population]
    h.update(file_sha256(root / rel).encode())
    for s in SOURCES:
        p = root / s
        if p.exists():
            h.update(file_sha256(p).encode())
    h.update(file_sha256(Path(__file__)).encode())
    for f in ("config.json", "model.safetensors"):
        q = weights / f
        if q.exists():
            h.update(f"{f}:{q.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def prepare_views(x: np.ndarray, views, seed: int | None = None) -> dict[str, np.ndarray]:
    """The two declared input views for one decoded clip."""
    from hearsay.embed import prepare_segment

    out = {}
    if "ours" in views:
        out["ours"] = prepare_segment(x, band_match=True)
    if "native" in views:
        # the model's own recipe: no band match, 8 s cap only
        from hearsay import SR
        from hearsay.embed import MAX_SEGMENT_S

        out["native"] = np.ascontiguousarray(x[: int(MAX_SEGMENT_S * SR)], dtype=np.float32)
    return out


def model_device(model) -> torch.device | None:
    """The device a model's weights live on, or `None` for a parameterless model.

    `next(model.parameters())` raises `StopIteration` on a module with no parameters, which is
    exactly what the hermetic tests' fake detector is -- so the lookup must tolerate it rather
    than making the production path untestable.
    """
    p = next(model.parameters(), None)
    return None if p is None else p.device


def standardize(x: np.ndarray, device=None) -> torch.Tensor:
    """Per-clip zero-mean unit-variance, matching `Wav2Vec2FeatureExtractor` (eps 1e-7).

    Standardise on CPU, then move: the statistics are identical either way, and the transfer is
    one small copy per clip. `device` is **not** optional in practice -- the input must land on
    the model's device, and an earlier revision left it on CPU, which only appeared to work
    because the loader was also leaving the model on CPU.
    """
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[None]
    t = (t - t.mean()) / (t.var() + 1e-7).sqrt()
    return t if device is None else t.to(device)


@torch.inference_mode()
def score_one(model, views: dict[str, np.ndarray]) -> dict:
    """`synth_logit` per view for one clip, or a flag naming why it could not be scored."""
    from hearsay.antideepfake import check_min_length, receptive_field

    out = {}
    need = receptive_field(model.wav2vec2.config.conv_kernel, model.wav2vec2.config.conv_stride)
    for name, x in views.items():
        if x.size == 0:
            return {"flag": "empty_after_prepare"}
        if not np.isfinite(x).all():
            return {"flag": "non_finite_audio"}
        if np.ptp(x) == 0:
            return {"flag": "silent_or_zeros"}  # the decode_error-as-zeros failure mode
        if x.size < need:
            return {"flag": f"below_receptive_field_{need}"}
        check_min_length(model.wav2vec2.config, x.size)
        logits = model(standardize(x, device=model_device(model)))
        if tuple(logits.shape) != (1, 2):
            raise RuntimeError(f"model returned {tuple(logits.shape)}, expected (1, 2)")
        arr = logits[0].float().cpu().numpy()
        if not np.isfinite(arr).all():
            return {"flag": "non_finite_logits"}
        out[f"logit_{name}"] = synth_logit(arr)
    out["flag"] = ""
    return out


def _decode(path: str, repo_root: Path, views: tuple[str, ...], seed: int):
    """Decode and prepare one clip. Runs in a worker thread; raises are carried to the caller."""
    from hearsay.audio import load_audio

    full = Path(path) if Path(path).is_absolute() else repo_root / path
    return prepare_views(load_audio(full), views, seed)


def score_rows(model, rows: pd.DataFrame, args, done: dict) -> tuple[list[dict], int]:
    """Score every row, checkpointing after each chunk so an interrupted run resumes.

    Decoding runs in a small thread pool while the model stays on the main thread: measured,
    decode is **79 ms/clip against a 165 ms forward**, a third of the budget, and it is pure
    I/O plus a C-level resampler so it overlaps cleanly. The model is deliberately *not*
    shared across threads -- MPS work is issued from one thread only.
    """
    from concurrent.futures import ThreadPoolExecutor

    views = tuple(v for v in args.views.split(",") if v)
    results: list[dict] = []
    # set by `run()`; a direct call falls back rather than raising on a private attribute
    partial = getattr(args, "_partial", None) or lane_path(f"{args.out_name}.partial.csv")
    todo = [i for i in range(len(rows)) if rows.path[i] not in done]
    print(f"{len(todo)} to score, {len(done)} already done")
    meta = rows[["path", "fold", "split", "label"]].to_dict("records")  # avoid per-row .iloc

    with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
        for chunks, start in enumerate(range(0, len(todo), args.chunk)):
            if args.max_chunks and chunks >= args.max_chunks:
                return results, 5
            block = todo[start : start + args.chunk]
            futures = [
                pool.submit(_decode, meta[i]["path"], args.repo_root, views, args.seed) for i in block
            ]
            for i, fut in zip(block, futures, strict=True):
                rec = dict(meta[i])
                try:
                    rec.update(score_one(model, fut.result()))
                except FileNotFoundError:
                    rec["flag"] = "missing_file"
                except Exception as e:  # noqa: BLE001 - deliberate, see below
                    # One bad clip must never lose a row: the row is kept with a flag and
                    # counted, the way `safe_run` handles a crashing detector. A narrower
                    # except would let an unanticipated decoder failure abort a multi-hour run.
                    rec["flag"] = f"error:{type(e).__name__}"
                results.append(rec)
            _write_partial(partial, list(done.values()) + results)
            pct = 100 * (start + len(block)) / max(1, len(todo))
            print(f"  {pct:5.1f}%  {len(results)} scored  {time.strftime('%H:%M:%S')}", flush=True)
    return results, 0


def _write_partial(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    cols = sorted({k for r in rows for k in r})
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def direction_ok(frame: pd.DataFrame, col: str) -> dict:
    """AUC > 0.5 and the spoof median above the bona fide median, on labelled finite rows.

    `ok: None` on one class or no rows, and the caller treats anything but `True` as a failure --
    the `is not True` subtlety from M3, where `None` must also block the export.
    """
    from sklearn.metrics import roc_auc_score

    f = frame[frame.label.isin(["bonafide", "spoof"])]
    s = pd.to_numeric(f[col], errors="coerce").to_numpy(dtype=float)
    y = (f.label == "spoof").to_numpy(int)
    m = np.isfinite(s)
    if m.sum() == 0 or np.unique(y[m]).size < 2:
        return {"ok": None, "n": int(m.sum())}
    auc = float(roc_auc_score(y[m], s[m]))
    med_s, med_b = float(np.median(s[m & (y == 1)])), float(np.median(s[m & (y == 0)]))
    return {"ok": bool(auc > 0.5 and med_s > med_b), "auc": auc,
            "median_spoof": med_s, "median_bonafide": med_b, "n": int(m.sum())}  # fmt: skip


def run(argv=None, loader=load_antideepfake) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve()
    # Basename only. `--model` may be a path, and `OUT / "/abs/name"` resolves to `/abs/name`
    # in pathlib -- so a path-like out_name silently wrote every artifact OUTSIDE the lane
    # directory while reporting success. That is the E5 violation E5 exists to prevent.
    args.out_name = args.out_name or f"{Path(args.model).name}__{args.population}"
    args.out_name = Path(args.out_name).name
    weights, identity = resolve_weights(args.model)
    rows = build_rows(args, root)
    if args.limit:
        # A pilot must see both classes or the direction gate reads `ok: None` and refuses --
        # correct behaviour, but useless as a smoke test. Plain concat of per-label samples,
        # deliberately avoiding `groupby.apply`, whose `include_groups` signature moved between
        # pandas versions.
        if rows.label.isin(["bonafide", "spoof"]).any():
            per = max(1, args.limit // 2)
            parts = [g.sample(n=min(len(g), per), random_state=0) for _, g in rows.groupby("label")]
            rows = pd.concat(parts).reset_index(drop=True)
        else:
            rows = rows.head(args.limit).reset_index(drop=True)
    cfg_id = config_hash(args, root, weights)

    # A pilot's partial is namespaced: otherwise a 60-clip pilot leaves
    # `<name>__itw.partial.csv` behind, the next real run finds it, resumes from it, and then
    # refuses the whole run as "stale partial: configuration changed". That cost a real run.
    stem = f"{args.out_name}{'.pilot' if args.limit else ''}"
    partial = lane_path(f"{stem}.partial.csv")
    side = lane_path(f"{stem}.partial.json")
    done: dict = {}
    if args.resume:
        if not (partial.exists() and side.exists()):
            die("--resume needs both the partial and its sidecar")
        if json.loads(side.read_text()).get("config_hash") != cfg_id:
            die("stale partial: configuration changed; delete it or drop --resume")
        for r in csv.DictReader(partial.open()):
            if r.get("flag", "") in ("", "silent_or_zeros"):
                done[r["path"]] = r
        print(f"resuming with {len(done)} rows")
    elif partial.exists() and not args.limit:
        die(f"{partial.name} exists; pass --resume or delete it")
    OUT.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps({"config_hash": cfg_id, "model": args.model,
                                "population": args.population}, indent=1) + "\n")  # fmt: skip

    lock = lane_path("MPS.lock")
    ctx = (
        mps_reservation(lock, scope=f"score:{args.out_name}", wait_s=1800.0)
        if not args.no_lock else _nullctx()
    )
    args._partial = partial
    with ctx:
        model = loader(weights, device=args.device, dtype=getattr(torch, args.dtype), identity=identity)
        results, code = score_rows(model, rows, args, done)
    all_rows = list(done.values()) + results
    if code:
        _write_partial(partial, all_rows)
        print(f"stopped early (code {code}); partial kept")
        return code

    raw = pd.DataFrame(all_rows)
    raw = raw.set_index("path").loc[rows.path].reset_index()  # manifest order
    fails = raw.groupby("split").flag.apply(lambda c: int((c.fillna("") != "").sum()))
    totals = raw.groupby("split").size()
    over = {s: (int(fails[s]), int(totals[s])) for s in totals.index if fails[s] > FAIL_RATE * totals[s]}
    meta = {"model": args.model, "population": args.population, "config_hash": cfg_id,
            "weights": str(weights), "identity_bound": identity is not None,
            "dtype": args.dtype, "views": args.views, "n_rows": len(raw),
            "flags": raw.flag.fillna("").value_counts().to_dict(),
            "pilot": bool(args.limit)}  # fmt: skip
    if over:
        meta["gate"] = {"failure_rate": over}
        _publish_raw(args, raw, meta, pilot=bool(args.limit))
        print(f"FAILURE GATE: {over}; nothing published", file=sys.stderr)
        return 3

    primary = "logit_ours" if "ours" in args.views else f"logit_{args.views.split(',')[0]}"
    _, _, labelled = POPULATIONS[args.population]
    if labelled:
        d = direction_ok(raw, primary)
        meta["direction"] = d
        if d.get("ok") is not True:
            _publish_raw(args, raw, meta, pilot=bool(args.limit))
            print(f"DIRECTION GATE: {d}; nothing published", file=sys.stderr)
            return 4
        y = (raw.label == "spoof").to_numpy(int)
        s = pd.to_numeric(raw[primary], errors="coerce").to_numpy(float)
        try:
            refuse_degenerate(y, s)
        except ValueError as e:
            meta["gate"] = {"degenerate": str(e)}
            _publish_raw(args, raw, meta, pilot=bool(args.limit))
            print(f"DEGENERATE: {e}; nothing published", file=sys.stderr)
            return 3
        meta["readings"] = readings(y, s)

    if args.limit:
        _publish_raw(args, raw, meta, pilot=True)
        print(json.dumps({k: meta[k] for k in ("flags", "direction", "readings") if k in meta}, indent=1))
        print("pilot: nothing published")
        return 0

    _publish_raw(args, raw, meta, pilot=bool(args.limit))
    _publish_export(args, raw, primary)
    partial.unlink(missing_ok=True)
    side.unlink(missing_ok=True)
    print(f"published {args.out_name}: {meta.get('readings', {})}")
    return 0


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _publish_raw(args, raw: pd.DataFrame, meta: dict, pilot: bool = False) -> None:
    stem = f"{args.out_name}{'.pilot' if pilot else ''}"
    raw.to_csv(lane_path(f"{stem}.raw.csv"), index=False)
    lane_path(f"{stem}.meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")


def _publish_export(args, raw: pd.DataFrame, primary: str) -> None:
    """The fusion contract: `path, fold, split, score, logit`, plus `flag` for integrity.

    `score = sigmoid(logit)` and nothing is calibrated (`D10`). Paths are the manifest's own
    spelling, so a join against the other exports matches row for row.
    """
    from hearsay.metrics import sigmoid

    logit = pd.to_numeric(raw[primary], errors="coerce").to_numpy(float)
    with np.errstate(over="ignore"):
        score = sigmoid(logit)
    out = pd.DataFrame({"path": raw.path, "fold": raw.fold, "split": raw.split,
                        "score": score, "logit": logit, "flag": raw.flag.fillna("")})  # fmt: skip
    if not out.path.is_unique:
        raise ValueError("duplicate paths in the export")
    out[[*BASE_COLS, "flag"]].to_csv(lane_path(f"{args.out_name}.csv"), index=False)


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
