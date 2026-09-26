"""Build the M5 audio bundle from outputs/manifests/m5_manifest.csv and nsa_test.csv.

Layout under --out (default: the external drive, bundle/v1):
  core/<id>.flac  extra/<id>.flac  test/<id>.flac
  manifest.csv       training rows + file, duration, raw_duration, peak, lead_s, pcm_sha256
  test/manifest.csv  1,671 rows in nsa_test.csv order (filename, path, id, file, ...)
  errors.csv  diagnostics.json  code.tgz  config_sha.txt  bundle_meta.json  TREE_SHA

Rules: core/test rows are never dropped (a decode error there aborts); extras under --min-extra-s
are dropped and counted; extra decode failures abort above --max-error-frac. ASV19 rows are
restricted to >= 2.0 s after trimming and spoof is duration-matched to bona fide (0.25 s bins,
at most 0.5x per bin). Cross-scope PCM-hash collisions abort. Resumable: existing FLACs that
open with the right frame count are reused.

Usage: uv run python scripts/m5_build_bundle.py [--workers 6] [--limit 200]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hearsay import SR
from hearsay.m5_bundle import clip_ok, tree_sha, write_clip, write_meta
from hearsay.m5_data import FOLDS, FOLDS_PLUS, shortcut_aucs

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/m5_bundle/v1")
XLSR = REPO / "weights" / "wav2vec2-xls-r-300m"


def _job(args):
    src, dst, min_s = args
    try:
        return write_clip(src, dst, min_s=min_s)
    except (OSError, RuntimeError, ValueError) as e:  # never lose a row silently
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def _run(jobs, workers: int, label: str):
    out, t0 = [], time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=16)):
            out.append(r)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {label}: {i + 1}/{len(jobs)} ({rate:.0f} clips/s)", flush=True)
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def make_code_tgz(out: Path) -> None:
    members = ["src/hearsay", "scripts", "splits/nsa_test_durations.csv", "splits/nsa_folds.csv",
               "splits/nsa_folds_plus_asv19.csv", "pyproject.toml", "uv.lock"]  # fmt: skip
    with tarfile.open(out / "code.tgz", "w:gz") as tar:
        for m in members:
            p = REPO / m
            if p.is_dir():
                for q in sorted(p.rglob("*")):
                    if q.is_file() and "__pycache__" not in q.parts and q.suffix in {".py", ".sh"}:
                        tar.add(q, arcname=str(q.relative_to(REPO)))
            elif p.exists():
                tar.add(p, arcname=m)
        for name in ("config.json", "preprocessor_config.json"):
            tar.add(XLSR / name, arcname=f"weights/wav2vec2-xls-r-300m/{name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=REPO / "outputs/manifests/m5_manifest.csv")
    ap.add_argument("--test-manifest", type=Path, default=REPO / "outputs/manifests/nsa_test.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="first N training rows + N test rows (smoke)")
    ap.add_argument("--min-extra-s", type=float, default=0.5)
    ap.add_argument("--asv-min-s", type=float, default=2.0)
    ap.add_argument("--asv-spoof-ratio", type=float, default=0.5)
    ap.add_argument("--max-error-frac", type=float, default=0.005)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.nice(args.nice)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest)
    t = pd.read_csv(args.test_manifest)
    if args.limit:
        m, t = m.iloc[: args.limit], t.iloc[: args.limit]
    t = t.assign(id="test_" + t.filename.str.replace(".wav", "", regex=False),
                 abs_path=[p if Path(p).is_absolute() else str(REPO / p) for p in t.path])

    # --- decode ---
    def plan(rows, sub, min_s_fn):
        jobs, skip = [], []
        for r in rows.itertuples():
            dst = out / sub / f"{r.id}.flac"
            if clip_ok(dst):
                skip.append(r.Index)
            src = getattr(r, "abs_path", None) or r.path
            jobs.append((src, dst, min_s_fn(r)))
        return jobs, skip

    core_mask = m.train_scope == "core"

    def min_s(r):
        if r.train_scope == "core":
            return None
        return args.asv_min_s if r.train_scope == "extra_asv19" else args.min_extra_s

    t0 = time.time()
    train_jobs, _ = plan(m, "", lambda r: None)  # placeholder to size
    train_jobs = [(src, out / ("core" if c else "extra") / f"{i}.flac", ms)
                  for (src, _, _), c, i, ms in zip(train_jobs, core_mask, m.id, [min_s(r) for r in m.itertuples()])]  # fmt: skip
    todo = [j for j in train_jobs if not clip_ok(j[1])]
    print(f"training rows: {len(m)} ({len(train_jobs) - len(todo)} reused)")
    res_train = {}
    done = _run(todo, args.workers, "train")
    for j, r in zip(todo, done):
        res_train[j[1].name] = r
    for j in train_jobs:
        if j[1].name not in res_train:  # reused: recompute stats cheaply from the file
            res_train[j[1].name] = {"file": j[1].name, "reused": True}
    test_jobs = [(ap_, out / "test" / f"{i}.flac", None) for ap_, i in zip(t.abs_path, t.id)]
    todo_t = [j for j in test_jobs if not clip_ok(j[1])]
    res_test = dict(zip([j[1].name for j in todo_t], _run(todo_t, args.workers, "test")))
    print(f"decode done in {time.time() - t0:.0f}s")

    # --- reused rows need stats: recompute from the FLAC (fast, no ffmpeg) ---
    import hashlib

    from hearsay.m5_bundle import leading_silence_s, read_clip

    def stats_from_file(p: Path) -> dict:
        x = read_clip(p)
        pcm16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        return {"file": p.name, "duration": round(x.size / SR, 4), "raw_duration": np.nan,
                "peak": round(float(np.abs(x).max()), 5), "lead_s": round(leading_silence_s(x), 4),
                "pcm_sha256": hashlib.sha256(pcm16.tobytes()).hexdigest()}  # fmt: skip

    for name, r in list(res_train.items()):
        if r.get("reused"):
            sub = "core" if name in {f"{i}.flac" for i in m.id[core_mask]} else "extra"
            res_train[name] = stats_from_file(out / sub / name)
    for j in test_jobs:
        if j[1].name not in res_test:
            res_test[j[1].name] = stats_from_file(j[1])

    # --- assemble manifests, enforce the drop rules ---
    stats = pd.DataFrame([res_train[f"{i}.flac"] for i in m.id])
    mm = pd.concat([m.reset_index(drop=True), stats], axis=1)
    errors = mm[mm.get("error", pd.Series(np.nan, index=mm.index)).notna()]
    dropped = mm[mm.get("dropped", pd.Series(np.nan, index=mm.index)).notna()]
    if (errors.train_scope == "core").any():
        print(errors[errors.train_scope == "core"][["path", "error"]].head(), file=sys.stderr)
        sys.exit("FATAL: decode error on a core row (core rows are never dropped)")
    n_extra = int((~core_mask).sum())
    if n_extra and len(errors) / n_extra > args.max_error_frac:
        sys.exit(f"FATAL: {len(errors)} extra decode errors > {args.max_error_frac:.1%}")
    errors.to_csv(out / "errors.csv", index=False)
    keep = mm[mm.file.notna()].copy()
    print(f"dropped short extras: {len(dropped)}; decode errors (extras): {len(errors)}")

    # ASV19: >= asv_min_s already enforced at write; duration-match spoof to bona fide per bin
    asv = keep.train_scope == "extra_asv19"
    if asv.any():
        rng = np.random.default_rng(args.seed)
        a = keep[asv]
        bins = (a.duration // 0.25).astype(int)
        drop_idx = []
        for b, d in a.groupby(bins):
            nb = int((d.label == "bonafide").sum())
            sp = d[d.label == "spoof"].index.to_numpy()
            cap = int(args.asv_spoof_ratio * nb)
            if len(sp) > cap:
                drop_idx.extend(rng.choice(sp, size=len(sp) - cap, replace=False))
        keep = keep.drop(index=drop_idx)
        print(f"ASV19 after matching: {keep[keep.train_scope == 'extra_asv19'].groupby('label').size().to_dict()}")

    # cross-scope PCM dedup: an extra row whose audio also exists in core/test is dropped (it
    # would leak a validation/test clip into training); a core<->test collision is reported
    # (the sponsor's sets may overlap; we cannot change either) but never hidden.
    tstats = pd.DataFrame([res_test[f"{i}.flac"] for i in t.id])
    tt = pd.concat([t.reset_index(drop=True), tstats], axis=1)
    if tt.file.isna().any() or ("error" in tt and tt.error.notna().any()):
        sys.exit("FATAL: a test row failed to decode")
    core_hashes = set(keep.loc[keep.train_scope == "core", "pcm_sha256"])
    test_hashes = set(tt.pcm_sha256)
    is_extra = keep.train_scope != "core"
    leak = is_extra & keep.pcm_sha256.isin(core_hashes | test_hashes)
    if leak.any():
        keep[leak][["id", "path", "train_scope"]].to_csv(out / "dropped_extra_collisions.csv",
                                                         index=False)
        keep = keep[~leak]
    overlap = keep[(keep.train_scope == "core") & keep.pcm_sha256.isin(test_hashes)]
    overlap[["id", "path", "label", "generator", "fold", "pcm_sha256"]].to_csv(
        out / "test_overlap.csv", index=False)
    dup_core = int(keep[keep.train_scope == "core"].pcm_sha256.duplicated().sum())
    print(f"extra rows dropped for colliding with core/test audio: {int(leak.sum())}; "
          f"core rows identical to a test clip: {len(overlap)} (test_overlap.csv); "
          f"duplicate audio within core: {dup_core}")

    cols = ["id", "path", "label", "generator", "speaker", "source", "group", "fold",
            "train_scope", "model_name", "file", "duration", "raw_duration", "peak", "lead_s",
            "pcm_sha256"]  # fmt: skip
    keep = keep[cols]
    keep.to_csv(out / "manifest.csv", index=False)
    tt[["filename", "path", "id", "file", "duration", "raw_duration", "peak", "lead_s",
        "pcm_sha256"]].to_csv(out / "test" / "manifest.csv", index=False)  # fmt: skip

    diag = {"shortcut_aucs_by_scope_all_training_rows": shortcut_aucs(keep),
            "n_rows": len(keep), "n_test": len(tt),
            "duration_by_scope_label": {f"{s}/{l}": {"median": round(float(d.duration.median()), 3),
                                                     "p90": round(float(d.duration.quantile(.9)), 3),
                                                     "n": len(d)}
                                        for (s, l), d in keep.groupby(["train_scope", "label"])}}  # fmt: skip
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(json.dumps(diag["shortcut_aucs_by_scope_all_training_rows"], indent=1))

    make_code_tgz(out)
    (out / "config_sha.txt").write_text(
        __import__("hashlib").sha256((XLSR / "config.json").read_bytes()).hexdigest())
    sha = tree_sha(out)
    (out / "TREE_SHA").write_text(sha)
    write_meta(out, tree_sha=sha, n_rows=len(keep), n_test=len(tt),
               folds_sha256=__import__("hearsay.m5_data", fromlist=["x"]).sha256_file(FOLDS),
               folds_plus_sha256=__import__("hearsay.m5_data", fromlist=["x"]).sha256_file(FOLDS_PLUS),
               git_sha=_git_sha(), xlsr_config_sha=(out / "config_sha.txt").read_text(),
               built_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               size_bytes=int(sum(p.stat().st_size for p in out.rglob("*.flac"))))  # fmt: skip
    print(f"bundle {out}: {len(keep)} training rows, {len(tt)} test rows, tree {sha[:12]}, "
          f"{sum(p.stat().st_size for p in out.rglob('*.flac')) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
