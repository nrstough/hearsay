#!/usr/bin/env python3
"""Cross-platform decode identity for the image checks (run spec T7, tests P2).

`pcm-hash DIR --template T [--n N] --out hashes.json` decodes each named file through
hearsay.audio.load_audio (the same FFmpeg command the detectors use) via scripts/hash_audio.hash_one
and records the sha256 of the decoded float32 PCM. Run it on the Mac and inside the image
(`--entrypoint python`), then `pcm-diff a.json b.json` lists any file whose decoded samples differ:
a non-empty list means the image's ffmpeg (5.1.9 in bookworm) does not decode like the Mac's
(7.1.1), and score parity cannot be attributed to torch alone.

Score parity itself is scripts/run_pipeline.py --compare-tsv (Spearman, max |diff|).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _hash_one():
    spec = importlib.util.spec_from_file_location("hash_audio", REPO / "scripts" / "hash_audio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.hash_one


def template_ids(path: Path) -> list[str]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows or "filename" not in rows[0]:
        raise SystemExit(f"{path}: no tab-separated 'filename' column")
    return [r["filename"] for r in rows]


def pcm_hash(data: Path, template: Path | None, n: int | None, out: Path) -> dict:
    hash_one = _hash_one()
    if template is not None:
        ids = template_ids(template)
    else:
        ids = sorted(p.name for p in data.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
    if n:
        ids = ids[:n]
    if not ids:
        raise SystemExit("pcm-hash: no files")
    res = {}
    for i in ids:
        _, file_sha, pcm_sha, n_samples = hash_one(str(data / i))
        res[i] = {"file_sha256": file_sha, "pcm_sha256": pcm_sha, "n_samples": n_samples}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    n_fail = sum(1 for v in res.values() if not v["pcm_sha256"])
    print(f"pcm-hash: {len(res)} files -> {out} ({n_fail} decode failures)")
    return res


def pcm_diff(a: Path, b: Path) -> list[str]:
    ha, hb = json.loads(a.read_text()), json.loads(b.read_text())
    ids = sorted(set(ha) | set(hb))
    bad = []
    for i in ids:
        va, vb = ha.get(i), hb.get(i)
        if va is None or vb is None:
            bad.append(f"{i}: only in {'A' if vb is None else 'B'}")
        elif not va["pcm_sha256"] or not vb["pcm_sha256"] or not va["n_samples"] or not vb["n_samples"]:
            fa = not va["pcm_sha256"] or not va["n_samples"]
            fb = not vb["pcm_sha256"] or not vb["n_samples"]
            where = "both" if fa and fb else ("A" if fa else "B")
            bad.append(f"{i}: decode failed ({where}); a failed decode never counts as a match")
        elif va["pcm_sha256"] != vb["pcm_sha256"] or va["n_samples"] != vb["n_samples"]:
            bad.append(f"{i}: pcm differs ({va['n_samples']} vs {vb['n_samples']} samples)")
    for line in bad:
        print(line)
    print(f"pcm-diff: {len(ids)} files, {len(bad)} differ")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("pcm-hash")
    h.add_argument("data", type=Path)
    h.add_argument("--template", type=Path)
    h.add_argument("--n", type=int)
    h.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("pcm-diff")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "pcm-hash":
        pcm_hash(args.data, args.template, args.n, args.out)
        return 0
    return 1 if pcm_diff(args.a, args.b) else 0


if __name__ == "__main__":
    sys.exit(main())
