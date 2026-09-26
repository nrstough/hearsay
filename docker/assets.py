#!/usr/bin/env python3
"""Shipped-asset manifest for the HEARSAY image (run spec D3, tests B10, R5).

`freeze` records sha256 and size of every regular file under the shipped model and weight
directories at build time; `verify` checks the tree at container start: every file present with
the recorded size, small files (<= --hash-over-mb) re-hashed, large ones re-hashed only with
--full, and no extra files in a shipped directory. Stdlib only: it runs in the image build
before the virtualenv exists. `__pycache__`, `.cache` directories and `*.pyc` are ignored on
both sides (Python and the Hub write them beside the weights).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".cache"}
SKIP_SUFFIXES = {".pyc"}
MB = 1024 * 1024


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(base: Path) -> list[Path]:
    out = []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(base).parts[:-1]):
            continue
        out.append(p)
    return out


def freeze(root: Path, dirs: list[str]) -> dict:
    files = []
    for d in dirs:
        base = root / d
        if not base.is_dir():
            raise SystemExit(f"assets freeze: {base} is not a directory")
        for p in _files(base):
            files.append({"path": p.relative_to(root).as_posix(), "size": p.stat().st_size,
                          "sha256": sha256_file(p)})
        if not any(f["path"].startswith(d.rstrip("/") + "/") for f in files):
            raise SystemExit(f"assets freeze: {base} holds no files")
    return {"created": datetime.now(UTC).isoformat(timespec="seconds"),
            "dirs": list(dirs), "files": files}


def verify(root: Path, manifest: dict, *, full: bool = False,
           hash_over_mb: float = 100.0, manifest_path: Path | None = None) -> list[str]:
    """Problems found (empty list = OK). Each entry names the file and the mismatch. The
    manifest file itself is never an "extra" file, wherever it lives."""
    problems: list[str] = []
    expected = {f["path"]: f for f in manifest["files"]}
    skip_rel = None
    if manifest_path is not None:
        try:
            skip_rel = manifest_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            skip_rel = None
    for rel, f in expected.items():
        p = root / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
            continue
        size = p.stat().st_size
        if size != f["size"]:
            problems.append(f"size mismatch: {rel} ({size} != {f['size']})")
            continue
        if full or size <= hash_over_mb * MB:
            got = sha256_file(p)
            if got != f["sha256"]:
                problems.append(f"sha256 mismatch: {rel}")
    for d in manifest.get("dirs", []):
        base = root / d
        if not base.is_dir():
            continue  # already reported as missing files
        for p in _files(base):
            rel = p.relative_to(root).as_posix()
            if rel not in expected and rel != skip_rel:
                problems.append(f"extra: {rel}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--root", type=Path, required=True)
    f.add_argument("--dirs", nargs="+", required=True)
    f.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--root", type=Path, required=True)
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--full", action="store_true", help="re-hash large files too")
    v.add_argument("--hash-over-mb", type=float, default=100.0)
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        m = freeze(args.root.resolve(), args.dirs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(m, indent=1))
        print(f"assets: froze {len(m['files'])} files from {len(m['dirs'])} dirs -> {args.out}")
        return 0
    m = json.loads(args.manifest.read_text())
    problems = verify(args.root.resolve(), m, full=args.full, hash_over_mb=args.hash_over_mb,
                      manifest_path=args.manifest)
    for p in problems:
        print(f"assets: {p}", file=sys.stderr)
    print(f"assets: {'OK' if not problems else f'{len(problems)} problem(s)'} "
          f"({len(m['files'])} files{', full hash' if args.full else ''})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
