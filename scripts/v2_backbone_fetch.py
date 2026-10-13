"""Fetch an AntiDeepfake checkpoint, bind its identity, and archive its card. V2-53 step 3.

Run spec `docs/specs/2026-10-11_v2-backbone.md`, Stage A (A1-A6).

Why this is a script and not a `huggingface-cli download`:

* **A1** the revision is resolved and recorded, so a later Hub commit cannot be mistaken for
  what we scored.
* **A2** every downloaded file's sha256 is checked against the Hub's LFS oid *before first
  use*. A truncated download otherwise scores as if it were fine.
* **A3** the target must resolve off the repo volume. `/` has ~19 GiB free against 16.4 GiB
  of weights, and this project has already lost three Docker builds to a full disk
  (`docs/specs/2026-09-26_k-docker-image.md`).
* **A4** the realpath is recorded, because a symlinked model dir resolving elsewhere has
  bitten this repo before.
* **A5** the licence (CC BY-NC-SA 4.0: non-commercial *and* share-alike) is recorded.
* **A6** the identity record is what every score file is later bound to. A rank reference
  built from one model and scored with another is silently wrong -- the `check_m5_identity`
  lesson from the M4/M5 lane.

The card is archived because the whole contamination argument rests on its training-corpora
table. Citing a web page read once, with nothing in the repo, is not evidence.

Usage:
  export HF_HOME="/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/hf"
  uv run python scripts/v2_backbone_fetch.py --model mms-300m-anti-deepfake
  uv run python scripts/v2_backbone_fetch.py --model xls-r-2b-anti-deepfake --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "v2" / "backbone"
ORG = "nii-yamagishilab"
LICENCE = "CC BY-NC-SA 4.0"
CANDIDATES = (
    "mms-300m-anti-deepfake",
    "xls-r-1b-anti-deepfake",
    "mms-1b-anti-deepfake",
    "xls-r-2b-anti-deepfake",
)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hub_meta(model: str) -> dict:
    """Revision sha plus each file's size and LFS oid. The oid of an LFS file is its sha256."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(f"{ORG}/{model}", files_metadata=True)
    files = {}
    for s in info.siblings:
        lfs = getattr(s, "lfs", None)
        oid = None
        if lfs is not None:
            oid = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files[s.rfilename] = {"size": getattr(s, "size", None), "lfs_sha256": oid}
    return {"revision": info.sha, "files": files}


class TargetRefused(ValueError):
    """Raised when a download target fails A3. Separate from `die` so tests can assert on it."""


class SameVolume(TargetRefused):
    """The target shares a volume with the repo. Overridable, because it is a placement policy."""


class NotEnoughSpace(TargetRefused):
    """The target volume lacks 2x the download. **Not** overridable: a full disk is not a policy.

    These are separate classes because catching the base type made HEARSAY_ALLOW_SAME_VOLUME
    silently bypass the free-space rule too -- the half of A3 that exists because this project
    has already lost three Docker builds to a full disk.
    """


def check_target(target: Path, need_bytes: int, repo: Path = REPO) -> Path:
    """A3: refuse the repo **volume**, and refuse a volume without 2x the download free.

    The first revision compared path prefixes, which is "not inside the repo directory", not
    "off the repo volume" -- and it passed on the one case it exists to prevent:
    `~/.cache/huggingface` is a different directory on the *same* device as the repo
    (`st_dev` 16777229 for both), so 16.4 GiB of weights would have been allowed onto the
    volume that has ~19 GiB free. Compare device ids, not strings.
    """
    resolved = target.resolve()
    probe = resolved
    while not probe.exists():
        probe = probe.parent
    if probe.stat().st_dev == repo.resolve().stat().st_dev:
        raise SameVolume(
            f"refusing: {resolved} is on the same volume as the repo (st_dev "
            f"{probe.stat().st_dev}); weights must go on separate storage (A3/D8)"
        )
    free = shutil.disk_usage(probe).free
    if free < 2 * need_bytes:
        raise NotEnoughSpace(
            f"refusing: {free / 2**30:.1f} GiB free at {probe} < 2x the {need_bytes / 2**30:.1f} GiB download"
        )
    return resolved


def fetch(model: str, dry_run: bool) -> dict:
    meta = hub_meta(model)
    need = sum(f["size"] or 0 for f in meta["files"].values())
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        die("HF_HOME is unset; point it at the external drive before downloading (A3)")
    try:
        target = check_target(Path(hf_home), need)
    except SameVolume as e:
        if os.environ.get("HEARSAY_ALLOW_SAME_VOLUME") != "1":
            die(f"{e}\n       set HEARSAY_ALLOW_SAME_VOLUME=1 to override, and record it as a deviation")
        print(f"WARNING: A3 overridden by HEARSAY_ALLOW_SAME_VOLUME -- {e}")
        target = Path(hf_home).resolve()

    print(f"{model}: revision {meta['revision'][:12]} | {need / 2**30:.2f} GiB | cache {target}")
    for name, f in sorted(meta["files"].items()):
        if f["size"] and f["size"] > 10**6:
            print(f"  {name}: {f['size'] / 2**30:.2f} GiB  lfs_sha256={(f['lfs_sha256'] or '-')[:12]}")
    if dry_run:
        print("dry run: nothing downloaded")
        return {"model_id": f"{ORG}/{model}", "revision": meta["revision"], "dry_run": True}

    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(f"{ORG}/{model}", revision=meta["revision"], allow_patterns=["*.json", "*.md", "*.safetensors"])
    )

    verified = {}
    for name, f in sorted(meta["files"].items()):
        p = local / name
        if not p.exists():
            continue
        got = file_sha256(p)
        want = f["lfs_sha256"]
        if want and got != want:
            die(f"A2 sha mismatch on {name}: got {got[:16]} want {want[:16]} -- refusing to use it")
        verified[name] = got
    print(f"  verified sha256 for {len(verified)} file(s); {sum(1 for v in meta['files'].values() if v['lfs_sha256'])} had a Hub oid to check against")

    cards = OUT / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    for src, dst in (("README.md", f"{model}.card.md"), ("config.json", f"{model}.config.json")):
        if (local / src).exists():
            shutil.copy2(local / src, cards / dst)
            print(f"  archived {dst}")

    record = {
        "model_id": f"{ORG}/{model}",
        "revision": meta["revision"],
        "licence": LICENCE,
        "realpath": str(local.resolve()),
        "file_sha256": verified,
        "download_bytes": need,
        "card_archived": str((cards / f"{model}.card.md").relative_to(REPO)),
    }
    cfg = local / "config.json"
    if cfg.exists():
        raw = json.loads(cfg.read_text())
        record["n_layers"] = raw.get("num_hidden_layers")
        record["hidden_size"] = raw.get("hidden_size")
        record["do_stable_layer_norm"] = raw.get("do_stable_layer_norm")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{model}.identity.json"
    dest.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"  wrote {dest.relative_to(REPO)}")
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=CANDIDATES)
    ap.add_argument("--dry-run", action="store_true", help="resolve and print, download nothing")
    args = ap.parse_args(argv)
    fetch(args.model, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
