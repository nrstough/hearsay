#!/bin/bash
# Rebuild the bundle's code.tgz from the CURRENT working tree, refresh TREE_SHA and
# bundle_meta.json, and push those three files to R2. Run this after ANY code change and before
# launching a box: a box that pulls a stale code.tgz runs stale code (the 04:51 pilot failure).
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=r2_guard.sh
. "$HERE/r2_guard.sh"
BUNDLE="${BUNDLE:-/Volumes/Crucial P3 NVME Gen 3 2TB/hearsay/m5_bundle/v1}"
DST="$(r2_path "${HEARSAY_R2_PREFIX}bundle/v1")"
cd "$REPO"
"$REPO/.venv/bin/python" - "$BUNDLE" <<'PY'
import json, sys, time
from pathlib import Path
sys.path.insert(0, "scripts")
from m5_build_bundle import make_code_tgz, _git_sha, xlsr_hf_revision, xlsr_hf_weight_sha256
from hearsay.m5_bundle import tree_sha
out = Path(sys.argv[1])
make_code_tgz(out)
sha = tree_sha(out)
(out / "TREE_SHA").write_text(sha)
meta = json.loads((out / "bundle_meta.json").read_text())
meta.update(tree_sha=sha, git_sha=_git_sha(), code_rebuilt_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            xlsr_hf_revision=meta.get("xlsr_hf_revision") or xlsr_hf_revision(),
            xlsr_hf_weight_sha256=meta.get("xlsr_hf_weight_sha256") or xlsr_hf_weight_sha256())
(out / "bundle_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
print("code.tgz rebuilt; tree", sha[:12], "git", meta["git_sha"][:8])
PY
for f in code.tgz TREE_SHA bundle_meta.json; do rclone copyto "$BUNDLE/$f" "$DST/$f"; done
echo "pushed to $DST"
