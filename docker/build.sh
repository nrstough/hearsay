#!/bin/bash
# Build the linux/amd64 HEARSAY image (run spec D3, D12). Stages the shipped model bundles into
# docker/build/models/ with symlinks resolved (cp -RL): the probe under its real name (the
# runner's default PROBE_DIR), models/hc_selected, models/cmp_selected and
# models/fusion_v0/constants.json. Records BUILD_INFO and the git sha, tags hearsay:<stamp> and
# hearsay:latest. Env: PROBE_DIR, IMAGE (default hearsay). --print-args: resolve only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO_ROOT:-$(cd "$HERE/.." && pwd)}"
cd "$REPO"

PROBE_DIR="${PROBE_DIR:-models/m1_wav2vec2-xls-r-300m_L7_20260926-0521}"
IMAGE="${IMAGE:-hearsay}"

resolve() {  # print the real path of $1 (symlinks followed) or fail loudly
  local p="$1" real
  real="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$p")"
  if [ ! -d "$real" ]; then echo "build.sh: $p does not resolve to a directory" >&2; exit 1; fi
  echo "$real"
}
PROBE_REAL="$(resolve "$PROBE_DIR")"
HC_REAL="$(resolve models/hc_selected)"
CMP_REAL="$(resolve models/cmp_selected)"
CONSTANTS="models/fusion_v0/constants.json"
[ -f "$PROBE_REAL/probe.joblib" ] || { echo "build.sh: $PROBE_REAL has no probe.joblib" >&2; exit 1; }
[ -f "$HC_REAL/model.joblib" ] || { echo "build.sh: $HC_REAL has no model.joblib" >&2; exit 1; }
[ -f "$CMP_REAL/model.joblib" ] || { echo "build.sh: $CMP_REAL has no model.joblib" >&2; exit 1; }
[ -f "$CONSTANTS" ] || { echo "build.sh: $CONSTANTS missing (scripts/fuse.py writes it)" >&2; exit 1; }

# Disk guard: the Colima VM disk is a sparse file on whichever drive hosts ~/.colima (a virtual
# cap does not protect that drive); a from-scratch build transiently adds over 10 GB. Refuse below
# MIN_FREE_GB on that drive and prune leftovers afterwards.
MIN_FREE_GB="${MIN_FREE_GB:-8}"
VM_DIR="$(python3 -c 'import os; p = os.path.expanduser("~/.colima"); print(os.path.realpath(p) if os.path.exists(p) else "/")')"
FREE_GB="$(df -g "$VM_DIR" | awk 'NR==2 {print $4}')"
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "build.sh: only ${FREE_GB} GB free on the drive hosting $VM_DIR; need ${MIN_FREE_GB} GB (prune images or free space)" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M)"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY="$(git status --porcelain 2>/dev/null | grep -q . && echo dirty || echo clean)"
BUILD_INFO="sha=$SHA $DIRTY stamp=$STAMP probe=$(basename "$PROBE_REAL") hc=$(basename "$HC_REAL") cmp=$(basename "$CMP_REAL") fusion=fusion_v0"

if [ "${1:-}" = "--print-args" ]; then
  echo "PROBE_REAL=$PROBE_REAL"; echo "HC_REAL=$HC_REAL"; echo "CMP_REAL=$CMP_REAL"
  echo "CONSTANTS=$CONSTANTS"; echo "BUILD_INFO=$BUILD_INFO"; exit 0
fi

rm -rf docker/build && mkdir -p docker/build/models/fusion_v0
cp -RL "$PROBE_REAL" "docker/build/models/$(basename "$PROBE_REAL")"
cp -RL "$HC_REAL" docker/build/models/hc_selected
cp -RL "$CMP_REAL" docker/build/models/cmp_selected
cp "$CONSTANTS" docker/build/models/fusion_v0/constants.json
echo "build.sh: $BUILD_INFO"
docker buildx build --platform linux/amd64 --load \
  --build-arg BUILD_INFO="$BUILD_INFO" --build-arg GIT_SHA="$SHA" \
  -t "$IMAGE:$STAMP" -t "$IMAGE:latest" .
docker image inspect "$IMAGE:latest" --format 'image {{.Architecture}} {{.Size}} bytes'
# Keep the footprint small: only this build's tag plus :latest survive; no dangling layers or cache.
for tag in $(docker images "$IMAGE" --format '{{.Tag}}' | grep -v -e latest -e "^$STAMP$"); do docker rmi "$IMAGE:$tag" >/dev/null 2>&1 || true; done
docker image prune -f >/dev/null 2>&1 || true
docker builder prune -f >/dev/null 2>&1 || true
echo "build.sh: $(df -g "$VM_DIR" | awk 'NR==2 {print $4}') GB free on the drive hosting $VM_DIR after pruning"
