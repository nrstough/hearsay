#!/bin/bash
# Build the linux/amd64 HEARSAY image (run spec D3, D12). Stages the shipped model bundles into
# docker/build/models/ with symlinks resolved (cp -RL): the probe under its real name (the
# runner's default PROBE_DIR), models/hc_selected, models/cmp_selected and every
# models/fusion_*/constants.json (the runner's default is fusion_v1). Records BUILD_INFO and the git sha, tags hearsay:<stamp> and
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
FUSION_DIRS=()
for d in models/fusion_*; do [ -f "$d/constants.json" ] && FUSION_DIRS+=("$d"); done
[ -f "$PROBE_REAL/probe.joblib" ] || { echo "build.sh: $PROBE_REAL has no probe.joblib" >&2; exit 1; }
[ -f "$HC_REAL/model.joblib" ] || { echo "build.sh: $HC_REAL has no model.joblib" >&2; exit 1; }
[ -f "$CMP_REAL/model.joblib" ] || { echo "build.sh: $CMP_REAL has no model.joblib" >&2; exit 1; }
[ "${#FUSION_DIRS[@]}" -ge 1 ] || { echo "build.sh: no models/fusion_*/constants.json (scripts/fuse.py writes them)" >&2; exit 1; }

# Disk guard: the Colima VM disk is a sparse file on whichever drive hosts ~/.colima (a virtual
# cap does not protect that drive); a from-scratch build transiently adds over 10 GB. Refuse below
# MIN_FREE_GB on that drive and prune leftovers afterwards.
MIN_FREE_GB="${MIN_FREE_GB:-8}"
VM_DIR="$(python3 -c 'import os; p = os.path.expanduser("~/.colima"); print(os.path.realpath(p) if os.path.exists(p) else "/")')"
FREE_GB="$(( $(df -Pk "$VM_DIR" | awk 'NR==2 {print $4}') / 1048576 ))"
if [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "build.sh: only ${FREE_GB} GB free on the drive hosting $VM_DIR; need ${MIN_FREE_GB} GB (prune images or free space)" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M)"
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY="$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo dirty || echo clean)"
FUSION_NAMES="$(printf '%s,' "${FUSION_DIRS[@]##*/}")"; FUSION_NAMES="${FUSION_NAMES%,}"
BUILD_INFO="sha=$SHA $DIRTY stamp=$STAMP probe=$(basename "$PROBE_REAL") hc=$(basename "$HC_REAL") cmp=$(basename "$CMP_REAL") fusion=$FUSION_NAMES"

if [ "${1:-}" = "--print-args" ]; then
  echo "PROBE_REAL=$PROBE_REAL"; echo "HC_REAL=$HC_REAL"; echo "CMP_REAL=$CMP_REAL"
  echo "FUSION_DIRS=${FUSION_DIRS[*]}"; echo "BUILD_INFO=$BUILD_INFO"; exit 0
fi

# Free VM disk before building: superseded tags (all but the current :latest) and dangling images go
# first, since the 20 GB cap cannot hold a third 7.5 GB tag mid-build (build 8, 08:41).
KEEP="$(docker image inspect "$IMAGE:latest" --format '{{.Id}}' 2>/dev/null || true)"
for tag in $(docker images "$IMAGE" --format '{{.Tag}}' | grep -v '^latest$'); do
  [ "$(docker image inspect "$IMAGE:$tag" --format '{{.Id}}')" = "$KEEP" ] || docker rmi "$IMAGE:$tag" >/dev/null 2>&1 || true
done
docker image prune -f >/dev/null 2>&1 || true

rm -rf docker/build && mkdir -p docker/build/models
cp -RL "$PROBE_REAL" "docker/build/models/$(basename "$PROBE_REAL")"
cp -RL "$HC_REAL" docker/build/models/hc_selected
cp -RL "$CMP_REAL" docker/build/models/cmp_selected
for d in "${FUSION_DIRS[@]}"; do mkdir -p "docker/build/models/${d##*/}" && cp "$d/constants.json" "docker/build/models/${d##*/}/constants.json"; done
echo "build.sh: $BUILD_INFO"
docker buildx build --platform linux/amd64 --load \
  --build-arg BUILD_INFO="$BUILD_INFO" --build-arg GIT_SHA="$SHA" \
  -t "$IMAGE:$STAMP" -t "$IMAGE:latest" .
docker image inspect "$IMAGE:latest" --format 'image {{.Architecture}} {{.Size}} bytes'
# Keep the footprint small: only this build's tag plus :latest survive, dangling images go; the
# builder cache stays so the next build reuses the dependency and weight layers (about 2 min
# instead of about 4 from scratch); the VM's hard disk cap bounds it.
for tag in $(docker images "$IMAGE" --format '{{.Tag}}' | grep -v -e latest -e "^$STAMP$"); do docker rmi "$IMAGE:$tag" >/dev/null 2>&1 || true; done
docker image prune -f >/dev/null 2>&1 || true
echo "build.sh: $(( $(df -Pk "$VM_DIR" | awk 'NR==2 {print $4}') / 1048576 )) GB free on the drive hosting $VM_DIR after pruning"
