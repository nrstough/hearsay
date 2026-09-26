#!/bin/bash
# Pull every M5 run directory from R2 to outputs/m5_runs/ (checkpoints excluded).
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=r2_guard.sh
. "$HERE/r2_guard.sh"
DST="$REPO/outputs/m5_runs"; mkdir -p "$DST"
r2_copy "${HEARSAY_R2_PREFIX}runs/" "$DST" --exclude '**/ckpt/**' --transfers 16 --stats-one-line
ls "$DST"
