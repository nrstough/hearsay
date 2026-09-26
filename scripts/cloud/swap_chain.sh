#!/bin/bash
# Hot-swap a running box's chain: ship the CURRENT box scripts, stop the old (waiting) chain,
# refresh code.tgz from R2, and start a new chain with the given JOBS. Setup re-runs but every
# pull is incremental (rclone skips existing files), so it costs ~2 min, not a relaunch.
# Usage: swap_chain.sh <job-name> "<JOBS>"
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="${1:?job}"; JOBS="${2:?jobs}"
STATE="$HOME/.hearsay_vast/$JOB"
read -r HOST PORT < "$STATE/SSH"
DEADLINE="${DEADLINE:-$(date -j -f '%H:%M' '11:45' '+%s' 2>/dev/null || date -d '11:45' '+%s')}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT root@$HOST"
tar czf - -C "$HERE" box_setup.sh box_chain.sh box_codecs.py r2_guard.sh | $SSH 'mkdir -p /root/m5/cloud && tar xzf - -C /root/m5/cloud'
$SSH "pkill -f box_chain.sh || true; sleep 1; rm -f /root/m5/code.tgz; setsid env JOB='$JOB' JOBS='$JOBS' DEADLINE='$DEADLINE' bash /root/m5/cloud/box_chain.sh >> /root/m5/chain.log 2>&1 < /dev/null & echo SWAPPED"
