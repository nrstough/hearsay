#!/bin/bash
# Hot-swap a running box's chain: ship the CURRENT box scripts, stop the old chain and any
# trainer, refresh code.tgz from R2, and start a new chain with the given JOBS. Setup re-runs
# but every pull is incremental (rclone skips existing files), so it costs ~2 min, not a relaunch.
#
# Two separate ssh sessions on purpose: a `pkill -f` pattern must never appear in the same
# command line that also names the script to start, or it kills its own shell. And build JOBS
# with ${VAR} braces: in zsh, `$C0:extra` applies the `:e` modifier and empties the value.
# Usage: swap_chain.sh <job-name> "<JOBS>"
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="${1:?job}"; JOBS="${2:?jobs}"
STATE="$HOME/.hearsay_vast/$JOB"
read -r HOST PORT < "$STATE/SSH"
DEADLINE="${DEADLINE:-$(date -j -f '%H:%M' '11:45' '+%s' 2>/dev/null || date -d '11:45' '+%s')}"
date +%s > "$STATE/HOLD"   # the reaper skips a held job (a stale FAIL status must not kill the box mid-swap)
trap 'rm -f "$STATE/HOLD"' EXIT   # released on every exit path, including an ssh failure under set -e
tar czf - -C "$HERE" box_setup.sh box_chain.sh box_codecs.py r2_guard.sh 2>/dev/null \
  | ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" "root@$HOST" 'mkdir -p /root/m5/cloud && tar xzf - -C /root/m5/cloud 2>/dev/null'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" "root@$HOST" 'pkill -f "[c]loud/box_chain" || true; pkill -f "[m]5_train" || true; sleep 1; echo "[swap] old chain stopped"'
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" "root@$HOST" "rm -f /root/m5/code.tgz; echo SWAPPING > /tmp/STATUS; rclone copyto /tmp/STATUS r2:pa-source/hearsay/runs/$JOB/STATUS 2>/dev/null; setsid env JOB='$JOB' JOBS='$JOBS' DEADLINE='$DEADLINE' bash /root/m5/cloud/box_chain.sh >> /root/m5/chain.log 2>&1 < /dev/null & sleep 4; echo \"[swap] new chain: \$(cat /tmp/STATUS)\""
rm -f "$STATE/HOLD"
