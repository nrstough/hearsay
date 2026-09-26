#!/bin/bash
# Box-side chain: setup -> jobs -> DONE. Writes STATUS to R2 for the Mac-side reaper, which
# destroys the box (the vast API key never comes here). Any failure writes FAIL and stops.
# Env: JOB, JOBS ("fold=4:steps=600:arm=nsa_extra codecs fold=full:steps=3000"), DEADLINE.
set -u
cd /root/m5
# shellcheck source=r2_guard.sh
. /root/m5/cloud/r2_guard.sh
RUNS="${HEARSAY_R2_PREFIX}runs/${JOB}"
LOGDIR=/root/m5/runs; mkdir -p "$LOGDIR"

status() { echo "$1" > /tmp/STATUS; echo "$(date -u '+%FT%TZ') $1" >> /tmp/STATUS.log
           rclone copyto /tmp/STATUS "$RUNS/STATUS" 2>/dev/null; rclone copyto /tmp/STATUS.log "$RUNS/STATUS.log" 2>/dev/null; }
push_logs() { rclone copy /root/m5/chain.log "$RUNS/" 2>/dev/null; rclone copy "$LOGDIR" "$RUNS/" --exclude 'ckpt/**' 2>/dev/null; }
fail() { status "FAIL $1"; push_logs; exit 1; }
trap 'push_logs' EXIT

status "SETUP"
bash /root/m5/cloud/box_setup.sh > /root/m5/setup.log 2>&1
rclone copy /root/m5/setup.log "$RUNS/" 2>/dev/null
grep -q SETUP-DONE /root/m5/setup.log || fail "setup"

export PYTHONPATH=/root/m5/code/src
cd /root/m5/code
FIRST_JOBS="$JOBS"

# Keep the box (and its warm setup) for follow-up jobs: poll R2 for a NEXT file holding more
# JOBS for up to WAIT_MIN minutes, heartbeating STATUS so the reaper sees it alive.
WAIT_MIN="${WAIT_MIN:-45}"
run_jobs() {
  for spec in "$@"; do
    case "$spec" in
      codecs) status "RUNNING codecs"
        python scripts/cloud/box_codecs.py --bundle /root/m5/bundle --frac 0.3 --workers "$(nproc)" > "$LOGDIR/codecs.log" 2>&1 || fail "codecs"
        rclone copy /root/m5/bundle/codecs "${HEARSAY_R2_PREFIX}bundle/v1/codecs" --transfers 16 || fail "codecs-push" ;;
      fold=*)
        FOLD=$(echo "$spec" | tr ':' '\n' | sed -n 's/^fold=//p')
        STEPS=$(echo "$spec" | tr ':' '\n' | sed -n 's/^steps=//p'); STEPS=${STEPS:-3000}
        ARM=$(echo "$spec" | tr ':' '\n' | sed -n 's/^arm=//p'); ARM=${ARM:-nsa_extra}
        CFG=$(echo "$spec" | tr ':' '\n' | sed -n 's/^cfg=//p'); CFG=${CFG:-{\}}
        EXTRA=$(echo "$spec" | tr ':' '\n' | sed -n 's/^extra=//p')
        NAME="fold${FOLD}_${ARM}"
        status "RUNNING $NAME"
        [ -f /root/m5/bundle/codecs/codec_manifest.csv ] || rclone copy "${HEARSAY_R2_PREFIX}bundle/v1/codecs" /root/m5/bundle/codecs --transfers 16 2>/dev/null || true
        # shellcheck disable=SC2086
        python scripts/m5_train.py --bundle /root/m5/bundle --fold "$FOLD" --arm "$ARM" --steps "$STEPS" \
          --out "$LOGDIR/$NAME" --r2-prefix "$RUNS/$NAME" --deadline "$DEADLINE" --workers 12 \
          --weights-dir /root/m5/weights --config "$CFG" $EXTRA > "$LOGDIR/$NAME.log" 2>&1 || fail "$NAME"
        grep -q '^DONE' "$LOGDIR/$NAME.log" || fail "$NAME-nodone" ;;
      *) fail "unknown job $spec" ;;
    esac
    push_logs
  done
}
# shellcheck disable=SC2086
run_jobs $FIRST_JOBS
while :; do
  waited=0
  while [ "$waited" -lt $((WAIT_MIN * 60)) ]; do
    if rclone lsf "$RUNS/NEXT" 2>/dev/null | grep -q NEXT; then
      NEXT=$(rclone cat "$RUNS/NEXT"); rclone delete "$RUNS/NEXT"
      echo "next jobs: $NEXT"; break
    fi
    [ $((waited % 300)) -eq 0 ] && status "WAITING $waited"
    sleep 30; waited=$((waited + 30))
  done
  [ -n "${NEXT:-}" ] || break
  # shellcheck disable=SC2086
  run_jobs $NEXT
  NEXT=""
done
status "DONE"
push_logs
