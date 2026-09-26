#!/bin/bash
# Mac-side reaper: polls each job's STATUS in R2 and destroys its box on DONE / FAIL, on a
# stalled STATUS (no change for STALL_MIN minutes), or at DEADLINE. The API key stays here.
# Run under caffeinate: `caffeinate -i scripts/cloud/reaper.sh` (Ctrl-C does not destroy).
# Env: DEADLINE (unix), STALL_MIN (default 40), POLL (seconds, default 60).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=r2_guard.sh
. "$HERE/r2_guard.sh"
DEADLINE="${DEADLINE:-$(date -j -f '%H:%M' '11:45' '+%s' 2>/dev/null || date -d '11:45' '+%s')}"
STALL_MIN="${STALL_MIN:-40}"; POLL="${POLL:-60}"
STATE="$HOME/.hearsay_vast"
LEDGER="$REPO/docs/reports/cloud-expense-ledger.md"
export VAST_API_KEY="${VAST_API_KEY:-$(cat "$HOME/.config/vastai/vast_api_key")}"
VAST="uvx vastai"
py() { "$REPO/.venv/bin/python" -c "$@"; }
kill_() { echo y | $VAST destroy instance "$1" >/dev/null 2>&1; }

# per-job state lives in files (macOS ships bash 3.2: no associative arrays)
while :; do
  NOW=$(date +%s)
  ACTIVE=0
  for d in "$STATE"/*/; do
    [ -f "$d/CID" ] || continue
    JOB=$(basename "$d"); CID=$(cat "$d/CID")
    [ -f "$d/DESTROYED" ] && continue
    [ -f "$d/HOLD" ] && { echo "$(date '+%H:%M:%S') [$JOB] held (swap in progress)"; continue; }
    ACTIVE=$((ACTIVE + 1))
    ST=$(rclone cat "${HEARSAY_R2_PREFIX}runs/$JOB/STATUS" 2>/dev/null || echo "?")
    LAST=$(cat "$d/LAST" 2>/dev/null || echo "")
    if [ "$LAST" != "$ST" ]; then echo "$ST" > "$d/LAST"; echo "$NOW" > "$d/SEEN"; fi
    SEEN=$(cat "$d/SEEN" 2>/dev/null || echo "$NOW")
    AGE=$(( (NOW - SEEN) / 60 ))
    REASON=""
    case "$ST" in
      DONE) REASON="done" ;;
      FAIL*) REASON="failed: $ST" ;;
    esac
    [ -z "$REASON" ] && [ "$NOW" -ge "$DEADLINE" ] && REASON="deadline"
    [ -z "$REASON" ] && [ "$AGE" -ge "$STALL_MIN" ] && [ "$ST" != "?" ] && REASON="stalled ${AGE}m at '$ST'"
    [ -z "$REASON" ] && [ "$ST" = "?" ] && [ "$AGE" -ge $((STALL_MIN * 2)) ] && REASON="no STATUS for ${AGE}m"
    if [ -n "$REASON" ]; then
      echo "$(date '+%H:%M:%S') [$JOB] $REASON -> destroy $CID"
      kill_ "$CID"; date +%s > "$d/DESTROYED"
      printf '| %s | vast.ai | %s | %s | destroyed (%s) | | | | \n' "$(date '+%Y-%m-%d %H:%M')" "$CID" "$JOB" "$REASON" >> "$LEDGER"
    else
      echo "$(date '+%H:%M:%S') [$JOB] $ST (${AGE}m)"
    fi
  done
  if [ "$ACTIVE" -eq 0 ]; then
    LEFT=$($VAST show instances --raw 2>/dev/null | py 'import sys,json; print(len(json.load(sys.stdin) or []))' 2>/dev/null || echo "?")
    echo "no active jobs; instances on the account: $LEFT"
    # never exit early: jobs are launched after the reaper starts; stop 30 min past the deadline
    [ "$NOW" -ge $((DEADLINE + 1800)) ] && [ "$LEFT" = 0 ] && exit 0
  fi
  sleep "$POLL"
done
