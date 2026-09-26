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
# instances(): the validated instance list (JSON array) from a SUCCESSFUL query, or exit 1.
# Every consumer treats a failure as "unknown", never as "no instances" (Codex round 9).
instances() {
  local raw rc
  raw=$($VAST show instances --raw 2>/dev/null); rc=$?
  [ "$rc" = 0 ] || return 1
  printf '%s' "$raw" | py "import sys,json
try:
    xs = json.load(sys.stdin)
except Exception:
    sys.exit(3)
if not isinstance(xs, list):
    sys.exit(3)
print(json.dumps(xs))" 2>/dev/null
}
gone() {
  local xs
  xs=$(instances) || { echo unknown; return; }
  printf '%s' "$xs" | py "import sys,json
xs=json.load(sys.stdin)
print('yes' if not any(str(i.get('id')) == '$1' for i in xs) else 'no')" 2>/dev/null || echo unknown
}

# per-job state lives in files (macOS ships bash 3.2: no associative arrays)
while :; do
  NOW=$(date +%s)
  ACTIVE=0
  for d in "$STATE"/*/; do
    [ -f "$d/CID" ] || continue
    JOB=$(basename "$d"); CID=$(cat "$d/CID")
    [ -f "$d/DESTROYED" ] && continue
    HELD=0
    if [ -f "$d/HOLD" ]; then
      HOLD_AGE=$(( (NOW - $(cat "$d/HOLD" 2>/dev/null || echo "$NOW")) / 60 ))
      if [ "$HOLD_AGE" -ge "${HOLD_MAX_MIN:-15}" ]; then rm -f "$d/HOLD"; echo "$(date '+%H:%M:%S') [$JOB] stale hold expired"; else HELD=1; fi
    fi
    ACTIVE=$((ACTIVE + 1))
    ST=$(rclone cat "${HEARSAY_R2_PREFIX}runs/$JOB/STATUS" 2>/dev/null || echo "?")
    LAST=$(cat "$d/LAST" 2>/dev/null || echo "")
    if [ "$LAST" != "$ST" ]; then echo "$ST" > "$d/LAST"; echo "$NOW" > "$d/SEEN"; fi
    SEEN=$(cat "$d/SEEN" 2>/dev/null || echo "$NOW")
    AGE=$(( (NOW - SEEN) / 60 ))
    # order of precedence: ORPHAN, then the absolute deadline, then a fresh HOLD (which defers
    # every status-based rule, including a stale FAIL left over from before a swap), then
    # DONE / FAIL / stall (Codex round 7)
    REASON=""
    [ -f "$d/ORPHAN" ] && REASON="orphan from a failed launch"
    [ -z "$REASON" ] && [ "$NOW" -ge "$DEADLINE" ] && REASON="deadline"
    [ -z "$REASON" ] && [ "$HELD" = 1 ] && { echo "$(date '+%H:%M:%S') [$JOB] held (swap in progress; status '$ST' deferred)"; continue; }
    if [ -z "$REASON" ]; then
      case "$ST" in
        DONE) REASON="done" ;;
        FAIL*) REASON="failed: $ST" ;;
      esac
    fi
    [ -z "$REASON" ] && [ "$AGE" -ge "$STALL_MIN" ] && [ "$ST" != "?" ] && REASON="stalled ${AGE}m at '$ST'"
    [ -z "$REASON" ] && [ "$ST" = "?" ] && [ "$AGE" -ge $((STALL_MIN * 2)) ] && REASON="no STATUS for ${AGE}m"
    if [ -n "$REASON" ]; then
      echo "$(date '+%H:%M:%S') [$JOB] $REASON -> destroy $CID"
      kill_ "$CID"; sleep 5
      if [ "$(gone "$CID")" = yes ]; then   # only a confirmed absence ends the watch (Codex 4)
        date +%s > "$d/DESTROYED"
        printf '| %s | vast.ai | %s | %s | destroyed (%s) | | | | \n' "$(date '+%Y-%m-%d %H:%M')" "$CID" "$JOB" "$REASON" >> "$LEDGER"
      else
        echo "$(date '+%H:%M:%S') [$JOB] destroy of $CID not confirmed; retrying next poll"
      fi
    else
      echo "$(date '+%H:%M:%S') [$JOB] $ST (${AGE}m)"
    fi
  done
  if [ "$ACTIVE" -eq 0 ]; then
    LEFT=$(instances | py 'import sys,json; print(len(json.load(sys.stdin)))' 2>/dev/null || echo "?")
    echo "no active jobs; instances on the account: ${LEFT:-?}"
    # never exit early: jobs are launched after the reaper starts; stop 30 min past the deadline,
    # and only on a VALIDATED empty list ("?" keeps watching)
    [ "$NOW" -ge $((DEADLINE + 1800)) ] && [ "$LEFT" = 0 ] && exit 0
  fi
  sleep "$POLL"
done
