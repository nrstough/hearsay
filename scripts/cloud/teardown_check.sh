#!/bin/bash
# Manual backstop: destroy every instance on the account (or just list them with --list) and
# append the invoice total to the ledger. Usage: teardown_check.sh [--list]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export VAST_API_KEY="${VAST_API_KEY:-$(cat "$HOME/.config/vastai/vast_api_key")}"
VAST="uvx vastai"
py() { "$REPO/.venv/bin/python" -c "$@"; }
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
IDS=$(instances | py 'import sys,json; print(" ".join(str(i["id"]) for i in json.load(sys.stdin)))' 2>/dev/null) || { echo "FATAL: cannot read the instance list; nothing confirmed" >&2; exit 2; }
if [ "${1:-}" = "--list" ]; then echo "instances: ${IDS:-none}"; exit 0; fi
for id in $IDS; do echo "destroying $id"; echo y | $VAST destroy instance "$id"; done
sleep "${DESTROY_WAIT:-5}"
LEFT=$(instances | py 'import sys,json; print(len(json.load(sys.stdin)))' 2>/dev/null) || { echo "FATAL: cannot confirm the instance list after destroy" >&2; exit 2; }
CREDIT=$($VAST show user --raw 2>/dev/null | py 'import sys,json; print(round(json.load(sys.stdin).get("credit",0),2))')
echo "instances left: $LEFT; credit now: \$$CREDIT"
printf '| %s | vast.ai | teardown check | | | | | credit left $%s | \n' "$(date '+%Y-%m-%d %H:%M')" "$CREDIT" >> "$REPO/docs/reports/cloud-expense-ledger.md"
[ "$LEFT" = 0 ]
