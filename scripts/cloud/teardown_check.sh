#!/bin/bash
# Manual backstop: destroy every instance on the account (or just list them with --list) and
# append the invoice total to the ledger. Usage: teardown_check.sh [--list]
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export VAST_API_KEY="${VAST_API_KEY:-$(cat "$HOME/.config/vastai/vast_api_key")}"
VAST="uvx vastai"
py() { "$REPO/.venv/bin/python" -c "$@"; }
IDS=$($VAST show instances --raw 2>/dev/null | py 'import sys,json; print(" ".join(str(i["id"]) for i in (json.load(sys.stdin) or [])))')
if [ "${1:-}" = "--list" ]; then echo "instances: ${IDS:-none}"; exit 0; fi
for id in $IDS; do echo "destroying $id"; echo y | $VAST destroy instance "$id"; done
sleep 5
LEFT=$($VAST show instances --raw 2>/dev/null | py 'import sys,json; print(len(json.load(sys.stdin) or []))')
CREDIT=$($VAST show user --raw 2>/dev/null | py 'import sys,json; print(round(json.load(sys.stdin).get("credit",0),2))')
echo "instances left: $LEFT; credit now: \$$CREDIT"
printf '| %s | vast.ai | teardown check | | | | | credit left $%s | \n' "$(date '+%Y-%m-%d %H:%M')" "$CREDIT" >> "$REPO/docs/reports/cloud-expense-ledger.md"
[ "$LEFT" = 0 ]
