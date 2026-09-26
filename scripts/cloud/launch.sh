#!/bin/bash
# Rent one vast.ai box for an M5 job and start its chain. Tries the given offer ids in order
# until one boots with working outbound network; a box that fails anywhere before its chain
# starts is destroyed by the trap below (never left running).
#
# Usage: scripts/cloud/launch.sh <job-name> "<JOBS>" <est-hours> <offer-id> [offer-id ...]
#   JOBS is the space-separated list run on the box, e.g. "fold=4:steps=600 codecs"
#   (box_chain.sh parses it). est-hours feeds the budget guard.
#
# Env: VAST_API_KEY (else read from ~/.config/vastai/vast_api_key; never printed),
#      DEADLINE (unix time; default: today 11:45 local), IMAGE, DISK_GB.
# State: ~/.hearsay_vast/<job>/{CID,SSH}. Ledger row appended to docs/reports/cloud-expense-ledger.md.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# shellcheck source=r2_guard.sh
. "$HERE/r2_guard.sh"

JOB="${1:?job name}"; JOBS="${2:?jobs}"; EST_HOURS="${3:?est hours}"; shift 3
[ $# -ge 1 ] || { echo "need at least one offer id" >&2; exit 2; }
IMAGE="${IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime}"
DISK_GB="${DISK_GB:-60}"
DEADLINE="${DEADLINE:-$(date -j -f '%H:%M' '11:45' '+%s' 2>/dev/null || date -d '11:45' '+%s')}"
STATE="$HOME/.hearsay_vast/$JOB"; rm -rf "$STATE"; mkdir -p "$STATE"   # never reuse stale state
LEDGER="$REPO/docs/reports/cloud-expense-ledger.md"
CREDS="$HOME/.config/cloudflare-r2-pa-source.txt"
KEY="${VAST_API_KEY:-$(cat "$HOME/.config/vastai/vast_api_key")}"
VAST="uvx vastai"
export VAST_API_KEY="$KEY"

py() { "$REPO/.venv/bin/python" -c "$@"; }

# --- budget guard: credit minus the commitments of boxes already running ---
CREDIT=$($VAST show user --raw 2>/dev/null | py 'import sys,json; print(json.load(sys.stdin).get("credit", 0))')
RUNNING=$($VAST show instances --raw 2>/dev/null | py 'import sys,json
xs=json.load(sys.stdin) or []
print(sum(float(i.get("dph_total") or 0) for i in xs) if isinstance(xs, list) else 0)')
NEED=$(py "print(round($EST_HOURS * 0.9 + 1.5, 2))")   # est hours at ~\$0.9/h upper bound + margin
COMMIT=$(py "print(round($RUNNING * 2.0, 2))")          # running boxes: assume 2 h more each
OK=$(py "print(int($CREDIT - $COMMIT >= $NEED))")
echo "[budget] credit \$$CREDIT, running commitments \$$COMMIT, need \$$NEED"
[ "$OK" = 1 ] || { echo "REFUSED: budget guard (top up or wait for a box to finish)" >&2; exit 3; }

ist() { $VAST show instances --raw 2>/dev/null | py "import sys,json
for i in json.load(sys.stdin) or []:
    if i['id']==$1: print(i.get('actual_status') or 'starting')"; }
issh() { $VAST show instances --raw 2>/dev/null | py "import sys,json
for i in json.load(sys.stdin) or []:
    if i['id']==$1: print(i.get('ssh_host'), i.get('ssh_port'))"; }
kill_() { echo y | $VAST destroy instance "$1" >/dev/null 2>&1; }

CID=""
cleanup() {
  # any exit before PROVISIONED destroys the box we created
  if [ -n "$CID" ] && [ ! -f "$STATE/PROVISIONED" ]; then
    echo "[cleanup] destroying $CID (launch did not complete)"; kill_ "$CID"
  fi
}
trap cleanup EXIT

for OFFER in "$@"; do
  echo "=== offer $OFFER ==="
  CID=$($VAST create instance "$OFFER" --image "$IMAGE" --disk "$DISK_GB" --ssh --raw 2>&1 | py 'import sys,json
try: print(json.load(sys.stdin).get("new_contract","") or "")
except Exception: print("")')
  [ -n "$CID" ] || { echo "  create failed"; CID=""; continue; }
  echo "  CID=$CID waiting for running..."; ST=""
  for _ in $(seq 1 40); do ST=$(ist "$CID"); [ "$ST" = running ] && break; sleep 15; done
  [ "$ST" = running ] || { echo "  stuck ($ST) -> destroy"; kill_ "$CID"; CID=""; continue; }
  read -r HOST PORT <<<"$(issh "$CID")"
  SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT root@$HOST"
  NET=BAD
  for _ in $(seq 1 8); do
    if $SSH 'echo nameserver 8.8.8.8 > /etc/resolv.conf; curl -sS -m 10 -o /dev/null https://rclone.org && echo NET-OK' 2>/dev/null | grep -q NET-OK; then NET=OK; break; fi
    sleep 10
  done
  [ "$NET" = OK ] || { echo "  no outbound network -> destroy"; kill_ "$CID"; CID=""; continue; }
  echo "  network OK; provisioning..."
  ACCT=$(grep '^account_id=' "$CREDS" | cut -d= -f2)
  AK=$(grep '^access_key_id=' "$CREDS" | cut -d= -f2)
  SK=$(grep '^secret_access_key=' "$CREDS" | cut -d= -f2)
  # rclone + creds: the creds travel on stdin (not argv, not the box's shell history)
  printf 'access_key_id=%s\nsecret_access_key=%s\nendpoint=https://%s.r2.cloudflarestorage.com\n' "$AK" "$SK" "$ACCT" \
    | $SSH 'set -e; cd /root; (command -v rclone >/dev/null) || (curl -sS -m 60 -O https://downloads.rclone.org/rclone-current-linux-amd64.deb && dpkg -i rclone-current-linux-amd64.deb >/dev/null);
      mkdir -p /root/.config/rclone; { echo "[r2]"; echo "type = s3"; echo "provider = Cloudflare"; echo "no_check_bucket = true"; cat; } > /root/.config/rclone/rclone.conf; chmod 600 /root/.config/rclone/rclone.conf;
      rclone lsd r2:pa-source/hearsay/ >/dev/null && echo RCLONE-OK' | grep -q RCLONE-OK \
    || { echo "  rclone setup failed -> destroy"; kill_ "$CID"; CID=""; continue; }
  # ship the box scripts and start the chain detached; PROVISIONED is written ONLY after the
  # chain is confirmed running (a box whose chain never started is destroyed, Codex round 3)
  if ! tar czf - -C "$HERE" box_setup.sh box_chain.sh box_codecs.py r2_guard.sh 2>/dev/null \
       | $SSH 'mkdir -p /root/m5/cloud && tar xzf - -C /root/m5/cloud 2>/dev/null'; then
    echo "  script transfer failed -> destroy"; kill_ "$CID"; CID=""; continue
  fi
  if ! $SSH "setsid env JOB='$JOB' JOBS='$JOBS' DEADLINE='$DEADLINE' bash /root/m5/cloud/box_chain.sh > /root/m5/chain.log 2>&1 < /dev/null & sleep 3; pgrep -f 'bash /root/m5/cloud/box_ch[a]in.sh' >/dev/null && echo CHAIN-UP" 2>/dev/null | grep -q CHAIN-UP; then
    echo "  chain did not start -> destroy"; kill_ "$CID"; CID=""; continue
  fi
  echo PROVISIONED
  echo "$CID" > "$STATE/CID"; echo "$HOST $PORT" > "$STATE/SSH"; date +%s > "$STATE/PROVISIONED"
  DPH=$($VAST show instances --raw | py "import sys,json
for i in json.load(sys.stdin) or []:
    if i['id']==$CID: print(i.get('dph_total'), i.get('gpu_name'))")
  printf '| %s | vast.ai | %s (%s) | %s | %s | | | est %s h | \n' "$(date '+%Y-%m-%d %H:%M')" "$CID" "$OFFER" "$JOB" "$DPH" "$EST_HOURS" >> "$LEDGER"
  echo "launched $JOB on $CID ($DPH); ssh -p $PORT root@$HOST; log: /root/m5/chain.log"
  exit 0
done
echo "no offer worked" >&2
exit 1
