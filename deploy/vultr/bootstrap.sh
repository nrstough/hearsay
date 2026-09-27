#!/usr/bin/env bash
# HEARSAY demo host: run once as root on a fresh Ubuntu 24.04 Vultr box. Idempotent enough to re-run.
# Usage: scp -r deploy/vultr root@<ip>:/root/vultr && ssh root@<ip> 'bash /root/vultr/bootstrap.sh'
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
HERE=$(cd "$(dirname "$0")" && pwd)
REPO_URL=${REPO_URL:-https://github.com/nrstough/hearsay}
BRANCH=${BRANCH:-main}
HOME_DIR=/home/hearsay
APP=$HOME_DIR/hearsay
log() { printf '\n== %s ==\n' "$*"; }

log "packages"
apt-get update -q
apt-get install -y -q ffmpeg nginx git curl rsync build-essential ufw
if ! command -v node >/dev/null 2>&1 || [[ "$(node -v)" != v22* ]]; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y -q nodejs
fi
node -v; npm -v; ffmpeg -version | head -1

log "swap: 4 GB, insurance for the model load and next build on an 8 GB box"
if ! swapon --show | grep -q '^/swapfile'; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
free -h | head -3

log "user hearsay (same authorized_keys as root, so rsync from the Mac works)"
id hearsay >/dev/null 2>&1 || adduser --disabled-password --gecos "" hearsay
install -d -m 700 -o hearsay -g hearsay "$HOME_DIR/.ssh"
if [ -f /root/.ssh/authorized_keys ]; then
  install -m 600 -o hearsay -g hearsay /root/.ssh/authorized_keys "$HOME_DIR/.ssh/authorized_keys"
fi

log "uv"
su - hearsay -c 'test -x ~/.local/bin/uv || curl -LsSf https://astral.sh/uv/install.sh | sh'
su - hearsay -c '~/.local/bin/uv --version'

log "clone $REPO_URL ($BRANCH)"
if [ ! -d "$APP/.git" ]; then
  su - hearsay -c "git clone --branch $BRANCH $REPO_URL $APP"
else
  su - hearsay -c "cd $APP && git fetch -q origin && git checkout -q $BRANCH && git pull -q --ff-only"
fi
# The deploy files may be newer than what is pushed: copy this directory into the checkout.
if [ "$HERE" != "$APP/deploy/vultr" ]; then
  install -d -o hearsay -g hearsay "$APP/deploy/vultr"
  install -m 644 -o hearsay -g hearsay "$HERE"/* "$APP/deploy/vultr/"
fi
su - hearsay -c "cd $APP && git log --oneline -1"

log "python venv (CPU torch; the Dockerfile's recipe)"
su - hearsay -c "cd $APP && bash deploy/vultr/venv.sh"

log "next production build"
su - hearsay -c "cd $APP && cp -n .env.example .env; npm ci --no-audit --no-fund && npm run build"

log "systemd units + nginx"
install -m 644 "$APP"/deploy/vultr/hearsay-api.service "$APP"/deploy/vultr/hearsay-web.service "$APP"/deploy/vultr/hearsay-warm.service /etc/systemd/system/
install -m 644 "$APP"/deploy/vultr/nginx-hearsay.conf /etc/nginx/sites-available/hearsay
ln -sf /etc/nginx/sites-available/hearsay /etc/nginx/sites-enabled/hearsay
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl enable --now nginx >/dev/null
systemctl reload nginx
systemctl daemon-reload
systemctl enable hearsay-api hearsay-web hearsay-warm >/dev/null

log "firewall: 22 and 80 only (3000 and 8000 bind to 127.0.0.1 anyway)"
ufw allow OpenSSH >/dev/null
ufw allow 80/tcp >/dev/null
ufw --force enable >/dev/null
ufw status | sed -n 1,6p

log "bootstrap done"
echo "Next, from the Mac:  bash deploy/vultr/sync_assets.sh <ip>"
echo "Then here:           systemctl start hearsay-api hearsay-web && su - hearsay -c 'cd hearsay && bash deploy/vultr/smoke.sh'"
