#!/usr/bin/env bash
# Wait for the API, then score one demo clip so the models are resident before the first real upload.
set -euo pipefail
for _ in $(seq 1 150); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 2; done
curl -sf -o /dev/null -w 'warm-up upload: HTTP %{http_code} in %{time_total}s\n' \
  -F file=@demo/synthetic_apple_tts_samantha.wav http://127.0.0.1:8000/analyze
curl -sf http://127.0.0.1:8000/health | python3 -c 'import json,sys; h=json.load(sys.stdin); print("models_loaded", h["models_loaded"], "load_seconds", h["model_load_seconds"], "scorers", h["scorers"])'
