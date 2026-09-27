#!/usr/bin/env bash
# Parity smoke on the box, from the repo root as any user. Exits 1 if a verdict differs from the Mac's
# (demo/README.md: 0.0012 real, 0.9979 synthetic, 0.9979 synthetic; rule A3_w0.2_E) or the app is not up.
set -uo pipefail
API=${API:-http://127.0.0.1:8000}
WEB=${WEB:-http://127.0.0.1:80}
fail=0

echo "== API health =="
for _ in $(seq 1 150); do curl -sf "$API/health" >/dev/null && break; sleep 2; done
curl -sf "$API/health" | python3 -c '
import json,sys; h=json.load(sys.stdin)
print("fusion:", h["version"]["fusion"], "| rule:", h["rule"], "| scorers:", h["scorers"], "| n_results:", h["n_results"], "| loaded:", h["models_loaded"], h["model_load_seconds"], "| git:", h["version"]["git_sha"])
sys.exit(0 if h["version"]["fusion"]=="fusion_v2/constants.json" and len(h["scorers"])==4 and h["rule"]!="m1b_only" else 1)' \
  || { echo "FAIL health: expected fusion_v2/constants.json, four scorers"; fail=1; }

check() {  # file expected_prob expected_verdict
  local out p v r
  out=$(curl -sf -F "file=@$1" "$API/analyze" | python3 -c '
import json,sys; d=json.load(sys.stdin); print(round(d["probability_synthetic"],4), d["verdict"], d["fusion"]["detail"]["final"])') || out="request failed"
  echo "$1 -> $out   (Mac: $2 $3 A3_w0.2_E)"
  read -r p v r <<<"$out"
  python3 -c "import sys; sys.exit(0 if abs(float('${p:-9}')-$2)<1e-3 and '${v:-}'=='$3' and '${r:-}'=='A3_w0.2_E' else 1)" 2>/dev/null \
    || { echo "FAIL parity on $1"; fail=1; }
}
echo "== demo uploads through the API =="
check demo/synthetic_apple_tts_samantha.wav 0.9979 synthetic
check demo/real_in_the_wild_28886.wav 0.0012 real
check demo/synthetic_in_the_wild_5518.wav 0.9979 synthetic

echo "== app through nginx =="
code=$(curl -s -o /dev/null -w '%{http_code}' "$WEB/"); echo "GET / -> $code"; [ "$code" = 200 ] || fail=1
curl -sf "$WEB/api/forensic/presets" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("presets:", len(d) if isinstance(d,list) else {k:(len(v) if isinstance(v,list) else v) for k,v in d.items()})' \
  || { echo "FAIL presets"; fail=1; }
sha=$(curl -sf "$WEB/api/forensic/export-tsv" | sha256sum | cut -c1-8); echo "export-tsv sha256 prefix: $sha (expected fb783076)"; [ "$sha" = fb783076 ] || fail=1
curl -sf "$WEB/api/forensic/batch" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("batch keys:", list(d)[:8] if isinstance(d,dict) else type(d).__name__)' || { echo "FAIL batch"; fail=1; }
curl -sf -o /dev/null -w 'web upload of the real clip: HTTP %{http_code} in %{time_total}s\n' -F "file=@demo/real_in_the_wild_28886.wav" "$WEB/api/forensic/analyze" \
  || { echo "FAIL web upload"; fail=1; }

if [ "$fail" = 0 ]; then echo "SMOKE OK"; else echo "SMOKE FAILED"; exit 1; fi
