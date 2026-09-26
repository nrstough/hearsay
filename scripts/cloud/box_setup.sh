#!/bin/bash
# Box bring-up for M5 on pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime: uses the image's torch,
# installs the rest, pulls the bundle + code from R2, verifies the tree sha, gets XLS-R.
set -eu
cd /root/m5
# shellcheck source=r2_guard.sh
. /root/m5/cloud/r2_guard.sh
B="${HEARSAY_R2_PREFIX}bundle/v1"

echo "[setup] apt (ffmpeg) ..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq ffmpeg git >/dev/null
ffmpeg -hide_banner -encoders 2>/dev/null | grep -Eo 'libmp3lame|libopus| aac |libopencore_amrnb|pcm_mulaw' | sort -u | tr '\n' ' '; echo

echo "[setup] python deps (image torch: $(python -c 'import torch;print(torch.__version__)'))"
pip install -q "transformers==5.17.0" safetensors soundfile scipy pandas scikit-learn tqdm huggingface_hub
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('CUDA OK', torch.cuda.get_device_name(0))"

echo "[setup] pulling code + bundle ..."
mkdir -p code bundle weights
rclone copy "$B/code.tgz" /root/m5/ && tar xzf /root/m5/code.tgz -C code && cp /root/m5/code.tgz bundle/
# the Mac's uplink is slow: wait (up to 90 min) for the upload marker before pulling audio
for i in $(seq 1 180); do
  rclone lsf "$B/UPLOAD_DONE" 2>/dev/null | grep -q UPLOAD_DONE && break
  [ "$i" = 1 ] && echo "  waiting for $B/UPLOAD_DONE ..."
  sleep 30
done
rclone lsf "$B/UPLOAD_DONE" 2>/dev/null | grep -q UPLOAD_DONE || { echo "FATAL: bundle upload never completed"; exit 1; }
rclone copy "$B" bundle --transfers 32 --checkers 32 --exclude 'codecs/**' --exclude 'code.tgz' --exclude 'UPLOAD_DONE' --stats 30s --stats-one-line
export PYTHONPATH=/root/m5/code/src
python - <<'EOF'
import json
from hearsay.m5_bundle import tree_sha
meta = json.load(open('/root/m5/bundle/bundle_meta.json'))
got = tree_sha('/root/m5/bundle')
assert got == meta['tree_sha'], f"tree sha {got[:12]} != {meta['tree_sha'][:12]}"
print('bundle tree OK', got[:12], meta['n_rows'], 'rows')
EOF

echo "[setup] XLS-R weights (pinned Hub revision, sha-checked against bundle_meta.json) ..."
WANT=$(cat bundle/config_sha.txt)
python - <<'EOF'
import hashlib, json
from huggingface_hub import snapshot_download
meta = json.load(open("/root/m5/bundle/bundle_meta.json"))
rev, want = meta.get("xlsr_hf_revision"), meta.get("xlsr_hf_weight_sha256") or {}
assert rev, "bundle_meta.json has no xlsr_hf_revision: refuse to pull unpinned weights"
assert want, "bundle_meta.json has no xlsr_hf_weight_sha256: refuse to pull unverifiable weights"
p = snapshot_download("facebook/wav2vec2-xls-r-300m", revision=rev, local_dir="/root/m5/weights",
                      allow_patterns=["config.json", "preprocessor_config.json", "*.bin", "*.safetensors"])
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
checked = 0
for name, expected in want.items():
    try:
        got = sha(f"/root/m5/weights/{name}")
    except FileNotFoundError:
        continue
    assert got == expected, f"{name}: sha {got[:12]} != expected {expected[:12]}"
    checked += 1
assert checked, "no weight file matched the expected manifest"
print("  from HF hub at revision", rev, "-", checked, "weight file(s) sha-verified")
EOF
GOT=$(sha256sum weights/config.json | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || { echo "FATAL: XLS-R config sha $GOT != bundled $WANT"; exit 1; }
python - <<'EOF'
from transformers import Wav2Vec2Model
m = Wav2Vec2Model.from_pretrained('/root/m5/weights')
print('XLS-R loads:', sum(p.numel() for p in m.parameters()) // 1_000_000, 'M params')
EOF
echo "SETUP-DONE"
