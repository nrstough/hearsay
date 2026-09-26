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
got = tree_sha('/root/m5/bundle', exclude=("TREE_SHA", "bundle_meta.json", "codecs"))
assert got == meta['tree_sha'], f"tree sha {got[:12]} != {meta['tree_sha'][:12]}"
print('bundle tree OK', got[:12], meta['n_rows'], 'rows')
EOF

echo "[setup] XLS-R weights ..."
WANT=$(cat bundle/config_sha.txt)
if rclone copy "${HEARSAY_R2_PREFIX}weights/wav2vec2-xls-r-300m" weights --transfers 8 2>/dev/null && [ -f weights/model.safetensors ]; then
  echo "  from R2"
else
  python - <<'EOF'
import json
from huggingface_hub import snapshot_download
# pinned revision (bundle_meta.json: xlsr_hf_revision), so every box gets the same weights
rev = json.load(open("/root/m5/bundle/bundle_meta.json")).get("xlsr_hf_revision")
assert rev, "bundle_meta.json has no xlsr_hf_revision: refuse to pull unpinned weights"
p = snapshot_download("facebook/wav2vec2-xls-r-300m", revision=rev, local_dir="/root/m5/weights",
                      allow_patterns=["config.json", "preprocessor_config.json", "*.bin", "*.safetensors"])
print("  from HF hub at revision", rev, ":", p)
EOF
fi
GOT=$(sha256sum weights/config.json | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || { echo "FATAL: XLS-R config sha $GOT != bundled $WANT"; exit 1; }
sha256sum weights/*.safetensors weights/*.bin 2>/dev/null | tee weights/SHA256SUMS  # recorded per box
python - <<'EOF'
from transformers import Wav2Vec2Model
m = Wav2Vec2Model.from_pretrained('/root/m5/weights')
print('XLS-R loads:', sum(p.numel() for p in m.parameters()) // 1_000_000, 'M params')
EOF
echo "SETUP-DONE"
