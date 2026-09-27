#!/usr/bin/env bash
# From the Mac, repo root: copy the gitignored assets the API needs to the box (about 3.3 GB).
# --copy-links turns the models/hc_selected symlink into a directory copy, which the runner accepts.
# Not copied: data/, outputs/ beyond the results dump, weights/wavlm-* (unused).
set -euo pipefail
IP=${1:?usage: sync_assets.sh <ip> [user]}
USER_=${2:-hearsay}
cd "$(dirname "$0")/../.."
ls weights/wav2vec2-xls-r-300m weights/Spectra-AASIST weights/spkrec-ecapa-voxceleb \
   models/m5_xlsr_ft_20260926-0741/model models/fusion_v2/constants.json >/dev/null && echo "assets present on the Mac"
rsync -az --stats --relative --copy-links \
  weights/wav2vec2-xls-r-300m weights/Spectra-AASIST weights/spkrec-ecapa-voxceleb \
  models/m1_wav2vec2-xls-r-300m_L7_20260926-0521 models/hc_lgbm_20260926-055451 models/hc_selected \
  models/m5_xlsr_ft_20260926-0741 models/fusion_v2 models/fusion_v1 \
  outputs/runner/v2_full/results submissions/CrossExam_predictions.tsv \
  "$USER_@$IP:~/hearsay/"
ssh "$USER_@$IP" 'cd ~/hearsay && test -f models/hc_selected/meta.json && test -f models/m5_xlsr_ft_20260926-0741/model/hashes.json && test -f models/fusion_v2/constants.json && ls outputs/runner/v2_full/results | wc -l && du -sh weights models outputs submissions && echo "assets on the box: ok"'
