#!/usr/bin/env bash
# CPU-only virtualenv from the lockfile, as the Dockerfile does it. On Linux the lock resolves torch
# to 2.14.0+cu130 with the whole CUDA tree, so torch/torchaudio are installed from the PyTorch CPU
# index and lightgbm (training only; never imported at inference) is skipped. `--no-deps` is
# load-bearing: with it, nothing re-resolves torch from PyPI. Run from the repo root as `hearsay`.
set -euo pipefail
UV=${UV:-$HOME/.local/bin/uv}
TMP=$(mktemp -d)
"$UV" export --frozen --no-dev --no-hashes --no-emit-project \
    --no-emit-package torch --no-emit-package torchaudio --no-emit-package lightgbm \
  | grep -vE '^(nvidia-|cuda-|triton)' > "$TMP/requirements-cpu.txt"
"$UV" export --frozen --only-dev --no-hashes --no-emit-project \
  | grep -vE '^(nvidia-|cuda-|triton)' > "$TMP/requirements-dev.txt"
[ -x .venv/bin/python ] || "$UV" venv .venv --python 3.12
"$UV" pip install --python .venv/bin/python --no-deps -r "$TMP/requirements-cpu.txt"
"$UV" pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchaudio==2.11.0
"$UV" pip install --python .venv/bin/python --no-deps -r "$TMP/requirements-dev.txt"
"$UV" pip check --python .venv/bin/python
"$UV" pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c "import torch, hearsay.api; print('torch', torch.__version__, 'cuda' if torch.cuda.is_available() else 'cpu', '| hearsay.api imports')"
rm -rf "$TMP"
