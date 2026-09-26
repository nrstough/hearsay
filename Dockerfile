# HEARSAY offline CPU inference image (linux/amd64). Run spec: docs/specs/2026-09-26_k-docker-image.md.
#   docker run --network none -v <test_dir>:/data:ro -v <out_dir>:/out hearsay
# Writes /out/<HEARSAY_TEAM>_predictions.tsv (1.0 = synthetic), one JSON per file under /out/results/,
# a resumable /out/results.jsonl and run_meta.json (scripts/run_pipeline.py, main chat); never downloads.
# Layer order: deps -> weights -> shipped models -> code -> manifest, so a code edit rebuilds
# only the last small layers and never re-copies the 2.6 GB of weights.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never PIP_DISABLE_PIP_VERSION_CHECK=1 NUMBA_CACHE_DIR=/tmp/numba \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /usr/local/bin/uv
WORKDIR /app

# Locked dependencies, CPU torch. `--no-deps` is load-bearing: without it speechbrain resolves
# torch from PyPI as 2.14.0+cu130 with the whole CUDA tree and the CPU step becomes a no-op.
# The export keeps torch's pure-Python deps pinned; the grep drops only the CUDA-only lines.
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --no-dev --no-hashes --no-emit-project \
      --no-emit-package torch --no-emit-package torchaudio --no-emit-package lightgbm \
    | grep -vE '^(nvidia-|cuda-|triton)' > /app/requirements-cpu.txt \
 && uv venv /opt/venv --python 3.12 \
 && uv pip install --python /opt/venv/bin/python --no-deps -r /app/requirements-cpu.txt \
 && uv pip install --python /opt/venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchaudio==2.11.0 \
 && uv pip check --python /opt/venv/bin/python \
 && rm -rf /root/.cache/uv

# Weights (all local, never downloaded) and the shipped model bundles staged by docker/build.sh.
COPY weights/wav2vec2-xls-r-300m ./weights/wav2vec2-xls-r-300m
COPY weights/Spectra-AASIST ./weights/Spectra-AASIST
COPY weights/spkrec-ecapa-voxceleb ./weights/spkrec-ecapa-voxceleb
COPY docker/build/models ./models

# Code (editable install so hearsay.*.REPO resolves to /app; PYTHONPATH is the fallback).
COPY src ./src
COPY scripts ./scripts
COPY docker/entrypoint.sh docker/assets.py docker/parity.py ./docker/
RUN uv pip install --python /opt/venv/bin/python --no-deps -e /app && rm -rf /root/.cache/uv

ARG BUILD_INFO=""
ARG GIT_SHA="unknown"
ENV HEARSAY_GIT_SHA=$GIT_SHA
RUN printf '%s\n' "$BUILD_INFO" > /app/BUILD_INFO \
 && python docker/assets.py freeze --root /app \
      --dirs models weights/wav2vec2-xls-r-300m weights/Spectra-AASIST weights/spkrec-ecapa-voxceleb \
      --out /app/assets.json

VOLUME ["/data", "/out"]
ENTRYPOINT ["bash", "/app/docker/entrypoint.sh"]
