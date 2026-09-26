#!/bin/bash
# HEARSAY image entrypoint (run spec D12): derive the thread count from the cgroup quota, pick
# up a template dropped beside the audio, verify the shipped assets, then exec the main chat's
# runner (scripts/run_pipeline.py) with /data and /out. Env the runner itself reads:
#   HEARSAY_TEAM (default HEARSAY)  HEARSAY_TEMPLATE (path inside the container)
#   HEARSAY_RULE (zmean | stack_nonlj, default zmean)
# Env this script reads: OMP_NUM_THREADS (default min(nproc, cgroup cpu quota, 6); the runner caps
# torch at 6 too, and the cap here keeps numpy/numba from oversubscribing on a big box).
# Extra arguments are passed through to the runner (e.g. --limit 50 --compare-tsv /ref/x.tsv).
set -euo pipefail

if [ -z "${OMP_NUM_THREADS:-}" ]; then
  n=$(nproc)
  if [ -r /sys/fs/cgroup/cpu.max ]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    if [ "$quota" != "max" ] && [ "$period" -gt 0 ] 2>/dev/null; then
      q=$(( (quota + period - 1) / period ))
      [ "$q" -lt "$n" ] && n=$q
    fi
  fi
  [ "$n" -gt 6 ] && n=6
  [ "$n" -ge 1 ] || n=1
  export OMP_NUM_THREADS="$n"
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

if [ -z "${HEARSAY_TEMPLATE:-}" ]; then
  tsvs=(/data/*.tsv)
  if [ "${#tsvs[@]}" -eq 1 ] && [ -f "${tsvs[0]}" ]; then
    export HEARSAY_TEMPLATE="${tsvs[0]}"
  elif [ "${#tsvs[@]}" -gt 1 ]; then
    echo "hearsay: ${#tsvs[@]} .tsv files under /data; set HEARSAY_TEMPLATE to pick one" >&2
  fi
fi

if [ -f /app/assets.json ]; then
  python /app/docker/assets.py verify --root /app --manifest /app/assets.json \
    || { echo "hearsay: shipped assets differ from the build manifest; refusing to run" >&2; exit 3; }
fi
[ -d /data ] || { echo "hearsay: /data is not mounted" >&2; exit 2; }
echo "hearsay: build=$(cat /app/BUILD_INFO 2>/dev/null || echo unknown) threads=$OMP_NUM_THREADS" \
     "team=${HEARSAY_TEAM:-HEARSAY} rule=${HEARSAY_RULE:-zmean} template=${HEARSAY_TEMPLATE:-none}"
exec python /app/scripts/run_pipeline.py --in /data --out /out --threads "$OMP_NUM_THREADS" --require-offline "$@"
