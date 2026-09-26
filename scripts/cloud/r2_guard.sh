#!/bin/bash
# Source this. Every R2 path M5 touches must live under r2:pa-source/hearsay/ (the bucket is
# shared with another project; its prefixes are never read or written from here).
HEARSAY_R2_PREFIX="r2:pa-source/hearsay/"

r2_path() {
  # usage: r2_path <path>  -> prints the path or fails loudly
  local p="$1"
  case "$p" in
    "${HEARSAY_R2_PREFIX}"*) printf '%s\n' "$p" ;;
    *) echo "REFUSED: R2 path '$p' is outside ${HEARSAY_R2_PREFIX}" >&2; return 1 ;;
  esac
}

r2_copy() {
  # usage: r2_copy <src> <dst> [rclone args...]; at least one side must be under the prefix,
  # and no side may be another r2: prefix.
  local src="$1" dst="$2"; shift 2
  for side in "$src" "$dst"; do
    case "$side" in
      r2:*) r2_path "$side" >/dev/null || return 1 ;;
    esac
  done
  rclone copy "$src" "$dst" "$@"
}
