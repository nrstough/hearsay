#!/bin/bash
set -euo pipefail

# Generic (global) Codex audit.
# A project's own .claude/review-audit.sh takes precedence when it exists — this is the
# fallback used from any project that doesn't ship one. Both resolve their rubric the
# same way: the project's .claude/codex-audit-prompt.md first, ~/.claude/codex-audit-prompt.md second.

# --- Validate inputs ---
if [[ $# -lt 1 ]] || [[ ! -f "$1" ]]; then
  echo "Usage: review-audit.sh <path-to-record-or-spec.md>" >&2
  echo "Error: File not found or not specified." >&2
  exit 1
fi

INPUT_FILE="$1"

# --- Resolve the main repo root ---
# In a worktree, --show-toplevel gives the WORKTREE root. Shared .claude/ config lives in
# the MAIN repo, so derive it from --git-common-dir (always the real repo's .git).
if GIT_COMMON_DIR="$(git rev-parse --git-common-dir 2>/dev/null)"; then
  MAIN_REPO="$(cd "$GIT_COMMON_DIR/.." && pwd)"
else
  MAIN_REPO="$PWD"   # not a git repo — degrade gracefully
fi

# --- Resolve the rubric: project first, global fallback ---
PROMPT_FILE=""
for candidate in "$MAIN_REPO/.claude/codex-audit-prompt.md" "$HOME/.claude/codex-audit-prompt.md"; do
  if [[ -f "$candidate" ]]; then PROMPT_FILE="$candidate"; break; fi
done

if [[ -z "$PROMPT_FILE" ]]; then
  echo "Error: No audit prompt found." >&2
  echo "  Looked in: $MAIN_REPO/.claude/codex-audit-prompt.md" >&2
  echo "             $HOME/.claude/codex-audit-prompt.md" >&2
  exit 1
fi

OUT_FILE="${INPUT_FILE%.md}-audit.md"

# --- Model pin (override per-invocation with CODEX_MODEL=<slug>) ---
# gpt-6-astra (GPT-6, rolled out 2026-09-03) needs codex-cli >= 0.153.1 to carry
# its metadata; an older CLI fails with a 400 "requires a newer version of Codex"
# (verified failing on 0.145.0, 2026-09-15 — 0.154.0 works). The failure branch
# below names the fix. Previous known-good pin: gpt-5.6-sol.
CODEX_MODEL="${CODEX_MODEL:-gpt-6-astra}"

# --- Load prompt and input content ---
SYSTEM_PROMPT=$(cat "$PROMPT_FILE")
INPUT_CONTENT=$(cat "$INPUT_FILE")

TMP_OUTPUT=$(mktemp)
CODEX_LOG=$(mktemp)
trap 'rm -f "$TMP_OUTPUT" "$CODEX_LOG"' EXIT

# --- Call Codex CLI (account-based auth, no API key needed) ---
echo "Sending audit input to Codex CLI (model: $CODEX_MODEL, rubric: $PROMPT_FILE)..."

COMBINED_PROMPT="${SYSTEM_PROMPT}

# Change Record to Audit:

${INPUT_CONTENT}"

set +e
echo "$COMBINED_PROMPT" | codex exec -m "$CODEX_MODEL" \
  --sandbox read-only \
  -o "$TMP_OUTPUT" \
  - 2>&1 | tee "$CODEX_LOG"
CODEX_RC=${PIPESTATUS[0]}
set -e

RESULT=$(cat "$TMP_OUTPUT")

# Success is "-o produced content", NOT $? — codex exec exits 0 even on a 400.
# Only diagnose when the output is empty: the log also carries model prose and repo
# text the agent read, so anything matched there is untrustworthy as a failure signal
# (an input quoting the upgrade error once tripped this and threw away a good result).
if [[ -z "$RESULT" ]]; then
  if grep -qE '^ERROR: .*"invalid_request_error".*requires a newer version of Codex' "$CODEX_LOG"; then
    echo "Error: model '$CODEX_MODEL' needs a newer codex-cli than $(codex --version)." >&2
    echo "  Fix:      sudo npm install -g @openai/codex@latest" >&2
    echo "  Fallback: CODEX_MODEL=gpt-5.6-sol bash $0 \"$INPUT_FILE\"" >&2
    exit 1
  fi
  # Any other API rejection (unsupported slug, auth, quota): surface its message
  # rather than the useless bare "empty output".
  API_ERR=$(grep -m1 -oE '"message":"[^"]+"' "$CODEX_LOG" | head -1 | cut -d'"' -f4 || true)
  if [[ -n "$API_ERR" ]]; then
    echo "Error: Codex rejected the request (model '$CODEX_MODEL'): $API_ERR" >&2
    exit 1
  fi
  if [[ $CODEX_RC -ne 0 ]]; then
    echo "Error: Codex CLI call failed" >&2
    exit 1
  fi
  echo "Warning: Codex produced empty audit output." >&2
  exit 2
fi

echo "$RESULT" > "$OUT_FILE"
echo "Audit written to: $OUT_FILE"
