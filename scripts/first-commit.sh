#!/usr/bin/env bash
# first-commit.sh — HEARSAY repository birth.
#
# RUN AT 8:00 PM FRIDAY, SEP 25, 2026 (HackGT 13 coding window opens) — NOT BEFORE.
# The script refuses to run earlier than that, by this machine's local clock.
#
# What it does, in order:
#   1. Pre-flight: time gate, no existing .git, gh authenticated, required files present.
#   2. git init with `main` as the default branch.
#   3. Stage everything .gitignore allows (including .claude/skills/ and CLAUDE.md, which
#      must travel with the repo so teammates get them on clone) and make the first commit.
#   4. Create a PRIVATE GitHub repo named `hearsay` under the authenticated gh account.
#   5. Add it as `origin` and push `main`.
# The local commit happens before the GitHub repo is created, so a failed commit never
# leaves an empty remote behind.
#
# Usage (from anywhere):  bash ~/Projects/hearsay/scripts/first-commit.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="hearsay"
NOT_BEFORE="202609252000"   # YYYYmmddHHMM, local time: Fri Sep 25 2026, 8:00 pm

die() { echo "first-commit: $*" >&2; exit 1; }

# --- 1. Pre-flight -----------------------------------------------------------------------

now="$(date +%Y%m%d%H%M)"
if (( 10#$now < 10#$NOT_BEFORE )); then
  die "refusing to run: it is $(date '+%a %b %d %Y %H:%M %Z'); this script runs at 8:00 pm Friday Sep 25 2026 or later."
fi

cd "$REPO_DIR"
[[ -f CLAUDE.md && -f pyproject.toml && -f .gitignore ]] || die "not in the HEARSAY repo root ($REPO_DIR)."
[[ -e .git ]] && die ".git already exists in $REPO_DIR — the first commit has already happened (or was started). Not touching it."
[[ -d .claude/skills ]] || die ".claude/skills/ is missing; teammates would clone a repo without the skills."

command -v gh >/dev/null || die "gh is not installed."
gh auth status >/dev/null 2>&1 || die "gh is not authenticated. Run: gh auth login"
OWNER="$(gh api user --jq .login)"
if gh repo view "$OWNER/$REPO_NAME" >/dev/null 2>&1; then
  die "github.com/$OWNER/$REPO_NAME already exists. Not creating or pushing over it."
fi

if ! git config user.name >/dev/null || ! git config user.email >/dev/null; then
  die "git user.name / user.email are not set. Set them with git config --global."
fi

missing_docs=()
for f in docs/scoping.md docs/master-doc.md; do [[ -s "$f" ]] || missing_docs+=("$f"); done
if (( ${#missing_docs[@]} )); then
  echo "first-commit: warning — missing or empty: ${missing_docs[*]}"
  read -r -p "Commit without them? [y/N] " ans
  [[ "$ans" == [yY] ]] || die "aborted; add the docs and re-run."
fi

# --- 2. git init -------------------------------------------------------------------------

git init -b main

# --- 3. Stage + commit -------------------------------------------------------------------

git add -A

# Everything that must travel with the repo is staged...
for must in CLAUDE.md README.md .gitignore .env.example pyproject.toml uv.lock scripts/first-commit.sh \
            submissions/log.csv .claude/settings.json .claude/review-plan.sh .claude/review-audit.sh \
            .claude/skills/plan-review/SKILL.md; do
  git ls-files --error-unmatch "$must" >/dev/null 2>&1 || { rm -rf .git; die "$must is not staged; aborting (removed the fresh .git)."; }
done
# ...and nothing that must never be committed is.
bad="$(git ls-files | grep -E '(^|/)\.env$|^\.venv/|\.(wav|mp3|m4a|flac|ogg)$|settings\.local\.json$|^(data|weights)/[^.]|^models/|^outputs/|^submissions/.+\.csv$' | grep -v '^submissions/log\.csv$' || true)"
if [[ -n "$bad" ]]; then
  rm -rf .git
  die "refusing to commit ignored-class files (removed the fresh .git):
$bad"
fi
big="$(git ls-files -z | xargs -0 du -k 2>/dev/null | awk '$1 > 5120 {print $2}' || true)"
if [[ -n "$big" ]]; then
  rm -rf .git
  die "refusing to commit files over 5 MB (removed the fresh .git):
$big"
fi

echo "first-commit: staging $(git ls-files | wc -l | tr -d ' ') files:"
git ls-files | sed 's/^/  /'

git commit -q -m "chore: HEARSAY scaffold, Claude Code skills, and CLAUDE.md

Pre-event environment setup: pyproject + uv.lock, .gitignore, empty
directory layout, project-level Claude Code skills (.claude/skills),
Codex review scripts + rubrics, .claude/settings.json, README, .env.example,
pytest/ruff config, CLAUDE.md, planning docs, submissions/log.csv header.
No application code.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"

# --- 4. Create the private GitHub repo ---------------------------------------------------

gh repo create "$OWNER/$REPO_NAME" --private \
  --description "HEARSAY — audio deepfake detector (HackGT 13, NSA challenge)"

# --- 5. origin + push --------------------------------------------------------------------

git remote add origin "https://github.com/$OWNER/$REPO_NAME.git"
git push -u origin main

echo
echo "first-commit: done — https://github.com/$OWNER/$REPO_NAME (private), main pushed."
echo "Next: add collaborators, e.g."
echo "  gh api -X PUT repos/$OWNER/$REPO_NAME/collaborators/<github-username> -f permission=push"
