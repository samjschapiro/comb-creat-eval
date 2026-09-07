#!/bin/bash
# Publish the Kombine human generation study (frontend) to GitHub Pages at a top-level path on the
# user's site. It lives in a SUBDIRECTORY of the user Pages repo (samjschapiro.github.io, whose CNAME
# is schapiro.ai), so this only ever touches `kombine/` — the homepage and other paths are left alone.
#
#   https://schapiro.ai/kombine/
#
# The repo is large, so we do a sparse, blobless clone that materializes ONLY the kombine/ subtree —
# no other site assets are downloaded or rewritten.
#
# FRONTEND ONLY: js/experiment.js has DATA_SUBMISSION_URL='' (debug mode — responses are shown at the
# end and POSTed nowhere). Wire a data sink before sending real participants.
#
#   ./scripts/kg_creat/deploy_study.sh --dry-run   # show what would change
#   ./scripts/kg_creat/deploy_study.sh
set -euo pipefail

STUDY="/Users/schapiro/Desktop/Experiments/llm_creativity_mech_interp/src/experiments/kombine_generation"
REMOTE="https://github.com/samjschapiro/samjschapiro.github.io.git"
CLONE="${TMPDIR:-/tmp}/kombine-study-deploy"
SUBDIR="docs/kombine"   # GitHub Pages for this repo builds from /docs, so the served path is /kombine/
DRY_RUN=""
[ "${1:-}" = "--dry-run" ] && DRY_RUN="--dry-run"

[ -f "$STUDY/index.html" ]         || { echo "FATAL: $STUDY/index.html not found."; exit 1; }
[ -f "$STUDY/js/experiment.js" ]   || { echo "FATAL: $STUDY/js/experiment.js not found."; exit 1; }
[ -f "$STUDY/js/stimuli-data.js" ] || { echo "FATAL: $STUDY/js/stimuli-data.js not found."; exit 1; }

rm -rf "$CLONE"
# sparse + blobless: fetch commit/tree metadata but no file blobs until needed, and check out only
# the kombine/ cone (empty until we add files) — so the big existing site is never materialized.
git clone --quiet --depth 1 --filter=blob:none --sparse "$REMOTE" "$CLONE"
git -C "$CLONE" sparse-checkout set "$SUBDIR"
mkdir -p "$CLONE/$SUBDIR"

# --delete keeps the published copy in step with the source, scoped to kombine/ only. vercel.json is a
# Vercel artifact unused on Pages; README/.gitignore are repo hygiene, not served content.
rsync -a --delete $DRY_RUN -v \
  --exclude .git --exclude 'vercel.json' --exclude .gitignore --exclude README.md \
  "$STUDY/" "$CLONE/$SUBDIR/"

if [ -n "$DRY_RUN" ]; then
  echo
  echo "DRY RUN — nothing pushed. Files above would land in $SUBDIR/ of $REMOTE"
  rm -rf "$CLONE"
  exit 0
fi

git -C "$CLONE" add -A
if git -C "$CLONE" diff --cached --quiet; then
  echo "No changes to publish."
else
  git -C "$CLONE" -c user.name="${GIT_AUTHOR_NAME:-Samuel Schapiro}" \
      -c user.email="${GIT_AUTHOR_EMAIL:-jaylenwarren55@gmail.com}" \
      commit --quiet -m "Add/update Kombine generation study"
  git -C "$CLONE" push --quiet origin HEAD
  echo "Published -> https://schapiro.ai/kombine/"
fi

rm -rf "$CLONE"
