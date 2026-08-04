#!/bin/bash
# Publish website/twistbench/ to the standalone site repo (GitHub Pages).
#
#   https://schapiro.ai/twistbench/   (github.com/samjschapiro/twistbench, main branch)
#
# The site repo holds ONLY the built page. Its generated data/ payload is gitignored in
# this repo, so rebuild it first if the scores or stories changed:
#
#   uv run python src/plot_twist/scripts/build_website_data.py configs/plot_twist/website.yaml --overwrite
#
# The README at the site repo root is maintained there, not here, so it is preserved.
set -euo pipefail

SITE="website/twistbench"
REMOTE="https://github.com/samjschapiro/twistbench.git"
CLONE="${TMPDIR:-/tmp}/twistbench-deploy"

[ -f "$SITE/index.html" ] || { echo "FATAL: $SITE/index.html not found — run from the repo root."; exit 1; }
[ -f "$SITE/data/leaderboard.json" ] || { echo "FATAL: $SITE/data not built — see build_website_data.py above."; exit 1; }

rm -rf "$CLONE"
git clone --quiet --depth 1 "$REMOTE" "$CLONE"

# Mirror the built site in. --delete drops files removed since the last deploy. The
# exclude is ANCHORED (leading slash) so only the site repo's own root README survives —
# an unanchored "README.md" would also drop static/authors/README.md.
rsync -a --delete --exclude .git --exclude /README.md "$SITE/" "$CLONE/"

git -C "$CLONE" add -A
if git -C "$CLONE" diff --cached --quiet; then
  echo "No changes to publish."
else
  git -C "$CLONE" commit --quiet -m "Update TwistBench site"
  git -C "$CLONE" push --quiet origin main
  echo "Published -> https://schapiro.ai/twistbench/"
fi

rm -rf "$CLONE"
