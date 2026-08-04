#!/bin/bash
# Publish website/twistbench/ to the gh-pages branch (GitHub Pages).
#
# The site is served at https://samjschapiro.github.io/comb-creat-eval/. The generated
# data/ payload is gitignored on main and lives only here, so rebuild it first if the
# scores or stories changed:
#
#   uv run python src/plot_twist/scripts/build_website_data.py configs/plot_twist/website.yaml --overwrite
#
# Uses a throwaway worktree so the checked-out branch and working tree are never touched.
set -euo pipefail

SITE="website/twistbench"
WORKTREE=".git/tmp-gh-pages"

[ -f "$SITE/index.html" ] || { echo "FATAL: $SITE/index.html not found — run from the repo root."; exit 1; }
[ -f "$SITE/data/leaderboard.json" ] || { echo "FATAL: $SITE/data not built — see build_website_data.py above."; exit 1; }

git fetch origin gh-pages --quiet 2>/dev/null || true
rm -rf "$WORKTREE"
if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
  git worktree add --quiet "$WORKTREE" -B gh-pages origin/gh-pages
else
  git worktree add --quiet --detach "$WORKTREE"
  git -C "$WORKTREE" checkout --orphan gh-pages
  git -C "$WORKTREE" rm -rf --quiet . 2>/dev/null || true
fi

# Mirror the site into the worktree (delete removes files dropped since last deploy).
rsync -a --delete --exclude .git "$SITE/" "$WORKTREE/"

git -C "$WORKTREE" add -A
if git -C "$WORKTREE" diff --cached --quiet; then
  echo "No changes to publish."
else
  git -C "$WORKTREE" commit --quiet -m "Deploy TwistBench site"
  git -C "$WORKTREE" push --quiet origin gh-pages
  echo "Published -> https://samjschapiro.github.io/comb-creat-eval/"
fi

git worktree remove --force "$WORKTREE"
