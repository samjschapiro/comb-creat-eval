#!/bin/bash
# Publish the TwistBench human preference study to the standalone site repo (GitHub Pages).
#
#   https://schapiro.ai/twistbench/study/   (github.com/samjschapiro/twistbench, main branch)
#
# It lives in a SUBDIRECTORY of the project-page repo, so this script only ever touches
# `study/` — the project page at /twistbench/ and its data/ payload are left alone.
#
# Only the static files go up. Two deliberate exclusions, both about the blind:
#   server/    holds pairs.json, the story_id -> human|llm key.
#   README.md  names the LLM source and says the human stories are Gutenberg classics.
# Either one served alongside the study tells a curious participant what they are comparing.
#
# Publishing is opt-in: it needs --publish. Without it the script dry-runs and pushes nothing,
# so nothing reaches the public site as a side effect of a build or a test.
#
#   ./scripts/plot_twist/deploy_study.sh             # dry run (default)
#   ./scripts/plot_twist/deploy_study.sh --publish   # actually push
set -euo pipefail

# Override with STUDY_DIR=... to deploy from another checkout of the study branch (a worktree).
STUDY="${STUDY_DIR:-/Users/schapiro/Desktop/Experiments/llm_creativity_mech_interp/src/experiments/twistbench_preference}"
REMOTE="https://github.com/samjschapiro/twistbench.git"
CLONE="${TMPDIR:-/tmp}/twistbench-study-deploy"
SUBDIR="study"
DRY_RUN="--dry-run"
[ "${1:-}" = "--publish" ] && DRY_RUN=""

[ -f "$STUDY/index.html" ]           || { echo "FATAL: $STUDY/index.html not found."; exit 1; }
[ -f "$STUDY/js/stimuli-data.js" ]   || { echo "FATAL: stimuli not built — run build_human_eval_stimuli.sh"; exit 1; }

# Refuse to publish a payload that gives authorship away. Checked on every deploy rather than
# trusted to stay fixed: an earlier payload shipped real story ids ("anthropic_claude-sonnet-...")
# that this check, then only looking for "author_kind", let straight through. Story prose and
# option wording are blanked first, so a character who happens to read a sonnet is not a failure.
python3 - "$STUDY/js/stimuli-data.js" <<'PYCHECK'
import json, re, sys
src = open(sys.argv[1], encoding="utf-8").read()
blobs = re.findall(r"^window\.\w+ = (.*?);\s*$", src, re.M | re.S)
if len(blobs) < 3:
    sys.exit(f"FATAL: expected STORIES, STIMULUS_PAIRS and EXPERIMENT_CONFIG in {sys.argv[1]}")
def blank(o):
    if isinstance(o, dict):
        return {k: ("" if k == "text" else blank(v)) for k, v in o.items()}
    return [blank(x) for x in o] if isinstance(o, list) else o
skeleton = (src.split("window.")[0] + json.dumps([blank(json.loads(b)) for b in blobs])).lower()
bad = [t for t in ("author_kind", "anthropic", "claude", "sonnet", "__vs__", "llm_source",
                   "real_id", '"correct"', "human", "llm") if t in skeleton]
if bad:
    sys.exit(f"FATAL: js/stimuli-data.js carries {bad} outside story text. Rebuild the stimuli.")
print("blinding check passed: no authorship markers outside story text")
PYCHECK

rm -rf "$CLONE"
git clone --quiet --depth 1 "$REMOTE" "$CLONE"
mkdir -p "$CLONE/$SUBDIR"

# --delete keeps the published copy in step with the source; the excludes are what must never
# be served.
rsync -a --delete $DRY_RUN -v \
  --exclude .git --exclude server --exclude data --exclude 'vercel.json' \
  --exclude '*.sqlite' --exclude .gitignore --exclude README.md \
  "$STUDY/" "$CLONE/$SUBDIR/"

if [ -n "$DRY_RUN" ]; then
  echo
  echo "DRY RUN — nothing pushed. Files above would land in $SUBDIR/ of $REMOTE"
  echo "Re-run with --publish to actually push."
  rm -rf "$CLONE"
  exit 0
fi

git -C "$CLONE" add -A
if git -C "$CLONE" diff --cached --quiet; then
  echo "No changes to publish."
else
  git -C "$CLONE" -c user.name="${GIT_AUTHOR_NAME:-Samuel Schapiro}" \
      -c user.email="${GIT_AUTHOR_EMAIL:-jaylenwarren55@gmail.com}" \
      commit --quiet -m "Update TwistBench preference study"
  git -C "$CLONE" push --quiet origin main
  echo "Published -> https://schapiro.ai/twistbench/study/"
fi

rm -rf "$CLONE"
