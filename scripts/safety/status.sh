#!/bin/bash
# status.sh — show everything Claude might have running.
# Run this whenever you want to verify the AI isn't doing something behind your back.

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "================================================================"
echo "AI OPERATIONS STATUS"
echo "Time: $(date)"
echo "================================================================"

echo ""
echo "## 1. RUNNING PYTHON / EVAL PROCESSES"
echo "----------------------------------------------------------------"
RUNNING=$(ps aux | grep -iE "(python.*src/(dat_eval|comb_eval)|uv run python)" | grep -v grep || true)
if [ -z "$RUNNING" ]; then
    echo "  ✓ No eval processes running"
else
    echo "  ⚠ PROCESSES STILL RUNNING:"
    echo "$RUNNING" | awk '{printf "    PID %s  age=%s  cmd=%s\n", $2, $9, substr($0, index($0,$11))}'
fi

echo ""
echo "## 2. RECENT FILE ACTIVITY (last 60 seconds in data/)"
echo "----------------------------------------------------------------"
RECENT=$(find "$REPO_ROOT/data" -type f -mmin -1 2>/dev/null || true)
if [ -z "$RECENT" ]; then
    echo "  ✓ No files modified in last 60 seconds"
else
    echo "  ⚠ FILES BEING WRITTEN RIGHT NOW:"
    echo "$RECENT" | sed 's|^|    |'
fi

echo ""
echo "## 3. ALL DATA WRITTEN IN LAST 10 MINUTES"
echo "----------------------------------------------------------------"
RECENT_10=$(find "$REPO_ROOT/data" -type f -mmin -10 2>/dev/null | head -20 || true)
if [ -z "$RECENT_10" ]; then
    echo "  ✓ No file activity in last 10 minutes"
else
    echo "$RECENT_10" | sed 's|^|    |'
    COUNT=$(find "$REPO_ROOT/data" -type f -mmin -10 2>/dev/null | wc -l | tr -d ' ')
    echo "    (total: $COUNT files)"
fi

echo ""
echo "## 4. EVAL PROGRESS BY MODEL"
echo "----------------------------------------------------------------"
if [ -d "$REPO_ROOT/data/dat_eval/run_v1" ]; then
    for d in "$REPO_ROOT/data/dat_eval/run_v1"/*/; do
        [ -d "$d" ] || continue
        model=$(basename "$d")
        n=$(ls "$d"/*.json 2>/dev/null | wc -l | tr -d ' ')
        last=$(stat -f "%Sm" -t "%H:%M:%S" "$d" 2>/dev/null || echo "?")
        echo "    $model: $n files, last_mod=$last"
    done
else
    echo "    (no dat_eval data dir)"
fi

echo ""
echo "## 5. NETWORK CONNECTIONS TO OPENROUTER (live API calls)"
echo "----------------------------------------------------------------"
NETSTAT=$(lsof -i -P 2>/dev/null | grep -iE "(openrouter|anthropic|openai)" | head -5 || true)
if [ -z "$NETSTAT" ]; then
    echo "  ✓ No active connections to LLM APIs"
else
    echo "  ⚠ ACTIVE LLM API CONNECTIONS:"
    echo "$NETSTAT" | sed 's|^|    |'
fi

echo ""
echo "================================================================"
echo "If anything looks wrong, run: bash scripts/safety/kill_all.sh"
echo "================================================================"
