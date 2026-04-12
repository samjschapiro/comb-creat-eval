#!/bin/bash
# kill_all.sh — nuclear option. Kills ANY process Claude might have spawned
# that's running eval scripts or making API calls.
#
# Use this whenever you suspect the AI has runaway processes or you want to be
# absolutely sure nothing is being charged to your API account.

set -e

echo "================================================================"
echo "KILLING ALL EVAL PROCESSES"
echo "Time: $(date)"
echo "================================================================"

PIDS=$(ps aux | grep -iE "(python.*src/(dat_eval|comb_eval)|uv run python.*src/)" | grep -v grep | awk '{print $2}' || true)

if [ -z "$PIDS" ]; then
    echo "✓ No eval processes found"
    exit 0
fi

echo "Found processes:"
ps -p $PIDS -o pid,stat,start,time,command 2>/dev/null || true

echo ""
echo "Sending SIGTERM..."
for pid in $PIDS; do
    kill "$pid" 2>/dev/null && echo "  killed $pid" || echo "  failed to kill $pid"
done

sleep 2

# Check what survived
SURVIVED=$(ps aux | grep -iE "(python.*src/(dat_eval|comb_eval)|uv run python.*src/)" | grep -v grep | awk '{print $2}' || true)
if [ -n "$SURVIVED" ]; then
    echo ""
    echo "⚠ Some processes survived SIGTERM. Sending SIGKILL..."
    for pid in $SURVIVED; do
        kill -9 "$pid" 2>/dev/null && echo "  SIGKILL'd $pid" || echo "  could not kill $pid"
    done
fi

sleep 1

# Final verification
FINAL=$(ps aux | grep -iE "(python.*src/(dat_eval|comb_eval)|uv run python.*src/)" | grep -v grep || true)
echo ""
if [ -z "$FINAL" ]; then
    echo "✓ ALL EVAL PROCESSES DEAD"
else
    echo "⚠ STILL RUNNING (manual intervention needed):"
    echo "$FINAL"
    exit 1
fi

# Also report any open API connections
NETSTAT=$(lsof -i -P 2>/dev/null | grep -iE "(openrouter|anthropic|openai)" | head -5 || true)
if [ -n "$NETSTAT" ]; then
    echo ""
    echo "⚠ Open API connections still exist (may be other apps):"
    echo "$NETSTAT"
fi
