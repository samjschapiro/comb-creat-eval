#!/bin/bash
# Serve an HF checkpoint via the transformers-backed OpenAI shim on :8000
# (project env, torch 2.6.0+cu124 — works on driver 570; no vLLM).
# Args: <hf_model_id>
set -e
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
MODEL="$1"
[ -z "$MODEL" ] && { echo "usage: serve_hf.sh <hf_model_id>"; exit 1; }

pkill -f serve_hf_openai.py 2>/dev/null || true
sleep 2
rm -f hf_serve.log
nohup bash -c "set -a; source .env; set +a; uv run python src/creativity_rl/scripts/serve_hf_openai.py --model '$MODEL' --port 8000" > hf_serve.log 2>&1 &
echo $! > hf_serve.pid
echo "serving $MODEL (pid $(cat hf_serve.pid)); waiting for ready..."

for i in $(seq 1 90); do
  if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "$MODEL"; then
    echo "READY: $MODEL"
    exit 0
  fi
  if grep -q "Traceback\|Error\|RuntimeError" hf_serve.log 2>/dev/null; then
    echo "FAILED; tail hf_serve.log:"; tail -15 hf_serve.log; exit 1
  fi
  sleep 10
done
echo "TIMEOUT; tail hf_serve.log:"; tail -15 hf_serve.log; exit 1
