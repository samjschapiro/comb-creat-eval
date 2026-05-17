#!/bin/bash
# Serve an HF checkpoint as an OpenAI-compatible endpoint on :8000.
# vLLM ships its own torch, so run it isolated with uvx (does not touch
# the project venv / our cu124 pin). Args: <hf_model_id>
set -e
MODEL="$1"
[ -z "$MODEL" ] && { echo "usage: serve_vllm.sh <hf_model_id>"; exit 1; }

pkill -f "vllm serve" 2>/dev/null || true
sleep 2

# Latest vLLM (self-consistent deps). Serving only — unrelated to the
# DARLING training stack, so no version pin needed.
nohup uvx vllm serve "$MODEL" \
  --port 8000 --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  > vllm_serve.log 2>&1 &
echo $! > vllm_serve.pid
echo "vllm serving $MODEL (pid $(cat vllm_serve.pid)); waiting for ready..."

for i in $(seq 1 120); do
  if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q "$MODEL"; then
    echo "READY: $MODEL"
    exit 0
  fi
  sleep 10
done
echo "TIMEOUT waiting for vllm; tail vllm_serve.log:"
tail -20 vllm_serve.log
exit 1
