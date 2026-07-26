#!/usr/bin/env bash
# Generate NoveltyBench (curated+wildchat) via OpenRouter LOCALLY (no GPU needed).
# Writes to data/new_tests/noveltybench_skywork/gen/<split>/<key>/generations.jsonl
# for later upload to a scoring GPU. Resumable. bash 3.2 safe (sleep-poll gate).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-mlx/bin/python
export OPENROUTER_API_KEY="$(grep '^OPENROUTER_API_KEY=' .env | cut -d= -f2-)"
export HF_TOKEN="$(grep '^HF_TOKEN=' .env | cut -d= -f2-)"
ROOT=data/new_tests/noveltybench_skywork/gen
GEN=scripts/new_tests/gen_noveltybench_openrouter.py

KEYS=(mistral-7b-v01 mistral-nemo phi-4 gemma-3-27b mistral-small-24b llama-3.1-70b nemotron-70b qwen-2.5-72b llama-4-scout)
ORIDS=(mistralai/mistral-7b-instruct-v0.1 mistralai/mistral-nemo microsoft/phi-4 google/gemma-3-27b-it mistralai/mistral-small-24b-instruct-2501 meta-llama/llama-3.1-70b-instruct nvidia/llama-3.1-nemotron-70b-instruct qwen/qwen-2.5-72b-instruct meta-llama/llama-4-scout)

gen_one() {
  key="$1"; orid="$2"
  for split in curated wildchat; do
    dir="$ROOT/$split/$key"; want=100; [ "$split" = wildchat ] && want=1000
    if [ -f "$dir/generations.jsonl" ] && [ "$(wc -l < "$dir/generations.jsonl")" -ge "$want" ]; then echo "[SKIP $key/$split]"; continue; fi
    echo "[GEN $key/$split ($orid)]"
    if $PY "$GEN" --model "$orid" --data "$split" --out "$dir" --k 10 --concurrency 8 >"/tmp/nbgen_${key}_${split}.log" 2>&1; then
      echo "[OK $key/$split $(wc -l < "$dir/generations.jsonl")]"
    else
      echo "[FAIL $key/$split -> $(grep -oE 'is not a valid model ID|No endpoints found[^"]*|RateLimitError|BadRequestError' "/tmp/nbgen_${key}_${split}.log" | head -1)]"
    fi
  done
}

MAXP=2
for i in "${!KEYS[@]}"; do
  gen_one "${KEYS[$i]}" "${ORIDS[$i]}" &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 5; done
done
wait
echo "[GEN_ALL_DONE]"
