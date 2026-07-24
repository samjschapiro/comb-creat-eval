#!/usr/bin/env bash
# NoveltyBench served backfill, models run in PARALLEL (API-bound). Resumable.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-mlx/bin/python
unset LLM_BASE_URL
export HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d= -f2)
MAXP=5
# pool_key|openrouter_id  (open, non-reasoning; mistral-7b-v0.1 dropped: no OpenRouter endpoint)
MODELS=(
  "mistralai_mistral-nemo|mistralai/mistral-nemo"
  "microsoft_phi-4|microsoft/phi-4"
  "mistralai_mistral-small-24b-instruct-2501|mistralai/mistral-small-24b-instruct-2501"
  "google_gemma-3-27b-it|google/gemma-3-27b-it"
  "meta-llama_llama-3-1-70b-instruct|meta-llama/llama-3.1-70b-instruct"
  "qwen_qwen-2-5-72b-instruct|qwen/qwen-2.5-72b-instruct"
  "nvidia_llama-3-1-nemotron-70b-instruct|nvidia/llama-3.1-nemotron-70b-instruct"
  "deepseek_deepseek-chat|deepseek/deepseek-chat"
  "deepseek_deepseek-chat-v3-0324|deepseek/deepseek-chat-v3-0324"
)
run_one() {
  local key="${1%%|*}" orid="${1##*|}"
  local out="data/new_tests/noveltybench/${key}_served" cfg="configs/new_tests/nb_served/${key}.yaml"
  if [ -f "$out/summary.json" ]; then echo "[SKIP $(date +%H:%M:%S)] $key"; return; fi
  cat > "$cfg" <<YAML
test_model: "$orid"
subset: "NB-Curated"
max_prompts: null
k: 10
temperature: 1.0
top_p: 1.0
max_tokens: 512
patience: 0.8
distinctness_method: "deberta"
distinctness_threshold: 0.5
distinctness_device: "mps"
quality_judge_model: "openai/gpt-4o-mini"
generation_concurrency: 12
judge_concurrency: 12
output_dir: "$out"
YAML
  echo "[START $(date +%H:%M:%S)] $key ($orid)"
  $PY src/new_tests/scripts/run_noveltybench.py "$cfg" --overwrite > "/tmp/nb_${key}.log" 2>&1
  echo "[DONE $(date +%H:%M:%S)] $key -> $(grep -o '\"mean_utility_k\":[^,]*' "$out/summary.json" 2>/dev/null || echo FAILED)"
}
export -f run_one; export PY
for e in "${MODELS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do wait -n; done
  run_one "$e" &
done
wait
echo "[BATCH COMPLETE $(date +%H:%M:%S)]"
