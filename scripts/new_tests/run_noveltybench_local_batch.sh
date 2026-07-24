#!/usr/bin/env bash
# Local NoveltyBench backfill via MLX generation + DeBERTa(MPS) + gpt-4o-mini judge.
# Resumable: skips a model whose summary.json already exists.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv-mlx/bin/python
export LLM_BASE_URL=http://localhost:8080/v1
export LLM_API_KEY=EMPTY
export HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d= -f2)

# pool_key | mlx_repo   (comfortable non-reasoning tier)
MODELS=(
  "mistralai_mistral-7b-instruct-v0-1|mlx-community/mistral-7b-instruct-v0.1-4bit-ngs"
  "mistralai_mistral-nemo|mlx-community/Mistral-Nemo-Instruct-2407-4bit"
  "microsoft_phi-4|mlx-community/phi-4-4bit"
  "mistralai_mistral-small-24b-instruct-2501|mlx-community/Mistral-Small-24B-Instruct-2501-4bit"
  "google_gemma-3-27b-it|mlx-community/gemma-3-27b-it-4bit"
)
for entry in "${MODELS[@]}"; do
  key="${entry%%|*}"; repo="${entry##*|}"
  out="data/new_tests/noveltybench/${key}_local"
  cfg="configs/new_tests/nb_local/${key}.yaml"
  if [ -f "$out/summary.json" ]; then echo "[SKIP] $key (summary exists)"; continue; fi
  cat > "$cfg" <<YAML
test_model: "$repo"
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
generation_concurrency: 8
judge_concurrency: 8
output_dir: "$out"
YAML
  echo "[START $(date +%H:%M:%S)] $key  ($repo)"
  SECONDS=0
  $PY src/new_tests/scripts/run_noveltybench.py "$cfg" --overwrite 2>&1 | tail -3
  echo "[DONE $(date +%H:%M:%S), ${SECONDS}s] $key -> $(grep -o '\"mean_utility_k\":[^,]*' $out/summary.json 2>/dev/null)"
done
echo "[BATCH COMPLETE $(date +%H:%M:%S)]"
