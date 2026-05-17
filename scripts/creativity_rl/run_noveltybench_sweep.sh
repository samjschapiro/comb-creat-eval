#!/bin/bash
# NoveltyBench generalization sweep. Per model: serve it via the local
# batched server (generation), run NoveltyBench. Distinctness uses the
# local deberta classifier (no API); the quality judge (gpt-4o-mini)
# goes to OpenRouter via get_async_client_openrouter (generation stays
# on the local server via LLM_BASE_URL).
#
# Usage: run_noveltybench_sweep.sh model1 [model2 ...]
set -e
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
set -a; source .env 2>/dev/null || true; set +a   # OPENROUTER_API_KEY, HF_TOKEN
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="EMPTY"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && { echo "usage: run_noveltybench_sweep.sh <model> [model...]"; exit 1; }

for M in "${MODELS[@]}"; do
  KEY=$(echo "$M" | sed 's#[/.]#_#g')
  OUT="data/creativity_rl/noveltybench_eval/$KEY"
  echo "=== $M -> $OUT ==="
  bash scripts/creativity_rl/serve_hf.sh "$M"

  python3 - "$M" "$OUT" <<'PY'
import sys, yaml
model, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("configs/new_tests/noveltybench.yaml"))
cfg["test_model"] = model
cfg["output_dir"] = out
cfg["subset"] = "NB-Curated"
cfg["distinctness_method"] = "deberta"   # local classifier, no API
cfg["distinctness_device"] = "cpu"       # GPU is busy serving the test model
cfg["quality_judge_models"] = [          # 3 independent judges via OpenRouter
    "openai/gpt-4o-mini",
    "anthropic/claude-haiku-4-5",
    "google/gemini-2.0-flash-001",
]  # quality is judge-dependent -> ensemble, median-aggregated per generation
cfg["generation_concurrency"] = 32       # feed the batched server
yaml.safe_dump(cfg, open("configs/creativity_rl/_nb_run.yaml", "w"))
print("wrote _nb_run.yaml for", model)
PY

  uv run python src/new_tests/scripts/run_noveltybench.py configs/creativity_rl/_nb_run.yaml --overwrite
  pkill -f serve_hf_openai.py 2>/dev/null || true
  sleep 3
done
echo "ALL DONE. Results under data/creativity_rl/noveltybench_eval/"
