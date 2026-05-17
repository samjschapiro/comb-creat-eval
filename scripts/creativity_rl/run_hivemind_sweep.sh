#!/bin/bash
# Hivemind generalization sweep: for each model, serve it via the
# transformers shim and run the Hivemind harness through the local
# endpoint. SBERT embeddings (local), infinite-chats-eval prompts.
# No OpenAI/OpenRouter dependency.
#
# Usage: run_hivemind_sweep.sh model1 [model2 ...]
set -e
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="EMPTY"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && { echo "usage: run_hivemind_sweep.sh <model> [model...]"; exit 1; }

for M in "${MODELS[@]}"; do
  KEY=$(echo "$M" | sed 's#[/.]#_#g')
  OUT="data/creativity_rl/hivemind_eval/$KEY"
  echo "=== $M -> $OUT ==="
  bash scripts/creativity_rl/serve_hf.sh "$M"

  python3 - "$M" "$OUT" <<'PY'
import sys, yaml
model, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("configs/new_tests/hivemind.yaml"))
cfg["test_model"] = model
cfg["output_dir"] = out
cfg["prompt_source"] = "infinite-chats-eval"
cfg["embedding_model"] = "sentence-transformers/all-mpnet-base-v2"
cfg["generation_concurrency"] = 32  # feed the batching server
yaml.safe_dump(cfg, open("configs/creativity_rl/_hivemind_run.yaml", "w"))
print("wrote _hivemind_run.yaml for", model)
PY

  uv run python src/new_tests/scripts/run_hivemind.py configs/creativity_rl/_hivemind_run.yaml --overwrite
  pkill -f serve_hf_openai.py 2>/dev/null || true
  sleep 3
done
echo "ALL DONE. Results under data/creativity_rl/hivemind_eval/"
