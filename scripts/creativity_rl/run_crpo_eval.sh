#!/bin/bash
# CrPO-vs-base eval: for each model, serve it via vLLM and run DAT/CDAT/PACE
# through the existing dat_eval harness pointed at the local endpoint.
# Scoring (score_evals) is local and uses cached embeddings.
#
# Usage: run_crpo_eval.sh model1 [model2 ...]
set -e
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
set -a; source .env 2>/dev/null || true; set +a
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="EMPTY"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && { echo "usage: run_crpo_eval.sh <model> [model...]"; exit 1; }

for M in "${MODELS[@]}"; do
  KEY=$(echo "$M" | sed 's#[/.]#_#g')
  OUT="data/creativity_rl/crpo_eval/$KEY"
  echo "=== $M -> $OUT ==="
  bash scripts/creativity_rl/serve_hf.sh "$M"

  # One-off dat_eval config from the canonical template: this model only,
  # one temp per eval to keep the first pass cheap.
  python3 - "$M" "$OUT" <<'PY'
import sys, yaml
model, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("configs/dat_eval/run_evals.yaml"))
cfg["models"] = [model]
cfg["output_dir"] = out
cfg["dat_temperatures"] = [1.0]
cfg["cdat_temperatures"] = [1.0]
cfg["pace_temperatures"] = [0.0]
cfg["budget_usd"] = 5.0
yaml.safe_dump(cfg, open("configs/creativity_rl/_crpo_run.yaml", "w"))
print("wrote configs/creativity_rl/_crpo_run.yaml for", model)
PY

  uv run python src/dat_eval/scripts/run_evals.py configs/creativity_rl/_crpo_run.yaml --overwrite
  pkill -f serve_hf_openai.py 2>/dev/null || true
  sleep 3
done
echo "ALL DONE. Score with score_evals.py over data/creativity_rl/crpo_eval/*"
