#!/usr/bin/env bash
# Score all uploaded models on the GPU: partition (DeBERTa) + score (bf16 Skywork),
# for both splits, then combine into the 1,100-prompt union utility per model.
set -uo pipefail
source ~/.creds; cd ~/novelty-bench
export PYTHONPATH=~/novelty-bench OPENAI_API_KEY=sk-dummy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(mistral-nemo phi-4 gemma-3-27b mistral-small-24b llama-3.1-70b qwen-2.5-72b llama-4-scout)

for k in "${MODELS[@]}"; do
  for split in curated wildchat; do
    dir="results/$split/$k"
    [ -f "$dir/generations.jsonl" ] || { echo "[MISSING $dir]"; continue; }
    if [ ! -s "$dir/partitions.jsonl" ]; then
      echo "[PARTITION $k/$split]"; python3 src/partition.py --eval-dir "$dir" --alg classifier >"/tmp/part_${k}_${split}.log" 2>&1
    fi
    if [ ! -s "$dir/scores.jsonl" ]; then
      echo "[SCORE $k/$split]"; python3 src/score.py --eval-dir "$dir" --patience 0.8 >"/tmp/score_${k}_${split}.log" 2>&1
    fi
  done
done

echo "=== UNION UTILITIES (paper llama-3.1-8b ref: 3.76) ==="
python3 - <<'PY'
import json, statistics
MODELS=["mistral-nemo","phi-4","gemma-3-27b","mistral-small-24b","llama-3.1-70b","qwen-2.5-72b","llama-4-scout"]
out={}
for k in MODELS:
    rows=[]
    for split in ("curated","wildchat"):
        p=f"results/{split}/{k}/scores.jsonl"
        try: rows+=[json.loads(l) for l in open(p)]
        except FileNotFoundError: pass
    if not rows: print(f"  {k:20s} NO SCORES"); continue
    u=statistics.mean(r["utility"] for r in rows)
    d=statistics.mean(r["distinct"] for r in rows)+1
    out[k]={"utility":round(u,4),"distinct10":round(d,3),"n_prompts":len(rows)}
    print(f"  {k:20s} n={len(rows):5d}  UTILITY={u:.3f}  DISTINCT10={d:.2f}")
json.dump(out, open("union_utilities.json","w"), indent=2)
print("\nwrote union_utilities.json")
PY
echo "=== SCORE_ALL_DONE ==="
