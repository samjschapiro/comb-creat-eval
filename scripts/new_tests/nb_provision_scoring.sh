#!/usr/bin/env bash
# Provision a fresh Lambda instance for NoveltyBench SCORING only (generation is done
# locally via OpenRouter). Encodes every fix from the 2026-07-24 session, bf16 config.
set -euo pipefail
source ~/.creds

echo "=== [1/4] clone repo ==="
cd ~; rm -rf novelty-bench
git clone --depth 1 https://github.com/novelty-bench/novelty-bench 2>&1 | tail -1

echo "=== [2/4] pip deps (transformers PINNED 4.48.3; jinja2 forced >=3.1) ==="
pip install -q --disable-pip-version-check \
  "transformers==4.48.3" accelerate datasets openai numpy pydantic aiofiles tiktoken \
  scikit-learn sacrebleu rouge-score bert_score evaluate sentencepiece protobuf >/tmp/pip1.log 2>&1
pip install -q --disable-pip-version-check --upgrade --force-reinstall --no-deps "jinja2==3.1.4" >/tmp/pip2.log 2>&1
python3 -c "import transformers, jinja2, sentencepiece; print('transformers', transformers.__version__, '| jinja2', jinja2.__version__)"

echo "=== [3/4] patch score.py (KEEP bf16; add max_length guard vs O(seq^2) OOM) ==="
cd ~/novelty-bench
python3 - <<'PY'
p='src/score.py'; s=open(p).read()
if 'max_length=2048' not in s:
    s=s.replace('tokenize=True,\n        padding=True,\n        truncation=True,',
                'tokenize=True,\n        padding=True,\n        truncation=True,\n        max_length=2048,')
open(p,'w').write(s)
print('  bf16 kept:', 'torch_dtype=torch.bfloat16' in s, '| max_length:', 'max_length=2048' in s, '| no 8bit:', 'load_in_8bit' not in s)
PY

echo "=== [4/4] download Skywork-Reward-Gemma-2-27B-v0.2 (~54GB) ==="
python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('Skywork/Skywork-Reward-Gemma-2-27B-v0.2', token=os.environ['HF_TOKEN'])
print('skywork downloaded')
"
echo "=== PROVISION_COMPLETE ==="
