#!/bin/bash
# One-shot provisioning for a fresh Lambda A100 box: install uv, sync the
# project env with the creativity_rl extra (fastapi/uvicorn for the shim,
# transformers, torch 2.6.0+cu124). Run from the repo root after the repo
# has been rsynced over.
set -e
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# torch 2.6.0+cu124 only ships cp311-cp313 wheels; the box default is
# CPython 3.14. Pin 3.12 (uv fetches a managed interpreter) so both
# `uv sync` here and `uv run` in serve_hf.sh resolve against it.
uv python pin 3.12

# Full env incl. the creativity_rl extra so the shim has fastapi/uvicorn.
uv sync --extra creativity_rl

# Sanity: the shim's hard imports must resolve in the synced venv.
uv run --no-sync python - <<'PY'
import torch, fastapi, uvicorn, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("fastapi", fastapi.__version__, "uvicorn", uvicorn.__version__)
print("transformers", transformers.__version__)
PY
echo "PROVISION_OK"
