#!/bin/bash
# Stage 1 then stage 2. Pass --overwrite / --debug through to both.
set -e
uv run python src/creativity_rl/scripts/build_conditioned_data.py configs/creativity_rl/conditioned_v1.yaml "$@"
uv run python src/creativity_rl/scripts/train_conditioned_sft.py configs/creativity_rl/conditioned_v1.yaml "$@"
