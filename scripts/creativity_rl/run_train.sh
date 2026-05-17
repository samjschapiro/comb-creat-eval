#!/bin/bash
uv run python src/creativity_rl/scripts/train_grpo.py configs/creativity_rl/llama_1b_mcns.yaml "$@"
