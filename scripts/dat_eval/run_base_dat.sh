#!/bin/bash
# DAT on local base models as a text completion. Loads .env first: gated Hugging Face models need HF_TOKEN.
set -a && source .env && set +a
uv run python -m src.dat_eval.scripts.run_base_dat configs/dat_eval/base_models_dat_20261105.yaml "$@"
