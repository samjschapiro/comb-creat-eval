#!/bin/bash
uv run python src/dat_eval/scripts/score_evals.py configs/dat_eval/score_evals.yaml "$@"
