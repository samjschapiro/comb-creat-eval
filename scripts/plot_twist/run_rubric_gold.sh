#!/bin/bash
# Score the public-domain gold set with the fixed-rubric LLM judge.
uv run python src/plot_twist/scripts/run_rubric_gold.py configs/plot_twist/rubric_gold.yaml "$@"
