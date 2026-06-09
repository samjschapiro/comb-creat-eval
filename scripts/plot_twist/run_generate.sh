#!/bin/bash
# Generate N open-ended plot-twist stories per model (length-matched to human gold).
uv run python src/plot_twist/scripts/run_generate.py configs/plot_twist/generate_llm_twists.yaml "$@"
