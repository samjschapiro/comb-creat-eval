#!/bin/bash
# Score a generated stimuli set with the fixed-rubric judge.
uv run python src/plot_twist/scripts/run_rubric_stimuli.py configs/plot_twist/rubric_llm_twists.yaml "$@"
