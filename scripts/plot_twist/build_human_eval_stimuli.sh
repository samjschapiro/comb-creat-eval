#!/bin/bash
# Build the stimulus pairs for the TwistBench human preference study and write them into the
# jsPsych experiment (llm_creativity_mech_interp/src/experiments/twistbench_preference/).
#
# Reads the already-built website payload, so rebuild that first if the scores or stories changed:
#   uv run python src/plot_twist/scripts/build_website_data.py configs/plot_twist/website.yaml --overwrite
uv run python src/plot_twist/scripts/build_human_eval_stimuli.py configs/plot_twist/human_eval_stimuli.yaml "$@"
