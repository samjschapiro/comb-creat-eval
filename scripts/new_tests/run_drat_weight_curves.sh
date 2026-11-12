#!/bin/bash
# DRAT correlates vs the DRAT-N / DRAT-A weight (downstream; reads saved tables).
set -e
cd "$(dirname "$0")/../.."
.venv/bin/python src/new_tests/scripts/plot_drat_weight_curves.py configs/new_tests/drat_weight_curves_20261112.yaml "$@"
