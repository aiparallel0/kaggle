#!/usr/bin/env bash
# aliases.sh — Project shortcuts for the Donut SROIE experiment pipeline.
#
# Usage: source aliases.sh
# To persist across sessions, add the following line to your ~/.bashrc or ~/.zshrc:
#   source /path/to/aliases.sh

# Run the full 8-experiment pipeline
alias run='python run_all.py'

# Run a single experiment by ID (e.g. exp 1)
alias runexp='python run_all.py --experiment'

# Run the evaluator standalone
alias evaluate='python donut_evaluator.py'

# Lint and format checks (mirrors CI)
alias lint='ruff check . && ruff format --check .'

# Run the full test suite
alias tests='pytest tests/ -v'
