#!/usr/bin/env bash
# lint.sh — Run before every commit.
# CI runs these exact same commands and fails if they produce any output.
set -e
ruff check --fix .
ruff format .
echo "Lint OK — safe to commit."
