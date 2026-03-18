#!/usr/bin/env bash
# bootstrap.sh — One-command setup for fresh Vast.ai / cloud GPU instances.
#
# Usage:
#   git clone https://github.com/aiparallel0/kaggle.git && cd kaggle && bash bootstrap.sh
#   # Or with args forwarded to run_all.py:
#   bash bootstrap.sh --experiment 6
#   bash bootstrap.sh --skip-trocr
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[bootstrap] Setting up DONUT SROIE pipeline..."

# 1. Auto-detect GITHUB_REPO from git remote (if not already set)
if [ -z "${GITHUB_REPO:-}" ]; then
    REMOTE_URL=$(git remote get-url origin 2>/dev/null || true)
    if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+/[^/.]+) ]]; then
        export GITHUB_REPO="${BASH_REMATCH[1]}"
        echo "[bootstrap] Auto-detected GITHUB_REPO=$GITHUB_REPO"
    fi
fi

# 2. Set workspace (Vast.ai default: /workspace)
export DONUT_WORKSPACE="${DONUT_WORKSPACE:-/workspace}"
mkdir -p "$DONUT_WORKSPACE" 2>/dev/null || true

# 3. Copy token files from /workspace if they exist (Vast.ai convention)
for f in hf_token.txt anthropic_api_key.txt mistral_api_key.txt github_token.txt; do
    if [ -f "/workspace/$f" ] && [ ! -f "$SCRIPT_DIR/$f" ]; then
        cp "/workspace/$f" "$SCRIPT_DIR/$f"
        echo "[bootstrap] Copied $f from /workspace"
    fi
done

# 4. Read tokens from files into env vars if not already set
if [ -z "${GITHUB_TOKEN:-}" ] && [ -f "$SCRIPT_DIR/github_token.txt" ]; then
    export GITHUB_TOKEN="$(cat "$SCRIPT_DIR/github_token.txt")"
    echo "[bootstrap] Loaded GITHUB_TOKEN from github_token.txt"
fi
if [ -z "${HF_TOKEN:-}" ] && [ -f "$SCRIPT_DIR/hf_token.txt" ]; then
    export HF_TOKEN="$(cat "$SCRIPT_DIR/hf_token.txt")"
    echo "[bootstrap] Loaded HF_TOKEN from hf_token.txt"
fi

# 5. Enable AI diagnostics and auto-push
export AI_DIAGNOSE="${AI_DIAGNOSE:-1}"
export AI_DIAGNOSE_PROVIDER="${AI_DIAGNOSE_PROVIDER:-auto}"
export AUTOPUSH_RESULTS="${AUTOPUSH_RESULTS:-1}"

# 6. Install minimal deps
echo "[bootstrap] Installing dependencies..."
pip install -q -r requirements.txt

# 7. Validate import chain
echo "[bootstrap] Validating import chain..."
python -c "from constants import FIELDS, BASE_MODEL, SEED" || {
    echo "[bootstrap] FATAL: import chain broken"
    exit 2
}

# 8. Run full pipeline (forward all CLI args)
echo "[bootstrap] Starting pipeline..."
python run_all.py --auto-push "$@"
