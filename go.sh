#!/usr/bin/env bash
# =============================================================================
# go.sh — Single-line entry point for the Vast.ai GPU training pipeline.
#
# Usage (from repo root — works on Windows Git Bash / MSYS2 and Linux/macOS):
#   bash go.sh
#   bash go.sh --dry-run          # validate without spending money
#   bash go.sh --experiment 6     # forward any flags to vastai_runner.sh
#
# Secrets (choose one approach):
#   1. Fill in .env.cloud (copy from .env.cloud.example) — simplest.
#   2. Export VASTAI_API_KEY and GITHUB_TOKEN before running.
#   3. Let go.sh prompt you interactively (no file needed).
#
# go.sh does NOT modify vastai_runner.sh — it is a thin wrapper that handles
# secret loading and interactive prompting, then delegates everything else.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve script directory in a way that works on MSYS2/Git Bash.
# "$(dirname "${BASH_SOURCE[0]}")" may return a relative path; the subshell
# cd + pwd turns it into an absolute POSIX path that MSYS2 does not mangle.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# 1. Auto-load .env.cloud (gitignored — never committed).
#    Same pattern used by vastai_runner.sh so the two scripts are consistent.
# ---------------------------------------------------------------------------
ENV_FILE="${SCRIPT_DIR}/.env.cloud"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
    echo "[go] Loaded secrets from .env.cloud"
else
    echo "[go] No .env.cloud found — will check env vars and prompt if needed."
    echo "[go] Tip: cp .env.cloud.example .env.cloud  then fill in your keys."
fi

# ---------------------------------------------------------------------------
# 2. Auto-detect GITHUB_REPO from git remote (same logic as bootstrap.sh).
# ---------------------------------------------------------------------------
if [[ -z "${GITHUB_REPO:-}" ]]; then
    _REMOTE_URL="$(git -C "$SCRIPT_DIR" remote get-url origin 2>/dev/null || true)"
    if [[ "$_REMOTE_URL" =~ github\.com[:/]([^/]+/[^/.]+) ]]; then
        export GITHUB_REPO="${BASH_REMATCH[1]}"
        echo "[go] Auto-detected GITHUB_REPO=${GITHUB_REPO}"
    else
        echo "[go] Could not detect GITHUB_REPO from git remote — will prompt if needed."
    fi
fi

# ---------------------------------------------------------------------------
# 3. Interactive prompts for any still-missing required variables.
#    Only runs when stdin is a TTY (i.e. user is at a terminal, not in CI).
# ---------------------------------------------------------------------------
_prompt_secret() {
    local var_name="$1"
    local description="$2"
    local value
    if [[ ! -e /dev/tty ]]; then
        echo "[go] ERROR: /dev/tty is not available — cannot prompt interactively." >&2
        echo "[go]   Add ${var_name} to .env.cloud or export it before running go.sh." >&2
        exit 1
    fi
    # read -s hides input (no echo) — suitable for API keys.
    # IFS= prevents leading/trailing whitespace stripping.
    IFS= read -r -s -p "[go] Enter ${description} (input hidden): " value </dev/tty
    echo "" >/dev/tty   # newline after hidden input
    printf -v "$var_name" '%s' "$value"
    export "${var_name?}"
}

if [[ -z "${VASTAI_API_KEY:-}" ]]; then
    if [[ -t 0 ]]; then
        echo "[go] VASTAI_API_KEY is not set."
        _prompt_secret VASTAI_API_KEY "VASTAI_API_KEY (from https://cloud.vast.ai/account/)"
    else
        echo "[go] ERROR: VASTAI_API_KEY is not set and stdin is not a TTY." >&2
        echo "[go]   Add it to .env.cloud or export it before running go.sh." >&2
        exit 1
    fi
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    if [[ -t 0 ]]; then
        echo "[go] GITHUB_TOKEN is not set."
        _prompt_secret GITHUB_TOKEN "GITHUB_TOKEN (GitHub PAT with 'repo' scope)"
    else
        echo "[go] ERROR: GITHUB_TOKEN is not set and stdin is not a TTY." >&2
        echo "[go]   Add it to .env.cloud or export it before running go.sh." >&2
        exit 1
    fi
fi

if [[ -z "${GITHUB_REPO:-}" ]]; then
    if [[ -t 0 ]]; then
        IFS= read -r -p "[go] Enter GITHUB_REPO (e.g. aiparallel0/kaggle): " GITHUB_REPO </dev/tty
        export GITHUB_REPO
    else
        echo "[go] ERROR: GITHUB_REPO is not set and could not be auto-detected." >&2
        echo "[go]   Add it to .env.cloud or export it before running go.sh." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 4. Print a clear banner showing what we are about to do.
# ---------------------------------------------------------------------------
_GPU="${VASTAI_GPU_NAME:-RTX 4090}"
_PRICE="${VASTAI_MAX_PRICE:-0.80}"
_MODE="${TRAINING_MODE:-micro}"
_REPO="${GITHUB_REPO}"

echo ""
echo "============================================================"
echo "  DONUT SROIE GPU Training — Vast.ai Launcher"
echo "============================================================"
echo "  Repository : ${_REPO}"
echo "  GPU target : ${_GPU} (max \$${_PRICE}/hr)"
echo "  Train mode : ${_MODE}"
echo "  Extra args : $*"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 5. Delegate to vastai_runner.sh (same directory — no assumption about cwd).
# ---------------------------------------------------------------------------
RUNNER="${SCRIPT_DIR}/vastai_runner.sh"
if [[ ! -f "$RUNNER" ]]; then
    echo "[go] ERROR: vastai_runner.sh not found at ${RUNNER}" >&2
    exit 1
fi

exec bash "$RUNNER" "$@"
