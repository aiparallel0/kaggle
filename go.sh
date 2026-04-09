#!/usr/bin/env bash
# =============================================================================
# go.sh — Fully autonomous GPU training pipeline with self-healing loop.
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
# Self-healing loop:
#   go.sh provisions a GPU, runs training.  If training fails the GPU workflow
#   creates a GitHub Issue for Copilot which opens a fix PR.  go.sh monitors
#   the repo for new commits on main (indicating a merged fix) and automatically
#   re-provisions a fresh GPU to re-run training — up to MAX_GPU_RERUNS times
#   or MAX_TOTAL_HOURS wall-clock time.  No human intervention required.
#
# Loop flow:
#   1. Provision Vast.ai GPU → register ephemeral runner → dispatch workflow
#   2. Wait for job to finish → destroy GPU
#   3. Check training outcome (via workflow run status)
#   4. If success → exit 0
#   5. If failure → wait for Copilot to merge a fix (poll main for new commits)
#   6. On new commit detected → go to step 1 (re-provision fresh GPU)
#   7. After MAX_GPU_RERUNS or MAX_TOTAL_HOURS → exit with summary
#
# go.sh does NOT modify vastai_runner.sh — it is a wrapper that handles
# secret loading, interactive prompting, and the re-provisioning loop.
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
# 5. Self-healing loop configuration
# ---------------------------------------------------------------------------
MAX_GPU_RERUNS="${MAX_GPU_RERUNS:-5}"
MAX_TOTAL_HOURS="${MAX_TOTAL_HOURS:-8}"
REPROVISION_COOLDOWN="${REPROVISION_COOLDOWN:-60}"
FIX_POLL_INTERVAL="${FIX_POLL_INTERVAL:-120}"
FIX_POLL_TIMEOUT="${FIX_POLL_TIMEOUT:-3600}"

RUNNER="${SCRIPT_DIR}/vastai_runner.sh"
if [[ ! -f "$RUNNER" ]]; then
    echo "[go] ERROR: vastai_runner.sh not found at ${RUNNER}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Helper: get HEAD SHA of a remote branch via GitHub API (no local git needed)
# ---------------------------------------------------------------------------
_get_remote_sha() {
    local branch="${1:-main}"
    local sha=""
    local attempt
    for attempt in 1 2 3; do
        sha=$(curl -sS \
            -H "Accept: application/vnd.github+json" \
            -H "Authorization: token ${GITHUB_TOKEN}" \
            "https://api.github.com/repos/${GITHUB_REPO}/commits/${branch}" 2>/dev/null \
        | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('sha',''))" 2>/dev/null) \
        || sha=""
        if [[ -n "$sha" ]]; then
            echo "$sha"
            return
        fi
        [[ $attempt -lt 3 ]] && sleep 5
    done
    echo "[go] WARNING: GitHub API call failed for ${branch} SHA after 3 attempts — check GITHUB_TOKEN is valid and network is reachable" >&2
    echo ""
}

# ---------------------------------------------------------------------------
# Helper: get the most recent workflow run status for gpu_training.yml
# Returns: "success", "failure", "in_progress", or "unknown"
# ---------------------------------------------------------------------------
_get_latest_training_status() {
    local response
    response=$(curl -sSf \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        "https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/gpu_training.yml/runs?per_page=1&branch=main" 2>/dev/null) || { echo "unknown"; return; }

    python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
runs = data.get('workflow_runs', [])
if not runs:
    print('unknown')
else:
    r = runs[0]
    if r.get('status') != 'completed':
        print('in_progress')
    else:
        print(r.get('conclusion', 'unknown'))
" <<< "$response" 2>/dev/null || echo "unknown"
}

# ---------------------------------------------------------------------------
# Helper: read auto_fix_state.json from remote main branch
# Returns iteration count (0 = no failures or reset)
# ---------------------------------------------------------------------------
_get_remote_iteration() {
    local content
    content=$(curl -sSf \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: token ${GITHUB_TOKEN}" \
        "https://api.github.com/repos/${GITHUB_REPO}/contents/.github/auto_fix_state.json?ref=main" 2>/dev/null) || { echo "0"; return; }

    python3 -c "
import json, sys, base64
data = json.loads(sys.stdin.read())
content = base64.b64decode(data.get('content', '')).decode()
state = json.loads(content)
print(state.get('iteration', 0))
" <<< "$content" 2>/dev/null || echo "0"
}

# ---------------------------------------------------------------------------
# 6. Main self-healing loop
#
# Flow per iteration:
#   a. Record current HEAD of main
#   b. Run vastai_runner.sh (provisions GPU, runs training, destroys GPU)
#   c. Check outcome: if training succeeded → exit 0
#   d. If training failed → wait for Copilot to merge a fix
#      (poll main branch for new commits past the recorded SHA)
#   e. On new commit → re-provision GPU (next iteration)
#   f. On timeout / max iterations → exit with summary
# ---------------------------------------------------------------------------
LOOP_START=$(date +%s)
ATTEMPT=0
DRY_RUN_FLAG=""
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN_FLAG="--dry-run"; done

echo ""
echo "[go] Self-healing loop: max ${MAX_GPU_RERUNS} GPU runs, ${MAX_TOTAL_HOURS}h wall-clock limit"
echo "[go] Fix poll: every ${FIX_POLL_INTERVAL}s, timeout ${FIX_POLL_TIMEOUT}s"
echo ""

while true; do
    ATTEMPT=$((ATTEMPT + 1))

    # ── Check limits ──────────────────────────────────────────────────────
    if [[ $ATTEMPT -gt $MAX_GPU_RERUNS ]]; then
        echo ""
        echo "[go] ════════════════════════════════════════════════════════"
        echo "[go]  Max GPU re-provisions reached ($MAX_GPU_RERUNS)."
        echo "[go]  The self-healing loop did not resolve the issue."
        echo "[go]  Check GitHub Issues for the latest failure context."
        echo "[go] ════════════════════════════════════════════════════════"
        exit 1
    fi

    ELAPSED_HOURS=$(( ($(date +%s) - LOOP_START) / 3600 ))
    if [[ $ELAPSED_HOURS -ge $MAX_TOTAL_HOURS ]]; then
        echo ""
        echo "[go] ════════════════════════════════════════════════════════"
        echo "[go]  Wall-clock limit reached (${MAX_TOTAL_HOURS}h)."
        echo "[go]  Exiting — check GitHub for current pipeline status."
        echo "[go] ════════════════════════════════════════════════════════"
        exit 1
    fi

    echo ""
    echo "┌────────────────────────────────────────────────────────┐"
    echo "│  GPU Run $ATTEMPT of $MAX_GPU_RERUNS"
    echo "│  Elapsed: ${ELAPSED_HOURS}h of ${MAX_TOTAL_HOURS}h max"
    echo "└────────────────────────────────────────────────────────┘"

    # ── Record main HEAD before this run ──────────────────────────────────
    PRE_RUN_SHA=$(_get_remote_sha main)
    echo "[go] main HEAD before run: ${PRE_RUN_SHA:0:12}"

    # ── Run vastai_runner.sh ──────────────────────────────────────────────
    set +e
    bash "$RUNNER" "$@"
    RUNNER_EXIT=$?
    set -e

    echo "[go] vastai_runner.sh exited with code $RUNNER_EXIT"

    # ── Dry-run: exit immediately ─────────────────────────────────────────
    if [[ -n "$DRY_RUN_FLAG" ]]; then
        echo "[go] Dry-run mode — exiting after first run."
        exit $RUNNER_EXIT
    fi

    # ── Check if training succeeded ───────────────────────────────────────
    # GitHub Actions API takes ~10-15s to update workflow run status after
    # the runner exits.  Wait briefly before querying.
    sleep 15
    TRAINING_STATUS=$(_get_latest_training_status)
    echo "[go] Latest gpu_training.yml status: $TRAINING_STATUS"

    if [[ "$TRAINING_STATUS" == "success" ]]; then
        echo ""
        echo "[go] ════════════════════════════════════════════════════════"
        echo "[go]  ✅ Training succeeded on attempt $ATTEMPT!"
        echo "[go] ════════════════════════════════════════════════════════"
        exit 0
    fi

    # ── Check if auto-fix loop is exhausted ───────────────────────────────
    ITERATION=$(_get_remote_iteration)
    MAX_ITER="${MAX_AUTO_FIX_ITERATIONS:-5}"
    echo "[go] Auto-fix iteration: $ITERATION / $MAX_ITER"

    if [[ "$ITERATION" -ge "$MAX_ITER" ]]; then
        echo ""
        echo "[go] ════════════════════════════════════════════════════════"
        echo "[go]  Auto-fix loop exhausted ($ITERATION >= $MAX_ITER)."
        echo "[go]  Human intervention required."
        echo "[go] ════════════════════════════════════════════════════════"
        exit 1
    fi

    # ── Wait for Copilot to merge a fix (poll for new commits on main) ────
    echo ""
    echo "[go] Training failed. Waiting for Copilot to merge a fix..."
    echo "[go] Polling main branch every ${FIX_POLL_INTERVAL}s (timeout: ${FIX_POLL_TIMEOUT}s)"

    FIX_WAIT_START=$(date +%s)
    FIX_FOUND=false

    while true; do
        FIX_ELAPSED=$(( $(date +%s) - FIX_WAIT_START ))

        if [[ $FIX_ELAPSED -ge $FIX_POLL_TIMEOUT ]]; then
            echo "[go] Fix poll timeout (${FIX_POLL_TIMEOUT}s) — no new commits detected."
            break
        fi

        CURRENT_SHA=$(_get_remote_sha main)

        if [[ -n "$CURRENT_SHA" && "$CURRENT_SHA" != "$PRE_RUN_SHA" ]]; then
            echo "[go] ✅ New commit on main: ${CURRENT_SHA:0:12} (was ${PRE_RUN_SHA:0:12})"
            echo "[go] A fix was likely merged — will re-provision GPU."
            FIX_FOUND=true
            break
        fi

        # Show progress
        REMAINING=$(( FIX_POLL_TIMEOUT - FIX_ELAPSED ))
        echo "[go]   ...waiting (${FIX_ELAPSED}s / ${FIX_POLL_TIMEOUT}s, main still at ${CURRENT_SHA:0:12})"
        sleep "$FIX_POLL_INTERVAL"
    done

    if [[ "$FIX_FOUND" != "true" ]]; then
        echo ""
        echo "[go] ════════════════════════════════════════════════════════"
        echo "[go]  No fix merged within ${FIX_POLL_TIMEOUT}s."
        echo "[go]  Copilot may still be working. Re-run 'bash go.sh' later"
        echo "[go]  or check GitHub Issues for status."
        echo "[go] ════════════════════════════════════════════════════════"
        exit 1
    fi

    # ── Cool-down before re-provisioning ──────────────────────────────────
    echo "[go] Cooling down ${REPROVISION_COOLDOWN}s before re-provisioning..."
    sleep "$REPROVISION_COOLDOWN"
done
