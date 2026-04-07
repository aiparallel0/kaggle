#!/usr/bin/env bash
# =============================================================================
# vastai_runner.sh — Vast.ai GPU provisioner + ephemeral GitHub Actions runner
#
# One-liner (loads .env.cloud from repo root automatically):
#   bash vastai_runner.sh
#
# Or from anywhere:
#   bash <(curl -sSfL https://raw.githubusercontent.com/aiparallel0/kaggle/main/vastai_runner.sh)
#
# Required env vars (auto-loaded from .env.cloud if the file exists):
#   VASTAI_API_KEY   — Vast.ai API key  (https://vast.ai/console/account/)
#   GITHUB_TOKEN     — GitHub PAT with "repo" scope
#   GITHUB_REPO      — e.g. "aiparallel0/kaggle"
#
# Optional overrides:
#   VASTAI_GPU_NAME  default "RTX 4090"
#   VASTAI_MAX_PRICE default "0.80"
#   VASTAI_MIN_VRAM  default "24"
#   VASTAI_DISK_GB   default "40"
#   VASTAI_IMAGE     default "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
#   RUNNER_LABELS    default "self-hosted,gpu,vast-ai"
#   RUNNER_VERSION   default "2.316.1"
#   SSH_KEY          default "~/.ssh/id_vastai"
#   BOOT_TIMEOUT     default "600"
#   JOB_TIMEOUT      default "7200"
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Auto-load .env.cloud (gitignored secrets — never committed)
# ---------------------------------------------------------------------------
SCRIPT_DIR="")(cd "")(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.cloud"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[env] Loaded secrets from $ENV_FILE"
fi

# ---------------------------------------------------------------------------
# Configuration with defaults
# ---------------------------------------------------------------------------
VASTAI_GPU_NAME="${VASTAI_GPU_NAME:-RTX 4090}"
VASTAI_MAX_PRICE="${VASTAI_MAX_PRICE:-0.80}"
VASTAI_MIN_VRAM="${VASTAI_MIN_VRAM:-24}"
VASTAI_DISK_GB="${VASTAI_DISK_GB:-40}"
VASTAI_IMAGE="${VASTAI_IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,gpu,vast-ai}"
RUNNER_NAME="${RUNNER_NAME:-vastai-gpu-$(date +%s)}"
RUNNER_VERSION="${RUNNER_VERSION:-2.316.1}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_vastai}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-600}"
POLL_INTERVAL=15
JOB_TIMEOUT="${JOB_TIMEOUT:-7200}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }
require_env() {
    local var="$1"
    [[ -n "
${!var:-}" ]] || die "Required env var \\$var is not set. Add it to .env.cloud or export it."
}

# ---------------------------------------------------------------------------
# Step 1: Detect Python — portable, no hardcoded username/path
# ---------------------------------------------------------------------------
log "Step 1: Detecting Python..."
PYEXE=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null \
       && "$candidate" -c "import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)" 2>/dev/null; then
        PYEXE="$candidate"
        break
    fi
done
[[ -n "$PYEXE" ]] || die "No working Python 3.8+ found. Install from https://python.org"
log "  Python: $PYEXE (
$($PYEXE --version 2>&1))"

# ---------------------------------------------------------------------------
# Step 2: Install vastai CLI if needed
# ---------------------------------------------------------------------------
log "Step 2: Checking vastai CLI..."
if ! $PYEXE -m vast --help &>/dev/null 2>&1; then
    log "  Installing vastai..."
    $PYEXE -m pip install --quiet vastai
fi
VASTAI_CMD="$PYEXE -m vast"
log "  vastai CLI: $($VASTAI_CMD --version 2>/dev/null || echo 'unknown version')"

# ---------------------------------------------------------------------------
# Step 3: Validate required vars + authenticate
# ---------------------------------------------------------------------------
require_env VASTAI_API_KEY
require_env GITHUB_TOKEN
require_env GITHUB_REPO

log "Step 3: Authenticating with Vast.ai..."
$VASTAI_CMD set api-key "$VASTAI_API_KEY"
log "  Authenticated."

# ---------------------------------------------------------------------------
# Step 4: Find cheapest matching offer
# ---------------------------------------------------------------------------
log "Step 4: Searching offers (${VASTAI_GPU_NAME}, max \$${VASTAI_MAX_PRICE}/hr)..."
SEARCH_RESULT=$(
    $VASTAI_CMD search offers \
        "rentable=true num_gpus=1 gpu_name=${VASTAI_GPU_NAME// /_} gpu_ram>=${VASTAI_MIN_VRAM} dph<=${VASTAI_MAX_PRICE}" \
        --order "dph asc" --raw 2>/dev/null
) || die "vastai search offers failed."

OFFER_ID=$(echo "$SEARCH_RESULT" | $PYEXE -c "
import json, sys
data = json.loads(sys.stdin.read() or '[]')
if not data: sys.exit(1)
print(data[0]['id'])
") || die "No matching GPU offers found. Try raising VASTAI_MAX_PRICE."
log "  Offer: $OFFER_ID"

# ---------------------------------------------------------------------------
# Step 5: Create instance
# ---------------------------------------------------------------------------
log "Step 5: Creating instance..."
CREATE_RESULT=$(
    $VASTAI_CMD create instance "$OFFER_ID" \
        --image "$VASTAI_IMAGE" \
        --disk "$VASTAI_DISK_GB" \
        --label "$RUNNER_NAME" \
        --raw 2>/dev/null
) || die "vastai create instance failed."

INSTANCE_ID=$(echo "$CREATE_RESULT" | $PYEXE -c "
import json, sys
d = json.loads(sys.stdin.read() or '{}')
iid = d.get('new_contract')
if not iid: sys.exit(1)
print(iid)
") || die "Could not parse instance ID: $CREATE_RESULT"
log "  Instance: $INSTANCE_ID"

# Destroy instance on any exit
cleanup() {
    local code=$?
    log "Cleanup: destroying instance $INSTANCE_ID..."
    $VASTAI_CMD destroy instance "$INSTANCE_ID" --raw 2>/dev/null || true
    log "  Instance $INSTANCE_ID destroyed."
    exit $code
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Step 6: Wait for SSH
# ---------------------------------------------------------------------------
log "Step 6: Waiting for SSH (timeout ${BOOT_TIMEOUT}s)..."
SSH_HOST=""
SSH_PORT=""
elapsed=0
while [[ $elapsed -lt $BOOT_TIMEOUT ]]; do
    INFO=
($VASTAI_CMD show instance "$INSTANCE_ID" --raw 2>/dev/null || echo "{}").
sSTATUS=$(echo "$INFO" | $PYEXE -c "import json,sys; print(json.loads(sys.stdin.read()).get('actual_status','?'))")
    SSH_HOST=$(echo "$INFO" | $PYEXE -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('ssh_host','') or d.get('public_ipaddr',''))")
    SSH_PORT=$(echo "$INFO" | $PYEXE -c "import json,sys; print(json.loads(sys.stdin.read()).get('ssh_port',22))")

    if [[ "$STATUS" == "running" ]] && [[ -n "$SSH_HOST" ]]; then
        if ssh -i "$SSH_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
               -o BatchMode=yes -p "$SSH_PORT" "root@${SSH_HOST}" "echo ok" 2>/dev/null | grep -q ok; then
            log "  SSH READY: ${SSH_HOST}:${SSH_PORT}"
            break
        fi
    fi
    log "  $((elapsed/POLL_INTERVAL+1)): $STATUS ${SSH_HOST}:${SSH_PORT} — waiting ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
    SSH_HOST=""
done
[[ -n "$SSH_HOST" ]] || die "Instance never became SSH-accessible after ${BOOT_TIMEOUT}s."

SSH="ssh -i $SSH_KEY -o StrictHostKeyChecking=no -o BatchMode=yes -p ${SSH_PORT} root@${SSH_HOST}"
SCP="scp -i $SSH_KEY -o StrictHostKeyChecking=no -P ${SSH_PORT}"

# ---------------------------------------------------------------------------
# Step 7: Get GitHub Actions runner registration token
# ---------------------------------------------------------------------------
log "Step 7: Getting runner registration token..."
REG_TOKEN=$(curl -sSfX POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${GITHUB_REPO}/actions/runners/registration-token" \
    | $PYEXE -c "import json,sys; print(json.loads(sys.stdin.read())['token'])") \
    || die "Failed to get runner registration token. Check GITHUB_TOKEN permissions."
log "  Token: ${REG_TOKEN:0:8}..."

# ---------------------------------------------------------------------------
# Step 8: Write setup script to temp file, scp it, run it
#         (avoids ALL heredoc/herestring variable-expansion bugs on Windows Git Bash)
# ---------------------------------------------------------------------------
log "Step 8: Preparing remote setup script..."
RUNNER_PKG="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_PKG}"

SETUP_TMP=$(mktemp)
cat > "$SETUP_TMP" << SETUP_SCRIPT
#!/bin/bash
set -euo pipefail
log_r() { echo "[remote \\$(date -u +%H:%M:%S)] \\$*"; }

log_r "Creating non-root runner user..."
useradd -m -s /bin/bash runner 2>/dev/null || true
mkdir -p /workspace/actions-runner /workspace/runner-work /workspace/repo
chown -R runner:runner /workspace

log_r "Cloning https://github.com/${GITHUB_REPO}..."
git clone --depth 1 "https://github.com/${GITHUB_REPO}.git" /workspace/repo
chown -R runner:runner /workspace/repo

log_r "Downloading Actions runner v${RUNNER_VERSION}..."
curl -sSfL "${RUNNER_URL}" -o "/tmp/${RUNNER_PKG}"
tar xzf "/tmp/${RUNNER_PKG}" -C /workspace/actions-runner
chown -R runner:runner /workspace/actions-runner

log_r "Configuring runner (name=${RUNNER_NAME}, labels=${RUNNER_LABELS})..."
su - runner -c "
  cd /workspace/actions-runner && \
  ./config.sh \
    --url 'https://github.com/${GITHUB_REPO}' \
    --token '${REG_TOKEN}' \
    --name '${RUNNER_NAME}' \
    --labels '${RUNNER_LABELS}' \
    --ephemeral --unattended \
    --work /workspace/runner-work
"

log_r "Starting runner..."
su - runner -c "
  cd /workspace/actions-runner
  nohup ./run.sh > /workspace/runner.log 2>&1 &
  echo \\$! > /tmp/runner.pid
  disown
"

sleep 3
echo "Runner PID: \
$(cat /tmp/runner.pid 2>/dev/null || echo unknown)"
echo "--- runner.log tail ---"
tail -15 /workspace/runner.log 2>/dev/null || echo "(log not yet available)"
SETUP_SCRIPT

log "  Copying setup script to instance..."
$SCP "$SETUP_TMP" "root@${SSH_HOST}:/tmp/runner_setup.sh"
rm -f "$SETUP_TMP"

log "  Running setup script on instance..."
$SSH "bash /tmp/runner_setup.sh"
log "  Remote setup complete."

# ---------------------------------------------------------------------------
# Step 9: Wait for job to finish
# ---------------------------------------------------------------------------
log "Step 9: Waiting for job to complete (timeout ${JOB_TIMEOUT}s)..."
job_elapsed=0
while [[ $job_elapsed -lt $JOB_TIMEOUT ]]; do
    ALIVE=
($SSH "kill -0 \
$(cat /tmp/runner.pid 2>/dev/null) 2>/dev/null && echo alive || echo gone" 2>/dev/null || echo "gone")
    if [[ "$ALIVE" == "gone" ]]; then
        log "  Runner exited — job complete."
        break
    fi
    log "  Runner running (${job_elapsed}/${JOB_TIMEOUT}s)..."
sleep "$POLL_INTERVAL"
    job_elapsed=$((job_elapsed + POLL_INTERVAL))
done
[[ $job_elapsed -lt $JOB_TIMEOUT ]] || log "WARNING: Job timed out — destroying instance anyway."

log "Done. EXIT trap will destroy instance $INSTANCE_ID."