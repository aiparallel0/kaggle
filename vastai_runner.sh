#!/usr/bin/env bash
# =============================================================================
# vastai_runner.sh — Vast.ai GPU provisioner + ephemeral GitHub Actions runner
#
# Usage (one-liner):
#   bash <(curl -sSfL https://raw.githubusercontent.com/aiparallel0/kaggle/main/vastai_runner.sh)
#
# Or from repo root (loads .env.cloud automatically):
#   bash vastai_runner.sh
#
# Required env vars (loaded from .env.cloud if present):
#   VASTAI_API_KEY   — Vast.ai API key
#   GITHUB_TOKEN     — GitHub PAT with "repo" scope
#   GITHUB_REPO      — e.g. "aiparallel0/kaggle"
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Load .env.cloud if present (gitignored secrets file)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "")(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.cloud"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
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
BOOT_TIMEOUT="${BOOT_TIMEOUT:-600}"
POLL_INTERVAL="${POLL_INTERVAL:-15}"
JOB_TIMEOUT="${JOB_TIMEOUT:-7200}"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }
require_env() { local v="$1"; [[ -n "
${!v:-}" ]] || die "Required env var \\$v is not set."; }

# ---------------------------------------------------------------------------
# Step 1: Detect Python (portable — works on Windows Git Bash without username)
# ---------------------------------------------------------------------------
log "Step 1: Detecting Python..."
if command -v python3 &>/dev/null && python3 -c "import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)" 2>/dev/null; then
    PYEXE="python3"
elif command -v python &>/dev/null && python -c "import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)" 2>/dev/null; then
    PYEXE="python"
else
    die "No working Python 3.8+ found. Install from https://python.org"
fi
log "  Python: $PYEXE ($($PYEXE --version 2>&1))"

# ---------------------------------------------------------------------------
# Step 2: Install vastai CLI if needed
# ---------------------------------------------------------------------------
log "Step 2: Checking vastai CLI..."
if ! $PYEXE -m vast --help &>/dev/null 2>&1; then
    log "  Installing vastai..."
    $PYEXE -m pip install --quiet vastai
fi
VASTAI_CMD="$PYEXE -m vast"
log "  vastai CLI ready: $($VASTAI_CMD --version 2>/dev/null || echo 'unknown')"

# ---------------------------------------------------------------------------
# Step 3: Validate required env vars and authenticate
# ---------------------------------------------------------------------------
require_env VASTAI_API_KEY
require_env GITHUB_TOKEN
require_env GITHUB_REPO

log "Step 3: Authenticating with Vast.ai..."
$VASTAI_CMD set api-key "$VASTAI_API_KEY"
log "  Authenticated."

# ---------------------------------------------------------------------------
# Step 4: Search for cheapest matching GPU offer
# ---------------------------------------------------------------------------
log "Step 4: Searching offers (GPU: ${VASTAI_GPU_NAME}, max \\$${VASTAI_MAX_PRICE}/hr)..."
SEARCH_RESULT=$(
    $VASTAI_CMD search offers \
        "rentable=true num_gpus=1 gpu_name=${VASTAI_GPU_NAME// /_} gpu_ram>=${VASTAI_MIN_VRAM} dph<=${VASTAI_MAX_PRICE}" \
        --order "dph asc" --raw 2>/dev/null
) || die "vastai search offers failed."

OFFER_ID=$(echo "$SEARCH_RESULT" | $PYEXE -c "
import json,sys
data=json.loads(sys.stdin.read() or '[]')
if not data: sys.exit(1)
print(data[0]['id'])
") || die "No matching GPU offers found. Try raising VASTAI_MAX_PRICE."
log "  Offer: $OFFER_ID"

# ---------------------------------------------------------------------------
# Step 5: Create instance
# ---------------------------------------------------------------------------
log "Step 5: Creating instance from offer $OFFER_ID..."
CREATE_RESULT=$(
    $VASTAI_CMD create instance "$OFFER_ID" \
        --image "$VASTAI_IMAGE" \
        --disk "$VASTAI_DISK_GB" \
        --label "$RUNNER_NAME" \
        --raw 2>/dev/null
) || die "vastai create instance failed."

INSTANCE_ID=$(echo "$CREATE_RESULT" | $PYEXE -c "
import json,sys
d=json.loads(sys.stdin.read() or '{}')
id=d.get('new_contract')
if not iid: sys.exit(1)
print(iid)
") || die "Could not parse instance ID: $CREATE_RESULT"
log "  Instance: $INSTANCE_ID"

# Destroy instance on exit (success or failure)
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
log "Step 6: Waiting for instance to boot (timeout ${BOOT_TIMEOUT}s)..."
SSH_HOST=""; SSH_PORT=""
for i in $(seq 1 $((BOOT_TIMEOUT / POLL_INTERVAL))); do
    INFO=$($VASTAI_CMD show instance "$INSTANCE_ID" --raw 2>/dev/null || echo "{}")[027] 2>/dev/null
    STATUS=$(echo "$INFO" | $PYEXE -c "import json,sys; print(json.loads(sys.stdin.read()).get('actual_status','?'))")
    SSH_HOST=$(echo "$INFO" | $PYEXE -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('ssh_host','') or d.get('public_ipaddr',''))")
    SSH_PORT=$(echo "$INFO" | $PYEXE -c "import json,sys; print(json.loads(sys.stdin.read()).get('ssh_port',22))")
    if [[ "$STATUS" == "running" ]] && \
       ssh -i ~/.ssh/id_vastai -o ConnectTimeout=5 -o StrictHostKeyChecking=no \
           -o BatchMode=yes -p "$SSH_PORT" "root@${SSH_HOST}" "echo ok" 2>/dev/null | grep -q ok; then
        log "  SSH READY: ${SSH_HOST}:${SSH_PORT}"
        break
    fi
    log "  $i: $STATUS ${SSH_HOST}:${SSH_PORT} — waiting ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
    SSH_HOST=""
done
[[ -n "$SSH_HOST" ]] || die "Instance never became SSH-accessible after ${BOOT_TIMEOUT}s."

SSH="ssh -i ~/.ssh/id_vastai -o StrictHostKeyChecking=no -o BatchMode=yes -p ${SSH_PORT} root@${SSH_HOST}"
SCP="scp -i ~/.ssh/id_vastai -o StrictHostKeyChecking=no -P ${SSH_PORT}"

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
    || die "Failed to get runner registration token. Check GITHUB_TOKEN and GITHUB_REPO."
log "  Token: ${REG_TOKEN:0:8}..."

# ---------------------------------------------------------------------------
# Step 8: Write setup script locally, scp it, run it
# ---------------------------------------------------------------------------
log "Step 8: Setting up runner on remote instance..."

RUNNER_PKG="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_PKG}"

# Write setup script to a temp file (avoids ALL heredoc/herestring issues on Windows Git Bash)
SETUP_SCRIPT=$(mktemp /tmp/runner_setup_XXXX.sh)
cat > "$SETUP_SCRIPT" << SETUPEOF
#!/bin/bash
set -euo pipefail
log_r() { echo "[remote \\$(date -u +%H:%M:%S)] \\$*"; }

# Create non-root user for runner (GitHub Actions runner refuses to run as root)
useradd -m -s /bin/bash runner 2>/dev/null || true
mkdir -p /workspace/actions-runner /workspace/runner-work /workspace/repo
chown -R runner:runner /workspace

# Clone repo
log_r "Cloning https://github.com/${GITHUB_REPO}..."
git clone --depth 1 "https://github.com/${GITHUB_REPO}.git" /workspace/repo
chown -R runner:runner /workspace/repo

# Download runner tarball as root (faster), extract into runner-owned dir
log_r "Downloading Actions runner v${RUNNER_VERSION}..."
curl -sSfL "${RUNNER_URL}" -o /tmp/${RUNNER_PKG}
tar xzf /tmp/${RUNNER_PKG} -C /workspace/actions-runner
chown -R runner:runner /workspace/actions-runner

# Configure and start runner as non-root user
log_r "Configuring runner..."
su - runner -c "
  cd /workspace/actions-runner
  ./config.sh \
    --url 'https://github.com/${GITHUB_REPO}' \
    --token '${REG_TOKEN}' \
    --name '${RUNNER_NAME}' \
    --labels '${RUNNER_LABELS}' \
    --ephemeral --unattended \
    --work /workspace/runner-work
"

log_r "Starting runner (nohup, log: /workspace/runner.log)..."
su - runner -c "
  cd /workspace/actions-runner
  nohup ./run.sh > /workspace/runner.log 2>&1 &
  echo \\$! > /tmp/runner.pid
  disown
"

sleep 3
echo "Runner PID: \\$(cat /tmp/runner.pid 2>/dev/null || echo unknown)"
tail -15 /workspace/runner.log 2>/dev/null || echo "(log not yet available)"
SETUPEOF

# Copy and execute
$SCP "$SETUP_SCRIPT" "root@${SSH_HOST}:/tmp/runner_setup.sh"
$SSH "bash /tmp/runner_setup.sh"
rm -f "$SETUP_SCRIPT"
log "  Remote setup complete."

# ---------------------------------------------------------------------------
# Step 9: Wait for runner job to finish
# ---------------------------------------------------------------------------
log "Step 9: Waiting for job to complete (timeout ${JOB_TIMEOUT}s)..."
job_elapsed=0
while [[ $job_elapsed -lt $JOB_TIMEOUT ]]; do
    ALIVE=$($SSH "kill -0 \\(cat /tmp/runner.pid 2>/dev/null) 2>/dev/null && echo alive || echo gone" 2>/dev/null || echo "gone")
    if [[ "$ALIVE" == "gone" ]]; then
        log "  Runner exited — job complete."
        break
    fi
    log "  Runner running (${job_elapsed}/${JOB_TIMEOUT}s)..."
sleep "$POLL_INTERVAL"
    job_elapsed=$((job_elapsed + POLL_INTERVAL))
done
[[ $job_elapsed -lt $JOB_TIMEOUT ]] || log "WARNING: Job timeout reached — destroying instance anyway."

log "Done. EXIT trap will destroy instance $INSTANCE_ID."
SETUPEOF
