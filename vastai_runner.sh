#!/usr/bin/env bash
# =============================================================================
# vastai_runner.sh — Vast.ai GPU provisioner + ephemeral GitHub Actions runner
#
# What this script does:
#   1. Installs the `vastai` CLI (pip install vastai)
#   2. Authenticates with VASTAI_API_KEY
#   3. Searches for a cheap GPU instance (RTX 4090 or similar, ≥24 GB VRAM)
#   4. Creates the instance with a PyTorch+CUDA Docker image
#   5. Waits for the instance to be ready (SSH accessible)
#   6. SSHs in and:
#        a. Clones aiparallel0/kaggle
#        b. Installs Python dependencies (pip install -r requirements.txt)
#        c. Registers the machine as an ephemeral GitHub Actions self-hosted runner
#        d. Starts the runner (./run.sh) — auto-deregisters after one job
#   7. Polls until the runner exits (job complete)
#   8. Destroys the Vast.ai instance to stop billing
#
# Required environment variables:
#   VASTAI_API_KEY   — Vast.ai API key (https://vast.ai/console/account/)
#   GITHUB_TOKEN     — GitHub PAT with "repo" scope (for runner registration)
#   GITHUB_REPO      — Target repository, e.g. "aiparallel0/kaggle"
#
# Optional environment variables:
#   VASTAI_GPU_NAME  — GPU model filter, default "RTX 4090"
#   VASTAI_MAX_PRICE — Max $/hr, default "0.50"
#   VASTAI_MIN_VRAM  — Minimum VRAM in GB, default "24"
#   VASTAI_DISK_GB   — Root disk size in GB, default "40"
#   VASTAI_IMAGE     — Docker image, default "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
#   RUNNER_LABELS    — Comma-separated runner labels, default "self-hosted,gpu,vast-ai"
#   RUNNER_NAME      — Runner name, default "vastai-gpu-$(date +%s)"
#
# Usage:
#   export VASTAI_API_KEY="your-key"
#   export GITHUB_TOKEN="ghp_..."
#   export GITHUB_REPO="aiparallel0/kaggle"
#   bash vastai_runner.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration with defaults
# ---------------------------------------------------------------------------
VASTAI_GPU_NAME="${VASTAI_GPU_NAME:-RTX 4090}"
VASTAI_MAX_PRICE="${VASTAI_MAX_PRICE:-0.50}"
VASTAI_MIN_VRAM="${VASTAI_MIN_VRAM:-24}"
VASTAI_DISK_GB="${VASTAI_DISK_GB:-40}"
VASTAI_IMAGE="${VASTAI_IMAGE:-pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,gpu,vast-ai}"
RUNNER_NAME="${RUNNER_NAME:-vastai-gpu-$(date +%s)}"
RUNNER_VERSION="${RUNNER_VERSION:-2.316.1}"

# Maximum seconds to wait for instance SSH to become available
BOOT_TIMEOUT=600
# Polling interval in seconds while waiting for the instance/runner
POLL_INTERVAL=15

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

require_env() {
    local var="$1"
    [[ -n "${!var:-}" ]] || die "Required env var \$$var is not set."
}

# ---------------------------------------------------------------------------
# Step 0: Validate required environment variables
# ---------------------------------------------------------------------------
require_env VASTAI_API_KEY
require_env GITHUB_TOKEN
require_env GITHUB_REPO

# ---------------------------------------------------------------------------
# Step 1: Detect Python binary and install vastai CLI if not already present
# ---------------------------------------------------------------------------
log "Step 1: Installing vastai CLI..."

# Detect Python binary (Windows Git Bash has `python` not `python3`)
if command -v python3 &>/dev/null && python3 -c "import sys; sys.exit(0)" 2>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null && python -c "import sys; sys.exit(0)" 2>/dev/null; then
    PYTHON_CMD="python"
else
    die "No working Python found. Install Python 3.11+ from https://python.org"
fi

log "  Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"

# Install vastai if not present
if ! $PYTHON_CMD -m vast --help &>/dev/null 2>&1; then
    $PYTHON_CMD -m pip install --quiet vastai
fi

# Detect correct invocation.
# NOTE: `pip install vastai` installs the module as `vast` (not `vastai`).
if command -v vastai &>/dev/null; then
    VASTAI_CMD="vastai"
elif $PYTHON_CMD -m vast --help &>/dev/null 2>&1; then
    VASTAI_CMD="$PYTHON_CMD -m vast"
else
    die "vastai CLI not found after install. Run: pip install vastai"
fi

log "  vastai CLI ready (cmd: ${VASTAI_CMD}): $($VASTAI_CMD --version 2>/dev/null || echo 'unknown version')"

# ---------------------------------------------------------------------------
# Step 2: Authenticate with Vast.ai
# ---------------------------------------------------------------------------
log "Step 2: Authenticating with Vast.ai..."
$VASTAI_CMD set api-key "$VASTAI_API_KEY"
log "  Authentication configured."

# ---------------------------------------------------------------------------
# Step 3: Search for a suitable GPU instance
# ---------------------------------------------------------------------------
log "Step 3: Searching for GPU instances (GPU: '${VASTAI_GPU_NAME}', ≥${VASTAI_MIN_VRAM} GB VRAM, max \$${VASTAI_MAX_PRICE}/hr)..."

# Build a query that filters by GPU RAM and price, then sort by price ascending.
# The `search offers` command returns JSON when --raw is specified.
# Note: ${VASTAI_GPU_NAME// /_} converts spaces to underscores as required by the
# vastai query syntax (e.g. "RTX 4090" → "RTX_4090"). rentable=true must be lowercase.
SEARCH_RESULT=$(
    $VASTAI_CMD search offers \
        "rentable=true num_gpus=1 gpu_name=${VASTAI_GPU_NAME// /_} gpu_ram>=${VASTAI_MIN_VRAM} dph<=${VASTAI_MAX_PRICE}" \
        --order "dph asc" \
        --raw 2>/dev/null
) || die "vastai search offers failed. Check your API key and network connectivity."

# Extract the cheapest offer's ID using Python (already available on the host)
OFFER_ID=$($PYTHON_CMD - <<'PYEOF'
import json, sys
data = json.loads(sys.stdin.read() or "[]")
if not data:
    sys.exit(1)
# data is a list of offer dicts; pick the cheapest (already sorted by dph asc)
print(data[0]["id"])
PYEOF
<<< "$SEARCH_RESULT") || die "No matching GPU offers found. Try relaxing VASTAI_GPU_NAME or VASTAI_MAX_PRICE."

log "  Found offer ID: $OFFER_ID"

# ---------------------------------------------------------------------------
# Step 4: Create the instance
# ---------------------------------------------------------------------------
log "Step 4: Creating instance from offer $OFFER_ID..."

CREATE_RESULT=$(
    $VASTAI_CMD create instance "$OFFER_ID" \
        --image "$VASTAI_IMAGE" \
        --disk "$VASTAI_DISK_GB" \
        --label "$RUNNER_NAME" \
        --raw 2>/dev/null
) || die "vastai create instance failed."

INSTANCE_ID=$($PYTHON_CMD - <<'PYEOF'
import json, sys
data = json.loads(sys.stdin.read() or "{}")
iid = data.get("new_contract")
if not iid:
    sys.exit(1)
print(iid)
PYEOF
<<< "$CREATE_RESULT") || die "Could not parse instance ID from create response: $CREATE_RESULT"

log "  Instance created: ID=$INSTANCE_ID"

# Trap to ensure the instance is destroyed on script exit (success or failure)
cleanup() {
    local exit_code=$?
    log "Cleanup: destroying instance $INSTANCE_ID..."
    $VASTAI_CMD destroy instance "$INSTANCE_ID" --raw 2>/dev/null || true
    log "  Instance $INSTANCE_ID destroyed."
    exit $exit_code
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Step 5: Wait for the instance to be ready (running + SSH accessible)
# ---------------------------------------------------------------------------
log "Step 5: Waiting for instance $INSTANCE_ID to boot (timeout ${BOOT_TIMEOUT}s)..."

elapsed=0
SSH_HOST=""
SSH_PORT=""

while [[ $elapsed -lt $BOOT_TIMEOUT ]]; do
    INSTANCE_JSON=$($VASTAI_CMD show instance "$INSTANCE_ID" --raw 2>/dev/null || echo "{}")
    STATUS=$($PYTHON_CMD - <<'PYEOF'
import json, sys
data = json.loads(sys.stdin.read() or "{}")
print(data.get("actual_status", "unknown"))
PYEOF
<<< "$INSTANCE_JSON")

    if [[ "$STATUS" == "running" ]]; then
        # Extract SSH host and port
        SSH_HOST=$($PYTHON_CMD - <<'PYEOF'
import json, sys
data = json.loads(sys.stdin.read() or "{}")
print(data.get("ssh_host", "") or data.get("public_ipaddr", ""))
PYEOF
<<< "$INSTANCE_JSON")
        SSH_PORT=$($PYTHON_CMD - <<'PYEOF'
import json, sys
data = json.loads(sys.stdin.read() or "{}")
print(data.get("ssh_port", 22))
PYEOF
<<< "$INSTANCE_JSON")

        # Probe SSH connectivity
        if ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes \
               -p "$SSH_PORT" "root@${SSH_HOST}" "echo ok" &>/dev/null; then
            log "  Instance is running and SSH is accessible at ${SSH_HOST}:${SSH_PORT}"
            break
        fi
    fi

    log "  Status: $STATUS — waiting ${POLL_INTERVAL}s (${elapsed}/${BOOT_TIMEOUT}s elapsed)..."
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
done

[[ -n "$SSH_HOST" ]] || die "Instance never became reachable via SSH after ${BOOT_TIMEOUT}s."

SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -p ${SSH_PORT}"

# ---------------------------------------------------------------------------
# Step 6: Set up the GitHub Actions runner on the instance via SSH
# ---------------------------------------------------------------------------
log "Step 6: Fetching GitHub Actions runner registration token..."

# Get a fresh registration token from GitHub API
REG_TOKEN=$($PYTHON_CMD - <<PYEOF
import json, sys, urllib.request
token = "${GITHUB_TOKEN}"
repo  = "${GITHUB_REPO}"
url   = f"https://api.github.com/repos/{repo}/actions/runners/registration-token"
req = urllib.request.Request(url, data=b"", headers={
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {token}",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "vastai-runner/1.0",
}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    print(data["token"])
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
) || die "Could not obtain GitHub Actions runner registration token. Check GITHUB_TOKEN and GITHUB_REPO."

log "  Registration token obtained."

# Determine the Actions runner download URL for Linux x64
RUNNER_PKG="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${RUNNER_PKG}"

# Escape variables for safe interpolation in the heredoc sent over SSH
_REPO=$(printf '%q' "$GITHUB_REPO")
_REG_TOKEN=$(printf '%q' "$REG_TOKEN")
_RUNNER_NAME=$(printf '%q' "$RUNNER_NAME")
_RUNNER_LABELS=$(printf '%q' "$RUNNER_LABELS")

log "Step 6b: Running remote setup on ${SSH_HOST}:${SSH_PORT}..."

# shellcheck disable=SC2087  # intentional: variables expand on the CLIENT side
ssh $SSH_OPTS "root@${SSH_HOST}" bash -s -- \
    "$_REPO" "$_REG_TOKEN" "$_RUNNER_NAME" "$_RUNNER_LABELS" \
    "$RUNNER_URL" "$RUNNER_PKG" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

REPO="$1"
REG_TOKEN="$2"
RUNNER_NAME="$3"
RUNNER_LABELS="$4"
RUNNER_URL="$5"
RUNNER_PKG="$6"

log_r() { echo "[remote] $*"; }

# -- Clone the repository --
log_r "Cloning https://github.com/${REPO} ..."
git clone --depth 1 "https://github.com/${REPO}.git" /workspace/repo
cd /workspace/repo

# -- Install Python dependencies --
log_r "Installing Python dependencies..."
pip install --quiet -r requirements.txt || true
# Always ensure core linting/validation tools are present (must succeed)
pip install --quiet ruff pyyaml
# Verify critical packages installed (torch/transformers may need GPU extras)
for pkg in torch transformers; do
    python3 -c "import $pkg" 2>/dev/null \
        || log_r "WARNING: $pkg not importable — GPU extras may be missing"
done

# -- Validate import chain (MUST succeed — constants.py must work without torch) --
log_r "Validating import chain..."
python3 -c "from constants import FIELDS, BASE_MODEL, SEED; print('  Import chain OK')"

# -- Download and configure the GitHub Actions runner --
log_r "Downloading Actions runner ${RUNNER_URL} ..."
cd /opt
mkdir -p actions-runner && cd actions-runner
curl -sSfL "$RUNNER_URL" -o "$RUNNER_PKG"
tar xzf "$RUNNER_PKG"

log_r "Configuring runner (name: ${RUNNER_NAME}, labels: ${RUNNER_LABELS}) ..."
./config.sh \
    --url "https://github.com/${REPO}" \
    --token "$REG_TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --ephemeral \
    --unattended \
    --work /workspace/runner-work

# -- Start the runner (ephemeral: auto-deregisters after one job) --
log_r "Starting runner..."
./run.sh &
RUNNER_PID=$!
log_r "Runner PID: ${RUNNER_PID}"

# Write PID to file so the host script can wait for completion
echo "$RUNNER_PID" > /tmp/runner.pid
REMOTE_SCRIPT

log "  Remote setup complete."

# ---------------------------------------------------------------------------
# Step 7: Wait for the runner job to finish
# ---------------------------------------------------------------------------
log "Step 7: Waiting for the runner to complete its job..."

# Poll the remote process until the runner PID disappears
# The runner exits automatically after one ephemeral job
JOB_TIMEOUT=7200  # 2 hours maximum
job_elapsed=0

while [[ $job_elapsed -lt $JOB_TIMEOUT ]]; do
    RUNNER_ALIVE=$(ssh $SSH_OPTS "root@${SSH_HOST}" \
        "kill -0 \$(cat /tmp/runner.pid 2>/dev/null) 2>/dev/null && echo alive || echo gone" \
        2>/dev/null || echo "gone")

    if [[ "$RUNNER_ALIVE" == "gone" ]]; then
        log "  Runner process has exited — job complete."
        break
    fi

    log "  Runner still running (${job_elapsed}/${JOB_TIMEOUT}s elapsed)..."
    sleep "$POLL_INTERVAL"
    job_elapsed=$((job_elapsed + POLL_INTERVAL))
done

if [[ $job_elapsed -ge $JOB_TIMEOUT ]]; then
    log "WARNING: Runner did not finish within ${JOB_TIMEOUT}s — destroying instance anyway."
fi

# ---------------------------------------------------------------------------
# Step 8: Destroy the instance (handled by the EXIT trap above)
# ---------------------------------------------------------------------------
log "Step 8: Script complete — EXIT trap will destroy instance $INSTANCE_ID."
