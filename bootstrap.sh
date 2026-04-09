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

# 0. Activate virtual environment if present (Vast.ai instances use /venv/main)
for _venv_path in /venv/main /opt/conda /usr/local; do
    if [ -f "$_venv_path/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$_venv_path/bin/activate"
        echo "[bootstrap] Activated venv: $_venv_path"
        break
    fi
done

# 1. Auto-detect GITHUB_REPO from git remote (if not already set)
if [ -z "${GITHUB_REPO:-}" ]; then
    REMOTE_URL=$(git remote get-url origin 2>/dev/null || true)
    if [[ "$REMOTE_URL" =~ github\.com[:/]([^/]+/[^/.]+) ]]; then
        export GITHUB_REPO="${BASH_REMATCH[1]}"
        echo "[bootstrap] Auto-detected GITHUB_REPO=$GITHUB_REPO"
    fi
fi

# 2. Set workspace — default to $SCRIPT_DIR/workspace on Windows/MSYS/Cygwin
# to avoid MSYS2 path translation of /workspace → C:\Program Files\Git\workspace.
if [ -z "${DONUT_WORKSPACE:-}" ]; then
    _UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
    case "$_UNAME_S" in
        MINGW* | MSYS* | CYGWIN*)
            # Windows with Git Bash / MSYS2 / Cygwin — use a local writable path
            export DONUT_WORKSPACE="$SCRIPT_DIR/workspace"
            echo "[bootstrap] Windows/MSYS detected ($_UNAME_S) — using local workspace: $DONUT_WORKSPACE"
            ;;
        *)
            # Linux / macOS / Vast.ai — use /workspace (standard cloud GPU path)
            export DONUT_WORKSPACE="/workspace"
            ;;
    esac
fi
if ! mkdir -p "$DONUT_WORKSPACE" 2>/dev/null; then
    # Fallback: default path is not writable (e.g. /workspace on a system where
    # the user lacks root); switch to a repo-local directory instead.
    export DONUT_WORKSPACE="$SCRIPT_DIR/workspace"
    mkdir -p "$DONUT_WORKSPACE"
    echo "[bootstrap] WARNING: default workspace not writable; using $DONUT_WORKSPACE"
fi

# 3. Copy token files from /workspace if they exist (Vast.ai convention)
for f in hf_token.txt anthropic_api_key.txt mistral_api_key.txt github_token.txt; do
    if [ -f "/workspace/$f" ] && [ ! -f "$SCRIPT_DIR/$f" ]; then
        cp "/workspace/$f" "$SCRIPT_DIR/$f"
        echo "[bootstrap] Copied $f from /workspace"
    fi
done

# 4. Read tokens from files into env vars if not already set
if [ -z "${GITHUB_TOKEN:-}" ] && [ -f "$SCRIPT_DIR/github_token.txt" ]; then
    GITHUB_TOKEN="$(tr -d '\r\n' < "$SCRIPT_DIR/github_token.txt")"
    export GITHUB_TOKEN
    echo "[bootstrap] Loaded GITHUB_TOKEN from github_token.txt"
fi
if [ -z "${HF_TOKEN:-}" ] && [ -f "$SCRIPT_DIR/hf_token.txt" ]; then
    HF_TOKEN="$(tr -d '\r\n' < "$SCRIPT_DIR/hf_token.txt")"
    export HF_TOKEN
    echo "[bootstrap] Loaded HF_TOKEN from hf_token.txt"
fi

# 5. Enable AI diagnostics and auto-push
export AI_DIAGNOSE="${AI_DIAGNOSE:-1}"
export AI_DIAGNOSE_PROVIDER="${AI_DIAGNOSE_PROVIDER:-auto}"
export AUTOPUSH_RESULTS="${AUTOPUSH_RESULTS:-1}"

# 6. Install dependencies in three stages.
# flash-attn's setup.py imports torch at build time, so a single
# `pip install -r requirements.txt` may attempt to build flash-attn before
# torch is present and fail with "ModuleNotFoundError: No module named 'torch'".
echo "[bootstrap] Installing dependencies..."

# Stage 1: Install PyTorch first — required at build time by flash-attn.
# Extract the version constraints from requirements.txt to stay in sync with it.
_TORCH_SPEC="$(grep -m1 '^torch>=' requirements.txt || echo 'torch>=2.0.0')"
_TORCHVISION_SPEC="$(grep -m1 '^torchvision>=' requirements.txt || echo 'torchvision>=0.15.0')"
echo "[bootstrap]   Stage 1/3: Installing PyTorch (required before flash-attn)..."
pip install -q "$_TORCH_SPEC" "$_TORCHVISION_SPEC"

# Stage 2: Install all remaining deps except the flash-attn package line.
# We use '^flash-attn' (line-start anchor) so only the package spec line is
# excluded; comment lines in requirements.txt are ignored by pip anyway.
echo "[bootstrap]   Stage 2/3: Installing remaining dependencies (excluding flash-attn)..."
_REQS_NO_FLASH="$(mktemp /tmp/requirements-no-flash-XXXXXX.txt)"
grep -vE '^flash-attn' requirements.txt > "$_REQS_NO_FLASH"
pip install -q -r "$_REQS_NO_FLASH"
rm -f "$_REQS_NO_FLASH"

# Stage 3 (optional/best-effort): Install flash-attn with --no-build-isolation so
# it can find the already-installed torch.  flash-attn is an optimisation only;
# the pipeline falls back to PyTorch SDPA if it is absent.  A failure here must
# NOT abort the bootstrap (hence the if/else rather than a bare pip call under
# set -euo pipefail).  pip output is NOT suppressed so failures are diagnosable.
_FLASH_SPEC="$(grep -m1 '^flash-attn' requirements.txt || echo 'flash-attn>=2.0.0')"
echo "[bootstrap]   Stage 3/3: Installing flash-attn (optional, best-effort)..."
if pip install -q "$_FLASH_SPEC" --no-build-isolation; then
    echo "[bootstrap]   flash-attn installed successfully."
else
    echo "[bootstrap]   WARNING: flash-attn could not be installed (non-fatal). Pipeline will use PyTorch SDPA attention fallback."
fi

# 7. Validate import chain
echo "[bootstrap] Validating import chain..."
python -c "from constants import FIELDS, BASE_MODEL, SEED" || {
    echo "[bootstrap] FATAL: import chain broken"
    exit 2
}

# 8. Run full pipeline (forward all CLI args)
echo "[bootstrap] Starting pipeline..."
python run_all.py --auto-push "$@"
