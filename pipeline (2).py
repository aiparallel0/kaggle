#!/usr/bin/env python3
"""
pipeline.py — AI-powered repository editing pipeline.
Reads all repository files, sends them with a prompt to a local Ollama model,
validates the response, applies file changes, and pushes to GitHub.

Supports three prompt modes:
  1. Interactive  — type a prompt at runtime (default)
  2. Single       — pass a prompt directly as a command-line argument
  3. Bulk         — provide a .txt file of newline-separated prompts; each is run sequentially
"""

import requests
import subprocess
import os
import datetime
import re
import sys
import logging
import time

# ─────────────────────────────────────────────
#  CONFIGURATION — Edit these values as needed
# ─────────────────────────────────────────────

REPO_PATH            = "/workspace/kaggle"        # Absolute path to the cloned repository
MODEL_NAME           = "codellama:34b"             # Ollama model to use
OLLAMA_URL           = "http://localhost:11434/api/generate"
TIMEOUT_SECONDS      = 1200                        # Max wait time per model response (20 min)
LOG_FILE             = "/var/log/pipeline.log"     # Log file path
DRY_RUN              = False                       # If True, shows changes without writing or pushing
COMMIT_PREFIX        = "AI fix"                    # Prefix used in git commit messages
MIN_RETENTION_RATIO  = 0.5                         # Reject files smaller than 50% of original size
BULK_DELAY_SECONDS   = 5                           # Pause between bulk prompts to avoid overloading the model

# File extensions to include when reading the repository
FILE_EXTENSIONS = (".py", ".yaml", ".yml", ".json", ".txt", ".md")

# Folders to skip when reading the repository
EXCLUDED_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules", ".venv", "venv"}

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  STEP 1 — Read all repository files
# ─────────────────────────────────────────────

def read_repo_files(repo_path: str) -> dict[str, str]:
    """Walk the entire repository and return a dict of {relative_path: file_content}."""
    files = {}
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for filename in filenames:
            if filename.endswith(FILE_EXTENSIONS):
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, repo_path)
                try:
                    with open(filepath, "r", errors="ignore") as f:
                        files[rel_path] = f.read()
                except Exception as e:
                    log.warning(f"Could not read {rel_path}: {e}")
    log.info(f"Read {len(files)} files from {repo_path}")
    return files


# ─────────────────────────────────────────────
#  STEP 2 — Build the prompt
# ─────────────────────────────────────────────

def build_prompt(files: dict[str, str], instruction: str) -> str:
    """Construct the full prompt by injecting all repository files as context."""
    file_context = ""
    for path, content in files.items():
        file_context += f"\n\n### FILE: {path}\n```\n{content}\n```"

    prompt = f"""You are an expert software engineer and code repair assistant.
Below are all the source files from the repository you must work on.
{file_context}

─────────────────────────────────────────────
INSTRUCTION:
{instruction.strip()}
─────────────────────────────────────────────

RESPONSE FORMAT — strictly follow this structure for every file you modify:

### FILE: relative/path/to/file.py
```python
<complete updated file content here — never use stubs, placeholders, or # ... comments>
```

Rules:
- Return the FULL, complete content of each changed file — never truncate with # ... or pass statements.
- Only include files that actually require changes.
- Do not add explanation text outside of inline code comments.
- If a file does not need changes, do not include it in your response.
"""
    log.info(f"Prompt built — {len(files)} files included, ~{len(prompt):,} characters total.")
    return prompt


# ─────────────────────────────────────────────
#  STEP 3 — Query the Ollama model
# ─────────────────────────────────────────────

def query_model(prompt: str) -> str:
    """Send the prompt to the Ollama API and return the raw response text."""
    log.info(f"Querying model '{MODEL_NAME}' — this may take several minutes...")
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 16384,
                "num_ctx": 65536,
            }
        }, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        result = response.json().get("response", "")
        log.info(f"Model response received — {len(result):,} characters.")
        return result
    except requests.exceptions.Timeout:
        log.error(f"Model query timed out after {TIMEOUT_SECONDS}s. Consider increasing TIMEOUT_SECONDS.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        log.error(f"Failed to reach Ollama API: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────
#  STEP 4 — Validate, parse response, apply changes
# ─────────────────────────────────────────────

def parse_and_apply(response: str, repo_path: str, original_files: dict[str, str]) -> list[str]:
    """
    Extract file changes from the model response, validate each against the
    original file size, and write approved changes to disk.
    """
    pattern = r"### FILE: (.+?)\n```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        log.warning("No file changes were parsed from the model response.")
        log.debug(f"Raw model response:\n{response}")
        return []

    changed  = []
    rejected = []

    for rel_path, content in matches:
        rel_path    = rel_path.strip()
        full_path   = os.path.join(repo_path, rel_path)
        new_content = content.strip()
        new_size    = len(new_content)

        # ── Size validation ──────────────────────────────────────────────
        if rel_path in original_files:
            original_size = len(original_files[rel_path])
            if original_size > 0 and new_size < original_size * MIN_RETENTION_RATIO:
                log.error(
                    f"REJECTED {rel_path}: new content is {new_size:,} chars, "
                    f"original was {original_size:,} chars "
                    f"({new_size / original_size:.0%} retention — below {MIN_RETENTION_RATIO:.0%} threshold). "
                    f"Model likely returned stubs or truncated output."
                )
                rejected.append(rel_path)
                continue

        # ── Stub detection ───────────────────────────────────────────────
        stub_indicators = ["# ...", "pass  #", "# TODO", "# placeholder", "# not implemented"]
        if any(indicator in new_content for indicator in stub_indicators):
            log.warning(
                f"WARNING {rel_path}: response contains stub indicators. "
                f"Applying anyway — review the commit manually."
            )

        if DRY_RUN:
            log.info(f"[DRY RUN] Would write: {rel_path} ({new_size:,} chars)")
            changed.append(rel_path)
            continue

        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(new_content + "\n")
            log.info(f"Applied: {rel_path} ({new_size:,} chars written)")
            changed.append(rel_path)
        except Exception as e:
            log.error(f"Failed to write {rel_path}: {e}")

    if rejected:
        log.warning(f"{len(rejected)} file(s) rejected due to size validation: {rejected}")

    return changed


# ─────────────────────────────────────────────
#  STEP 5 — Commit and push to GitHub
# ─────────────────────────────────────────────

def commit_and_push(changed_files: list[str], prompt_label: str = "") -> None:
    """Stage all changes, commit with a timestamped message, and push to origin."""
    timestamp      = datetime.datetime.now(datetime.UTC).isoformat()
    label          = f" [{prompt_label}]" if prompt_label else ""
    commit_message = f"{COMMIT_PREFIX}{label}: {timestamp} — modified {len(changed_files)} file(s)"

    try:
        subprocess.run(["git", "-C", REPO_PATH, "add", "."], check=True)

        status = subprocess.run(
            ["git", "-C", REPO_PATH, "status", "--porcelain"],
            capture_output=True, text=True
        )
        if not status.stdout.strip():
            log.info("No changes detected by git — nothing to commit.")
            return

        subprocess.run(["git", "-C", REPO_PATH, "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "-C", REPO_PATH, "push"], check=True)
        log.info(f"Pushed commit: {commit_message}")
        log.info(f"Files changed: {changed_files}")
    except subprocess.CalledProcessError as e:
        log.error(f"Git operation failed: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────
#  PROMPT MODE HELPERS
# ─────────────────────────────────────────────

def prompt_interactive() -> list[str]:
    """Ask the developer to type a prompt directly into the terminal."""
    print("\n" + "=" * 60)
    print("  MODE: Interactive")
    print("  Enter your prompt below.")
    print("  Press Enter twice when done.")
    print("=" * 60 + "\n")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    prompt = "\n".join(lines).strip()
    if not prompt:
        log.error("No prompt entered. Exiting.")
        sys.exit(1)
    return [prompt]


def prompt_single(text: str) -> list[str]:
    """Use a prompt passed directly as a command-line argument."""
    log.info("MODE: Single prompt (command-line argument)")
    return [text.strip()]


def prompt_bulk(filepath: str) -> list[str]:
    """
    Load prompts from a plain text file.
    Each prompt must be separated by a blank line.
    Lines beginning with # are treated as comments and ignored.

    Example prompts.txt:
    ─────────────────────────────────────────────
    Fix all CORD tag parsing issues causing low recall.

    Refactor the dataset loader to support lazy loading.

    Add docstrings to all public methods in train_donut.py.
    ─────────────────────────────────────────────
    """
    if not os.path.isfile(filepath):
        log.error(f"Bulk prompt file not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r") as f:
        raw = f.read()

    # Split on blank lines, strip comments and whitespace
    blocks = re.split(r"\n\s*\n", raw)
    prompts = []
    for block in blocks:
        lines = [l for l in block.splitlines() if not l.strip().startswith("#")]
        cleaned = "\n".join(lines).strip()
        if cleaned:
            prompts.append(cleaned)

    if not prompts:
        log.error(f"No valid prompts found in {filepath}.")
        sys.exit(1)

    log.info(f"MODE: Bulk — loaded {len(prompts)} prompt(s) from {filepath}")
    return prompts


# ─────────────────────────────────────────────
#  RUN A SINGLE PROMPT THROUGH THE PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(user_prompt: str, prompt_index: int = 1, total_prompts: int = 1) -> None:
    """Execute the full read → query → validate → apply → push cycle for one prompt."""
    log.info(f"── Prompt {prompt_index}/{total_prompts} ──────────────────────────────")
    log.info(f"Prompt preview: {user_prompt[:120].strip()}{'...' if len(user_prompt) > 120 else ''}")

    files    = read_repo_files(REPO_PATH)
    prompt   = build_prompt(files, user_prompt)
    response = query_model(prompt)

    # Use the first 40 chars of the prompt as a short commit label
    label   = re.sub(r"[^\w\s-]", "", user_prompt[:40]).strip().replace(" ", "_")
    changed = parse_and_apply(response, REPO_PATH, original_files=files)

    if changed and not DRY_RUN:
        commit_and_push(changed, prompt_label=label)
    elif not changed:
        log.warning(f"Prompt {prompt_index}/{total_prompts}: no files changed — review the model response.")
    else:
        log.info(f"Prompt {prompt_index}/{total_prompts}: dry run complete, no files written.")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Pipeline started")
    log.info(f"Repository : {REPO_PATH}")
    log.info(f"Model      : {MODEL_NAME}")
    log.info(f"Dry run    : {DRY_RUN}")
    log.info("=" * 60)

    if not os.path.isdir(REPO_PATH):
        log.error(f"Repository path does not exist: {REPO_PATH}")
        sys.exit(1)

    # ── Determine prompt mode from command-line arguments ────────────────
    #
    #   Interactive (default):
    #     python3 pipeline.py
    #
    #   Single prompt:
    #     python3 pipeline.py "Fix the CORD tag parsing issue."
    #
    #   Bulk from file:
    #     python3 pipeline.py --bulk prompts.txt
    #
    if len(sys.argv) == 1:
        prompts = prompt_interactive()

    elif sys.argv[1] == "--bulk":
        if len(sys.argv) < 3:
            log.error("Usage: python3 pipeline.py --bulk <path/to/prompts.txt>")
            sys.exit(1)
        prompts = prompt_bulk(sys.argv[2])

    else:
        prompts = prompt_single(" ".join(sys.argv[1:]))

    # ── Execute all prompts sequentially ────────────────────────────────
    total = len(prompts)
    for i, user_prompt in enumerate(prompts, start=1):
        run_pipeline(user_prompt, prompt_index=i, total_prompts=total)
        if i < total:
            log.info(f"Waiting {BULK_DELAY_SECONDS}s before next prompt...")
            time.sleep(BULK_DELAY_SECONDS)

    log.info("=" * 60)
    log.info(f"Pipeline finished — {total} prompt(s) processed.")
    log.info("=" * 60)
