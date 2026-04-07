# Contributing to DONUT SROIE Multi-Dataset Pipeline

## Development Workflow

This repo is developed using a **Copilot agent → PR → merge to main** pattern.

1. Identify a bug or improvement
2. Open a GitHub Issue describing the problem
3. GitHub Copilot coding agent implements the fix on a feature branch
4. PR is reviewed and merged to `main`
5. The feature branch is deleted after merge

## Repository Settings Reminder

> **Action required for repo owner:** Enable **"Automatically delete head branches"** in
> **Settings → General → Pull Requests** to prevent stale branches from accumulating.

## Running Tests Locally

The test suite is CPU-only and requires only `pytest` (no GPU, no model downloads):

```bash
# Install test dependencies
pip install pytest ruff

# Run all tests
pytest tests/ -v --tb=short

# Run tests and stop on first failure
pytest tests/ -v --tb=short -x
```

## Running Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install          # install hooks once
pre-commit run --all-files  # run against all files
```

## Linting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

Or use the provided helper script:

```bash
bash lint.sh
```

## Makefile Shortcuts

```bash
make test       # pytest tests/ -v --tb=short
make test-fast  # pytest tests/ -v --tb=short -x  (stop on first failure)
make lint       # ruff check + format check
make check      # lint + test
```

## Current Baseline

The best known result is **Experiment 6** (SROIE + Invoices-DONUT, 2× SROIE oversampling):

| Metric | Value |
|---|---|
| Global F1 | **0.8982** |
| Company F1 | 0.905 |
| Date F1 | 0.984 |
| Address F1 | 0.790 |
| Total F1 | 0.912 |

Any PR that changes training code should validate against this baseline by running:

```bash
python run_all.py --experiment 6
```

## Key Invariants (never break these)

1. **`constants.py` must import without torch** — `python -c "from constants import FIELDS, BASE_MODEL, SEED"` must always work on a CPU-only machine
2. **`tie_word_embeddings = False`** must be set after `resize_token_embeddings()` to prevent F1 collapse
3. **Never mutate** the global `EXPERIMENTS` dict — always use `dataclasses.replace()`
4. **Always use list form** for `convert_tokens_to_ids(["<s_sroie>"])[0]`

See `CLAUDE.md` for the full list of invariants and known failure patterns.
