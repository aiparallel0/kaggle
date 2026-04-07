# =============================================================================
# Makefile — Single-command entry points for the DONUT SROIE pipeline
# =============================================================================
# Stop being the glue: every common operation is one command.
#
# Usage:
#   make              # full pipeline (all experiments + paper)
#   make setup        # install deps + validate import chain
#   make run          # full pipeline with auto-push
#   make exp-1        # single experiment (replace 1 with any ID)
#   make check        # run CI tests (lint + imports + smoke)
#   make fix          # run autonomous CI auto-fix loop (max 5 attempts)
#   make sync         # push current results to GitHub
#   make clean        # delete cached results (triggers re-run)
#   make help         # print this message
#
# Environment variables (set before make or export):
#   AI_DIAGNOSE=1                  enable AI-powered diagnostics
#   AI_DIAGNOSE_PROVIDER=auto      claude | mistral | auto
#   ANTHROPIC_API_KEY=sk-ant-...   Claude key (or put in anthropic_api_key.txt)
#   GITHUB_TOKEN=ghp_...           GitHub token for auto-push / PR comments
#   GITHUB_REPO=owner/repo         GitHub repo slug for PR comments
#   DONUT_WORKSPACE=/workspace     checkpoint and data path
# =============================================================================

PYTHON ?= python3

.PHONY: all run setup check fix sync clean help
.PHONY: test test-fast lint
.PHONY: $(addprefix exp-,1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17)

# Default: full pipeline
all: run

# ── Primary targets ────────────────────────────────────────────────────────

## run: full pipeline — train all experiments, generate paper, push results
run:
	$(PYTHON) run_all.py --auto-push

## setup: install deps, validate import chain, write processor_config.json
setup:
	pip install -r requirements.txt
	$(PYTHON) -c "from constants import FIELDS, BASE_MODEL, SEED; print('Import chain OK')"
	$(PYTHON) -c "\
import json, pathlib; \
p = pathlib.Path('processor_config.json'); \
p.write_text(json.dumps({'image_processor': {'size': {'height': 1280, 'width': 960}}}, indent=2) + '\n') if not p.exists() else None; \
print('processor_config.json ready')"

## check: run CI test suite (lint + imports + smoke test)
check: lint test

## test: run fast CPU-only pytest suite
test:
	pytest tests/ -v --tb=short

## test-fast: run pytest and stop on first failure
test-fast:
	pytest tests/ -v --tb=short -x

## lint: run ruff linter and format check
lint:
	ruff check . --select E,F,W --ignore E501
	ruff format --check .

## fix: run autonomous CI auto-fix loop (AI rewrites broken files, max 5 attempts)
fix:
	$(PYTHON) autonomous_ci.py --no-merge --max-fix-attempts 5

## sync: commit any pending result JSONs and push to origin
sync:
	@mkdir -p results; \
	BRANCH=$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main); \
	RESULT_FILES=$$(git status --short -- results/ 2>/dev/null | grep '\.json' | awk '{print $$2}' | tr '\n' ' '); \
	if [ -z "$$RESULT_FILES" ]; then echo "[sync] No new results to push"; exit 0; fi; \
	git add $$RESULT_FILES && \
	git commit -m "[auto] Sync results" && \
	git push -u origin $$BRANCH && \
	echo "[sync] Pushed to $$BRANCH"

## clean: delete cached result files (next run will re-train all experiments)
clean:
	@find results/ -name "experiment_*.json" -delete 2>/dev/null; \
	find results/ -name "all_experiments.json" -delete 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	echo "[clean] Done"

# ── Single-experiment shortcuts ────────────────────────────────────────────
# Usage: make exp-6  (runs only Experiment 6)
exp-%:
	$(PYTHON) run_all.py --experiment $* --auto-push

# ── Paper generation ───────────────────────────────────────────────────────
## paper: regenerate paper_filled.tex from existing results (no re-training)
paper:
	$(PYTHON) run_all.py --paper-only

## tensorboard: launch TensorBoard on port 6006 (run after training)
tensorboard:
	tensorboard --logdir results/tb_logs --host 0.0.0.0 --port 6006

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "DONUT SROIE Pipeline — available make targets"
	@echo "=============================================="
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
	@echo ""
	@echo "  exp-N     run single experiment N (e.g. make exp-6)"
	@echo "  paper     regenerate paper from cached results"
	@echo ""
	@echo "Set AI_DIAGNOSE=1 ANTHROPIC_API_KEY=... for AI-powered diagnostics."
	@echo ""
