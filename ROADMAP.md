# ROADMAP — DONUT + TrOCR/YOLO KIE Pipeline

A development roadmap based on codebase analysis as of 2026-03.

---

## Completed ✅

- Multi-dataset DONUT fine-tuning pipeline (8 experiments with different dataset combinations)
- TrOCR + YOLOv8 two-stage pipeline for receipt KIE
- Unified SROIE Task-3 evaluation metrics (global F1, per-field F1, NED, exact-match)
- LaTeX paper auto-generation from results (`inject_results.py` + `paper.tex`)
- Constants centralization (`constants.py` — single source of truth, no duplication)
- Automated preflight validation (`preflight_checks.py` + `validators/`)
- Cloud pipeline orchestrator (`cloud_pipeline.py`) with mode routing
- `LmHeadCloneCallback` + safetensors weight-tying fix (prevents F1 ≈ 0.42 bug)
- `token2json` list-output handling (prevents F1 ≈ 0.008 bug)
- Val/test data split separation (prevents data leakage)
- `evaluate.py` shim removed — all callers use `donut_evaluator.py` directly
- `setup.py` removed — entry points migrated to `pyproject.toml`
- Per-file MIT license headers removed — single `pyproject.toml` license declaration

---

## In Progress 🔧

- **Cloud GPU training automation** — `mode_ml_training.py` uses subprocess stubs; Vast.ai integration not yet wired
- **Ollama-based code repair** — `mode_code_repair.py` has the structure but Ollama fix-application is a stub
- **Hyperparameter sweep runner** — `training_config.py` defines grids but there is no sweep orchestrator that actually loops over them

---

## Leaky / Fragile Areas ⚠️

- **`run_all.py` is a 67 KB monolith** — needs decomposition into `stages/` modules (data_prep, training, evaluation, paper_gen)
- **No integration tests** — `tests/` has unit tests for constants, loaders, metrics, and smoke tests, but no end-to-end single-epoch test
- **Hardcoded paths** — several files hardcode `/workspace/` (e.g. `SROIE_DATA_DIR`); should be env-var-driven throughout
- **No type checking / mypy** — project uses type hints but `mypy` is not part of CI
- **Large result JSON files tracked in git** — should use `.gitignore` (now added) or Git LFS
- **`run_all.py` writes `terminal.txt` log** — file now in `.gitignore`; the code itself should use `logging` instead of writing to a fixed filename

---

## Future Roadmap 🚀

1. **Decompose `run_all.py`** into `stages/` modules: `stages/data_prep.py`, `stages/training.py`, `stages/evaluation.py`, `stages/paper_gen.py`
2. **Complete Ollama integration** in `mode_code_repair.py` — implement actual fix-application using Ollama API
3. **Add end-to-end integration test** that runs a single mini-epoch experiment (1 train sample, 1 val sample, 1 epoch) to verify the full pipeline without real training time
4. **Implement hyperparameter sweep runner** using the grids defined in `training_config.py`
5. **Add Weights & Biases / MLflow tracking** — replace manual JSON result files with proper experiment tracking
6. **Support additional receipt datasets** — CORD v2, RVL-CDIP, SROIE 2019 full set
7. **Docker / devcontainer setup** for fully reproducible environments (`Dockerfile` + `.devcontainer/devcontainer.json`)
8. **CI pipeline improvements** — add `mypy` strict checking, `ruff format --check`, and test coverage reporting
9. **Model serving** — add a simple FastAPI endpoint (`serve.py`) for single-image inference
10. **Multi-GPU / distributed training** — add `accelerate launch` support in `train.py` for multi-GPU DONUT fine-tuning
