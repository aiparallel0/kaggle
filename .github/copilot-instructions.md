# Copilot Instructions — aiparallel0/kaggle

## Project description

This repository implements a systematic study of **multi-dataset fine-tuning for receipt Key Information Extraction (KIE)** using two architectures:

1. **DONUT** (`naver-clova-ix/donut-base`) — end-to-end vision-language model that autoregressively decodes 4 XML field tags from a receipt image.
2. **TrOCR + YOLOv8** — two-stage OCR pipeline: YOLOv8x detects text regions, TrOCR reads crops, heuristics assign fields.

The goal is to evaluate how adding auxiliary training datasets (WildReceipt, FUNSD, Invoices-DONUT) to SROIE fine-tuning affects performance on the SROIE Task-3 benchmark, across **8 dataset-combination experiments**. Best known F1: **0.8982** (Experiment 6: SROIE + Invoices-DONUT, 2× SROIE oversampling).

The 4 SROIE fields extracted: `company`, `date`, `address`, `total`.

---

## Key files

| File | Purpose |
|---|---|
| `run_all.py` | **Main entry point.** Full dual-architecture pipeline. Use `--micro` for fast ~10 min smoke test. |
| `train_trocr_yolo.py` | TrOCR + YOLOv8 training, evaluation, and field-assignment heuristics. |
| `run_experiments.py` | DONUT experiment configs (`ExperimentConfig`) and 8-experiment orchestrator. |
| `diagnostics.py` | Smoke tests, known failure pattern detection, AI-powered root cause analysis. |
| `constants.py` | **Single source of truth** for all shared constants. **Must be importable without torch.** |
| `cloud_orchestration.py` | Cloud pipeline orchestrator including Vast.ai GPU provisioner (`VastAIProvisioner`). |
| `train.py` | `DonutTrainer` wrapper around HuggingFace `Seq2SeqTrainer`. |
| `data_pipeline.py` | Dataset loaders and normalization (all extend `BaseDatasetLoader` ABC). |
| `resource_manager.py` | GPU/RAM resource management and training config validation. |
| `reporting.py` | Benchmarking, plots, and LaTeX paper generation. |

---

## Critical invariants (never violate these)

### 1. `constants.py` must import without torch

```python
# This must always work on CPU-only machines:
python -c "from constants import FIELDS, BASE_MODEL, SEED"
```

Never add torch/transformers imports to `constants.py`.

### 2. `tie_word_embeddings = False` after token embedding resize

```python
# After resize_token_embeddings(), always set:
model.config.tie_word_embeddings = False
```

Without this, safetensors drops `lm_head.weight` on save → F1 ≈ 0.42 on reload.

### 3. Always use list form for `convert_tokens_to_ids`

```python
# ✅ Correct
token_id = tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
# ❌ Wrong — returns ID of '<' character
token_id = tokenizer.convert_tokens_to_ids("<s_sroie>")
```

### 4. Never mutate the global `EXPERIMENTS` dict

```python
import dataclasses
config = dataclasses.replace(EXPERIMENTS[exp_id], batch_size=4)  # ✅
config = EXPERIMENTS[exp_id]; config.batch_size = 4              # ❌ mutates global
```

---

## Known failure patterns

| Symptom | Root cause | Fix |
|---|---|---|
| **F1 ≈ 0.42** | `lm_head.weight` dropped by safetensors deduplication | `LmHeadCloneCallback` + `tie_word_embeddings=False` |
| **F1 = 0.00** | Wrong `decoder_start_token_id` | Use list form: `tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]` |
| **F1 ≈ 0.008** | `token2json` returned a list (CORD `<sep/>` tokens) | Merge page-list in `_parse_prediction()` |
| **Loss = NaN** | fp16 overflow | Switch to bf16 or add gradient clipping |
| **Import error** | Broken import chain in `constants.py` | Fix the import; never add GPU-only imports to `constants.py` |
| **YOLO 0% detection** | `imgsz` mismatch between training and inference | Pass `imgsz=YOLO_IMG_SIZE` at every inference call site |
| **TrOCR all empty** | `TROCR_EPOCHS=1` in speed mode → `val_loss≈9.1` | Raise `TROCR_EPOCHS` floor to ≥5 in all speed modes |

---

## Validation commands

Always run these before pushing a fix:

```bash
# 1. Import chain (must work without torch)
python -c "from constants import FIELDS, BASE_MODEL, SEED"

# 2. Lint and format
ruff check . && ruff format --check .

# 3. Smoke test (requires GPU/model weights — use || true in CI)
python diagnostics.py --smoke-test

# 4. Fast end-to-end training test (~10 min on GPU)
python run_all.py --micro --verify
```

---

## Base model and training setup

- **Base model:** `naver-clova-ix/donut-base`
- **Task:** Seq2Seq fine-tuning (full parameters, not LoRA)
- **Target sequence format:** `<s_sroie><s_company>NAME</s_company><s_date>DATE</s_date><s_address>ADDR</s_address><s_total>TOTAL</s_total></s_sroie>`
- **Max decode length:** `MAX_LENGTH = 768` tokens (from `constants.py`)
- **Image size:** 960 × 1280 px (DONUT canonical — do NOT change)
- **Best experiment:** Exp 6 — SROIE + Invoices-DONUT with 2× SROIE oversampling → **F1 = 0.8982**

---

## Code style

- Python 3.10+ type hints everywhere (use `X | None` not `Optional[X]`)
- Double quotes for strings (enforced by ruff)
- Line length 100 (enforced by ruff)
- All constants imported from `constants.py` — never redeclare them
- Exception narrowing: use specific types (`json.JSONDecodeError`, `OSError`, etc.) not bare `Exception`

---

## How to reproduce a failure

```bash
# Fast test (no GPU required for import/lint checks):
python -c "from constants import FIELDS, BASE_MODEL, SEED"
ruff check .

# Full smoke test (GPU recommended):
python diagnostics.py --smoke-test --ai-diagnose --ai-provider auto

# Fast end-to-end test:
python run_all.py --micro --verify 2>&1 | tee training.log
```

---

## Autonomous GPU CI auto-fix loop

This repository uses a fully autonomous training → diagnose → fix → re-train loop.
**When you are assigned a `training-failure-fix` issue, follow these rules exactly.**

### How the loop works

```
push to main
  → gpu_training.yml runs on Vast.ai GPU runner
    IF SUCCESS: commit results, reset iteration counter ✅
    IF FAILURE:
      1. AI diagnoses via diagnostics.py
      2. Increments .github/auto_fix_state.json iteration counter
      3. Creates GitHub Issue assigned to @copilot with:
         - Full training log tail
         - Recently changed files (to avoid re-breaking them)
         - Iteration count (N of MAX)
      4. Copilot (you) picks up the issue and opens a PR
      5. PR must have label `training-failure-fix`
      6. auto_fix_loop.yml detects the PR, runs CI checks
      7. If CI passes: auto-merges (squash) → triggers gpu_training.yml again
      8. If CI fails: closes PR, creates new issue for next attempt
      9. After MAX_AUTO_FIX_ATTEMPTS (5): escalates to human
```

### Rules for Copilot when assigned a training-failure-fix issue

1. **Focus only on the specific failure** described in the issue. Read the log tail carefully.
2. **Do NOT modify** `.github/workflows/`, `vastai_runner.sh`, or `.github/auto_fix_state.json`.
3. **Always run these validations** before pushing:
   ```bash
   python -c "from constants import FIELDS, BASE_MODEL, SEED"
   ruff check . && ruff format --check .
   ```
4. **Check the iteration history** in the issue body — do not repeat a fix that was already tried.
5. **Add the label `training-failure-fix`** to your PR (the auto-merge loop requires it).
6. **Target `main`** — not any other branch.
7. **Keep changes surgical** — fix the root cause, do not refactor unrelated code.

### Auto-fix state file

`.github/auto_fix_state.json` tracks the loop state:

```json
{
  "iteration": 2,
  "max_iterations": 5,
  "last_failure_run_id": "12345678",
  "last_failure_sha": "abc1234",
  "history": [...]
}
```

- `iteration` is auto-incremented by `gpu_training.yml` on every failure.
- Reset to `0` automatically when training succeeds.
- **Never modify this file manually** unless resetting after human intervention.
