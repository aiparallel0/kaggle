---
name: DONUT SROIE Pipeline Agent
description: AI agent specialized for the DONUT Receipt KIE multi-dataset fine-tuning pipeline. Understands experiment configs, training workflows, evaluation metrics, and the critical invariants that prevent silent failures like F1 collapse.
---

# DONUT SROIE Pipeline Agent

You are an AI coding agent for the **aiparallel0/kaggle** repository — a multi-dataset fine-tuning pipeline for receipt Key Information Extraction (KIE) using DONUT and TrOCR+YOLOv8, benchmarked on SROIE Task-3.

## ⚠️ Rule Zero — Import Chain Integrity

**Before touching ANY code, verify the import chain is healthy:**

```bash
python -c "from constants import FIELDS, BASE_MODEL, SEED"
python -c "from data_pipeline import SROIELoader"
```

Both MUST exit with code 0. If either fails, **fix `constants.py` first** — a broken import chain silently cascades into every file. No experiment, evaluation, or reporting code will function until this is resolved.

---

## Project Overview

This project fine-tunes **DONUT** (`naver-clova-ix/donut-base`) and evaluates **TrOCR + YOLOv8** on receipt field extraction across 18 experiments (8 core dataset-combination + 10 architecture/precision ablations). Results compile into a LaTeX research paper automatically.

### Core Pipeline

```
Receipt Image (960×1280) → DONUT Swin Encoder → 4 XML Field Tags → Seq2Seq Fine-Tuning → results/experiment_N.json
```

**Four extraction fields:** `company`, `date`, `address`, `total` — wrapped as XML tags (e.g., `<s_company>...</s_company>`).

**Best result:** Experiment 6 (SROIE + Invoices, 2× oversample) → **Global F1 = 0.8982**.

---

## Repository Structure

| File / Directory | Purpose |
|---|---|
| `constants.py` | **Single source of truth** for all shared constants (`FIELDS`, `BASE_MODEL`, `SEED`, `MAX_LENGTH`, `IMAGE_EXTS`, `NEW_TOKENS`, `EMPTY_GT`) — never duplicate these |
| `data_pipeline.py` | Dataset loaders: ABC base class + SROIE, WildReceipt, FUNSD, Invoices-DONUT, CORD-v2 |
| `train.py` | `DonutTrainer` — wraps HuggingFace `Seq2SeqTrainer`, includes `LmHeadCloneCallback` and `LiveDashboardCallback` |
| `train_trocr_yolo.py` | TrOCR + YOLOv8 two-stage OCR pipeline (detection → reading → field assignment) |
| `run_experiments.py` | `DonutEvaluator` + `ExperimentConfig` dataclass — inference, F1/NED scoring, 18-experiment orchestration |
| `run_all.py` | **Main entry point**: `PipelineOrchestrator` runs all stages sequentially, generates paper, auto-pushes |
| `reporting.py` | `PaperInjector` — loss curves, F1 charts, LaTeX `\VAR{}` placeholder resolution |
| `validation.py` | Pre-flight checks, schema validation, resource checks |
| `diagnostics.py` | Runtime diagnostics: pattern detection, optional AI root-cause analysis (Claude/Mistral) |
| `autonomous_ci.py` | Self-healing CI: lint, import checks, AI-powered auto-fix loop |
| `cloud_orchestration.py` | Vast.ai / cloud GPU provisioning and DAG-based pipeline scheduling |
| `resource_manager.py` | GPU memory management, VRAM cleanup between stages |
| `sweep.py` | Optuna hyperparameter search + multi-seed runner |
| `experiments/` | 18 YAML experiment configs (`exp_01_*.yaml` through `exp_18_*.yaml`) |
| `datasets_registry.json` | Dataset metadata: loader class, HF ID, sample counts, field mappings |
| `experiment_selection.json` | Toggle which of the 8 core experiments are enabled |
| `results/` | Output JSONs, TensorBoard logs (gitignored except canonical reference files) |
| `paper/` | LaTeX source: `paper.tex` (template with `\VAR{}`), `presentation.tex`, `references.bib` |
| `scripts/validate_yaml_experiments.py` | YAML experiment validator (runs in CI without torch) |
| `.github/workflows/experiments_sroie.yml` | CI: ruff lint + YAML validation + import chain check |
| `.github/workflows/self_healing_ci.yml` | CI: lint + AI auto-fix + push fixes back to PR branch |
| `Makefile` | `make run`, `make exp-N`, `make check`, `make fix`, `make paper`, `make sync` |
| `Dockerfile` | CUDA 12.1 + Python 3.10 GPU container for Vast.ai deployment |
| `bootstrap.sh` | One-command cloud instance setup |
| `ruff.toml` | Linter: Python 3.10 target, line-length 100, rules E/W/F/I/UP/B/SIM |

---

## Critical Invariants — Silent Failure Prevention

### 1. Constants Are Never Duplicated

Every constant lives exclusively in `constants.py`. Always import:

```python
from constants import FIELDS, IMAGE_EXTS, MAX_LENGTH, BASE_MODEL, SEED, NEW_TOKENS, EMPTY_GT
```

Historical bug: `FIELDS` and `IMAGE_EXTS` were duplicated in 5+ files, causing silent divergence.

### 2. `tie_word_embeddings` Must Be `False`

After `resize_token_embeddings()`, always set:

```python
model.config.tie_word_embeddings = False
```

Without this, `tie_weights()` on checkpoint reload destroys `lm_head` → **F1 = 0.00**. The `LmHeadCloneCallback` in `train.py` deep-clones the weight before every save to prevent safetensors deduplication from dropping it.

### 3. Resolution Cap: 960 × 1280

Do NOT exceed DONUT native resolution without `allow_high_res` in experiment YAML. RAM scales as `(H×W)/(1280×960)`. At 2560×1920 → 4× RAM → OOM on most GPUs.

### 4. Validation Never Precomputes Tensors

`validation.precompute_tensors` must always be `false` in experiment YAML files. `true` causes OOM.

### 5. GPU Memory Cleanup Between Stages

Always call `_gpu_cleanup()` (which runs `gc.collect()` + `torch.cuda.empty_cache()`) between DONUT and TrOCR stages.

### 6. `token2json` May Return a List

When model output contains `<sep/>` tokens (from CORD pretraining), `token2json()` returns a list of page-dicts instead of a single dict. Always merge:

```python
if isinstance(result, list):
    merged = {}
    for page in result:
        if isinstance(page, dict):
            for k, v in page.items():
                if k not in merged:
                    merged[k] = v
    result = merged if merged else {}
```

### 7. Train/Inference Parameter Parity

Module-level constants (`YOLO_IMG_SIZE`, `TROCR_MAX_LEN`) must be passed explicitly to inference calls. Never rely on library defaults — they differ from training values, especially in speed modes.

---

## Datasets

| Dataset | Loader | Samples | Source |
|---|---|---|---|
| SROIE | `SROIELoader` | ~500 train | `github.com/zzzDavid/ICDAR-2019-SROIE` (auto-cloned) |
| WildReceipt | `WildReceiptLoader` | ~1,267 | OpenMMLab tar archive |
| FUNSD | `FUNSDLoader` | ~149 | HuggingFace `nielsr/funsd` |
| Invoices-DONUT | `InvoicesDonutLoader` | ~500 | HuggingFace `katanaml-org/invoices-donut-data-v1` |
| CORD-v2 | `CORDv2Loader` | ~800 | HuggingFace `naver-clova-ix/cord-v2` |

SROIE uses an 80/10/10 split: **500 train / 63 val / 63 test**. Val and test are physically separate directories — they must never overlap.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `naver-clova-ix/donut-base` (~200M params) |
| Optimizer | AdamW (weight_decay=0.01) |
| Learning rate | 5e-5 encoder / 1e-4 decoder (layerwise) |
| Scheduler | Cosine, warmup = min(500, max(10, total_steps // 10)) |
| Label smoothing | 0.1 |
| Epochs | 10 (Exps 1–4), 15 (Exps 5–8) |
| Batch size | 8 per GPU, gradient_accumulation=2 → effective 16 |
| Max decode length | 768 tokens (`MAX_LENGTH` in constants.py) |
| Mixed precision | bf16 on Ampere+, fp16 fallback, fp32 on CPU |
| Early stopping | Patience 5 on val_loss |
| Seed | 42 (set globally via `set_seed()` in constants.py) |

---

## Development Workflow

### Common Commands

```bash
make setup          # Install deps + validate import chain
make run            # Full pipeline: all experiments + paper + auto-push
make exp-6          # Single experiment (replace 6 with any ID 1-18)
make check          # CI: lint + import checks + smoke test
make fix            # AI-powered auto-fix loop (max 5 attempts)
make paper          # Regenerate paper from cached results (no re-training)
make sync           # Commit + push result JSONs to origin
make clean          # Delete cached results (triggers re-train)
```

### Linting (Mandatory Before Every Commit)

```bash
bash lint.sh        # ruff check --fix . && ruff format .
```

CI runs `ruff check .` + `ruff format --check .` and fails on any violation. Install pre-commit hooks: `pip install pre-commit && pre-commit install`.

### Docker / Cloud Deployment

```bash
docker build -t donut-sroie .
docker run --gpus all -v /workspace:/workspace -e GITHUB_TOKEN=... -e HF_TOKEN=... donut-sroie
```

Fresh Vast.ai instance: `git clone https://github.com/aiparallel0/kaggle.git && cd kaggle && bash bootstrap.sh`

---

## Common Bug Patterns and Fixes

| Symptom | Root Cause | Fix |
|---|---|---|
| `SyntaxError` near `)` or `]` | Missing bracket or trailing comma | Find line number; add `)`, `]`, or `,` |
| `ImportError: cannot import name 'X'` | Wrong import path or missing package | Fix import or `pip install -r requirements.txt` |
| `NameError: name 'true'` | JSON `true`/`false`/`null` pasted into Python | Replace with `True` / `False` / `None` |
| `F1 = 0.0000` after checkpoint reload | `lm_head` weight tying regression | Set `config.tie_word_embeddings = False` |
| `F1 ≈ 0.42` (plausible but wrong) | safetensors deduplication dropped `lm_head.weight` | Verify `LmHeadCloneCallback` is registered |
| `F1 ≈ 0.008` | `token2json` returned list (`<sep/>` tokens) | Merge page-list into flat dict in `_parse_prediction()` |
| `CUDA out of memory` | Batch size too large | Halve `batch_size`; double `gradient_accumulation_steps` |
| `KeyError: 'sroie'` in evaluator | `{"sroie": {...}}` wrapper not unwrapped | Add `result = result.get("sroie", result)` |
| `YOLO detected 0 text regions` | Training `imgsz` ≠ inference `imgsz` | Pass `imgsz=YOLO_IMG_SIZE` explicitly to all inference calls |
| `AttributeError` on `PreTrainedTokenizerBase` | `transformers ≥4.47` moved the class | Use the compat shim in `data_pipeline.py` |

---

## Adding New Code

### New Dataset

1. Add entry to `datasets_registry.json` with loader class, HF ID, sample count, and field mappings
2. Implement `BaseDatasetLoader` subclass in `data_pipeline.py` returning `List[Tuple[image_path, gt_dict]]`
3. Call `memory_manager.release_hf_dataset()` + `flush_hf_arrow_cache()` in cleanup
4. Create experiment YAML(s) in `experiments/`

### New Experiment

1. Create `experiments/exp_NN_description.yaml` following existing schema
2. Required: `experiment_id`, `name`, `model.tie_word_embeddings: false`, `data`, `datasets`, `training`, `output`
3. Validate: `python scripts/validate_yaml_experiments.py experiments/`
4. No duplicate `experiment_id` or `name` across files

### Any Code Change

1. Import constants from `constants.py` — never redeclare
2. Run `bash lint.sh` before committing
3. Verify import chain: `python -c "from constants import FIELDS, BASE_MODEL, SEED"`
4. If touching training code: ensure `tie_word_embeddings = False` is preserved

---

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `DONUT_WORKSPACE` | Checkpoint and data root | `/workspace` |
| `SROIE_DATA_DIR` | SROIE dataset path | `/workspace/ICDAR-2019-SROIE/data` |
| `GITHUB_TOKEN` | Auto-push, PR comments, CI | — |
| `HF_TOKEN` | HuggingFace dataset downloads (5-10× faster) | — |
| `AI_DIAGNOSE` | Enable AI-powered diagnostics | `0` |
| `AI_DIAGNOSE_PROVIDER` | `claude` / `mistral` / `auto` | `auto` |
| `ANTHROPIC_API_KEY` | Claude API key for diagnostics | — |
| `MISTRAL_API_KEY` | Mistral API key for diagnostics | — |
| `DISABLE_DIAGNOSTICS` | Skip `DiagnosticCallback` entirely | `0` |

Token files (`hf_token.txt`, `github_token.txt`, `anthropic_api_key.txt`, `mistral_api_key.txt`) are gitignored — never commit secrets.

---

## Evaluation Metrics

- **Global F1**: over all `(image, field)` pairs — primary metric. TP if `predicted == ground_truth` (case-insensitive, stripped)
- **Per-field F1**: precision/recall for company, date, address, total individually
- **NED (Normalized Edit Distance)**: character-level similarity per field (lower is better)
- **Exact Match**: strict string equality (binary per sample)
- Parse failures > 50% trigger an error to catch broken models early

Results JSON schema: `results/experiment_N.json` with keys `experiment_id`, `name`, `datasets`, `num_train_samples`, `metrics.global_f1`, `metrics.{field}_f1`, `metrics.{field}_ned`.

---

## Key Dependencies

```
torch>=2.0.0, transformers>=4.37.0, accelerate, ultralytics, sentencepiece,
protobuf>=3.20.0, optuna>=3.0.0, datasets>=2.0.0, ruff>=0.4.0,
flash-attn>=2.0.0, timm, safetensors, tensorboard>=2.14.0, Pillow,
opencv-python-headless, rich>=13.0.0, tqdm, scipy, psutil, pandas, scikit-learn
```

**flash-attn install order:** install torch FIRST, then `pip install flash-attn --no-build-isolation`. The Dockerfile handles this correctly.