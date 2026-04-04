---
# my-agent.agent.md — GitHub Copilot Agent Instructions
# DONUT SROIE Multi-Dataset Receipt KIE Pipeline
---

## Identity

You are an AI coding agent working on the **aiparallel0/kaggle** repository — a multi-dataset fine-tuning pipeline for receipt Key Information Extraction (KIE) using DONUT and TrOCR+YOLOv8 models, benchmarked on SROIE Task-3.

---

## ⚠️ Rule Zero — Import Chain Integrity

**Before touching ANY code, verify the import chain is healthy:**

```bash
python -c "from constants import FIELDS, BASE_MODEL, SEED"
python -c "from data_pipeline import SROIELoader"
```

Both MUST exit with code 0. If either fails, **fix `constants.py` first** — a broken import chain silently cascades into every file in the project. No experiment, evaluation, or reporting code will function.

---

## Project Architecture

### What This Project Does

Fine-tunes **DONUT** (`naver-clova-ix/donut-base`) and evaluates **TrOCR + YOLOv8** on receipt field extraction across 8 dataset-combination experiments. Results compile into a LaTeX research paper automatically.

### Core Pipeline (5 Stages)

```
Receipt Image (960×1280) → DONUT Swin Encoder → 4 XML Field Tags → Seq2Seq Fine-Tuning → results/experiment_N.json
```

**Four extraction fields:** `company`, `date`, `address`, `total` — wrapped as XML tags (e.g., `<s_company>...</s_company>`).

### Repository Structure

```
├── constants.py              # Single source of truth for ALL constants — NEVER duplicate
├── data_pipeline.py          # Dataset loaders (ABC base + SROIE, WildReceipt, FUNSD, Invoices, CORD)
├── train.py                  # DonutTrainer — Seq2Seq fine-tuning with HuggingFace Trainer
├── train_trocr_yolo.py       # TrOCR + YOLOv8 two-stage OCR pipeline
├── run_experiments.py         # DonutEvaluator — inference + F1/NED scoring on SROIE test set
├── run_all.py                # Orchestrator: runs all experiments, generates paper, auto-pushes
├── reporting.py              # Loss curves, F1 bar charts, LaTeX table injection
├── validation.py             # Pre-flight checks, schema validation, resource checks
├── diagnostics.py            # Runtime diagnostics with optional AI-powered auto-fix
├── autonomous_ci.py          # Self-healing CI: lint, import checks, AI-powered fixes
├── cloud_orchestration.py    # Vast.ai / cloud GPU provisioning & remote execution
├── resource_manager.py       # GPU memory management, VRAM cleanup between stages
├── sweep.py                  # Optuna hyperparameter sweep
├── experiments/              # YAML experiment configs (exp_01 through exp_18)
├── datasets_registry.json    # Dataset metadata (loader class, HF IDs, sample counts)
├── experiment_selection.json # Which of the 8 core experiments are enabled
├── results/                  # Output JSONs + TensorBoard logs
├── paper/                    # LaTeX paper template, presentation template, references.bib
├── scripts/                  # Utility scripts (YAML validator)
├── .github/workflows/        # CI: experiments_sroie.yml (lint+YAML validation), self_healing_ci.yml
├── Makefile                  # make run, make exp-N, make check, make fix, make paper
├── Dockerfile                # CUDA 12.1 + Python 3.10 GPU container for Vast.ai
├── bootstrap.sh              # One-command cloud instance setup
└── ruff.toml                 # Linter config: Python 3.10+, line-length=100
```

---

## Critical Invariants — Things That Must Never Break

### 1. `constants.py` Is the Single Source of Truth

Every constant lives here: `FIELDS`, `BASE_MODEL`, `SEED`, `MAX_LENGTH`, `IMAGE_EXTS`, `SROIE_FIELD_TAGS`, etc. **Never duplicate a constant** in another file — always import from `constants.py`. If you need a new constant, add it here.

### 2. `tie_word_embeddings` Must Be `False`

After `resize_token_embeddings()`, always set:
```python
model.config.tie_word_embeddings = False
```
Without this, checkpoint reload destroys `lm_head` → **F1 = 0.00 on every prediction**. This is the single most destructive silent failure.

### 3. Image Resolution: 960 × 1280 (Native DONUT)

Do NOT increase beyond this without `allow_high_res` in the experiment YAML. RAM scales as `(H×W)/(1280×960)`. At 2560×1920 → 4× RAM → OOM.

### 4. Validation Never Precomputes Tensors

`validation.precompute_tensors` must always be `false` in experiment YAML files. Setting it to `true` causes OOM during training.

### 5. GPU Memory Cleanup Between Stages

Always call `torch.cuda.empty_cache()` + `gc.collect()` between DONUT and TrOCR stages. Add `_gpu_cleanup()` at the start of `train_trocr()`.

---

## Datasets

| Dataset | Loader Class | Samples | Source |
|---------|-------------|---------|--------|
| SROIE | `SROIELoader` | ~500 train | Local (ICDAR-2019-SROIE git clone) |
| WildReceipt | `WildReceiptLoader` | ~1,267 | OpenMMLab tar archive |
| FUNSD | `FUNSDLoader` | ~149 | HuggingFace `nielsr/funsd` |
| Invoices-DONUT | `InvoicesDonutLoader` | ~500 | HuggingFace `katanaml-org/invoices-donut-data-v1` |
| CORD-v2 | `CORDv2Loader` | ~800 | HuggingFace `naver-clova-ix/cord-v2` |

All loaders extend the ABC base class in `data_pipeline.py`. New datasets: add to `datasets_registry.json`, implement the loader, call `memory_manager.release_hf_dataset()` + `flush_hf_arrow_cache()`.

---

## 8 Core Experiments

| Exp | Description | Best Global F1 |
|-----|-------------|---------------|
| 1 | SROIE only (baseline) | 0.8503 |
| 2 | SROIE + WildReceipt | — |
| 3 | SROIE + Invoices-DONUT | — |
| 4 | SROIE + WR + Invoices | — |
| 5 | SROIE + WR (2× SROIE oversample) | — |
| **6** | **SROIE + Invoices (2× SROIE oversample)** | **0.8982 (BEST)** |
| 7 | SROIE + All (2× SROIE oversample) | — |
| 8 | SROIE + WR + Inv (3× SROIE oversample) | — |

Experiments are configured via `experiments/exp_*.yaml` files and enabled/disabled in `experiment_selection.json`.

---

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (weight_decay=0.01) |
| Learning rate | 5e-5 encoder / 1e-4 decoder (layerwise) |
| Scheduler | Cosine with adaptive warmup (500 steps, capped at 10% of total) |
| Label smoothing | 0.1 |
| Epochs | 10 (Exps 1-4), 15 (Exps 5-8) |
| Batch size | 8 per GPU, grad_accum=2 → effective 16 |
| Max decode length | 768 tokens |
| Mixed precision | bf16 (Ampere+), fp16 fallback, fp32 on CPU |
| Early stopping | Patience 5 on val_loss |
| Seed | 42 |

---

## Development Workflow

### Running Things

```bash
make setup          # Install deps + validate import chain
make run            # Full pipeline (all experiments + paper + auto-push)
make exp-6          # Single experiment (replace 6 with any ID)
make check          # CI: lint + imports + smoke test
make fix            # AI-powered auto-fix loop (max 5 attempts)
make paper          # Regenerate paper from cached results only
make sync           # Push result JSONs to GitHub
make clean          # Delete cached results (triggers re-train)
```

### Linting

```bash
bash lint.sh        # ruff check --fix . && ruff format .
```

Ruff config: `ruff.toml` — Python 3.10 target, line-length 100, rules: E, W, F, I, UP, B, SIM. Pre-commit hooks configured in `.pre-commit-config.yaml`.

### CI Workflows

1. **`experiments_sroie.yml`** — Runs on every PR/push: ruff lint, YAML experiment validation (`scripts/validate_yaml_experiments.py`), import chain verification. No torch required.
2. **`self_healing_ci.yml`** — Runs on PR: lint + import checks + AI-powered auto-fix (Claude/Mistral). Pushes fixes back to PR branch automatically.

### Docker / Cloud

```bash
docker build -t donut-sroie .
docker run --gpus all -v /workspace:/workspace -e GITHUB_TOKEN=... -e HF_TOKEN=... donut-sroie
```

Or on a fresh Vast.ai instance:
```bash
git clone https://github.com/aiparallel0/kaggle.git && cd kaggle && bash bootstrap.sh
```

---

## The ~2-PR Bug Pattern

Roughly every 2 PRs, a trivial syntax or compatibility issue breaks the CLI. These are always 1-5 line fixes. Common symptoms and fixes:

| Symptom | Fix |
|---------|-----|
| `SyntaxError: invalid syntax` near `)` or `]` | Missing bracket or trailing comma — find line, add it |
| `ImportError: cannot import name 'X'` | Wrong import path or missing package |
| `NameError: name 'true' is not defined` | JSON `true`/`false`/`null` → Python `True`/`False`/`None` |
| `F1 = 0.0000` after checkpoint reload | `config.tie_word_embeddings = False` regression |
| `CUDA out of memory` | Halve `batch_size`, double `gradient_accumulation_steps` |
| `KeyError: 'sroie'` in evaluator | Missing `result.get("sroie", result)` unwrap |

---

## Code Patterns

### OOP Design

- All dataset loaders extend the ABC base in `data_pipeline.py`
- `DonutTrainer` in `train.py` wraps HuggingFace `Seq2SeqTrainer`
- `DonutEvaluator` in `run_experiments.py` handles inference + scoring
- `ExperimentConfig` dataclass holds all hyperparameters — loaded from YAML

### Adding a New Dataset

1. Add entry to `datasets_registry.json`
2. Implement `BaseDatasetLoader` subclass in `data_pipeline.py`
3. Call `memory_manager.release_hf_dataset()` + `flush_hf_arrow_cache()` in loader cleanup
4. Create experiment YAML(s) in `experiments/`
5. Update `experiment_selection.json` if needed

### Adding a New Experiment

1. Create `experiments/exp_NN_description.yaml` following the schema of existing files
2. Required fields: `experiment_id`, `name`, `model`, `data`, `datasets`, `training`, `output`
3. Run `python scripts/validate_yaml_experiments.py experiments/` to catch errors early
4. No duplicate `experiment_id` or `name` across YAML files

---

## Evaluation Metrics

- **Global F1**: harmonic mean of per-field F1 scores (primary metric)
- **Per-field F1**: precision/recall on company, date, address, total
- **NED (Normalized Edit Distance)**: character-level similarity per field
- **Exact Match**: strict string equality (binary per sample)

Results written to `results/experiment_N.json`. Reporting reads these to generate LaTeX tables and plots.

---

## Environment

| Variable | Purpose | Default |
|----------|---------|---------|
| `DONUT_WORKSPACE` | Checkpoint and data root | `/workspace` |
| `SROIE_DATA_DIR` | SROIE dataset path | `/workspace/ICDAR-2019-SROIE/data` |
| `GITHUB_TOKEN` | Auto-push + PR comments | — |
| `HF_TOKEN` | HuggingFace dataset downloads | — |
| `AI_DIAGNOSE` | Enable AI-powered diagnostics | `1` |
| `AI_DIAGNOSE_PROVIDER` | `claude` / `mistral` / `auto` | `auto` |
| `ANTHROPIC_API_KEY` | Claude API key | — |

Token files: `hf_token.txt`, `github_token.txt`, `anthropic_api_key.txt`, `mistral_api_key.txt` (all gitignored).

---

## Key Dependencies

```
torch>=2.0.0, transformers>=4.37.0, accelerate, ultralytics (YOLOv8),
sentencepiece, protobuf>=3.20.0, optuna>=3.0.0, datasets>=2.0.0,
ruff>=0.4.0, flash-attn>=2.0.0, timm, editdistance, jiwer,
opencv-python-headless, tensorboard>=2.14.0, safetensors
```

**flash-attn install order matters:** install torch FIRST, then `pip install flash-attn --no-build-isolation`.

---

## Guardrails — What Must Never Be Done

1. **Never duplicate constants** — import from `constants.py`
2. **Never set `tie_word_embeddings: true`** — causes F1=0.00
3. **Never set `validation.precompute_tensors: true`** — causes OOM
4. **Never exceed 1280×960 resolution** without `allow_high_res` and explicit RAM budgeting
5. **Never skip the import chain check** before running experiments
6. **Never bake secrets into Docker images** — inject at runtime via env vars
7. **Never mix `is:issue` and `is:pr`** in GitHub search queries
8. **Always run `bash lint.sh`** (ruff check + format) before committing
9. **Always validate YAML** with `python scripts/validate_yaml_experiments.py experiments/` after editing experiment configs
10. **Always free GPU memory** between DONUT and TrOCR stages