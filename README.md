# Multi-Dataset Fine-Tuning of DONUT for Receipt Information Extraction

A systematic study of multi-dataset fine-tuning for receipt key information extraction (KIE) using the DONUT (Document Understanding Transformer) model and TrOCR+YOLO comparison.

**Quick fact:** Starting from a CORD-pretrained DONUT checkpoint, this pipeline trains on the SROIE benchmark combined with three auxiliary datasets (WildReceipt, CORD, Invoices-DONUT) across 8 experiment configurations, evaluates each model on the SROIE test set, and generates a complete LaTeX paper with real results.

---

## 🚀 Quick Start

### Installation & Run (Recommended)

```bash
pip install -r requirements.txt
python run_all.py
```

This runs the full end-to-end pipeline:
1. Downloads and splits SROIE data (500 train / 63 val / 63 test)
2. Downloads and normalizes 3 auxiliary datasets
3. Evaluates pretrained CORD baseline (zero-shot)
4. Trains 8 DONUT experiments (train + evaluate each)
5. Trains 8 TrOCR+YOLO experiments (train + evaluate each)
6. Generates comparison tables and plots
7. Fills `paper.tex` with all real metrics → `paper_filled.tex`

**Time:** ~12+ hours on A100 GPU

### Quick Test Mode (30 minutes)

Test the pipeline faster with just Experiment 1 + TrOCR+YOLO:

```bash
python run_all.py --quick
```

**What it does:**
- Installs dependencies automatically
- Sets up SROIE data (Stage 0)
- Downloads auxiliary datasets (Stage 1)
- Trains DONUT Experiment 1 (SROIE baseline)
- Trains TrOCR+YOLO
- Generates `results.tex` with 2D loss curves and metrics table
- All output logged to `terminal.txt`

**Outputs:**
- `results.tex` — LaTeX document with loss plots and metrics
- `results_plots/` — PNG figures (donut_loss.png, trocr_loss.png)
- `terminal.txt` — Complete execution log
- `results/experiment_1.json` — DONUT Exp 1 metrics

### Hyperparameter Sweep Mode (Optional)

Test multiple hyperparameter combinations to optimize performance:

```bash
python run_all.py --quick --all
```

**What it does:**
- Runs quick test with default parameter grids:
  - Batch sizes: 4, 8, 16
  - Epochs: 5, 10, 15
  - Learning rates: 1e-5, 5e-5, 1e-4
  - Schedulers: linear, cosine
- Generates comprehensive `results.tex` with parameter variation tables and overlay plots

**Custom parameter grid:**

```bash
python run_all.py --quick --all --param-grid batch_size 8 16 --param-grid epochs 8 10
```

This tests only batch_size=[8, 16] and epochs=[8, 10] for faster iteration.

**Configure defaults in `run_all.py`:**

Edit the `TRAINING_PARAMS` dict to change defaults for quick mode:

```python
TRAINING_PARAMS = {
    "batch_size": 8,           # Change default batch size
    "epochs": 10,              # Change default epochs
    "learning_rate": 5e-5,     # Change default learning rate (1e-5 to 1e-4 typical)
    "lr_scheduler_type": "cosine",  # or "linear" / "constant"
}
```

### Single Experiment

```bash
python run_all.py --experiment 2
```

### Generate Paper Only

```bash
python run_all.py --paper-only
```

Regenerates `paper_filled.tex` from existing results without re-training.

---

## 🔬 Standalone Scripts (Alternative Workflows)

These scripts provide isolated entry points for development and debugging. Use them for hyperparameter exploration, data validation, or testing individual components.

### dataset_preparation.py — Dataset & Annotation Prep

Prepares YOLO bounding box labels and TrOCR line crops from SROIE images.

```bash
python dataset_preparation.py                # Prepare all data
python dataset_preparation.py --validate     # Validate existing data
python dataset_preparation.py --force        # Re-prepare (clear existing)
```

**Outputs:** `data/yolo/{train,val,test}/` and `data/trocr/{train,val,test}/`

### train_donut.py — DONUT Reference Implementation

Standalone DONUT fine-tuning (useful for hyperparameter exploration).

```bash
python train_donut.py                        # Train DONUT model
python train_donut.py --dry-run              # Validate setup
python train_donut.py --sweep                # Generate hyperparameter configs
python train_donut.py --config N             # Train specific config
```

**Outputs:** `models/donut_finetuned/best/` and `models/donut_finetuned/training_history.json`

### train_trocr_yolo.py — TrOCR + YOLO Pipeline

Two-stage OCR pipeline: YOLOv8 detection + TrOCR reading + heuristic assignment.

```bash
python train_trocr_yolo.py                   # Train TrOCR+YOLO
```

**Outputs:** `models/yolo_finetuned/run/weights/best.pt` and `models/trocr_finetuned/best/`

### evaluate_models.py — Unified Model Evaluation

Evaluate both architectures on the same 63 SROIE test images with standardized metrics.

```bash
python evaluate_models.py                           # Evaluate both architectures
python evaluate_models.py --donut-only              # DONUT only
python evaluate_models.py --trocr-only              # TrOCR+YOLO only
python evaluate_models.py --report                  # Generate HTML report
```

**Outputs:**
- `results/metrics.json` — Raw metrics
- `results/evaluation_summary.json` — Structured summary
- `results/evaluation_report.html` — Interactive HTML report

---

## 📋 Command Reference

### `run_all.py` — Main Pipeline

```bash
python run_all.py [options]

Options:
  --experiment N      Run only experiment N (1–8)
  --force             Delete cached results and re-run from scratch
  --paper-only        Skip training; generate paper from existing results
  --skip-install      Skip Stage 0 SROIE auto-install (data already present)
  --skip-download     Skip dataset download/verification stage
  --skip-pretrained   Skip pretrained baseline evaluation step
  --skip-trocr        Skip TrOCR+YOLO pipeline (DONUT only)
  --quick             Quick test mode: train only Exp 1 + TrOCR+YOLO
  --sroie-dir PATH    Path to SROIE data (default: /workspace/ICDAR-2019-SROIE/data)
  --workspace PATH    Workspace root for model checkpoints (default: /workspace)
  --paper-template F  LaTeX template to fill (default: paper.tex)
  --output F          Output filled LaTeX file (default: paper_filled.tex)
```

### `run_experiments.py` — DONUT Only

```bash
python run_experiments.py --all              # all 8 experiments
python run_experiments.py --experiment 3     # single experiment
python run_experiments.py --all --force      # force re-run
```

### `inject_results.py` — Paper Generation

```bash
python inject_results.py --all --paper paper.tex --output paper_filled.tex
```

---

## 📝 Understanding the Output

### terminal.txt

Complete execution log with timestamps for each major stage, training progress, inference results, and errors. Check this file if something goes wrong.

Example:
```
2025-02-28 14:32:10 | run_all | INFO | ========================================================================
2025-02-28 14:32:10 | run_all | INFO | QUICK MODE: Single DONUT Experiment + TrOCR+YOLO
2025-02-28 14:35:22 | run_all | INFO | [Stage 0] SROIE data install...
2025-02-28 14:42:15 | run_all | INFO | [Stage 2] Training DONUT Experiment 1 (SROIE baseline)...
```

### results.tex (Quick Mode)

LaTeX document containing:
- Training configuration (batch size, epochs, LR, scheduler)
- 2D loss curves (train vs val)
- Metrics table (F1, NED per field)
- Optional terminal output snippets

Compile to PDF:
```bash
pdflatex results.tex
```

---

## 💾 Installation Methods

### Method 1: Direct Execution (Recommended)

```bash
pip install -r requirements.txt
python run_all.py
```

### Method 2: Package Installation

```bash
pip install .                    # Install from local directory
pip install -e .                 # Install in development mode
python -m run_all                # Run via module invocation
```

### Method 3: GitHub Installation

```bash
pip install git+https://github.com/aiparallel0/kaggle.git
python -m run_all
```

**Auto-Install Features:**
- Dependencies automatically installed from `requirements.txt` if needed
- Dual-stream logging: all output to `terminal.txt`, console shows filtered progress
- Professional CLI interface with argparse-based options

---

## 📊 Experiment Definitions

8 DONUT fine-tuning experiments with different dataset combinations. All use 80/10/10 SROIE split: **500 train / 63 val / 63 test**.

| Exp | Training Data | Approx. Samples | Expected F1 |
|---|---|---|---|
| 1 | SROIE only (baseline) | ~500 | 0.83–0.84 |
| 2 | SROIE + WildReceipt | ~2,240 | 0.85–0.86 |
| 3 | SROIE + Invoices-DONUT | ~1,300 | 0.84–0.85 |
| 4 | SROIE + FUNSD | ~1,400 | 0.86–0.87 |
| 5 | SROIE + WildReceipt + FUNSD | ~3,140 | 0.87–0.88 |
| 6 | SROIE + WildReceipt + Invoices | ~3,040 | 0.87–0.88 |
| 7 | SROIE + FUNSD + Invoices | ~2,200 | 0.86–0.87 |
| 8 | SROIE + All datasets | ~3,940 | 0.88–0.90 |

---

## 📚 Data Sources

| Dataset | Source | Notes |
|---|---|---|
| **SROIE** | Auto-downloaded from `https://github.com/zzzDavid/ICDAR-2019-SROIE.git` | 80/10/10 split applied (500 train / 63 val / 63 test) |
| **WildReceipt** | `https://download.openmmlab.com/mmocr/data/wildreceipt.tar` | OpenMMLab tar download |
| **FUNSD** | HuggingFace `nielsr/funsd` | No token required |
| **Invoices-DONUT** | HuggingFace `katanaml-org/invoices-donut-data-v1` | HF token recommended for 5–10× faster download |

**HuggingFace Token:** Place your token in `hf_token.txt` (single line, gitignored). Enables faster downloads.

---

## 🏗️ Repository Structure

```
kaggle/
├── run_all.py               # MAIN ENTRY POINT: full dual-architecture pipeline
├── run_experiments.py       # 8-experiment DONUT orchestrator
├── CLAUDE.md                # Authoritative AI guide & project rules
├── README.md                # This file
│
├── [Core Pipeline]
├── constants.py             # Shared constants (SINGLE SOURCE OF TRUTH)
├── dataset_loaders.py       # ABC-based dataset loaders
├── train.py                 # DonutTrainer class
├── donut_evaluator.py       # Evaluation & metrics computation
├── inject_results.py        # LaTeX paper generation
├── paper.tex                # LaTeX paper template
├── requirements.txt         # Python dependencies
│
├── [Standalone/Educational Scripts]
├── dataset_preparation.py     # Data prep with validation CLI
├── train_donut.py             # DONUT training with sweep support
├── train_trocr_yolo.py        # TrOCR+YOLO pipeline
├── evaluate_models.py         # Unified evaluation + HTML reporting
│
├── [Configuration & Templates]
├── paper.tex                # LaTeX research paper template
├── references.bib           # Bibliography
├── training_config.py       # Hyperparameter grids
├── requirements.txt         # Python dependencies
│
└── [Runtime Artifacts (gitignored)]
    ├── data/                # Cached datasets
    ├── models/              # Checkpoints & fine-tuned weights
    ├── results/             # Experiment JSON outputs & plots
    └── paper_filled.tex     # Generated paper (auto-produced)
```

---

## 📖 Key Invariants

**Train/test split isolation:**
- `val_img/` and `test_img/` are populated once by `stage_install()` and are never mixed
- Training early stopping reads only `val_img/`
- Final scoring reads only `test_img/`
- Auxiliary datasets (WildReceipt, FUNSD, Invoices) contribute to training only—their held-out 15% test portion is discarded

**Constants centralization:**
- All shared constants live **exclusively** in `constants.py`
- Never redeclare `FIELDS`, `IMAGE_EXTS`, `MAX_LENGTH`, etc. in other files
- Single source of truth prevents silent drift bugs

---

## ⚡ Performance Tips

- **HuggingFace Token:** Create `hf_token.txt` with your token for 5-10× faster downloads
  - Get one at https://huggingface.co/settings/tokens
- **Parallel downloads:** All auxiliary datasets download in parallel automatically
- **RAM cache:** Images pre-loaded into RAM when sufficient memory available
- **DataLoader:** Optimized with `pin_memory=True`, `prefetch_factor=4`, `persistent_workers=True`
- **GPU:** CUDA auto-detected; CPU fallback supported but 10–20× slower
- **Test on smaller data first:** Run `--quick` mode before full pipeline

---

## 🧪 Results Format

Each experiment saves `results/experiment_N.json`:

```json
{
  "experiment_id": 1,
  "name": "SROIE only (baseline)",
  "datasets": ["sroie"],
  "num_train_samples": 500,
  "metrics": {
    "global_f1": 0.9123,
    "global_precision": 0.9200,
    "global_recall": 0.9048,
    "overall_exact_match": 0.8500,
    "company_f1": 0.9500,
    "company_ned": 0.0312,
    "date_f1": 0.9800,
    "date_ned": 0.0150,
    "address_f1": 0.8700,
    "address_ned": 0.0890,
    "total_f1": 0.9500,
    "total_ned": 0.0280
  }
}
```

---

## 📊 Evaluation Metric

**Global F1** over all (image, field) pairs:
- A pair is **TP** if `predicted_string == ground_truth_string` (case-insensitive, stripped)
- **NED** (Normalized Edit Distance via `editdistance`) reported per field — lower is better ↓
- **Exact Match** counts full 4-field predictions where all fields match

---

## 🛠️ Troubleshooting

### `python run_all.py --quick` hangs

- Check that SROIE data is cloned (see Stage 0 output in `terminal.txt`)
- Try adding `--skip-install` if SROIE already exists: `python run_all.py --quick --skip-install`
- Check GPU memory: quick mode needs ~8 GB VRAM

### results.tex won't compile

- Check for missing packages: `pip install matplotlib numpy`
- Verify `results_plots/` directory exists and contains PNG files
- Try opening the tex file in a text editor to see if there are obvious LaTeX errors

### Terminal output not appearing in terminal.txt

- Check file permissions: `ls -la terminal.txt`
- Try removing and re-running: `rm terminal.txt && python run_all.py --quick`

### Out of memory (OOM) errors

- Reduce batch size: `--param-grid batch_size 4 8` (instead of default 8)
- Skip TrOCR+YOLO: `python run_all.py --skip-trocr` (DONUT only)

---

## 📚 Common Use Cases

### Test if a hyperparameter change helps

```bash
# Edit TRAINING_PARAMS to set batch_size=16 (instead of default 8)
python run_all.py --quick

# Results in results.tex show F1, NED, and loss curves
# Compare against previous runs to see if change helped
```

### Find optimal batch size

```bash
python run_all.py --quick --all --param-grid batch_size 4 8 16 32
```

Generates results.tex with batch size comparison. Look at the "Comparison Plots" section.

### Test multiple learning rates

```bash
python run_all.py --quick --all --param-grid learning_rate 1e-5 3e-5 1e-4 3e-4
```

### Debug a broken configuration

Run quick test first (fast failure detection):

```bash
python run_all.py --quick
```

If it fails, check `terminal.txt` for error messages. Once fixed, run full pipeline:

```bash
python run_all.py
```

### Validate data setup

```bash
python dataset_preparation.py --validate
```

---

## 🐛 Known Issues Fixed

Three critical bugs that previously caused catastrophic F1 score collapses. All are fixed and guarded with tests.

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| **F1 ≈ 0.42** (plausible-looking) | `safetensors` omits `lm_head.weight` from checkpoint shards because it shares data pointer with `embed_tokens.weight` after `resize_token_embeddings()`. On reload, `lm_head` randomly re-initialized. | `LmHeadCloneCallback` deep-clones weight before every save. `load_model_with_tied_weights()` raises `RuntimeError` immediately if `lm_head.weight` still missing. |
| **F1 ≈ 0.008** (near-zero, not zero) | `token2json()` returns **list** of page-dicts when generated sequence contains `<sep/>` tokens (inherited from CORD pretraining). `_parse_prediction()` treated any non-dict as parse failure, returning `{}`. | `_parse_prediction()` and `_self_test()` merge list of pages into single flat dict (first occurrence of each key wins). |
| **F1 unreliable / overfitted** | `val_img/` directory not created; `load_sroie_val()` returned `[]`; `do_eval=False`; no early stopping; model evaluated on test data indirectly. | `stage_install()` explicitly moves 63 images into `val_img/` and 63 into `test_img/` — physically distinct directories checked by tests. |

---

## 📖 Full Documentation

**For complete architecture details, known issues & historical fixes, training time estimates, hyperparameter optimization, and development workflows, see [`CLAUDE.md`](CLAUDE.md).**

This file serves as the authoritative technical guide for AI agents and developers working on this codebase.

---

## 🤝 Contributing

See [`CLAUDE.md`](CLAUDE.md) for:
- Development workflows
- OOP design patterns
- Known issues & historical fixes
- Performance tips
- Full end-to-end pipeline flow diagram

---

## 📝 Citation

If you use this project in your research, please cite the authors and datasets used:

```bibtex
@misc{donut-multi-dataset-fine-tuning,
  title = {Multi-Dataset Fine-Tuning of DONUT for Receipt Information Extraction},
  year = {2026}
}
```

---

## ⚠️ Limitations

- Test set (63 images from SROIE training split) is **not** the official SROIE test set (347 images with no public ground truth)
- Leaderboard comparisons in paper are for contextual reference only
- Single GPU, limited hyperparameter search, English-centric evaluation

---

---

## 🔧 Advanced Configuration

### Hyperparameter Defaults

Edit `training_config.py` to change default hyperparameter grids for sweep mode:

```python
PARAM_GRIDS_DEFAULT = {
    "batch_sizes": [8, 16],           # Test fewer sizes for speed
    "epochs_list": [5, 10],           # Fewer epochs for speed
    "learning_rates": [5e-5, 1e-4],   # Focus on high-performing rates
    "schedulers": ["cosine"],         # Test only cosine scheduler
}
```

This reduces the number of combinations and speeds up sweeps.

### Environment Variables

- `SROIE_DATA_DIR` — Path to SROIE data (default: `/workspace/ICDAR-2019-SROIE/data`)
- `DONUT_WORKSPACE` — Workspace root for model checkpoints (default: `/workspace`)
- `HF_TOKEN` — HuggingFace token (prefer `hf_token.txt` instead)

### Package Installation

`setup.py` enables package-style installation:

```bash
pip install -e .                     # Development mode
pip install git+https://github.com/aiparallel0/kaggle.git
donut-kie                            # CLI alias (if installed via setup.py)
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | One or more experiments had no training data (partial results saved) |
| `2` | Fatal error (missing SROIE data, unrecoverable failure) |

---

## 📚 Key Features

### ✅ Features Included

- **Auto-Install Dependencies** — Automatically installs `requirements.txt` before training
- **Dual-Stream Logging** — All output logged to `terminal.txt`; console shows filtered progress
- **Quick Mode (`--quick`)** — Run only SROIE baseline + TrOCR+YOLO in ~30 minutes
- **Hyperparameter Sweep** — Test multiple parameter combinations with `--quick --all`
- **Professional CLI** — Clean argparse-based interface with backward compatibility
- **2D Loss Curves** — Generates training/validation loss plots in results.tex
- **Paper Generation** — Automatic LaTeX paper with all metrics dynamically injected

### ⚠️ Known Limitations

- Test set (63 images from SROIE training split) is **not** the official SROIE test set (347 images with no public ground truth)
- Leaderboard comparisons in paper are for contextual reference only
- Single GPU, limited hyperparameter search, English-centric evaluation
- Loss history capture requires trainer.py integration to fully save and display loss curves
- Hyperparameter application in quick mode may require additional trainer integration

---

## 📖 Documentation Organization

| Document | Purpose | Audience |
|---|---|---|
| **README.md** (this file) | Quick start, CLI reference, common tasks | All users |
| **CLAUDE.md** | Complete technical architecture, bug fixes, development workflows | Developers, AI agents |
| Standalone scripts | Educational, isolated components | Advanced users |

---

**For help with Claude Code features and hooks, see `/help` or report issues at https://github.com/anthropics/claude-code/issues**
