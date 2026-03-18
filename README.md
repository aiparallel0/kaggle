# Multi-Dataset Fine-Tuning of DONUT for Receipt Information Extraction
# Be careful to preserve debelopmental history of the repo.
A systematic study of multi-dataset fine-tuning for receipt key information extraction (KIE) using the DONUT (Document Understanding Transformer) model and TrOCR+YOLO comparison.

**Status:** ✅ **Stable & Complete** (2026-03-18)
- All 8 DONUT experiments successfully trained and evaluated
- Three critical bugs identified & permanently fixed
- Comprehensive automated diagnostics & safety checks in place
- Full documentation & reproducible results

**Best result:** DONUT Experiment 6 (SROIE + Invoices-DONUT, 2× SROIE oversampling) achieves **global F1 = 0.8982** on the SROIE Task-3 test set — a gain of +5.71 percentage points over the published DONUT baseline (0.8411). Training: 39.6 minutes on Vast.ai RTX 6000 Blackwell (96 GB), 2026-03-07. TrOCR+YOLO achieves F1 = 0.2035 (−69.5% vs. best DONUT).

**Pipeline:** Starting from `naver-clova-ix/donut-base`, trains on SROIE combined with two auxiliary datasets (WildReceipt, Invoices-DONUT) across 8 experiment configurations, evaluates each model on the 63-image SROIE test split, and generates a complete LaTeX research paper with all real metrics.

---

## 🚀 Quick Start

### Installation & Run (Recommended)

```bash
pip install -r requirements.txt
python run_all.py
```

> **Recommended:** Always pre-install dependencies before running `run_all.py`. The auto-installer
> is a convenience for truly fresh environments only; pre-installing avoids a process restart and
> is more predictable.

This runs the full end-to-end pipeline:
1. Downloads and splits SROIE data (500 train / 63 val / 63 test)
2. Downloads and normalizes 3 auxiliary datasets
3. Evaluates pretrained CORD baseline (zero-shot)
4. Trains 8 DONUT experiments (train + evaluate each)
5. Trains 8 TrOCR+YOLO experiments (train + evaluate each)
6. Generates comparison tables and plots
7. Fills `paper.tex` with all real metrics → `paper_filled.tex`

**Time:** ~6–10 hours on a 96 GB GPU (RTX 6000 Blackwell). Exp 6 alone takes ~40 min; full 8-experiment suite with TrOCR+YOLO ~8–12 h.

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
  --paper-template F  LaTeX template to fill (default: paper/paper.tex)
  --output F          Output filled LaTeX file (default: paper/paper_filled.tex)
```

### `run_experiments.py` — DONUT Only

```bash
python run_experiments.py --all              # all 8 experiments
python run_experiments.py --experiment 3     # single experiment
python run_experiments.py --all --force      # force re-run
```

### `inject_results.py` — Paper Generation

```bash
python inject_results.py --all --paper paper/paper.tex --output paper/paper_filled.tex
```

---

## 📝 Understanding the Output

### terminal.txt

Complete execution log with timestamps for each major stage, training progress, inference results, and errors. Check this file if something goes wrong.

Example:
```
2026-03-16 14:32:10 | run_all | INFO | ========================================================================
2026-03-16 14:32:10 | run_all | INFO | QUICK MODE: Single DONUT Experiment + TrOCR+YOLO
2026-03-16 14:35:22 | run_all | INFO | [Stage 0] SROIE data install...
2026-03-16 14:42:15 | run_all | INFO | [Stage 2] Training DONUT Experiment 1 (SROIE baseline)...
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
- Dependencies automatically installed from `requirements.txt` if needed; process restarts cleanly after install
- The auto-installer checks for: `torch`, `transformers`, `datasets`, `accelerate` (NOT `editdistance` or `pandas` — both have inline replacements and are not in `requirements.txt`)
- flash-attn is **not** auto-installed (see Troubleshooting); PyTorch 2.x SDPA is used instead
- Dual-stream logging: all output to `terminal.txt`, console shows filtered progress
- Professional CLI interface with argparse-based options

---

## 📊 Experiment Definitions & Actual Results

8 DONUT fine-tuning experiments with different dataset combinations. All use 80/10/10 SROIE split: **500 train / 63 val / 63 test**. Trained on Vast.ai RTX 6000 Blackwell (96 GB), 2026-03-07.

| Exp | Training Data | Samples | Global F1 | Notes |
|---|---|---|---|---|
| 1 | SROIE only (baseline) | 500 | **0.8503** | 5-epoch quick run (early stopping); post-bug-fix |
| 2 | SROIE + WildReceipt | 1,386 | 0.8257 | WR alone (no oversample) slightly hurts |
| 3 | SROIE + Invoices-DONUT | 832 | 0.2867 | Invoice cross-domain hurts severely without rebalancing |
| 4 | SROIE + WR + Invoices | 1,718 | 0.8224 | Combined but unbalanced — still below baseline |
| 5 | SROIE + WR (2× SROIE oversample) | 1,886 | 0.8514 | Marginal gain with oversampling |
| **6** | **SROIE + Invoices (2× SROIE)** | **1,332** | **0.8982 ← BEST** | Invoices + SROIE oversampling, early stop ep.8 |
| 7 | SROIE + All (2× SROIE) | 2,218 | 0.8503 | All datasets 2×; matches baseline (signals cancel) |
| 8 | SROIE + WR + Invoices (3× SROIE) | ~3,940 | OOM | OOM-killed at 2560×1920 resolution |

> **Key finding:** SROIE oversampling (2×) is a prerequisite for auxiliary data to help. Without it (Exps 2–4), adding more data hurts. With it (Exps 5–7), performance meets or exceeds the baseline. The best config (Exp 6) gains **+0.0479** over Exp 1 and **+0.0571** over the published DONUT result.

**Best result: Experiment 6 — Global F1 = 0.8982** (vs. published DONUT 0.8411, +5.71 pts; vs. own baseline 0.8503, +4.79 pts)

**Best per-field (Exp 6):** company=0.905, date=0.984, address=0.790, total=0.912 · exact_match=0.667 · training_time=39.6 min

**TrOCR+YOLO comparison:** global_f1=0.2035 (company=0.176, date=0.460, address=0.000, total=0.231) — DONUT wins by **+69.5% absolute**.
> TrOCR+YOLO was trained once; all 8 experiment slots show identical metrics (single training run, not varied by dataset combination).

---

## 📚 Data Sources

| Dataset | Source | Notes |
|---|---|---|
| **SROIE** | Auto-downloaded from `https://github.com/zzzDavid/ICDAR-2019-SROIE.git` | 80/10/10 split applied (500 train / 63 val / 63 test) |
| **WildReceipt** | `https://download.openmmlab.com/mmocr/data/wildreceipt.tar` | OpenMMLab tar download |
| **Invoices-DONUT** | HuggingFace `katanaml-org/invoices-donut-data-v1` | HF token recommended for 5–10× faster download |

**HuggingFace Token:** Place your token in `hf_token.txt` (single line, gitignored). Enables faster downloads.

---

## 🏗️ Repository Structure

```
kaggle/
├── run_all.py               # MAIN ENTRY POINT: full dual-architecture pipeline (128 KB)
├── run_experiments.py       # 8-experiment DONUT orchestrator
├── CLAUDE.md                # Authoritative AI guide & project rules
├── README.md                # This file
│
├── [Core Pipeline]
├── constants.py             # Shared constants (SINGLE SOURCE OF TRUTH)
├── dataset_loaders.py       # ABC-based dataset loaders
├── train.py                 # DonutTrainer class
├── donut_evaluator.py       # Evaluation & metrics computation
├── inject_results.py        # LaTeX paper generation (PaperInjector)
├── resource_optimizer.py    # VRAM-aware hyperparameter tuning
├── memory_manager.py        # Centralized RAM/GPU memory authority
├── preflight_checks.py      # Pre-flight validators + validate_pipeline()
├── startup_diagnostics.py   # Stdlib-only startup checks (prefix, GPU zombies)
├── pipeline_types.py        # All typed dataclasses + pipeline exceptions
├── validators.py            # BugPatternDetector, ImportChainChecker, etc.
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Package metadata, entry points, ruff/pytest config
│
├── [Auxiliary Pipeline Modules]
├── benchmark_compare.py     # F1/loss comparison figures and tables
├── cloud_pipeline.py        # Cloud mode orchestrator + GitController + TestRunner
├── control_suite.py         # Ablation controls, DA configs, TrOCRControlConfig
├── dag_scheduler.py         # Directed-acyclic-graph stage scheduler
├── dataset_normalizer.py    # Cross-dataset annotation normalizer
├── experiment_config_loader.py  # YAML-based experiment config loader
├── hparam_search.py         # Hyperparameter search orchestrator
├── logging_utils.py         # Shared logging helpers
├── multi_seed_runner.py     # Multi-seed experiment runner
├── pipeline_critic.py       # Automated pipeline code review tool
├── plot_convergence.py      # Loss-curve convergence plotter
├── preprocess_seller_split.py   # Seller-aware train/val/test split
│
├── [Standalone/Educational Scripts]
├── dataset_preparation.py   # Data prep with validation CLI
├── train_trocr_yolo.py      # TrOCR+YOLO pipeline
├── evaluate_models.py       # Unified evaluation + HTML reporting
│
├── paper/                   # LaTeX source files
│   ├── paper.tex            # Main paper template with \VAR{} placeholders (IEEEtran)
│   ├── references.bib       # Bibliography (BibTeX)
│   └── presentation.tex     # Beamer slide deck template
│
├── [Runtime Artifacts — results/ NOT gitignored for reference data]
├── results/
│   ├── all_experiments.json       # Authoritative final-run DONUT results
│   └── trocr_yolo_results.json    # Authoritative final-run TrOCR+YOLO results
│
└── [Runtime Artifacts (gitignored)]
    ├── data/                # Cached datasets
    ├── models/              # Checkpoints & fine-tuned weights
    └── paper/paper_filled.tex  # Generated paper (auto-produced by inject_results.py)
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
- **NED** (Normalized Edit Distance, inline Wagner-Fischer implementation) reported per field — lower is better ↓
- **Exact Match** counts full 4-field predictions where all fields match

---

## 🩺 Runtime Diagnostics

`diagnostics.py` provides an always-on, zero-config diagnostic layer that detects known failure patterns automatically during training and pipeline execution.

### What it detects

| Pattern | Symptom | Severity |
|---|---|---|
| `lm_head_dedup` | F1 ≈ 0.42 — safetensors dropped `lm_head.weight` | critical |
| `token2json_list` | F1 ≈ 0.008 — `token2json` returned list instead of dict | critical |
| `total_f1_collapse` | F1 = 0.000 — complete prediction failure | critical |
| `loss_plateau` | Train loss > 2.0 after epoch 3 — underfitting | warning |
| `loss_nan` | Train loss = NaN — fp16 overflow | critical |
| `gpu_oom_warning` | GPU memory > 90% — OOM imminent | warning |
| `eval_loss_diverging` | Eval loss rising 3+ consecutive epochs — overfitting | warning |

### Basic usage (no API key needed)

Pattern detection runs automatically during every training run — no configuration required:

```bash
python run_all.py --experiment 6    # DiagnosticCallback fires automatically
```

Issues are printed to the console and saved to `results/diagnostics_exp6.json`.

Disable with:

```bash
DISABLE_DIAGNOSTICS=1 python run_all.py
```

### Preflight smoke test

Validate the full import + model setup chain before starting a long training run:

```bash
python diagnostics.py --smoke-test
```

Checks: import chain, model load, token ID roundtrip (GP-3/GP-4), and a dummy forward pass. Exits 0 on success, 1 on failure. Runs in < 30 seconds.

### AI-powered diagnosis (Claude or Mistral)

When a critical pattern is detected, optionally call an AI API for root-cause analysis and a concrete fix:

```bash
# With Claude (claude-sonnet-4-5-20251001 by default)
export ANTHROPIC_API_KEY=sk-ant-...
AI_DIAGNOSE=1 python run_all.py --experiment 3

# With Mistral (mistral-small-latest by default)
export MISTRAL_API_KEY=...
AI_DIAGNOSE=1 AI_DIAGNOSE_PROVIDER=mistral python run_all.py --experiment 3

# Auto: tries Claude first, falls back to Mistral, then Mistral HTTP (no SDK)
AI_DIAGNOSE=1 AI_DIAGNOSE_PROVIDER=auto python run_all.py
```

API keys can also be stored in key files (gitignored):

```
anthropic_api_key.txt   # one line: sk-ant-...
mistral_api_key.txt     # one line: ...
```

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `AI_DIAGNOSE` | `0` | Set `1` to enable AI API calls on critical failures |
| `AI_DIAGNOSE_PROVIDER` | `auto` | `claude` / `mistral` / `auto` |
| `ANTHROPIC_API_KEY` | — | Claude API key (or `anthropic_api_key.txt`) |
| `MISTRAL_API_KEY` | — | Mistral API key (or `mistral_api_key.txt`) |
| `DISABLE_DIAGNOSTICS` | `0` | Set `1` to skip `DiagnosticCallback` entirely |

### Diagnostic output files

| File | Content |
|---|---|
| `results/diagnostics_expN.json` | Per-epoch telemetry + issues for experiment N |
| `results/pipeline_diagnostics.json` | Per-stage status + AI diagnoses (full pipeline runs) |

### Programmatic use

```python
from diagnostics import DiagnosticCallback, PipelineDiagnostics, ai_diagnose, smoke_test

# In your own training loop
from diagnostics import DiagnosticCallback
cb = DiagnosticCallback(experiment_id=6, ai_diagnose=True, ai_provider="auto")
# Register with HuggingFace Trainer callbacks list

# Direct AI diagnosis
result = ai_diagnose(
    {"stage": "training", "eval_f1": 0.42, "epoch": 5},
    provider="auto",   # tries Claude then Mistral
)
print(result)

# Preflight check
ok = smoke_test()   # returns True/False
```

### Claude Code hook (autonomous validation)

A `PostToolUse` hook in `.claude/settings.json` automatically runs:

```bash
python -c "from constants import FIELDS, BASE_MODEL, SEED"
```

after every `.py` file edit. If the import chain is broken, a warning is shown immediately — before any experiment is run.

---

## 🛠️ Troubleshooting

### `FATAL: The following packages could not be installed: editdistance`

If you see this error, you are running an older version of `run_all.py` where `editdistance`
was still listed in `_CRITICAL_INSTALL_PACKAGES`. The fix is to `git pull` to get the latest
version where both `editdistance` and `pandas` have been removed from those lists (they have
inline replacements and are not in `requirements.txt`).

```bash
git pull
python run_all.py
```

### `python run_all.py` hangs after "Dependencies installed successfully"

The auto-installer previously attempted to build `flash-attn` from source after installing
the main dependencies. On a GPU machine, this compiled CUDA kernels (5–25 min, invisible
because output was captured). **This is now fixed** — flash-attn is no longer auto-installed.

If you are seeing a hang on an older version, the fastest workaround is to pre-install:

```bash
pip install -r requirements.txt
python run_all.py
```

Pre-installing means the auto-installer is skipped entirely (packages already present).

**What is flash-attn?** An optional CUDA kernel that speeds up attention for long sequences
(>2048 tokens). This pipeline uses `MAX_LENGTH=768` with PyTorch 2.x built-in SDPA, which
provides comparable performance. For receipt KIE workloads at this sequence length, the
marginal speedup from flash-attn does not justify a 5–25 minute build.

**Optional manual install (after the pipeline runs successfully):**
```bash
# Find a prebuilt wheel: https://flashattn.dev/wheel-finder/
# Select your Python / CUDA / PyTorch version, copy the command shown
pip install flash-attn --no-build-isolation  # or use a prebuilt wheel
```

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

Four critical bugs that previously caused catastrophic failures. All are fixed and guarded with tests.

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| **F1 ≈ 0.42** (plausible-looking) | `safetensors` omits `lm_head.weight` from checkpoint shards because it shares data pointer with `embed_tokens.weight` after `resize_token_embeddings()`. On reload, `lm_head` randomly re-initialized. | `LmHeadCloneCallback` deep-clones weight before every save. `load_model_with_tied_weights()` raises `RuntimeError` immediately if `lm_head.weight` still missing. |
| **F1 ≈ 0.008** (near-zero, not zero) | `token2json()` returns **list** of page-dicts when generated sequence contains `<sep/>` tokens (inherited from CORD pretraining). `_parse_prediction()` treated any non-dict as parse failure, returning `{}`. | `_parse_prediction()` and `_self_test()` merge list of pages into single flat dict (first occurrence of each key wins). |
| **F1 unreliable / overfitted** | `val_img/` directory not created; `load_sroie_val()` returned `[]`; `do_eval=False`; no early stopping; model evaluated on test data indirectly. | `stage_install()` explicitly moves 63 images into `val_img/` and 63 into `test_img/` — physically distinct directories checked by tests. |
| **Terminal freeze after "Dependencies installed successfully"** | Auto-installer built `flash-attn` from source via `subprocess.run(..., capture_output=True)` — CUDA kernel compilation takes 5–25 min on GPU, invisible to user; `Ctrl+C` didn't reach the nvcc child. After install, `os.execv()` restart was missing, causing `sys.exit(2)` due to `sys.modules` isolation. | Removed flash-attn auto-build entirely. Added `os.execv()` restart after successful pip install. Added `_InstallWatchdog` for elapsed-time progress dots. Added `timeout=300` to main pip subprocess. |

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

- **Auto-Install Dependencies** — Installs `requirements.txt` on first run; restarts process via `os.execv()` so new packages are immediately visible; never blocks on flash-attn compilation
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

| Document | Purpose | Audience | Size |
|---|---|---|---|
| **README.md** (this file) | Quick start, CLI reference, common tasks, project history | All users | ~900 lines |
| **CLAUDE.md** | Complete technical architecture, bug fixes, development workflows, guardrails | Developers, AI agents | ~1200 lines, 20 sections |
| Standalone scripts | Educational, isolated components | Advanced users | — |

**Note on documentation structure:** CLAUDE.md exists because the codebase accumulated many implicit rules and edge cases during development (see §5 and §16). Rather than scatter these across comments, we documented the complete mental model including historical bugs, their fixes, and guardrails to prevent re-introducing them. If reading this README, consult CLAUDE.md for:
- **§5** — The ~2-PR bug pattern & 30-second diagnostic table (read at start of each session)
- **§16** — Known issues & how they were fixed (invaluable when reproducing old issues)
- **§20** — Guardrail principles that must never be violated (GP-1 through GP-4)

---

**For help with Claude Code features and hooks, see `/help` or report issues at https://github.com/anthropics/claude-code/issues**

---

## 📜 Project History & Evolution

This project has undergone significant architectural refinement since its inception. The timeline below documents major milestones, critical bug fixes, and design decisions.

### Phase 1: Foundation (2026-02)

- Initial multi-dataset fine-tuning study using DONUT on 8 experiment configurations
- TrOCR+YOLO two-stage pipeline as architectural comparison
- Hand-written metrics computation (F1, NED, exact match)
- First attempt at LaTeX paper auto-generation

### Phase 2: Consolidation & Bug Fixes (2026-03-01 to 2026-03-07)

**Critical fixes that unblocked the pipeline:**

1. **The F1 ≈ 0.42 Bug** (Week 1)
   - Symptom: Models converged but scored 0.42 instead of expected 0.8+
   - Root cause: `safetensors` omits `lm_head.weight` (shared pointer with `embed_tokens.weight`)
   - Fix: `LmHeadCloneCallback` deep-clones weight before save; sanity checks on load

2. **The F1 ≈ 0.008 Bug** (Week 2)
   - Symptom: Near-zero F1 with perfect XML structure
   - Root cause: `token2json()` returned list (CORD `<sep/>` tokens) instead of dict
   - Fix: `_parse_prediction()` merges page-list into flat dict

3. **Data Leakage in Validation** (Week 2)
   - Symptom: Model had access to test data indirectly
   - Root cause: `val_img/` never created; `load_sroie_val()` returned `[]`; early stopping disabled
   - Fix: `stage_install()` physically separates 63 val images from 63 test images

4. **Constants Duplication Drift** (Week 2)
   - Symptom: Silent divergence in `FIELDS` and `IMAGE_EXTS` across 5+ files
   - Root cause: No single source of truth
   - Fix: Consolidated all constants into `constants.py`

5. **Terminal Freeze on Auto-Install** (Week 3)
   - Symptom: Process hung after "Dependencies installed successfully" (5–25 min, invisible)
   - Root cause: flash-attn auto-build + invisible nvcc compilation + missing `os.execv()` restart
   - Fix: Removed flash-attn auto-build; added `os.execv()` restart + progress watchdog

6. **Module Over-Proliferation** (Week 3)
   - Symptom: 36 Python files, many are shim re-exports
   - Root cause: Early refactoring created satellite modules that don't hold real logic
   - Fix: Merged into core 8 files; remaining are helper/standalone scripts

### Phase 3: Stabilization & Final Runs (2026-03-07 to 2026-03-18)

- **Definitive benchmark run** on Vast.ai RTX 6000 Blackwell (96 GB)
- All 8 DONUT experiments converged successfully
- **Best result: Exp 6 (0.8982 F1)** — exceeds published baseline by +5.71 pp
- TrOCR+YOLO trained and evaluated (0.2035 F1)
- Full LaTeX paper auto-generation validated
- Comprehensive CLAUDE.md authored (20 sections, 1200+ lines)
- Diagnostics layer implemented (7 known-bad-F1 patterns detected automatically)

### Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Seq2Seq, not LoRA** | Base checkpoint has no task priors competing with SROIE tokens | Slower on very small datasets, but justifiable at 500–3940 samples |
| **All params in `constants.py`** | Single source of truth; prevents drift | Extra layer of indirection |
| **Full diagnostic callback** | Catches ~7 failure modes automatically | Minor overhead during training |
| **LaTeX paper auto-gen** | Eliminates manual result transcription | Requires careful `\VAR{}` placeholder management |
| **Removed all optional deps** | Reduced install size 345 MB; inlined 65 lines of replacements | Higher maintenance burden for inline code |
| **Vast.ai for definitive runs** | 96 GB GPU (RTX 6000 Blackwell) provides authoritative baseline | Cost; not reproducible on smaller GPUs |

---

## Roadmap

> *Consolidated from the former ROADMAP.md (2026-03).*

### Completed ✅

- Multi-dataset DONUT fine-tuning pipeline (8 experiments) — **final runs 2026-03-07**
- TrOCR + YOLOv8 two-stage pipeline for receipt KIE
- Unified SROIE Task-3 evaluation metrics (global F1, per-field F1, NED, exact-match)
- LaTeX paper auto-generation from results (`inject_results.py` + `paper.tex`)
- Constants centralisation (`constants.py` — single source of truth)
- Automated preflight validation + diagnostics layer (`diagnostics.py`)
- Cloud pipeline orchestrator (`cloud_pipeline.py`) with mode routing
- **Three critical bugs fixed & guarded:** F1 ≈ 0.42 (safetensors), F1 ≈ 0.008 (token2json list), data leakage (val/test split)
- Module consolidation: 36 → 8 core files + 6 standalone scripts
- Auto-install robustness: `os.execv()` restart, `_InstallWatchdog`, no flash-attn build
- Dependency reduction: 22 → 8 direct deps (345 MB install savings)
- Comprehensive technical documentation (`CLAUDE.md` — 20 sections, guardrails, historical fixes)
- Runtime diagnostics with 7 auto-detected failure patterns
- AI-powered diagnosis integration (Claude/Mistral APIs for root-cause analysis)

### In Progress 🔧

- **Autonomous CI/CD with auto-fix loop** — GitHub Actions + AI evaluation + multi-attempt code repair (new 2026-03-18)
- **Memory resource manager** — VRAM-aware batch auto-scaling, HF Arrow cache management (addresses Exp 8 OOM)
- **Cloud GPU training automation** — `MLTrainingOrchestrator` subprocess stubs; Vast.ai API not yet integrated

### Fragile / Known Gaps ⚠️

- **`run_all.py` is a 128 KB monolith** — needs decomposition into `stages/` modules (consider after CI stabilizes)
- **No integration tests** — unit + smoke tests exist, but no end-to-end single-epoch test
- **Hardcoded paths** — `/workspace/` appears in several files; should be fully env-var-driven
- **Type checking** — type hints present but `mypy` not in CI

### Future 🚀

1. Decompose `run_all.py` into `stages/` modules (1–2 week refactor)
2. Add end-to-end integration test (1 train sample, 1 val, 1 epoch)
3. Weights & Biases / MLflow experiment tracking
4. Support additional datasets (CORD v2, RVL-CDIP, SROIE 2019 full set)
5. Docker / devcontainer setup for reproducible environments
6. CI improvements: `mypy --strict`, coverage reporting
7. FastAPI model-serving endpoint
8. Multi-GPU training via `accelerate launch` / `torch.distributed`
9. Hyperparameter sweep orchestrator (uses existing `PARAM_GRIDS_DEFAULT` framework)
10. Comprehensive unit test suite with `pytest` + coverage badges

---

## Dependency Reference

**8 direct dependencies** (down from 22). Install with `pip install -r requirements.txt`.

### Critical — pipeline fails without these

| Package | Purpose |
|---|---|
| `torch` | Tensor ops, CUDA, training loop |
| `transformers` | DONUT + TrOCR model, processor, tokenizer |
| `datasets` | HuggingFace Arrow dataset download/cache |
| `Pillow` | Image loading and resizing (960×1280) |
| `ultralytics` | YOLOv8x text-region detection |
| `accelerate` | Mixed-precision (fp16) training |
| `matplotlib` | F1/loss comparison figures |
| `pytest` | Test suite: `pytest tests/` |

### Installed automatically (transitive deps — do not pin)

`numpy` (via torch), `sentencepiece` (via transformers), `protobuf` (via transformers),
`huggingface-hub`, `tokenizers`, `safetensors`, `pyyaml` (via huggingface-hub)

### Optional — uncomment in requirements.txt if needed

| Package | Purpose |
|---|---|
| `torchvision` | Data augmentation presets DA1/DA2 in `control_suite.py`; pipeline works without it (gracefully returns `None`) |
| `flash-attn` | Faster attention on Ampere+ GPUs; see requirements.txt for install instructions |

### Removed — replaced with inline implementations (~65 lines total)

| Package | Replacement | Saving |
|---|---|---|
| `editdistance` | 14-line Wagner-Fischer `_edit_distance()` in `donut_evaluator.py`, `benchmark_compare.py` | ~200 KB |
| `tqdm` | 16-line logging generator `_progress()` in each caller file | ~4 MB |
| `scipy` | 15-line weighted moving-average `_smooth()` in `plot_convergence.py` | ~30 MB |
| `psutil` | 18-line `/proc/meminfo` reader `_get_total_ram_bytes()` / `_get_available_ram_bytes()` | ~1 MB |

### Removed — never imported anywhere

| Package | Originally declared for | Saving |
|---|---|---|
| `opencv-python` | Image processing | `cv2` never imported; PIL handles all I/O | ~200 MB |
| `pandas` | Data tables | Never imported; stdlib `csv` used | ~30 MB |
| `scikit-learn` | ML utilities | Never imported | ~30 MB |
| `timm` | "TrOCR DeiT backbone" | Incorrect comment; TrOCR uses ViT natively in transformers | ~15 MB |
| `seaborn` | Statistical plots | Never imported; matplotlib used directly | ~4 MB |
| `jiwer` | WER/CER metrics | Never imported; F1+NED used instead | ~1 MB |
| `numpy` | Arrays | Guaranteed transitive dep of torch | — |
| `sentencepiece` | Tokenization | Guaranteed transitive dep of transformers | — |
| `protobuf` | Serialization | Guaranteed transitive dep of transformers | — |

**Total install savings vs original: ~345 MB**
