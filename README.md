# Multi-Dataset Fine-Tuning of DONUT for Receipt Information Extraction
# Be careful to preserve debelopmental history of the repo.
A systematic study of multi-dataset fine-tuning for receipt key information extraction (KIE) using the DONUT (Document Understanding Transformer) model and TrOCR+YOLO comparison.

**Status:** ✅ **Stable & Complete** (2026-04-02)
- **18-experiment comprehensive suite**: 8 DONUT multi-dataset core experiments + 10 extended experiments (precision grid, resolution grid, TrOCR+YOLO, zero-shot baselines)
- Five critical bugs identified & permanently fixed (see Known Issues)
- Comprehensive automated diagnostics & safety checks in place
- Full documentation & reproducible results
- **Note (2026-04-02):** `transformers 5.5.0` is now the default install (`pip install transformers`). The codebase is compatible with transformers 5.x — the `PreTrainedTokenizerBase` compat shim in `data_pipeline.py` handles the 4.47+ relocation. Pre-install via `pip install -r requirements.txt` before running; the auto-installer requires `transformers` to be present.

**Best result:** DONUT Experiment 6 (SROIE + Invoices-DONUT, 2× SROIE oversampling) achieves **global F1 = 0.8982** on the SROIE Task-3 test set — a gain of +5.71 percentage points over the published DONUT baseline (0.8411). Training: 39.6 minutes on Vast.ai RTX 6000 Blackwell (96 GB), 2026-03-07. TrOCR+YOLO (inline fallback, no `ultralytics`) achieves F1 = 0.2035 (−69.5% vs. best DONUT).

**Pipeline:** Starting from `naver-clova-ix/donut-base`, trains on SROIE combined with two auxiliary datasets (WildReceipt, Invoices-DONUT) across 18 experiment configurations, evaluates each model on the 63-image SROIE test split, and generates a complete LaTeX research paper with all real metrics.

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
5. Trains TrOCR+YOLO pipeline (single training run; identical metrics reported across all experiment slots)
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

These scripts provide isolated entry points for development and debugging. The following scripts are the actual files present in the repository.

### train_trocr_yolo.py — TrOCR + YOLO Pipeline

Two-stage OCR pipeline: YOLOv8 detection + TrOCR reading + heuristic assignment.

```bash
python train_trocr_yolo.py                   # Train TrOCR+YOLO
```

**Outputs:** `models/yolo_finetuned/run/weights/best.pt` and `models/trocr_finetuned/best/`

> **Note:** `dataset_preparation.py`, `train_donut.py`, and `evaluate_models.py` listed in older documentation no longer exist as standalone files. Their functionality is integrated into `run_all.py` and `run_experiments.py`.

---

## 📋 Command Reference

### `run_all.py` — Main Pipeline

```bash
python run_all.py [options]

Options:
  --experiment N      Run only experiment N (1–18)
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
python run_experiments.py --all              # all experiments (1–18)
python run_experiments.py --experiment 3     # single experiment
python run_experiments.py --all --force      # force re-run
```

### Paper Generation (`reporting.py` PaperInjector)

```bash
python reporting.py --all --paper paper/paper.tex --output paper/paper_filled.tex
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
- The auto-installer checks for: `torch`, `transformers` only (NOT `datasets`, `accelerate`, `editdistance` or `pandas` — all have inline replacements and are not in the critical install list)
- flash-attn is **not** auto-installed (see Troubleshooting); PyTorch 2.x SDPA is used instead
- Dual-stream logging: all output to `terminal.txt`, console shows filtered progress
- Professional CLI interface with argparse-based options

---

## 📊 Experiment Definitions & Actual Results

18-experiment comprehensive suite. **Core DONUT experiments (1–8)** use the 80/10/10 SROIE split: **500 train / 63 val / 63 test**. **Extended series (9–18)** cover precision grid, high-resolution ablation, zero-shot baselines, and TrOCR+YOLO architecture comparison. Trained on Vast.ai RTX 6000 Blackwell (96 GB), 2026-03-07.

### Core Multi-Dataset DONUT Experiments (1–8)

| Exp | Training Data | Samples | Global F1 | Status | Notes |
|---|---|---|---|---|---|
| 1 | SROIE only (baseline) | 500 | **0.8503** | ✅ Pass | 10 epochs, fp16, ~32 min |
| 2 | SROIE + WildReceipt | 1,386 | 0.8257 | ✅ Pass | No oversample — WR dilutes SROIE signal |
| 3 | SROIE + Invoices-DONUT | 832 | 0.2867 | ✅ Pass¹ | No oversample — severe cross-domain collapse |
| 4 | SROIE + WR + Invoices | 1,718 | 0.8224 | ✅ Pass | Unbalanced all-in — still below baseline |
| 5 | SROIE + WR (2× SROIE) | 1,886 | 0.8514 | ✅ Pass | Oversampling helps marginally (+0.0011) |
| **6** | **SROIE + Invoices (2× SROIE)** | **1,332** | **0.8982** | ✅ **BEST** | Early stop ep.8; 39.6 min training |
| 7 | SROIE + All (2× SROIE) | 2,218 | 0.8503 | ✅ Pass | All datasets 2×; competing signals cancel |
| 8 | SROIE + All (3× SROIE) | ~2,718 | OOM | ⚠️ **OOM** | Previous run failed at wrong 2560×1920 resolution. At correct 1280×960: ~110 min, ~69 GB RAM required. |

> ¹ Exp 3 "passes" in the sense it trains to completion, but the resulting model is near-unusable (F1=0.2867). This is the expected control result, not a bug.

> **Key finding:** SROIE oversampling (2×) is a **prerequisite** for auxiliary data to help. Without it (Exps 2–4), adding more data hurts. The best config (Exp 6) gains **+0.0479** over Exp 1 and **+0.0571** over the published DONUT result.

**Exp 6 per-field:** company=0.905, date=0.984, address=0.790, total=0.912 · exact_match=0.667

### Extended Architecture Comparison Series (9–18)

| Exp | Name | Architecture | Precision | Resolution | Status | Notes |
|---|---|---|---|---|---|---|
| 9 | DONUT zero-shot | DONUT (no training) | fp16 | 1280×960 | ✅ Pass | Skip=true; inference only, expected F1≈0.10–0.30 |
| 10 | DONUT fine-tuned (best recipe) | DONUT | fp16 | 1280×960 | ✅ Pass | Mirrors Exp 6; F1≈0.8982 |
| 11 | DONUT fine-tuned bf16 | DONUT | **bf16** | 1280×960 | ✅ Pass² | bf16 vs fp16 ablation; requires Ampere+ GPU |
| 12 | TrOCR+YOLO (2× SROIE + Invoices) | TrOCR+YOLO | fp16 | 1280×960 | ⚠️ Degraded³ | 279M params; inline YOLO fallback if ultralytics absent |
| 13 | DONUT fine-tuned fp32 | DONUT | **fp32** | 1280×960 | ✅ Pass | Full precision reference; batch=4, grad_accum=4; ~2× slower |
| 14 | DONUT high-res fp16 | DONUT | fp16 | **2560×1920** | ⚠️ **OOM risk** | 4× pixels; batch=2, grad_accum=8; `allow_high_res: true`; processor_config.json must be updated |
| 15 | TrOCR+YOLO (SROIE only) | TrOCR+YOLO | fp16 | 1280×960 | ⚠️ Degraded³ | Minimal pipeline; 500 samples, no oversample |
| 16 | DONUT high-res bf16 | DONUT | bf16 | **2560×1920** | ⚠️ **OOM risk** | High-res + bf16; Ampere+ required; processor_config.json must be updated |
| 17 | DONUT high-res fp32 | DONUT | fp32 | **2560×1920** | ❌ **Expected OOM** | 4× pixels × 2× fp32 memory; batch=1, grad_accum=16; **≥60 GB VRAM required** — RTX 4090 (24 GB) will OOM |
| 18 | DONUT zero-shot high-res | DONUT (no training) | fp16 | **2560×1920** | ✅ Pass | Inference only; resolution without training gains nothing |

> ² bf16 requires Ampere+ (RTX 30xx / A100 / H100). Falls back to fp32 on older hardware.

> ³ TrOCR+YOLO quality is severely limited when `ultralytics` is not installed. The inline YOLO fallback uses a proxy L2 loss (trains the backbone away from random init) but does **not** produce a calibrated detector. Real detection quality requires: `pip install ultralytics`. The inline fallback is a smoke-test stand-in only.

### TrOCR+YOLO vs DONUT

| Metric | DONUT Exp 6 (best) | TrOCR+YOLO Exp 12 | Gap |
|---|---|---|---|
| Global F1 | **0.8982** | 0.2035 | +69.5% absolute |
| company F1 | 0.905 | 0.176 | +72.9% |
| date F1 | 0.984 | 0.460 | +52.4% |
| address F1 | 0.790 | 0.000 | +79.0% |
| total F1 | 0.912 | 0.231 | +68.1% |
| Parameters | ~200M | ~279M (+40%) | DONUT smaller AND better |

> TrOCR+YOLO address F1=0.000 is structural: address spans multiple lines which YOLO crops independently; TrOCR reads each crop in isolation; the heuristic rule-based field assignment cannot reconstruct multi-line addresses. DONUT's end-to-end attention handles multi-line fields natively.

### Summary: Will Any Experiment Fail?

| Category | Experiments | Outcome on RTX 4090 (24 GB) |
|---|---|---|
| Core DONUT (normal resolution) | 1–7, 9–11, 13 | ✅ All expected to complete |
| Exp 8 (3× oversample, 1280×960) | 8 | ⚠️ Completes on ≥64 GB RAM; may OOM on <32 GB system RAM |
| High-res fp16/bf16 | 14, 16 | ⚠️ Possible OOM — batch halved to 2; progressive recovery may succeed |
| **High-res fp32** | **17** | ❌ **Will OOM on RTX 4090** — requires ≥60 GB VRAM |
| TrOCR+YOLO (no ultralytics) | 12, 15 | ⚠️ Runs but gives degraded quality (inline fallback only) |
| Zero-shot / inference-only | 9, 18 | ✅ Always pass (no training, minimal VRAM) |

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
├── run_experiments.py       # 8-experiment DONUT orchestrator + ExperimentConfig + DonutEvaluator
├── CLAUDE.md                # Authoritative AI guide & project rules
├── README.md                # This file
│
├── [Core Pipeline]
├── constants.py             # Shared constants (SINGLE SOURCE OF TRUTH) + _edit_distance() + logging utils
├── data_pipeline.py         # Merged: dataset_loaders + dataset_normalizer + preprocess_seller_split
├── train.py                 # DonutTrainer class + LiveDashboardCallback (inlined)
├── reporting.py             # Merged: benchmark_compare + plot_convergence + inject_results (PaperInjector)
├── resource_manager.py      # Merged: memory_manager + resource_optimizer (VRAM-aware scaling)
├── validation.py            # Merged: validators + preflight_checks + startup_diagnostics
├── diagnostics.py           # DiagnosticCallback + PipelineDiagnostics + ai_diagnose()
├── cloud_orchestration.py   # Merged: pipeline_types + dag_scheduler + cloud_pipeline
├── sweep.py                 # Merged: hparam_search + multi_seed_runner
├── autonomous_ci.py         # Autonomous CI/CD: test runner + AI evaluator + auto-fix loop
├── requirements.txt         # Python dependencies (5 direct: torch, transformers, pyyaml, accelerate, ruff)
├── ruff.toml                # Linter/formatter config
│
├── [TrOCR + YOLO]
├── train_trocr_yolo.py      # TrOCR+YOLO training pipeline
│
├── [Scripts]
├── scripts/
│   └── validate_yaml_experiments.py  # CI YAML experiment validator
│
├── [Experiment Definitions]
├── experiments/             # YAML experiment definition files (exp_01_*.yaml … exp_18_*.yaml)
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
    └── paper/paper_filled.tex  # Generated paper (auto-produced by reporting.py PaperInjector)
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

### `FATAL: The following packages could not be installed: transformers` (on first run)

**Root cause:** Line 42 of `requirements.txt` was missing the `#` comment character, so `pip install -r requirements.txt` tried to install the literal string `ultralytics   → 100% replaced...` and failed — aborting the entire install, including `transformers`.

**Fixed in 2026-04-02 commit.** After `git pull`, run:

```bash
pip install -r requirements.txt   # now installs cleanly
python run_all.py
```

If you hit this on an older checkout, install manually:

```bash
pip install transformers accelerate torch pyyaml ruff
python run_all.py
```

### `transformers 5.x` compatibility

`pip install transformers` now installs **5.5.0** (as of 2026-04). The codebase is compatible:
- `PreTrainedTokenizerBase` compat shim in `data_pipeline.py` handles the ≥4.47 import path change
- `Seq2SeqTrainer` and `VisionEncoderDecoderModel` APIs are stable across 4.x–5.x
- `safetensors` 0.7.0 is required by transformers 5.x and is installed automatically

If you see `AttributeError` on a transformers class after upgrading, check `data_pipeline.py` for the compat shim — it must be present and not deleted.

### YOLO training produces terrible detection (address F1 = 0.000)

If `ultralytics` is **not** installed, the pipeline uses an inline fallback (`_YOLOv8Inline`) that:
- Trains a ResNet-style backbone with L2 regression loss
- Does **not** use anchor boxes, NMS, or IoU thresholds
- Produces bounding boxes that are structurally plausible but poorly calibrated
- Results in near-zero address F1 because multi-line address crops are not reliably found

**Fix:** `pip install ultralytics` for real YOLOv8x detection quality. The pipeline then automatically uses the real YOLO trainer instead of the inline fallback.

### YOLO detects 0 text regions despite training showing mAP > 0

**Root cause:** Train/inference parameter drift. The YOLO model was trained at resolution X (e.g., 320px in superfast mode) but inference used ultralytics' default of 640px. The anchor grid scale mismatch causes all detection confidence scores to drop below the 0.25 threshold, producing zero detections.

**General principle:** Every inference call must pass the same configuration constants used during training. Never rely on library defaults — they may differ from training settings, especially in micro/superfast modes where constants are patched to smaller values.

**Fixed in:** All YOLO inference sites now pass `imgsz=YOLO_IMG_SIZE`; all TrOCR generation sites now use `TROCR_MAX_LEN`; `reporting.py` imports constants from `train_trocr_yolo.py` instead of hardcoding values. See Pattern 8 in `CLAUDE.md` §16.

### "YOLO detected 0 text regions" but YOLO mAP is high (> 0.9)

**Root cause:** Despite the warning message, YOLO is not the culprit. The log message "YOLO detected 0 text regions" is misleading — it fires whenever `ocr_lines` is empty, which can happen for two independent reasons:

1. **YOLO found 0 boxes** (genuine YOLO failure) — fix by checking `imgsz`, model quality, or installing ultralytics
2. **YOLO found boxes but TrOCR decoded every crop to empty text** (TrOCR failure) — YOLO is working, but the TrOCR model is undertrained

If YOLO mAP is high (e.g., mAP50 = 0.935) and the log now says **"YOLO detected N text region(s) but TrOCR decoded all N crop(s) to empty text"**, the fix is TrOCR-specific:

**Fix:** Check that `TROCR_EPOCHS ≥ 5` in speed modes. One epoch of TrOCR fine-tuning produces `val_loss ≈ 9.1` — the decoder is non-functional and outputs garbage for every crop. Five epochs brings `val_loss` to ~2.5–3.0, sufficient for basic text decoding. Also verify `TROCR_MAX_LEN ≥ 64` so ~50-character address and name lines are not truncated.

```bash
# Verify the TROCR_EPOCHS floor in your speed-mode handler:
grep -n "TROCR_EPOCHS" run_all.py
# All speed-mode assignments should be 5, not 1
```

This bug was masked before PR #195 because YOLO was also broken (0 detections). Fixing YOLO exposed the TrOCR failure. See Pattern 9 in `CLAUDE.md` §16 for the general lesson about masked cascading failures in multi-component pipelines.

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
python run_all.py --skip-pretrained --experiment 1  # dry run with just Exp 1
```

---

## 🐛 Known Issues Fixed

Six critical bugs that previously caused catastrophic failures. All are fixed.

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| **F1 ≈ 0.42** (plausible-looking) | `safetensors` omits `lm_head.weight` from checkpoint shards because it shares data pointer with `embed_tokens.weight` after `resize_token_embeddings()`. On reload, `lm_head` randomly re-initialized. | `LmHeadCloneCallback` deep-clones weight before every save. `load_model_with_tied_weights()` raises `RuntimeError` immediately if `lm_head.weight` still missing. |
| **F1 ≈ 0.008** (near-zero, not zero) | `token2json()` returns **list** of page-dicts when generated sequence contains `<sep/>` tokens (inherited from CORD pretraining). `_parse_prediction()` treated any non-dict as parse failure, returning `{}`. | `_parse_prediction()` and `_self_test()` merge list of pages into single flat dict (first occurrence of each key wins). |
| **F1 unreliable / overfitted** | `val_img/` directory not created; `load_sroie_val()` returned `[]`; `do_eval=False`; no early stopping; model evaluated on test data indirectly. | `stage_install()` explicitly moves 63 images into `val_img/` and 63 into `test_img/` — physically distinct directories checked by tests. |
| **Terminal freeze after "Dependencies installed successfully"** | Auto-installer built `flash-attn` from source via `subprocess.run(..., capture_output=True)` — CUDA kernel compilation takes 5–25 min on GPU, invisible to user; `Ctrl+C` didn't reach the nvcc child. After install, `os.execv()` restart was missing, causing `sys.exit(2)` due to `sys.modules` isolation. | Removed flash-attn auto-build entirely. Added `os.execv()` restart after successful pip install. Added `_InstallWatchdog` for elapsed-time progress dots. Added `timeout=300` to main pip subprocess. |
| **`FATAL: transformers could not be installed`** on fresh environment | `requirements.txt` line 42 was missing the `#` prefix on the `ultralytics` comment, so `pip install -r requirements.txt` tried to parse `ultralytics   → 100% replaced...` as a package name and aborted the entire install. | Added `#` to line 42. `pip install -r requirements.txt` now completes cleanly. (Fixed 2026-04-02) |
| **YOLO detects 0 text regions (100%) despite training mAP > 0** | YOLO inference used library default `imgsz=640` instead of the training resolution (`YOLO_IMG_SIZE=320` in superfast, `256` in micro). Anchor grid scale mismatch caused all confidence scores to drop below threshold. Same anti-pattern affected `max_new_tokens` (hardcoded `128` vs patchable `TROCR_MAX_LEN`) and model path defaults in `reporting.py`. | Pass module-level constants (`YOLO_IMG_SIZE`, `TROCR_MAX_LEN`, `YOLO_BASE`, `TROCR_MODEL_ID`) explicitly at every inference call site. Added Pattern 8 to CLAUDE.md §16. |
| **"YOLO detected 0 text regions" (92% of images) despite YOLO mAP50 = 0.935** | PR #195 fixed YOLO parameter drift, unmasking a second independent bug: `TROCR_EPOCHS=1` in speed modes produces `val_loss=9.1268` — the decoder outputs garbage/empty strings for every crop. `ocr_lines` stays empty on 92% of images; `_verify_yolo_detection_rate` crashes with a RuntimeError that incorrectly blames YOLO. The two failure modes (YOLO found 0 boxes vs TrOCR decoded all crops to empty) now produce distinct log messages. | Raised `TROCR_EPOCHS` floor to 5 (brings `val_loss` from ~9.1 to ~2.5–3.0) and `TROCR_MAX_LEN` to 64 in all speed modes. `_extract_ocr_lines()` now returns `yolo_box_count`; warning messages distinguish YOLO vs TrOCR failure. Added Pattern 9 (masked cascading failures) to CLAUDE.md §16. Added runtime `ValueError` guard at the top of `train_trocr()` rejecting `TROCR_EPOCHS < 3`. Superfast mode now achieves **Global F1 = 0.4810** (Date=0.919, Company=0.476, Total=0.355, Address=0.176). |

---

### 💡 Lessons Learned: Masked Cascading Failures

> **Lesson: Masked Cascading Failures.** In multi-component pipelines (YOLO → TrOCR → field assignment), a bug in component A can mask a bug in component B. Fixing A then reveals B with a confusing error message. Always:
>
> 1. **Independently verify each stage's output** after fixing any single component — don't declare the fix done until the full chain (not just the fixed component) produces valid output.
> 2. **Ensure error messages identify the failing component**, not just the failing symptom. A generic "YOLO detected 0 text regions" message that fires for both YOLO failures *and* TrOCR failures wastes hours of debugging time chasing the wrong component.
> 3. **Enforce minimum quality floors** (e.g., `TROCR_EPOCHS ≥ 3` — now enforced via `ValueError` in `train_trocr()`) so that extreme parameter patching in speed modes cannot silently produce a non-functional model.
>
> **Before the fix:** YOLO was broken (imgsz drift) and crashed first, hiding TrOCR's 1-epoch failure. Fixing YOLO (PR #195) revealed TrOCR with the **identical error message** — the same `RuntimeError` blaming YOLO, despite YOLO now having mAP50=0.935. Adding separate `yolo_zero_count` and `trocr_empty_count` counters in `_verify_yolo_detection_rate()` immediately identified TrOCR as the new culprit.
>
> See **CLAUDE.md §16 Pattern 9** and the post-mortem block comment in `run_all.py` near `_superfast_mode_handler` for the full analysis.

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
- LaTeX paper auto-generation from results (`reporting.py` PaperInjector + `paper.tex`)
- Constants centralisation (`constants.py` — single source of truth)
- Automated preflight validation + diagnostics layer (`diagnostics.py`)
- Cloud pipeline orchestrator (`cloud_orchestration.py`) with mode routing
- **Three critical bugs fixed & guarded:** F1 ≈ 0.42 (safetensors), F1 ≈ 0.008 (token2json list), data leakage (val/test split)
- Module consolidation: 36 → 13 core files
- Auto-install robustness: `os.execv()` restart, `_InstallWatchdog`, no flash-attn build
- Dependency reduction: 22 → 5 direct deps (345 MB install savings)
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

**5 direct dependencies** (down from 22). Install with `pip install -r requirements.txt`.

### Critical — pipeline fails without these

| Package | Purpose |
|---|---|
| `torch` | Tensor ops, CUDA, training loop |
| `transformers` | DONUT + TrOCR model, processor, tokenizer |
| `pyyaml` | YAML experiment config loading |
| `accelerate` | Declared dependency; used by transformers `Seq2SeqTrainer` internals |
| `ruff` | Linter/formatter — required to pass CI |

### Optional — enhances pipeline but has inline fallback

| Package | Purpose | Inline fallback |
|---|---|---|
| `datasets` | HuggingFace Arrow dataset download/cache | `_hf_download_dataset_inline()` in `data_pipeline.py` (urllib-only) |
| `Pillow` | Image loading and resizing (960×1280) | `_load_png/_load_bmp/_load_jpeg_pure` in `train.py`; optional but recommended for speed |
| `ultralytics` | YOLOv8x text-region detection | `_YOLO_CLS` proxy in `train_trocr_yolo.py` — trains a placeholder model that cannot reliably detect text regions; install `ultralytics` for real YOLO detection quality |
| `matplotlib` | F1/loss comparison figures | SVG generators in `reporting.py` (`_svg_bar_chart`, `_svg_speed_chart`, etc.) |

### Installed automatically (transitive deps — do not pin)

`numpy` (via torch), `sentencepiece` (via transformers), `protobuf` (via transformers),
`huggingface-hub`, `tokenizers`, `safetensors`

### Optional — uncomment in requirements.txt if needed

| Package | Purpose |
|---|---|
| `torchvision` | Data augmentation presets DA1/DA2; pipeline works without it (gracefully returns `None`) |
| `flash-attn` | Faster attention on Ampere+ GPUs; see requirements.txt for install instructions |

### Removed — replaced with inline implementations (~65 lines total)

| Package | Replacement | Saving |
|---|---|---|
| `editdistance` | 14-line Wagner-Fischer `_edit_distance()` in `constants.py` | ~200 KB |
| `tqdm` | 16-line logging generator `_progress()` in each caller file | ~4 MB |
| `scipy` | 15-line weighted moving-average `_smooth()` in `reporting.py` | ~30 MB |
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

---

## Code Quality & SOLID Analysis

A comprehensive SOLID principles audit was performed on 2026-04-04 across all 14 Python files (34,213 LOC). See [`SOLID_VIOLATIONS.md`](SOLID_VIOLATIONS.md) for the complete inventory.

### Summary

| Category | Status |
|----------|--------|
| Linting (ruff) | ✅ All checks pass |
| Formatting (ruff format) | ✅ All files formatted |
| Import chain integrity | ✅ Verified |
| Broad exception catches narrowed | ✅ 13 instances fixed |
| Duplicate code removed | ✅ 5 instances fixed |
| Silent failures → logged warnings | ✅ 1 instance fixed |
| Remaining SOLID violations | 🔲 Documented in SOLID_VIOLATIONS.md |

### Key Findings

- **22 SRP violations** — god classes/functions that handle too many concerns
- **8 OCP violations** — code requiring modification instead of extension
- **4 LSP violations** — inconsistent subclass/callback contracts
- **9 DIP violations** — concrete dependencies where abstractions are needed
- **35+ functions exceeding 50 lines** (largest: `DonutTrainer.train()` at 617 lines)
- **58+ public functions missing return type hints**

### Priority Refactoring

See `SOLID_VIOLATIONS.md` § 18 for the full roadmap. Top 5:

1. Split `DonutTrainer.train()` (617 lines → 5+ methods)
2. Split `train_experiment()` (402 lines) and `run_experiment()` (378 lines)
3. Extract god classes: `TrOCRReceiptDataset`, `TrOCRYOLOPipeline`, `PaperInjector`
4. Template method for dataset loader boilerplate (4 classes × 3 methods)
5. Strategy pattern for task prompt parsers (5+ hardcoded locations)
