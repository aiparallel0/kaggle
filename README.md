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

Outputs:
- `results.tex` — LaTeX document with loss plots and metrics
- `results_plots/` — PNG figures
- `terminal.txt` — Complete execution log
- `results/experiment_1.json` — DONUT Exp 1 metrics

### Single Experiment

```bash
python run_all.py --experiment 2
```

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
├── 01_dataset_preparation.py  # Data prep with validation CLI
├── 02_train_donut.py          # DONUT training with sweep support
├── 03_train_trocr_yolo.py     # TrOCR+YOLO pipeline
├── 04_evaluate.py             # Unified evaluation + HTML reporting
├── 05_compare_results.py      # Comparison with CSV export & filtering
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

**For help with Claude Code features and hooks, see `/help` or report issues at https://github.com/anthropics/claude-code/issues**
