# Receipt OCR Comparison: DONUT vs TrOCR + YOLO

## 🚀 Quick Start (Recommended)

For the complete, automated pipeline with all 8 experiments:

```bash
python run_all.py                 # Full pipeline
python run_all.py --skip-trocr    # DONUT only
python run_all.py --experiment 1  # Single experiment
python run_all.py --paper-only    # Generate paper from existing results
```

See `CLAUDE.md` Section 10 for all `run_all.py` command-line options.

---

## 📚 Standalone Scripts (Alternative Workflows)

These scripts provide isolated entry points for development and debugging:

### 01_dataset_preparation.py — Dataset & Annotation Prep
Prepares YOLO bounding box labels and TrOCR line crops from SROIE images.

```bash
python 01_dataset_preparation.py                # Prepare all data
python 01_dataset_preparation.py --validate     # Validate existing data
python 01_dataset_preparation.py --force        # Re-prepare (clear existing)
```

**Outputs:** `data/yolo/{train,val,test}/` and `data/trocr/{train,val,test}/`

### 02_train_donut.py — DONUT Reference Implementation
Standalone DONUT fine-tuning (useful for hyperparameter exploration).

```bash
python 02_train_donut.py                        # Train DONUT model
python 02_train_donut.py --dry-run              # Validate setup
python 02_train_donut.py --sweep                # Generate hyperparameter configs
python 02_train_donut.py --config N             # Train specific config
```

**Outputs:** `models/donut_finetuned/best/` and `models/donut_finetuned/training_history.json`

### 03_train_trocr_yolo.py — TrOCR + YOLO Pipeline
Two-stage OCR pipeline: YOLOv8 detection + TrOCR reading + heuristic assignment.

```bash
python 03_train_trocr_yolo.py                   # Train TrOCR+YOLO
```

**Outputs:** `models/yolo_finetuned/run/weights/best.pt` and `models/trocr_finetuned/best/`

### 04_evaluate.py — Unified Model Evaluation
Evaluate both architectures on the same 63 SROIE test images with standardized metrics.

```bash
python 04_evaluate.py                           # Evaluate both architectures
python 04_evaluate.py --donut-only              # DONUT only
python 04_evaluate.py --trocr-only              # TrOCR+YOLO only
python 04_evaluate.py --report                  # Generate HTML report
```

**Outputs:**
- `results/metrics.json` — Raw metrics
- `results/evaluation_summary.json` — Structured summary
- `results/evaluation_report.html` — Interactive HTML report

### 05_compare_results.py — Cross-Architecture Comparison
Generate comparison plots and analysis tables across all experiments.

```bash
python 05_compare_results.py                    # Full comparison (all plots)
python 05_compare_results.py --export-csv       # Export to CSV
python 05_compare_results.py --filter 0.85      # Show experiments with F1 >= 0.85
python 05_compare_results.py --ned-plot         # Generate NED comparison plot
```

**Outputs:**
- `results/plot_f1_comparison.png` — F1 bar chart
- `results/plot_field_f1.png` — Per-field F1
- `results/plot_convergence.png` — Training curves
- `results/plot_ned_comparison.png` — NED analysis
- `results/comparison_results.csv` — Export table

---

## 📋 Project Structure

```
kaggle/
├── ⭐ run_all.py               CANONICAL ENTRY POINT (recommended)
├── CLAUDE.md                  Authoritative AI guide & project rules
├── README.md                  Project overview
│
├── [Core Pipeline]
├── constants.py               Shared constants (SINGLE SOURCE OF TRUTH)
├── dataset_loaders.py         ABC-based dataset loaders
├── train.py                   DonutTrainer class
├── donut_evaluator.py         Evaluation & metrics computation
├── inject_results.py          LaTeX paper generation
├── benchmark_compare.py       Architecture comparison
│
├── [Standalone/Educational Scripts]
├── 01_dataset_preparation.py  ← Data prep with validation CLI
├── 02_train_donut.py          ← DONUT training with sweep support
├── 03_train_trocr_yolo.py     ← TrOCR+YOLO pipeline
├── 04_evaluate.py             ← Unified evaluation + HTML reporting
├── 05_compare_results.py      ← Comparison with CSV export & filtering
│
├── [Configuration & Templates]
├── paper.tex                  LaTeX research paper template
├── references.bib             Bibliography
├── training_config.py         Hyperparameter grids for quick mode
├── requirements.txt           Python dependencies
│
├── [Runtime Artifacts (gitignored)]
├── data/                      Cached datasets
├── models/                    Checkpoints & fine-tuned weights
├── results/                   Experiment JSON outputs & plots
└── paper_filled.tex           Generated paper (auto-produced)
```

---

## 🎯 When to Use Each Script

| Goal | Command |
|------|---------|
| Full 8-experiment pipeline | `python run_all.py` |
| Quick test (single exp) | `python run_all.py --experiment 1` |
| Prepare data only | `python 01_dataset_preparation.py` |
| Validate data | `python 01_dataset_preparation.py --validate` |
| Test DONUT setup | `python 02_train_donut.py --dry-run` |
| Explore hyperparameters | `python 02_train_donut.py --sweep` |
| Evaluate trained models | `python 04_evaluate.py` |
| Get HTML comparison report | `python 04_evaluate.py --report` |
| Export results to CSV | `python 05_compare_results.py --export-csv` |
| Find best-performing exps | `python 05_compare_results.py --filter 0.85` |

---

## 📖 Full Documentation

See **`CLAUDE.md`** for:
- Complete pipeline architecture (Section 2)
- Hyperparameter optimization (Section 3)
- Training time estimates (Section 4)
- Known bugs & fixes (Section 16)
- Full end-to-end flow diagram (Section 18)
- Development workflows (Section 10)
