# Multi-Dataset Fine-Tuning of DONUT for Receipt Information Extraction

A systematic study of multi-dataset fine-tuning for receipt key information
extraction (KIE) using the DONUT (Document Understanding Transformer) model.

## Overview

Starting from a CORD-pretrained DONUT checkpoint, this pipeline trains on
the SROIE benchmark combined with three auxiliary datasets (WildReceipt,
CORU, CORD) across 8 experiment configurations, evaluates each model
on the SROIE test set, and generates a complete LaTeX paper with real results.

## Repository Structure (7-file pipeline)

```
.
├── dataset_loaders.py   # Download & normalize WildReceipt, CORU, CORD
├── train.py             # Standalone SROIE-only fine-tuning script
├── evaluate.py          # Inference + SROIE Task-3 metric computation
├── run_experiments.py   # 8-experiment orchestrator (train + eval)
├── run_all.py           # Single entry point: download → train → eval → paper
├── inject_results.py    # Generate LaTeX tables; fill paper_filled.tex
├── paper.tex            # LaTeX paper template with \VAR{} placeholders
├── requirements.txt     # Python dependencies
├── results/             # Per-experiment JSON results (created at runtime)
└── legacy/              # Archived files from previous iterations
```

## Quick Start (vast.ai / Jupyter terminal)

```bash
# Install dependencies and run the full pipeline — SROIE data is auto-downloaded
pip install -r requirements.txt && python run_all.py

# Run a single experiment
python run_all.py --experiment 2

# Force re-run (ignore cached results)
python run_all.py --force

# Generate paper only (results must already exist)
python run_all.py --paper-only

# Skip SROIE auto-install (data already present)
python run_all.py --skip-install

# Skip pretrained baseline evaluation
python run_all.py --skip-pretrained
```

## Experiment Definitions

| Exp | Training Data                    | Description                              |
|-----|----------------------------------|------------------------------------------|
| 1   | SROIE only                      | Baseline — 626 SROIE train images        |
| 2   | SROIE + WildReceipt             | +~1 740 WildReceipt images               |
| 3   | SROIE + CORU                    | +CORU multilingual receipts (~20k)       |
| 4   | SROIE + CORD                    | +~900 CORD receipt images                |
| 5   | SROIE + WildReceipt + CORD      | Combined receipt datasets                |
| 6   | SROIE + CORU + CORD             | CORU + CORD                              |
| 7   | SROIE + WildReceipt + CORU      | WildReceipt + CORU                       |
| 8   | SROIE + All                     | All four datasets combined               |

## Dataset Sources

- **SROIE**: **Auto-downloaded** from `https://github.com/zzzDavid/ICDAR-2019-SROIE.git`
  (shallow-cloned into the parent of `--sroie-dir`; uses the official 626/347 train/test split).
  Use `--skip-install` if data is already present.
- **WildReceipt**: Auto-downloaded from
  `https://download.openmmlab.com/mmocr/data/wildreceipt.tar`
- **CORU**: Auto-downloaded from HuggingFace
  (`abdoelsayed/CORU`, `Information_Extraction` subset)
- **CORD**: Auto-downloaded from HuggingFace
  (`naver-clova-ix/cord-v2`)
- **Invoices-DONUT**: Available but not in active experiments — Auto-downloaded from HuggingFace
  (`katanaml-org/invoices-donut-data-v1`)

## CLI Reference

### `run_all.py`

```
python run_all.py [options]

Options:
  --experiment N      Run only experiment N (1–8)
  --force             Delete cached results and re-run from scratch
  --paper-only        Skip training; generate paper from existing results
  --skip-install      Skip Stage 0 SROIE auto-install (data already present)
  --skip-download     Skip dataset download/verification stage
  --skip-pretrained   Skip pretrained baseline evaluation step
  --sroie-dir PATH    Path to SROIE data (default: /workspace/ICDAR-2019-SROIE/data)
  --workspace PATH    Workspace root for model checkpoints (default: /workspace)
  --paper-template F  LaTeX template to fill (default: paper.tex)
  --output F          Output filled LaTeX file (default: paper_filled.tex)
```

### `run_experiments.py`

```
python run_experiments.py --all [--force]
python run_experiments.py --experiment N [--force]
```

### `inject_results.py`

```
python inject_results.py --all [--results PATH] [--paper paper.tex] [--output paper_filled.tex]
```

## Evaluation Metric

SROIE Task 3 global F1 over all (image, field) pairs, where a pair is TP
if the predicted string equals the ground truth string (case-insensitive,
stripped). NED (Normalized Edit Distance) is also reported per field; lower
is better (↓).

## Results Format

Each experiment saves `results/experiment_N.json`:

```json
{
  "experiment_id": 1,
  "name": "SROIE only (baseline)",
  "datasets": ["sroie"],
  "num_train_samples": 626,
  "metrics": {
    "global_f1": 0.9123,
    "global_precision": 0.9200,
    "global_recall": 0.9048,
    "overall_exact_match": 0.8500,
    "company_f1": 0.9500,
    "company_ned": 0.0312,
    ...
  }
}
```
