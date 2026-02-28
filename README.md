# Multi-Dataset Fine-Tuning of DONUT for Receipt Information Extraction

A systematic study of multi-dataset fine-tuning for receipt key information
extraction (KIE) using the DONUT (Document Understanding Transformer) model.

## Overview

Starting from a CORD-pretrained DONUT checkpoint, this pipeline trains on
the SROIE benchmark combined with three auxiliary datasets (WildReceipt,
CORD, Invoices-DONUT) across 8 experiment configurations, evaluates each model
on the SROIE test set, and generates a complete LaTeX paper with real results.

## Pipeline Architecture

```
Raw sources                 Normalization              Unified schema
────────────               ─────────────              ──────────────
SROIE (626 imgs)     ──►  SROIELoader                {"company": "...",
WildReceipt (tar)    ──►  WildReceiptLoader     ──►   "date":    "...",
CORD (HuggingFace)   ──►  CORDLoader                  "address": "...",
Invoices (HF)        ──►  InvoicesDonutLoader          "total":   "..."}
                                                             │
              80/10/10 split                                 │ SROIEDataset
          ┌─────┴──────┬──────┐                             │ __getitem__
       img/         val_img/ test_img/                       ▼
     (500 train)   (63 val) (63 test)          <s_sroie><s_company>…</s_company>
          │                    │               <s_date>…</s_date>
          │                    │               <s_address>…</s_address>
          ▼                    │               <s_total>…</s_total></s_sroie>
   DonutTrainer                │                             │
   (Seq2SeqTrainer)            │                             │ labels tensor
   early-stop on val_img/      │                             │
   LmHeadCloneCallback ────────┼──────────── checkpoint ─────┘
   (prevents lm_head loss)     │                │
          │                    │      load_model_with_tied_weights()
          │                    │      RuntimeError if lm_head missing
          ▼                    ▼                │
   best checkpoint      test_img/ (63)          ▼
          │                    │        DonutEvaluator._parse_prediction()
          └────────────────────┘        token2json() → list? → merge pages
                                        _unwrap_prediction() → {"company":…}
                                                 │
                                        compute_metrics()
                                        global F1, NED per field
                                                 │
                                        results/experiment_N.json
                                                 │
                                        PaperInjector → paper_filled.tex
```

**Key invariant:** `val_img/` and `test_img/` are populated once by
`stage_install()` and are never mixed. Training early stopping reads only
`val_img/`; final scoring reads only `test_img/`. Auxiliary datasets
(WildReceipt, CORD, Invoices) contribute to training only — their held-out
15% test portion is discarded to prevent contamination.

## Known Bugs Fixed

Three silent bugs were causing catastrophically wrong F1 scores. All are fixed
and guarded with tests in `tests/`.

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| **F1 ≈ 0.42** (plausible-looking) | `safetensors` omits `lm_head.weight` from checkpoint shards because it shares a data pointer with `embed_tokens.weight` after `resize_token_embeddings()`. On reload, `lm_head` is randomly re-initialized. | `LmHeadCloneCallback` deep-clones the weight before every save. `load_model_with_tied_weights()` raises `RuntimeError` immediately if `lm_head.weight` is still missing. |
| **F1 ≈ 0.008** (near-zero, not zero) | `token2json()` returns a **list** of page-dicts when the generated sequence contains `<sep/>` tokens (inherited from CORD pretraining). `_parse_prediction()` was treating any non-dict as a parse failure and returning `{}`. | `_parse_prediction()` and `_self_test()` now merge the list of pages into a single flat dict (first occurrence of each key wins). |
| **F1 unreliable / overfitted** | `val_img/` directory not created; `load_sroie_val()` returned `[]`; `do_eval=False`; no early stopping; model could be evaluated on test data indirectly. | `stage_install()` explicitly moves 63 images into `val_img/` and 63 into `test_img/` — physically distinct directories checked by tests. |

## Repository Structure (7-file pipeline)

```
.
├── dataset_loaders.py   # Download & normalize WildReceipt, CORD, Invoices-DONUT
├── train.py             # Standalone SROIE-only fine-tuning script
├── evaluate.py          # Inference + SROIE Task-3 metric computation
├── run_experiments.py   # 8-experiment orchestrator (train + eval)
├── run_all.py           # Single entry point: download → train → eval → paper
├── inject_results.py    # Generate LaTeX tables; fill paper_filled.tex
├── paper.tex            # LaTeX paper template with \VAR{} placeholders
├── requirements.txt     # Python dependencies
├── hf_token.txt         # HuggingFace token placeholder (gitignored)
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

## Performance Tips

- **HF_TOKEN**: Create `hf_token.txt` with your HuggingFace token for 5-10x faster downloads.
  Get one at https://huggingface.co/settings/tokens
- **Parallel downloads**: All auxiliary datasets are downloaded in parallel automatically.
- **RAM cache**: Images are pre-loaded into RAM when sufficient memory is available.
- **DataLoader**: Optimized with pin_memory, prefetch_factor=4, and persistent workers.

## Experiment Definitions

| Exp | Training Data                       | Description                              |
|-----|-------------------------------------|------------------------------------------|
| 1   | SROIE only                         | Baseline — 500 SROIE train images        |
| 2   | SROIE + WildReceipt                | +~1 740 WildReceipt images               |
| 3   | SROIE + Invoices-DONUT             | +~800 invoice images                     |
| 4   | SROIE + CORD                       | +~900 CORD receipt images                |
| 5   | SROIE + WildReceipt + CORD         | Combined receipt datasets                |
| 6   | SROIE + WildReceipt + Invoices     | WildReceipt + Invoices-DONUT             |
| 7   | SROIE + CORD + Invoices            | CORD + Invoices-DONUT                    |
| 8   | SROIE + All                        | All four datasets combined               |

## Dataset Sources

- **SROIE**: **Auto-downloaded** from `https://github.com/zzzDavid/ICDAR-2019-SROIE.git`
  (shallow-cloned; uses 80/10/10 split: 500 train / 63 val / 63 test).
  The official 347-image test split has no public ground truth labels.
  Use `--skip-install` if data is already present.
- **WildReceipt**: Auto-downloaded from
  `https://download.openmmlab.com/mmocr/data/wildreceipt.tar`
- **CORD**: Auto-downloaded from HuggingFace
  (`naver-clova-ix/cord-v2`)
- **Invoices-DONUT**: Auto-downloaded from HuggingFace
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
  "num_train_samples": 500,
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
