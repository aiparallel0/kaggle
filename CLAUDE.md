# CLAUDE.md — AI Assistant Guide for DONUT SROIE Multi-Dataset Pipeline

## Project Overview

This repository implements a systematic study of **multi-dataset fine-tuning for receipt Key Information Extraction (KIE)** using two model architectures:

1. **DONUT** (Document Understanding Transformer) — end-to-end vision-language model, `naver-clova-ix/donut-base-finetuned-cord-v2` as base checkpoint
2. **TrOCR + YOLOv8** — two-stage OCR pipeline: YOLOv8n detects text regions, TrOCR reads crops, heuristics assign fields

The goal is to evaluate how adding auxiliary training datasets (WildReceipt, CORD, Invoices-DONUT) to SROIE fine-tuning affects performance on the SROIE Task-3 benchmark, across 8 dataset-combination experiments. Results are automatically compiled into a LaTeX research paper.

---

## Repository Structure

```
kaggle/
├── constants.py              # SINGLE SOURCE OF TRUTH for all shared constants
├── dataset_loaders.py        # ABC-based download & normalization for all datasets
├── train.py                  # DonutTrainer OOP wrapper (legacy standalone)
├── evaluate.py               # DonutEvaluator OOP wrapper (legacy standalone)
├── run_experiments.py        # 8-experiment DONUT orchestrator (ExperimentConfig)
├── run_all.py                # MAIN ENTRY POINT: full dual-architecture pipeline
├── inject_results.py         # PaperInjector: generates LaTeX from results JSON
├── paper.tex                 # LaTeX template with \VAR{} placeholders
├── references.bib            # BibTeX references for paper
├── requirements.txt          # Python dependencies with version pins
├── constants.py              # Shared constants (import from here, never duplicate)
├── hf_token.txt              # HuggingFace token (gitignored — never commit)
│
├── 00_project_structure.md   # Alternative numbered-script project overview
├── 01_dataset_preparation.py # Alternative: dataset prep script
├── 02_train_donut.py         # Alternative: DONUT training script
├── 03_train_trocr_yolo.py    # TrOCR+YOLO training (also called by run_all.py)
├── 04_evaluate.py            # Alternative: unified evaluation
├── 05_compare_results.py     # Alternative: visualization & comparison
│
├── results/                  # Runtime: per-experiment JSON (gitignored)
├── data/                     # Runtime: dataset cache (gitignored)
├── models/                   # Runtime: model checkpoints (gitignored)
└── paper_filled.tex          # Runtime: generated paper output (gitignored)
```

**Note on numbered scripts (`01_`–`05_`):** These are an alternative, simpler workflow for standalone use. The canonical production pipeline uses `run_all.py` as the single entry point.

---

## Technology Stack

| Category | Libraries / Tools |
|---|---|
| Deep Learning | PyTorch ≥2.0, Transformers ≥4.35, Accelerate ≥0.24 |
| Vision-Language | DONUT (`VisionEncoderDecoderModel`), TrOCR (`microsoft/trocr-base-printed`) |
| Object Detection | YOLOv8 (`ultralytics ≥8.0`) |
| Datasets | HuggingFace `datasets ≥2.14`, Pillow, OpenCV |
| Metrics | `editdistance` (NED), `jiwer` (WER/CER), `scikit-learn` |
| Visualization | matplotlib, seaborn |
| Paper Generation | LaTeX (`paper.tex` → `paper_filled.tex`) |
| Tokenization | `sentencepiece` |

---

## Development Workflows

### Full Pipeline (recommended)

```bash
pip install -r requirements.txt

# Full pipeline: download → train all 8 experiments → TrOCR+YOLO → paper
python run_all.py

# Single DONUT experiment only
python run_all.py --experiment 2

# Skip TrOCR+YOLO stages (DONUT only)
python run_all.py --skip-trocr

# Force re-run ignoring cached results
python run_all.py --force

# Generate paper from existing results (no training)
python run_all.py --paper-only

# Skip SROIE auto-install (data already present)
python run_all.py --skip-install

# Skip pretrained baseline evaluation
python run_all.py --skip-pretrained
```

### DONUT Experiments Only

```bash
# Run all 8 experiments
python run_experiments.py --all

# Run single experiment
python run_experiments.py --experiment 3

# Force re-run
python run_experiments.py --all --force
```

### Paper Generation

```bash
python inject_results.py --all --paper paper.tex --output paper_filled.tex
```

### Alternative Numbered-Script Workflow

```bash
python 01_dataset_preparation.py
python 02_train_donut.py
python 03_train_trocr_yolo.py
python 04_evaluate.py
python 05_compare_results.py
```

### Exit Codes (`run_all.py`)

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | One or more experiments had no training data (partial results saved) |
| `2` | Fatal error (missing SROIE data, unrecoverable failure) |

---

## Experiment Definitions

8 DONUT fine-tuning experiments with different dataset combinations:

| Exp | Training Data | Approx. Samples |
|---|---|---|
| 1 | SROIE only (baseline) | ~500 |
| 2 | SROIE + WildReceipt | ~2,240 |
| 3 | SROIE + Invoices-DONUT | ~1,300 |
| 4 | SROIE + CORD | ~1,400 |
| 5 | SROIE + WildReceipt + CORD | ~3,140 |
| 6 | SROIE + WildReceipt + Invoices | ~3,040 |
| 7 | SROIE + CORD + Invoices | ~2,200 |
| 8 | SROIE + All datasets | ~3,940 |

All experiments use an 80/10/10 split of SROIE: **500 train / 63 val / 63 test**.

---

## Dataset Sources

| Dataset | Source | Notes |
|---|---|---|
| SROIE | `https://github.com/zzzDavid/ICDAR-2019-SROIE.git` | Auto-cloned; 80/10/10 split applied |
| WildReceipt | `https://download.openmmlab.com/mmocr/data/wildreceipt.tar` | OpenMMLab tar |
| CORD | HuggingFace `naver-clova-ix/cord-v2` | Requires HF token for faster download |
| Invoices-DONUT | HuggingFace `katanaml-org/invoices-donut-data-v1` | Requires HF token for faster download |

**HF Token:** Place your HuggingFace token in `hf_token.txt` (single line). This file is gitignored and must never be committed. It enables 5–10× faster downloads.

---

## Key Conventions

### 1. Constants — Never Duplicate

All shared constants live exclusively in `constants.py`. **Never redeclare them in other files.** Always import:

```python
from constants import FIELDS, IMAGE_EXTS, MAX_LENGTH, BASE_MODEL, SEED, NEW_TOKENS, EMPTY_GT
```

| Constant | Value | Purpose |
|---|---|---|
| `FIELDS` | `["company", "date", "address", "total"]` | SROIE Task-3 target fields |
| `IMAGE_EXTS` | `frozenset({".jpg", ".jpeg", ...})` | Accepted image extensions |
| `MAX_LENGTH` | `512` | DONUT decoder max token length |
| `BASE_MODEL` | `"naver-clova-ix/donut-base-finetuned-cord-v2"` | Base DONUT checkpoint |
| `SEED` | `42` | Global random seed |
| `NEW_TOKENS` | `["<s_sroie>", ...]` | SROIE special tokens added to tokenizer |
| `EMPTY_GT` | `{"company": "", ...}` | Empty ground-truth template |

### 2. OOP Design Patterns

- **Dataset loaders**: ABC hierarchy in `dataset_loaders.py` — all loaders extend the base class and return `List[Tuple[image_path, gt_dict]]`
- **Trainer**: `DonutTrainer` class in `train.py` wraps `Seq2SeqTrainer`
- **Evaluator**: `DonutEvaluator` class in `evaluate.py`
- **Orchestrator**: `PipelineOrchestrator` in `run_all.py` with `StageResult` dataclass
- **Paper injection**: `PaperInjector` class in `inject_results.py`
- **Experiment config**: `ExperimentConfig` dataclass in `run_experiments.py`

### 3. ExperimentConfig Is the Single Source of Truth

All training hyperparameters (epochs, learning rate, batch size, etc.) come from `ExperimentConfig`. There are **no hardcoded hyperparameters** outside this dataclass. `DonutTrainer` reads all values via duck-typed attribute access on the config object.

### 4. Sequential Execution (No Parallel GPU Stages)

All pipeline stages in `run_all.py` run **sequentially** to prevent GPU memory contention. GPU memory is explicitly freed between stages using `torch.cuda.empty_cache()` and `gc.collect()`.

### 5. Critical lm_head Weight Tying Bug Fix

After `resize_token_embeddings()`, always set:
```python
model.config.tie_word_embeddings = False
```
This ensures `lm_head` and `embed_tokens` are saved independently. Without this fix, `tie_weights()` on reload destroys the learned `lm_head`, causing F1=0 on all predictions.

### 6. Paper Template System

`paper.tex` uses `\VAR{variable_name}` placeholders. `inject_results.py` resolves these from `results/*.json` files and writes `paper_filled.tex`. Never edit `paper_filled.tex` directly — it is regenerated.

### 7. DONUT Output Format

DONUT decodes to structured XML-like token sequences:
```
<s_sroie><s_company>COMPANY NAME</s_company><s_date>01/01/2024</s_date>...</s_sroie>
```
The evaluator unwraps the `{"sroie": {...}}` wrapper from `token2json()` output.

### 8. Evaluation Metric

SROIE Task-3 **global F1** over all `(image, field)` pairs:
- A pair is **TP** if predicted string == ground truth string (case-insensitive, stripped)
- **NED** (Normalized Edit Distance via `editdistance`) is also reported per field (lower = better ↓)
- Parse failures (>50% threshold) raise an error to catch broken models early

---

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

## TrOCR + YOLO Architecture

Two-stage pipeline (`03_train_trocr_yolo.py`, called by `run_all.py`):

1. **YOLOv8n** (`yolov8n.pt`) — detects text regions as bounding boxes
   - Config: `YOLO_EPOCHS=50`, `YOLO_IMG_SIZE=640`, `YOLO_BATCH=8`
2. **TrOCR** (`microsoft/trocr-base-printed`) — reads text from cropped regions
   - Config: `TROCR_EPOCHS=10`, `TROCR_BATCH=8`, `TROCR_LR=5e-5`, `TROCR_MAX_LEN=128`
3. **Rule-based heuristics** — assign extracted text to SROIE fields

YOLO training data: YAML at `data/yolo/dataset.yaml`. TrOCR crops: `data/trocr/`.

---

## Environment & Paths

| Path | Default | Override |
|---|---|---|
| Workspace (model checkpoints) | `/workspace` | `--workspace PATH` or `DONUT_WORKSPACE` env var |
| SROIE data directory | `/workspace/ICDAR-2019-SROIE/data` | `--sroie-dir PATH` |
| Results directory | `results/` (relative) | hardcoded |
| Paper template | `paper.tex` | `--paper-template F` |
| Paper output | `paper_filled.tex` | `--output F` |

---

## Gitignored Runtime Artifacts

These are **generated at runtime** and must never be committed:

- `results/` — experiment JSON outputs
- `data/` — dataset cache
- `models/` — model checkpoints (`*.pt`, `*.pth`, `*.pkl`, `*.h5`)
- `paper_filled.tex` — generated LaTeX paper
- `hf_token.txt` — HuggingFace authentication token

---

## Performance Tips

- **HF Token**: `hf_token.txt` with your token gives 5–10× faster dataset downloads
- **Dataset downloads**: All auxiliary datasets download in parallel automatically
- **RAM cache**: Images are pre-loaded into RAM when sufficient memory is available
- **DataLoader**: Uses `pin_memory=True`, `prefetch_factor=4`, `persistent_workers=True`
- **GPU**: CUDA is used automatically when available; CPU fallback is supported but slow

---

## Known Issues & Historical Fixes

| Issue | Fix Applied |
|---|---|
| `protobuf` missing → 100% pipeline crash | Added `protobuf>=3.20.0` to `requirements.txt` |
| `lm_head` weight tying → F1=0 on reload | `config.tie_word_embeddings=False` after `resize_token_embeddings()` |
| Key file loading failure | Try `.txt` first, then `.json` (BUG A/E fix in `train.py`) |
| Data leakage in eval | Fixed split logic in `run_experiments.py` |
| `{"sroie": {...}}` wrapper in token2json | Unwrapped in `evaluate.py` |
| FIELDS/IMAGE_EXTS duplicated in 5+ files | Consolidated in `constants.py` |
| transformers ≥4.47 `PreTrainedTokenizerBase` move | Compat shim in `dataset_loaders.py` |
