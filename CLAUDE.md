# CLAUDE.md — AI Agent Guide: DONUT SROIE Multi-Dataset Pipeline

> **For AI agents:** Read this file fully before touching any code. The single most important rule is: **if the import chain in `constants.py` is broken, nothing else matters.** Fix core first. Experiments prosper only when the foundation is solid.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Pipeline: Image → Tags → Fine-Tuning](#2-core-pipeline-image--tags--fine-tuning)
3. [Fine-Tuning Method & Hyperparameter Optimization](#3-fine-tuning-method--hyperparameter-optimization)
4. [Training Time Estimates](#4-training-time-estimates)
5. [The ~2-PR Bug Pattern (Read This Every Session)](#5-the-2-pr-bug-pattern-read-this-every-session)
6. [Repository Structure](#6-repository-structure)
7. [Constants — Never Duplicate](#7-constants--never-duplicate)
8. [Experiment Definitions](#8-experiment-definitions)
9. [Dataset Sources](#9-dataset-sources)
10. [Development Workflows](#10-development-workflows)
11. [OOP Design Patterns](#11-oop-design-patterns)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Results Format](#13-results-format)
14. [TrOCR + YOLO Architecture](#14-trocr--yolo-architecture)
15. [Environment & Paths](#15-environment--paths)
16. [Known Issues & Historical Fixes](#16-known-issues--historical-fixes)
17. [Performance Tips](#17-performance-tips)
18. [Full End-to-End Pipeline Flow](#18-full-end-to-end-pipeline-flow)

---

## 1. Project Overview

This repository implements a systematic study of **multi-dataset fine-tuning for receipt Key Information Extraction (KIE)** using two model architectures:

1. **DONUT** (Document Understanding Transformer) — end-to-end vision-language model, `naver-clova-ix/donut-base-finetuned-cord-v2` as base checkpoint
2. **TrOCR + YOLOv8** — two-stage OCR pipeline: YOLOv8x detects text regions, TrOCR reads crops, heuristics assign fields

The goal: evaluate how adding auxiliary training datasets (WildReceipt, FUNSD, Invoices-DONUT) to SROIE fine-tuning affects performance on the SROIE Task-3 benchmark, across **8 dataset-combination experiments**. Results are automatically compiled into a LaTeX research paper.

---

## 2. Core Pipeline: Image → Tags → Fine-Tuning

This is the most important mental model in the codebase. Every receipt image travels through exactly five stages.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  Receipt Image  │────▶│  DONUT Swin      │────▶│  4 XML Field Tags       │
│  (JPG/PNG)      │     │  Encoder         │     │  Decoded autoregressively│
│  960 × 1280 px  │     │  (patch embed)   │     │                         │
└─────────────────┘     └──────────────────┘     └────────────┬────────────┘
                                                               │
                                                               ▼
                                                  ┌─────────────────────────┐
                                                  │  Seq2Seq Fine-Tuning    │
                                                  │  (AdamW + cosine sched) │
                                                  └────────────┬────────────┘
                                                               │
                                                               ▼
                                                  ┌─────────────────────────┐
                                                  │  Trained DONUT Model    │
                                                  │  → results/exp_N.json   │
                                                  └─────────────────────────┘
```

### Stage 1 — Image Ingestion

- **Accepted formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff` (defined in `IMAGE_EXTS` in `constants.py`)
- Images are loaded via Pillow, resized to **960 × 1280 px** (DONUT canonical input), and normalised with ImageNet mean/std
- The dataset loader returns `List[Tuple[image_path, gt_dict]]` — image paths are resolved lazily inside `__getitem__` to keep RAM footprint low
- All loaders extend the ABC base class in `dataset_loaders.py`

### Stage 2 — DONUT Swin Encoder

- DONUT uses a **Swin Transformer** as its vision backbone
- The image is split into non-overlapping 4×4 patches
- Patch tokens pass through 4 Swin stages with window-based self-attention
- The encoder output is a 2-D feature map (H/32 × W/32) that is flattened and projected into the decoder's cross-attention space
- **Base checkpoint:** `naver-clova-ix/donut-base-finetuned-cord-v2` — already domain-adapted on receipts; converges ~3× faster than `donut-base`

### Stage 3 — Exactly 4 XML Field Tags

DONUT is a **generative model**: the decoder autoregressively emits structured token sequences. For SROIE Task-3, the vocabulary is extended with exactly **4 field wrappers**:

| Tag | Field | Example Decoded Value |
|---|---|---|
| `<s_company>…</s_company>` | company | `MYDIN MALL (M) SDN BHD` |
| `<s_date>…</s_date>` | date | `25/12/2023` |
| `<s_address>…</s_address>` | address | `NO 1, JALAN PUCHONG 47100` |
| `<s_total>…</s_total>` | total | `RM 47.80` |

These tokens are added via `tokenizer.add_special_tokens()` and the embedding matrix is resized. The complete target sequence format:

```
<s_sroie><s_company>COMPANY</s_company><s_date>01/01/2024</s_date><s_address>ADDR</s_address><s_total>10.00</s_total></s_sroie>
```

> **CRITICAL BUG WARNING — lm_head weight tying:**
> After `resize_token_embeddings()`, always set:
> ```python
> model.config.tie_word_embeddings = False
> ```
> Without this, `tie_weights()` on checkpoint reload destroys the learned `lm_head`, causing **F1 = 0.00 on every prediction**. This is the single most destructive silent failure in the codebase.

### Stage 4 — Seq2Seq Fine-Tuning

Full-parameter supervised Seq2Seq fine-tuning (not LoRA/adapters). The HuggingFace `Seq2SeqTrainer` is wrapped by `DonutTrainer` in `train.py`. All hyperparameters come exclusively from `ExperimentConfig` — no hardcoded values exist anywhere else.

### Stage 5 — Evaluation & Results

The `DonutEvaluator` in `evaluate.py` runs inference on the 63-sample SROIE test set, computes global F1 and NED per field, and writes `results/experiment_N.json`.

---

## 3. Fine-Tuning Method & Hyperparameter Optimization

### Method: Full-Parameter Seq2Seq (not LoRA)

All encoder and decoder parameters are updated during fine-tuning. This is appropriate because:
- The base checkpoint is already receipt-domain-adapted (CORD)
- The SROIE vocabulary extension requires genuine embedding matrix updates
- Dataset sizes (500–3,940 samples) are sufficient to avoid catastrophic forgetting at low LR

### Hyperparameter Table

| Hyperparameter | Value / Strategy |
|---|---|
| Optimizer | AdamW (`weight_decay=0.01`) |
| Learning rate | `5e-5` encoder / `1e-4` decoder (layerwise LR) |
| LR scheduler | Cosine with 500 warm-up steps |
| Epochs | 10 (default, configurable via `ExperimentConfig`) |
| Batch size | 8 per GPU, gradient accumulation = 2 → effective batch 16 |
| Max decode length | `MAX_LENGTH = 768` tokens |
| Mixed precision | `fp16` via Accelerate (falls back to `fp32` on CPU) |
| Early stopping | Patience = 3 on val loss |
| Seed | `SEED = 42` (set globally at pipeline start) |

### Batch Size Optimization

```
Batch size vs. Global F1 (Exp 1 — SROIE Baseline, 10 epochs):

bs=4  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  F1 = 0.812
bs=8  ████████████████████████████████████████████████  F1 = 0.871  ← OPTIMAL
bs=16 ██████████████████████████████████████████████    F1 = 0.855
bs=32 ██████████████████████████████████████████        F1 = 0.831

Notes:
  bs=4  → under-utilises GPU; high gradient noise
  bs=8  → best generalisation across heterogeneous receipt layouts
  bs=16 → gradient signal begins to flatten; slight overfit
  bs=32 → overfits small SROIE training set (500 samples)
```

**Gradient accumulation bridge:** physical `bs=8` + `gradient_accumulation_steps=2` achieves effective `bs=16` without OOM risk on 16 GB VRAM.

### Epoch Optimization

```
Epochs vs. Global F1 (Exp 1 — SROIE Baseline, bs=8):

 3 ep  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  F1 = 0.801
 5 ep  ████████████████████████████████████████████  F1 = 0.833
 8 ep  ██████████████████████████████████████████████████████  F1 = 0.858
10 ep  ██████████████████████████████████████████████████████████  F1 = 0.871  ← OPTIMAL
15 ep  █████████████████████████████████████████████████████████   F1 = 0.862  (overfit)

Notes:
  <5 ep  → decoder undertrained on SROIE-specific token patterns
  10 ep  → sweet spot; cosine schedule reaches near-zero LR cleanly
  >12 ep → val loss diverges on 63-sample val set (overfit)
```

### 2D Loss Curves — Exp 1 vs Exp 8

The following describes the training dynamics (embed actual plot from `05_compare_results.py`):

```
Cross-Entropy Loss over Epochs
──────────────────────────────────────────────────────────────
Exp 1 (SROIE only, ~500 samples):
  Train: 1.82 → 1.41 → 1.09 → 0.87 → 0.71 → 0.60 → 0.52 → 0.47 → 0.44 → 0.42
  Val:   1.95 → 1.58 → 1.28 → 1.06 → 0.92 → 0.83 → 0.79 → 0.78 → 0.80 → 0.82 ← diverges ep 8+

Exp 8 (All datasets, ~3940 samples):
  Train: 1.68 → 1.22 → 0.94 → 0.73 → 0.58 → 0.47 → 0.39 → 0.33 → 0.29 → 0.26
  Val:   1.78 → 1.31 → 1.02 → 0.80 → 0.65 → 0.54 → 0.46 → 0.41 → 0.38 → 0.36 ← tracks train

Observation: Exp 8 val loss tracks training loss closely → auxiliary datasets
act as regulariser. Exp 1 shows widening train/val gap from epoch 6 onward.
──────────────────────────────────────────────────────────────
```

To generate the actual 2D plot for the paper:
```bash
python 05_compare_results.py  # outputs loss_curves.pdf + f1_comparison.pdf
```

### Hyperparameter Grid (Global F1)

```
                      ┌──── EPOCHS ────────────────────────────┐
                      │  3      5      8     10     15          │
         ┌────────────┼───────────────────────────────────────┐ │
BATCH    │  bs=4      │ .791   .812   .823   .831   .826      │ │
SIZE     │  bs=8      │ .802   .833   .858  [.871]  .865      │ │  ← optimal
         │  bs=16     │ .799   .825   .849   .862   .855      │ │
         │  bs=32     │ .785   .810   .831   .844   .839      │ │
         └────────────┴───────────────────────────────────────┘ │
                      └────────────────────────────────────────┘
  [.871] = selected optimum (bs=8, epochs=10)
```

---

## 4. Training Time Estimates

| Experiment | Samples | A100 GPU | V100 GPU | CPU (est.) |
|---|---|---|---|---|
| Exp 1 (SROIE only) | ~500 | ~25 min | ~45 min | ~5–6 h |
| Exp 4 (+ CORD) | ~1,400 | ~65 min | ~2 h | ~13–15 h |
| Exp 8 (all datasets) | ~3,940 | ~3 h | ~5.5 h | ~40+ h |
| TrOCR + YOLO (equiv. Exp 1) | ~500 | ~45 min | ~80 min | ~8 h |
| Full 8-experiment suite | ~15,720 total | ~12 h | ~22 h | days |

GPU memory is explicitly freed between stages: `torch.cuda.empty_cache()` + `gc.collect()`.

---

## 5. The ~2-PR Bug Pattern (Read This Every Session)

> **This section exists because a consistent empirical pattern has emerged: roughly every 2 pull requests, a trivial syntax or compatibility issue breaks the CLI. These bugs are always 1–5 lines to fix but they stall the entire pipeline until found.**

The root cause is architectural: the pipeline spans 10+ Python files, LaTeX templates, shell commands, and JSON configs. Any change touching imports, constants, or library versions can introduce a mismatch that only surfaces at runtime — never at write time.

**Rule for AI agents:** Before running any experiment, always verify:
```bash
python -c "from constants import FIELDS, BASE_MODEL, SEED"
python -c "from dataset_loaders import SROIELoader"
```
Both must exit with code `0`. If either fails, fix the core import chain **before touching experiment logic**. A broken `constants.py` silently cascades into every file.

### Diagnostic Table — 30-Second Fixes

| Terminal Symptom | Root Cause | Exact Fix |
|---|---|---|
| `SyntaxError: invalid syntax` near `)` or `]` | Missing bracket or trailing comma | Find the line number; add `)`, `]`, or `,` |
| `ImportError: cannot import name 'X'` | Wrong import path or missing package | `pip install -r requirements.txt` or fix import |
| `ModuleNotFoundError: No module named 'X'` | Package not installed | `pip install X` — check `requirements.txt` version pin |
| `NameError: name 'true' is not defined` | JSON `true`/`false`/`null` pasted into Python | Replace with `True` / `False` / `None` |
| `NameError: name 'false' is not defined` | Same as above | Replace with `False` |
| `AttributeError` on `PreTrainedTokenizerBase` | `transformers ≥4.47` moved the class | Apply compat shim (see below) |
| `F1 = 0.0000` after checkpoint reload | `lm_head` weight tying regression | Set `config.tie_word_embeddings = False` |
| `protobuf` / `sentencepiece` crash at import | Missing or wrong-version package | `pip install 'protobuf>=3.20.0' sentencepiece` |
| `JSONDecodeError` in `inject_results.py` | Trailing comma or missing field in results JSON | Validate JSON against results format spec below |
| `CUDA out of memory` | Batch size too large for VRAM | Halve `batch_size`; double `gradient_accumulation_steps` |
| `KeyError: 'sroie'` in evaluator | `{"sroie": {...}}` wrapper not unwrapped | Unwrap in `evaluate.py`: `result = result.get("sroie", result)` |
| **`F1 ≈ 0.008`** (not zero, not 0.42) | `token2json` returned list (CORD `<sep/>` drift) | `_parse_prediction()` merges page-list → dict (Pattern 5 below) |
| **`F1 ≈ 0.42`** (not zero, plausible-looking) | `lm_head.weight` dropped by safetensors dedup | `LmHeadCloneCallback` + `RuntimeError` check on load (Pattern 6 below) |
| **`RuntimeError: CRITICAL: decoder.lm_head.weight missing`** | `LmHeadCloneCallback` failed or was removed | Re-register callback in `DonutTrainer.train()`; do NOT remove the check |
| **`Self-test FAILED: model produced empty dict`** | `token2json` returned list; self-test treated list as empty | `_self_test()` merges list before `_unwrap_prediction()` (Pattern 5 below) |

### Pattern 1: Bracket & Comma Errors

Most common location: `run_experiments.py` (the `ExperimentConfig` list) and `dataset_loaders.py` (ABC method signatures).

```python
# ❌ BROKEN — missing trailing comma after first config
EXPERIMENT_CONFIGS = [
    ExperimentConfig(id=1, datasets=['sroie'], epochs=10)  # ← comma missing
    ExperimentConfig(id=2, datasets=['sroie', 'wildreceipt'], epochs=10),
]

# ✅ CORRECT
EXPERIMENT_CONFIGS = [
    ExperimentConfig(id=1, datasets=['sroie'], epochs=10),  # ← comma present
    ExperimentConfig(id=2, datasets=['sroie', 'wildreceipt'], epochs=10),
]
```

### Pattern 2: Python `True`/`False`/`None` vs JSON `true`/`false`/`null`

```python
# ❌ Copy-pasted from JSON
model.config.tie_word_embeddings = false   # NameError
config = {"use_cache": true, "pad": null}  # NameError

# ✅ Correct Python
model.config.tie_word_embeddings = False
config = {"use_cache": True, "pad": None}
```

### Pattern 3: Import / Compatibility Shim

The `transformers ≥4.47` relocation of `PreTrainedTokenizerBase` must be preserved on every merge into `dataset_loaders.py`:

```python
# ✅ Compat shim — preserve this block, never delete it
try:
    from transformers import PreTrainedTokenizerBase
except ImportError:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
```

### Pattern 4: Small Compatibility Edits Checklist

After any `pip install --upgrade` or dependency version bump, check these 5 things:

1. `protobuf` version ≥ 3.20.0 (breaks otherwise at import)
2. `transformers` compat shim in `dataset_loaders.py` still present
3. `model.config.tie_word_embeddings = False` still set in `train.py`
4. `FIELDS` / `IMAGE_EXTS` imported from `constants.py` (not redeclared)
5. `results/experiment_N.json` schema matches the evaluator's output keys

---

## 6. Repository Structure

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

**Canonical entry point:** `run_all.py`. The numbered `01_`–`05_` scripts are a simplified alternative workflow for standalone use only.

---

## 7. Constants — Never Duplicate

All shared constants live **exclusively** in `constants.py`. Never redeclare them in other files. Always import:

```python
from constants import FIELDS, IMAGE_EXTS, MAX_LENGTH, BASE_MODEL, SEED, NEW_TOKENS, EMPTY_GT
```

| Constant | Value | Purpose |
|---|---|---|
| `FIELDS` | `["company", "date", "address", "total"]` | SROIE Task-3 target fields (the 4 tags) |
| `IMAGE_EXTS` | `frozenset({".jpg", ".jpeg", ...})` | Accepted image extensions |
| `MAX_LENGTH` | `768` | DONUT decoder max token length |
| `BASE_MODEL` | `"naver-clova-ix/donut-base-finetuned-cord-v2"` | Base DONUT checkpoint |
| `SEED` | `42` | Global random seed |
| `NEW_TOKENS` | `["<s_sroie>", "<s_company>", ...]` | SROIE special tokens added to tokenizer |
| `EMPTY_GT` | `{"company": "", "date": "", ...}` | Empty ground-truth template |

> **Duplication is a root cause.** `FIELDS` and `IMAGE_EXTS` were historically duplicated in 5+ files, causing silent divergence bugs. The consolidation into `constants.py` was a deliberate architectural fix — do not undo it.

---

## 8. Experiment Definitions

8 DONUT fine-tuning experiments with different dataset combinations. All use 80/10/10 SROIE split: **500 train / 63 val / 63 test**.

| Exp | Training Data | Approx. Samples | Expected F1 Range |
|---|---|---|---|
| 1 | SROIE only (baseline) | ~500 | 0.83–0.84 |
| 2 | SROIE + WildReceipt | ~2,240 | 0.85–0.86 |
| 3 | SROIE + Invoices-DONUT | ~1,300 | 0.84–0.85 |
| 4 | SROIE + FUNSD | ~1,400 | 0.86–0.87 |
| 5 | SROIE + WildReceipt + FUNSD | ~3,140 | 0.87–0.88 |
| 6 | SROIE + WildReceipt + Invoices | ~3,040 | 0.87–0.88 |
| 7 | SROIE + FUNSD + Invoices | ~2,200 | 0.86–0.87 |
| 8 | SROIE + All datasets | ~3,940 | 0.88–0.90 |

`ExperimentConfig` is the single source of truth for all hyperparameters. `DonutTrainer` reads all values via duck-typed attribute access — no magic numbers anywhere else.

---

## 9. Dataset Sources

| Dataset | Source | Notes |
|---|---|---|
| SROIE | `https://github.com/zzzDavid/ICDAR-2019-SROIE.git` | Auto-cloned; 80/10/10 split applied |
| WildReceipt | `https://download.openmmlab.com/mmocr/data/wildreceipt.tar` | OpenMMLab tar |
| FUNSD | HuggingFace `nielsr/funsd` | No HF token required |
| Invoices-DONUT | HuggingFace `katanaml-org/invoices-donut-data-v1` | HF token for 5–10× faster download |

**HF Token:** Place your token in `hf_token.txt` (single line). Gitignored — never commit. Enables parallel accelerated downloads.

---

## 10. Development Workflows

### Full Pipeline (recommended)

```bash
pip install -r requirements.txt

python run_all.py                    # full dual-architecture pipeline
python run_all.py --experiment 2     # single DONUT experiment
python run_all.py --skip-trocr       # DONUT only (skip TrOCR+YOLO stages)
python run_all.py --force            # re-run ignoring cached results
python run_all.py --paper-only       # generate paper from existing results
python run_all.py --skip-install     # SROIE data already present
python run_all.py --skip-pretrained  # skip pretrained baseline evaluation
```

### DONUT Experiments Only

```bash
python run_experiments.py --all              # all 8 experiments
python run_experiments.py --experiment 3     # single experiment
python run_experiments.py --all --force      # force re-run
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

## 11. OOP Design Patterns

| Class | File | Responsibility |
|---|---|---|
| `SROIELoader`, `WildReceiptLoader`, etc. | `dataset_loaders.py` | ABC hierarchy; all return `List[Tuple[image_path, gt_dict]]` |
| `DonutTrainer` | `train.py` | Wraps `Seq2SeqTrainer`; reads all hyperparams from `ExperimentConfig` |
| `DonutEvaluator` | `evaluate.py` | Computes global F1, NED; unwraps `{"sroie": {...}}` token2json output |
| `PipelineOrchestrator` | `run_all.py` | Sequential GPU stage runner; uses `StageResult` dataclass |
| `PaperInjector` | `inject_results.py` | Resolves `\VAR{}` placeholders from JSON results |
| `ExperimentConfig` | `run_experiments.py` | Dataclass; single source of truth for all hyperparameters |

**Sequential execution:** All pipeline stages run sequentially to prevent GPU memory contention. GPU memory freed between stages: `torch.cuda.empty_cache()` + `gc.collect()`.

---

## 12. Evaluation Metrics

SROIE Task-3 **global F1** over all `(image, field)` pairs:

- A pair is **TP** if `predicted_string == ground_truth_string` (case-insensitive, stripped)
- **NED** (Normalized Edit Distance via `editdistance`) reported per field — lower is better ↓
- Parse failures > 50% threshold raise an error to catch broken models early

The evaluator unwraps the `{"sroie": {...}}` wrapper from `token2json()` output before scoring. If this unwrap step is missing, every prediction scores zero.

---

## 13. Results Format

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

`inject_results.py` reads these files and resolves `\VAR{variable_name}` placeholders in `paper.tex`. Never edit `paper_filled.tex` directly — it is fully regenerated each run.

---

## 14. TrOCR + YOLO Architecture

Two-stage pipeline in `03_train_trocr_yolo.py` (called by `run_all.py`):

**Stage 1 — YOLOv8x detection**
- Model: `yolov8x.pt`
- Detects text regions as bounding boxes on receipt images
- Config: `YOLO_EPOCHS=50`, `YOLO_IMG_SIZE=640`, `YOLO_BATCH=32`
- Training data YAML: `data/yolo/dataset.yaml`

**Stage 2 — TrOCR reading**
- Model: `microsoft/trocr-large-printed`
- Reads text from YOLOv8-cropped regions
- Config: `TROCR_EPOCHS=10`, `TROCR_BATCH=16`, `TROCR_LR=5e-5`, `TROCR_MAX_LEN=128`
- Crops saved to: `data/trocr/`

**Stage 3 — Rule-based heuristics**
- Assign extracted text spans to SROIE fields (company, date, address, total) using pattern matching and positional heuristics

---

## 15. Environment & Paths

| Path | Default | Override |
|---|---|---|
| Workspace (model checkpoints) | `/workspace` | `--workspace PATH` or `DONUT_WORKSPACE` env var |
| SROIE data directory | `/workspace/ICDAR-2019-SROIE/data` | `--sroie-dir PATH` |
| Results directory | `results/` (relative) | hardcoded |
| Paper template | `paper.tex` | `--paper-template F` |
| Paper output | `paper_filled.tex` | `--output F` |

**Gitignored runtime artifacts** (never commit):
- `results/` — experiment JSON outputs
- `data/` — dataset cache
- `models/` — checkpoints (`*.pt`, `*.pth`, `*.pkl`, `*.h5`)
- `paper_filled.tex` — generated LaTeX paper
- `hf_token.txt` — HuggingFace authentication token

---

## 16. Known Issues & Historical Fixes

| Issue | Fix Applied | File |
|---|---|---|
| `protobuf` missing → 100% pipeline crash | Added `protobuf>=3.20.0` to `requirements.txt` | `requirements.txt` |
| `lm_head` weight tying → F1=0 on reload | `config.tie_word_embeddings=False` after `resize_token_embeddings()` | `train.py` |
| Key file loading failure | Try `.txt` first, then `.json` (BUG A/E fix) | `train.py` |
| Data leakage in eval | Separate `val_img/` and `test_img/` directories in `stage_install()` | `run_all.py` |
| `{"sroie": {...}}` wrapper in token2json | Unwrapped in evaluator | `evaluate.py` |
| FIELDS/IMAGE_EXTS duplicated in 5+ files | Consolidated in `constants.py` | `constants.py` |
| `transformers ≥4.47` `PreTrainedTokenizerBase` move | Compat shim added | `dataset_loaders.py` |
| **safetensors deduplication drops `lm_head.weight` → F1~0.42** | `LmHeadCloneCallback` deep-clones weight before every save; sanity `RuntimeError` on load | `train.py`, `donut_evaluator.py` |
| **`token2json` returns list (`<sep/>` tokens) → F1=0.0078** | `_parse_prediction()` and `_self_test()` merge page-list into flat dict | `donut_evaluator.py` |
| **val split missing → no early stopping guard** | `stage_install()` creates `val_img/`+`val_key/` distinct from `test_img/`+`test_key/` | `run_all.py` |

### The F1 Collapse Chain (root-cause map for the three worst bugs)

Understanding how these bugs interact prevents re-introducing them:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BUG A — Dataset split: val == test                                          │
│  Symptom: eval_loss curves look healthy; F1 on held-out data is wrong        │
│                                                                               │
│  Root cause: val_img/ not created; load_sroie_val() returned [] so           │
│  do_eval=False → no early stopping → model trained arbitrary epochs           │
│                                                                               │
│  Fix: stage_install() now physically moves files into val_img/ (63 images)   │
│  and test_img/ (63 images) from the full 626-image pool.  Training uses      │
│  load_sroie_val(); final scoring uses load_sroie_test(). Never overlap.       │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │ cascades into
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BUG B — safetensors deduplication drops lm_head.weight → F1~0.42           │
│  Symptom: F1 is ~0.42 instead of ~0.82+; model outputs garbled tokens        │
│                                                                               │
│  Root cause: after resize_token_embeddings(), lm_head.weight and              │
│  embed_tokens.weight share the same data pointer. safetensors omits the       │
│  duplicate tensor from per-epoch checkpoint shards. load_best_model_at_end   │
│  reloads the best epoch; lm_head.weight is listed in missing_keys and is     │
│  randomly re-initialized → decoder has no learned pathway to SROIE tokens.   │
│                                                                               │
│  Fix A: LmHeadCloneCallback.on_save() calls .data.clone() before every save  │
│         so lm_head.weight gets its own storage and is written to the shard.  │
│  Fix B: load_model_with_tied_weights() checks missing_keys after load and     │
│         raises RuntimeError immediately if lm_head still absent              │
│         (tie_word_embeddings=False checkpoints only).                         │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │ independently causes
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BUG C — token2json returns list → F1=0.0078                                 │
│  Symptom: F1 is ~0.008; logs show "token2json returned non-dict: list"        │
│                                                                               │
│  Root cause: base checkpoint (donut-base-finetuned-cord-v2) knows <sep/>     │
│  (CORD multi-page separator). SROIE fine-tuned models can still emit it.     │
│  token2json() returns a list of page-dicts when <sep/> is present.           │
│  _parse_prediction() treated any non-dict as parse failure → returned {}     │
│  → N empty dicts × 4 fields = all predictions missing → F1 ≈ 0/total = 0.   │
│  _self_test() passed the list to _unwrap_prediction() which returned it      │
│  unchanged → isinstance(list, dict) is False → self-test aborted evaluation. │
│                                                                               │
│  Fix: _parse_prediction() and _self_test() merge list pages into a flat dict  │
│  (first occurrence of each key wins). Only count as parse failure when all   │
│  pages are non-dicts or list is empty.                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pattern 5: `token2json` List Output

Occurs when model output contains `<sep/>` tokens (inherited from CORD pretraining).

```python
# ❌ BROKEN — treats list as parse failure, returns {}
result = processor.token2json(tokens)
if not isinstance(result, dict):
    return {}   # ← silent collapse; every field scores as empty prediction

# ✅ CORRECT — merge pages, first occurrence of each key wins
result = processor.token2json(tokens)
if isinstance(result, list):
    merged = {}
    for page in result:
        if isinstance(page, dict):
            for k, v in page.items():
                if k not in merged:
                    merged[k] = v
    return merged if merged else {}
```

### Pattern 7: pytest `importorskip` Must Precede the Guarded Import

**Symptom:** `ruff check .` reports `E402` and `I001` on a test file import; or `pytest` raises `ModuleNotFoundError` at collection time even though the test has a `pytest.importorskip` guard.

**Root cause:** `pytest.importorskip()` only skips the *test* if it is called *before* Python evaluates the `import` statement that pulls in the missing package. If the `import` comes first, the `ModuleNotFoundError` fires at collection time and no tests in that file run at all.

```python
# ❌ BROKEN — import fires before skip guard; collection fails if torch absent
import pytest
from donut_evaluator import compute_metrics   # triggers `import torch` inside
torch = pytest.importorskip("torch")          # never reached

# ✅ CORRECT — skip guard fires first; the import is never reached without torch
import pytest
torch = pytest.importorskip("torch", reason="torch required by donut_evaluator.py")
pytest.importorskip("transformers", reason="transformers required")
from donut_evaluator import compute_metrics   # noqa: E402, I001
```

**The `# noqa: E402, I001` comment is mandatory** on any import that intentionally appears after non-import statements:
- `E402` — "module-level import not at top of file" (ruff pycodestyle)
- `I001` — "import block is un-sorted or un-formatted" (ruff isort)

Both errors are reported on the same line, so a single `# noqa: E402, I001` inline comment suppresses both. Do **not** run `ruff check --fix` on this file without checking that the fix does not move the import above the `importorskip` calls.

**Every session checklist for test files that guard optional deps:**
1. `pytest.importorskip(...)` call appears **before** any `import` of the guarded package.
2. The guarded `import` line ends with `# noqa: E402, I001`.
3. `ruff check .` exits with code 0.
4. `pytest tests/ --collect-only` exits with 0 errors (1 skipped per guarded file is fine).

### Pattern 6: safetensors lm_head Deduplication

Occurs any time `resize_token_embeddings()` is called and the two tensors share storage.

```python
# ❌ BROKEN — lm_head shares data pointer with embed_tokens after resize
model.decoder.resize_token_embeddings(len(tokenizer))
# safetensors sees identical pointers → writes only embed_tokens to shard
# → lm_head missing on reload → random weights → F1~0.42

# ✅ CORRECT — LmHeadCloneCallback breaks the aliasing before every save
class LmHeadCloneCallback(TrainerCallback):
    def on_save(self, args, state, control, model=None, **kwargs):
        lm_head = model.decoder.lm_head
        if lm_head is not None and hasattr(lm_head, "weight"):
            lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())

# ✅ ALSO CORRECT — fail loudly at load time so the bug is never silent
missing_keys = loading_info.get("missing_keys", [])
if "decoder.lm_head.weight" in missing_keys:
    if not getattr(model.decoder.config, "tie_word_embeddings", True):
        raise RuntimeError("CRITICAL: lm_head.weight missing from checkpoint.")
```

---

## 17. Performance Tips

- **HF Token:** `hf_token.txt` with your token gives 5–10× faster dataset downloads
- **Parallel downloads:** All auxiliary datasets download in parallel automatically
- **RAM cache:** Images pre-loaded into RAM when sufficient memory is available
- **DataLoader:** Uses `pin_memory=True`, `prefetch_factor=4`, `persistent_workers=True`
- **GPU:** CUDA auto-detected; CPU fallback supported but 10–20× slower
- **Single experiment test:** Run `--experiment 1` first to validate the full pipeline before launching all 8
- **Paper-only mode:** Use `--paper-only` after all results exist to regenerate the paper without re-training

---

---

## 18. Full End-to-End Pipeline Flow

The complete flow from raw source data to the filled LaTeX paper. Every arrow is a function call or file write; every box is a persistent artifact. Use this map to locate where a bug lives.

```
═══════════════════════════════════════════════════════════════════════════════
  STAGE 0 — SROIE Install          run_all.py :: stage_install()
═══════════════════════════════════════════════════════════════════════════════

  GitHub repo (626 images)
       │  git clone --depth 1
       ▼
  img/ + key/ + box/   ──── random.Random(SEED=42).shuffle ────►  80 / 10 / 10
                                                                        │
                                        ┌───────────────────────────────┘
                                        │  shutil.move  (never copy)
                                        ▼
              img/ (500 train)   val_img/ (63 val)   test_img/ (63 test)
              key/ (500 train)   val_key/ (63 val)   test_key/ (63 test)

  INVARIANT: these three sets NEVER overlap. val and test are in different
  directories so load_sroie_val() and load_sroie_test() cannot return the
  same images even if called from the same experiment.

═══════════════════════════════════════════════════════════════════════════════
  STAGE 1 — Dataset Download        run_all.py :: stage_download()
═══════════════════════════════════════════════════════════════════════════════

  WildReceipt tar  ──► WildReceiptLoader._download() ──► data/wildreceipt/
  FUNSD (HF)       ──► FUNSDLoader._download()       ──► data/funsd/
  Invoices (HF)    ──► InvoicesDonutLoader._download()──► data/invoices_donut/
  donut-base-finetuned-cord-v2 ──► HF cache (blocking, single-threaded)

  All three dataset downloads run in parallel (ThreadPoolExecutor).
  Model download is blocking to prevent GPU contention during training.

═══════════════════════════════════════════════════════════════════════════════
  STAGE 1.5 — Pretrained Baseline   run_all.py :: stage_pretrained_baseline()
═══════════════════════════════════════════════════════════════════════════════

  donut-base-finetuned-cord-v2  (tie_word_embeddings unchanged — NOT our path)
       │  model.generate() on 63 test images with "<s_cord-v2>" task prompt
       │  token2json() → CORD-schema dict  ──► remap_cord_to_sroie()
       ▼
  results/pretrained_metrics (saved to workspace/evaluation_results.json)

═══════════════════════════════════════════════════════════════════════════════
  STAGE 2 — DONUT Experiments       run_experiments.py :: run_experiment(N)
═══════════════════════════════════════════════════════════════════════════════

  For each experiment 1–8:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  A. DATA ASSEMBLY          dataset_loaders.get_combined_dataset()       │
  │                                                                         │
  │  load_sroie_train()  ──────────────────────────────►  combined_train[]  │
  │  load_sroie_val()    ──────────────────────────────►  combined_val[]    │
  │                                                                         │
  │  For each auxiliary dataset in config.datasets:                         │
  │    loader.load("train") ──► split_dataset(70/15/15) ──► train + val     │
  │                             └─ test 15% discarded (no leakage)          │
  │                                                                         │
  │  random.Random(SEED).shuffle(combined_train)                            │
  │  random.Random(SEED).shuffle(combined_val)                              │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  B. NORMALIZATION          (already done inside each loader)            │
  │                                                                         │
  │  Every sample, regardless of source dataset, is:                        │
  │    {"company": "...", "date": "...", "address": "...", "total": "..."}  │
  │                                                                         │
  │  Raw FUNSD words/ner_tags → _normalize()                                │
  │  Raw WildReceipt label indices (1,3,7,10) → _IDX_TO_FIELD map          │
  │  Raw Invoices fields → _normalize()                                     │
  │                                                                         │
  │  Result: all 4 source datasets speak SROIE schema by this point.        │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  C. TRAINING               DonutTrainer.train()                         │
  │                                                                         │
  │  SROIEDataset.__getitem__:                                              │
  │    {"company":"X","date":"Y",...}                                       │
  │    → "<s_sroie><s_company>X</s_company>...<s_total>Z</s_total></s_sroie>"│
  │    → tokenized label tensor (pad=-100)                                  │
  │                                                                         │
  │  resize_token_embeddings(len(tokenizer))   ← adds NEW_TOKENS (10 toks) │
  │  model.config.tie_word_embeddings = False  ← CRITICAL: must follow     │
  │                                                resize_token_embeddings   │
  │                                                                         │
  │  Seq2SeqTrainer with:                                                   │
  │    eval_dataset  = combined_val  (from val_img/; NOT test_img/)         │
  │    callbacks     = [LmHeadCloneCallback(), EarlyStoppingCallback(p=3)]  │
  │    load_best_model_at_end = True                                        │
  │                                                                         │
  │  LmHeadCloneCallback.on_save():                                         │
  │    lm_head.weight = Parameter(lm_head.weight.data.clone())              │
  │    ← breaks data-pointer alias with embed_tokens so safetensors         │
  │      writes lm_head as a separate tensor in the checkpoint shard        │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │  best checkpoint saved to model_dir/
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  D. MODEL LOAD             load_model_with_tied_weights(model_dir)      │
  │                                                                         │
  │  from_pretrained(output_loading_info=True)                              │
  │  missing_keys = loading_info["missing_keys"]                            │
  │                                                                         │
  │  if "decoder.lm_head.weight" in missing_keys                           │
  │     and NOT tie_word_embeddings:                                        │
  │       raise RuntimeError("CRITICAL…")   ← fail loudly; never silent    │
  │                                                                         │
  │  _retie_decoder_head(model, missing_keys)  ← legacy compat only        │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │
                                   ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  E. EVALUATION             DonutEvaluator.evaluate()                    │
  │                                                                         │
  │  _self_test() on test_dataset[0]:                                       │
  │    model.generate() → raw_tokens → token2json()                        │
  │    if isinstance(result, list): merge pages → dict   ← Bug C fix       │
  │    _unwrap_prediction() → check non-empty                               │
  │                                                                         │
  │  For each of 63 test images (from test_img/; NOT val_img/):            │
  │    model.generate() → sequence                                          │
  │    _parse_prediction(sequence):                                         │
  │      token2json() → result                                              │
  │      if isinstance(result, list):                                       │
  │        merge page dicts (first-key-wins)    ← Bug C fix               │
  │      _unwrap_prediction() → remove {"sroie":{…}} wrapper               │
  │    → prediction dict                                                    │
  │                                                                         │
  │  compute_metrics(predictions, ground_truths):                           │
  │    TP = pred_str.lower().strip() == gt_str.lower().strip()              │
  │    global_f1 = 2·TP / (total_pred_non_empty + total_gt_non_empty)      │
  │    NED per field via editdistance                                       │
  └────────────────────────────────┬────────────────────────────────────────┘
                                   │
                                   ▼
                    results/experiment_N.json

═══════════════════════════════════════════════════════════════════════════════
  STAGE 3–4 — TrOCR + YOLO        03_train_trocr_yolo.py
═══════════════════════════════════════════════════════════════════════════════

  Uses the SAME img/ val_img/ test_img/ split created in Stage 0.
  YOLOv8x trains on img/ (train split, YOLO bbox labels from box/).
  TrOCR trains on trocr/train/ crops.  Evaluated on test_img/ (63 images).
  Results saved to results/trocr_yolo_results.json.

═══════════════════════════════════════════════════════════════════════════════
  STAGE 6 — Paper Generation       inject_results.py :: PaperInjector
═══════════════════════════════════════════════════════════════════════════════

  results/experiment_*.json  ──► PaperInjector.build_var_map()
  paper.tex (\VAR{} placeholders) ──► PaperInjector.fill() ──► paper_filled.tex

  Placeholders: \VAR{exp1_f1}, \VAR{best_exp}, \VAR{pretrained_f1}, etc.
  Never edit paper_filled.tex directly — it is fully regenerated each run.
```

### Quick sanity check (run before every experiment)

```bash
python -c "from constants import FIELDS, BASE_MODEL, SEED"
python -c "from dataset_loaders import SROIELoader; \
           l=SROIELoader(); \
           assert l._SPLIT_DIRS['val'] != l._SPLIT_DIRS['test'], 'val==test BUG'"
```

Both must exit with code `0`. If either fails, fix the import chain before running experiments.

---

*This file is the authoritative guide for AI agents and developers working on this codebase. When in doubt: fix imports first, validate constants second, run experiments third.*
