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
18. [Pipeline Stage Summary](#18-pipeline-stage-summary)
19. [Runtime Diagnostics — `diagnostics.py`](#19-runtime-diagnostics--diagnosticspy)
20. [Guardrail Principles — What Must Never Be Done](#20-guardrail-principles--what-must-never-be-done)

---

## 1. Project Overview

This repository implements a systematic study of **multi-dataset fine-tuning for receipt Key Information Extraction (KIE)** using two model architectures:

1. **DONUT** (Document Understanding Transformer) — end-to-end vision-language model, `naver-clova-ix/donut-base` as base checkpoint
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
- All loaders extend the ABC base class in `data_pipeline.py`

### Stage 2 — DONUT Swin Encoder

- DONUT uses a **Swin Transformer** as its vision backbone
- The image is split into non-overlapping 4×4 patches
- Patch tokens pass through 4 Swin stages with window-based self-attention
- The encoder output is a 2-D feature map (H/32 × W/32) that is flattened and projected into the decoder's cross-attention space
- **Base checkpoint:** `naver-clova-ix/donut-base` — clean base, no CORD task-specific prior that could compete with SROIE tokens

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

The `DonutEvaluator` in `run_experiments.py` runs inference on the 63-sample SROIE test set, computes global F1 and NED per field, and writes `results/experiment_N.json`.

---

## 3. Fine-Tuning Method & Hyperparameter Optimization

### Method: Full-Parameter Seq2Seq (not LoRA)

All encoder and decoder parameters are updated during fine-tuning. This is appropriate because:
- The base checkpoint (`donut-base`) has no task-specific priors competing with SROIE tokens
- The SROIE vocabulary extension requires genuine embedding matrix updates
- Dataset sizes (500–3,940 samples) are sufficient to avoid catastrophic forgetting at low LR

### Hyperparameter Table

| Hyperparameter | Value / Strategy |
|---|---|
| Optimizer | AdamW (`weight_decay=0.01`) |
| Learning rate | `5e-5` encoder / `1e-4` decoder (layerwise LR) |
| LR scheduler | Cosine, warmup = `min(500, max(10, total_steps // 10))` — adaptive cap in `train.py` |
| Label smoothing | `0.1` — reduces overconfident predictions on rare SROIE tokens |
| Epochs | 10 (Exps 1–4), 15 (Exps 5–8, configurable via `ExperimentConfig`) |
| Batch size | 8 per GPU, gradient accumulation = 2 → effective batch 16 |
| Max decode length | `MAX_LENGTH = 768` tokens |
| Mixed precision | `bf16` on Ampere/Ada, `fp16` fallback, `fp32` on CPU |
| Early stopping | Patience = 5 on val loss |
| Seed | `SEED = 42` (set globally at pipeline start) |

> **Warmup note:** `ExperimentConfig.warmup_steps = 500`. `DonutTrainer.train()` automatically caps this to ≤10% of total optimizer steps (minimum 10) so it is safe for all dataset sizes (Exp 1 with ~320 total steps → effective warmup ≈ 32).

The optimal settings give Global F1 ≈ 0.8503 on Exp 1 (SROIE baseline), 0.8982 on Exp 6 (BEST).
Use `python reporting.py` to regenerate loss curves and F1 comparison plots.

---

## 4. Training Time Estimates (Measured 2026-03-07 on Vast.ai RTX 6000 Blackwell 96 GB)

| Experiment | Samples | Measured Time | Notes |
|---|---|---|---|
| Exp 2 (+ WildReceipt) | 1,386 | ~42 min (2511 s) | Actual training_time_sec from JSON |
| Exp 3 (+ Invoices) | 832 | — | Not measured (parse failure run) |
| Exp 4 (+ WR+Inv) | 1,718 | — | Not measured |
| Exp 5 (+ WR 2×) | 1,886 | — | Not measured |
| **Exp 6 (+ Inv 2×)** | **1,332** | **~39.6 min (2374 s)** | **BEST** |
| Exp 7 (+ All 2×) | 2,218 | ~84 min (5061 s) | Actual |
| Exp 8 (All 3×) | ~3,940 | OOM | OOM at 2560×1920 image resolution |

**Reference GPU estimates (other hardware):**
| GPU | Exp 6 est. | Full 7-exp suite est. |
|---|---|---|
| A100 (80 GB) | ~25 min | ~4–5 h |
| RTX 4090 (24 GB) ← current workspace GPU | ~60 min | ~10–12 h |
| V100 (16 GB) | ~90 min | ~15–20 h |

The RTX 4090 (24 GB) row matches the current workspace GPU (`GPU 0 (NVIDIA GeForce RTX 4090): 0.0/24.0 GB used`). The Vast.ai RTX 6000 Blackwell figures above are the authoritative reference run; RTX 4090 figures are estimates only.

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
| `JSONDecodeError` in `reporting.py` | Trailing comma or missing field in results JSON | Validate JSON against results format spec below |
| `CUDA out of memory` | Batch size too large for VRAM | Halve `batch_size`; double `gradient_accumulation_steps` |
| **`OutOfMemoryError` in `train_trocr()` after DONUT experiments** | Prior stages leaked GPU memory; no defensive cleanup at TrOCR/YOLO start | Add `_gpu_cleanup()` at start of `train_trocr()` and `train_yolo()`; add VRAM-aware batch auto-scaling |
| `KeyError: 'sroie'` in evaluator | `{"sroie": {...}}` wrapper not unwrapped | Unwrap in `run_experiments.py`: `result = result.get("sroie", result)` |
| **`F1 ≈ 0.008`** (not zero, not 0.42) | `token2json` returned list (CORD `<sep/>` drift) | `_parse_prediction()` merges page-list → dict (Pattern 5 below) |
| **`F1 ≈ 0.42`** (not zero, plausible-looking) | `lm_head.weight` dropped by safetensors dedup | `LmHeadCloneCallback` + `RuntimeError` check on load (Pattern 6 below) |
| **`RuntimeError: CRITICAL: decoder.lm_head.weight missing`** | `LmHeadCloneCallback` failed or was removed | Re-register callback in `DonutTrainer.train()`; do NOT remove the check |
| **`Self-test FAILED: model produced empty dict`** | `token2json` returned list; self-test treated list as empty | `_self_test()` merges list before `_unwrap_prediction()` (Pattern 5 below) |
| **Experiment results inconsistent across runs** | Mutable global `EXPERIMENTS` dict mutated by `run_experiment()` | Use `dataclasses.replace()` — never assign to `EXPERIMENTS[N].field` |
| **`ruff check .` fails in CI** | pre-commit hook not installed; `lint.sh` not run before push | Run `lint.sh` before every `git commit` (see § 10 Mandatory Lint) |
| **`FATAL: The following packages could not be installed: editdistance`** | `_CRITICAL_INSTALL_PACKAGES` still contains `editdistance` which was removed from `requirements.txt` (replaced with inline `_edit_distance()`) | Remove `editdistance` and `pandas` from both `_CRITICAL_INSTALL_PACKAGES` and `_CRITICAL_VERIFY_PACKAGES` in `run_all.py` |

### Pattern 1: Bracket & Comma Errors

Most common location: `run_experiments.py` (the `ExperimentConfig` list) and `data_pipeline.py` (ABC method signatures).

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

The `transformers ≥4.47` relocation of `PreTrainedTokenizerBase` must be preserved on every merge into `data_pipeline.py`:

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
2. `transformers` compat shim in `data_pipeline.py` still present
3. `model.config.tie_word_embeddings = False` still set in `train.py`
4. `FIELDS` / `IMAGE_EXTS` imported from `constants.py` (not redeclared)
5. `results/experiment_N.json` schema matches the evaluator's output keys

---

## 6. Repository Structure

```
kaggle/
├── constants.py              # SINGLE SOURCE OF TRUTH for all shared constants + logging utils + project metadata
├── data_pipeline.py          # Merged: dataset_loaders + dataset_normalizer + preprocess_seller_split + dataset_preparation
├── run_experiments.py        # Merged: experiment_config + evaluation + 8-experiment DONUT orchestrator
├── reporting.py              # Merged: benchmark_compare + plot_convergence + inject_results
├── validation.py             # Merged: validators + preflight_checks + startup_diagnostics
├── cloud_orchestration.py    # Merged: pipeline_types + pipeline_critic + dag_scheduler + cloud_pipeline
├── sweep.py                  # Merged: hparam_search + multi_seed_runner
├── resource_manager.py       # Merged: memory_manager + resource_optimizer
│
├── train.py                  # DonutTrainer OOP wrapper + LiveDashboardCallback (inlined)
├── run_all.py                # MAIN ENTRY POINT: full dual-architecture pipeline
├── train_trocr_yolo.py       # TrOCR+YOLO training (also called by run_all.py)
│
├── requirements.txt          # Python dependencies with version pins (2 required: torch, transformers)
├── ruff.toml                 # Linter/formatter config (moved from pyproject.toml)
├── lint.sh                   # Run before every commit: ruff check --fix . && ruff format .
│
├── paper/                    # LaTeX source files
│   ├── paper.tex             # Main paper template with \VAR{} placeholders
│   ├── references.bib        # BibTeX references
│   └── presentation.tex      # Presentation slides template
│
├── experiments/              # YAML experiment definition files (exp_01_*.yaml … exp_17_*.yaml)
│
├── results/                  # Runtime: per-experiment JSON (gitignored)
├── data/                     # Runtime: dataset cache (gitignored)
├── models/                   # Runtime: model checkpoints (gitignored)
└── paper/paper_filled.tex    # Runtime: generated paper output (gitignored)
```

**Canonical entry point:** `run_all.py`.

**Consolidation history (all shim files deleted):**
- `dataset_loaders.py`, `dataset_normalizer.py`, `dataset_preparation.py` → merged into `data_pipeline.py`
- `evaluation.py`, `experiment_config.py` → merged into `run_experiments.py`
- `logging_utils.py` → merged into `constants.py`
- `pyproject.toml` project metadata → merged into `constants.py` as `PROJECT_*` constants
- `pipeline_critic.py`, `validators.py`, `control_suite.py`, `memory_manager.py`, `resource_optimizer.py` → all were re-export shims, deleted
- `tests/` directory, `.github/` directory, `__main__.py` → deleted
- `live_dashboard.py` → inlined into `train.py` as `LiveDashboardCallback`
- `paper_diff.py` → inlined into `reporting.py`

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
| `BASE_MODEL` | `"naver-clova-ix/donut-base"` | Base DONUT checkpoint |
| `SEED` | `42` | Global random seed |
| `NEW_TOKENS` | `["<s_sroie>", "<s_company>", ...]` | SROIE special tokens added to tokenizer |
| `EMPTY_GT` | `{"company": "", "date": "", ...}` | Empty ground-truth template |

> **Duplication is a root cause.** `FIELDS` and `IMAGE_EXTS` were historically duplicated in 5+ files, causing silent divergence bugs. The consolidation into `constants.py` was a deliberate architectural fix — do not undo it.

---

## 8. Experiment Definitions

8 DONUT fine-tuning experiments with different dataset combinations. All use 80/10/10 SROIE split: **500 train / 63 val / 63 test**. Final runs completed 2026-03-07 on Vast.ai RTX 6000 Blackwell (96 GB).

| Exp | Training Data | Samples | **Actual F1** | Notes |
|---|---|---|---|---|
| 1 | SROIE only (baseline) | 500 | **0.8503** | 5-epoch quick run (post-bug-fix). company=0.841, date=0.984, address=0.790, total=0.784 |
| 2 | SROIE + WildReceipt | 1,386 | **0.8257** | WR alone (no oversample) slightly hurts vs baseline |
| 3 | SROIE + Invoices-DONUT | 832 | **0.2867** | Cross-domain hurts severely without rebalancing |
| 4 | SROIE + WR + Invoices | 1,718 | **0.8224** | Combined unbalanced — still below baseline |
| 5 | SROIE + WR (2× SROIE) | 1,886 | **0.8514** | Marginal gain with oversampling |
| **6** | **SROIE + Invoices (2× SROIE)** | **1,332** | **0.8982 ← BEST** | Early stop ep.8, 39.6 min |
| 7 | SROIE + All (2× SROIE) | 2,218 | **0.8503** | Matches baseline (competing signals cancel) |
| 8 | SROIE + WR + Invoices (3× SROIE) | ~3,940 | **OOM** | OOM at 2560×1920; fixed in resource_manager.py |

**Key insight:** SROIE oversampling (2×) is a prerequisite for auxiliary data to help. Without it, Exps 2–4 all score at or below the baseline.

**Known: Exp 1 company F1 at 10 epochs is a convergence failure, not the SROIE-only ceiling.** The CORD-pretrained encoder needs more than 10 epochs on 500 samples to converge on Southeast Asian merchant names (short, capitalised, mixed English–Malay). Experiments 5–7 ran more epochs, so the reported gain of +0.0479 over Exp 1 conflates auxiliary-data benefit with extended-training benefit. A controlled 15-epoch Exp 1 run would establish a fairer baseline before drawing magnitude claims.

**Exp 6 per-field:** company=0.9048, date=0.9841, address=0.7903, total=0.9120, exact_match=0.6667

**Gain over baseline:** Exp 6 (0.8982) − Exp 1 (0.8503) = **+0.0479**; vs published DONUT (0.8411) = **+0.0571**

**TrOCR+YOLO (trained):** global_f1=0.2035, company=0.176, date=0.460, address=0.000, total=0.231

**Gap:** DONUT best (0.8982) vs TrOCR+YOLO (0.2035) = **+69.5% absolute**

`ExperimentConfig` is the single source of truth for all hyperparameters. `DonutTrainer` reads all values via duck-typed attribute access — no magic numbers anywhere else.

---

## 9. Dataset Sources

| Dataset | Source | Notes |
|---|---|---|
| SROIE | `https://github.com/zzzDavid/ICDAR-2019-SROIE.git` | Auto-cloned; 80/10/10 split applied |
| WildReceipt | `https://download.openmmlab.com/mmocr/data/wildreceipt.tar` | OpenMMLab tar |
| Invoices-DONUT | HuggingFace `katanaml-org/invoices-donut-data-v1` | HF token for 5–10× faster download |

**HF Token:** Place your token in `hf_token.txt` (single line). Gitignored — never commit. Enables parallel accelerated downloads.

**Dataset version pinning:** If a dataset repo updates and breaks the pipeline, pin to a known-good commit via env vars:
```bash
export HF_FUNSD_REVISION=<commit-sha>       # nielsr/funsd
export HF_INVOICES_REVISION=<commit-sha>    # katanaml-org/invoices-donut-data-v1
export HF_CORD_REVISION=<commit-sha>        # naver-clova-ix/cord-v2
```
Delete the dataset's `.done` marker file to force re-download with the new revision.

---

## 10. Development Workflows

### Full Pipeline (recommended)

**requirements.txt is now minimal (2 packages: torch + transformers).**
All other dependencies are inlined:
- `datasets` → `_hf_download_dataset_inline()` in `data_pipeline.py`
- `ultralytics` → `_YOLO_CLS` in `train_trocr_yolo.py`
- `accelerate` → never used; `torch.cuda.amp.GradScaler` directly
- `matplotlib` → `_svg_bar_chart/_svg_radar_chart/...` in `reporting.py`
- `pytest` → `_Pytest` stub in `run_all.py` (tests/ directory removed)
- `Pillow` → `_load_png/_load_bmp/_load_jpeg_pure` in `train.py`

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
python inject_results.py --all --paper paper/paper.tex --output paper/paper_filled.tex
```

### Paper PDF Compilation

`stage_paper()` in `run_all.py` automatically compiles both filled `.tex` files to PDF if a LaTeX compiler is found:
- Tries `pdflatex` → `latexmk` → `xelatex` in order
- Runs each compiler **twice** to resolve `\ref` / `\cite` cross-references
- Outputs: `paper/paper_filled.pdf` and `paper/presentation_filled.pdf`
- Falls back gracefully with an install hint if no compiler is present

Install a compiler: `apt-get install texlive-latex-base texlive-fonts-recommended` (Linux)
or download [MiKTeX](https://miktex.org/) (Windows/macOS).

```python
# Programmatic usage from Python:
from reporting import compile_pdf
pdf = compile_pdf("paper/paper_filled.tex")   # returns Path or None
```

### Console Lab (LiveDashboardCallback)

When `rich` is installed, a persistent in-place table is rendered during training:

```
┌────────────────────────────────────────────────┐
│  Exp 6 — Training [8/15]                       │
├────┬──────────┬──────────┬────────────┬────────┤
│ Ep │ Train ↓  │ Val ↓    │  Best F1 ↑ │  Δ F1  │
├────┼──────────┼──────────┼────────────┼────────┤
│  1 │   2.1234 │   1.8901 │     0.4120 │     —  │
│  2 │   1.4210 │   1.2345 │     0.6310 │+0.2190 │
│  8 │   0.3891 │   0.4102 │     0.8982 │+0.0120 │
└────┴──────────┴──────────┴────────────┴────────┘
```

The table updates in-place after every epoch. Disable with `DISABLE_LIVE_DASHBOARD=1`.

### Alternative Standalone Workflow

```bash
python dataset_preparation.py
python train_donut.py
python train_trocr_yolo.py
python evaluate_models.py
```

### Exit Codes (`run_all.py`)

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | One or more experiments had no training data (partial results saved) |
| `2` | Fatal error (missing SROIE data, unrecoverable failure) |

### Mandatory: Run lint before every commit

CI runs `ruff check .` + `ruff format --check .` — both fail on any unformatted code.
**Always run before `git commit`:**

```bash
lint.sh          # auto-fixes + formats everything (preferred)
# or equivalently:
ruff check --fix . && ruff format .
```

Install the pre-commit hook once to make this automatic:

```bash
pip install pre-commit && pre-commit install
```

**Common ruff violations that break CI:**

| Error | Rule | Cause | Fix |
|---|---|---|---|
| Import not at top of file | E402 | Intentional late import | Add `# noqa: E402, I001` (see Pattern 7) |
| Unused import | F401 | Removed dependency still imported | Delete the import line |
| Import order wrong | I001 | Added import in wrong block | Run `lint.sh` |
| f-string without placeholder | F541 | `f"plain string"` | Remove `f` prefix |
| Comparison to None | E711 | `x == None` | Use `x is None` |
| Old-style type hint | UP006/UP007 | `Optional[X]`, `List[X]` | Use `X \| None`, `list[X]` |

---

## 11. OOP Design Patterns

| Class | File | Responsibility |
|---|---|---|
| `SROIELoader`, `WildReceiptLoader`, etc. | `data_pipeline.py` | ABC hierarchy; all return `List[Tuple[image_path, gt_dict]]` |
| `DonutTrainer` | `train.py` | Wraps `Seq2SeqTrainer`; reads all hyperparams from `ExperimentConfig` |
| `DonutEvaluator` | `run_experiments.py` | Computes global F1, NED; unwraps `{"sroie": {...}}` token2json output |
| `PipelineOrchestrator` | `run_all.py` | Sequential GPU stage runner; uses `StageResult` dataclass |
| `PaperInjector` | `reporting.py` | Resolves `\VAR{}` placeholders from JSON results |
| `ExperimentConfig` | `run_experiments.py` | Dataclass; single source of truth for all hyperparameters |

**Sequential execution:** All pipeline stages run sequentially to prevent GPU memory contention. GPU memory freed between stages: `torch.cuda.empty_cache()` + `gc.collect()`.

---

## 12. Evaluation Metrics

SROIE Task-3 **global F1** over all `(image, field)` pairs:

- A pair is **TP** if `predicted_string == ground_truth_string` (case-insensitive, stripped)
- **NED** (Normalized Edit Distance, inline Wagner-Fischer `_edit_distance()`) reported per field — lower is better ↓
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

`reporting.py` reads these files and resolves `\VAR{variable_name}` placeholders in `paper/paper.tex`. Never edit `paper/paper_filled.tex` directly — it is fully regenerated each run.

---

## 14. TrOCR + YOLO Architecture

Two-stage pipeline in `train_trocr_yolo.py` (called by `run_all.py`):

**Stage 1 — YOLOv8x detection**
- Model: `yolov8x.pt`
- Detects text regions as bounding boxes on receipt images
- Config: `YOLO_EPOCHS=50`, `YOLO_IMG_SIZE=512`, `YOLO_BATCH=8`, `YOLO_AMP=True`
- Training data YAML: `data/yolo/dataset.yaml`

**Stage 2 — TrOCR reading**
- Model: `microsoft/trocr-base-printed`
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
| Paper template | `paper/paper.tex` | `--paper-template F` |
| Paper output | `paper/paper_filled.tex` | `--output F` |

**Gitignored runtime artifacts** (never commit):
- `results/` — experiment JSON outputs
- `data/` — dataset cache
- `models/` — checkpoints (`*.pt`, `*.pth`, `*.pkl`, `*.h5`)
- `paper/paper_filled.tex` — generated LaTeX paper
- `hf_token.txt` — HuggingFace authentication token
- `github_token.txt` — GitHub personal access token (for diagnostics issue creation)

---

## 16. Known Issues & Historical Fixes

| Issue | Fix Applied | File |
|---|---|---|
| `protobuf` missing → 100% pipeline crash | Added `protobuf>=3.20.0` to `requirements.txt` | `requirements.txt` |
| `lm_head` weight tying → F1=0 on reload | `config.tie_word_embeddings=False` after `resize_token_embeddings()` | `train.py` |
| Key file loading failure | Try `.txt` first, then `.json` (BUG A/E fix) | `train.py` |
| Data leakage in eval | Separate `val_img/` and `test_img/` directories in `stage_install()` | `run_all.py` |
| `{"sroie": {...}}` wrapper in token2json | Unwrapped in evaluator | `run_experiments.py` |
| FIELDS/IMAGE_EXTS duplicated in 5+ files | Consolidated in `constants.py` | `constants.py` |
| `transformers ≥4.47` `PreTrainedTokenizerBase` move | Compat shim added | `data_pipeline.py` |
| **safetensors deduplication drops `lm_head.weight` → F1~0.42** | `LmHeadCloneCallback` deep-clones weight before every save; sanity `RuntimeError` on load | `train.py`, `run_experiments.py` |
| **`token2json` returns list (`<sep/>` tokens) → F1=0.0078** | `_parse_prediction()` and `_self_test()` merge page-list into flat dict | `run_experiments.py` |
| **val split missing → no early stopping guard** | `stage_install()` creates `val_img/`+`val_key/` distinct from `test_img/`+`test_key/` | `run_all.py` |
| **mutable global `EXPERIMENTS` dict corrupted by `run_experiment()`** | `dataclasses.replace()` creates isolated copy; `_config_to_dict()` records actual training params | `run_experiments.py` |
| **TrOCR OOM after DONUT experiments** | Prior stages' GPU memory not freed before TrOCR model load | Added defensive `_gpu_cleanup()` at start of `train_trocr()` and `train_yolo()`; explicit cleanup between YOLO and TrOCR in `stage_trocr_experiments()`; VRAM-aware batch auto-scaling halves batch when free VRAM < needed | `train_trocr_yolo.py`, `run_all.py` |
| **`flash-attn` build fails with nvcc segfault on torch≥2.9+cu126** | Pipeline already falls back to PyTorch SDPA — no action needed. If you want FA2: use prebuilt wheel from https://flashattn.dev/wheel-finder/ (Python 3.12 / CUDA 12.6). Fixed `run_experiments.py` to catch `RuntimeError` in addition to `ImportError`. | `run_experiments.py`, `validation.py` |
| **Terminal freeze after "Dependencies installed successfully" on GPU machine** | `_install_dependencies()` ran `subprocess.run(..., capture_output=True)` for flash-attn after the main install — CUDA kernel compilation takes 5–25 min, invisible; `Ctrl+C` didn't reach nvcc child. Post-install `_verify_critical_packages()` called `sys.exit(2)` because newly-installed packages aren't visible to the running process (`sys.modules` isolation). | Removed flash-attn auto-build entirely. Added `os.execv()` restart (with `_DONUT_RESTARTED=1` sentinel) after successful pip install. Added `_InstallWatchdog` thread for elapsed-time progress. Added `timeout=300` to main pip subprocess. | `run_all.py` |
| **`FATAL: editdistance could not be installed` after dependency auto-install** | `editdistance` and `pandas` still in `_CRITICAL_INSTALL_PACKAGES`/`_CRITICAL_VERIFY_PACKAGES` after being removed from `requirements.txt` (both replaced with inline implementations) | Removed both from `_CRITICAL_INSTALL_PACKAGES` and `_CRITICAL_VERIFY_PACKAGES` in `run_all.py` | `run_all.py` |

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

## 18. Pipeline Stage Summary

`run_all.py` executes these stages in order:

| Stage | Function | Output |
|---|---|---|
| 0 | `stage_install()` | SROIE 80/10/10 split: `img/`, `val_img/`, `test_img/` |
| 1 | `stage_download()` | Aux datasets + base model in HF cache |
| 1.5 | `stage_pretrained_baseline()` | `results/evaluation_results.json` (zero-shot F1) |
| 2 | `run_experiments.run_experiment(N)` × 8 | `results/experiment_N.json` each |
| 3-4 | `train_trocr_yolo.py` | `results/trocr_yolo_results.json` |
| 5 | `benchmark_compare.main()` | `results/benchmark_results.json` + plots |
| 6 | `inject_results.PaperInjector.fill()` | `paper/paper_filled.tex` |

**Critical SROIE split invariant:** `val_img/` and `test_img/` are physically separate directories. `load_sroie_val()` and `load_sroie_test()` must never return overlapping images.

**Training loop invariant:** `resize_token_embeddings()` must always be followed by `model.config.tie_word_embeddings = False` and the `LmHeadCloneCallback` must be registered — see Section 16 bug patterns.

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

---

## 19. Runtime Diagnostics — `diagnostics.py`

`diagnostics.py` is an always-on, self-contained diagnostic layer. It requires no API keys to run. AI-powered diagnosis is opt-in via environment variables.

### Architecture

| Component | Class / Function | When it runs |
|---|---|---|
| Per-epoch pattern detection | `DiagnosticCallback` (TrainerCallback) | After every `on_evaluate` during training |
| Post-evaluation F1 check | inline in `run_experiment()` | After `evaluate_experiment()` returns |
| Stage-level error capture | `PipelineDiagnostics` in `PipelineOrchestrator` | After every stage in `run_all.py` |
| AI root-cause analysis | `ai_diagnose()` | On demand: triggered by critical pattern or stage failure when `AI_DIAGNOSE=1` |
| Pre-run validation | `smoke_test()` / `run_preflight()` | Called from CLI: `python diagnostics.py --smoke-test` |

### Known patterns detected automatically

| Pattern name | F1 signature | Root cause | Fix pointer |
|---|---|---|---|
| `lm_head_dedup` | 0.40 < F1 < 0.44 | safetensors omits `lm_head.weight` | `LmHeadCloneCallback` + `tie_word_embeddings=False` (Pattern 6) |
| `token2json_list` | F1 < 0.02 | `token2json` returned list (`<sep/>` tokens) | `_parse_prediction()` list merge (Pattern 5) |
| `total_f1_collapse` | F1 = 0.000 | Wrong `decoder_start_token_id` or missing tokens | Use list form in `convert_tokens_to_ids` (GP-3) |
| `loss_plateau` | train_loss > 2.0 at epoch > 3 | Fewer than 200 optimizer steps | `validate_training_config()` (GP-2) |
| `loss_nan` | train_loss = NaN | fp16 overflow or bad LR | Switch to bf16; clip gradients |
| `gpu_oom_warning` | allocated / reserved > 0.90 | Batch too large for VRAM | Halve batch_size, double grad_accum |
| `eval_loss_diverging` | 3 consecutive eval loss increases | Overfitting | Increase `weight_decay`; rely on early stopping |

### Running diagnostics

```bash
# Smoke test — validates import chain, model load, token roundtrip, forward pass (<30s)
python diagnostics.py --smoke-test

# Smoke test + AI diagnosis on failure
AI_DIAGNOSE=1 python diagnostics.py --smoke-test --ai-diagnose --ai-provider auto

# DiagnosticCallback fires automatically during training — no flag needed
python run_experiments.py --experiment 6

# Disable pattern detection entirely
DISABLE_DIAGNOSTICS=1 python run_all.py
```

### Enabling AI-powered diagnosis

```bash
# Claude (claude-sonnet-4-5-20251001 default)
export ANTHROPIC_API_KEY=sk-ant-...
AI_DIAGNOSE=1 python run_all.py --experiment 3

# Mistral (mistral-small-latest default; SDK or raw HTTP fallback)
export MISTRAL_API_KEY=...
AI_DIAGNOSE=1 AI_DIAGNOSE_PROVIDER=mistral python run_all.py

# Auto: tries Claude → Mistral SDK → Mistral HTTP (no SDK required for HTTP fallback)
AI_DIAGNOSE=1 AI_DIAGNOSE_PROVIDER=auto python run_all.py
```

Keys can also be stored in gitignored flat files:
```
anthropic_api_key.txt   # single line: sk-ant-...
mistral_api_key.txt     # single line: ...
```

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `AI_DIAGNOSE` | `0` | `1` enables AI API calls on critical failures |
| `AI_DIAGNOSE_PROVIDER` | `auto` | `claude` / `mistral` / `auto` |
| `ANTHROPIC_API_KEY` | — | Claude key (or `anthropic_api_key.txt`) |
| `MISTRAL_API_KEY` | — | Mistral key (or `mistral_api_key.txt`) |
| `GITHUB_TOKEN` | — | GitHub PAT with `repo` scope (or `github_token.txt`); enables auto-notify |
| `GITHUB_REPO` | — | Target repo slug, e.g. `aiparallel0/kaggle`; required for GitHub notify |
| `DISABLE_DIAGNOSTICS` | `0` | `1` skips `DiagnosticCallback` entirely |

### Output files

| File | When created | Content |
|---|---|---|
| `results/diagnostics_expN.json` | After each training run | Per-epoch checkpoints, issues, AI diagnoses |
| `results/pipeline_diagnostics.json` | After `run_all.py` completes | Per-stage status, GPU snapshots, AI diagnoses |

### Claude Code autonomous validation hook

`.claude/settings.json` registers a `PostToolUse` hook on `Edit|Write` that runs:

```bash
python -c "from constants import FIELDS, BASE_MODEL, SEED"
```

after every `.py` file edit in this session. If the import breaks, a warning is injected into the assistant context immediately — before any experiment runs. This satisfies the CLAUDE.md §5 rule automatically without requiring a manual check.

To disable the hook for a session: `DISABLE_DIAGNOSTICS=1` (does not affect the hook — the hook is controlled via `.claude/settings.json`).

### Programmatic API

```python
from diagnostics import (
    DiagnosticCallback,   # TrainerCallback — add to HF Trainer callbacks list
    PipelineDiagnostics,  # Stage wrapper — use in PipelineOrchestrator
    ai_diagnose,          # Direct AI call: ai_diagnose(context, provider="auto")
    RuntimeCheckpoint,    # Telemetry dataclass
    smoke_test,           # Returns True/False; prints pass/fail per check
    run_preflight,        # smoke_test() + optional AI diagnosis on failure
)

# Minimal usage in a custom training loop:
from diagnostics import DiagnosticCallback
cb = DiagnosticCallback(experiment_id=6, ai_diagnose=True)
trainer = Seq2SeqTrainer(..., callbacks=[..., cb])

# Direct AI call (no training needed):
from diagnostics import ai_diagnose
diagnosis = ai_diagnose(
    {"eval_f1": 0.42, "epoch": 7, "stage": "training"},
    provider="auto",
)
```

### Integration points (for AI agents modifying this codebase)

- **`train.py`:** `DiagnosticCallback` is registered in `DonutTrainer.train()` after `LiveDashboardCallback`. Do not remove it. It must come after `LmHeadCloneCallback`.
- **`run_experiments.py`:** Post-evaluation block (after `evaluate_experiment()`) checks F1 against known-bad ranges. Adding new known-bad F1 ranges: add to the `if/elif/elif` chain, not to `_BUG_PATTERNS` (those are for training-time checks).
- **`run_all.py`:** `PipelineOrchestrator.__init__()` creates `self._diag`. The `_run_stage()` method records each stage result. `run()` calls `self._diag.save_report()` at pipeline end.
- **`diagnostics.py`:** `_BUG_PATTERNS` is a list of dicts with `name`, `description`, `detect` (lambda), `severity`, `fix`. Add new patterns here for training-time detection. The `detect` lambda receives a `RuntimeCheckpoint` and must not raise.

---

## Autonomous CI/CD Pipeline

> **For developers:** This pipeline runs fully automatically after each commit. Tests run, AI evaluates results, PR is updated with findings, and on success the PR auto-merges. Setup requires two external API tokens — see "Token Setup" below.

### How It Works

1. **Commit detection** — After a successful `git commit`, a hook in `.claude/settings.json` fires automatically
2. **Test execution** — `autonomous_ci.py` runs:
   - Import chain validation
   - Linting check (`ruff check`)
   - Smoke test (`diagnostics.py --smoke-test`)
3. **AI evaluation** — Test results sent to Claude or Mistral API for verdict: `MERGE` / `BLOCK` / `COMMENT_ONLY`
4. **GitHub posting** — Results posted as a comment on the PR
5. **Auto-merge** — If all tests pass and AI verdict is `MERGE`, PR auto-merges with squash strategy

**Time from commit to merge:** ~15–30 seconds (tests) + API call latency (2–5s) = ~20–35s total.

### Token Setup

Two tokens required for full functionality (both optional; pipeline degrades gracefully without them):

#### 1. GitHub Token (for PR auto-merge)

**Scope required:** `repo` (full repository access)

**Setup:**
```bash
# Option A: Save token to gitignored file
echo "ghp_xxxxx..." > github_token.txt   # replace with your token
# Never commit github_token.txt

# Option B: Set environment variable
export GITHUB_TOKEN="ghp_xxxxx..."

# Set target repo (required)
export GITHUB_REPO="aiparallel0/kaggle"
```

**Test connectivity:**
```bash
python diagnostics.py --github-test
# Output: [GitHub Test] PASS — issue created: https://github.com/...
```

**Create a GitHub token:**
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)" or "Generate new token (fine-grained)"
3. Select scope: `repo` (full control of private repositories)
4. Click "Generate token"
5. Copy token immediately (GitHub won't show it again)

#### 2. AI API Key (for test evaluation)

**Option A: Claude (recommended)**

```bash
echo "sk-ant-xxxxx..." > anthropic_api_key.txt
export ANTHROPIC_API_KEY="sk-ant-xxxxx..."
```

[Get Claude API key](https://console.anthropic.com/account/keys)

**Option B: Mistral**

```bash
echo "xxxxx..." > mistral_api_key.txt
export MISTRAL_API_KEY="xxxxx..."
```

[Get Mistral API key](https://console.mistral.ai/api-keys/)

**Option C: Auto** (tries Claude, falls back to Mistral HTTP, then silent mode)

If either API key is present, it will be used. Both can coexist; auto-select tries Claude first.

### Usage

#### After a commit (automatic)

```bash
git commit -m "Add my feature"
# → Hook fires automatically
# → Tests run in background
# → Results posted to PR
# → PR auto-merges if tests pass
```

Monitor progress:
```bash
# View latest evaluation
tail -f /tmp/autonomous_ci.log

# Check PR comment
gh pr view <pr-number>
```

#### Manual trigger (testing)

```bash
# Test on current branch
python autonomous_ci.py

# Test specific PR
python autonomous_ci.py --pr 42

# Test without merging
python autonomous_ci.py --no-merge

# Test and post comment, but don't merge
python autonomous_ci.py --pr-only

# Specify GitHub repo
python autonomous_ci.py --repo myorg/myrepo

# Use specific AI provider
python autonomous_ci.py --ai-provider claude

# [NEW] Auto-fix with custom attempt limit
python autonomous_ci.py --max-fix-attempts 5     # max 5 auto-fix retries
python autonomous_ci.py --max-fix-attempts 20    # max 20 attempts
python autonomous_ci.py --no-auto-fix            # disable auto-fix entirely
```

### Auto-Fix Loop (NEW — Autonomous Code Repair)

When tests fail, the pipeline **automatically generates and applies code fixes** without waiting for human intervention. This is the key innovation that makes the pipeline fully autonomous.

#### How Auto-Fix Works

```
Test fails
    ↓
AI analyzes failure (import error, syntax error, etc.)
    ↓
AI generates Python code that fixes the issue
    ↓
Code is validated (syntax check + type validation)
    ↓
Fix is applied to whitelisted files
    ↓
git commit -m "[auto-fix 1/N] Apply generated fix"
    ↓
Tests re-run automatically
    ↓
IF pass → auto-merge to main ✅
IF fail AND attempt < max → loop back to "AI analyzes failure"
IF fail AND attempt == max → stop and report (N attempts exhausted)
```

#### Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `--max-fix-attempts` | `10` | Maximum number of auto-fix retry loops |
| `--no-auto-fix` | `false` | Disable auto-fix (test-only mode) |

**N is configurable:**
```bash
# Conservative: only 3 attempts
python autonomous_ci.py --max-fix-attempts 3

# Aggressive: 20 attempts
python autonomous_ci.py --max-fix-attempts 20

# Classic mode: no auto-fix (manual review only)
python autonomous_ci.py --no-auto-fix
```

#### Whitelist of Modifiable Files

For safety, auto-fix can only modify these files:
```
✅ constants.py
✅ train.py
✅ data_pipeline.py
✅ run_experiments.py
✅ run_all.py
✅ diagnostics.py
✅ autonomous_ci.py
```

Any fix affecting files outside this list will be **rejected before applying**.

#### Example: Auto-Fix in Action

```
$ git commit -m "Add new feature"

[hook fires automatically]

$ python autonomous_ci.py

[CI] Test suite: 2/3 FAIL
[CI] AI verdict: BLOCK
[CI] Tests failed. Starting auto-fix loop...

[AutoFix] Attempt 1/10
[AutoFix] Generating fix code from AI...
[AutoFix] ✓ Generated fix for constants.py
[AutoFix] Committed fix attempt 1
[AutoFix] Re-running tests...
[CI] Test suite: 3/3 PASS ✅

[AutoFix] ✅ Tests PASSED on attempt 1!
[CI] ✅ Auto-fix succeeded! Re-evaluating...
[CI] AI verdict: MERGE
[CI] Auto-merging PR #42...
✅ PR #42 merged to main (squash)
```

#### Multi-Attempt Example (Harder Fix)

```
[AutoFix] Attempt 1/10: Fix import error
[AutoFix] Re-running tests...
[CI] Test suite: 2/3 FAIL (different error now)

[AutoFix] Attempt 2/10: Fix syntax error (from previous fix)
[AutoFix] Re-running tests...
[CI] Test suite: 2/3 FAIL (different error)

[AutoFix] Attempt 3/10: Fix missing dependency
[AutoFix] Re-running tests...
[CI] Test suite: 3/3 PASS ✅

[AutoFix] ✅ Tests PASSED on attempt 3!
```

#### GitHub PR Comments (With Fix Attempts)

When auto-fix runs, the PR comment includes a summary of all attempts:

```markdown
## Autonomous CI/CD Report
**Status:** ✅ PASS
**Duration:** 45.3s

### Auto-Fix Attempts
**Attempt 1/10:** ❌
- Fix: Applied import fix
- Error: Import still missing after fix

**Attempt 2/10:** ❌
- Fix: Applied syntax correction
- Error: Different error appeared

**Attempt 3/10:** ✅
- Fix: Fixed missing constant definition

### Test Results
| Test | Status | Duration |
|---|---|---|
| import_check | ✅ | 2.1s |
| ruff_lint | ✅ | 1.3s |
| smoke_test | ✅ | 12.5s |

### AI Evaluation
**Verdict:** `MERGE`
**Reasoning:** All tests pass after 3 auto-fix attempts. Code is ready.
```

#### Safety Guarantees

1. **Syntax validation:** Generated code is validated before committing
2. **Whitelist enforcement:** Only specific files can be modified
3. **Max attempts limit:** Prevents infinite loops (default N=10)
4. **Rollback on catastrophe:** If N attempts fail, the system stops and reports
5. **Git trail:** Each fix is a separate commit: `[auto-fix 1/10]`, `[auto-fix 2/10]`, etc.
6. **AI transparency:** Full fix attempt history posted to GitHub PR

#### When Auto-Fix Stops

Auto-fix will **stop and report failure** if:
- All N attempts exhausted and tests still failing
- AI can't generate valid Python code
- Fix would require modifying non-whitelisted files
- Git commit fails (filesystem issue)

In all cases: **PR comment is posted with full diagnostic info**, allowing human review.

#### Disable for a session

```bash
# Don't run CI hooks for this session
DISABLE_DIAGNOSTICS=1 bash

# Or set in .claude/settings.local.json (gitignored):
# { "disableAllHooks": true }
```

### Test Suite Details

| Test | What it checks | Fails if |
|---|---|---|
| `import_check` | Can import `constants`, `data_pipeline`, `diagnostics` | Any import error |
| `ruff_lint` | Code style and type hints | `ruff check .` returns non-zero |
| `smoke_test` | Model loads, tokenizer configs, forward pass | <30s test fails |

**All three must pass for `MERGE` verdict.** If any fail, verdict is `BLOCK` and PR is not merged.

### AI Evaluation Prompts

The AI evaluator receives:
```json
{
  "test_results": {
    "total_tests": 3,
    "passed_tests": 3,
    "failed_tests": 0,
    "all_passed": true,
    "summary": "3/3 tests passed",
    "tests": [...]
  },
  "evaluation_task": "Determine if build is safe to merge"
}
```

**Prompt sent to Claude/Mistral:**

> You are a CI/CD evaluation agent... Analyze the test results below and provide: 1) Pass/Fail verdict: Is the build safe to merge? 2) Root cause (if failed) 3) Recommended action: MERGE / BLOCK / COMMENT_ONLY

The AI looks for:
- All tests passing → verdict `MERGE`
- Any test failing → verdict `BLOCK`
- Warnings/issues → verdict `COMMENT_ONLY` (posts to PR, doesn't block)

### Automation Rules

**Auto-merge triggers:**
- All tests: ✅ PASS
- AI verdict: `MERGE`
- Flag `--no-merge` not set
- Flag `--pr-only` not set
- GitHub token + repo available

**Auto-merge strategy:** Squash (combines all feature branch commits into one)

**PR comment always posted if:**
- GITHUB_TOKEN + GITHUB_REPO env vars set
- PR number detected from branch

### Examples

#### Example 1: Successful commit → auto-merge

```bash
# Make a feature commit on branch `claude/my-feature`
git commit -m "Add feature X"

# → Hook fires
# → Tests: 3/3 pass ✅
# → AI: "All tests pass; code looks good"
# → Verdict: MERGE
# → PR auto-merged to main
# → Console log: ✅ PR #42 merged to main
```

#### Example 2: Failed test → blocked, PR comment posted

```bash
git commit -m "Add feature Y"

# → Tests: 2/3 pass (smoke_test fails)
# → AI: "smoke_test failure indicates model loading issue"
# → Verdict: BLOCK
# → PR NOT merged
# → GitHub PR #43 gets comment:
#   ```
#   ## Autonomous CI/CD Report
#   **Status:** ❌ FAIL
#   | Test | Status |
#   | import_check | ✅ |
#   | ruff_lint | ✅ |
#   | smoke_test | ❌ |
#   **Verdict:** `BLOCK`
#   **Reasoning:** smoke_test failure indicates...
#   ```
```

#### Example 3: Manual test without merge

```bash
python autonomous_ci.py --no-merge --branch feature/my-work

# → Tests run
# → Results logged to console
# → PR NOT merged (--no-merge flag)
# → Useful for: testing CI logic before enabling auto-merge
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Hook doesn't fire after commit | `disableAllHooks: true` in settings.json | Remove or set to `false` |
| `ImportError: no module named anthropic` | Claude SDK not installed | Not critical; falls back to no AI evaluation |
| `GitHub API call failed: 401` | Bad/expired GitHub token | `python diagnostics.py --github-test` to verify |
| `No GitHub credentials; skipping PR lookup` | GITHUB_TOKEN or GITHUB_REPO not set | Export both env vars; see "Token Setup" above |
| Tests pass but verdict is `BLOCK` | AI evaluation failed or unavailable | Check API keys; run with `--ai-provider auto` |
| PR doesn't auto-merge even on MERGE verdict | `--no-merge` or `--pr-only` flag set | Use `python autonomous_ci.py` without flags |

### Architecture Files

| File | Purpose |
|---|---|
| `autonomous_ci.py` | Test runner, AI evaluator, GitHub poster, auto-merger |
| `.claude/settings.json` | Hook config: detects `git commit`, runs `autonomous_ci.py --pr-only` async |
| `diagnostics.py` | Exports `ai_diagnose()`, GitHub API wrappers for autonomous_ci to use |

### Workflow Integration (For Developers)

No special steps needed. The pipeline is transparent:

1. **Make changes** → `git add`, `git commit`
2. **Hook fires automatically** — you don't have to do anything
3. **Watch console** (optional) → `tail -f /tmp/autonomous_ci.log`
4. **PR updates automatically** → results posted as comment
5. **Merge happens automatically** → if tests pass

If a test fails:
- **Read the PR comment** → AI explains the failure
- **Fix the issue** → make a new commit
- **Hook fires again** → tests re-run automatically
- **Repeat until merge** → once all pass, PR auto-merges

### Performance

**Typical pipeline time:**
| Stage | Time |
|---|---|
| Test suite | ~10–15s (all 3 tests) |
| AI evaluation | ~2–5s (Claude/Mistral API call) |
| GitHub posting | ~1–2s (API call) |
| Auto-merge | ~1–2s (API call) |
| **Total** | **~15–25s** |

**Parallel execution:** Hook runs `async: true` — doesn't block your terminal. You can continue working while tests run in background.

---



> These rules exist because each one corresponds to a real silent failure that was hard to diagnose. They are not style suggestions.

### GP-1 — Never Mutate `EXPERIMENTS` Global State

```python
# ❌ WRONG — mutates the global singleton
config = EXPERIMENTS[exp_id]
config.batch_size = 4  # silently corrupts all future experiments in the same process

# ✅ CORRECT — creates an isolated copy
import dataclasses
config = dataclasses.replace(EXPERIMENTS[exp_id], batch_size=4)
```

**Why:** Python dataclasses are mutable. `config = EXPERIMENTS[exp_id]` is a reference, not a copy. Any field assignment propagates back to the global dict and corrupts the cache-validity check for all subsequent experiments.

### GP-2 — Always Validate Optimizer Step Count Before Training

After any resource-optimizer override, call `validate_training_config()` from `resource_manager.py`:

```python
from resource_manager import validate_training_config
validate_training_config(
    batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    num_train_samples=len(train_samples),
    epochs=config.epochs,
)
```

**Why:** A large batch on a high-VRAM GPU can reduce total optimizer steps below DONUT's convergence threshold (~200 steps). The model will learn XML scaffolding but not field content, producing perfectly structured but completely empty predictions. This never raises an exception — it is a silent failure.

**Minimum safe steps:** 200. DONUT typically converges at ~250–300 steps for SROIE-sized datasets.

### GP-3 — Always Use List Form for `convert_tokens_to_ids`

```python
# ❌ WRONG — returns ID of '<' character, not the full token
token_id = tokenizer.convert_tokens_to_ids("<s_sroie>")

# ✅ CORRECT — returns ID of the whole special token
token_id = tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
```

**Why:** The string form of `convert_tokens_to_ids` iterates over characters, returning the ID for `<` — not the full `<s_sroie>` token. This causes the decoder to start from the wrong token, producing garbage output. Always wrap the token string in a list.

### GP-4 — Always Verify `decoder_start_token_id` Roundtrips Correctly

After setting `model.config.decoder_start_token_id`, verify the ID decodes back to the expected token:

```python
_decoded = tokenizer.decode([model.config.decoder_start_token_id])
if _decoded != "<s_sroie>":
    raise RuntimeError(
        f"decoder_start_token_id={model.config.decoder_start_token_id} decodes to "
        f"'{_decoded}', not '<s_sroie>'. "
        f"Use: tokenizer.convert_tokens_to_ids(['<s_sroie>'])[0]"
    )
```

**Why:** Silent wrong token IDs produce models that generate valid XML structure but wrong content. The roundtrip check catches misconfiguration immediately at setup time.

---

## § 9. Memory Management Rules (Added 2026-03-08)

### The Authority Module
All memory budget decisions go through `resource_manager.py`. Do not hardcode per-sample MB estimates anywhere else. Do not add `* 3` or `* 14.2` or `* 56.6` constants to any file.

### RAM Memory Map (at 1280×960 — the correct DONUT native resolution)

| What | Where | Size | When freed |
|---|---|---|---|
| DONUT model weights | GPU VRAM | ~800 MB | After `del model` + `torch.cuda.empty_cache()` |
| AdamW optimizer (m+v moments) | GPU VRAM | ~1,600 MB | After `del trainer` |
| Gradient activations | GPU VRAM | ~4,000 MB at batch=8 | After each backward pass |
| DataLoader prefetch buffers | System RAM (worker procs) | ~230 MB × n_workers | After `resource_manager.shutdown_dataloader_workers(trainer)` |
| PIL image cache (`_image_cache`) | System RAM | 3.516 MB × n_samples | After `train_ds.clear_caches()` |
| Float32 pixel cache (`_pixel_cache`) | System RAM | 14.064 MB × n_samples | After `train_ds.clear_caches()` |
| Label tensor cache (`_label_cache`) | System RAM | ~0.003 MB × n_samples | After `train_ds.clear_caches()` |
| HF Arrow mmaps (FUNSD/InvoicesDonut) | System RAM | 50–200 MB | After `resource_manager.release_hf_dataset(ds)` + `flush_hf_arrow_cache()` |
| Processor (tokenizer + image processor) | System RAM | ~100 MB | After `del processor` |
| Base model pre-load copy | System RAM | ~800 MB | After all experiments + `del _base_model` |

### The Correct Cleanup Order (per experiment)

```python
# 1. Shutdown DataLoader workers FIRST (before del trainer)
resource_manager.shutdown_dataloader_workers(trainer)
# 2. Capture log history (plain list, safe to keep)
log_history = result.log_history
# 3. Clear dataset caches in-place (empties dicts without waiting for GC)
train_ds.clear_caches()
if val_ds is not None:
    val_ds.clear_caches()
# 4. Delete all references
del trainer, model, processor, train_ds
if val_ds is not None:
    del val_ds
# 5. GPU VRAM flush
_gpu_cleanup()   # gc.collect() + torch.cuda.empty_cache()
```

### The `processor_config.json` Rule

**NEVER set `height` above 1280 or `width` above 960** for `naver-clova-ix/donut-base`.

The model was pretrained at 1280×960. Higher resolutions:
- Do NOT improve quality (out-of-distribution for pretrained weights)
- Multiply RAM per sample by `(H × W) / (1280 × 960)` — at 2560×1920 this is 4×
- All threshold constants in `resource_manager.py` (`_REF_IMAGE_SIZE`, `_VRAM_PER_SAMPLE_AT_REF_GB`) are calibrated to 1280×960

If you want to experiment with resolution: update `_REF_IMAGE_SIZE` and `_VRAM_PER_SAMPLE_AT_REF_GB` in `resource_manager.py` to match, so all threshold arithmetic stays correct.

### Val Dataset Rule (2026-03-08 fix)

**Always construct the val `MultiDataset` with `precompute_tensors=False`.**

Pixel tensor precompute on the val set is never worth the RAM cost:
- Val runs once per epoch; the per-step overhead is negligible vs training (which visits each sample many times)
- With 1332 train images already cached (Exp 6), the remaining ~4.9 GB RAM cannot absorb an additional 1902 MB val tensor cache → kernel SIGKILL
- The label tensor cache is also skipped when `precompute_tensors=False` (safe: val tokenisation runs once per epoch)

```python
# ✅ CORRECT — val dataset never precomputes tensors
val_ds = MultiDataset(
    val_samples,
    processor,
    max_length=config.max_length,
    precompute_tensors=False,   # val set never needs pixel tensor precompute
)

# ❌ WRONG — default precompute_tensors=True causes ~1902 MB val tensor allocation
val_ds = MultiDataset(val_samples, processor, max_length=config.max_length)
```

### `_RAM_SAFETY_FRACTION` Rule (2026-03-08 fix)

`_RAM_SAFETY_FRACTION` in `resource_manager.py` is **0.06** (6%), not 0.15.

The previous 15% was calibrated for a single allocation. Across sequential train→val allocations within the same experiment, the effective headroom needed is much higher. At 6%:
- Exp 6 train PIL check: 4683 MB vs 6% × 37637 MB = **2258 MB threshold → SKIP** (belt+suspenders catch that blocks the drain before the val check runs)
- 500-sample baseline: 1758 MB vs 6% × 65536 MB = 3932 MB → still ALLOW on 64 GB RAM

**Never raise `_RAM_SAFETY_FRACTION` above 0.10.**

### Adding a New Dataset

1. Drop images and annotations into `/workspace/datasets/<your_dataset>/`
2. Normalize annotations to SROIE schema: `{"company": "", "date": "", "address": "", "total": ""}`
3. Add entry to `datasets_registry.json` in repository root
4. Add loader class to `data_pipeline.py` following `BaseDatasetLoader` ABC
5. If loader uses `load_from_disk()` or `load_dataset()`: call `resource_manager.release_hf_dataset(ds)` + `resource_manager.flush_hf_arrow_cache()` after sample extraction
6. Add dataset name to relevant `ExperimentConfig.datasets` lists in `run_experiments.py`
7. No changes to `resource_manager.py`, `train.py`, `constants.py`, or `resource_manager.py`

### Why 20 Previous PRs Failed

Every OOM PR from #80 to #109 fixed GPU VRAM symptoms. The actual 192 GB RAM explosion was:
- A RAM problem (not GPU), caused by `processor_config.json` at 4× resolution (`2560×1920` instead of `1280×960`)
- Compounded by a wrong `* 3 MB/sample` constant in `train.py` (should be `3 × H × W / 1_048_576`)
- Compounded by HF Arrow cache never being released between experiments in `data_pipeline.py`
- Compounded by DataLoader worker processes staying alive across experiments (`persistent_workers=True` + `prefetch_factor=4`)

The fix is in commit adding this section. Do not revert `processor_config.json` to `2560×1920`.
