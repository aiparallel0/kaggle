---
# Fill in the fields below to create a basic custom agent for your repository.
# The Copilot CLI can be used for local testing: https://gh.io/customagents/cli
# To make this agent available, merge this file into the default repository branch.
# For format details, see: https://gh.io/customagents/config
name: donut-sroie-kie-agent
description: AI coding agent for the aiparallel0/kaggle repository — a multi-dataset fine-tuning pipeline for receipt Key Information Extraction (KIE) using DONUT (naver-clova-ix/donut-base) and TrOCR+YOLOv8 models, benchmarked on SROIE Task-3. Extracts four fields (company, date, address, total) as XML tags via Seq2Seq fine-tuning across 8 dataset-combination experiments (SROIE, WildReceipt, FUNSD, Invoices-DONUT, CORD-v2). Enforces import chain integrity (constants.py as single source of truth), tie_word_embeddings=False invariant, 960x1280 image resolution limits, GPU memory cleanup between stages, and YAML experiment config validation. Supports autonomous CI with AI-powered self-healing, cloud GPU orchestration via Vast.ai, Optuna hyperparameter sweeps, and automatic LaTeX paper generation from experiment results.
---
# my-agent.agent.md — GitHub Copilot Agent Instructions
# DONUT SROIE Multi-Dataset Receipt KIE Pipeline
---

## Identity

You are an AI coding agent working on the **aiparallel0/kaggle** repository — a multi-dataset fine-tuning pipeline for receipt Key Information Extraction (KIE) using DONUT and TrOCR+YOLOv8 models, benchmarked on SROIE Task-3.

---

## ⚠️ Rule Zero — Import Chain Integrity

**Before touching ANY code, verify the import chain is healthy:**
```bash
python -c "from constants import FIELDS, BASE_MODEL, SEED"
python -c "from data_pipeline import SROIELoader"
```

Both MUST exit with code 0. If either fails, **fix `constants.py` first** — a broken import chain silently cascades into every file in the project. No experiment, evaluation, or reporting code will function.

---

## Project Architecture

### What This Project Does

Fine-tunes **DONUT** (`naver-clova-ix/donut-base`) and evaluates **TrOCR + YOLOv8** on receipt field extraction across 8 dataset-combination experiments. Results compile into a LaTeX research paper automatically.

### Core Pipeline (5 Stages)
