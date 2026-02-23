---
# Fill in the fields below to create a basic custom agent for your repository.
# The Copilot CLI can be used for local testing: https://gh.io/customagents/cli
# To make this agent available, merge this file into the default repository branch.
# For format details, see: https://gh.io/customagents/config
name: DONUT Receipt Audit Copilot
description: Assists with the multi-dataset DONUT fine-tuning pipeline for receipt key information extraction (KIE). Helps debug training scripts, interpret SROIE evaluation metrics, configure experiment runs, troubleshoot dataset loading for WildReceipt/CORD/SROIE-NER, and understand or modify the LaTeX paper generation workflow.
---
# DONUT Receipt KIE Pipeline Agent

This agent helps contributors and researchers work with the multi-dataset DONUT fine-tuning pipeline for receipt Key Information Extraction (KIE).

## What this agent can help with

- **Running experiments**: Explaining CLI flags for `run_all.py`, `run_experiments.py`, and `inject_results.py`; helping select the right experiment configuration (Experiments 1–8) for your research goals.
- **Dataset setup**: Troubleshooting auto-download and normalization of WildReceipt, SROIE, SROIE-NER, CORD, and Invoices-DONUT datasets via `dataset_loaders.py`.
- **Training & evaluation**: Debugging `train.py` and `evaluate.py`, interpreting SROIE Task-3 global F1 and NED metrics, and understanding results stored in `results/experiment_N.json`.
- **Paper generation**: Explaining how `inject_results.py` fills `\VAR{}` placeholders in `paper.tex` to produce `paper_filled.tex`.
- **Environment setup**: Helping configure `requirements.txt` dependencies for vast.ai, Jupyter terminals, or local GPU environments.

## Key files to reference

| File | Purpose |
|---|---|
| `run_all.py` | Single entry point for the full pipeline |
| `dataset_loaders.py` | Dataset download & normalization |
| `train.py` | SROIE fine-tuning |
| `evaluate.py` | Inference + metric computation |
| `inject_results.py` | LaTeX table generation |
| `paper.tex` | Paper template |
