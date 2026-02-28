# Quick Mode Guide — Fast Testing & Parameter Sweeps

This guide documents the new `-quick` mode and related features added to `run_all.py`.

## Overview

The refactored `run_all.py` now includes:

1. **Auto-Install Dependencies** — Automatically installs `requirements.txt` before training
2. **Dual-Stream Logging** — All output logged to `terminal.txt`; console shows filtered output
3. **Quick Mode (`-quick`)** — Run only SROIE baseline + TrOCR+YOLO, generate `results.tex` with loss plots
4. **Hyperparameter Sweep (`-quick -all`)** — Test multiple hyperparameter combinations, generate comparison tables
5. **Professional CLI Interface** — Clean argparse-based interface with examples

## Quick Start

### 1. Simple Quick Test (30 min on A100)

```bash
python run_all.py -quick
```

**What it does:**
- Installs dependencies automatically
- Sets up SROIE data
- Trains DONUT Experiment 1 (SROIE baseline)
- Trains TrOCR+YOLO
- Generates `results.tex` with 2D loss curves and metrics table
- All output logged to `terminal.txt`

**Output files:**
- `results.tex` — LaTeX document with loss plots and metrics
- `results_plots/` — PNG figures (donut_loss.png, trocr_loss.png)
- `terminal.txt` — Complete execution log
- `results/experiment_1.json` — DONUT Exp 1 metrics

### 2. Hyperparameter Sweep (with parameter testing)

```bash
python run_all.py -quick -all
```

**What it does:**
- Runs quick test with default parameter grids:
  - Batch sizes: 4, 8, 16
  - Epochs: 5, 10, 15
  - Learning rates: 1e-5, 5e-5, 1e-4
  - Schedulers: linear, cosine
- Generates comparison `results.tex` with parameter variation tables and overlay plots

**To customize parameter grid:**

```bash
python run_all.py -quick -all --param-grid batch_size 8 16 --param-grid epochs 8 10
```

This tests only batch_size=[8, 16] and epochs=[8, 10] (smaller grid = faster testing).

### 3. Full Pipeline (unchanged behavior)

```bash
python run_all.py
```

Runs all 8 DONUT experiments + TrOCR+YOLO + benchmarking + paper generation (12+ hours).

## Configuring Hyperparameters

Edit `TRAINING_PARAMS` dict in `run_all.py` to change defaults for quick mode:

```python
TRAINING_PARAMS: dict[str, Any] = {
    "batch_size": 8,           # Change default batch size
    "epochs": 10,              # Change default epochs
    "learning_rate": 5e-5,     # Change default learning rate (1e-5 to 1e-4 typical)
    "lr_scheduler_type": "cosine",  # or "linear" / "constant"
}
```

These values are used by:
- Quick mode (`-quick`)
- Full pipeline (if applied globally in future versions)
- Reflected in final `results.tex`

## Understanding the Output

### terminal.txt

Complete execution log with:
- Timestamps for each major stage
- Training progress (epochs, loss values)
- Inference results
- Errors and warnings (if any)

Example:
```
2025-02-28 14:32:10 | run_all | INFO | ========================================================================
2025-02-28 14:32:10 | run_all | INFO | QUICK MODE: Single DONUT Experiment + TrOCR+YOLO
2025-02-28 14:32:10 | run_all | INFO | ========================================================================
2025-02-28 14:32:10 | run_all | INFO | [Stage 0] SROIE data install...
2025-02-28 14:35:22 | run_all | INFO | [Stage 2] Training DONUT Experiment 1 (SROIE baseline)...
...
```

### results.tex

LaTeX document containing:

**Quick Mode Section:**
- Training configuration (batch size, epochs, LR, scheduler)
- 2D loss curves (train vs val, one plot per architecture)
- Metrics table (F1, NED per field)
- Terminal output snippets (errors/warnings)

**Sweep Mode Section (with -quick -all):**
- Parameter variations table (all combinations with F1 scores)
- Comparison plots:
  - F1 vs batch size (colored by scheduler)
  - F1 vs epochs
  - F1 vs learning rate
  - etc.

Compile to PDF:
```bash
pdflatex results.tex
```

### results_plots/

PNG figures embedded in `results.tex`:
- `donut_loss.png` — DONUT train/val loss vs epoch
- `trocr_yolo_loss.png` — TrOCR+YOLO losses
- `sweep_f1_vs_batchsize.png` — Overlay comparison (sweep mode)
- etc.

## Common Use Cases

### Test if a hyperparameter change helps

```bash
# Edit TRAINING_PARAMS to set batch_size=16 (instead of default 8)
# Then run quick test
python run_all.py -quick

# Results in results.tex show F1, NED, and loss curves
# Compare against previous runs to see if change helped
```

### Find optimal batch size

```bash
python run_all.py -quick -all --param-grid batch_size 4 8 16 32
```

Generates results.tex with batch size comparison. Look at the "Comparison Plots" section to see which batch size achieves best F1.

### Test multiple learning rates

```bash
python run_all.py -quick -all --param-grid learning_rate 1e-5 3e-5 1e-4 3e-4
```

### Debug a broken configuration

Run quick test first (fast failure detection):
```bash
python run_all.py -quick
```

If it fails, check `terminal.txt` for error messages. Once fixed, run full pipeline:
```bash
python run_all.py
```

## Files Modified / Created

### New Files
- `training_config.py` — TrainingConfig dataclass + parameter grids
- `quick_results_generator.py` — LaTeX generation + plot generation
- `QUICK_MODE_GUIDE.md` — This file

### Modified Files
- `run_all.py` (~2,100 lines now):
  - Added `_install_dependencies()` function
  - Added `_setup_logging()` + `_DualStreamHandler` class
  - Added `QuickResults` dataclass
  - Added `HyperparameterGrid` dataclass
  - Added `_quick_mode_handler()` function
  - Added `_quick_all_mode_handler()` function
  - Extended `build_parser()` with new arguments
  - Modified `main()` to call auto-install + logging setup + quick mode dispatch

## Backward Compatibility

**All existing usage is unchanged:**

```bash
python run_all.py                          # Full pipeline (same as before)
python run_all.py --experiment 2           # Single experiment (same as before)
python run_all.py --paper-only             # Paper only (same as before)
python run_all.py --skip-trocr             # Skip TrOCR (same as before)
```

The new features (`-quick`, `-all`, etc.) are additive and don't break existing workflows.

## Troubleshooting

### `python run_all.py -quick` hangs

- Check that SROIE data is cloned (see Stage 0 output in terminal.txt)
- Try adding `--skip-install` if SROIE already exists: `python run_all.py -quick --skip-install`
- Check GPU memory: quick mode needs ~8 GB VRAM

### Results.tex won't compile

- Check for missing packages: `pip install matplotlib numpy`
- Verify `results_plots/` directory exists and contains PNG files
- Try opening the tex file in a text editor to see if there are obvious LaTeX errors

### Terminal output not appearing in terminal.txt

- Check file permissions: `ls -la terminal.txt`
- Try removing and re-running: `rm terminal.txt && python run_all.py -quick`

## Performance Tips

1. **Speed up dataset downloads** — Create `hf_token.txt` with your HuggingFace token for 5-10× faster downloads
2. **GPU memory** — If OOM, reduce batch_size: `--param-grid batch_size 4 8`
3. **Test on smaller data first** — Use quick mode (`-quick`) before running full pipeline
4. **Run at night** — Full pipeline takes 12+ hours; start before sleep/weekend

## Advanced: Modifying Parameter Grids

Edit `training_config.py` to change default parameter grids:

```python
PARAM_GRIDS_DEFAULT = {
    "batch_sizes": [8, 16],           # Test fewer sizes for speed
    "epochs_list": [5, 10],           # Fewer epochs for speed
    "learning_rates": [5e-5, 1e-4],   # Focus on high-performing rates
    "schedulers": ["cosine"],         # Test only cosine scheduler
}
```

This reduces the number of combinations and speeds up sweeps.

## See Also

- `CLAUDE.md` — Full architecture documentation
- `constants.py` — Shared constants (FIELDS, IMAGE_EXTS, etc.)
- `run_experiments.py` — Experiment configuration
- `quick_results_generator.py` — LaTeX generation internals
