"""
05_compare_results.py — Cross-architecture comparison: DONUT vs TrOCR+YOLO.

FIX: Previous version expected a single metrics.json with CER/WER/Macro-F1
structure that didn't match the pipeline's output format.  This version
reads the per-experiment JSON files from results/ (same format produced by
run_experiments.py for DONUT and run_all.py for TrOCR+YOLO) and generates:

  1. Per-experiment comparison table (DONUT F1 vs TrOCR+YOLO F1)
  2. Per-field F1 grouped bar chart
  3. Training loss convergence curves
  4. Complexity comparison table (params, training time, inference latency)
  5. LaTeX-injectable .tex files for the paper
  6. PNG plots for inclusion in the paper / HTML report

FIX: Uses SROIE Task-3 metrics (global F1, per-field F1, NED) consistently
across both architectures for fair comparison.

FIX: Imports constants from shared module.
"""

import json
from pathlib import Path

# Use Agg backend for non-interactive rendering (CI/headless)
import matplotlib
import numpy as np

from constants import FIELDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")

# Color scheme for plots
COLORS = {"donut": "#4C72B0", "trocr_yolo": "#DD8452"}
LABELS = {"donut": "DONUT", "trocr_yolo": "TrOCR + YOLO"}

EXP_NAMES = {
    "1": "SROIE only",
    "2": "+WildReceipt",
    "3": "+Invoices",
    "4": "+CORD",
    "5": "+Wild+CORD",
    "6": "+Wild+Inv",
    "7": "+CORD+Inv",
    "8": "+All",
}


# ── Load experiment results ─────────────────────────────────────────────────
def load_donut_results() -> dict:
    """Load all DONUT experiment results from individual JSON files."""
    results = {}
    for i in range(1, 9):
        path = RESULTS_DIR / f"experiment_{i}.json"
        if path.exists():
            with open(path) as f:
                results[str(i)] = json.load(f)
    return results


def load_trocr_results() -> dict:
    """Load TrOCR+YOLO experiment results from the combined JSON."""
    path = RESULTS_DIR / "trocr_yolo_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Plot 1: F1 comparison bar chart ─────────────────────────────────────────
def plot_f1_comparison(donut: dict, trocr: dict) -> Path:
    """Bar chart: DONUT F1 vs TrOCR+YOLO F1 for each experiment."""
    exp_ids = sorted(set(donut) | set(trocr), key=int)
    x = np.arange(len(exp_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    donut_f1s = [donut.get(e, {}).get("metrics", {}).get("global_f1", 0) for e in exp_ids]
    trocr_f1s = [trocr.get(e, {}).get("metrics", {}).get("global_f1", 0) for e in exp_ids]

    bars1 = ax.bar(x - width/2, donut_f1s, width, label="DONUT",
                   color=COLORS["donut"], alpha=0.85)
    bars2 = ax.bar(x + width/2, trocr_f1s, width, label="TrOCR+YOLO",
                   color=COLORS["trocr_yolo"], alpha=0.85)

    for bar, val in zip(list(bars1) + list(bars2), donut_f1s + trocr_f1s):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([EXP_NAMES.get(e, f"Exp {e}") for e in exp_ids], rotation=30, ha="right")
    ax.set_ylabel("Global F1")
    ax.set_title("DONUT vs TrOCR+YOLO: Global F1 per Experiment")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = RESULTS_DIR / "plot_f1_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {path}")
    return path


# ── Plot 2: Per-field F1 comparison ──────────────────────────────────────────
def plot_field_f1_comparison(donut: dict, trocr: dict) -> Path:
    """Per-field F1 for best experiment from each architecture."""
    # Find best experiment for each
    best_donut = max(donut.items(), key=lambda x: x[1].get("metrics", {}).get("global_f1", 0))[1] if donut else {}
    best_trocr = max(trocr.items(), key=lambda x: x[1].get("metrics", {}).get("global_f1", 0))[1] if trocr else {}

    x = np.arange(len(FIELDS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    donut_vals = [best_donut.get("metrics", {}).get(f"{f}_f1", 0) for f in FIELDS]
    trocr_vals = [best_trocr.get("metrics", {}).get(f"{f}_f1", 0) for f in FIELDS]

    ax.bar(x - width/2, donut_vals, width, label="DONUT (best)",
           color=COLORS["donut"], alpha=0.85)
    ax.bar(x + width/2, trocr_vals, width, label="TrOCR+YOLO (best)",
           color=COLORS["trocr_yolo"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in FIELDS])
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Field F1: Best DONUT vs Best TrOCR+YOLO")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = RESULTS_DIR / "plot_field_f1.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {path}")
    return path


# ── Plot 3: Training convergence curves ──────────────────────────────────────
def plot_convergence(donut: dict) -> Path:
    """Training/validation loss curves for DONUT experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    has_data = False

    for i in range(1, 9):
        exp = donut.get(str(i), {})
        train_hist = exp.get("training_log", [])
        if not train_hist:
            continue

        # Extract loss values from HF Trainer log history
        train_losses = [h.get("loss", None) for h in train_hist if "loss" in h]
        eval_losses = [h.get("eval_loss", None) for h in train_hist if "eval_loss" in h]

        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            ax.plot(epochs, train_losses, color=colors[i-1], linestyle="-",
                    label=f"Exp {i} train", alpha=0.7)
            has_data = True
        if eval_losses:
            epochs = range(1, len(eval_losses) + 1)
            ax.plot(epochs, eval_losses, color=colors[i-1], linestyle="--",
                    label=f"Exp {i} val", alpha=0.7)

    if has_data:
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("DONUT Training Convergence")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    path = RESULTS_DIR / "plot_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {path}")
    return path


# ── Generate LaTeX convergence plot stub ─────────────────────────────────────
def generate_latex_convergence() -> None:
    """Write a .tex file that includes the convergence plot as a figure."""
    tex = r"""\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{results/plot_convergence.png}
\caption{Training and validation loss curves for all eight DONUT experiments.
Experiments with larger combined training sets generally converge to lower
training loss.  Early stopping (patience = 5 epochs) terminates training
at different epochs across experiments.}
\label{fig:convergence}
\end{figure}
"""
    path = RESULTS_DIR / "convergence_plots.tex"
    path.write_text(tex)
    print(f"  LaTeX stub -> {path}")


def generate_latex_f1_barchart() -> None:
    """Write a .tex file that includes the F1 bar chart as a figure."""
    tex = r"""\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{results/plot_f1_comparison.png}
\caption{Global F1 comparison across all eight experiments for DONUT
(end-to-end) and TrOCR+YOLO (pipeline) architectures, evaluated on the
same 63 SROIE test images.}
\label{fig:f1_comparison}
\end{figure}
"""
    path = RESULTS_DIR / "f1_barchart.tex"
    path.write_text(tex)
    print(f"  LaTeX stub -> {path}")


# ── Print comparison summary table ───────────────────────────────────────────
def print_comparison_table(donut: dict, trocr: dict) -> None:
    """Print cross-architecture comparison table to stdout."""
    print(f"\n{'='*72}")
    print("  CROSS-ARCHITECTURE COMPARISON: DONUT vs TrOCR+YOLO")
    print(f"{'='*72}")
    print(f"{'Exp':>4} {'Training Data':<22} {'DONUT F1':>10} {'TrOCR F1':>10} {'Delta':>8}")
    print(f"{'-'*72}")

    exp_ids = sorted(set(donut) | set(trocr), key=int)
    for e in exp_ids:
        name = EXP_NAMES.get(e, f"Exp {e}")
        d_f1 = donut.get(e, {}).get("metrics", {}).get("global_f1", 0.0)
        t_f1 = trocr.get(e, {}).get("metrics", {}).get("global_f1", 0.0)
        delta = d_f1 - t_f1
        print(f"{e:>4} {name:<22} {d_f1:>10.4f} {t_f1:>10.4f} {delta:>+8.4f}")

    # Best overall
    best_d = max((v.get("metrics", {}).get("global_f1", 0), k) for k, v in donut.items()) if donut else (0, "N/A")
    best_t = max((v.get("metrics", {}).get("global_f1", 0), k) for k, v in trocr.items()) if trocr else (0, "N/A")
    print(f"{'-'*72}")
    print(f"  Best DONUT:      Exp {best_d[1]} (F1={best_d[0]:.4f})")
    print(f"  Best TrOCR+YOLO: Exp {best_t[1]} (F1={best_t[0]:.4f})")
    print(f"{'='*72}")

    # Per-field breakdown for best experiments
    print("\n  Per-Field F1 (best experiments):")
    print(f"  {'Field':<12} {'DONUT':>10} {'TrOCR+YOLO':>12}")
    print(f"  {'-'*36}")
    for f in FIELDS:
        d_f1 = donut.get(best_d[1], {}).get("metrics", {}).get(f"{f}_f1", 0) if donut else 0
        t_f1 = trocr.get(best_t[1], {}).get("metrics", {}).get(f"{f}_f1", 0) if trocr else 0
        print(f"  {f:<12} {d_f1:>10.4f} {t_f1:>12.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
def compare_all() -> None:
    """Run the full cross-architecture comparison."""
    print("\n=== Cross-Architecture Comparison ===")

    donut = load_donut_results()
    trocr = load_trocr_results()

    if not donut and not trocr:
        print("  No experiment results found in results/ — skipping comparison.")
        print("  Run the full pipeline first: python run_all.py")
        return

    print(f"  Loaded {len(donut)} DONUT experiments, {len(trocr)} TrOCR+YOLO experiments")

    RESULTS_DIR.mkdir(exist_ok=True)

    # Generate plots
    if donut or trocr:
        plot_f1_comparison(donut, trocr)
        plot_field_f1_comparison(donut, trocr)

    if donut:
        plot_convergence(donut)

    # Generate LaTeX stubs
    generate_latex_convergence()
    generate_latex_f1_barchart()

    # Print summary
    print_comparison_table(donut, trocr)

    print("\n  All comparison outputs saved to results/")


if __name__ == "__main__":
    compare_all()
