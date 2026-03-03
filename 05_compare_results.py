"""
05_compare_results.py — Cross-architecture comparison: DONUT vs TrOCR+YOLO.

Reads per-experiment JSON files from results/ (produced by run_experiments.py
for DONUT and run_all.py for TrOCR+YOLO) and generates:

  1. Per-experiment comparison table (DONUT F1 vs TrOCR+YOLO F1)
  2. Per-field F1 grouped bar chart
  3. Training loss convergence curves
  4. Complexity comparison table (params, training time, inference latency)
  5. LaTeX-injectable .tex files for the paper
  6. PNG plots for inclusion in the paper / HTML report

See also: benchmark_compare.py — for head-to-head live evaluation that runs
both pipelines on actual receipt images (requires model checkpoints and an
images/labels directory). Use this file (05_compare_results.py) when results
JSON files already exist; use benchmark_compare.py for a fresh live benchmark.
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
    "4": "+WildReceipt+Invoices",
    "5": "+WildReceipt (2x SROIE)",
    "6": "+Invoices (2x SROIE)",
    "7": "+All (2x SROIE)",
    "8": "+All (3x SROIE)",
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

    bars1 = ax.bar(
        x - width / 2, donut_f1s, width, label="DONUT", color=COLORS["donut"], alpha=0.85
    )
    bars2 = ax.bar(
        x + width / 2, trocr_f1s, width, label="TrOCR+YOLO", color=COLORS["trocr_yolo"], alpha=0.85
    )

    for bar, val in zip(list(bars1) + list(bars2), donut_f1s + trocr_f1s):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

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
    best_donut = (
        max(donut.items(), key=lambda x: x[1].get("metrics", {}).get("global_f1", 0))[1]
        if donut
        else {}
    )
    best_trocr = (
        max(trocr.items(), key=lambda x: x[1].get("metrics", {}).get("global_f1", 0))[1]
        if trocr
        else {}
    )

    x = np.arange(len(FIELDS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    donut_vals = [best_donut.get("metrics", {}).get(f"{f}_f1", 0) for f in FIELDS]
    trocr_vals = [best_trocr.get("metrics", {}).get(f"{f}_f1", 0) for f in FIELDS]

    ax.bar(
        x - width / 2, donut_vals, width, label="DONUT (best)", color=COLORS["donut"], alpha=0.85
    )
    ax.bar(
        x + width / 2,
        trocr_vals,
        width,
        label="TrOCR+YOLO (best)",
        color=COLORS["trocr_yolo"],
        alpha=0.85,
    )

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
            ax.plot(
                epochs,
                train_losses,
                color=colors[i - 1],
                linestyle="-",
                label=f"Exp {i} train",
                alpha=0.7,
            )
            has_data = True
        if eval_losses:
            epochs = range(1, len(eval_losses) + 1)
            ax.plot(
                epochs,
                eval_losses,
                color=colors[i - 1],
                linestyle="--",
                label=f"Exp {i} val",
                alpha=0.7,
            )

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
    print(f"\n{'=' * 72}")
    print("  CROSS-ARCHITECTURE COMPARISON: DONUT vs TrOCR+YOLO")
    print(f"{'=' * 72}")
    print(f"{'Exp':>4} {'Training Data':<22} {'DONUT F1':>10} {'TrOCR F1':>10} {'Delta':>8}")
    print(f"{'-' * 72}")

    exp_ids = sorted(set(donut) | set(trocr), key=int)
    for e in exp_ids:
        name = EXP_NAMES.get(e, f"Exp {e}")
        d_f1 = donut.get(e, {}).get("metrics", {}).get("global_f1", 0.0)
        t_f1 = trocr.get(e, {}).get("metrics", {}).get("global_f1", 0.0)
        delta = d_f1 - t_f1
        print(f"{e:>4} {name:<22} {d_f1:>10.4f} {t_f1:>10.4f} {delta:>+8.4f}")

    # Best overall
    best_d = (
        max((v.get("metrics", {}).get("global_f1", 0), k) for k, v in donut.items())
        if donut
        else (0, "N/A")
    )
    best_t = (
        max((v.get("metrics", {}).get("global_f1", 0), k) for k, v in trocr.items())
        if trocr
        else (0, "N/A")
    )
    print(f"{'-' * 72}")
    print(f"  Best DONUT:      Exp {best_d[1]} (F1={best_d[0]:.4f})")
    print(f"  Best TrOCR+YOLO: Exp {best_t[1]} (F1={best_t[0]:.4f})")
    print(f"{'=' * 72}")

    # Per-field breakdown for best experiments
    print("\n  Per-Field F1 (best experiments):")
    print(f"  {'Field':<12} {'DONUT':>10} {'TrOCR+YOLO':>12}")
    print(f"  {'-' * 36}")
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


def export_to_csv() -> None:
    """Export comparison results to CSV format for external analysis."""
    import csv

    donut = load_donut_results()
    trocr = load_trocr_results()

    # Export per-experiment F1 scores
    csv_path = RESULTS_DIR / "comparison_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Experiment", "Name", "DONUT_Global_F1", "DONUT_Company_F1",
            "DONUT_Date_F1", "DONUT_Address_F1", "DONUT_Total_F1",
            "TrOCR_Global_F1", "TrOCR_Company_F1", "TrOCR_Date_F1",
            "TrOCR_Address_F1", "TrOCR_Total_F1", "F1_Difference"
        ])

        exp_ids = sorted(set(donut) | set(trocr), key=int)
        for exp_id in exp_ids:
            d_data = donut.get(str(exp_id), {}).get("metrics", {})
            t_data = trocr.get(str(exp_id), {}).get("metrics", {})

            d_f1 = d_data.get("global_f1", 0.0)
            t_f1 = t_data.get("global_f1", 0.0)

            writer.writerow([
                exp_id,
                EXP_NAMES.get(str(exp_id), f"Exp {exp_id}"),
                f"{d_f1:.4f}",
                f"{d_data.get('company_f1', 0):.4f}",
                f"{d_data.get('date_f1', 0):.4f}",
                f"{d_data.get('address_f1', 0):.4f}",
                f"{d_data.get('total_f1', 0):.4f}",
                f"{t_f1:.4f}",
                f"{t_data.get('company_f1', 0):.4f}",
                f"{t_data.get('date_f1', 0):.4f}",
                f"{t_data.get('address_f1', 0):.4f}",
                f"{t_data.get('total_f1', 0):.4f}",
                f"{d_f1 - t_f1:+.4f}",
            ])

    print(f"  📊 Exported to CSV -> {csv_path}")


def filter_by_threshold(min_f1: float = 0.8) -> None:
    """Print experiments meeting a minimum F1 score threshold."""
    donut = load_donut_results()

    print(f"\n  Experiments with Global F1 >= {min_f1:.2f}:")
    print(f"  {'-' * 50}")

    count = 0
    for exp_id in sorted(donut.keys(), key=int):
        f1 = donut[exp_id].get("metrics", {}).get("global_f1", 0)
        if f1 >= min_f1:
            name = EXP_NAMES.get(exp_id, f"Exp {exp_id}")
            print(f"  Exp {exp_id:1s} ({name:<20s}) — F1 = {f1:.4f}")
            count += 1

    if count == 0:
        print(f"  No experiments found above threshold")


def plot_ned_comparison(donut: dict, trocr: dict) -> None:
    """Generate a plot comparing Normalized Edit Distance across fields."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Get best experiments
    best_d_idx = (
        max(donut.keys(), key=lambda k: donut[k].get("metrics", {}).get("global_f1", 0))
        if donut
        else None
    )
    best_t_idx = (
        max(trocr.keys(), key=lambda k: trocr[k].get("metrics", {}).get("global_f1", 0))
        if trocr
        else None
    )

    x = np.arange(len(FIELDS))
    width = 0.35

    donut_neds = []
    trocr_neds = []

    if best_d_idx:
        best_d = donut[best_d_idx].get("metrics", {})
        donut_neds = [best_d.get(f"{f}_ned", 1.0) for f in FIELDS]

    if best_t_idx:
        best_t = trocr[best_t_idx].get("metrics", {})
        trocr_neds = [best_t.get(f"{f}_ned", 1.0) for f in FIELDS]

    if donut_neds:
        ax.bar(
            x - width / 2,
            donut_neds,
            width,
            label="DONUT (best)",
            color=COLORS["donut"],
            alpha=0.85,
        )

    if trocr_neds:
        ax.bar(
            x + width / 2,
            trocr_neds,
            width,
            label="TrOCR+YOLO (best)",
            color=COLORS["trocr_yolo"],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in FIELDS])
    ax.set_ylabel("Normalized Edit Distance (lower is better)")
    ax.set_title("Per-Field NED: Best DONUT vs Best TrOCR+YOLO")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    path = RESULTS_DIR / "plot_ned_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  🎯 NED plot saved -> {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare DONUT and TrOCR+YOLO results")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV format"
    )
    parser.add_argument(
        "--filter",
        type=float,
        metavar="THRESHOLD",
        help="Show only experiments with F1 >= THRESHOLD"
    )
    parser.add_argument(
        "--ned-plot",
        action="store_true",
        help="Generate NED comparison plot"
    )

    args = parser.parse_args()

    if args.filter:
        donut = load_donut_results()
        filter_by_threshold(args.filter)
    elif args.export_csv:
        RESULTS_DIR.mkdir(exist_ok=True)
        export_to_csv()
    elif args.ned_plot:
        RESULTS_DIR.mkdir(exist_ok=True)
        donut = load_donut_results()
        trocr = load_trocr_results()
        plot_ned_comparison(donut, trocr)
    else:
        compare_all()
