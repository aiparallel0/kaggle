# MIT License
#
# Copyright (c) 2024
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
quick_results_generator.py — Generate LaTeX results.tex from quick mode and sweeps.

This module converts QuickResults and hyperparameter sweep results into
publication-ready LaTeX documents with embedded matplotlib plots and tables.

Classes:
    ResultsGenerator: Main class for generating results.tex
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# Unified Plot Generation (for both results.tex and paper.tex)
# ============================================================================


def generate_loss_plots_from_results(results_dir: Path = Path("results")) -> dict[str, Path]:
    """Generate 2D loss plots from experiment result JSON files.

    Reads loss history from results/experiment_*.json and generates:
    - donut_loss_all_experiments.png: Overlay plot of all DONUT experiments
    - donut_loss_exp1.png: Single plot for Experiment 1

    Args:
        results_dir: Path to results directory

    Returns:
        Dict mapping plot names to PNG file paths
    """
    if not HAS_MATPLOTLIB:
        return {}

    plots_generated = {}
    plots_dir = results_dir / "figures"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Attempt to load loss histories from experiment files
        # Note: These are typically saved by the trainer via callbacks
        import json

        loss_data = {}
        for exp_file in sorted(results_dir.glob("experiment_*.json")):
            try:
                with open(exp_file) as f:
                    exp_data = json.load(f)
                    exp_id = exp_data.get("experiment_id", 1)
                    # Extract loss history if available (format depends on trainer)
                    if "training_log" in exp_data:
                        loss_data[exp_id] = exp_data["training_log"]
            except Exception:
                pass

        # Generate overlay plot if we have loss data
        if loss_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, len(loss_data)))

            for (exp_id, loss_hist), color in zip(sorted(loss_data.items()), colors):
                if isinstance(loss_hist, dict) and "train_loss" in loss_hist:
                    epochs = range(1, len(loss_hist["train_loss"]) + 1)
                    ax.plot(
                        epochs,
                        loss_hist["train_loss"],
                        marker="o",
                        label=f"Exp {exp_id} (train)",
                        color=color,
                        linewidth=2,
                    )
                elif isinstance(loss_hist, list):
                    epochs = range(1, len(loss_hist) + 1)
                    ax.plot(
                        epochs,
                        loss_hist,
                        marker="o",
                        label=f"Exp {exp_id}",
                        color=color,
                        linewidth=2,
                    )

            ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
            ax.set_ylabel("Training Loss", fontsize=12, fontweight="bold")
            ax.set_title("DONUT Training Loss Across Experiments", fontsize=13, fontweight="bold")
            ax.legend(fontsize=10, loc="best")
            ax.grid(True, alpha=0.3, linestyle="--")
            fig.tight_layout()

            plot_path = plots_dir / "donut_loss_all_experiments.png"
            fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            plots_generated["donut_loss_overlay"] = plot_path

    except Exception:
        pass  # Gracefully skip if loss data unavailable

    return plots_generated


@dataclass
class QuickResults:
    """Container for quick mode execution results."""

    donut_train_losses: list[float]  # per epoch
    donut_val_losses: list[float]  # per epoch
    donut_metrics: dict[str, float]  # global_f1, per-field F1/NED
    trocr_yolo_losses: dict[str, list[float]]  # {"yolo_train": [...], "trocr_train": [...]}
    trocr_yolo_metrics: dict[str, float]
    training_config: dict[str, Any]  # batch_size, epochs, lr, scheduler
    terminal_output_file: Path
    training_time_seconds: float


class ResultsGenerator:
    """Generate results.tex with 2D plots, tables, and terminal output snippets."""

    def __init__(self, quick_results: QuickResults | None = None):
        """Initialize generator.

        Args:
            quick_results: QuickResults object (for simple quick mode)
                          None if using from_sweep_results() (for --quick -all)
        """
        self.quick_results = quick_results
        self.sweep_results: dict[str, dict[str, Any]] = {}
        self.plots_dir = Path("results_plots")
        self.plots_dir.mkdir(exist_ok=True)

    @classmethod
    def from_sweep_results(cls, sweep_results: dict[str, dict[str, Any]]):
        """Alternative constructor for hyperparameter sweep results.

        Args:
            sweep_results: Dict mapping param combination keys to result dicts

        Returns:
            ResultsGenerator instance configured for sweep mode
        """
        obj = cls(None)
        obj.sweep_results = sweep_results
        return obj

    def generate(self, output_path: Path = Path("results.tex")) -> None:
        """Generate complete results.tex document.

        Args:
            output_path: Output file path for generated LaTeX
        """
        lines = []

        # Document header
        lines.extend(self._generate_header())

        # Content based on mode
        if self.sweep_results:
            lines.extend(self._generate_sweep_section())
        elif self.quick_results:
            lines.extend(self._generate_quick_section())

        # Document footer
        lines.extend(self._generate_footer())

        # Write to file
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"✓ Results saved to {output_path}")

    def _generate_header(self) -> list[str]:
        """Generate LaTeX document header."""
        lines = [
            r"\documentclass{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{graphicx, booktabs, amsmath, hyperref, xcolor}",
            r"\usepackage{geometry}",
            r"\geometry{margin=1in}",
            r"\usepackage{listings}",
            r"\lstset{basicstyle=\ttfamily\small, breaklines=true, "
            r"backgroundcolor=\color{gray!10}, frame=single}",
            r"\title{Quick Mode Results — DONUT + TrOCR+YOLO}",
            r"\author{Multi-Dataset KIE Pipeline}",
            r"\date{" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "}",
            r"\begin{document}",
            r"\maketitle",
            r"\tableofcontents",
            r"\newpage",
        ]
        return lines

    def _generate_quick_section(self) -> list[str]:
        """Generate single quick mode results section."""
        if not self.quick_results:
            return []

        lines = [
            r"\section{Quick Mode Results}",
            r"\subsection{Training Configuration}",
        ]

        # Configuration table
        config = self.quick_results.training_config
        lines.extend(
            [
                r"\begin{center}",
                r"\begin{tabular}{|l|r|}",
                r"\hline",
                f"Batch Size & {config.get('batch_size', 8)} \\\\\\hline",
                f"Epochs & {config.get('epochs', 10)} \\\\\\hline",
                f"Learning Rate & {config.get('learning_rate', 5e-5):.0e} \\\\\\hline",
                f"Scheduler & {config.get('lr_scheduler', 'cosine')} \\\\\\hline",
                f"Training Time & {self.quick_results.training_time_seconds / 60:.1f} min \\\\\\hline",
                r"\end{tabular}",
                r"\end{center}",
                r"",
            ]
        )

        # Loss plots
        if HAS_MATPLOTLIB and (
            self.quick_results.donut_train_losses or self.quick_results.trocr_yolo_losses
        ):
            lines.extend(self._generate_loss_plots())

        # Metrics table
        lines.extend(self._generate_metrics_table())

        # Terminal output summary
        lines.extend(self._generate_terminal_summary())

        return lines

    def _generate_loss_plots(self) -> list[str]:
        """Generate 2D loss vs epoch plots; save as PNG, embed in LaTeX."""
        lines = [r"\subsection{Training Loss Curves}"]

        # DONUT loss plot
        if self.quick_results.donut_train_losses and self.quick_results.donut_val_losses:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                epochs = range(1, len(self.quick_results.donut_train_losses) + 1)
                ax.plot(
                    epochs,
                    self.quick_results.donut_train_losses,
                    "b-o",
                    label="Train Loss",
                    linewidth=2,
                    markersize=4,
                )
                ax.plot(
                    epochs,
                    self.quick_results.donut_val_losses,
                    "r--s",
                    label="Val Loss",
                    linewidth=2,
                    markersize=4,
                )
                ax.set_xlabel("Epoch", fontsize=11)
                ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
                ax.set_title(
                    "DONUT Exp 1 (SROIE Baseline) Training", fontsize=12, fontweight="bold"
                )
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle="--")
                fig.tight_layout()

                plot_path = self.plots_dir / "donut_loss.png"
                fig.savefig(str(plot_path), dpi=100, bbox_inches="tight")
                plt.close(fig)

                lines.extend(
                    [
                        r"\subsubsection{DONUT Loss}",
                        r"\begin{center}",
                        r"\includegraphics[width=0.75\textwidth]{results_plots/donut_loss.png}",
                        r"\end{center}",
                        r"",
                    ]
                )
            except Exception as e:
                lines.append(f"% Error generating DONUT loss plot: {e}")

        # TrOCR+YOLO loss plot (if available)
        if self.quick_results.trocr_yolo_losses:
            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                for label, losses in self.quick_results.trocr_yolo_losses.items():
                    if losses:
                        ax.plot(losses, marker="o", label=label, linewidth=2, markersize=4)
                ax.set_xlabel("Epoch", fontsize=11)
                ax.set_ylabel("Loss", fontsize=11)
                ax.set_title("TrOCR + YOLO Training", fontsize=12, fontweight="bold")
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, linestyle="--")
                fig.tight_layout()

                plot_path = self.plots_dir / "trocr_yolo_loss.png"
                fig.savefig(str(plot_path), dpi=100, bbox_inches="tight")
                plt.close(fig)

                lines.extend(
                    [
                        r"\subsubsection{TrOCR + YOLO Loss}",
                        r"\begin{center}",
                        r"\includegraphics[width=0.75\textwidth]{results_plots/trocr_yolo_loss.png}",
                        r"\end{center}",
                        r"",
                    ]
                )
            except Exception as e:
                lines.append(f"% Error generating TrOCR loss plot: {e}")

        return lines

    def _generate_metrics_table(self) -> list[str]:
        """Generate evaluation metrics LaTeX table."""
        lines = [r"\subsection{Evaluation Metrics}"]

        if not self.quick_results:
            return lines

        metrics = self.quick_results.donut_metrics
        lines.extend(
            [
                r"\begin{center}",
                r"\begin{tabular}{|l|r|r|}",
                r"\hline",
                r"\textbf{Field} & \textbf{F1 Score} & \textbf{NED} \\",
                r"\hline",
            ]
        )

        # Per-field metrics
        fields = ["company", "date", "address", "total"]
        for field in fields:
            f1 = metrics.get(f"{field}_f1", 0.0)
            ned = metrics.get(f"{field}_ned", 0.0)
            lines.append(rf"{field.capitalize():12s} & {f1:8.4f} & {ned:8.4f} \\")

        lines.extend(
            [
                r"\hline",
                r"\textbf{Global} & "
                f"{metrics.get('global_f1', 0.0):8.4f} & - \\",
                r"\hline",
                r"\end{tabular}",
                r"\end{center}",
                r"",
            ]
        )

        # TrOCR+YOLO metrics if available
        if self.quick_results.trocr_yolo_metrics:
            lines.extend(
                [
                    r"\subsubsection{TrOCR + YOLO Metrics}",
                    r"\begin{center}",
                    r"\begin{tabular}{|l|r|}",
                    r"\hline",
                ]
            )
            for key, value in self.quick_results.trocr_yolo_metrics.items():
                lines.append(rf"{key:30s} & {value:10.4f} \\")
            lines.extend(
                [
                    r"\hline",
                    r"\end{tabular}",
                    r"\end{center}",
                    r"",
                ]
            )

        return lines

    def _generate_terminal_summary(self) -> list[str]:
        """Extract and embed key terminal output (errors, warnings)."""
        lines = [r"\subsection{Terminal Output Summary}"]

        if not self.quick_results.terminal_output_file.exists():
            lines.append("No terminal output file found.")
            return lines

        try:
            content = self.quick_results.terminal_output_file.read_text(encoding="utf-8")
            # Extract ERROR, WARNING, and CRITICAL lines
            errors = []
            for line in content.split("\n"):
                if any(x in line for x in ["ERROR", "CRITICAL", "WARNING", "FAILED"]):
                    errors.append(line.strip())

            if errors:
                lines.extend(
                    [
                        r"\subsubsection{Errors \& Warnings}",
                        r"\begin{lstlisting}",
                    ]
                )
                # Show first 10 errors
                for error in errors[:10]:
                    # Escape LaTeX special characters
                    safe_error = error.replace("_", r"\_").replace("#", r"\#")
                    lines.append(safe_error)
                lines.append(r"\end{lstlisting}")
            else:
                lines.append("No errors or warnings recorded.")

            lines.append("")
        except Exception as e:
            lines.append(f"Error reading terminal output: {e}")

        return lines

    def _generate_sweep_section(self) -> list[str]:
        """Generate hyperparameter sweep comparison section."""
        lines = [
            r"\section{Hyperparameter Sweep Results}",
            r"\subsection{Parameter Variations Table}",
        ]

        if not self.sweep_results:
            return lines

        # Create comparison table
        lines.extend(
            [
                r"\begin{center}",
                r"\begin{tabular}{|l|r|r|r|r|r|}",
                r"\hline",
                r"\textbf{Batch} & \textbf{Epochs} & \textbf{LR} & \textbf{Sched} & "
                r"\textbf{DONUT F1} & \textbf{Time (s)} \\",
                r"\hline",
            ]
        )

        for _key, result in sorted(self.sweep_results.items()):
            bs = result.get("batch_size", "-")
            ep = result.get("epochs", "-")
            lr = result.get("learning_rate", "-")
            if isinstance(lr, float):
                lr = f"{lr:.0e}"
            sched = result.get("scheduler", "-")[:3]  # abbreviate
            f1 = result.get("donut_f1", 0.0)
            t = result.get("training_time", 0.0)
            lines.append(rf"{bs} & {ep} & {lr} & {sched} & {f1:.4f} & {t:.0f} \\")

        lines.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"\end{center}",
                r"",
            ]
        )

        # Generate comparison plots
        try:
            lines.extend(self._generate_sweep_comparison_plots())
        except Exception as e:
            lines.append(f"% Error generating sweep comparison plots: {e}")

        return lines

    def _generate_sweep_comparison_plots(self) -> list[str]:
        """Generate overlay comparison plots for sweep."""
        if not HAS_MATPLOTLIB or not self.sweep_results:
            return []

        lines = [r"\subsection{Comparison Plots}"]

        try:
            # Plot 1: F1 vs batch_size (colored by scheduler)
            fig, ax = plt.subplots(figsize=(8, 5))

            schedulers = set(r.get("scheduler", "unknown") for r in self.sweep_results.values())
            colors = {"cosine": "blue", "linear": "orange", "constant": "green", "unknown": "red"}

            for scheduler in sorted(schedulers):
                results_for_sched = [
                    (r.get("batch_size", 0), r.get("donut_f1", 0.0))
                    for r in self.sweep_results.values()
                    if r.get("scheduler") == scheduler
                ]
                if results_for_sched:
                    results_for_sched.sort()
                    bs_values, f1_values = zip(*results_for_sched)
                    ax.plot(
                        bs_values,
                        f1_values,
                        marker="o",
                        label=scheduler,
                        linewidth=2,
                        markersize=6,
                        color=colors.get(scheduler, "gray"),
                    )

            ax.set_xlabel("Batch Size", fontsize=11)
            ax.set_ylabel("Global F1", fontsize=11)
            ax.set_title("F1 Score vs Batch Size (by Scheduler)", fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, linestyle="--")
            fig.tight_layout()

            plot_path = self.plots_dir / "sweep_f1_vs_batchsize.png"
            fig.savefig(str(plot_path), dpi=100, bbox_inches="tight")
            plt.close(fig)

            lines.extend(
                [
                    r"\subsubsection{F1 vs Batch Size}",
                    r"\begin{center}",
                    r"\includegraphics[width=0.75\textwidth]{results_plots/sweep_f1_vs_batchsize.png}",
                    r"\end{center}",
                    r"",
                ]
            )
        except Exception as e:
            lines.append(f"% Error generating F1 vs batch size plot: {e}")

        return lines

    def _generate_footer(self) -> list[str]:
        """Generate LaTeX document footer."""
        return [
            r"\end{document}",
        ]
