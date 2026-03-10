"""
plot_convergence.py
====================
Generates cubic-spline-smoothed convergence plot .tex files from per-experiment
CSV files in the results/ directory.

Each input CSV has the format:
    epoch,train_loss,eval_loss

Generated output files (all in results/):
    convergence_combined_paper.tex  -- combined axes for paper.tex (figure body only)
    convergence_combined_slides.tex -- same, scaled for Beamer slides
    convergence_grid_1_8_slides.tex -- 2x4 grid for experiments 1-8
    convergence_grid_9_18_slides.tex -- grid for experiments 9-18

All output files are self-contained tikzpicture/pgfplots blocks that can be
input-ed directly.  No begin{document} wrapper is written.

Usage
-----
    python plot_convergence.py                       # uses results/ and experiment_selection.json
    python plot_convergence.py --results-dir results --selection experiment_selection.json

API
---
    import plot_convergence
    plot_convergence.generate_all()
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

# Optional scientific computing imports — used for cubic spline smoothing.
# Both scipy and numpy are listed in requirements.txt.  If unavailable, the
# module degrades gracefully to passing raw (unsmoothed) data points.
try:
    import numpy as _np
    from scipy.interpolate import CubicSpline as _CubicSpline

    _SCIPY_AVAILABLE = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _CubicSpline = None  # type: ignore[assignment,misc]
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Colour palette — 12-colour qualitative palette (ColorBrewer Set1 + Set2 mix)
# Assigned by experiment ID modulo palette length.
# ---------------------------------------------------------------------------

_PALETTE = [
    "red!80!black",
    "blue!80!black",
    "green!60!black",
    "orange!90!black",
    "purple!80!black",
    "cyan!70!black",
    "brown!80!black",
    "pink!80!black",
    "teal!80!black",
    "violet!80!black",
    "lime!70!black",
    "magenta!70!black",
]

# Short experiment names for legend labels
_EXP_LABELS: dict[str, str] = {
    "1": "Exp~1 SROIE",
    "2": "Exp~2 +WR",
    "3": "Exp~3 +Inv",
    "4": "Exp~4 +WR+Inv",
    "5": "Exp~5 +WR(2×)",
    "6": "Exp~6 +Inv(2×)",
    "7": "Exp~7 +All(2×)",
    "8": "Exp~8 +All(3×)",
    "9": "Exp~9 ZS",
    "10": "Exp~10 FT-fp16",
    "11": "Exp~11 FT-bf16",
    "12": "Exp~12 TrOCR",
    "13": "Exp~13 FT-fp32",
    "14": "Exp~14 HR-fp16",
    "15": "Exp~15 TrOCR-S",
    "16": "Exp~16 HR-bf16",
    "17": "Exp~17 HR-fp32",
    "18": "Exp~18 ZS-HR",
}


# ---------------------------------------------------------------------------
# CSV reading helpers
# ---------------------------------------------------------------------------


def _read_csv(csv_path: Path) -> tuple[list[float], list[float | None], list[float | None]]:
    """
    Read a convergence CSV file.

    Returns (epochs, train_losses, eval_losses).
    Empty cells → None.
    """
    epochs: list[float] = []
    train_losses: list[float | None] = []
    eval_losses: list[float | None] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                ep = float(row["epoch"])
            except (ValueError, KeyError):
                continue
            epochs.append(ep)

            tl = row.get("train_loss", "").strip()
            train_losses.append(float(tl) if tl else None)

            el = row.get("eval_loss", "").strip()
            eval_losses.append(float(el) if el else None)

    return epochs, train_losses, eval_losses


# ---------------------------------------------------------------------------
# Cubic spline smoothing
# ---------------------------------------------------------------------------


def smooth_curve(
    epochs: Sequence[float],
    values: Sequence[float | None],
    n_points: int = 200,
) -> tuple[list[float] | None, list[float] | None]:
    """
    Cubic spline interpolation.  Skips None / NaN values.

    Returns (x_smooth, y_smooth) or (None, None) if fewer than 2 valid points.
    Falls back to returning raw data points if scipy/numpy are unavailable.
    """
    if not _SCIPY_AVAILABLE:
        # scipy / numpy unavailable — return raw data without smoothing
        valid = [(e, v) for e, v in zip(epochs, values) if v is not None]
        if len(valid) < 2:
            return None, None
        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        return xs, ys

    valid = [(e, v) for e, v in zip(epochs, values) if v is not None and not _is_nan(v)]
    if len(valid) < 2:
        return None, None

    x = _np.array([p[0] for p in valid], dtype=float)
    y = _np.array([p[1] for p in valid], dtype=float)

    try:
        cs = _CubicSpline(x, y)
        x_smooth = _np.linspace(x[0], x[-1], n_points)
        y_smooth = cs(x_smooth)
        return x_smooth.tolist(), y_smooth.tolist()
    except Exception:
        # Fallback if CubicSpline fails (e.g. repeated x values)
        return x.tolist(), y.tolist()


def _is_nan(v: float) -> bool:
    try:
        return v != v  # NaN check without math import
    except TypeError:
        return True


# ---------------------------------------------------------------------------
# pgfplots coordinate string
# ---------------------------------------------------------------------------


def _coords_str(xs: list[float], ys: list[float]) -> str:
    """Return pgfplots coordinates string: (x1,y1) (x2,y2) ..."""
    return " ".join(f"({x:.6g},{y:.6g})" for x, y in zip(xs, ys))


# ---------------------------------------------------------------------------
# Per-experiment data loader
# ---------------------------------------------------------------------------


def _load_exp_data(
    exp_id: str,
    results_dir: Path,
) -> dict | None:
    """Load and smooth one experiment's CSV.  Returns None if CSV absent or has <2 points."""
    csv_path = results_dir / f"convergence_exp{exp_id}.csv"
    if not csv_path.exists():
        return None

    try:
        epochs, train_losses, eval_losses = _read_csv(csv_path)
    except Exception:
        return None

    train_xs, train_ys = smooth_curve(epochs, train_losses)
    eval_xs, eval_ys = smooth_curve(epochs, eval_losses)

    if train_xs is None and eval_xs is None:
        return None

    return {
        "exp_id": exp_id,
        "label": _EXP_LABELS.get(exp_id, f"Exp~{exp_id}"),
        "color": _PALETTE[int(exp_id) % len(_PALETTE)],
        "train_xs": train_xs,
        "train_ys": train_ys,
        "eval_xs": eval_xs,
        "eval_ys": eval_ys,
    }


# ---------------------------------------------------------------------------
# Combined axis generators
# ---------------------------------------------------------------------------

_HEADER = "% Auto-generated by plot_convergence.py — do not edit manually\n"


def _combined_axis_content(
    exp_data: list[dict],
    width: str = r"\linewidth",
    height: str = "5cm",
) -> str:
    """Return lines for a single axis showing all experiments."""
    lines: list[str] = []
    for ed in exp_data:
        color = ed["color"]
        label = ed["label"]

        if ed["train_xs"] is not None:
            coords = _coords_str(ed["train_xs"], ed["train_ys"])
            lines.append(
                f"\\addplot[color={color}, solid, line width=0.8pt] "
                f"coordinates {{{coords}}};\n"
                f"\\addlegendentry{{{label} train}}"
            )
        if ed["eval_xs"] is not None:
            coords = _coords_str(ed["eval_xs"], ed["eval_ys"])
            lines.append(
                f"\\addplot[color={color}, dashed, line width=0.8pt] "
                f"coordinates {{{coords}}};\n"
                f"\\addlegendentry{{{label} val}}"
            )

    plots_str = "\n".join(lines)
    return (
        f"\\begin{{tikzpicture}}\n"
        f"\\begin{{axis}}[\n"
        f"  width={width}, height={height},\n"
        f"  xlabel={{Epoch}}, ylabel={{Loss}},\n"
        f"  legend pos=north east,\n"
        f"  legend style={{font=\\tiny}},\n"
        f"  grid=major, grid style={{gray!30}},\n"
        f"  every axis plot/.append style={{line width=0.8pt}},\n"
        f"]\n"
        f"{plots_str}\n"
        f"\\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def generate_combined_paper(
    exp_data: list[dict],
    results_dir: Path,
) -> None:
    """Write results/convergence_combined_paper.tex — figure body for paper.tex."""
    content = _HEADER + _combined_axis_content(exp_data, width=r"\linewidth", height="5cm")
    out = results_dir / "convergence_combined_paper.tex"
    out.write_text(content, encoding="utf-8")


def generate_combined_slides(
    exp_data: list[dict],
    results_dir: Path,
) -> None:
    """Write results/convergence_combined_slides.tex — for Beamer slides."""
    content = _HEADER + _combined_axis_content(exp_data, width=r"\textwidth", height="5cm")
    out = results_dir / "convergence_combined_slides.tex"
    out.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Grid generators
# ---------------------------------------------------------------------------


def _single_axis_block(
    ed: dict,
    width: str = "0.48\\textwidth",
    height: str = "3.5cm",
) -> str:
    """Return a minipage + axis for one experiment."""
    label = ed["label"]
    color = ed["color"]
    lines: list[str] = []

    if ed["train_xs"] is not None:
        coords = _coords_str(ed["train_xs"], ed["train_ys"])
        lines.append(
            f"  \\addplot[color={color}, solid, line width=0.8pt] "
            f"coordinates {{{coords}}};\n"
            f"  \\addlegendentry{{train}}"
        )
    if ed["eval_xs"] is not None:
        coords = _coords_str(ed["eval_xs"], ed["eval_ys"])
        lines.append(
            f"  \\addplot[color={color}, dashed, line width=0.8pt] "
            f"coordinates {{{coords}}};\n"
            f"  \\addlegendentry{{val}}"
        )

    plots_str = "\n".join(lines)
    return (
        f"\\begin{{minipage}}{{{width}}}\n"
        f"\\begin{{tikzpicture}}\n"
        f"\\begin{{axis}}[\n"
        f"  title={{{label}}},\n"
        f"  width=\\textwidth, height={height},\n"
        f"  xlabel={{Epoch}}, ylabel={{Loss}},\n"
        f"  legend pos=north east,\n"
        f"  legend style={{font=\\tiny}},\n"
        f"  grid=major, grid style={{gray!30}},\n"
        f"  title style={{font=\\small\\bfseries}},\n"
        f"  label style={{font=\\tiny}},\n"
        f"  tick label style={{font=\\tiny}},\n"
        f"]\n"
        f"{plots_str}\n"
        f"\\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
        f"\\end{{minipage}}\n"
    )


def _grid_tex(
    exp_data: list[dict],
    cols: int = 2,
    width: str = "0.48\\textwidth",
    height: str = "3.5cm",
) -> str:
    """Return a sequence of minipage blocks arranged in a grid."""
    blocks: list[str] = []
    for i, ed in enumerate(exp_data):
        block = _single_axis_block(ed, width=width, height=height)
        if i > 0 and i % cols == 0:
            blocks.append("\\\\\n")
        elif i > 0:
            blocks.append("\\hfill\n")
        blocks.append(block)
    return "".join(blocks)


def generate_grid_1_8(
    exp_data: list[dict],
    results_dir: Path,
) -> None:
    """Write results/convergence_grid_1_8_slides.tex — 2×4 grid for exps 1–8."""
    subset = [ed for ed in exp_data if int(ed["exp_id"]) <= 8]
    if not subset:
        content = _HEADER + "% No convergence data available for experiments 1--8.\n"
    else:
        content = _HEADER + _grid_tex(subset, cols=2, width="0.48\\textwidth", height="3.0cm")
    out = results_dir / "convergence_grid_1_8_slides.tex"
    out.write_text(content, encoding="utf-8")


def generate_grid_9_18(
    exp_data: list[dict],
    results_dir: Path,
) -> None:
    """Write results/convergence_grid_9_18_slides.tex — grid for exps 9–18."""
    subset = [ed for ed in exp_data if int(ed["exp_id"]) >= 9]
    if not subset:
        content = _HEADER + "% No convergence data available for experiments 9--18.\n"
    else:
        content = _HEADER + _grid_tex(subset, cols=2, width="0.48\\textwidth", height="3.0cm")
    out = results_dir / "convergence_grid_9_18_slides.tex"
    out.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_all(
    results_dir: str | Path = "results",
    selection_file: str | Path = "experiment_selection.json",
) -> None:
    """
    Generate all four convergence .tex output files.

    Missing CSV files are silently skipped.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine which experiment IDs to plot
    selection_file = Path(selection_file)
    exp_ids: list[str] = []
    if selection_file.exists():
        try:
            with open(selection_file, encoding="utf-8") as fh:
                sel = json.load(fh)
            for entry in sel.get("experiments", []):
                if entry.get("enabled", True):
                    exp_ids.append(str(int(str(entry.get("id", "")))))
        except Exception:
            pass

    if not exp_ids:
        # Fallback: discover from available CSV files
        exp_ids = sorted(
            [p.stem.replace("convergence_exp", "") for p in results_dir.glob("convergence_exp*.csv")],
            key=lambda s: int(s) if s.isdigit() else 999,
        )

    # Load and smooth data for each experiment
    exp_data: list[dict] = []
    for eid in exp_ids:
        ed = _load_exp_data(eid, results_dir)
        if ed is not None:
            exp_data.append(ed)

    if not exp_data:
        # Write placeholder files so \inputifexists compiles cleanly
        placeholder = _HEADER + "% No convergence CSV files found yet.\n"
        for fname in [
            "convergence_combined_paper.tex",
            "convergence_combined_slides.tex",
            "convergence_grid_1_8_slides.tex",
            "convergence_grid_9_18_slides.tex",
        ]:
            (results_dir / fname).write_text(placeholder, encoding="utf-8")
        return

    generate_combined_paper(exp_data, results_dir)
    generate_combined_slides(exp_data, results_dir)
    generate_grid_1_8(exp_data, results_dir)
    generate_grid_9_18(exp_data, results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate convergence plot .tex files from experiment CSV data."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing convergence_expN.csv files (default: results)",
    )
    parser.add_argument(
        "--selection",
        default="experiment_selection.json",
        help="Path to experiment_selection.json (default: experiment_selection.json)",
    )
    args = parser.parse_args()
    generate_all(results_dir=args.results_dir, selection_file=args.selection)
    print("plot_convergence: convergence .tex files written to", args.results_dir)


if __name__ == "__main__":
    main()
