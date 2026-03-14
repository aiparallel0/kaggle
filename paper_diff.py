# =============================================================================
# paper_diff.py
# Purpose: Compare old and new paper_filled.tex to show changed VAR values
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
paper_diff.py — Diff old vs new paper_filled.tex after inject_results runs.

After ``inject_results.py`` regenerates ``paper_filled.tex``, this module
compares the previously existing version (read before overwriting) against
the newly generated version.  For every ``\\VAR{name}`` placeholder that
changed value, it prints a coloured table:

  field | old_value | new_value | Δ (absolute) | direction (↑↓=)

The diff is also written to ``results/paper_diff_YYYYMMDD.txt``
(UTC date: ``datetime.now(timezone.utc)``).

When ``rich`` is installed, the table is printed with colour.  Otherwise,
plain-text ASCII table is used.

Usage
-----
Called automatically from ``inject_results.main()`` at the end.
Can also be run standalone:
    python paper_diff.py --old paper_old.tex --new paper/paper_filled.tex
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path


# Pre-compiled regex that matches \VAR{name} substitutions in filled LaTeX.
# In a filled file the \VAR{} call is replaced by its value, so we match
# any content between \\VAR tag remnants or look for placeholder patterns.
# Instead, we compare line-by-line to find changed lines and extract values.
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _extract_var_values(text: str) -> dict[str, str]:
    """Extract variable values from a filled LaTeX file.

    The filled file has already had ``\\VAR{name}`` placeholders replaced.
    We cannot easily recover variable names from the filled file, so instead
    we compare line-by-line and report changed lines.

    This function returns a dict mapping line numbers (as str) to line text,
    used for line-level diffing.
    """
    return {str(i): line for i, line in enumerate(text.splitlines())}


def compute_diff(old_text: str, new_text: str) -> list[dict]:
    """Compute line-level diff between old and new filled LaTeX.

    Returns a list of change dicts:
        {"line": int, "old": str, "new": str,
         "delta": float|None, "direction": str}
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()

    changes: list[dict] = []
    max_len = max(len(old_lines), len(new_lines))
    for i in range(max_len):
        old_line = old_lines[i] if i < len(old_lines) else ""
        new_line = new_lines[i] if i < len(new_lines) else ""
        if old_line != new_line:
            # Try to compute numeric delta
            old_stripped = old_line.strip()
            new_stripped = new_line.strip()
            delta: float | None = None
            direction = "~"
            if _NUMERIC_RE.match(old_stripped) and _NUMERIC_RE.match(new_stripped):
                try:
                    o = float(old_stripped)
                    n = float(new_stripped)
                    delta = n - o
                    if delta > 0:
                        direction = "↑"
                    elif delta < 0:
                        direction = "↓"
                    else:
                        direction = "="
                except ValueError:
                    pass
            changes.append({
                "line": i + 1,
                "old": old_line,
                "new": new_line,
                "delta": delta,
                "direction": direction,
            })
    return changes


def print_diff_table(
    changes: list[dict],
    output_file: Path | None = None,
    use_rich: bool | None = None,
) -> None:
    """Print the diff as a table (rich-coloured if available, plain text otherwise).

    Parameters
    ----------
    changes:
        List of change dicts from :func:`compute_diff`.
    output_file:
        If provided, also write plain-text diff to this file.
    use_rich:
        If None, auto-detect based on rich availability.
    """
    if use_rich is None:
        try:
            import rich  # noqa: F401
            use_rich = True
        except ImportError:
            use_rich = False

    lines: list[str] = []
    lines.append(f"paper_diff — {len(changes)} line(s) changed")
    lines.append("=" * 80)
    lines.append(f"{'Line':<6} {'Old value':<30} {'New value':<30} {'Δ':<12} {'Dir':<4}")
    lines.append("-" * 80)
    for c in changes:
        old_short = c["old"][:28].replace("\n", "").replace("\r", "")
        new_short = c["new"][:28].replace("\n", "").replace("\r", "")
        delta_str = f"{c['delta']:+.4f}" if c["delta"] is not None else ""
        lines.append(
            f"{c['line']:<6} {old_short:<30} {new_short:<30} {delta_str:<12} {c['direction']:<4}"
        )
    lines.append("=" * 80)

    plain_text = "\n".join(lines)

    if use_rich:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title=f"Paper Diff — {len(changes)} line(s) changed",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Line", justify="right", style="dim", width=6)
            table.add_column("Old value", style="red")
            table.add_column("New value", style="green")
            table.add_column("Δ", justify="right")
            table.add_column("Dir", justify="center")

            for c in changes:
                delta_str = f"{c['delta']:+.4f}" if c["delta"] is not None else ""
                dir_colour = (
                    "green" if c["direction"] == "↑"
                    else "red" if c["direction"] == "↓"
                    else "yellow"
                )
                table.add_row(
                    str(c["line"]),
                    c["old"][:35],
                    c["new"][:35],
                    delta_str,
                    f"[{dir_colour}]{c['direction']}[/{dir_colour}]",
                )
            console.print(table)
        except Exception:
            print(plain_text)
    else:
        print(plain_text)

    # Write to file
    if output_file is not None:
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(plain_text, encoding="utf-8")
            print(f"[PaperDiff] Diff written to: {output_file}")
        except OSError as exc:
            print(f"[PaperDiff] WARNING: Could not write diff file: {exc}")


def run_paper_diff(
    old_text: str,
    new_text: str,
    results_dir: str | Path = "results",
) -> None:
    """Compare old and new filled paper text; print and save the diff.

    Parameters
    ----------
    old_text:
        Content of the previous paper_filled.tex (read before overwriting).
    new_text:
        Content of the newly generated paper_filled.tex.
    results_dir:
        Directory where the diff file is saved.
    """
    changes = compute_diff(old_text, new_text)
    if not changes:
        print("[PaperDiff] No changes detected in paper_filled.tex.")
        return

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_file = Path(results_dir) / f"paper_diff_{date_str}.txt"
    print_diff_table(changes, output_file=out_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff old vs new paper_filled.tex")
    parser.add_argument("--old", required=True, help="Path to old paper_filled.tex")
    parser.add_argument("--new", required=True, help="Path to new paper_filled.tex")
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory to save diff file (default: results)",
    )
    args = parser.parse_args()

    old_text = Path(args.old).read_text(encoding="utf-8", errors="replace")
    new_text = Path(args.new).read_text(encoding="utf-8", errors="replace")
    run_paper_diff(old_text, new_text, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
