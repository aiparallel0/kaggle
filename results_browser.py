# =============================================================================
# results_browser.py
# Purpose: Unified CLI results browser and leaderboard
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
results_browser.py — Unified CLI results leaderboard.

Scans ``results/``, ``results/v2/``, and ``results/multi_seed/`` for all
``*.json`` result files, then prints a ranked leaderboard sorted by F1
descending.

Flags
-----
--field company|date|address|total
    Filter / sort by per-field F1 instead of global F1.
--arch donut|trocr
    Filter by architecture (inferred from result file path).
--export csv
    Dump leaderboard to ``results/leaderboard_YYYYMMDD.csv`` (UTC date).

Uses ``rich.table.Table`` if available, plain text otherwise.

Usage
-----
    python results_browser.py
    python results_browser.py --field company --arch donut
    python results_browser.py --export csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _scan_results(results_root: Path) -> list[dict]:
    """Scan results directories and return a list of result dicts."""
    scan_dirs = [
        results_root,
        results_root / "v2",
        results_root / "multi_seed",
    ]
    results: list[dict] = []
    seen_paths: set[Path] = set()

    for d in scan_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            if f in seen_paths:
                continue
            seen_paths.add(f)
            # Skip summary / non-experiment files
            if f.name in {"all_experiments.json", "leaderboard.json"}:
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            # Enrich with path metadata
            data["_result_file"] = str(f)
            data["_dir"] = str(d)
            # Infer arch from path
            if "trocr" in f.name.lower() or "yolo" in f.name.lower():
                data["_arch"] = "trocr"
            else:
                data["_arch"] = "donut"
            results.append(data)

    return results


def _get_f1(result: dict, field: str | None) -> float:
    """Extract the relevant F1 from a result dict."""
    metrics = result.get("metrics", {})
    if field:
        return float(metrics.get(f"{field}_f1", math.nan))
    # For multi-seed results, use mean F1
    global_f1 = metrics.get("global_f1")
    if isinstance(global_f1, dict):
        return float(global_f1.get("mean", math.nan))
    return float(global_f1 if global_f1 is not None else math.nan)


def build_leaderboard(
    results_root: Path,
    field: str | None = None,
    arch: str | None = None,
) -> list[dict]:
    """Build a sorted leaderboard list."""
    all_results = _scan_results(results_root)

    if arch:
        all_results = [r for r in all_results if r.get("_arch", "donut") == arch.lower()]

    # Attach F1 for sorting
    for r in all_results:
        r["_f1"] = _get_f1(r, field)

    # Sort by F1 descending (NaN last)
    all_results.sort(
        key=lambda r: (not math.isnan(r["_f1"]), r["_f1"]),
        reverse=True,
    )
    return all_results


def print_leaderboard(
    leaderboard: list[dict],
    field: str | None = None,
    use_rich: bool | None = None,
) -> None:
    """Print leaderboard as a table."""
    if use_rich is None:
        try:
            import rich  # noqa: F401
            use_rich = True
        except ImportError:
            use_rich = False

    f1_header = f"{field}_f1" if field else "global_f1"

    if use_rich:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title=f"DONUT SROIE Results Leaderboard — ranked by {f1_header}",
                show_header=True,
                header_style="bold cyan",
                show_lines=True,
            )
            table.add_column("Rank", justify="right", style="dim", width=5)
            table.add_column("Exp ID", justify="right", width=7)
            table.add_column("Name", width=40)
            table.add_column("Arch", width=6)
            table.add_column(f1_header, justify="right", style="green", width=10)
            table.add_column("Samples", justify="right", width=9)
            table.add_column("File", style="dim", width=30)

            for rank, r in enumerate(leaderboard, 1):
                exp_id = str(r.get("experiment_id", "?"))
                name = str(r.get("name", ""))[:38]
                arch = r.get("_arch", "donut")
                f1_val = r["_f1"]
                f1_str = f"{f1_val:.4f}" if not math.isnan(f1_val) else "—"
                samples = str(r.get("num_train_samples", "?"))
                file_path = Path(r.get("_result_file", "")).name
                table.add_row(
                    str(rank), exp_id, name, arch, f1_str, samples, file_path
                )
            console.print(table)
            return
        except Exception:
            pass  # fall through to plain text

    # Plain text
    header = (
        f"{'Rank':<5} {'ID':<5} {'Name':<40} {'Arch':<6} "
        f"{f1_header:<12} {'Samples':<9} {'File':<30}"
    )
    print("\n" + "=" * len(header))
    print(f"DONUT SROIE Leaderboard — sorted by {f1_header}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(leaderboard, 1):
        exp_id = str(r.get("experiment_id", "?"))
        name = str(r.get("name", ""))[:38]
        arch = r.get("_arch", "donut")
        f1_val = r["_f1"]
        f1_str = f"{f1_val:.4f}" if not math.isnan(f1_val) else "—"
        samples = str(r.get("num_train_samples", "?"))
        file_path = Path(r.get("_result_file", "")).name[:28]
        print(
            f"{rank:<5} {exp_id:<5} {name:<40} {arch:<6} "
            f"{f1_str:<12} {samples:<9} {file_path:<30}"
        )
    print("=" * len(header) + "\n")


def export_csv(leaderboard: list[dict], results_root: Path, field: str | None = None) -> Path:
    """Export leaderboard to a dated CSV file."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = results_root / f"leaderboard_{date_str}.csv"
    f1_header = f"{field}_f1" if field else "global_f1"

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "experiment_id", "name", "arch", f1_header, "num_train_samples", "result_file"])
        for rank, r in enumerate(leaderboard, 1):
            f1_val = r["_f1"]
            writer.writerow([
                rank,
                r.get("experiment_id", ""),
                r.get("name", ""),
                r.get("_arch", ""),
                f"{f1_val:.4f}" if not math.isnan(f1_val) else "",
                r.get("num_train_samples", ""),
                r.get("_result_file", ""),
            ])
    print(f"[ResultsBrowser] Leaderboard exported to: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse and rank DONUT SROIE experiment results"
    )
    parser.add_argument(
        "--field",
        choices=["company", "date", "address", "total"],
        default=None,
        help="Sort by per-field F1 instead of global F1",
    )
    parser.add_argument(
        "--arch",
        choices=["donut", "trocr"],
        default=None,
        help="Filter by architecture",
    )
    parser.add_argument(
        "--export",
        choices=["csv"],
        default=None,
        help="Export leaderboard to CSV (results/leaderboard_YYYYMMDD.csv)",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root results directory (default: results)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.exists():
        print(f"[ResultsBrowser] No results directory found at: {results_root}")
        return

    leaderboard = build_leaderboard(results_root, field=args.field, arch=args.arch)

    if not leaderboard:
        print("[ResultsBrowser] No results found.")
        return

    print_leaderboard(leaderboard, field=args.field)

    if args.export == "csv":
        export_csv(leaderboard, results_root, field=args.field)


if __name__ == "__main__":
    main()
