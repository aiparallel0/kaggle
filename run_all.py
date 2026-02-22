"""
run_all.py — Single entry point that runs the complete DONUT SROIE pipeline.

Calling this one script does everything:
  1. Verifies / downloads all auxiliary datasets
  2. Trains DONUT (from the CORD checkpoint) for each of the 7 experiment configurations
  3. Evaluates every fine-tuned model on the SROIE test set
  4. Saves per-experiment JSON metrics to results/
  5. Writes a summary JSON   results/all_experiments.json
  6. Generates LaTeX table bodies (printed to stdout)
  7. Produces paper_filled.tex — the complete paper with all numeric results filled in

Usage
-----
  # Full pipeline (all 7 experiments):
      python run_all.py

  # Single experiment only (skip the others, still generate paper at end):
      python run_all.py --experiment 2

  # Skip training and go straight to paper generation (results must exist):
      python run_all.py --paper-only

  # Override where SROIE data and workspace dirs live:
      python run_all.py --sroie-dir /data/SROIE --workspace /workspace

  # Change output paper filename:
      python run_all.py --output my_paper.tex

Exit codes
----------
  0  — Success
  1  — One or more experiments had no training data (results saved as partial)
  2  — Fatal error (missing SROIE data, etc.)
"""

import argparse
import json
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    width = 72
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}")


def _step(n: int, total: int, desc: str) -> None:
    print(f"\n[{n}/{total}] {desc}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Stage 1 — Dataset verification / download
# ---------------------------------------------------------------------------

def stage_download(args) -> None:
    """Verify SROIE exists and pre-fetch all auxiliary datasets."""
    import dataset_loaders  # local module

    _banner("STAGE 1 — Dataset verification & download")

    # SROIE must already be present
    sroie_img = Path(args.sroie_dir) / "img"
    sroie_test_img = Path(args.sroie_dir) / "test_img"
    if not sroie_img.exists() or not sroie_test_img.exists():
        print(f"ERROR: SROIE data not found at {args.sroie_dir}.", file=sys.stderr)
        print("       Expected subdirs: img/, key/, test_img/, test_key/", file=sys.stderr)
        sys.exit(2)

    train_samples = dataset_loaders.load_sroie_train()
    test_samples = dataset_loaders.load_sroie_test()
    print(f"  SROIE train : {len(train_samples)} samples")
    print(f"  SROIE test  : {len(test_samples)} samples")

    # Trigger downloads for all auxiliary datasets so they are cached before training
    aux_datasets = ["wildreceipt", "funsd", "xfund", "eaten", "cord", "kaggle_scanned"]
    for ds_name in aux_datasets:
        print(f"  Fetching '{ds_name}' ...")
        try:
            data = dataset_loaders.get_combined_dataset([ds_name])
            print(f"    → {len(data)} samples available")
        except Exception as exc:
            print(f"    → WARNING: failed to fetch '{ds_name}': {exc}. "
                  f"Any experiment that includes this dataset will produce 0 samples "
                  f"from it and fall back to SROIE-only training.")


# ---------------------------------------------------------------------------
# Stage 2 — Experiments (train + eval)
# ---------------------------------------------------------------------------

def stage_experiments(args) -> int:
    """Run all (or a single) experiment(s). Returns 0 on full success, 1 on partial."""
    import run_experiments as re_mod  # local module

    _banner("STAGE 2 — Experiments (train + evaluate)")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    exp_ids = [args.experiment] if args.experiment else list(re_mod.EXPERIMENTS.keys())
    total = len(exp_ids)
    had_empty = False

    for i, exp_id in enumerate(exp_ids, 1):
        _step(i, total, f"Experiment {exp_id}: {re_mod.EXPERIMENTS[exp_id]['name']}")
        t0 = time.monotonic()
        result = re_mod.run_experiment(exp_id)
        elapsed = time.monotonic() - t0
        f1 = result.get("metrics", {}).get("global_f1", "N/A")
        print(f"  Finished in {elapsed/60:.1f} min  |  Global F1 = {f1}")
        if result.get("num_train_samples", 0) == 0:
            had_empty = True

    re_mod.save_summary()
    return 1 if had_empty else 0


# ---------------------------------------------------------------------------
# Stage 3 — LaTeX paper generation
# ---------------------------------------------------------------------------

def stage_paper(args) -> None:
    """Generate LaTeX tables and fill paper_filled.tex."""
    import inject_results as ir  # local module

    _banner("STAGE 3 — LaTeX paper generation")

    results_path = Path("results") / "all_experiments.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found — run experiments first.", file=sys.stderr)
        sys.exit(2)

    with open(results_path) as fh:
        all_exp = json.load(fh)

    ir.print_table1_dataset_stats()
    ir.print_table2_experiments(all_exp)
    ir.print_table3_perfield(all_exp)
    ir.print_table4_leaderboard(all_exp)

    paper_template = Path(args.paper_template)
    output_paper = Path(args.output)

    if paper_template.exists():
        var_map = ir.build_var_map(all_exp)
        ir.fill_paper(str(paper_template), str(output_paper), var_map)
        print(f"\n  Complete paper written → {output_paper}")
    else:
        print(f"  WARNING: paper template not found at {paper_template}; "
              f"skipping paper_filled.tex generation.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_all.py",
        description="Complete DONUT SROIE pipeline: download → train → evaluate → paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--experiment", type=int, metavar="N",
        help="Run only experiment N (1–7) instead of all 7",
    )
    p.add_argument(
        "--paper-only", action="store_true",
        help="Skip download & training; only generate the paper from existing results",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip the dataset download/verification stage",
    )
    p.add_argument(
        "--sroie-dir", default="/workspace/ICDAR-2019-SROIE/data",
        metavar="PATH",
        help="Path to SROIE data directory (default: /workspace/ICDAR-2019-SROIE/data)",
    )
    p.add_argument(
        "--workspace", default="/workspace",
        metavar="PATH",
        help="Workspace root for model checkpoints (default: /workspace)",
    )
    p.add_argument(
        "--paper-template", default="paper.tex",
        metavar="FILE",
        help="LaTeX template to fill (default: paper.tex)",
    )
    p.add_argument(
        "--output", default="paper_filled.tex",
        metavar="FILE",
        help="Output filled LaTeX file (default: paper_filled.tex)",
    )
    return p


def main() -> None:
    t_start = time.monotonic()
    parser = build_parser()
    args = parser.parse_args()

    # Propagate workspace override to sub-modules before importing them
    import os
    os.environ.setdefault("DONUT_WORKSPACE", args.workspace)

    exit_code = 0

    if args.paper_only:
        stage_paper(args)
    else:
        if not args.skip_download:
            stage_download(args)
        exit_code = stage_experiments(args)
        stage_paper(args)

    elapsed = time.monotonic() - t_start
    _banner(f"DONE — total wall time {elapsed/60:.1f} min  |  exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
