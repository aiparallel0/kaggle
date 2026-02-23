"""
run_all.py — Single entry point that runs the complete DONUT SROIE pipeline.

Calling this one script does everything:
  0. Installs SROIE data (auto-clones from GitHub; uses official 626/347 train/test split)
  1. Verifies / downloads all auxiliary datasets
  2. Trains DONUT (from the CORD checkpoint) for each of the 8 experiment configurations
  3. Evaluates every fine-tuned model on the SROIE test set
  4. Saves per-experiment JSON metrics to results/
  5. Writes a summary JSON   results/all_experiments.json
  6. Generates LaTeX table bodies (printed to stdout)
  7. Produces paper_filled.tex — the complete paper with all numeric results filled in

Usage
-----
  # Full pipeline (all 8 experiments):
      python run_all.py

  # Single experiment only (skip the others, still generate paper at end):
      python run_all.py --experiment 2

  # Skip training and go straight to paper generation (results must exist):
      python run_all.py --paper-only

  # Override where SROIE data and workspace dirs live:
      python run_all.py --sroie-dir /data/SROIE --workspace /workspace

  # Skip SROIE auto-install (data already present):
      python run_all.py --skip-install

  # Skip pretrained baseline evaluation:
      python run_all.py --skip-pretrained

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
import os
import shutil
import subprocess
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
# Stage 0 — SROIE auto-install
# ---------------------------------------------------------------------------

def stage_install(args) -> None:
    """Clone SROIE from GitHub and set up train/test directories if not present."""
    _banner("STAGE 0 — SROIE data install")

    sroie_data_dir = Path(args.sroie_dir)
    sroie_img = sroie_data_dir / "img"
    sroie_test_img = sroie_data_dir / "test_img"

    if sroie_img.exists() and sroie_test_img.exists():
        print(f"  SROIE data already present at {sroie_data_dir} — skipping install.")
        return

    # Clone from GitHub into the parent directory of sroie_data_dir
    parent_dir = sroie_data_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    repo_url = "https://github.com/zzzDavid/ICDAR-2019-SROIE.git"
    clone_target = parent_dir / "ICDAR-2019-SROIE-repo"

    if not clone_target.exists():
        print(f"  Cloning {repo_url} ...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(clone_target)],
                check=True,
            )
        except subprocess.CalledProcessError:
            print(
                f"ERROR: Failed to clone SROIE repository from {repo_url}.",
                file=sys.stderr,
            )
            sys.exit(2)

    # The repo has data/img/, data/key/, data/box/ but no test split.
    # Point sroie_data_dir at the cloned repo's data/ subdirectory if needed.
    cloned_data = clone_target / "data"
    if not sroie_data_dir.exists() and cloned_data.exists():
        sroie_data_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(cloned_data), str(sroie_data_dir))

    # Verify source directories exist
    img_dir = sroie_data_dir / "img"
    key_dir = sroie_data_dir / "key"
    if not img_dir.exists():
        print(
            f"ERROR: Expected {img_dir} after clone — directory not found.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not key_dir.exists():
        print(
            f"ERROR: Expected {key_dir} after clone — directory not found.",
            file=sys.stderr,
        )
        sys.exit(2)

    image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

    # Try to use the official SROIE test split.
    # Check if the repo includes a separate test folder or split-list files.
    test_img_dir = sroie_data_dir / "test_img"
    test_key_dir = sroie_data_dir / "test_key"

    # Look for official test folder variants in the cloned repo
    official_test_found = False
    for test_candidate in (
        clone_target / "data" / "test",
        clone_target / "test",
        clone_target / "data" / "test_img",
    ):
        if test_candidate.exists() and any(
            p.is_file() and p.suffix.lower() in image_exts
            for p in test_candidate.iterdir()
        ):
            # Official test images found — copy to test_img/
            test_img_dir.mkdir(exist_ok=True)
            test_key_dir.mkdir(exist_ok=True)
            for img_path in test_candidate.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in image_exts:
                    shutil.copy2(str(img_path), str(test_img_dir / img_path.name))
            # Look for matching key files in a parallel key folder
            key_candidate = test_candidate.parent / (test_candidate.name.replace("img", "key"))
            if not key_candidate.exists():
                key_candidate = test_candidate.parent / "test_key"
            if key_candidate.exists():
                for kf in key_candidate.iterdir():
                    if kf.is_file():
                        shutil.copy2(str(kf), str(test_key_dir / kf.name))
            official_test_found = True
            img_count = sum(1 for p in test_img_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in image_exts)
            print(f"  Official SROIE test split found: {img_count} images at {test_img_dir}")
            break

    # Check for task3 split list files
    if not official_test_found:
        for split_file in (
            clone_target / "task3_test.txt",
            clone_target / "data" / "task3_test.txt",
            sroie_data_dir / "task3_test.txt",
        ):
            if split_file.exists():
                test_stems = set(split_file.read_text(encoding="utf-8").splitlines())
                test_stems = {s.strip() for s in test_stems if s.strip()}
                test_img_dir.mkdir(exist_ok=True)
                test_key_dir.mkdir(exist_ok=True)
                for img_path in img_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in image_exts:
                        if img_path.stem in test_stems:
                            shutil.move(str(img_path), str(test_img_dir / img_path.name))
                            for ext in [".txt", ".json"]:
                                key_src = key_dir / (img_path.stem + ext)
                                if key_src.exists():
                                    shutil.move(str(key_src), str(test_key_dir / key_src.name))
                                    break
                official_test_found = True
                img_count = sum(1 for p in test_img_dir.iterdir()
                                if p.is_file() and p.suffix.lower() in image_exts)
                print(f"  Official SROIE test split (task3_test.txt): {img_count} images")
                break

    if not official_test_found:
        # Fallback: use ALL available images as training; no test split.
        print(
            "  WARNING: The official SROIE test split (347 images) could not be determined "
            "from the cloned repository. ALL available images will be used as training data. "
            "Evaluation on the SROIE test set will be skipped (0 test samples).",
            file=sys.stderr,
        )
        # Create empty test directories to prevent FileNotFoundError downstream
        test_img_dir.mkdir(exist_ok=True)
        test_key_dir.mkdir(exist_ok=True)

    train_count = sum(1 for p in img_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in image_exts)
    test_count = sum(1 for p in test_img_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in image_exts)
    print(f"  Train images : {train_count}")
    print(f"  Test images  : {test_count}")
    print(f"  SROIE data ready at {sroie_data_dir}")


# ---------------------------------------------------------------------------
# Stage 1 — Dataset verification / download
# ---------------------------------------------------------------------------

def stage_download(args) -> None:
    """Verify SROIE exists and pre-fetch all auxiliary datasets."""
    import dataset_loaders  # local module

    _banner("STAGE 1 — Dataset verification & download")

    # SROIE must already be present (img/ directory at minimum)
    sroie_img = Path(args.sroie_dir) / "img"
    sroie_test_img = Path(args.sroie_dir) / "test_img"
    if not sroie_img.exists():
        print(f"ERROR: SROIE data not found at {args.sroie_dir}.", file=sys.stderr)
        print("       Expected subdirs: img/, key/", file=sys.stderr)
        sys.exit(2)

    train_samples = dataset_loaders.load_sroie_train()
    test_samples = dataset_loaders.load_sroie_test()
    print(f"  SROIE train : {len(train_samples)} samples")
    print(f"  SROIE test  : {len(test_samples)} samples")
    if not sroie_test_img.exists() or len(test_samples) == 0:
        print(
            "  WARNING: SROIE test split not found or empty — evaluation will be skipped.",
            file=sys.stderr,
        )

    # Trigger downloads for all auxiliary datasets so they are cached before training
    aux_datasets = ["wildreceipt", "coru", "cord", "invoices_donut"]
    failed_datasets = []
    for ds_name in aux_datasets:
        print(f"  Fetching '{ds_name}' ...")
        try:
            train_data, val_data = dataset_loaders.get_combined_dataset([ds_name])
            print(f"    → {len(train_data)} train / {len(val_data)} val samples available")
        except Exception as exc:
            print(f"    → WARNING: failed to fetch '{ds_name}': {exc}. "
                  f"Any experiment that includes this dataset will produce 0 samples "
                  f"from it and fall back to SROIE-only training.")
            failed_datasets.append(ds_name)

    if failed_datasets:
        # Report which experiment IDs are affected by the failed downloads
        import run_experiments as re_mod
        affected_exp_ids = [
            exp_id for exp_id, exp in re_mod.EXPERIMENTS.items()
            if any(ds in exp["datasets"] for ds in failed_datasets)
        ]
        print(
            f"\n  WARNING: {len(failed_datasets)} auxiliary dataset(s) failed to load: "
            f"{failed_datasets}",
            file=sys.stderr,
        )
        print(
            f"  Affected experiment IDs: {affected_exp_ids}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Stage 1.5 — Pretrained baseline evaluation
# ---------------------------------------------------------------------------

def stage_pretrained_baseline(args) -> None:
    """Evaluate the pretrained CORD model as a zero-shot baseline on SROIE test."""
    import torch
    import dataset_loaders
    import evaluate as eval_mod
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    _banner("STAGE 1.5 — Pretrained baseline evaluation (zero-shot CORD)")

    workspace = Path(args.workspace)
    output_path = workspace / "evaluation_results.json"

    test_samples = dataset_loaders.load_sroie_test()
    if len(test_samples) == 0:
        print(
            "  WARNING: SROIE test split is empty — skipping pretrained baseline evaluation.",
            file=sys.stderr,
        )
        return

    print(f"  Evaluating on {len(test_samples)} test images ...")

    ground_truths = [s[1] for s in test_samples]
    image_paths = [s[0] for s in test_samples]

    pretrained_model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
    print(f"  Loading pretrained model: {pretrained_model_id}")
    pre_processor = DonutProcessor.from_pretrained(pretrained_model_id)
    pre_model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_id).to(
        eval_mod.DEVICE
    )
    pre_model.eval()

    pretrained_preds = []
    with torch.no_grad():
        for img_path in image_paths:
            raw = eval_mod.run_inference(pre_model, pre_processor, img_path, "<s_cord-v2>")
            pretrained_preds.append(eval_mod.remap_cord_to_sroie(raw))

    pretrained_metrics = eval_mod.compute_metrics(pretrained_preds, ground_truths)
    print(f"  Pretrained Global F1 = {pretrained_metrics.get('global_f1', 'N/A')}")

    workspace.mkdir(parents=True, exist_ok=True)
    output = {"pretrained_metrics": pretrained_metrics}
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    print(f"  Saved → {output_path}")


# ---------------------------------------------------------------------------
# Stage 2 — Experiments (train + eval)
# ---------------------------------------------------------------------------

def stage_experiments(args) -> int:
    """Run all (or a single) experiment(s). Returns 0 on full success, 1 on partial."""
    import run_experiments as re_mod  # local module

    _banner("STAGE 2 — Experiments (train + evaluate)")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # FIX (BUG 3): honour --force by clearing cached result files first
    if getattr(args, "force", False):
        for result_file in results_dir.glob("experiment_*.json"):
            result_file.unlink()
            print(f"[force] Deleted cached result: {result_file}")

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
    import dataset_loaders
    import inject_results as ir  # local module

    _banner("STAGE 3 — LaTeX paper generation")

    results_path = Path("results") / "all_experiments.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found — run experiments first.", file=sys.stderr)
        sys.exit(2)

    with open(results_path) as fh:
        all_exp = json.load(fh)

    # Compute actual dataset counts for Table 1
    try:
        actual_counts = {
            "sroie_train": len(dataset_loaders.load_sroie_train()),
            "sroie_test": len(dataset_loaders.load_sroie_test()),
        }
    except Exception:
        actual_counts = None

    ir.print_table1_dataset_stats(actual_counts)
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
        help="Run only experiment N (1–8) instead of all 8",
    )
    p.add_argument(
        "--force", action="store_true",
        # FIX (BUG 3): Delete all cached results and re-run every experiment.
        help="Delete all cached experiment results and re-run from scratch",
    )
    p.add_argument(
        "--paper-only", action="store_true",
        help="Skip download & training; only generate the paper from existing results",
    )
    p.add_argument(
        "--skip-install", action="store_true",
        help="Skip Stage 0 SROIE auto-install (data already present)",
    )
    p.add_argument(
        "--skip-download", action="store_true",
        help="Skip the dataset download/verification stage",
    )
    p.add_argument(
        "--skip-pretrained", action="store_true",
        help="Skip pretrained baseline evaluation step",
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

    # Propagate workspace and SROIE dir overrides to sub-modules before importing them
    os.environ["DONUT_WORKSPACE"] = args.workspace
    os.environ["SROIE_DATA_DIR"] = args.sroie_dir

    exit_code = 0

    if args.paper_only:
        stage_paper(args)
    else:
        if not args.skip_install:
            t0 = time.monotonic()
            stage_install(args)
            print(f"  [Stage 0 elapsed: {time.monotonic()-t0:.1f}s]")
        if not args.skip_download:
            t0 = time.monotonic()
            stage_download(args)
            print(f"  [Stage 1 elapsed: {time.monotonic()-t0:.1f}s]")
        if not args.skip_pretrained:
            t0 = time.monotonic()
            stage_pretrained_baseline(args)
            print(f"  [Stage 1.5 elapsed: {time.monotonic()-t0:.1f}s]")
        t0 = time.monotonic()
        exit_code = stage_experiments(args)
        print(f"  [Stage 2 elapsed: {time.monotonic()-t0:.1f}s]")
        t0 = time.monotonic()
        stage_paper(args)
        print(f"  [Stage 3 elapsed: {time.monotonic()-t0:.1f}s]")

    elapsed = time.monotonic() - t_start
    _banner(f"DONE — total wall time {elapsed/60:.1f} min  |  exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
