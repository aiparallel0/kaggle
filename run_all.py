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
import random
import shutil
import subprocess
import sys
import threading
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
# HuggingFace authentication
# ---------------------------------------------------------------------------

def _setup_hf_auth() -> None:
    """Load HuggingFace token from hf_token.txt or environment for faster downloads.

    Authenticated HF downloads are 5-10x faster and avoid rate limiting (429 errors).
    """
    # Priority: HF_TOKEN env var > hf_token.txt file
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token_file = Path("hf_token.txt")
        if token_file.exists():
            token = token_file.read_text().strip()
            if token and not token.startswith("#"):
                print(f"  [HF Auth] Loaded token from {token_file}")
            else:
                token = None
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token  # legacy env var
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("  [HF Auth] Authenticated — faster downloads enabled")
        except Exception as e:
            print(f"  [HF Auth] Login failed: {e} — continuing unauthenticated")
    else:
        print(
            "  [HF Auth] No token found. Downloads may be slow/rate-limited.\n"
            "            To fix: create hf_token.txt with your HF token,\n"
            "            or set HF_TOKEN environment variable."
        )


# ---------------------------------------------------------------------------
# Background model pre-download
# ---------------------------------------------------------------------------

def _predownload_model_background() -> threading.Thread:
    """Start downloading the base model in a background thread.

    This runs concurrently with dataset downloads so the model is
    already cached when training starts.
    """
    def _download():
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
            print("  [Background] Pre-downloading base model ...")
            DonutProcessor.from_pretrained(model_id)
            VisionEncoderDecoderModel.from_pretrained(model_id)
            print("  [Background] Base model cached ✓")
        except Exception as e:
            print(f"  [Background] Model pre-download failed: {e}")

    t = threading.Thread(target=_download, daemon=True, name="model-predownload")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Stage 0 — SROIE auto-install
# ---------------------------------------------------------------------------

def stage_install(args) -> None:
    """Clone SROIE from GitHub and set up train/val/test directories using 80/10/10 split."""
    _banner("STAGE 0 — SROIE data install")

    sroie_data_dir = Path(args.sroie_dir)
    sroie_img = sroie_data_dir / "img"
    sroie_test_img = sroie_data_dir / "test_img"
    sroie_val_img = sroie_data_dir / "val_img"

    if sroie_img.exists() and sroie_test_img.exists() and sroie_val_img.exists():
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

    # The repo has data/img/, data/key/, data/box/ with ALL 626 images.
    # The official SROIE test set (347 images) has NO public ground truth labels,
    # so we use an 80/10/10 split instead: 500 train / 63 val / 63 test.
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

    # Collect all images with matching key files
    all_images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    )
    # Keep only images that have a corresponding key file
    valid_images = [
        p for p in all_images
        if (key_dir / (p.stem + ".txt")).exists() or (key_dir / (p.stem + ".json")).exists()
    ]

    # 80/10/10 split with seed 42
    # Official test split has no public ground truth — use our own split instead.
    rng = random.Random(42)
    shuffled = list(valid_images)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = (n - n_train) // 2
    train_imgs = shuffled[:n_train]
    val_imgs = shuffled[n_train:n_train + n_val]
    test_imgs = shuffled[n_train + n_val:]

    val_img_dir = sroie_data_dir / "val_img"
    val_key_dir = sroie_data_dir / "val_key"
    test_img_dir = sroie_data_dir / "test_img"
    test_key_dir = sroie_data_dir / "test_key"

    val_img_dir.mkdir(exist_ok=True)
    val_key_dir.mkdir(exist_ok=True)
    test_img_dir.mkdir(exist_ok=True)
    test_key_dir.mkdir(exist_ok=True)

    def _copy_split(img_list, dst_img_dir, dst_key_dir):
        for img_path in img_list:
            shutil.copy2(str(img_path), str(dst_img_dir / img_path.name))
            for ext in [".txt", ".json"]:
                key_src = key_dir / (img_path.stem + ext)
                if key_src.exists():
                    shutil.copy2(str(key_src), str(dst_key_dir / key_src.name))
                    break

    _copy_split(val_imgs, val_img_dir, val_key_dir)
    _copy_split(test_imgs, test_img_dir, test_key_dir)

    print(f"  Train images : {len(train_imgs)}")
    print(f"  Val images   : {len(val_imgs)}")
    print(f"  Test images  : {len(test_imgs)}")
    print(f"  SROIE data ready at {sroie_data_dir}")


# ---------------------------------------------------------------------------
# Stage 1 — Dataset verification / download
# ---------------------------------------------------------------------------

def stage_download(args) -> None:
    """Verify SROIE exists and pre-fetch all auxiliary datasets IN PARALLEL."""
    import dataset_loaders  # local module
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _banner("STAGE 1 — Dataset verification & parallel download")

    # Start model download in background (runs during dataset downloads)
    model_thread = _predownload_model_background()

    # SROIE must already be present (img/ directory at minimum)
    sroie_img = Path(args.sroie_dir) / "img"
    sroie_test_img = Path(args.sroie_dir) / "test_img"
    if not sroie_img.exists():
        print(f"ERROR: SROIE data not found at {args.sroie_dir}.", file=sys.stderr)
        print("       Expected subdirs: img/, key/", file=sys.stderr)
        sys.exit(2)

    train_samples = dataset_loaders.load_sroie_train()
    test_samples = dataset_loaders.load_sroie_test()
    val_samples = dataset_loaders.load_sroie_val()
    print(f"  SROIE train : {len(train_samples)} samples")
    print(f"  SROIE val   : {len(val_samples)} samples")
    print(f"  SROIE test  : {len(test_samples)} samples")
    if not sroie_test_img.exists() or len(test_samples) == 0:
        print(
            "  WARNING: SROIE test split not found or empty — evaluation will be skipped.",
            file=sys.stderr,
        )

    # Download all auxiliary datasets in parallel (I/O-bound, threads are fine)
    aux_datasets = ["wildreceipt", "cord", "invoices_donut"]
    failed_datasets = []

    def _fetch_one(ds_name: str) -> tuple:
        """Download a single dataset. Returns (name, count, error)."""
        try:
            t0 = time.monotonic()
            data = dataset_loaders.get_combined_dataset([ds_name])
            train_count = len(data[0]) if isinstance(data, tuple) else len(data)
            elapsed = time.monotonic() - t0
            return (ds_name, train_count, None, elapsed)
        except Exception as exc:
            return (ds_name, 0, str(exc), 0.0)

    print(f"  Downloading {len(aux_datasets)} datasets in parallel ...")
    with ThreadPoolExecutor(max_workers=len(aux_datasets)) as pool:
        futures = {pool.submit(_fetch_one, ds): ds for ds in aux_datasets}
        for future in as_completed(futures):
            ds_name, count, error, elapsed = future.result()
            if error:
                print(f"    → WARNING: '{ds_name}' failed: {error}")
                failed_datasets.append(ds_name)
            else:
                print(f"    → '{ds_name}' ready: {count} train samples ({elapsed:.1f}s)")

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

    # Wait for background model download to finish
    model_thread.join(timeout=600)  # 10 min max


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
            "sroie_val": len(dataset_loaders.load_sroie_val()),
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

    # Set up HuggingFace authentication for faster downloads
    _setup_hf_auth()

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
