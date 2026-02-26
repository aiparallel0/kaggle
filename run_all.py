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
run_all.py — Single entry point that runs the complete DONUT SROIE pipeline.

Calling this one script does everything:
  0. Installs SROIE data (auto-clones from GitHub; uses official 626/347 train/test split)
  1. Verifies / downloads all auxiliary datasets + pre-downloads the base model
  2. Evaluates the pretrained CORD model as a zero-shot baseline on SROIE test
  3. Trains DONUT (from the CORD checkpoint) for each of the 8 experiment configurations
  4. Evaluates every fine-tuned model on the SROIE test set
  5. Saves per-experiment JSON metrics to results/
  6. Writes a summary JSON   results/all_experiments.json
  7. Generates LaTeX table bodies (printed to stdout)
  8. Produces paper_filled.tex — the complete paper with all numeric results filled in

All stages run SEQUENTIALLY to prevent GPU memory contention.  No background
threads or concurrent GPU access.

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

  # Force re-run (delete cached results first):
      python run_all.py --force

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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# StageResult — structured output from each pipeline stage
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Result of a single pipeline stage execution."""

    name: str
    duration: float  # seconds
    exit_status: int  # 0=success, 1=partial, 2=fatal
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


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
# Stage 0 — SROIE auto-install
# ---------------------------------------------------------------------------

def stage_install(args) -> StageResult:
    """Clone SROIE from GitHub and set up train/val/test directories using 80/10/10 split."""
    _banner("STAGE 0 — SROIE data install")
    warnings: List[str] = []

    sroie_data_dir = Path(args.sroie_dir)
    sroie_img = sroie_data_dir / "img"
    sroie_test_img = sroie_data_dir / "test_img"
    sroie_val_img = sroie_data_dir / "val_img"

    if sroie_img.exists() and sroie_test_img.exists() and sroie_val_img.exists():
        print(f"  SROIE data already present at {sroie_data_dir} — skipping install.")
        return StageResult(name="SROIE Install", duration=0.0, exit_status=0,
                           warnings=warnings)

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

    def _move_split(img_list, dst_img_dir, dst_key_dir):
        """Move (not copy) val/test images out of img/ so train set is clean."""
        for img_path in img_list:
            shutil.move(str(img_path), str(dst_img_dir / img_path.name))
            for ext in [".txt", ".json"]:
                key_src = key_dir / (img_path.stem + ext)
                if key_src.exists():
                    shutil.move(str(key_src), str(dst_key_dir / key_src.name))
                    break

    _move_split(val_imgs, val_img_dir, val_key_dir)
    _move_split(test_imgs, test_img_dir, test_key_dir)

    # Verify: img/ should now contain exactly the train images
    remaining = sum(
        1 for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    )
    print(f"  Train images : {remaining} (in img/, after moving val+test out)")
    print(f"  Val images   : {len(val_imgs)} (in val_img/)")
    print(f"  Test images  : {len(test_imgs)} (in test_img/)")
    if remaining != len(train_imgs):
        print(
            f"  WARNING: Expected {len(train_imgs)} train images in img/ "
            f"but found {remaining}",
            file=sys.stderr,
        )
    print(f"  SROIE data ready at {sroie_data_dir}")

    return StageResult(name="SROIE Install", duration=0.0, exit_status=0,
                       warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 1 — Dataset verification / download + inline model pre-download
# ---------------------------------------------------------------------------

def stage_download(args) -> StageResult:
    """Verify SROIE exists, fetch auxiliary datasets in parallel, then download model inline."""
    import dataset_loaders  # local module
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _banner("STAGE 1 — Dataset verification & download")
    warnings: List[str] = []

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
        w = "SROIE test split not found or empty — evaluation will be skipped."
        print(f"  WARNING: {w}", file=sys.stderr)
        warnings.append(w)

    # Download all auxiliary datasets in parallel (I/O-bound, threads are fine)
    aux_datasets = ["wildreceipt", "cord", "invoices_donut"]
    failed_datasets: List[str] = []

    def _fetch_one(ds_name: str) -> tuple:
        """Download a single dataset. Returns (name, count, error, elapsed)."""
        try:
            t0 = time.monotonic()
            data = dataset_loaders.get_combined_dataset([ds_name])
            train_count = len(data[0]) if isinstance(data, tuple) else len(data)
            elapsed = time.monotonic() - t0
            return (ds_name, train_count, None, elapsed)
        except Exception as exc:
            return (ds_name, 0, str(exc), 0.0)

    print(f"  Downloading {len(aux_datasets)} datasets in parallel ...")
    counts = {
        "sroie": len(train_samples),
        "wildreceipt": 0,
        "cord": 0,
        "invoices_donut": 0,
    }
    with ThreadPoolExecutor(max_workers=len(aux_datasets)) as pool:
        futures = {pool.submit(_fetch_one, ds): ds for ds in aux_datasets}
        for future in as_completed(futures):
            ds_name, count, error, elapsed = future.result()
            if error:
                w = f"'{ds_name}' failed: {error}"
                print(f"    → WARNING: {w}")
                failed_datasets.append(ds_name)
                warnings.append(w)
            else:
                counts[ds_name] = count
                print(f"    → '{ds_name}' ready: {count} train samples ({elapsed:.1f}s)")

    print(f"\n  ┌{'─' * 50}┐")
    print(f"  │ {'Dataset':<25} {'Samples':>10} {'Status':>12} │")
    print(f"  ├{'─' * 50}┤")
    for ds_name in ["sroie", "wildreceipt", "cord", "invoices_donut"]:
        count = counts.get(ds_name, 0)
        status = "✓ OK" if count > 0 else "✗ EMPTY"
        print(f"  │ {ds_name:<25} {count:>10} {status:>12} │")
    print(f"  └{'─' * 50}┘")

    if failed_datasets:
        # Report which experiment IDs are affected by the failed downloads
        import run_experiments as re_mod
        affected_exp_ids = [
            exp_id for exp_id, exp in re_mod.EXPERIMENTS.items()
            if any(ds in exp.datasets for ds in failed_datasets)
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

    # Inline (blocking) model pre-download — NO background thread.
    # This ensures the model is fully cached before any training starts,
    # preventing GPU memory contention from concurrent downloads.
    print("\n  Pre-downloading base model (blocking) ...")
    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
        DonutProcessor.from_pretrained(model_id)
        VisionEncoderDecoderModel.from_pretrained(model_id)
        print("  Base model cached ✓")
    except Exception as e:
        w = f"Model pre-download failed: {e}"
        print(f"  WARNING: {w}")
        warnings.append(w)

    exit_status = 1 if failed_datasets else 0
    return StageResult(name="Dataset Download", duration=0.0, exit_status=exit_status,
                       warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 1.5 — Pretrained baseline evaluation
# ---------------------------------------------------------------------------

def stage_pretrained_baseline(args) -> StageResult:
    """Evaluate the pretrained CORD model as a zero-shot baseline on SROIE test."""
    import torch
    import dataset_loaders
    import evaluate as eval_mod
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    _banner("STAGE 1.5 — Pretrained baseline evaluation (zero-shot CORD)")
    warnings: List[str] = []

    workspace = Path(args.workspace)
    output_path = workspace / "evaluation_results.json"

    test_samples = dataset_loaders.load_sroie_test()
    if len(test_samples) == 0:
        w = "SROIE test split is empty — skipping pretrained baseline evaluation."
        print(f"  WARNING: {w}", file=sys.stderr)
        warnings.append(w)
        return StageResult(name="Pretrained Eval", duration=0.0, exit_status=1,
                           warnings=warnings)

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

    return StageResult(name="Pretrained Eval", duration=0.0, exit_status=0,
                       warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 2 — Experiments (train + eval)
# ---------------------------------------------------------------------------

def stage_experiments(args) -> StageResult:
    """Run all (or a single) experiment(s) SEQUENTIALLY. Returns StageResult."""
    import run_experiments as re_mod  # local module

    _banner("STAGE 2 — Experiments (train + evaluate)")
    warnings: List[str] = []

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Honour --force by clearing cached result files first
    if getattr(args, "force", False):
        for result_file in results_dir.glob("experiment_*.json"):
            result_file.unlink()
            print(f"  [force] Deleted cached result: {result_file}")
        summary_file = results_dir / "all_experiments.json"
        if summary_file.exists():
            summary_file.unlink()
            print(f"  [force] Deleted cached summary: {summary_file}")

    exp_ids = [args.experiment] if args.experiment else list(re_mod.EXPERIMENTS.keys())
    total = len(exp_ids)
    had_empty = False
    completed = 0
    succeeded = 0
    failed_experiments: List[int] = []

    for i, exp_id in enumerate(exp_ids, 1):
        _step(i, total, f"Experiment {exp_id}: {re_mod.EXPERIMENTS[exp_id].name}")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"  ▶ Experiment {exp_id} started at {ts}")
        print(f"    Datasets: {re_mod.EXPERIMENTS[exp_id].datasets}")
        t0 = time.monotonic()
        try:
            result = re_mod.run_experiment(exp_id)
        except Exception as exc:
            import traceback
            elapsed = time.monotonic() - t0
            ts_end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            w = f"Experiment {exp_id} crashed: {type(exc).__name__}: {exc}"
            print(f"\n  ✗ FATAL: {w}")
            print("    Traceback follows:")
            traceback.print_exc()
            failed_experiments.append(exp_id)
            warnings.append(w)
            had_empty = True
            continue
        elapsed = time.monotonic() - t0
        ts_end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        f1 = result.get("metrics", {}).get("global_f1", "N/A")
        print(f"  ◀ Experiment {exp_id} finished at {ts_end} ({elapsed:.1f}s)")
        print(f"    done in {elapsed / 60:.1f}min | F1={f1} "
              f"| Samples={result.get('num_train_samples', '?')}")
        completed += 1
        if result.get("num_train_samples", 0) == 0:
            had_empty = True
            failed_experiments.append(exp_id)
            warnings.append(f"Experiment {exp_id} had 0 training samples")
        else:
            succeeded += 1

    re_mod.save_summary()

    exit_status = 1 if had_empty else 0
    return StageResult(name="Experiments", duration=0.0, exit_status=exit_status,
                       warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 3 — LaTeX paper generation
# ---------------------------------------------------------------------------

def stage_paper(args) -> StageResult:
    """Generate LaTeX tables and fill paper_filled.tex."""
    import dataset_loaders
    import inject_results as ir  # local module

    _banner("STAGE 3 — LaTeX paper generation")
    warnings: List[str] = []

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

    ir.generate_convergence_data(str(results_path))
    ir.generate_convergence_tex(str(results_path))
    ir.generate_f1_barchart_tex(str(results_path))

    paper_template = Path(args.paper_template)
    output_paper = Path(args.output)

    if paper_template.exists():
        var_map = ir.build_var_map(all_exp)
        ir.fill_paper(str(paper_template), str(output_paper), var_map)
        print(f"\n  Complete paper written → {output_paper}")
    else:
        w = (f"paper template not found at {paper_template}; "
             f"skipping paper_filled.tex generation.")
        print(f"  WARNING: {w}")
        warnings.append(w)

    return StageResult(name="Paper Generation", duration=0.0, exit_status=0,
                       warnings=warnings)


# ---------------------------------------------------------------------------
# PipelineOrchestrator — sequential execution with structured results
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Orchestrates all pipeline stages SEQUENTIALLY with timing and status tracking."""

    def __init__(self, args):
        self.args = args
        self.stages: List[StageResult] = []

    def _run_stage(self, name: str, func) -> StageResult:
        """Time a stage, catch exceptions, and record the StageResult."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n  ▶ {name} started at {ts}")
        t0 = time.monotonic()
        try:
            result = func(self.args)
            result.duration = time.monotonic() - t0
        except SystemExit:
            raise
        except Exception as exc:
            import traceback
            elapsed = time.monotonic() - t0
            traceback.print_exc()
            result = StageResult(
                name=name, duration=elapsed, exit_status=2,
                warnings=[f"Uncaught exception: {type(exc).__name__}: {exc}"],
            )
        ts_end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"  ◀ {name} finished at {ts_end} ({result.duration:.1f}s)")
        self.stages.append(result)
        return result

    def run(self) -> int:
        """Run all stages sequentially and return the pipeline exit code."""
        exit_code = 0

        if self.args.paper_only:
            self._run_stage("Paper Generation", stage_paper)
            print(repr(self))
            return exit_code

        # Stage 0 — SROIE install
        if not self.args.skip_install:
            self._run_stage("SROIE Install", stage_install)

        # Stage 1 — Dataset download + inline model pre-download
        if not self.args.skip_download:
            r = self._run_stage("Dataset Download", stage_download)
            if r.exit_status > exit_code:
                exit_code = r.exit_status

        # Stage 1.5 — Pretrained baseline evaluation
        if not self.args.skip_pretrained:
            self._run_stage("Pretrained Eval", stage_pretrained_baseline)

        # Stage 2 — Experiments (sequential, one at a time)
        r = self._run_stage("Experiments", stage_experiments)
        if r.exit_status > exit_code:
            exit_code = r.exit_status

        # Stage 3 — Paper generation
        self._run_stage("Paper Generation", stage_paper)

        print(repr(self))
        return exit_code

    def __repr__(self) -> str:
        """Phase 5A visual output — formatted pipeline execution summary table."""
        W = 70  # inner width
        lines = []
        lines.append(f"╔{'═' * W}╗")
        title = "PIPELINE EXECUTION SUMMARY"
        lines.append(f"║{title:^{W}}║")
        lines.append(f"╠{'═' * W}╣")

        # Header
        hdr = (f"  {'Stage':<24}│ {'Duration':>8} │ {'Status':<7} "
               f"│ {'Warnings':<20}")
        lines.append(f"║{hdr:<{W}}║")

        sep = f"{'═' * 25}╪{'═' * 10}╪{'═' * 9}╪{'═' * (W - 46)}"
        lines.append(f"╠{sep}╣")

        # Stage labels for display order
        stage_labels = {
            "SROIE Install": "0. SROIE Install",
            "Dataset Download": "1. Dataset Download",
            "Pretrained Eval": "1.5 Pretrained Eval",
            "Experiments": "2. Experiments",
            "Paper Generation": "3. Paper Generation",
        }

        for sr in self.stages:
            label = stage_labels.get(sr.name, sr.name)
            dur = f"{sr.duration:.1f}s"
            if sr.exit_status == 0:
                status = "✓ OK"
            elif sr.exit_status == 1:
                status = "⚠ PART"
            else:
                status = "✗ FAIL"
            warn_text = f"{len(sr.warnings)} warning{'s' if len(sr.warnings) != 1 else ''}" if sr.warnings else ""
            row = (f"  {label:<24}│ {dur:>8} │ {status:<7} "
                   f"│ {warn_text:<20}")
            lines.append(f"║{row:<{W}}║")

        lines.append(f"╚{'═' * W}╝")
        return "\n".join(lines)


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

    # ── Diagnostic: environment snapshot ──────────────────────────────────
    import platform
    import importlib
    import torch
    _banner("ENVIRONMENT DIAGNOSTICS")
    print(f"  Python       : {platform.python_version()} ({sys.executable})")
    print(f"  Platform     : {platform.platform()}")
    print(f"  CUDA avail   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device  : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version : {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU memory   : {gpu_mem:.1f} GB")
    for pkg in ["torch", "transformers", "datasets", "accelerate", "huggingface_hub",
                "sentencepiece", "editdistance", "pandas", "numpy", "Pillow"]:
        try:
            mod = importlib.import_module(
                pkg.replace("-", "_").lower() if pkg != "Pillow" else "PIL"
            )
            ver = getattr(mod, "__version__", "?")
            print(f"  {pkg:20s}: {ver}")
        except ImportError:
            print(f"  {pkg:20s}: NOT INSTALLED")
    print(f"  Workspace    : {args.workspace}")
    print(f"  SROIE dir    : {args.sroie_dir}")
    print(f"  CWD          : {Path.cwd()}")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  RAM          : {mem.total / (1024 ** 3):.1f} GB total, "
              f"{mem.available / (1024 ** 3):.1f} GB available")
    except ImportError:
        pass
    print(f"  CPU cores    : {os.cpu_count()}")

    # Run the pipeline via the orchestrator (all stages sequential)
    orchestrator = PipelineOrchestrator(args)
    exit_code = orchestrator.run()

    total_elapsed = time.monotonic() - t_start
    _banner(f"DONE — total wall time {total_elapsed / 60:.1f} min  |  exit code {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
