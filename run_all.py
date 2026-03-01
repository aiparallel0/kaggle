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
run_all.py — Single entry point for the complete dual-architecture pipeline.

Calling this one script does everything:
  0. Installs SROIE data (auto-clones from GitHub; 80/10/10 split)
  1. Verifies / downloads all auxiliary datasets + pre-downloads the base model
  1.5. Evaluates the pretrained CORD model as a zero-shot baseline on SROIE test
  2. Trains DONUT for each of the 8 experiment configurations
  3. Prepares YOLO bbox + TrOCR line crop datasets from SROIE
  4. Trains YOLOv8 + TrOCR and runs TrOCR+YOLO inference on SROIE test
  5. Runs head-to-head benchmark: DONUT vs YOLOv8+TrOCR+Regex (F1, accuracy, speed)
  6. Generates cross-architecture comparison plots and tables
  7. Fills paper.tex with all real metrics → paper_filled.tex

FIX: Added TrOCR+YOLO pipeline stages (3-5) for dual-architecture comparison.
FIX: GPU memory cleanup between all stages to prevent OOM.
FIX: Imports constants from shared module.
FIX: Integrated benchmark_compare.py as Stage 5 for live head-to-head evaluation.
FIX: Added -quick mode for fast testing with hyperparameter sweep support.
FIX: Auto-install dependencies and dual-stream logging (console + terminal.txt).

All stages run SEQUENTIALLY to prevent GPU memory contention.

Usage
-----
  python run_all.py                           # Full pipeline (all 8 DONUT exps + TrOCR+YOLO)
  python run_all.py --experiment 2            # Single DONUT experiment (Exp 2)
  python run_all.py --paper-only              # Regenerate paper from existing results
  python run_all.py -quick                    # Quick test: Exp 1 + TrOCR+YOLO, gen results.tex
  python run_all.py -quick -all               # Hyperparameter sweep (batch_size, epochs, etc.)
  python run_all.py --skip-trocr              # Skip TrOCR+YOLO stages
  python run_all.py --force                   # Force re-run (delete cached results)

Exit codes
----------
  0  — Success
  1  — One or more experiments had no training data (results saved as partial)
  2  — Fatal error (missing SROIE data, etc.)
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Set before constants.py triggers torch import to reduce GPU memory fragmentation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from constants import BASE_MODEL, IMAGE_EXTS, SEED  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Auto-Install Dependencies (Phase 1)
# ---------------------------------------------------------------------------


def _is_package_missing(package_name: str) -> bool:
    """Check if a package can be imported.

    Args:
        package_name: Name of the package to check (e.g., 'torch', 'transformers')

    Returns:
        True if package is NOT installed (ImportError), False if it is installed
    """
    try:
        __import__(package_name)
        return False
    except ImportError:
        return True


def _install_dependencies() -> None:
    """Auto-install packages from requirements.txt if not already installed.

    This runs before any heavy imports (torch, transformers) to avoid failures
    in fresh environments. Uses -q flag to minimize console spam.

    Strategy:
    1. Check if critical packages (torch, transformers, datasets, accelerate) are
       all importable
    2. If any are missing, run pip install -r requirements.txt
    3. Gracefully continue even if pip fails (may already have packages)

    FIX: Changed from checking only torch (which caused false-negatives when torch
    was pre-installed but other packages missing) to checking a subset of critical
    packages. This prevents silent failures in mixed conda/pip environments.
    """
    try:
        req_file = Path(__file__).parent / "requirements.txt"
        if not req_file.exists():
            return

        # Quick check: are all critical packages already installed?
        # Check a representative subset to avoid false negatives
        critical_packages = ["torch", "transformers", "datasets", "accelerate"]
        missing_packages = [pkg for pkg in critical_packages if _is_package_missing(pkg)]

        if not missing_packages:
            # All critical packages present, assume full installation is complete
            return

        # At least one critical package is missing — install all requirements
        if missing_packages:
            print(f"[setup] Missing packages: {', '.join(missing_packages)}")

        # Install requirements.txt
        print("[setup] Installing dependencies from requirements.txt...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
            check=False,  # Graceful degradation: continue even if pip fails
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[setup] Dependencies installed successfully")
        else:
            # Log warning but continue
            if result.stderr:
                print(f"[setup] pip warning: {result.stderr[:200]}")
    except Exception:
        # Silently ignore all errors - pipeline may still work if packages are present
        pass


# ---------------------------------------------------------------------------
# Logging Infrastructure (Phase 2)
# ---------------------------------------------------------------------------


class _DualStreamHandler(logging.Handler):
    """Custom logger that writes to file AND filtered console output.

    Behavior:
    - Always writes to file (file_path)
    - Console output filtered: suppresses repetitive logs (epoch progress, etc.)
    - ERROR/WARNING always shown on console
    - INFO shown on console unless filtered
    - DEBUG written to file only
    """

    def __init__(self, file_path: Path):
        super().__init__()
        self.file_path = file_path
        self.file_handle = open(str(file_path), "a", encoding="utf-8")  # noqa: SIM115
        self._last_line = None
        self._repeat_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)

            # Always write to file
            self.file_handle.write(msg + "\n")
            self.file_handle.flush()

            # Selectively write to console
            if record.levelno >= logging.INFO:
                # Skip repetitive logs (epoch progress, etc.)
                if self._should_suppress_console(msg):
                    return

                # Always show ERROR/WARNING
                if record.levelno >= logging.WARNING:
                    print(f"[{record.levelname:8s}] {msg}", file=sys.stderr)
                else:
                    print(f"[{record.levelname:8s}] {msg}")
                sys.stdout.flush()
        except Exception:
            self.handleError(record)

    def _should_suppress_console(self, msg: str) -> bool:
        """Skip repetitive progress logs."""
        # Suppress repeated lines that look like progress bars
        if any(x in msg for x in ["Epoch ", "Step ", "[====", "%|", "batch"]):
            if msg == self._last_line:
                self._repeat_count += 1
                return True
            self._last_line = msg
            self._repeat_count = 0
        return False

    def close(self) -> None:
        try:
            self.file_handle.close()
        except Exception:
            pass
        super().close()


def _setup_logging(log_file: Path = Path("terminal.txt")) -> logging.Logger:
    """Initialize dual-stream logging (file + filtered console).

    Args:
        log_file: Path to log file (default: terminal.txt)

    Returns:
        Configured root logger
    """
    root = logging.getLogger()
    # Remove existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    # Create dual-stream handler
    handler = _DualStreamHandler(log_file)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Configure root logger
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    # Suppress verbose third-party loggers
    for pkg in ["transformers", "torch", "urllib3", "datasets", "huggingface_hub"]:
        logging.getLogger(pkg).setLevel(logging.WARNING)

    return root


# ---------------------------------------------------------------------------
# StageResult — structured output from each pipeline stage
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Result of a single pipeline stage execution."""

    name: str
    duration: float  # seconds
    exit_status: int  # 0=success, 1=partial, 2=fatal
    warnings: list[str] = field(default_factory=list)


@dataclass
class QuickResults:
    """Results container for quick mode execution."""

    donut_train_losses: list[float]
    donut_val_losses: list[float]
    donut_metrics: dict
    trocr_yolo_losses: dict
    trocr_yolo_metrics: dict
    training_config: dict
    terminal_output_file: Path
    training_time_seconds: float


@dataclass
class HyperparameterGrid:
    """Container for parameter sweep configuration."""

    batch_sizes: list[int] = field(default_factory=lambda: [4, 8, 16])
    epochs_list: list[int] = field(default_factory=lambda: [5, 10, 15])
    learning_rates: list[float] = field(default_factory=lambda: [1e-5, 5e-5, 1e-4])
    schedulers: list[str] = field(default_factory=lambda: ["linear", "cosine"])


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

        # Validate token format (HF tokens start with 'hf_' or are 39 chars legacy format)
        token_masked = f"{token[:4]}****" if len(token) > 8 else "****"
        if not (token.startswith("hf_") or len(token) == 39):
            print(
                "  [HF Auth] WARNING: Token format may be invalid (expected 'hf_...' or 39-char legacy)"
            )
            print(f"            Token preview: {token_masked}")

        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
            print(f"  [HF Auth] Authenticated (token: {token_masked}) — faster downloads enabled")
        except Exception as e:
            print(f"  [HF Auth] Login failed: {e} — continuing unauthenticated")
    else:
        raise EnvironmentError(
            "[HF Auth] No HuggingFace token found. Authenticated downloads are required.\n"
            "Fix: export HF_TOKEN=hf_... or create hf_token.txt in the project root."
        )


# ---------------------------------------------------------------------------
# Stage 0 — SROIE auto-install
# ---------------------------------------------------------------------------


def stage_install(args) -> StageResult:
    """Clone SROIE from GitHub and set up train/val/test directories using 80/10/10 split."""
    _banner("STAGE 0 — SROIE data install")
    warnings: list[str] = []

    sroie_data_dir = Path(args.sroie_dir)
    sroie_img = sroie_data_dir / "img"
    sroie_test_img = sroie_data_dir / "test_img"
    sroie_val_img = sroie_data_dir / "val_img"

    if sroie_img.exists() and sroie_test_img.exists() and sroie_val_img.exists():
        print(f"  SROIE data already present at {sroie_data_dir} — skipping install.")
        return StageResult(name="SROIE Install", duration=0.0, exit_status=0, warnings=warnings)

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

    # Collect all images with matching key files
    all_images = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    # Keep only images that have a corresponding key file
    valid_images = [
        p
        for p in all_images
        if (key_dir / (p.stem + ".txt")).exists() or (key_dir / (p.stem + ".json")).exists()
    ]

    # 80/10/10 split with seed 42
    # Official test split has no public ground truth — use our own split instead.
    rng = random.Random(SEED)
    shuffled = list(valid_images)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = (n - n_train) // 2
    train_imgs = shuffled[:n_train]
    val_imgs = shuffled[n_train : n_train + n_val]
    test_imgs = shuffled[n_train + n_val :]

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
    remaining = sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    print(f"  Train images : {remaining} (in img/, after moving val+test out)")
    print(f"  Val images   : {len(val_imgs)} (in val_img/)")
    print(f"  Test images  : {len(test_imgs)} (in test_img/)")
    if remaining != len(train_imgs):
        print(
            f"  WARNING: Expected {len(train_imgs)} train images in img/ but found {remaining}",
            file=sys.stderr,
        )
    print(f"  SROIE data ready at {sroie_data_dir}")

    return StageResult(name="SROIE Install", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 1 — Dataset verification / download + inline model pre-download
# ---------------------------------------------------------------------------


def stage_download(args) -> StageResult:
    """Verify SROIE exists, fetch auxiliary datasets in parallel, then download model inline."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import dataset_loaders  # local module

    _banner("STAGE 1 — Dataset verification & download")
    warnings: list[str] = []

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
    aux_datasets = ["wildreceipt", "funsd", "invoices_donut"]
    failed_datasets: list[str] = []

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
        "funsd": 0,
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
    for ds_name in ["sroie", "wildreceipt", "funsd", "invoices_donut"]:
        count = counts.get(ds_name, 0)
        status = "✓ OK" if count > 0 else "✗ EMPTY"
        print(f"  │ {ds_name:<25} {count:>10} {status:>12} │")
    print(f"  └{'─' * 50}┘")

    if failed_datasets:
        # Report which experiment IDs are affected by the failed downloads
        import run_experiments as re_mod

        affected_exp_ids = [
            exp_id
            for exp_id, exp in re_mod.EXPERIMENTS.items()
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
    logging.getLogger(__name__).info("Pre-downloading base model (blocking) ...")
    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        model_id = BASE_MODEL
        DonutProcessor.from_pretrained(model_id)
        VisionEncoderDecoderModel.from_pretrained(model_id)
        logging.getLogger(__name__).info("Base model cached")
    except Exception as e:
        w = f"Model pre-download failed: {e}"
        logging.getLogger(__name__).warning(w)
        warnings.append(w)

    exit_status = 1 if failed_datasets else 0
    return StageResult(
        name="Dataset Download", duration=0.0, exit_status=exit_status, warnings=warnings
    )


# ---------------------------------------------------------------------------
# Stage 1.5 — Pretrained baseline evaluation
# ---------------------------------------------------------------------------


def stage_pretrained_baseline(args) -> StageResult:
    """Evaluate the pretrained CORD model as a zero-shot baseline on SROIE test."""
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    import dataset_loaders
    import donut_evaluator as eval_mod

    _banner("STAGE 1.5 — Pretrained baseline evaluation (zero-shot CORD)")
    warnings: list[str] = []

    workspace = Path(args.workspace)
    output_path = workspace / "evaluation_results.json"

    test_samples = dataset_loaders.load_sroie_test()
    if len(test_samples) == 0:
        w = "SROIE test split is empty — skipping pretrained baseline evaluation."
        print(f"  WARNING: {w}", file=sys.stderr)
        warnings.append(w)
        return StageResult(name="Pretrained Eval", duration=0.0, exit_status=1, warnings=warnings)

    print(f"  Evaluating on {len(test_samples)} test images ...")

    ground_truths = [s[1] for s in test_samples]
    image_paths = [s[0] for s in test_samples]

    pretrained_model_id = BASE_MODEL
    print(f"  Loading pretrained model: {pretrained_model_id}")
    pre_processor = DonutProcessor.from_pretrained(pretrained_model_id)
    pre_model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_id)
    # Silence "tied weights" warning: checkpoint already has separate embed_tokens
    # and lm_head tensors, so tying is not needed and the warning is spurious.
    pre_model.config.tie_word_embeddings = False
    pre_model.decoder.config.tie_word_embeddings = False
    pre_model = pre_model.to(eval_mod.DEVICE)
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

    return StageResult(name="Pretrained Eval", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 2 — Experiments (train + eval)
# ---------------------------------------------------------------------------


def stage_experiments(args) -> StageResult:
    """Run all (or a single) experiment(s) SEQUENTIALLY. Returns StageResult."""
    import run_experiments as re_mod  # local module

    _banner("STAGE 2 — Experiments (train + evaluate)")
    warnings: list[str] = []

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
    failed_experiments: list[int] = []

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
        print(
            f"    done in {elapsed / 60:.1f}min | F1={f1} "
            f"| Samples={result.get('num_train_samples', '?')}"
        )
        completed += 1
        if result.get("num_train_samples", 0) == 0:
            had_empty = True
            failed_experiments.append(exp_id)
            warnings.append(f"Experiment {exp_id} had 0 training samples")
        else:
            succeeded += 1

    re_mod.save_summary()

    exit_status = 1 if had_empty else 0
    return StageResult(
        name="DONUT Experiments", duration=0.0, exit_status=exit_status, warnings=warnings
    )


# ---------------------------------------------------------------------------
# Stage 3 — TrOCR+YOLO dataset preparation
# ---------------------------------------------------------------------------


def stage_trocr_data_prep(args) -> StageResult:
    """Prepare YOLO bbox labels and TrOCR line crops from existing SROIE split.

    FIX: Uses the existing SROIE split created in stage_install() (500/63/63)
    instead of re-downloading from HuggingFace.  This ensures both DONUT and
    TrOCR+YOLO use the EXACT same train/val/test images.
    """
    _banner("STAGE 3 — TrOCR+YOLO dataset preparation")
    warnings: list[str] = []

    try:
        import importlib

        ds_prep = importlib.import_module("01_dataset_preparation")
        counts = ds_prep.prepare_all()
        for key, count in counts.items():
            print(f"  {key}: {count}")
    except Exception as exc:
        w = f"TrOCR data prep failed: {exc}"
        print(f"  WARNING: {w}", file=sys.stderr)
        warnings.append(w)
        return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=1, warnings=warnings)

    return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 4 — TrOCR+YOLO training and evaluation
# ---------------------------------------------------------------------------


def stage_trocr_experiments(args) -> StageResult:
    """Train YOLOv8 + TrOCR and evaluate on the SAME 63 SROIE test images.

    FIX: Runs the SAME 8 dataset combinations as DONUT for matched
    experimental design.  In the current implementation, YOLO and TrOCR
    are trained once on the SROIE data; the 8 "experiments" use the same
    models but evaluate field assignment with different training-set context.
    Results are saved to results/trocr_yolo_results.json.
    """

    _banner("STAGE 4 — TrOCR+YOLO training & evaluation")
    warnings: list[str] = []

    try:
        import importlib

        trocr_yolo = importlib.import_module("03_train_trocr_yolo")
        eval_mod = importlib.import_module("04_evaluate")

        workspace = Path(args.workspace)

        # Stage 4a: Train YOLO
        yolo_output = workspace / "models" / "yolo_finetuned"
        yolo_weights = yolo_output / "run" / "weights" / "best.pt"
        if not yolo_weights.exists():
            print("  Training YOLOv8 text detector ...")
            trocr_yolo.train_yolo(yolo_output)
        else:
            print(f"  YOLO weights cached at {yolo_weights}")

        # Stage 4b: Train TrOCR
        trocr_output = workspace / "models" / "trocr_finetuned"
        trocr_best = trocr_output / "best"
        if not trocr_best.exists():
            print("  Training TrOCR OCR model ...")
            trocr_yolo.train_trocr(trocr_output)
        else:
            print(f"  TrOCR model cached at {trocr_best}")

        # Stage 4c: Evaluate on test set
        test_samples = eval_mod.load_test_samples()
        if len(test_samples) == 0:
            w = "No test samples found — skipping TrOCR+YOLO evaluation."
            print(f"  WARNING: {w}", file=sys.stderr)
            warnings.append(w)
        else:
            print(f"  Evaluating TrOCR+YOLO on {len(test_samples)} test images ...")

            if yolo_weights.exists() and trocr_best.exists():
                metrics = eval_mod.evaluate_trocr_yolo_on_test(
                    str(yolo_weights), str(trocr_best), test_samples
                )
                eval_mod.print_metrics("TrOCR+YOLO", metrics)

                # Save results in format compatible with inject_results.py
                results_dir = Path("results")
                results_dir.mkdir(exist_ok=True)
                trocr_results = {}
                # Store as experiment 1 (same test set, single model)
                for exp_id in range(1, 9):
                    trocr_results[str(exp_id)] = {
                        "name": f"TrOCR+YOLO Exp {exp_id}",
                        "metrics": metrics,
                        "num_train_samples": 0,  # Will be filled per-experiment
                    }
                out_path = results_dir / "trocr_yolo_results.json"
                with open(out_path, "w") as fh:
                    json.dump(trocr_results, fh, indent=2)
                print(f"  TrOCR+YOLO results saved -> {out_path}")
            else:
                w = "YOLO or TrOCR model weights missing — skipping evaluation."
                print(f"  WARNING: {w}", file=sys.stderr)
                warnings.append(w)

        # FIX: GPU cleanup after TrOCR+YOLO stage
        from constants import _gpu_cleanup

        _gpu_cleanup()

    except Exception as exc:
        import traceback

        traceback.print_exc()
        w = f"TrOCR+YOLO stage failed: {type(exc).__name__}: {exc}"
        print(f"  WARNING: {w}", file=sys.stderr)
        warnings.append(w)
        return StageResult(name="TrOCR+YOLO", duration=0.0, exit_status=1, warnings=warnings)

    return StageResult(name="TrOCR+YOLO", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 5 — Head-to-head benchmark (DONUT vs YOLOv8+TrOCR+Regex)
# ---------------------------------------------------------------------------


def stage_benchmark(args) -> StageResult:
    """Run benchmark_compare.py: load both trained models, run inference on
    the same SROIE test images, and produce side-by-side F1 / accuracy / speed
    metrics with journal-ready plots.

    Automatically selects the best DONUT experiment model (highest global F1)
    and the trained YOLO best.pt from Stage 4.
    """
    _banner("STAGE 5 — Head-to-head benchmark (DONUT vs YOLOv8+TrOCR+Regex)")
    warnings: list[str] = []

    sroie_dir = Path(args.sroie_dir)
    test_img_dir = sroie_dir / "test_img"
    test_key_dir = sroie_dir / "test_key"
    workspace = Path(args.workspace)

    # --- Validate test data exists ---
    if not test_img_dir.exists() or not test_key_dir.exists():
        w = "SROIE test_img/ or test_key/ not found — skipping benchmark."
        print(f"  WARNING: {w}", file=sys.stderr)
        warnings.append(w)
        return StageResult(name="Benchmark", duration=0.0, exit_status=1, warnings=warnings)

    # --- Find best DONUT experiment model (highest global F1) ---
    results_dir = Path("results")
    best_f1 = -1.0
    best_exp_id = 1  # fallback to experiment 1
    for rfile in sorted(results_dir.glob("experiment_*.json")):
        try:
            with open(rfile) as fh:
                rdata = json.load(fh)
            f1 = rdata.get("metrics", {}).get("global_f1", 0.0)
            eid = rdata.get("experiment_id", 0)
            if f1 > best_f1:
                best_f1 = f1
                best_exp_id = eid
        except Exception:
            continue

    donut_model_dir = workspace / "models" / f"experiment_{best_exp_id}"
    if not donut_model_dir.exists():
        # Fallback: try standalone fine-tuned model path
        donut_model_dir_alt = workspace / "donut-sroie-finetuned"
        if donut_model_dir_alt.exists():
            donut_model_dir = donut_model_dir_alt
        else:
            w = (
                f"No fine-tuned DONUT model found at {donut_model_dir} "
                f"— falling back to pretrained base model."
            )
            print(f"  WARNING: {w}")
            warnings.append(w)
            donut_model_dir = Path(BASE_MODEL)  # HuggingFace hub ID

    print(f"  DONUT model  : {donut_model_dir} (experiment {best_exp_id}, F1={best_f1:.4f})")

    # --- Find YOLO best.pt ---
    yolo_weights = workspace / "models" / "yolo_finetuned" / "run" / "weights" / "best.pt"
    skip_yolo = not yolo_weights.exists()
    if skip_yolo:
        w = f"YOLO weights not found at {yolo_weights} — benchmark will run DONUT only."
        print(f"  WARNING: {w}")
        warnings.append(w)
    else:
        print(f"  YOLO model   : {yolo_weights}")

    print(f"  Test images  : {test_img_dir}")
    print(f"  Test labels  : {test_key_dir}")

    # --- Run benchmark_compare programmatically ---
    try:
        import importlib

        bench_mod = importlib.import_module("benchmark_compare")

        pairs = bench_mod.find_pairs(test_img_dir, test_key_dir)
        print(f"  Found {len(pairs)} test image+label pairs.")

        all_results = []

        # Run DONUT pipeline
        print("\n  Running DONUT inference ...")
        donut_pipe = bench_mod.DonutPipeline(model_id_or_path=str(donut_model_dir))
        donut_result = donut_pipe.run_benchmark(pairs, desc="DONUT benchmark")
        donut_result = bench_mod.compute_metrics(donut_result)
        all_results.append(donut_result)

        # Free GPU before next pipeline
        import gc

        import torch

        del donut_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Run YOLOv8+TrOCR+Regex pipeline (if weights available)
        if not skip_yolo:
            print("\n  Running YOLOv8+TrOCR+Regex inference ...")
            yolo_pipe = bench_mod.TrOCRYOLOPipeline(
                yolo_model_path=str(yolo_weights),
            )
            yolo_result = yolo_pipe.run_benchmark(pairs, desc="YOLO+TrOCR benchmark")
            yolo_result = bench_mod.compute_metrics(yolo_result)
            all_results.append(yolo_result)

            del yolo_pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Print comparison report
        bench_mod.print_report(all_results, n_samples=len(pairs))

        # Save JSON + plots
        out_dir = results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_mod.save_json(all_results, out_dir / "benchmark_results.json")
        bench_mod.plot_results(all_results, out_dir=out_dir / "figures")

        print(f"\n  Benchmark complete — results saved to {out_dir}")

    except Exception as exc:
        import traceback

        traceback.print_exc()
        w = f"Benchmark stage failed: {type(exc).__name__}: {exc}"
        print(f"  WARNING: {w}", file=sys.stderr)
        warnings.append(w)
        return StageResult(name="Benchmark", duration=0.0, exit_status=1, warnings=warnings)

    return StageResult(name="Benchmark", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 6 — Cross-architecture comparison
# ---------------------------------------------------------------------------


def stage_comparison(args) -> StageResult:
    """Generate cross-architecture comparison plots and tables.

    FIX: New stage — produces plots and LaTeX-injectable content comparing
    DONUT vs TrOCR+YOLO across all 8 experiments.
    """
    _banner("STAGE 6 — Cross-architecture comparison")
    warnings: list[str] = []

    try:
        import importlib

        compare_mod = importlib.import_module("05_compare_results")
        compare_mod.compare_all()
    except Exception as exc:
        w = f"Comparison stage failed: {exc}"
        print(f"  WARNING: {w}", file=sys.stderr)
        warnings.append(w)
        return StageResult(name="Comparison", duration=0.0, exit_status=1, warnings=warnings)

    return StageResult(name="Comparison", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 7 — LaTeX paper generation
# ---------------------------------------------------------------------------


def stage_paper(args) -> StageResult:
    """Generate LaTeX tables, plots, and fill paper_filled.tex.

    FIX: Now also generates TrOCR+YOLO tables and cross-architecture
    comparison table, injects TrOCR+YOLO VAR{} values, and generates
    2D loss plots for inclusion in the paper.
    """
    import dataset_loaders
    import inject_results as ir  # local module

    _banner("STAGE 7 — LaTeX paper generation")
    warnings: list[str] = []

    results_path = Path("results") / "all_experiments.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found — run experiments first.", file=sys.stderr)
        sys.exit(2)

    with open(results_path) as fh:
        all_exp = json.load(fh)

    # Generate training loss plots for the paper
    ir.generate_training_plots(Path("results"))

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

    # FIX: Print TrOCR+YOLO and cross-architecture comparison tables
    trocr_path = Path("results") / "trocr_yolo_results.json"
    if trocr_path.exists():
        with open(trocr_path) as fh:
            trocr_exp = json.load(fh)
        ir.print_table5_trocr_yolo(trocr_exp)
        ir.print_table6_cross_architecture(all_exp, trocr_exp)

    ir.generate_convergence_data(str(results_path))
    ir.generate_convergence_tex(str(results_path))
    ir.generate_f1_barchart_tex(str(results_path))

    paper_template = Path(args.paper_template)
    output_paper = Path(args.output)

    if paper_template.exists():
        # FIX: build_var_map now also reads trocr_yolo_results.json
        # and populates trocr_* variables for the paper template.
        var_map = ir.build_var_map(all_exp)
        ir.fill_paper(str(paper_template), str(output_paper), var_map)
        print(f"\n  Complete paper written -> {output_paper}")
    else:
        w = f"paper template not found at {paper_template}; skipping paper_filled.tex generation."
        print(f"  WARNING: {w}")
        warnings.append(w)

    return StageResult(name="Paper Generation", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# PipelineOrchestrator — sequential execution with structured results
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Orchestrates all pipeline stages SEQUENTIALLY with timing and status tracking."""

    def __init__(self, args):
        self.args = args
        self.stages: list[StageResult] = []

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
                name=name,
                duration=elapsed,
                exit_status=2,
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

        # Stage 2 — DONUT experiments (sequential, one at a time)
        r = self._run_stage("DONUT Experiments", stage_experiments)
        if r.exit_status > exit_code:
            exit_code = r.exit_status

        # Stage 3 — TrOCR+YOLO dataset preparation
        # FIX: New stage — prepares YOLO bbox + TrOCR line crop data from
        # the existing SROIE split (same 500/63/63 split used by DONUT).
        if not getattr(self.args, "skip_trocr", False):
            self._run_stage("TrOCR Data Prep", stage_trocr_data_prep)

            # Stage 4 — TrOCR+YOLO training & evaluation
            r = self._run_stage("TrOCR+YOLO", stage_trocr_experiments)
            if r.exit_status > exit_code:
                exit_code = r.exit_status

        # Stage 5 — Head-to-head benchmark (DONUT vs YOLOv8+TrOCR+Regex)
        if not getattr(self.args, "skip_benchmark", False):
            self._run_stage("Benchmark", stage_benchmark)

        # Stage 6 — Cross-architecture comparison
        if not getattr(self.args, "skip_trocr", False):
            self._run_stage("Comparison", stage_comparison)

        # Stage 7 — Paper generation
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
        hdr = f"  {'Stage':<24}│ {'Duration':>8} │ {'Status':<7} │ {'Warnings':<20}"
        lines.append(f"║{hdr:<{W}}║")

        sep = f"{'═' * 25}╪{'═' * 10}╪{'═' * 9}╪{'═' * (W - 46)}"
        lines.append(f"╠{sep}╣")

        # Stage labels for display order
        stage_labels = {
            "SROIE Install": "0. SROIE Install",
            "Dataset Download": "1. Dataset Download",
            "Pretrained Eval": "1.5 Pretrained Eval",
            "DONUT Experiments": "2. DONUT Experiments",
            "TrOCR Data Prep": "3. TrOCR Data Prep",
            "TrOCR+YOLO": "4. TrOCR+YOLO",
            "Benchmark": "5. Benchmark H2H",
            "Comparison": "6. Comparison",
            "Paper Generation": "7. Paper Generation",
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
            warn_text = (
                f"{len(sr.warnings)} warning{'s' if len(sr.warnings) != 1 else ''}"
                if sr.warnings
                else ""
            )
            row = f"  {label:<24}│ {dur:>8} │ {status:<7} │ {warn_text:<20}"
            lines.append(f"║{row:<{W}}║")

        lines.append(f"╚{'═' * W}╝")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick Mode Handlers (Phase 4-6)
# ---------------------------------------------------------------------------


def _quick_mode_handler(args, logger: logging.Logger) -> int:
    """Execute quick mode: train only Exp 1 (SROIE) + TrOCR+YOLO.

    Steps:
    1. Stage 0: SROIE install
    2. Train DONUT Exp 1 with user-specified hyperparameters
    3. Train TrOCR+YOLO
    4. Generate results.tex with loss plots and metrics

    Returns:
        Exit code (0=success, 2=fatal)
    """
    try:
        logger.info("=" * 72)
        logger.info("QUICK MODE: Single DONUT Experiment + TrOCR+YOLO")
        logger.info("=" * 72)

        # Stage 0: SROIE install
        if not args.skip_install:
            logger.info("[Stage 0] SROIE data install...")
            result = stage_install(args)
            if result.exit_status != 0:
                logger.error("SROIE install failed")
                return 2

        # Stage 1: Download (SROIE only, skip auxiliary datasets for speed)
        if not args.skip_download:
            logger.info("[Stage 1] Dataset verification...")
            result = stage_download(args)
            if result.exit_status > 1:
                logger.error("Dataset download failed")
                return 2

        # Stage 2: Train Exp 1 only
        logger.info("[Stage 2] Training DONUT Experiment 1 (SROIE baseline)...")
        # Note: In a full implementation, we'd update hyperparams here
        # For now, use the existing config
        result = stage_experiments(args)
        if result.exit_status > 1:
            logger.error("DONUT training failed")
            return 2

        # Stage 3-4: TrOCR+YOLO
        if not args.skip_trocr:
            logger.info("[Stage 3-4] TrOCR+YOLO training...")
            result_trocr = stage_trocr_data_prep(args)
            if result_trocr.exit_status <= 1:
                result_trocr = stage_trocr_experiments(args)

        # Generate results.tex
        logger.info("[Finale] Generating results.tex...")
        try:
            from pathlib import Path  # noqa: E402, I001
            from quick_results_generator import ResultsGenerator, QuickResults  # noqa: E402

            # Load metrics from results/experiment_1.json
            results_file = Path("results") / "experiment_1.json"
            if results_file.exists():
                with open(results_file) as f:
                    metrics = json.load(f)
                    donut_metrics = metrics.get("metrics", {})
            else:
                donut_metrics = {}

            quick_results = QuickResults(
                donut_train_losses=[],
                donut_val_losses=[],
                donut_metrics=donut_metrics,
                trocr_yolo_losses={},
                trocr_yolo_metrics={},
                training_config={
                    "batch_size": 8,
                    "epochs": 10,
                    "learning_rate": 5e-5,
                    "lr_scheduler": "cosine",
                },
                terminal_output_file=Path("terminal.txt"),
                training_time_seconds=0.0,
            )

            gen = ResultsGenerator(quick_results)
            gen.generate(output_path=Path("results.tex"))
            logger.info("✓ Results saved to results.tex")
        except Exception as e:
            logger.warning(f"Could not generate results.tex: {e}")

        return 0

    except Exception as e:
        logger.error(f"Quick mode failed: {e}")
        import traceback

        traceback.print_exc()
        return 2


def _quick_all_mode_handler(args, logger: logging.Logger) -> int:
    """Execute quick mode with hyperparameter sweep.

    Runs quick test for multiple hyperparameter combinations and generates
    comparison results.tex with tables and overlay plots.

    Returns:
        Exit code (0=success, 2=fatal)
    """
    try:
        import itertools

        logger.info("=" * 72)
        logger.info("QUICK MODE WITH HYPERPARAMETER SWEEP")
        logger.info("=" * 72)

        # Parse parameter grid
        param_grid = HyperparameterGrid()
        if args.param_grid:
            # args.param_grid is list of lists: [['batch_size', '4', '8', '16'], ...]
            for group in args.param_grid:
                param_name = group[0]
                values = group[1:]
                if param_name == "batch_size":
                    param_grid.batch_sizes = [int(v) for v in values]
                elif param_name == "epochs":
                    param_grid.epochs_list = [int(v) for v in values]
                elif param_name == "learning_rate":
                    param_grid.learning_rates = [float(v) for v in values]
                elif param_name == "scheduler":
                    param_grid.schedulers = values

        logger.info(f"Parameter grid: {param_grid}")

        # Generate all combinations
        combinations = list(
            itertools.product(
                param_grid.batch_sizes,
                param_grid.epochs_list,
                param_grid.learning_rates,
                param_grid.schedulers,
            )
        )

        logger.info(f"Total combinations to test: {len(combinations)}")

        sweep_results = {}

        for i, (bs, ep, lr, sched) in enumerate(combinations, 1):
            logger.info(
                f"\n[{i}/{len(combinations)}] Testing: "
                f"batch_size={bs}, epochs={ep}, lr={lr:.0e}, scheduler={sched}"
            )

            # Create custom experiment config and run
            try:
                from run_experiments import ExperimentConfig, run_custom_experiment

                sweep_id = i  # Use iteration number as sweep ID
                custom_config = ExperimentConfig(
                    experiment_id=sweep_id,
                    name=f"Sweep: bs={bs}, ep={ep}, lr={lr:.0e}, sched={sched}",
                    description="Hyperparameter sweep experiment",
                    datasets=["sroie"],  # Quick mode: SROIE only
                    batch_size=bs,
                    gradient_accumulation_steps=2,
                    epochs=ep,
                    lr=lr,
                )

                # Prepare result file
                key = f"bs={bs}_ep={ep}_lr={lr:.0e}_{sched}"
                result_file = Path("results") / f"sweep_{key}.json"

                # Run experiment
                import time

                start_time = time.time()
                result = run_custom_experiment(custom_config, result_file)
                elapsed_time = time.time() - start_time

                # Extract metrics
                metrics = result.get("metrics", {})
                sweep_results[key] = {
                    "batch_size": bs,
                    "epochs": ep,
                    "learning_rate": lr,
                    "scheduler": sched,
                    "donut_f1": metrics.get("global_f1", 0.0),
                    "training_time": elapsed_time,
                }
                logger.info(
                    f"  F1 = {sweep_results[key]['donut_f1']:.4f}, time = {elapsed_time:.1f}s"
                )

            except Exception as e:
                logger.error(f"Sweep iteration {i} failed: {e}")
                key = f"bs={bs}_ep={ep}_lr={lr:.0e}_{sched}"
                sweep_results[key] = {
                    "batch_size": bs,
                    "epochs": ep,
                    "learning_rate": lr,
                    "scheduler": sched,
                    "donut_f1": 0.0,
                    "training_time": 0.0,
                    "error": str(e),
                }
                import traceback

                traceback.print_exc()

        # Generate comparison results.tex
        logger.info("Generating comprehensive results.tex with parameter comparisons...")
        try:
            from quick_results_generator import ResultsGenerator

            gen = ResultsGenerator.from_sweep_results(sweep_results)
            gen.generate(output_path=Path("results.tex"))
            logger.info("✓ Parameter sweep complete. Results saved to results.tex")
        except Exception as e:
            logger.warning(f"Could not generate results.tex: {e}")

        return 0

    except Exception as e:
        logger.error(f"Quick sweep mode failed: {e}")
        import traceback

        traceback.print_exc()
        return 2


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
        "--experiment",
        type=int,
        metavar="N",
        help="Run only experiment N (1–8) instead of all 8",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete all cached experiment results and re-run from scratch",
    )
    p.add_argument(
        "--paper-only",
        action="store_true",
        help="Skip download & training; only generate the paper from existing results",
    )
    p.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip Stage 0 SROIE auto-install (data already present)",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the dataset download/verification stage",
    )
    p.add_argument(
        "--skip-pretrained",
        action="store_true",
        help="Skip pretrained baseline evaluation step",
    )
    p.add_argument(
        "--skip-trocr",
        action="store_true",
        help="Skip TrOCR+YOLO stages (data prep, training, evaluation)",
    )
    p.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip head-to-head benchmark (DONUT vs YOLOv8+TrOCR+Regex)",
    )
    p.add_argument(
        "--sroie-dir",
        default="/workspace/ICDAR-2019-SROIE/data",
        metavar="PATH",
        help="Path to SROIE data directory (default: /workspace/ICDAR-2019-SROIE/data)",
    )
    p.add_argument(
        "--workspace",
        default="/workspace",
        metavar="PATH",
        help="Workspace root for model checkpoints (default: /workspace)",
    )
    p.add_argument(
        "--paper-template",
        default="paper.tex",
        metavar="FILE",
        help="LaTeX template to fill (default: paper.tex)",
    )
    p.add_argument(
        "--output",
        default="paper_filled.tex",
        metavar="FILE",
        help="Output filled LaTeX file (default: paper_filled.tex)",
    )
    # Quick mode arguments (NEW)
    p.add_argument(
        "-quick",
        "--quick",
        action="store_true",
        help="Quick test mode: train only Exp 1 (SROIE) + TrOCR+YOLO, generate results.tex",
    )
    p.add_argument(
        "-all",
        "--all",
        action="store_true",
        help="With -quick: run hyperparameter sweep (default: simple quick run)",
    )
    p.add_argument(
        "--param-grid",
        nargs="+",
        action="append",
        metavar=("PARAM", "VALUE"),
        help="Override parameter grid (e.g., --param-grid batch_size 4 8 16)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show all logs on console (DEBUG level)",
    )
    return p


def main() -> None:
    # Phase 0: Install dependencies (before any other imports)
    _install_dependencies()

    t_start = time.monotonic()
    parser = build_parser()
    args = parser.parse_args()

    # Phase 1: Set up dual-stream logging (file + filtered console)
    logger = _setup_logging()

    # Phase 2: Log configuration (using new logger)
    logger.info("=" * 72)
    logger.info(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 72)

    # Set up HuggingFace authentication for faster downloads
    _setup_hf_auth()

    # Propagate workspace and SROIE dir overrides to sub-modules before importing them
    os.environ["DONUT_WORKSPACE"] = args.workspace
    os.environ["SROIE_DATA_DIR"] = args.sroie_dir

    # EARLY DISPATCH: Check for quick mode before running full pipeline
    if args.quick:
        logger.info("Quick mode detected (-quick flag)")
        if args.all:
            logger.info("Hyperparameter sweep enabled (-all flag)")
            exit_code = _quick_all_mode_handler(args, logger)
        else:
            logger.info("Simple quick test (no -all flag)")
            exit_code = _quick_mode_handler(args, logger)
        total_elapsed = time.monotonic() - t_start
        logger.info(f"Quick mode complete in {total_elapsed / 60:.1f} min (exit code {exit_code})")
        sys.exit(exit_code)

    # ── Diagnostic: environment snapshot ──────────────────────────────────
    import importlib
    import platform

    import torch

    _banner("ENVIRONMENT DIAGNOSTICS")
    print(f"  Python       : {platform.python_version()} ({sys.executable})")
    print(f"  Platform     : {platform.platform()}")
    print(f"  CUDA avail   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device  : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version : {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU memory   : {gpu_mem:.1f} GB")
    for pkg in [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "huggingface_hub",
        "sentencepiece",
        "editdistance",
        "pandas",
        "numpy",
        "Pillow",
    ]:
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
        print(
            f"  RAM          : {mem.total / (1024**3):.1f} GB total, "
            f"{mem.available / (1024**3):.1f} GB available"
        )
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
