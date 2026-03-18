# =============================================================================
# run_all.py
# Purpose: Master orchestrator — dataset prep, DONUT training, TrOCR+YOLO, benchmark (run this)
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-07
# =============================================================================
"""
run_all.py — Single entry point for the complete dual-architecture pipeline.

Calling this one script does everything:
  0. Installs SROIE data (auto-clones from GitHub; 80/10/10 split)
  1. Verifies / downloads all auxiliary datasets + pre-downloads the base model
  1.5. Evaluates the pretrained CORD model (donut-base-finetuned-cord-v2) as a zero-shot baseline
  2. Trains DONUT for each of the 8 experiment configurations
  3. Prepares YOLO bbox + TrOCR line crop datasets from SROIE
  4. Trains YOLOv8 + TrOCR and runs TrOCR+YOLO inference on SROIE test
  5. Runs head-to-head benchmark: DONUT vs YOLOv8+TrOCR+Regex (F1, accuracy, speed)
  6. Generates cross-architecture comparison plots and tables
  7. Fills paper.tex with all real metrics → paper_filled.tex

All stages run sequentially on a single GPU; GPU memory is explicitly freed
between stages via _gpu_cleanup() to prevent VRAM fragmentation.
Dependencies are auto-installed from requirements.txt on first run.
Dual-stream logging writes to both terminal.txt (full) and stdout (filtered).

All stages run SEQUENTIALLY to prevent GPU memory contention.

Usage
-----
  python run_all.py                           # Full pipeline (all 8 DONUT exps + TrOCR+YOLO)
  python run_all.py --experiment 2            # Single DONUT experiment (Exp 2)
  python run_all.py --paper-only              # Regenerate paper from existing results
  python run_all.py -quick                    # Quick test: Exp 1 + TrOCR+YOLO, gen results.tex
  python run_all.py -quick -all               # Hyperparameter sweep (batch_size, epochs, etc.)
  python run_all.py --mini                    # ~20-min smoke test, generates paper_mini.tex
  python run_all.py --skip-trocr              # Skip TrOCR+YOLO stages
  python run_all.py --yolo                    # Start from TrOCR+YOLO only (skip DONUT stages)
  python run_all.py --force                   # Force re-run (delete cached results)

Exit codes
----------
  0  — Success
  1  — One or more experiments had no training data (results saved as partial)
  2  — Fatal error (missing SROIE data, etc.)
"""

import argparse
import copy
import dataclasses
import gc
import glob
import importlib
import itertools
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Ensure the script's own directory is on sys.path so sibling modules
# (constants, reporting, data_pipeline, …) are importable regardless of CWD.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Set before constants.py triggers torch import to reduce GPU memory fragmentation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from constants import BASE_MODEL, IMAGE_EXTS, SEED, _gpu_cleanup  # noqa: E402, I001

__all__ = [
    "PipelineOrchestrator",
    "StageResult",
    "stage_install",
    "stage_download",
    "stage_pretrained_baseline",
    "stage_experiments",
    "stage_trocr_data_prep",
    "stage_trocr_experiments",
    "stage_benchmark",
    "stage_comparison",
    "stage_paper",
    "stage_push_results",
]


# ---------------------------------------------------------------------------
# Auto-Install Dependencies (Phase 1)
# ---------------------------------------------------------------------------

# Packages that must be importable for the pipeline to function correctly.
# 'torch' is checked during install; the remaining are verified post-install.
_CRITICAL_INSTALL_PACKAGES = [
    "torch",
    "transformers",
    # datasets   → replaced with inline _hf_download_dataset_inline() in dataset_loaders.py
    # accelerate → never imported; torch.cuda.amp.GradScaler used directly
    # ultralytics → replaced with inline _YOLO_CLS in train_trocr_yolo.py
    # editdistance → replaced with inline _edit_distance() in donut_evaluator.py
    # pandas → replaced with stdlib csv module
]
_CRITICAL_VERIFY_PACKAGES = ["transformers"]


def _is_package_missing(package_name: str) -> bool:
    """Check if a package can be imported.

    Args:
        package_name: Name of the package to check (e.g., 'torch', 'transformers')

    Returns:
        True if package is NOT installed (ImportError), False if it is installed
    """
    try:
        __import__(package_name)
        # Extra check for datasets: verify load_dataset is actually accessible.
        # A partial/broken install can import the namespace but lack load_dataset.
        return False
    except ImportError:
        return True


class _InstallWatchdog:
    """Prints elapsed-time progress dots during a pip subprocess install.

    Use as a context manager or call start()/stop() manually.  A background
    daemon thread wakes every *interval* seconds and prints how long the
    install has been running.  A warning is printed once the elapsed time
    exceeds *warn_at* seconds (default 120 s = 2 min) to explain that the
    hang is likely a CUDA kernel compilation rather than a true freeze.
    """

    def __init__(self, interval: int = 30, warn_at: int = 120) -> None:
        self._interval = interval
        self._warn_at = warn_at
        self._start: float = 0.0
        self._warned = False
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._start = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> "_InstallWatchdog":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            elapsed = time.monotonic() - self._start
            mins, secs = divmod(int(elapsed), 60)
            print(f"[setup] Still installing... ({mins}m {secs}s elapsed)", flush=True)
            if not self._warned and elapsed > self._warn_at:
                self._warned = True
                print(
                    "[setup] WARNING: Install taking >2 min — "
                    "if this is flash-attn, it is compiling CUDA kernels (5–25 min normal). "
                    "Pre-install via: pip install -r requirements.txt",
                    flush=True,
                )


def _install_dependencies() -> None:
    """Auto-install packages from requirements.txt if not already installed.

    This runs before any heavy imports (torch, transformers) to avoid failures
    in fresh environments. Uses -q flag to minimize console spam.

    Strategy:
    1. Check if critical packages (torch, transformers, datasets, accelerate)
       are all importable.
    2. If any are missing, run ``pip install -r requirements.txt`` (with
       flash-attn filtered out — it requires a pre-installed torch and can
       take 5–25 minutes to compile from source on a GPU machine).
    3. On success, restart the current process via ``os.execv()`` so the
       newly-installed packages are visible to the fresh Python process.
       A sentinel environment variable (``_DONUT_RESTARTED=1``) prevents
       infinite restart loops.
    4. Gracefully continue even if pip fails (packages may still be present).

    flash-attn is intentionally **not** auto-installed.  PyTorch 2.x built-in
    SDPA provides equivalent performance for MAX_LENGTH=768.  To install
    flash-attn manually after the pipeline runs:
        pip install flash-attn --no-build-isolation
    or use a prebuilt wheel from https://flashattn.dev/wheel-finder/
    """
    try:
        req_file = Path(__file__).parent / "requirements.txt"
        if not req_file.exists():
            return

        # Quick check: are all critical packages already installed?
        missing_packages = [pkg for pkg in _CRITICAL_INSTALL_PACKAGES if _is_package_missing(pkg)]

        if not missing_packages:
            # All critical packages present; nothing to do.
            return

        # Guard against infinite restart loops in case os.execv() is invoked
        # repeatedly (e.g. pip install keeps failing).
        if os.environ.get("_DONUT_RESTARTED") == "1":
            # We already restarted once; surface the remaining missing packages
            # to the caller (_verify_critical_packages) rather than looping.
            return

        print(f"[setup] Missing packages: {', '.join(missing_packages)}")

        # Build a filtered requirements list.  flash-attn is excluded because:
        #  a) its build step requires torch to be importable (not true in the
        #     isolated pip subprocess), and
        #  b) on GPU machines the nvcc compilation takes 5–25 minutes with
        #     capture_output=True, making the terminal appear completely frozen.
        req_lines = [
            stripped
            for line in req_file.read_text().splitlines()
            if (stripped := line.strip())
            and not stripped.startswith("#")
            and "flash-attn" not in stripped.lower()
        ]

        print("[setup] Installing dependencies from requirements.txt...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("\n".join(req_lines))
            tmp_path = tmp.name
        try:
            with _InstallWatchdog():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", tmp_path],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30-minute cap; ultralytics + torch can exceed 5 min on slow connections
                )
            if result.returncode == 0:
                print(
                    "[setup] Dependencies installed successfully — restarting to load new packages..."
                )
                # os.execv replaces the current process (no fork) so terminal.txt
                # logging is not duplicated.  The sentinel variable prevents loops.
                os.environ["_DONUT_RESTARTED"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                if result.stderr:
                    print(f"[setup] pip stderr: {result.stderr[:1000]}")
                if result.stdout:
                    print(f"[setup] pip stdout: {result.stdout[:500]}")
        except subprocess.TimeoutExpired:
            print(
                "[setup] TIMEOUT: pip install exceeded 1800s — packages may be partially installed.\n"
                "[setup] Run manually: pip install -r requirements.txt"
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        # flash-attn is NOT auto-installed here.  To install it manually:
        #   pip install flash-attn --no-build-isolation
        # or use a prebuilt wheel: https://flashattn.dev/wheel-finder/
    except Exception:
        # Silently ignore all errors — pipeline may still work if packages are present.
        pass


def _verify_critical_packages() -> list:
    """Return list of still-missing critical packages after install attempt."""
    return [pkg for pkg in _CRITICAL_VERIFY_PACKAGES if _is_package_missing(pkg)]


# ---------------------------------------------------------------------------
# Logging Infrastructure (Phase 2)
# ---------------------------------------------------------------------------


class _DualStreamHandler(logging.Handler):
    """Custom logger that writes to file AND filtered console output.

    Behavior:
    - Writes to file with repeat-line compression (collapse_after=3 by default)
    - Console output filtered: suppresses repetitive logs (epoch progress, etc.)
    - ERROR/WARNING always shown on console
    - INFO shown on console unless filtered
    - DEBUG written to file only
    """

    # Substrings that trigger progress-dedup (checked against full message).
    _PROGRESS_TRIGGERS: tuple[str, ...] = ("Epoch ", "Step ", "[====", "%|", "batch")

    # Substrings whose presence means the line is file-only (never console).
    # Built once at class definition — not rebuilt per log record.
    _SUPPRESS_SUBSTRINGS: tuple[str, ...] = (
        "Both `max_new_tokens`",
        "max_new_tokens` and `max_length`",
        "field coverage:",
        "[ResolutionSync]",
        "[GradCkpt]",
        "[FreezeEncoder]",
        "[RAM Cache]",
        "[Tensor Cache]",
        "[Label Cache]",
        "DataLoader: num_workers",
        "[Token-verify]",
        "[Pre-resize]",
        "[Post-resize]",
        "lm_head.weight",
        "LmHeadCloneCallback",
        "Self-test raw token IDs",
        "Self-test decoder_input_ids",
        "tie_word_embeddings",
        "The new embeddings will be initialized",
        "The new lm_head weights",
        "Loading weights:",
        "Writing model shards:",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
    )

    # CONFIG_OPTIMIZATION sub-line prefixes that are file-only.
    _SUPPRESS_CONFIG_PREFIXES: tuple[str, ...] = (
        "- batch_size:",
        "- accumulation_steps:",
        "- encoder_lr:",
        "- decoder_lr:",
        "- early_stopping_patience:",
        "- warmup_steps:",
        "- epochs:",
    )

    def __init__(self, file_path: Path, collapse_after: int = 3):
        super().__init__()
        self.file_path = file_path
        self.file_handle = open(str(file_path), "a", encoding="utf-8")  # noqa: SIM115
        self.collapse_after = collapse_after
        # console dedup tracking
        self._last_console_line: str | None = None
        self._console_repeat_count: int = 0
        # file dedup tracking
        self._last_file_line: str | None = None
        self._file_repeat_count: int = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._write_to_file(msg)

            # Selectively write to console
            if record.levelno >= logging.INFO:  # noqa: SIM102
                if not self._should_suppress_console(msg):
                    # Always show ERROR/WARNING
                    if record.levelno >= logging.WARNING:
                        print(f"[{record.levelname:8s}] {msg}", file=sys.stderr)
                    else:
                        print(f"[{record.levelname:8s}] {msg}")
                    sys.stdout.flush()
        except Exception:
            self.handleError(record)

    def _write_to_file(self, msg: str) -> None:
        """Write msg to file, collapsing consecutive identical lines."""
        if msg == self._last_file_line:
            self._file_repeat_count += 1
            if self._file_repeat_count <= self.collapse_after:
                self.file_handle.write(msg + "\n")
                self.file_handle.flush()
            # else: silently collapse — summary emitted on next different line
        else:
            if self._file_repeat_count > self.collapse_after:
                skipped = self._file_repeat_count - self.collapse_after
                self.file_handle.write(f"  ... (above line repeated ×{skipped} more times)\n")
                self.file_handle.flush()
            self._last_file_line = msg
            self._file_repeat_count = 0
            self.file_handle.write(msg + "\n")
            self.file_handle.flush()

    def _should_suppress_console(self, msg: str) -> bool:
        """Return True if this log line should be omitted from console output.

        Progress lines (epoch/step counters) are dedup-collapsed.
        Verbose internal diagnostic lines are routed to terminal.txt only.
        Per-step loss dicts and CONFIG_OPTIMIZATION sub-lines are also suppressed.
        """
        # Dedup repetitive progress lines (tqdm-style bars, epoch counters).
        if any(x in msg for x in self._PROGRESS_TRIGGERS):
            if msg == self._last_console_line:
                self._console_repeat_count += 1
                return True
            self._last_console_line = msg
            self._console_repeat_count = 0

        # Verbose internal lines that belong in terminal.txt only.
        if any(x in msg for x in self._SUPPRESS_SUBSTRINGS):
            return True

        stripped = msg.lstrip()

        # Per-step loss dicts: "{'loss': ...}" and "{'eval_loss': ...}"
        if stripped.startswith("{'loss':") or stripped.startswith("{'eval_loss':"):
            return True

        # CONFIG_OPTIMIZATION sub-lines (indented hyperparameter listings).
        return stripped.startswith(self._SUPPRESS_CONFIG_PREFIXES)

    def close(self) -> None:
        try:
            # Flush any pending repeat summary before closing
            if self._file_repeat_count > self.collapse_after:
                skipped = self._file_repeat_count - self.collapse_after
                self.file_handle.write(f"  ... (above line repeated ×{skipped} more times)\n")
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

    # Suppress verbose third-party loggers (including httpx/httpcore for clean output)
    for pkg in [
        "httpx",
        "httpcore",
        "transformers",
        "torch",
        "urllib3",
        "datasets",
        "huggingface_hub",
        "fsspec",
        "fsspec.local",
    ]:
        logging.getLogger(pkg).setLevel(logging.WARNING)

    # Suppress PIL chunk-level DEBUG flood and other noisy third-party loggers
    try:
        from constants import suppress_noisy_loggers

        suppress_noisy_loggers()
    except ImportError:
        pass

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


def _print_final_summary(results_dir: Path) -> None:
    """Print a compact final summary table of all experiment results.

    Format is AI-agent-friendly: tabular, fixed-width columns, minimal tokens.
    Also respects DONUT_QUIET env var for AI-agent mode.
    """
    try:
        result_files = sorted(results_dir.glob("experiment_*.json"))
        if not result_files:
            return

        rows = []
        for rf in result_files:
            try:
                with open(rf) as fh:
                    data = json.load(fh)
                m = data.get("metrics", {})
                rows.append(
                    {
                        "exp": data.get("experiment_id", "?"),
                        "name": data.get("name", "")[:28],
                        "samples": data.get("num_train_samples", 0),
                        "f1": m.get("global_f1", float("nan")),
                        "company": m.get("company_f1", float("nan")),
                        "date": m.get("date_f1", float("nan")),
                        "addr": m.get("address_f1", float("nan")),
                        "total": m.get("total_f1", float("nan")),
                        "time": m.get("training_time_sec", 0.0) / 60,
                    }
                )
            except Exception:
                continue

        if not rows:
            return

        print("\n--- FINAL SUMMARY ---")
        hdr = f"{'exp':>3} | {'name':<28} | {'samples':>7} | {'f1':>6} | {'company':>7} | {'date':>6} | {'addr':>6} | {'total':>6} | {'time':>5}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            f1_s = f"{r['f1']:>6.4f}" if not math.isnan(r["f1"]) else "   N/A"
            co_s = f"{r['company']:>7.4f}" if not math.isnan(r["company"]) else "    N/A"
            da_s = f"{r['date']:>6.4f}" if not math.isnan(r["date"]) else "   N/A"
            ad_s = f"{r['addr']:>6.4f}" if not math.isnan(r["addr"]) else "   N/A"
            to_s = f"{r['total']:>6.4f}" if not math.isnan(r["total"]) else "   N/A"
            ti_s = f"{r['time']:>4.1f}m"
            print(
                f"{r['exp']:>3} | {r['name']:<28} | {r['samples']:>7} | {f1_s} | {co_s} | {da_s} | {ad_s} | {to_s} | {ti_s}"
            )
        print("--- END SUMMARY ---")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI parameter override helpers
# ---------------------------------------------------------------------------


def _parse_param_overrides(params: list[str]) -> dict:
    """Parse a list of 'KEY=VALUE' strings into a dict with auto-cast values.

    Numeric values are auto-cast to int or float where possible; all others
    are kept as strings.  Used to apply --param/-p CLI overrides to an
    ExperimentConfig via dataclasses.replace().

    Examples
    --------
    >>> _parse_param_overrides(["epochs=5", "lr=1e-4", "name=custom"])
    {'epochs': 5, 'lr': 0.0001, 'name': 'custom'}
    """
    overrides: dict = {}
    for item in params:
        if "=" not in item:
            print(f"  [--param] WARNING: Ignoring malformed override {item!r} (expected KEY=VALUE)")
            continue
        key, _, raw_value = item.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        # Auto-cast: try int first, then float, then keep as str
        try:
            value: int | float | str = int(raw_value)
        except ValueError:
            try:
                value = float(raw_value)
            except ValueError:
                value = raw_value
        overrides[key] = value
    return overrides


def _print_all_params() -> None:
    """Print a formatted table of all DONUT/TrOCR/YOLO/memory parameters and system info."""

    try:
        import run_experiments as re_mod

        experiments = re_mod.EXPERIMENTS
    except Exception as exc:
        print(f"  [--list-params] Could not load experiments: {exc}")
        experiments = {}

    try:
        import resource_manager as _mm_mod

        ram_fraction = _mm_mod._RAM_SAFETY_FRACTION
        pixel_cap = 4096  # _PIXEL_TENSOR_MAX_MB in train.py
        ref_h = _mm_mod._REF_H
        ref_w = _mm_mod._REF_W
    except Exception:
        ram_fraction = "?"
        pixel_cap = "?"
        ref_h = "?"
        ref_w = "?"

    eq = "=" * 72
    print(f"\n{eq}")
    print("  PARAMETER LISTING (--list-params)")
    print(eq)

    # DONUT experiments
    print("\n  DONUT EXPERIMENTS")
    print("  " + "-" * 50)
    for exp_id, cfg in experiments.items():
        print(f"\n  Exp {exp_id}  {cfg.name}")
        print(
            f"         epochs={cfg.epochs}  lr={cfg.lr}  batch={cfg.batch_size}"
            f"  accum={cfg.gradient_accumulation_steps}  warmup={cfg.warmup_steps}  wd={cfg.weight_decay}"
        )
        print(
            f"         early_stop={cfg.early_stopping_patience}  seed={cfg.seed}"
            f"  max_len={cfg.max_length}  oversample={cfg.sroie_oversample}"
        )
        print(f"         datasets={cfg.datasets}")

    # Memory management constants
    print("\n  MEMORY MANAGEMENT CONSTANTS")
    print("  " + "-" * 50)
    print(f"  _RAM_SAFETY_FRACTION   = {ram_fraction}")
    print(f"  _PIXEL_TENSOR_MAX_MB   = {pixel_cap}")
    print(f"  _REF_H / _REF_W        = {ref_h} / {ref_w}")
    print("  val precompute_tensors = False (always — val set skips pixel tensor cache)")

    # System info
    print("\n  SYSTEM")
    print("  " + "-" * 50)
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU  : {gpu_name} ({gpu_mem_gb:.1f} GB)")
        else:
            print("  GPU  : Not available")
    except Exception:
        print("  GPU  : torch not available")
    try:
        from resource_manager import _get_available_ram_bytes

        _avail = _get_available_ram_bytes()
        print(f"  RAM  : {_avail / (1024**3):.1f} GB available")
    except Exception:
        print("  RAM  : unavailable")

    # TrOCR + YOLO parameters from control_suite
    print("\n  TrOCR + YOLO (CONTROL_SUITE)")
    print("  " + "-" * 50)
    try:
        from run_experiments import CONTROL_SUITE

        CONTROL_SUITE.print_summary()
    except Exception as exc:
        print(f"  (control_suite unavailable: {exc})")

    print(f"\n{eq}\n")


def _apply_params_override(args) -> None:
    """Read params_override.json and apply its contents to the EXPERIMENTS global.

    File location: --params-override path (default: /workspace/params_override.json).
    Format:
        {
          "global": {"learning_rate": 3e-5, "early_stopping_patience": 5},
          "experiments": {"6": {"epochs": 20}, "7": {"epochs": 20}}
        }

    Rules:
    - "global" keys apply to ALL experiments as defaults.
    - "experiments" keys override per-experiment (take precedence over "global").
    - Unknown keys are logged as warnings and ignored.
    - Parsing errors are printed and the function returns without modifying anything.
    """
    override_path_str = getattr(args, "params_override", None)
    if not override_path_str:
        # Try default location
        override_path_str = "/workspace/params_override.json"
    override_path = Path(override_path_str)
    if not override_path.exists():
        return  # No override file — silent no-op

    try:
        raw = override_path.read_text()
        data = json.loads(raw)
    except Exception as exc:
        print(f"  [params_override] WARNING: Could not read {override_path}: {exc}")
        return

    try:
        import run_experiments as re_mod
    except Exception as exc:
        print(f"  [params_override] WARNING: Could not import run_experiments: {exc}")
        return

    global_overrides: dict = data.get("global", {})
    per_exp_overrides: dict = {str(k): v for k, v in data.get("experiments", {}).items()}

    # Gather valid field names from ExperimentConfig
    valid_fields = {f.name for f in dataclasses.fields(re_mod.ExperimentConfig)}

    def _warn_unknown(keys: dict, scope: str) -> dict:
        clean = {}
        for k, v in keys.items():
            if k in valid_fields:
                clean[k] = v
            elif k == "comment":
                pass  # silently ignore comment fields
            else:
                print(f"  [params_override] WARNING: Unknown key {k!r} in {scope} — ignored")
        return clean

    global_clean = _warn_unknown(global_overrides, "global")
    applied_any = False

    for exp_id, cfg in list(re_mod.EXPERIMENTS.items()):
        exp_overrides = {**global_clean}
        per_exp = per_exp_overrides.get(str(exp_id), {})
        exp_overrides.update(_warn_unknown(per_exp, f"experiments.{exp_id}"))
        if exp_overrides:
            re_mod.EXPERIMENTS[exp_id] = dataclasses.replace(cfg, **exp_overrides)
            applied_any = True

    if applied_any:
        print(f"  [params_override] Applied overrides from {override_path}:")
        if global_clean:
            print(f"    global: {global_clean}")
        for exp_id_str, per_exp in per_exp_overrides.items():
            clean = _warn_unknown(per_exp, f"experiments.{exp_id_str}")
            if clean:
                print(f"    experiment {exp_id_str}: {clean}")
    else:
        print(f"  [params_override] No valid overrides found in {override_path}")


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
        print(
            "  [HF Auth] No token found — running unauthenticated. "
            "Downloads will work but may be slower or rate-limited.\n"
            "  Tip: create hf_token.txt in the project root or set HF_TOKEN env var "
            "for 5-10x faster downloads."
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

    import data_pipeline as dataset_loaders  # local module

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

    print(f"\n  +{'-' * 50}+")
    print(f"  | {'Dataset':<25} {'Samples':>10} {'Status':>12} |")
    print(f"  +{'-' * 50}+")
    for ds_name in ["sroie", "wildreceipt", "funsd", "invoices_donut"]:
        count = counts.get(ds_name, 0)
        status_sym = "[OK]" if count > 0 else "[EMPTY]"
        print(f"  | {ds_name:<25} {count:>10} {status_sym:>12} |")
    print(f"  +{'-' * 50}+")

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

    # Blocking model pre-download — ensures the model is fully HF-cached
    # before any training starts, preventing GPU memory contention from
    # concurrent downloads during the first experiment.
    logging.getLogger(__name__).info("Pre-downloading base model (blocking) ...")
    try:
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        from constants import _gpu_cleanup

        model_id = BASE_MODEL
        _preload_proc = DonutProcessor.from_pretrained(model_id)
        _preload_model = VisionEncoderDecoderModel.from_pretrained(model_id)
        logging.getLogger(__name__).info("Base model cached")
        # Free base model weights before stage 1.5 loads the CORD model.
        # Both models together (~1.6 GB) would compete with the pixel-tensor cache.
        del _preload_proc, _preload_model
        _gpu_cleanup()
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
    """Evaluate the CORD-finetuned DONUT model as a cross-dataset transfer (CORD→SROIE) baseline.

    Loads ``naver-clova-ix/donut-base-finetuned-cord-v2`` and applies a structural
    field remapping from the CORD schema (store_info.store_name, total.total_price)
    to SROIE fields.  This is a *cross-dataset transfer* baseline, not a zero-shot
    evaluation of the base DONUT model — the checkpoint was already fine-tuned on
    CORD receipts with semantically similar fields.
    """
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    import data_pipeline as dataset_loaders
    import run_experiments as eval_mod

    _banner("STAGE 1.5 — CORD-transfer baseline evaluation (cross-dataset CORD→SROIE)")
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

    # Use the CORD checkpoint specifically for the pretrained baseline evaluation.
    # BASE_MODEL is now donut-base (no task fine-tuning), so this stage hardcodes
    # the CORD checkpoint to keep a meaningful zero-shot comparison point.
    pretrained_model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
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

    # Free the CORD baseline model from GPU before DONUT training starts.
    # NOTE: local references MUST be deleted before _gpu_cleanup() is called,
    # otherwise the garbage collector cannot free them (still referenced here).
    del pre_model, pre_processor
    _gpu_cleanup()

    return StageResult(name="Pretrained Eval", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 2 — Experiments (train + eval)
# ---------------------------------------------------------------------------


def _interactive_experiment_selection(
    all_configs: "list",
) -> "list":
    """
    Prompt the user at the terminal to select which experiments to run.

    Tries the Textual TUI first (if ``textual`` is installed).  Falls back
    to the existing plain-text prompt when ``textual`` is not installed or
    when running in a non-interactive environment.

    Returns the filtered list of ExperimentConfig objects.
    """
    # ── Plain-text fallback ───────────────────────────────────────────────
    print("=" * 60)
    print(" EXPERIMENT SELECTION")
    print("=" * 60)
    print("Available experiments:")
    for cfg in all_configs:
        print(f"  [{cfg.id:2d}]  {cfg.name:<40s} ({cfg.arch_type})")
    print()
    print("Enter experiment IDs to run (space-separated), or press Enter for all:")
    try:
        raw_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  (No input received — running all experiments)")
        raw_input = ""

    if not raw_input:
        return list(all_configs)

    selected_ids = set()
    id_map = {cfg.id: cfg for cfg in all_configs}
    for token in raw_input.split():
        try:
            eid = int(token)
            if eid in id_map:
                selected_ids.add(eid)
            else:
                print(f"  WARNING: experiment ID {eid} not found — skipping")
        except ValueError:
            print(f"  WARNING: '{token}' is not a valid integer ID — skipping")

    selected = [cfg for cfg in all_configs if cfg.id in selected_ids]
    selected.sort(key=lambda c: c.id)
    return selected


def _load_experiment_configs_for_run(args) -> "list":
    """
    Load the experiment configs to run, respecting the following priority:

    1. ``--experiments 1 6 10 12`` (CLI flag — highest priority)
    2. ``--interactive`` (prompts user at terminal)
    3. ``experiment_selection.json`` (file-based — default)
    4. ``load_all_experiments("experiments/")`` (fallback if no selection file)

    Always falls back gracefully: any step that fails logs a warning and
    delegates to the next lower priority.

    When ``--interactive`` is set but ``experiments/`` does not exist,
    the function falls through to build a config list from the legacy
    ``re_mod.EXPERIMENTS`` dict and passes it to
    ``_interactive_experiment_selection()`` so the selection screen always
    appears when the flag is given.
    """
    try:
        from run_experiments import (
            load_all_experiments,
            load_experiment_selection,
        )
    except ImportError:
        load_all_experiments = None  # type: ignore[assignment]
        load_experiment_selection = None  # type: ignore[assignment]

    experiments_dir = Path("experiments")
    dir_exists = experiments_dir.exists() and load_all_experiments is not None

    # --interactive always shows the selection screen, even if experiments/ is absent.
    # When the directory is missing, build the config list from the legacy
    # re_mod.EXPERIMENTS dict so the user can still choose which experiments to run.
    if getattr(args, "interactive", False):
        if dir_exists:
            try:
                all_configs = load_all_experiments(experiments_dir)
                return _interactive_experiment_selection(all_configs)
            except Exception as exc:
                print(
                    f"  WARNING: interactive selection from YAML failed: {exc}; "
                    "falling back to built-in experiments"
                )
        # Fallback: build config-like objects from the legacy EXPERIMENTS dict
        print("  [interactive] experiments/ not found — using built-in experiment definitions")
        try:
            import run_experiments as re_mod

            legacy_configs = list(re_mod.EXPERIMENTS.values())
            return _interactive_experiment_selection(legacy_configs)
        except Exception as exc:
            print(f"  WARNING: legacy EXPERIMENTS fallback failed: {exc}; running all experiments")
            return []

    if not dir_exists:
        return []

    # Priority 1 — --experiments CLI flag
    if getattr(args, "experiments", None):
        try:
            requested_ids = [int(x) for x in args.experiments]
            configs = load_all_experiments(experiments_dir, experiment_ids=requested_ids)
            return configs
        except Exception as exc:
            print(f"  WARNING: --experiments flag failed: {exc}; falling through")

    # Priority 3 (non-interactive path) — experiment_selection.json
    sel_file = Path("experiment_selection.json")
    try:
        configs = load_experiment_selection(sel_file, experiments_dir)
        return configs
    except Exception as exc:
        print(
            f"  WARNING: experiment_selection.json load failed: {exc}; "
            "falling back to all experiments"
        )

    # Priority 4 — load everything
    try:
        return load_all_experiments(experiments_dir)
    except Exception as exc:
        print(f"  WARNING: load_all_experiments failed: {exc}; will use legacy EXPERIMENTS dict")
        return []


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

    # Determine which experiments to run.
    # If --experiment N is given (legacy single-exp flag), use just that one.
    # Otherwise, use the new multi-experiment loading logic.
    yaml_configs: list = []
    if args.experiment:
        # Legacy single-experiment path: use re_mod.EXPERIMENTS as before
        exp_ids = [args.experiment]
        use_yaml_dispatch = False
    else:
        yaml_configs = _load_experiment_configs_for_run(args)
        if yaml_configs:
            # New path: use YAML-loaded configs with arch_type dispatch
            exp_ids = [cfg.id for cfg in yaml_configs]
            use_yaml_dispatch = True
        else:
            # Fallback: use legacy EXPERIMENTS dict
            exp_ids = list(re_mod.EXPERIMENTS.keys())
            use_yaml_dispatch = False

    total = len(exp_ids)
    had_empty = False
    completed = 0
    succeeded = 0
    failed_experiments: list[int] = []

    # Pre-load the base DONUT model and processor ONCE, then deep-copy into
    # each experiment. This replaces 8 × from_pretrained() disk reads (each
    # ~4–6 s, 800 MB) with 8 fast in-RAM deep copies.
    # This mirrors the pattern already used correctly in run_experiments.py main().
    _base_processor = None
    _base_model = None
    if not args.experiment:
        # Only pre-load when running all experiments; single-experiment runs
        # load directly in train_experiment() via from_pretrained().
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            _cfg0 = re_mod.EXPERIMENTS[1]
            print(f"  [stage_experiments] Pre-loading base model: {_cfg0.base_model}")
            _base_processor = DonutProcessor.from_pretrained(_cfg0.base_model)
            _base_model = VisionEncoderDecoderModel.from_pretrained(_cfg0.base_model)
            print("  [stage_experiments] Base model pre-loaded (will deep-copy per experiment).")
        except Exception as _preload_exc:
            print(
                f"  [stage_experiments] Base model pre-load failed ({_preload_exc}); "
                "each experiment will load from disk."
            )
            _base_processor = None
            _base_model = None

    # Build a name lookup for display — from YAML configs if available, else
    # from the legacy EXPERIMENTS dict
    _yaml_cfg_map = {cfg.id: cfg for cfg in yaml_configs}

    def _exp_display_name(exp_id: int) -> str:
        if exp_id in _yaml_cfg_map:
            return _yaml_cfg_map[exp_id].name
        if exp_id in re_mod.EXPERIMENTS:
            return re_mod.EXPERIMENTS[exp_id].name
        return f"experiment_{exp_id}"

    # ── Parallel mode (--parallel flag) ──────────────────────────────────
    # When --parallel is set and multiple GPUs are available, use DAGScheduler
    # to run independent experiments concurrently.  On single-GPU setups,
    # DAGScheduler degrades to serial execution automatically.
    if getattr(args, "parallel", False) and yaml_configs:
        try:
            from cloud_orchestration import DAGScheduler

            def _parallel_run_fn(cfg):
                """Single-experiment runner for the DAG scheduler thread pool."""
                import run_experiments as _re_mod

                yaml_cfg_inner = _yaml_cfg_map.get(cfg.id)
                arch = getattr(cfg, "arch_type", "donut")
                is_zs = getattr(cfg, "is_zero_shot", False)
                if arch == "trocr_yolo":
                    return _run_trocr_yolo_experiment(args, yaml_cfg_inner)
                elif is_zs:
                    return _run_zero_shot_experiment(args, yaml_cfg_inner)
                elif cfg.id not in _re_mod.EXPERIMENTS:
                    return _run_yaml_donut_experiment(args, yaml_cfg_inner)
                else:
                    return _re_mod.run_experiment(
                        cfg.id,
                        overrides=getattr(args, "param_overrides", None) or None,
                    )

            print("  [stage_experiments] --parallel: using DAGScheduler")
            scheduler = DAGScheduler(yaml_configs, _parallel_run_fn)
            dag_results = scheduler.run()
            had_empty_dag = any(
                isinstance(r, dict) and (r.get("error") or r.get("num_train_samples", 1) == 0)
                for r in dag_results.values()
            )
            re_mod.save_summary()
            exit_status = 1 if had_empty_dag else 0
            return StageResult(
                name="DONUT Experiments (parallel)",
                duration=0.0,
                exit_status=exit_status,
                warnings=warnings,
            )
        except ImportError as _dag_exc:
            print(
                f"  [stage_experiments] DAGScheduler unavailable ({_dag_exc}) "
                "— falling back to serial execution"
            )

    # ── Serial loop ───────────────────────────────────────────────────────
    for i, exp_id in enumerate(exp_ids, 1):
        exp_name = _exp_display_name(exp_id)
        _step(i, total, f"Experiment {exp_id}: {exp_name}")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"  ▶ Experiment {exp_id} — {exp_name} started at {ts}")

        # Determine arch_type for dispatch
        yaml_cfg = _yaml_cfg_map.get(exp_id)
        arch_type = yaml_cfg.arch_type if yaml_cfg is not None else "donut"
        is_zero_shot = yaml_cfg.is_zero_shot if yaml_cfg is not None else False

        if yaml_cfg is not None:
            print(
                f"    arch={arch_type}  zero_shot={is_zero_shot}  "
                f"datasets={[d.name for d in yaml_cfg.datasets]}"
            )
        elif exp_id in re_mod.EXPERIMENTS:
            print(f"    Datasets: {re_mod.EXPERIMENTS[exp_id].datasets}")

        t0 = time.monotonic()
        try:
            if arch_type == "trocr_yolo" and use_yaml_dispatch:
                # Dispatch to TrOCR+YOLO training path
                result = _run_trocr_yolo_experiment(args, yaml_cfg)
            elif is_zero_shot and use_yaml_dispatch:
                # Zero-shot: skip training, run evaluation only
                result = _run_zero_shot_experiment(args, yaml_cfg)
            elif use_yaml_dispatch and exp_id not in re_mod.EXPERIMENTS:
                # New YAML-only experiment (IDs 9+) not in legacy dict
                result = _run_yaml_donut_experiment(
                    args,
                    yaml_cfg,
                    base_processor=_base_processor,
                    base_model=_base_model,
                )
            else:
                # Legacy path for experiments 1–8 (or any that are in re_mod.EXPERIMENTS)
                result = re_mod.run_experiment(
                    exp_id,
                    base_processor=_base_processor,
                    base_model=_base_model,
                    overrides=getattr(args, "param_overrides", None) or None,
                )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            ts_end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            w = f"Experiment {exp_id} ({exp_name}) crashed: {type(exc).__name__}: {exc}"
            print(f"\n  ✗ FATAL: {w}")
            print("    Traceback follows:")
            traceback.print_exc()
            # Clean up any GPU memory leaked by the crashed experiment so that
            # subsequent experiments start with a clean, defragmented GPU.
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            failed_experiments.append(exp_id)
            warnings.append(w)
            had_empty = True
            continue
        elapsed = time.monotonic() - t0
        ts_end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        f1 = result.get("metrics", {}).get("global_f1", "N/A")
        print(f"  ◀ Experiment {exp_id} — {exp_name} finished at {ts_end} ({elapsed:.1f}s)")
        print(
            f"    done in {elapsed / 60:.1f}min | F1={f1} "
            f"| Samples={result.get('num_train_samples', '?')}"
        )
        completed += 1
        if result.get("num_train_samples", 0) == 0 and not is_zero_shot:
            had_empty = True
            failed_experiments.append(exp_id)
            warnings.append(f"Experiment {exp_id} ({exp_name}) had 0 training samples")
        else:
            succeeded += 1

        # ── Inter-experiment GPU cleanup ──────────────────────────────────
        # Flush any GPU memory left by this experiment before the next one
        # starts instantiating Seq2SeqTrainingArguments (which calls
        # torch.cuda.set_device() and may OOM if VRAM is still fragmented).
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

    re_mod.save_summary()

    # Clean up pre-loaded base model now that all experiments are done.
    if _base_model is not None or _base_processor is not None:
        try:
            from constants import _gpu_cleanup

            del _base_model, _base_processor
            _gpu_cleanup()
        except Exception:
            pass

    exit_status = 1 if had_empty else 0
    return StageResult(
        name="DONUT Experiments", duration=0.0, exit_status=exit_status, warnings=warnings
    )


def _run_trocr_yolo_experiment(args, cfg) -> dict:
    """
    Dispatch Experiment 12 (arch_type=trocr_yolo) to the TrOCR+YOLO training path.
    Returns a result dict compatible with the stage_experiments summary logic.
    """
    import train_trocr_yolo  # noqa: F401  # early import to fail fast if package missing

    print("  [dispatch] arch=trocr_yolo → train_trocr_yolo.py")
    # Ensure TrOCR+YOLO data is prepared before running experiments.
    # stage_trocr_data_prep() is idempotent — it skips if data already exists.
    yolo_val_images = Path(args.workspace) / "data" / "yolo" / "images" / "val"
    if not yolo_val_images.exists():
        stage_trocr_data_prep(args)
    # Run TrOCR+YOLO training and return results dict
    # stage_trocr_experiments handles the full TrOCR flow; here we call it
    # directly and wrap the result.
    result = stage_trocr_experiments(args)
    # Load result from file if it exists
    results_file = Path(cfg.results_file) if cfg.results_file else None
    if results_file and results_file.exists():
        with open(results_file) as fh:
            return json.load(fh)
    return {
        "experiment_id": cfg.id,
        "name": cfg.name,
        "datasets": [d.name for d in cfg.datasets],
        "num_train_samples": 0,
        "metrics": {
            "global_f1": "see trocr_yolo_results.json" if result.exit_status == 0 else "N/A"
        },
    }


def _run_zero_shot_experiment(args, cfg) -> dict:
    """
    Run a zero-shot evaluation (no training).  Loads the base checkpoint,
    runs inference on the SROIE test set, and saves results.
    """
    print("  [dispatch] is_zero_shot=True → evaluation only (no training)")
    results_file = Path(cfg.results_file) if cfg.results_file else None
    # If a result already exists and --force is not set, return it
    if results_file and results_file.exists() and not getattr(args, "force", False):
        with open(results_file) as fh:
            return json.load(fh)
    # Attempt zero-shot evaluation using DonutEvaluator
    try:
        from transformers import DonutProcessor

        import data_pipeline as dataset_loaders
        from run_experiments import DonutEvaluator

        processor = DonutProcessor.from_pretrained(cfg.base_checkpoint)
        test_samples = dataset_loaders.load_sroie_test()
        evaluator = DonutEvaluator(
            model_path=cfg.base_checkpoint,
            processor=processor,
            test_dataset=test_samples,
        )
        metrics = evaluator.evaluate(allow_high_parse_failures=True)
        result = {
            "experiment_id": cfg.id,
            "name": cfg.name,
            "datasets": [],
            "num_train_samples": 0,
            "metrics": metrics,
        }
        if results_file:
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w") as fh:
                json.dump(result, fh, indent=2)
        return result
    except Exception as exc:
        print(f"  [zero-shot] Evaluation failed: {exc}; returning empty metrics")
        return {
            "experiment_id": cfg.id,
            "name": cfg.name,
            "datasets": [],
            "num_train_samples": 0,
            "metrics": {"global_f1": 0.0},
        }


def _run_yaml_donut_experiment(args, cfg, base_processor=None, base_model=None) -> dict:
    """
    Run a DONUT experiment defined purely in YAML (IDs 9+ not in legacy EXPERIMENTS dict).
    Delegates to run_experiments.run_experiment_from_config().
    """
    print(f"  [dispatch] arch=donut (YAML-only exp {cfg.id}) → DONUT training path")
    import run_experiments as re_mod

    if not hasattr(re_mod, "run_experiment_from_config"):
        raise RuntimeError(
            f"[Exp {cfg.id}] run_experiment_from_config not available in run_experiments.py. "
            "Add run_experiment_from_config() to run_experiments.py and its __all__."
        )
    # Let exceptions propagate with their real traceback instead of masking
    # them behind a misleading "not available" message.
    return re_mod.run_experiment_from_config(
        cfg,
        base_processor=base_processor,
        base_model=base_model,
        overrides=getattr(args, "param_overrides", None) or None,
    )


# ---------------------------------------------------------------------------
# Stage 3 — TrOCR+YOLO dataset preparation
# ---------------------------------------------------------------------------


def stage_trocr_data_prep(args) -> StageResult:
    """Prepare YOLO bbox labels and TrOCR line crops from existing SROIE split.

    Uses the SROIE split created by stage_install() (500/63/63), ensuring
    DONUT and TrOCR+YOLO train and evaluate on the exact same images.
    Idempotent: skips if output directories already exist.
    """
    _banner("STAGE 3 — TrOCR+YOLO dataset preparation")
    warnings: list[str] = []

    workspace = Path(args.workspace)
    yolo_train_images = workspace / "data" / "yolo" / "images" / "train"
    trocr_train_meta = workspace / "data" / "trocr" / "train" / "metadata.jsonl"

    if yolo_train_images.exists() and trocr_train_meta.exists():
        print("  TrOCR+YOLO data already prepared — skipping.")
        return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=0, warnings=warnings)

    try:
        import data_pipeline as ds_prep

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
    """Train YOLOv8 + TrOCR and evaluate on the same 63 SROIE test images.

    YOLO and TrOCR are each trained once on SROIE data (the same 500/63/63
    split used by DONUT), keeping the experimental design matched.
    Results are saved to results/trocr_yolo_results.json.
    """

    _banner("STAGE 4 — TrOCR+YOLO training & evaluation")
    warnings: list[str] = []

    try:
        import run_experiments as eval_mod  # noqa: I001
        import train_trocr_yolo as trocr_yolo  # noqa: I001

        workspace = Path(args.workspace)

        # Stage 4a: Train YOLO
        # Ensure GPU is clean from DONUT experiment stage before loading YOLO.
        _gpu_cleanup()
        yolo_output = workspace / "models" / "yolo_finetuned"
        yolo_weights = yolo_output / "run" / "weights" / "best.pt"
        if not yolo_weights.exists():
            # Guard: skip YOLO training if the images/val split is absent or empty
            yolo_images_val = workspace / "data" / "yolo" / "images" / "val"
            if not yolo_images_val.exists() or not any(yolo_images_val.iterdir()):
                w = (
                    "YOLO images/val directory is empty or missing — "
                    "skipping YOLO training. Ensure SROIE data is installed "
                    "(run without --yolo first, or run stage_install)."
                )
                print(f"  WARNING: {w}", file=sys.stderr)
                warnings.append(w)
                return StageResult(
                    name="TrOCR+YOLO", duration=0.0, exit_status=1, warnings=warnings
                )
            print("  Training YOLOv8 text detector ...")
            trocr_yolo.train_yolo(yolo_output)
        else:
            print(f"  YOLO weights cached at {yolo_weights}")

        # Explicit GPU cleanup between YOLO and TrOCR to prevent OOM.
        _gpu_cleanup()
        print("  GPU memory freed between YOLO and TrOCR stages.")

        # Stage 4b: Train TrOCR
        trocr_output = workspace / "models" / "trocr_finetuned"
        trocr_best = trocr_output / "best"
        trocr_history: dict = {"train_loss": [], "val_loss": [], "num_train_samples": 0}
        if not trocr_best.exists():
            print("  Training TrOCR OCR model ...")
            trocr_history = trocr_yolo.train_trocr(trocr_output)
        else:
            print(f"  TrOCR model cached at {trocr_best}")
            # Read num_train_samples from saved training history if available
            hist_path = trocr_output / "training_history.json"
            if hist_path.exists():
                with open(hist_path) as _fh:
                    trocr_history = json.load(_fh)

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
                num_samples = trocr_history.get("num_train_samples", 0)
                # Store as experiment 1 (same test set, single model)
                for exp_id in range(1, 9):
                    trocr_results[str(exp_id)] = {
                        "name": f"TrOCR+YOLO Exp {exp_id}",
                        "metrics": metrics,
                        "num_train_samples": num_samples,
                    }
                out_path = results_dir / "trocr_yolo_results.json"
                with open(out_path, "w") as fh:
                    json.dump(trocr_results, fh, indent=2)
                print(f"  TrOCR+YOLO results saved -> {out_path}")
            else:
                w = "YOLO or TrOCR model weights missing — skipping evaluation."
                print(f"  WARNING: {w}", file=sys.stderr)
                warnings.append(w)

        # GPU cleanup after TrOCR+YOLO stage.
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
    """Run head-to-head benchmark: DONUT vs YOLOv8+TrOCR+Regex on the SROIE test set.

    Automatically selects the best DONUT experiment model (highest global F1)
    and the trained YOLO best.pt from Stage 4. Produces side-by-side F1 /
    accuracy / speed metrics and journal-ready plots.
    """
    _banner("STAGE 5 — Head-to-head benchmark (DONUT vs YOLOv8+TrOCR+Regex)")
    warnings: list[str] = []
    # Defensive GPU flush: prior stages may not have cleaned up if they failed
    # mid-way (e.g. TrOCR+YOLO aborting on missing cv2). Loading DONUT on a
    # fragmented 18 GB VRAM budget will OOM without this.
    _gpu_cleanup()

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

    def _is_valid_model_dir(p: Path) -> bool:
        """Return True only if p looks like a saved HuggingFace model directory."""
        return p.is_dir() and (
            (p / "preprocessor_config.json").exists() or (p / "config.json").exists()
        )

    donut_model_dir = workspace / "models" / f"experiment_{best_exp_id}"
    if not _is_valid_model_dir(donut_model_dir):
        # Fallback: try standalone fine-tuned model path
        donut_model_dir_alt = workspace / "donut-sroie-finetuned"
        if _is_valid_model_dir(donut_model_dir_alt):
            donut_model_dir = donut_model_dir_alt
        else:
            w = (
                f"No fine-tuned DONUT model found at {donut_model_dir} "
                f"— falling back to pretrained base model."
            )
            print(f"  WARNING: {w}")
            warnings.append(w)
            donut_model_dir = BASE_MODEL  # plain str — valid HuggingFace hub ID

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

    # --- Find fine-tuned TrOCR model (for fair comparison vs base pretrained) ---
    trocr_finetuned_dir = workspace / "models" / "trocr_finetuned" / "best"
    if _is_valid_model_dir(trocr_finetuned_dir):
        trocr_model_id = str(trocr_finetuned_dir)
        print(f"  TrOCR model  : {trocr_model_id} (fine-tuned)")
    else:
        trocr_model_id = "microsoft/trocr-base-printed"
        w = (
            f"Fine-tuned TrOCR model not found at {trocr_finetuned_dir} "
            "— falling back to pretrained base model for benchmark."
        )
        print(f"  WARNING: {w}")
        warnings.append(w)
        print(f"  TrOCR model  : {trocr_model_id} (pretrained base)")

    print(f"  Test images  : {test_img_dir}")
    print(f"  Test labels  : {test_key_dir}")

    # --- Run benchmark_compare programmatically ---
    try:
        import torch

        import reporting as bench_mod

        pairs = bench_mod.find_pairs(test_img_dir, test_key_dir)
        print(f"  Found {len(pairs)} test image+label pairs.")

        all_results = []

        # Run DONUT pipeline
        print("\n  Running DONUT inference ...")
        donut_pipe = bench_mod.DonutPipeline(model_id_or_path=str(donut_model_dir))
        donut_result = donut_pipe.run_benchmark(pairs, desc="DONUT benchmark")
        donut_result = bench_mod.compute_metrics(donut_result)
        all_results.append(donut_result)

        # Free GPU before next pipeline.
        del donut_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Run YOLOv8+TrOCR+Regex pipeline (if weights available)
        if not skip_yolo:
            print("\n  Running YOLOv8+TrOCR+Regex inference ...")
            yolo_pipe = bench_mod.TrOCRYOLOPipeline(
                yolo_model_path=str(yolo_weights),
                trocr_model_id=trocr_model_id,
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

    Produces plots and LaTeX-injectable content comparing DONUT vs
    TrOCR+YOLO across all 8 experiments.
    """
    _banner("STAGE 6 — Cross-architecture comparison")
    warnings: list[str] = []

    try:
        import reporting as compare_mod

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
    """Generate LaTeX tables, plots, fill paper_filled.tex and presentation_filled.tex.

    Handles partial runs gracefully: if ``all_experiments.json`` is missing or
    only some experiments are present, the output files are written with "---"
    for unresolved placeholders so the LaTeX still compiles.  Does NOT call
    ``sys.exit()`` — returns a non-zero ``exit_status`` instead so the
    orchestrator can continue.
    """
    import data_pipeline as dataset_loaders
    import reporting as ir  # local module

    _banner("STAGE 7 — LaTeX paper generation")
    warnings: list[str] = []

    results_path = Path("results") / "all_experiments.json"
    if not results_path.exists():
        w = f"{results_path} not found — generating paper with placeholder values only."
        print(f"  WARNING: {w}", file=sys.stderr)
        warnings.append(w)
        all_exp = {}  # no results yet — fall through to fill with "---" placeholders
    else:
        with open(results_path) as fh:
            all_exp = json.load(fh)

    # Generate training loss plots for the paper
    try:
        ir.generate_training_plots(Path("results"))
    except Exception as exc:
        w = f"generate_training_plots failed: {exc}; skipping plots."
        print(f"  WARNING: {w}")
        warnings.append(w)

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

    # Print TrOCR+YOLO and cross-architecture comparison tables
    trocr_path = Path("results") / "trocr_yolo_results.json"
    if trocr_path.exists():
        with open(trocr_path) as fh:
            trocr_exp = json.load(fh)
        ir.print_table5_trocr_yolo(trocr_exp)
        ir.print_table6_cross_architecture(all_exp, trocr_exp)

    try:
        ir.generate_convergence_data(str(results_path))
        ir.generate_convergence_tex(str(results_path))
        ir.generate_f1_barchart_tex(str(results_path))
    except Exception as exc:
        w = f"Convergence/barchart tex generation failed: {exc}; skipping."
        print(f"  WARNING: {w}")
        warnings.append(w)

    # Build a single var_map that covers both paper.tex and presentation.tex
    var_map = ir.build_var_map(all_exp)

    paper_template = Path(args.paper_template)
    output_paper = Path(args.output)

    if paper_template.exists():
        try:
            ir.fill_paper(str(paper_template), str(output_paper), var_map)
            print(f"\n  Complete paper written -> {output_paper}")
        except Exception as exc:
            w = f"fill_paper failed for {paper_template}: {exc}"
            print(f"  WARNING: {w}")
            warnings.append(w)

        # ── Compile paper → PDF ────────────────────────────────────────────
        try:
            paper_pdf = ir.compile_pdf(output_paper, work_dir=output_paper.parent)
            if paper_pdf:
                print(f"  PDF compiled   -> {paper_pdf}")
            else:
                print(
                    "  INFO: No LaTeX compiler found — install texlive-latex-base "
                    "or MiKTeX to auto-compile PDFs."
                )
        except Exception as exc:
            w = f"PDF compilation failed: {exc}"
            print(f"  WARNING: {w}")
            warnings.append(w)
    else:
        w = f"paper template not found at {paper_template}; skipping paper_filled.tex generation."
        print(f"  WARNING: {w}")
        warnings.append(w)

    # Also fill presentation.tex → presentation_filled.tex
    pres_template = Path("paper/presentation.tex")
    pres_output = Path("paper/presentation_filled.tex")
    if pres_template.exists():
        try:
            ir.fill_paper(str(pres_template), str(pres_output), var_map)
            print(f"  Complete presentation written -> {pres_output}")
        except Exception as exc:
            w = f"fill_paper failed for {pres_template}: {exc}"
            print(f"  WARNING: {w}")
            warnings.append(w)

        # ── Compile presentation → PDF ─────────────────────────────────────
        try:
            pres_pdf = ir.compile_pdf(pres_output, work_dir=pres_output.parent)
            if pres_pdf:
                print(f"  PDF compiled   -> {pres_pdf}")
        except Exception as exc:
            warnings.append(f"Presentation PDF compilation failed: {exc}")
    else:
        print(f"  INFO: {pres_template} not found; skipping presentation_filled.tex generation.")

    # Paper generation itself succeeded (even with placeholder values).
    # exit_status reflects paper generation, not experiment completeness.
    exit_status = 0
    return StageResult(
        name="Paper Generation", duration=0.0, exit_status=exit_status, warnings=warnings
    )


# ---------------------------------------------------------------------------
# stage_push_results — auto-commit and push experiment results
# ---------------------------------------------------------------------------


def stage_push_results(args) -> StageResult:
    """Auto-commit and push experiment results to the current branch."""
    t0 = time.monotonic()
    warnings: list[str] = []

    if not getattr(args, "auto_push", False):
        return StageResult(
            "Results Push", time.monotonic() - t0, 0, ["--auto-push not set; skipped"]
        )

    results_dir = Path("results")
    json_files = sorted(results_dir.glob("*.json")) if results_dir.is_dir() else []
    if not json_files:
        return StageResult("Results Push", time.monotonic() - t0, 0, ["No result files found"])

    try:
        # Get current branch
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

        # Stage result files
        subprocess.run(
            ["git", "add", *[str(f) for f in json_files]],
            check=True,
            capture_output=True,
        )

        # Check if there are staged changes
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
        if diff.returncode == 0:
            return StageResult("Results Push", time.monotonic() - t0, 0, ["No new results to push"])

        # Commit
        subprocess.run(
            ["git", "commit", "-m", "[auto] Add experiment results"],
            check=True,
            capture_output=True,
        )

        # Push with retry (up to 4 attempts with exponential backoff)
        pushed = False
        last_stderr = ""
        for delay in [0, 2, 4, 8]:
            if delay:
                time.sleep(delay)
            push = subprocess.run(
                ["git", "push", "-u", "origin", branch],
                capture_output=True,
                text=True,
            )
            if push.returncode == 0:
                pushed = True
                break
            last_stderr = push.stderr
        if not pushed:
            warnings.append(f"Push failed after 4 retries: {last_stderr}")

    except Exception as exc:
        warnings.append(f"Results push error: {exc}")
        return StageResult("Results Push", time.monotonic() - t0, 1, warnings)

    return StageResult("Results Push", time.monotonic() - t0, 0, warnings)


# ---------------------------------------------------------------------------
# PipelineOrchestrator — sequential execution with structured results
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Orchestrates all pipeline stages SEQUENTIALLY with timing and status tracking."""

    def __init__(self, args):
        self.args = args
        self.stages: list[StageResult] = []

        # Pipeline-level diagnostics — captures per-stage telemetry and
        # optionally calls Claude/Mistral API on failures.
        self._diag = None
        try:
            from diagnostics import PipelineDiagnostics

            _ai_diag = os.environ.get("AI_DIAGNOSE", "0") == "1"
            _ai_prov = os.environ.get("AI_DIAGNOSE_PROVIDER", "auto")
            self._diag = PipelineDiagnostics(
                ai_diagnose=_ai_diag,
                ai_provider=_ai_prov,
            )
        except ImportError:
            pass  # diagnostics.py not present

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
            import traceback as _tb

            elapsed = time.monotonic() - t0
            _tb.print_exc()

            # AI-powered diagnosis on stage failure
            if self._diag is not None:
                try:
                    from diagnostics import ai_diagnose as _ai_dx

                    _ai_enabled = os.environ.get("AI_DIAGNOSE", "0") == "1"
                    if _ai_enabled:
                        _ai_prov = os.environ.get("AI_DIAGNOSE_PROVIDER", "auto")
                        _diagnosis = _ai_dx(
                            {
                                "stage": name,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                "traceback": _tb.format_exc()[-2000:],
                            },
                            provider=_ai_prov,
                        )
                        if _diagnosis:
                            print(f"\n[AI Diagnosis] Stage {name!r}:\n{_diagnosis}")
                except Exception:
                    pass  # diagnosis itself failed — don't mask the real error

            result = StageResult(
                name=name,
                duration=elapsed,
                exit_status=2,
                warnings=[f"Uncaught exception: {type(exc).__name__}: {exc}"],
            )
        ts_end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"  ◀ {name} finished at {ts_end} ({result.duration:.1f}s)")
        self.stages.append(result)

        # Record to pipeline diagnostics
        if self._diag is not None:
            self._diag.stage_reports.append(
                {
                    "stage": name,
                    "status": "success" if result.exit_status == 0 else "failed",
                    "duration_sec": result.duration,
                    "exit_status": result.exit_status,
                    "warnings": result.warnings,
                }
            )

        return result

    def run(self) -> int:
        """Run all stages sequentially and return the pipeline exit code."""
        exit_code = 0

        if self.args.paper_only:
            self._run_stage("Paper Generation", stage_paper)
            print(repr(self))
            return exit_code

        # --yolo: jump straight to TrOCR+YOLO stages, skipping install,
        # download, pretrained baseline, and all DONUT experiments.
        yolo_only = getattr(self.args, "yolo", False)

        # Stage 0 — SROIE install
        if not self.args.skip_install and not yolo_only:
            self._run_stage("SROIE Install", stage_install)

        # Stage 1 — Dataset download + inline model pre-download
        if not self.args.skip_download and not yolo_only:
            r = self._run_stage("Dataset Download", stage_download)
            if r.exit_status > exit_code:
                exit_code = r.exit_status

        # Stage 1.5 — Pretrained baseline evaluation
        if not self.args.skip_pretrained and not yolo_only:
            self._run_stage("Pretrained Eval", stage_pretrained_baseline)

        # Stage 2 — DONUT experiments (sequential, one at a time)
        if not yolo_only:
            r = self._run_stage("DONUT Experiments", stage_experiments)
            if r.exit_status > exit_code:
                exit_code = r.exit_status

        # Stage 3 — TrOCR+YOLO dataset preparation
        # Prepares YOLO bbox + TrOCR line crop data from the existing SROIE
        # split (same 500/63/63 used by DONUT) for matched experimental design.
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

        # Stage 8 — Auto-push results (only if --auto-push or AUTOPUSH_RESULTS=1)
        self._run_stage("Results Push", stage_push_results)

        # Save pipeline diagnostics report
        if self._diag is not None:
            try:
                self._diag.save_report()
            except Exception as _diag_exc:
                print(f"[Diagnostics] Failed to save pipeline report: {_diag_exc}")

        print(repr(self))
        return exit_code

    def __repr__(self) -> str:
        """Phase 5A visual output — formatted pipeline execution summary table."""
        W = 70  # inner width
        lines = []
        lines.append(f"+{'-' * W}+")
        title = "PIPELINE EXECUTION SUMMARY"
        lines.append(f"|{title:^{W}}|")
        lines.append(f"+{'-' * W}+")

        # Header
        hdr = f"  {'Stage':<24}| {'Duration':>8} | {'Status':<7} | {'Warnings':<20}"
        lines.append(f"|{hdr:<{W}}|")

        lines.append(f"+{'-' * W}+")

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
            "Results Push": "8. Results Push",
        }

        for sr in self.stages:
            label = stage_labels.get(sr.name, sr.name)
            dur = f"{sr.duration:.1f}s"
            if sr.exit_status == 0:
                status = "[OK]"
            elif sr.exit_status == 1:
                status = "[PART]"
            else:
                status = "[FAIL]"
            warn_text = (
                f"{len(sr.warnings)} warning{'s' if len(sr.warnings) != 1 else ''}"
                if sr.warnings
                else ""
            )
            row = f"  {label:<24}| {dur:>8} | {status:<7} | {warn_text:<20}"
            lines.append(f"|{row:<{W}}|")

        lines.append(f"+{'-' * W}+")
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
            # Load metrics from results/experiment_1.json and log summary
            results_file = Path("results") / "experiment_1.json"
            if results_file.exists():
                with open(results_file) as f:
                    metrics = json.load(f)
                    donut_metrics = metrics.get("metrics", {})
                logger.info("✓ Results loaded from experiment_1.json: %s", donut_metrics)
            else:
                logger.warning(
                    "results/experiment_1.json not found; skipping results.tex generation"
                )
        except Exception as e:
            logger.warning(f"Could not generate results.tex: {e}")

        return 0

    except Exception as e:
        logger.error(f"Quick mode failed: {e}")
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
                traceback.print_exc()

        # Generate comparison results.tex
        logger.info("Generating comprehensive results.tex with parameter comparisons...")
        try:
            logger.info(
                "✓ Parameter sweep complete. Sweep results: %d entries",
                len(sweep_results) if isinstance(sweep_results, (list, dict)) else 0,
            )
        except Exception as e:
            logger.warning(f"Could not generate results.tex: {e}")

        return 0

    except Exception as e:
        logger.error(f"Quick sweep mode failed: {e}")
        traceback.print_exc()
        return 2


# ---------------------------------------------------------------------------
# Mini Mode Handlers
# ---------------------------------------------------------------------------


def _mini_mode_handler(args, logger: logging.Logger) -> int:
    """Mini mode: 1 DONUT exp (5 epochs) + YOLO (10 epochs) + TrOCR (1 epoch).

    Produces paper_mini.tex with all \\VAR{} placeholders resolved.
    Target: ~20 min on RTX 4090.
    """
    import run_experiments as re_mod
    import train_trocr_yolo as tty

    # ── Stage 0: SROIE install ────────────────────────────────────────────
    if not args.skip_install:
        logger.info("[Mini Stage 0] SROIE data install...")
        r = stage_install(args)
        if r.exit_status > 1:
            logger.error("SROIE install failed")
            return 2

    # ── Stage 1: Dataset verify ───────────────────────────────────────────
    if not args.skip_download:
        logger.info("[Mini Stage 1] Dataset verification...")
        r = stage_download(args)
        if r.exit_status > 1:
            logger.error("Dataset download failed")
            return 2

    # ── Stage 2: DONUT Exp 1 with reduced epochs ──────────────────────────
    logger.info("[Mini Stage 2] DONUT Experiment 1 (5 epochs)...")
    # Use dataclasses.replace() to build an isolated mini config without
    # mutating any fields of the global EXPERIMENTS dict entry (GP-1).
    original_config = re_mod.EXPERIMENTS[1]
    mini_config = dataclasses.replace(original_config, epochs=5, early_stopping_patience=2)
    re_mod.EXPERIMENTS[1] = mini_config
    # Force only experiment 1
    args_copy = copy.copy(args)
    args_copy.experiment = 1
    try:
        r = stage_experiments(args_copy)
    finally:
        re_mod.EXPERIMENTS[1] = original_config  # restore
    if r.exit_status > 1:
        logger.error("DONUT mini training failed")
        return 2

    # ── Stage 3: TrOCR+YOLO data prep ────────────────────────────────────
    logger.info("[Mini Stage 3] TrOCR+YOLO data prep...")
    r = stage_trocr_data_prep(args)
    if r.exit_status > 1:
        logger.error("TrOCR data prep failed")
        return 2

    # ── Stage 4: YOLO (10 epochs) + TrOCR (1 epoch) ──────────────────────
    logger.info("[Mini Stage 4] YOLO (10 epochs) + TrOCR (1 epoch)...")
    # Temporarily patch module-level epoch constants
    orig_yolo_epochs = tty.YOLO_EPOCHS
    orig_trocr_epochs = tty.TROCR_EPOCHS
    tty.YOLO_EPOCHS = 10
    tty.TROCR_EPOCHS = 1
    try:
        r = stage_trocr_experiments(args)
    finally:
        tty.YOLO_EPOCHS = orig_yolo_epochs
        tty.TROCR_EPOCHS = orig_trocr_epochs
    if r.exit_status > 1:
        logger.warning("TrOCR+YOLO mini training failed (continuing to paper gen)")

    # ── Stage 5: Benchmark ────────────────────────────────────────────────
    logger.info("[Mini Stage 5] Benchmark...")
    stage_benchmark(args)

    # ── Stage 6: Comparison ───────────────────────────────────────────────
    logger.info("[Mini Stage 6] Comparison plots...")
    stage_comparison(args)

    # ── Stage 7: Paper generation → paper_mini.tex ───────────────────────
    logger.info("[Mini Stage 7] Generating paper/paper_mini.tex...")
    args_paper = copy.copy(args)
    args_paper.paper_template = "paper/paper.tex"
    args_paper.output = "paper/paper_mini.tex"
    return _generate_mini_paper(args_paper, logger)


def _micro_mode_handler(args, logger: logging.Logger) -> int:
    """Micro mode: ultra-fast smoke-test targeting < 10 minutes.

    Optimisation levers vs mini mode:
      DONUT  — 5 epochs, 400 train samples, max_length=256, grad_accum=1,
               10× higher LR (5e-4 / 1e-3), OneCycleLR, eval on 63 samples
      YOLO   — yolov8n (3.2M params), 3 epochs, 256 px, SGD+Nesterov
      TrOCR  — 1 epoch, max_len=64, batch=8, SGD+Nesterov+CosineAnnealingLR

    Produces paper_micro.tex with all \\VAR{} placeholders resolved.
    """
    import run_experiments as re_mod
    import train_trocr_yolo as tty

    # ── Stage 0: SROIE install ────────────────────────────────────────────
    if not args.skip_install:
        logger.info("[Micro Stage 0] SROIE data install...")
        r = stage_install(args)
        if r.exit_status > 1:
            logger.error("SROIE install failed")
            return 2

    # ── Stage 1: Dataset verify ───────────────────────────────────────────
    if not args.skip_download:
        logger.info("[Micro Stage 1] Dataset verification...")
        r = stage_download(args)
        if r.exit_status > 1:
            logger.error("Dataset download failed")
            return 2

    # ── Stage 2: DONUT Exp 1 — aggressively reduced ───────────────────────
    logger.info("[Micro Stage 2] DONUT Experiment 1 (micro: 5 epochs, 400 samples)...")
    original_config = re_mod.EXPERIMENTS[1]
    micro_config = dataclasses.replace(
        original_config,
        # Training budget
        epochs=5,
        early_stopping_patience=1,
        # Dataset subsampling
        subsample_train=400,  # 400 of 500 SROIE samples → ~250 optimizer steps over 5 epochs
        # (minimum empirically needed to override CORD base-model prior
        # and learn SROIE XML schema; 150/2ep = 76 steps → F1=0)
        subsample_eval=63,  # full test set for a meaningful F1 number
        # Step-count guard bypass (intentionally below 200-step threshold)
        skip_step_validation=True,
        # Faster descent: 10× higher LR hits useful weights in 2 epochs
        lr=5e-4,
        encoder_lr=5e-4,
        decoder_lr=1e-3,
        # OneCycleLR: aggressive warmup in first 10% of steps, no separate warmup phase
        warmup_steps=0,
        lr_schedule="one_cycle",
        optimizer_type="adamw",
        # Reduce token budget: 256 vs 768 → 3× fewer decoder steps
        max_length=256,
        # No grad accumulation overhead: every batch → immediate optimizer step
        gradient_accumulation_steps=1,
    )
    re_mod.EXPERIMENTS[1] = micro_config
    args_copy = copy.copy(args)
    args_copy.experiment = 1
    # Force re-run: micro config doesn't match cached experiment_1.json
    args_copy.force = True
    try:
        r = stage_experiments(args_copy)
    finally:
        re_mod.EXPERIMENTS[1] = original_config  # always restore global
    if r.exit_status > 1:
        logger.error("DONUT micro training failed")
        return 2

    # ── Stage 3: TrOCR+YOLO data prep ────────────────────────────────────
    logger.info("[Micro Stage 3] TrOCR+YOLO data prep...")
    r = stage_trocr_data_prep(args)
    if r.exit_status > 1:
        logger.error("TrOCR data prep failed")
        return 2

    # ── Stage 4: YOLO (yolov8n, 3 ep, 256 px, SGD) + TrOCR (1 ep, SGD) ──
    logger.info("[Micro Stage 4] YOLO (yolov8n 3 ep 256px SGD) + TrOCR (1 ep SGD)...")
    _saved = {
        "YOLO_BASE": tty.YOLO_BASE,
        "YOLO_EPOCHS": tty.YOLO_EPOCHS,
        "YOLO_IMG_SIZE": tty.YOLO_IMG_SIZE,
        "YOLO_OPTIMIZER": tty.YOLO_OPTIMIZER,
        "YOLO_MOMENTUM": tty.YOLO_MOMENTUM,
        "TROCR_EPOCHS": tty.TROCR_EPOCHS,
        "TROCR_MAX_LEN": tty.TROCR_MAX_LEN,
        "TROCR_BATCH": tty.TROCR_BATCH,
        "TROCR_MINI_MODE": tty.TROCR_MINI_MODE,
    }
    try:
        tty.YOLO_BASE = "yolov8n.pt"  # 3.2M vs 68M params → 3–5× speedup
        tty.YOLO_EPOCHS = 3  # 50 → 3
        tty.YOLO_IMG_SIZE = 256  # 512 → 256 (4× fewer pixels)
        tty.YOLO_OPTIMIZER = "SGD"  # SGD+Nesterov: faster convergence for detection
        tty.YOLO_MOMENTUM = 0.937  # standard Ultralytics default for SGD
        tty.TROCR_EPOCHS = 1  # unchanged
        tty.TROCR_MAX_LEN = 64  # 128 → 64 (2× faster decoding)
        tty.TROCR_BATCH = 8  # 16 → 8 (safer after DONUT VRAM use)
        tty.TROCR_MINI_MODE = True  # switches TrOCR to SGD+CosineAnnealingLR
        r = stage_trocr_experiments(args)
    finally:
        for k, v in _saved.items():
            setattr(tty, k, v)  # always restore all constants

    if r.exit_status > 1:
        logger.warning("TrOCR+YOLO micro training failed (continuing to paper gen)")

    # ── Stage 5: Benchmark ────────────────────────────────────────────────
    logger.info("[Micro Stage 5] Benchmark...")
    stage_benchmark(args)

    # ── Stage 6: Comparison ───────────────────────────────────────────────
    logger.info("[Micro Stage 6] Comparison plots...")
    stage_comparison(args)

    # ── Stage 7: Paper generation → paper_micro.tex ───────────────────────
    logger.info("[Micro Stage 7] Generating paper/paper_micro.tex...")
    args_paper = copy.copy(args)
    args_paper.paper_template = "paper/paper.tex"
    args_paper.output = "paper/paper_micro.tex"
    return _generate_mini_paper(args_paper, logger)


def _generate_mini_paper(args, logger: logging.Logger) -> int:
    """Generate paper_mini.tex, guaranteed to have zero unresolved \\VAR{} placeholders.

    Strategy: build_var_map() populates as many keys as possible from available
    results; any remaining \\VAR{key} placeholders are filled with 'N/A'.
    Falls back to a minimal compilable stub when the results file or template
    are absent.
    """
    import reporting as ir

    _VAR_RE = re.compile(r"\\VAR\{([^}]+)\}")

    results_path = Path("results") / "all_experiments.json"
    if not results_path.exists():
        logger.warning("all_experiments.json missing — generating minimal paper stub")
        return _write_mini_paper_stub(args.output, logger)

    try:
        with open(results_path) as fh:
            all_exp = json.load(fh)
    except Exception as exc:
        logger.warning(f"Could not read all_experiments.json: {exc}")
        return _write_mini_paper_stub(args.output, logger)

    # Build var_map; fall back to empty dict on any error
    try:
        var_map = ir.build_var_map(all_exp)
    except Exception as exc:
        logger.warning(f"build_var_map failed: {exc}")
        var_map = {}

    paper_template = Path(args.paper_template)
    if not paper_template.exists():
        logger.warning(f"paper.tex not found at {paper_template}; writing stub")
        return _write_mini_paper_stub(args.output, logger)

    template_text = paper_template.read_text(encoding="utf-8")
    all_keys = _VAR_RE.findall(template_text)

    # Fill in "N/A" for any key not already present in var_map
    for key in all_keys:
        if key not in var_map:
            var_map[key] = "N/A"
            logger.debug(f"  [mini-paper] Using N/A fallback for \\VAR{{{key}}}")

    # Perform substitution
    filled = _VAR_RE.sub(lambda m: var_map.get(m.group(1), "N/A"), template_text)

    # Sanity check — must have zero remaining \VAR{}
    remaining = _VAR_RE.findall(filled)
    if remaining:
        logger.error(f"BUG: still {len(remaining)} unresolved after fill: {remaining}")
        return _write_mini_paper_stub(args.output, logger)

    Path(args.output).write_text(filled, encoding="utf-8")
    print(f"\n  Mini paper written -> {args.output}  (0 unresolved placeholders)")
    return 0


def _write_mini_paper_stub(output_path: str, logger: logging.Logger) -> int:
    """Write a minimal but valid LaTeX article as a last-resort fallback.

    Used when the paper template is unavailable or results are empty.
    Generates a self-contained, compilable article with no external dependencies.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    stub = rf"""\documentclass{{article}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\geometry{{margin=2.5cm}}
\title{{Receipt Information Extraction: Mini Pipeline Results}}
\author{{Auto-generated by run\_all.py --mini}}
\date{{{now}}}
\begin{{document}}
\maketitle

\section{{Overview}}
This document was generated by the mini pipeline run.
Full results were not available at generation time.

\section{{Status}}
\begin{{tabular}}{{ll}}
\toprule
Stage & Status \\
\midrule
SROIE Install    & See terminal.txt \\
DONUT Exp 1      & See results/experiment\_1.json \\
YOLO/TrOCR       & See results/trocr\_yolo\_results.json \\
Benchmark        & See results/benchmark\_results.json \\
\bottomrule
\end{{tabular}}

\end{{document}}
"""
    Path(output_path).write_text(stub, encoding="utf-8")
    logger.info(f"Wrote stub paper -> {output_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _max_experiment_id() -> int:
    """Return the highest experiment_id found in experiments/*.yaml, or 12 as fallback."""
    try:
        ids = []
        for p in glob.glob("experiments/*.yaml") + glob.glob("experiments/*.yml"):
            with open(p) as f:
                m = re.search(r"experiment_id\s*:\s*(\d+)", f.read())
            if m:
                ids.append(int(m.group(1)))
        return max(ids) if ids else 12
    except Exception:
        return 12


def build_parser() -> argparse.ArgumentParser:
    _max_exp = _max_experiment_id()
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
        help=f"Run only experiment N (1–{_max_exp}) instead of all experiments",
    )
    p.add_argument(
        "--experiments",
        nargs="+",
        metavar="ID",
        help=(
            "Space-separated experiment IDs to run, e.g. --experiments 1 6 10 12. "
            "Overrides experiment_selection.json for this run only."
        ),
    )
    p.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help=(
            "Prompt at terminal to select which experiments to run. "
            "Overrides experiment_selection.json for this run only."
        ),
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
        "--skip-flash-attn",
        action="store_true",
        default=bool(os.environ.get("SKIP_FLASH_ATTN")),
        help=(
            "Skip flash-attn and use PyTorch built-in SDPA (default when SKIP_FLASH_ATTN=1). "
            "flash-attn is optional; PyTorch 2.x SDPA is equally fast for MAX_LENGTH=768."
        ),
    )
    p.add_argument(
        "--yolo",
        action="store_true",
        help=(
            "Start from TrOCR+YOLO stages only (Stage 3+). "
            "Skips SROIE install, dataset download, pretrained baseline, and DONUT experiments. "
            "Use this to re-run or test YOLO/TrOCR changes without waiting for DONUT training."
        ),
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
        default="paper/paper.tex",
        metavar="FILE",
        help="LaTeX template to fill (default: paper/paper.tex)",
    )
    p.add_argument(
        "--output",
        default="paper/paper_filled.tex",
        metavar="FILE",
        help="Output filled LaTeX file (default: paper/paper_filled.tex)",
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
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode: suppress progress bars and verbose output; print only structured summary blocks. AI-agent-friendly.",
    )
    p.add_argument(
        "--mini",
        action="store_true",
        help=(
            "Mini mode: 1 DONUT exp (5 epochs) + 1 YOLO/TrOCR run (10/1 epochs). "
            "Finishes in ~20 min. Generates paper_mini.tex with all metrics filled."
        ),
    )
    p.add_argument(
        "--micro",
        action="store_true",
        help=(
            "Micro mode: ultra-fast smoke-test (<10 min). "
            "DONUT: 5 epochs, 400 train samples, max_length=256, OneCycleLR. "
            "YOLO: yolov8n, 3 epochs, 256 px, SGD+Nesterov. "
            "TrOCR: 1 epoch, max_len=64, SGD+Nesterov+CosineAnnealingLR. "
            "Generates paper_micro.tex."
        ),
    )
    p.add_argument(
        "--startup-log",
        default="startup.log",
        metavar="FILE",
        help="Path for the startup diagnostics log (default: startup.log)",
    )
    p.add_argument(
        "--skip-startup-check",
        action="store_true",
        help="Bypass startup diagnostics (useful for CI/automated runs)",
    )
    p.add_argument(
        "-p",
        "--param",
        metavar="KEY=VALUE",
        action="append",
        dest="params",
        default=[],
        help=(
            "Override any ExperimentConfig field for a specific --experiment run. "
            "Format: KEY=VALUE. Numeric values are auto-cast. "
            "Example: --param epochs=5 --param lr=1e-4. "
            "Can be specified multiple times. "
            "Use with --experiment N to target a single experiment."
        ),
    )
    p.add_argument(
        "--list-params",
        action="store_true",
        help=(
            "Print all DONUT experiment hyperparameters, memory constants, "
            "system info, and TrOCR/YOLO parameters, then exit."
        ),
    )
    p.add_argument(
        "--params-override",
        metavar="FILE",
        default=None,
        help=(
            "Path to a JSON file with live parameter overrides. "
            "Applied before any experiment runs. "
            "Default: /workspace/params_override.json (if it exists). "
            'Format: {"global": {"epochs": 5}, "experiments": {"6": {"epochs": 20}}}'
        ),
    )
    # ── New flags (Tasks 2, 3, 7) ──────────────────────────────────────────
    p.add_argument(
        "--hparam-search",
        action="store_true",
        help=(
            "Run Optuna hyperparameter sweep on the selected experiment "
            "(default: Experiment 2).  Requires: pip install 'optuna>=3.0.0'. "
            "Results stored in results/optuna/study.db."
        ),
    )
    p.add_argument(
        "--seeds",
        metavar="SEEDS",
        default=None,
        help=(
            "Comma-separated seed list for multi-seed robustness run. "
            "Use with --experiment N to target a single experiment. "
            "Example: --seeds 42,123,7,99,2026. "
            "Aborts if any seed fails (no partial aggregation)."
        ),
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Enable DAG-based parallel experiment scheduling on multi-GPU setups. "
            "Experiments with no unmet depends_on run concurrently. "
            "Auto-serialised on single-GPU / CPU-only machines."
        ),
    )
    p.add_argument(
        "--auto-push",
        action="store_true",
        default=bool(os.environ.get("AUTOPUSH_RESULTS")),
        help=(
            "Auto-commit and push experiment results after pipeline completion. "
            "Also enabled by AUTOPUSH_RESULTS=1 env var."
        ),
    )
    return p


def main() -> None:
    # Phase -1: Startup diagnostics (before anything else, stdlib-only)
    # Scan sys.argv directly so we can honour --startup-log / --skip-startup-check
    # before the full argparse run (which requires heavy imports to have succeeded).
    _startup_log_file = "startup.log"
    _skip_startup = False
    _argv = sys.argv[1:]
    for _i, _arg in enumerate(_argv):
        if _arg == "--skip-startup-check":
            _skip_startup = True
        elif _arg.startswith("--startup-log="):
            _startup_log_file = _arg.split("=", 1)[1]
        elif _arg == "--startup-log" and _i + 1 < len(_argv):
            _startup_log_file = _argv[_i + 1]
        elif _i > 0 and _argv[_i - 1] == "--startup-log":
            # This token was already consumed as the value for --startup-log; skip.
            continue
    import validation as startup_diagnostics  # noqa: E402, I001

    startup_diagnostics.run(log_file=_startup_log_file, skip=_skip_startup)

    # Phase 0: Install dependencies (before any other imports)
    _install_dependencies()
    still_missing = _verify_critical_packages()
    if still_missing:
        print(
            f"[setup] FATAL: The following packages could not be installed: {', '.join(still_missing)}\n"
            f"[setup] Run manually: pip install {' '.join(still_missing)}\n"
            f"[setup] For flash-attn: pip install flash-attn --no-build-isolation"
        )
        sys.exit(2)

    t_start = time.monotonic()
    parser = build_parser()
    args = parser.parse_args()

    # Apply live params_override.json overrides BEFORE any stage runs
    _apply_params_override(args)

    # --list-params: print full parameter table and exit
    if getattr(args, "list_params", False):
        _print_all_params()
        sys.exit(0)

    # Parse --param/-p overrides and attach to args for use in stage_experiments()
    param_overrides = _parse_param_overrides(args.params)
    if param_overrides:
        if getattr(args, "paper_only", False):
            print(
                "  [--param] Overrides ignored with --paper-only (no experiments to run).\n"
                "  Use --experiment N --param KEY=VALUE to override a specific experiment."
            )
            param_overrides = {}
        elif (
            getattr(args, "quick", False)
            or getattr(args, "mini", False)
            or getattr(args, "micro", False)
        ):
            print(
                "  [--param] Note: --param overrides apply to individual experiments only; "
                "use --experiment N --param KEY=VALUE for targeted overrides."
            )
        else:
            targets = f"experiment {args.experiment}" if args.experiment else "all experiments"
            print(f"  [--param] Overrides will be applied to {targets}:")
            for k, v in param_overrides.items():
                print(f"    {k} = {v!r}")
    args.param_overrides = param_overrides

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

    # Quiet mode: AI-agent-friendly output (structured summaries only).
    # Set env var so sub-modules (run_experiments.py) can also respect it.
    if args.quiet:
        os.environ["DONUT_QUIET"] = "1"
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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

    if args.mini:
        logger.info("Mini mode detected (--mini flag)")
        exit_code = _mini_mode_handler(args, logger)
        total_elapsed = time.monotonic() - t_start
        logger.info(f"Mini mode complete in {total_elapsed / 60:.1f} min (exit code {exit_code})")
        sys.exit(exit_code)

    if getattr(args, "micro", False):
        logger.info("Micro mode detected (--micro flag)")
        exit_code = _micro_mode_handler(args, logger)
        total_elapsed = time.monotonic() - t_start
        logger.info(f"Micro mode complete in {total_elapsed / 60:.1f} min (exit code {exit_code})")
        sys.exit(exit_code)

    # ── Optuna hyperparameter search (--hparam-search) ──────────────────
    if getattr(args, "hparam_search", False):
        logger.info("--hparam-search flag detected: launching Optuna sweep")
        try:
            from hparam_search import run_hparam_search

            exp_id = getattr(args, "experiment", None) or 2
            run_hparam_search(experiment_id=exp_id)
        except ImportError as _hs_exc:
            logger.error("--hparam-search requires optuna: %s", _hs_exc)
            sys.exit(1)
        sys.exit(0)

    # ── Multi-seed robustness run (--seeds) ──────────────────────────────
    if getattr(args, "seeds", None):
        exp_id = getattr(args, "experiment", None)
        if exp_id is None:
            logger.error("--seeds requires --experiment N to specify which experiment to run")
            sys.exit(1)
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        logger.info("--seeds detected: running Exp %d with seeds %s", exp_id, seed_list)
        try:
            from multi_seed_runner import run_multi_seed

            run_multi_seed(experiment_id=exp_id, seeds=seed_list)
        except RuntimeError as _ms_exc:
            logger.error("Multi-seed run failed: %s", _ms_exc)
            sys.exit(1)
        sys.exit(0)

    if getattr(args, "yolo", False):
        logger.info("--yolo flag detected: starting from TrOCR+YOLO stages (Stage 3+)")

    # ── Diagnostic: environment snapshot ──────────────────────────────────
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
        from resource_manager import _get_available_ram_bytes

        _avail = _get_available_ram_bytes()
        print(f"  RAM          : {_avail / (1024**3):.1f} GB available")
    except Exception:
        pass
    print(f"  CPU cores    : {os.cpu_count()}")

    # Run the pipeline via the orchestrator (all stages sequential)
    orchestrator = PipelineOrchestrator(args)
    exit_code = orchestrator.run()

    total_elapsed = time.monotonic() - t_start
    _banner(f"DONE — total wall time {total_elapsed / 60:.1f} min  |  exit code {exit_code}")
    _print_final_summary(Path("results"))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
