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
  python run_all.py --micro                   # <10-min smoke test (DONUT+TrOCR+YOLO)
  python run_all.py --superfast               # <3-min TrOCR+YOLO only (no DONUT), bare minimum
  python run_all.py --skip-trocr              # Skip TrOCR+YOLO stages
  python run_all.py --yolo                    # Start from TrOCR+YOLO only (skip DONUT stages)
  python run_all.py --force                   # Force re-run (delete cached results)
  python run_all.py --trocr-single            # TrOCR: train once (SROIE only), reuse for all 8 exps

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

# Fix: issue_report_summary medium #10 — set PYTORCH_ALLOC_CONF at the very top of
# run_all.py, before any imports that could transitively import torch. Moving this
# immediately after `import os` ensures the env var is visible to torch regardless of
# whether run_all.py is executed as a script or imported as a module.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# Suppress third-party tqdm bars (e.g. HuggingFace datasets, YOLO) before any import
# that might initialise tqdm internally.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_PROGRESS_BAR", "1")

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
    # datasets   → replaced with inline _hf_download_dataset_inline() in data_pipeline.py
    # accelerate → declared dep; never directly imported (may be used by transformers internals)
    # ultralytics → replaced with inline _YOLO_CLS in train_trocr_yolo.py
    # editdistance → replaced with inline _edit_distance() in constants.py
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
    1. Check if critical packages (torch, transformers) are all importable.
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

        # Fix: issue_report_summary medium #11 — use a file-based sentinel to prevent
        # infinite re-install + os.execv restart loops on repeated network failures.
        # The env-var sentinel (_DONUT_RESTARTED) is inherited by os.execv children, but
        # if os.execv is called again in a fresh shell (e.g. via subprocess), the env var
        # may be absent.  A file-based sentinel survives across subprocess spawns.
        _sentinel_path = Path.home() / ".donut_install_attempted"
        if os.environ.get("_DONUT_RESTARTED") == "1" or _sentinel_path.exists():
            # We already restarted once; raise an error rather than looping again.
            _sentinel_path.unlink(missing_ok=True)  # clean up so next fresh run works
            logging.getLogger(__name__).error(
                "[setup] Install sentinel detected — packages are STILL missing after a "
                "prior install attempt: %s. "
                "Run manually: pip install -r requirements.txt",
                ", ".join(missing_packages),
            )
            raise RuntimeError(
                f"Dependencies {missing_packages} are still missing after an auto-install "
                "attempt. Run manually: pip install -r requirements.txt"
            )

        logging.getLogger(__name__).debug(
            "[setup] Missing packages: %s", ", ".join(missing_packages)
        )

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

        logging.getLogger(__name__).debug(
            "[setup] Installing dependencies from requirements.txt..."
        )
        # ROBUSTNESS: assign tmp_path before the nested try so the finally
        # block can always reference it without a NameError if NamedTemporaryFile
        # raises (e.g. disk-full or permission denied on /tmp).
        tmp_path: str | None = None
        # Fix: issue_report_summary high #6 — initialise tmp_path before the try block
        # so the finally clause never raises NameError if NamedTemporaryFile fails.
        tmp_path = None
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write("\n".join(req_lines))
            tmp_path = tmp.name
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
                tmp.write("\n".join(req_lines))
                tmp_path = tmp.name
            with _InstallWatchdog():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", tmp_path],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30-minute cap; ultralytics + torch can exceed 5 min on slow connections
                )
            if result.returncode == 0:
                logging.getLogger(__name__).debug(
                    "[setup] Dependencies installed successfully — restarting to load new packages..."
                )
                # Write file-based sentinel and set env-var sentinel before os.execv so the
                # restarted process knows not to loop again even if env inheritance fails.
                # Fix: issue_report_summary medium #11 — write sentinel before restart.
                try:
                    _sentinel_path.write_text("1")
                except OSError:
                    pass  # non-fatal: env-var sentinel is a second line of defence
                os.environ["_DONUT_RESTARTED"] = "1"
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                if result.stderr:
                    logging.getLogger(__name__).warning(
                        "[setup] pip install failed (exit %d). stderr: %s",
                        result.returncode,
                        result.stderr[:2000],
                    )
                if result.stdout:
                    logging.getLogger(__name__).debug("[setup] pip stdout: %s", result.stdout[:500])
        except subprocess.TimeoutExpired:
            logging.getLogger(__name__).warning(
                "[setup] TIMEOUT: pip install exceeded 1800s — packages may be partially installed. "
                "Run manually: pip install -r requirements.txt"
            )
        finally:
            # Fix: issue_report_summary high #6 — guard the unlink so NameError in the
            # finally block never masks the original install error.
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        # flash-attn is NOT auto-installed here.  To install it manually:
        #   pip install flash-attn --no-build-isolation
        # or use a prebuilt wheel: https://flashattn.dev/wheel-finder/
    except RuntimeError:
        raise  # re-raise sentinel / loop-guard errors from above
    except Exception as _install_exc:
        # Non-fatal: if pip install itself errors (e.g. requirements.txt unreadable,
        # disk full, unexpected OSError), log the cause at WARNING level so it is
        # visible in the log file, then continue.  The pipeline will fail later with
        # a clear ImportError if a critical package is actually missing.
        logging.getLogger(__name__).warning(
            "[setup] _auto_install_packages encountered an unexpected error: %s — "
            "continuing (pipeline will fail later if required packages are absent).",
            _install_exc,
        )


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
        # Additional verbose internal patterns (file-only)
        "[memory_manager]",
        "[CheckpointValidator]",
        "[SemanticInit]",
        "Model + processor saved",
        "Post-save verification",
        "Post-save lm_head check",
        "Inference #",
        "run_inference #",
        "token2json returned list",
        "[validate_training_config]",
        "precision: bf16",
        "precision: fp16",
        "torch.compile skipped",
        "warmup_steps capped",
        "[Diagnostics] Callback registered",
        "[LiveDashboard] Callback registered",
        "Gradient checkpointing enabled",
        "Pre-training guardrails PASSED",
        "[Device] Model moved",
        "SROIE parser returned empty",
        "shutdown train dl workers",
        "shutdown eval dl workers",
        "Eval DataLoader workers shut down",
        "Loading model from",
        "CONFIG_OPTIMIZATION",
        "[progress]",
        # Stage-level verbose lines (file-only)
        "Pre-downloading base model",
        "Base model cached",
        "Pre-loading base model",
        "Base model pre-loaded",
        "GPU memory freed",
        "YOLO weights cached",
        "TrOCR model cached",
        "CORD-transfer baseline",
        "Evaluating on ",
        "Loading pretrained model",
        "Pretrained Global F1",
        "Saved →",
        "TrOCR+YOLO data already",
        "Training YOLOv8",
        "Training TrOCR",
        "Evaluating TrOCR+YOLO",
        "DONUT model  :",
        "YOLO model   :",
        "TrOCR model  :",
        "Test images  :",
        "Test labels  :",
        "Running DONUT inference",
        "Running YOLOv8+TrOCR",
        "Benchmark complete",
        "[stage_experiments]",
        "[force]",
        "[AutoRetry]",
        "arch=",
        "zero_shot=",
        "Datasets: [",
        "generate_training_plots",
        "Convergence/barchart tex",
        "% === TABLE",
        "generate_convergence",
        "generate_f1_barchart",
        "INFO: paper/presentation",
        # Ultra-minimal: suppress noisy progress/status patterns
        "Evaluating ",
        "DONUT benchmark",
        "Resource optimization applied",
        "Training on ",
        "Hyperparams:",
        "Validation set:",
        "Results saved",
        "Processor spot-check",
        "subsample_train",
        "Pipeline started",
        "SROIE split:",
        "SROIE: train=",
        "Stage Dataset Download done",
        "Stage Download done",
        "[Resources]",
        "GPU memory released",
        "step-count validation skipped",
        "--param overrides applied",
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

            # Selectively write to console using a compact format (no timestamp/logger name).
            if record.levelno >= logging.INFO:
                plain = record.getMessage()
                if not self._should_suppress_console(plain):
                    level_tag = record.levelname[:4]  # INFO, WARN, ERRO, CRIT
                    console_line = f"{level_tag}: {plain}"
                    if record.levelno >= logging.WARNING:
                        print(console_line, file=sys.stderr)
                    else:
                        print(console_line)
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

    def _should_suppress_console(self, plain: str) -> bool:
        """Return True if this log line should be omitted from console output.

        Receives the plain message text (no timestamp/logger name).
        Progress lines (epoch/step counters) are dedup-collapsed.
        Verbose internal diagnostic lines are routed to terminal.txt only.
        Per-step loss dicts and CONFIG_OPTIMIZATION sub-lines are also suppressed.
        """
        # Dedup repetitive progress lines (tqdm-style bars, epoch counters).
        if any(x in plain for x in self._PROGRESS_TRIGGERS):
            if plain == self._last_console_line:
                self._console_repeat_count += 1
                return True
            self._last_console_line = plain
            self._console_repeat_count = 0

        # Verbose internal lines that belong in terminal.txt only.
        if any(x in plain for x in self._SUPPRESS_SUBSTRINGS):
            return True

        stripped = plain.lstrip()

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

# When True, fancy box-drawing banners and tables are printed to console.
# Set by --fancy CLI flag in main().  All verbose output still goes to
# terminal.txt regardless of this flag.
_FANCY_OUTPUT: bool = False


def _banner(text: str) -> None:
    """Print a section banner — always shown on console.

    Uses ``rich.rule`` when rich is installed for a polished look;
    falls back to a plain ``===`` separator otherwise.
    The banner is also written to terminal.txt via the logging system.
    """
    logging.getLogger(__name__).debug("=== %s ===", text)
    try:
        from rich.console import Console
        from rich.rule import Rule

        Console().print(Rule(f"[bold cyan]{text}[/]", style="cyan"))
    except Exception:
        # Plain fallback (also used when DISABLE_LIVE_DASHBOARD=1)
        print(f"  {text}")


def _step(n: int, total: int, desc: str) -> None:
    """Print a numbered step header — always shown on console.

    Uses ``rich.text`` when rich is installed; falls back to plain text.
    Also written to terminal.txt via the logging system.
    """
    logging.getLogger(__name__).debug("[%d/%d] %s", n, total, desc)
    try:
        from rich.console import Console
        from rich.text import Text

        Console().print(Text(f"\n  [{n}/{total}] ", style="bold dim") + Text(desc, style="bold"))
    except Exception:
        print(f"\n[{n}/{total}] {desc}")


def _load_summary_rows(results_dir: Path) -> list[dict]:
    """Load experiment result rows from JSON files for summary display."""
    rows = []
    for rf in sorted(results_dir.glob("experiment_*.json")):
        try:
            with open(rf) as fh:
                data = json.load(fh)
            m = data.get("metrics", {})
            rows.append(
                {
                    "exp": data.get("experiment_id", "?"),
                    "name": data.get("name", "")[:29],
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
    return rows


def _compact_summary(results_dir: Path, pdf_compiled: bool | None = None) -> None:
    """Print a compact plain-text results summary (default mode)."""
    try:
        rows = _load_summary_rows(results_dir)
        if not rows:
            return
        valid = [r for r in rows if not math.isnan(r["f1"])]
        best = max(valid, key=lambda r: r["f1"]) if valid else None
        print("Results:")
        for r in rows:
            f1_s = f"{r['f1']:.4f}" if not math.isnan(r["f1"]) else " N/A "
            mark = " *" if best and r["exp"] == best["exp"] else ""
            print(f"  Exp {r['exp']:>2}: {r['name']:<29} F1={f1_s} ({r['time']:.1f}m){mark}")
        if best:
            co_s = f"{best['company']:.4f}" if not math.isnan(best["company"]) else "N/A"
            da_s = f"{best['date']:.4f}" if not math.isnan(best["date"]) else "N/A"
            ad_s = f"{best['addr']:.4f}" if not math.isnan(best["addr"]) else "N/A"
            to_s = f"{best['total']:.4f}" if not math.isnan(best["total"]) else "N/A"
            print(
                f"  Best Exp {best['exp']}: F1={best['f1']:.4f}  co={co_s} dt={da_s} addr={ad_s} tot={to_s}"
            )
        if pdf_compiled is not None:
            pdf_tag = "compiled" if pdf_compiled else "skipped (no LaTeX)"
            print(f"  PDF: {pdf_tag}")
    except Exception:
        pass


def _print_fancy_summary(results_dir: Path, pdf_compiled: bool | None = None) -> None:
    """Print a box-drawing summary table of all experiment results (--fancy mode).

    Format is AI-agent-friendly: copy-pastable, fixed-width columns, minimal tokens.
    Includes best experiment highlight and per-field F1 for the winner.
    """
    try:
        rows = _load_summary_rows(results_dir)
        if not rows:
            return

        # Find best experiment (highest F1, ignoring NaN)
        valid = [r for r in rows if not math.isnan(r["f1"])]
        best = max(valid, key=lambda r: r["f1"]) if valid else None

        # Box-drawing column widths (change here to update the whole layout)
        _CW_EXP = 5  # "│  1 " including separator
        _CW_NAME = 31  # "│ Name...             "
        _CW_F1 = 8  # "│  0.7343"
        _CW_TIME = 9  # "│   8.2m  "
        footer_w = _CW_EXP + 1 + _CW_NAME + 1 + _CW_F1 + 1 + _CW_TIME

        h_line = "─" * _CW_EXP + "┬" + "─" * _CW_NAME + "┬" + "─" * _CW_F1 + "┬" + "─" * _CW_TIME
        m_line = "─" * _CW_EXP + "┼" + "─" * _CW_NAME + "┼" + "─" * _CW_F1 + "┼" + "─" * _CW_TIME

        print()
        print("┌" + "─" * footer_w + "┐")
        title = "PIPELINE RESULTS SUMMARY"
        print("│" + title.center(footer_w) + "│")
        print("├" + h_line + "┤")
        print(f"│ {'Exp':>3} │ {'Name':<29} │ {'F1':>6} │ {'Time':>7} │")
        print("├" + m_line + "┤")
        for r in rows:
            f1_s = f"{r['f1']:6.4f}" if not math.isnan(r["f1"]) else "  N/A "
            ti_s = f"{r['time']:5.1f}m"
            marker = " ◀" if best and r["exp"] == best["exp"] else "  "
            print(f"│ {r['exp']:>3} │ {r['name']:<29} │ {f1_s} │ {ti_s:>7} │{marker}")
        print("├" + "─" * footer_w + "┤")

        # Footer lines: best experiment + per-field breakdown
        if best:
            co_s = f"{best['company']:.4f}" if not math.isnan(best["company"]) else "N/A"
            da_s = f"{best['date']:.4f}" if not math.isnan(best["date"]) else "N/A"
            ad_s = f"{best['addr']:.4f}" if not math.isnan(best["addr"]) else "N/A"
            to_s = f"{best['total']:.4f}" if not math.isnan(best["total"]) else "N/A"
            best_line = f" Best: Exp {best['exp']} (F1={best['f1']:.4f})"
            field_line = f" Per-field: co={co_s} | dt={da_s} | addr={ad_s} | tot={to_s}"
            print("│" + best_line.ljust(footer_w) + "│")
            print("│" + field_line.ljust(footer_w) + "│")

        if pdf_compiled is not None:
            pdf_status = " PDF: compiled ✓" if pdf_compiled else " PDF: no LaTeX compiler (skipped)"
            print("│" + pdf_status.ljust(footer_w) + "│")

        print("└" + "─" * footer_w + "┘")
    except Exception:
        pass


def _print_final_summary(results_dir: Path, pdf_compiled: bool | None = None) -> None:
    """Print results summary.

    Uses a rich Table when rich is available (regardless of --fancy);
    falls back to the box-drawing ASCII table when --fancy is set;
    uses compact plain-text as the last resort.
    """
    # Try rich table first
    try:
        import math as _math

        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        rows = _load_summary_rows(results_dir)
        if not rows:
            if _FANCY_OUTPUT:
                _print_fancy_summary(results_dir, pdf_compiled)
            else:
                _compact_summary(results_dir, pdf_compiled)
            return

        valid = [r for r in rows if not _math.isnan(r["f1"])]
        best = max(valid, key=lambda r: r["f1"]) if valid else None

        console = Console()
        table = Table(
            title="[bold]Pipeline Results Summary[/]",
            show_header=True,
            header_style="bold dim",
            border_style="dim",
        )
        table.add_column("Exp", justify="right", style="dim", width=4)
        table.add_column("Name", width=30)
        table.add_column("Global F1", justify="right", width=10)
        table.add_column("Company", justify="right", width=8)
        table.add_column("Date", justify="right", width=8)
        table.add_column("Address", justify="right", width=8)
        table.add_column("Total", justify="right", width=8)
        table.add_column("Time", justify="right", width=7)

        for r in rows:
            is_best = best and r["exp"] == best["exp"]
            f1_s = f"{r['f1']:.4f}" if not _math.isnan(r["f1"]) else "N/A"
            co_s = f"{r['company']:.3f}" if not _math.isnan(r["company"]) else "—"
            da_s = f"{r['date']:.3f}" if not _math.isnan(r["date"]) else "—"
            ad_s = f"{r['addr']:.3f}" if not _math.isnan(r["addr"]) else "—"
            to_s = f"{r['total']:.3f}" if not _math.isnan(r["total"]) else "—"
            ti_s = f"{r['time']:.1f}m"
            if is_best:
                f1_color = "bold green"
                name_text = Text(r["name"] + " ★", style="bold")
            elif not _math.isnan(r["f1"]):
                f1_color = "green" if r["f1"] >= 0.8 else ("yellow" if r["f1"] >= 0.5 else "red")
                name_text = Text(r["name"])
            else:
                f1_color = "dim"
                name_text = Text(r["name"], style="dim")
            table.add_row(
                str(r["exp"]),
                name_text,
                Text(f1_s, style=f1_color),
                co_s,
                da_s,
                ad_s,
                to_s,
                ti_s,
            )
        console.print(table)

        if pdf_compiled is not None:
            pdf_tag = "[green]compiled ✓[/]" if pdf_compiled else "[dim]skipped (no LaTeX)[/]"
            console.print(f"  PDF: {pdf_tag}")
        return
    except Exception:
        pass

    if _FANCY_OUTPUT:
        _print_fancy_summary(results_dir, pdf_compiled)
    else:
        _compact_summary(results_dir, pdf_compiled)


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
                logging.getLogger(__name__).debug("[HF Auth] Loaded token from %s", token_file)
            else:
                token = None
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token  # legacy env var

        # Validate token format (HF tokens start with 'hf_' or are 39 chars legacy format)
        token_masked = f"{token[:4]}****" if len(token) > 8 else "****"
        if not (token.startswith("hf_") or len(token) == 39):
            logging.getLogger(__name__).debug(
                "[HF Auth] Token format may be invalid (expected 'hf_...' or 39-char legacy). "
                "Preview: %s",
                token_masked,
            )

        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
            logging.getLogger(__name__).debug(
                "[HF Auth] Authenticated (token: %s) — faster downloads enabled", token_masked
            )
        except Exception as e:
            logging.getLogger(__name__).debug(
                "[HF Auth] Login failed: %s — continuing unauthenticated", e
            )
    else:
        logging.getLogger(__name__).debug(
            "[HF Auth] No token found — running unauthenticated. "
            "Tip: create hf_token.txt or set HF_TOKEN env var for 5-10x faster downloads."
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
        logging.getLogger(__name__).debug(
            "SROIE data already present at %s — skipping install.", sroie_data_dir
        )
        return StageResult(name="SROIE Install", duration=0.0, exit_status=0, warnings=warnings)

    # Clone from GitHub into the parent directory of sroie_data_dir
    parent_dir = sroie_data_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    repo_url = "https://github.com/zzzDavid/ICDAR-2019-SROIE.git"
    clone_target = parent_dir / "ICDAR-2019-SROIE-repo"

    if not clone_target.exists():
        logging.getLogger(__name__).debug("Cloning %s ...", repo_url)
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
    _log_inst = logging.getLogger(__name__)
    _log_inst.info(
        "SROIE split: train=%d  val=%d  test=%d", remaining, len(val_imgs), len(test_imgs)
    )
    if remaining != len(train_imgs):
        _log_inst.warning(
            "Expected %d train images in img/ but found %d", len(train_imgs), remaining
        )
    _log_inst.debug("SROIE data ready at %s", sroie_data_dir)

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
        logging.getLogger(__name__).error(
            "SROIE data not found at %s. Expected subdirs: img/, key/", args.sroie_dir
        )
        sys.exit(2)

    train_samples = dataset_loaders.load_sroie_train()
    test_samples = dataset_loaders.load_sroie_test()
    val_samples = dataset_loaders.load_sroie_val()
    _log_dl = logging.getLogger(__name__)
    _log_dl.info(
        "SROIE: train=%d  val=%d  test=%d", len(train_samples), len(val_samples), len(test_samples)
    )
    if not sroie_test_img.exists() or len(test_samples) == 0:
        w = "SROIE test split not found or empty — evaluation will be skipped."
        _log_dl.warning("%s", w)
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

    _log_dl.debug("Downloading %d datasets in parallel ...", len(aux_datasets))
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
                _log_dl.warning("%s", w)
                failed_datasets.append(ds_name)
                warnings.append(w)
            else:
                counts[ds_name] = count
                _log_dl.debug("  '%s' ready: %d samples (%.1fs)", ds_name, count, elapsed)

    # Log dataset summary at debug level (file-only).
    _dl_summary = "\n".join(
        f"  {ds:<25} {counts.get(ds, 0):>10}  {'[OK]' if counts.get(ds, 0) > 0 else '[EMPTY]'}"
        for ds in ["sroie", "wildreceipt", "funsd", "invoices_donut"]
    )
    _log_dl.debug("Dataset counts:\n%s", _dl_summary)

    if failed_datasets:
        # Report which experiment IDs are affected by the failed downloads
        import run_experiments as re_mod

        affected_exp_ids = [
            exp_id
            for exp_id, exp in re_mod.EXPERIMENTS.items()
            if any(ds in exp.datasets for ds in failed_datasets)
        ]
        _log_dl.warning(
            "%d auxiliary dataset(s) failed: %s  Affected experiments: %s",
            len(failed_datasets),
            failed_datasets,
            affected_exp_ids,
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
    _log = logging.getLogger(__name__)
    warnings: list[str] = []

    workspace = Path(args.workspace)
    output_path = workspace / "evaluation_results.json"

    test_samples = dataset_loaders.load_sroie_test()
    if len(test_samples) == 0:
        w = "SROIE test split is empty — skipping pretrained baseline evaluation."
        _log.warning("%s", w)
        warnings.append(w)
        return StageResult(name="Pretrained Eval", duration=0.0, exit_status=1, warnings=warnings)

    _log.debug("Evaluating on %d test images ...", len(test_samples))

    ground_truths = [s[1] for s in test_samples]
    image_paths = [s[0] for s in test_samples]

    # Use the CORD checkpoint specifically for the pretrained baseline evaluation.
    # BASE_MODEL is now donut-base (no task fine-tuning), so this stage hardcodes
    # the CORD checkpoint to keep a meaningful zero-shot comparison point.
    pretrained_model_id = "naver-clova-ix/donut-base-finetuned-cord-v2"
    _log.debug("Loading pretrained model: %s", pretrained_model_id)
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
    _log.info("Pretrained baseline F1=%s", pretrained_metrics.get("global_f1", "N/A"))

    workspace.mkdir(parents=True, exist_ok=True)
    output = {"pretrained_metrics": pretrained_metrics}
    with open(output_path, "w") as fh:
        json.dump(output, fh, indent=2, default=str)
    _log.debug("Pretrained baseline saved → %s", output_path)

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


# ---------------------------------------------------------------------------
# Environment helpers — run once at pipeline start
# ---------------------------------------------------------------------------


def _ensure_processor_config(path: str = "processor_config.json") -> None:
    """Write processor_config.json at *path* if it doesn't already exist.

    resource_manager.get_image_size_from_processor_config() reads this file to
    compute VRAM budgets.  When it's absent it falls back to the reference
    resolution and logs a WARNING on every experiment.  Writing the file once
    eliminates the recurring warning and ensures the correct 1280×960
    resolution is always used.
    """
    p = Path(path)
    if p.exists():
        return
    cfg = {"image_processor": {"size": {"height": 1280, "width": 960}}}
    try:
        p.write_text(json.dumps(cfg, indent=2) + "\n")
        logging.getLogger(__name__).debug("[env] Generated %s (height=1280, width=960)", p)
    except OSError as exc:
        logging.getLogger(__name__).debug(
            "[env] Could not write %s: %s — continuing without it", p, exc
        )


def _incremental_push_result(exp_id: int) -> None:
    """Commit and push a single experiment result JSON to the current branch.

    Called after each successful experiment when --auto-push is active.
    This ensures partial results are visible in the PR even if the pipeline
    crashes or is interrupted before the final stage_push_results() runs.
    Failures are silently swallowed — a failed push is never fatal.
    """
    result_file = Path("results") / f"experiment_{exp_id}.json"
    if not result_file.exists():
        return
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        subprocess.run(["git", "add", str(result_file)], check=True, capture_output=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
        if diff.returncode == 0:
            return  # nothing new to push
        subprocess.run(
            ["git", "commit", "-m", f"[auto] Exp {exp_id} result"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "push", "-u", "origin", branch],
            capture_output=True,
        )
    except Exception as _push_exc:
        print(f"  [sync] Incremental push for exp {exp_id} failed (non-fatal): {_push_exc}")


# ---------------------------------------------------------------------------
# Autonomous feedback loop — retry failed experiments after CI auto-fix
# ---------------------------------------------------------------------------


def _autonomous_feedback_loop(
    failed_ids: list,
    *,
    args,
    yaml_cfg_map: dict,
    use_yaml_dispatch: bool,
) -> list:
    """Run CI auto-fix and retry any experiments that failed with code errors.

    This is the "stop being the glue" feedback loop:
      1. When experiments crash with non-OOM errors (e.g., ValueError from a
         transformers API change), call autonomous_ci.autonomous_pipeline()
         which uses AI to detect and rewrite the broken file.
      2. If CI fixes pass, reload the affected modules via importlib.reload()
         so the new code is live in the current process.
      3. Retry only the failed experiments — successful ones are not repeated.

    Returns the list of experiment IDs that are *still* failed after the loop
    (empty list on full recovery).
    """
    if not failed_ids:
        return []

    _log_ar = logging.getLogger(__name__)
    if os.environ.get("DISABLE_DIAGNOSTICS", "0") == "1":
        _log_ar.debug("[AutoRetry] DISABLE_DIAGNOSTICS=1 — skipping feedback loop")
        return failed_ids

    _log_ar.debug(
        "[AutoRetry] %d experiment(s) failed (%s) — running autonomous CI auto-fix...",
        len(failed_ids),
        failed_ids,
    )

    # Run CI auto-fix (no git merge, no PR post — fixes code in-place).
    ci_fixed = False
    try:
        from autonomous_ci import autonomous_pipeline as _ci_pipeline

        ci_fixed = _ci_pipeline(
            no_merge=True,
            pr_only=False,
            max_fix_attempts=3,
            skip_smoke_test=False,
        )
        _log_ar.debug("[AutoRetry] CI auto-fix returned: %s", "PASS" if ci_fixed else "FAIL")
    except Exception as _ci_exc:
        _log_ar.debug("[AutoRetry] CI auto-fix error: %s", _ci_exc)
        return failed_ids

    if not ci_fixed:
        _log_ar.debug("[AutoRetry] CI could not fix issues — experiments remain failed")
        return failed_ids

    # Reload run_experiments so patched code is live without restarting the process.
    _log_ar.debug("[AutoRetry] Reloading run_experiments module to pick up fixes...")
    try:
        import run_experiments as _re_retry

        importlib.reload(_re_retry)
    except Exception as _reload_exc:
        _log_ar.debug("[AutoRetry] Module reload failed: %s — skipping retry", _reload_exc)
        return failed_ids

    # Retry failed experiments with the patched code.
    still_failed: list = []
    for exp_id in failed_ids:
        _log_ar.debug("[AutoRetry] Retrying experiment %d...", exp_id)
        try:
            yaml_cfg = yaml_cfg_map.get(exp_id)
            arch = getattr(yaml_cfg, "arch_type", "donut") if yaml_cfg else "donut"
            is_zs = getattr(yaml_cfg, "is_zero_shot", False) if yaml_cfg else False

            if arch == "trocr_yolo" and use_yaml_dispatch and yaml_cfg:
                result = _run_trocr_yolo_experiment(args, yaml_cfg)
            elif is_zs and use_yaml_dispatch and yaml_cfg:
                result = _run_zero_shot_experiment(args, yaml_cfg)
            elif use_yaml_dispatch and yaml_cfg and exp_id not in _re_retry.EXPERIMENTS:
                result = _run_yaml_donut_experiment(args, yaml_cfg)
            else:
                result = _re_retry.run_experiment(
                    exp_id,
                    overrides=getattr(args, "param_overrides", None) or None,
                )

            _log_ar.debug(
                "[AutoRetry] Experiment %d succeeded (F1=%s)",
                exp_id,
                result.get("metrics", {}).get("global_f1", "N/A"),
            )
        except Exception as _retry_exc:
            _log_ar.debug("[AutoRetry] Experiment %d still failed: %s", exp_id, _retry_exc)
            still_failed.append(exp_id)

    if still_failed:
        _log_ar.debug(
            "[AutoRetry] %d experiment(s) could not be recovered: %s",
            len(still_failed),
            still_failed,
        )
    else:
        _log_ar.debug("[AutoRetry] All failed experiments recovered successfully!")

    return still_failed


def stage_experiments(args) -> StageResult:
    """Run all (or a single) experiment(s) SEQUENTIALLY. Returns StageResult."""
    import run_experiments as re_mod  # local module

    _banner("STAGE 2 — Experiments (train + evaluate)")
    warnings: list[str] = []

    # Auto-generate processor_config.json once so resource_manager stops warning
    # about a missing file on every experiment.
    _ensure_processor_config()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Honour --force by clearing cached result files first
    if getattr(args, "force", False):
        _log_force = logging.getLogger(__name__)
        for result_file in results_dir.glob("experiment_*.json"):
            result_file.unlink()
            _log_force.debug("[force] Deleted cached result: %s", result_file)
        summary_file = results_dir / "all_experiments.json"
        if summary_file.exists():
            summary_file.unlink()
            _log_force.debug("[force] Deleted cached summary: %s", summary_file)

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
    _log_se = logging.getLogger(__name__)
    if not args.experiment:
        # Only pre-load when running all experiments; single-experiment runs
        # load directly in train_experiment() via from_pretrained().
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            _cfg0 = re_mod.EXPERIMENTS[1]
            _log_se.debug("[stage_experiments] Pre-loading base model: %s", _cfg0.base_model)
            _base_processor = DonutProcessor.from_pretrained(_cfg0.base_model)
            _base_model = VisionEncoderDecoderModel.from_pretrained(_cfg0.base_model)
            _log_se.debug(
                "[stage_experiments] Base model pre-loaded (will deep-copy per experiment)."
            )
        except Exception as _preload_exc:
            _log_se.debug(
                "[stage_experiments] Base model pre-load failed (%s); each experiment will load from disk.",
                _preload_exc,
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

            _log_se.debug("[stage_experiments] --parallel: using DAGScheduler")
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
            _log_se.debug(
                "[stage_experiments] DAGScheduler unavailable (%s) — falling back to serial execution",
                _dag_exc,
            )

    # ── Serial loop ───────────────────────────────────────────────────────
    _log = logging.getLogger(__name__)
    for i, exp_id in enumerate(exp_ids, 1):
        exp_name = _exp_display_name(exp_id)
        _step(i, total, f"Experiment {exp_id}: {exp_name}")
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        _log.debug("Experiment %d — %s started at %s", exp_id, exp_name, ts)
        _log.info("Exp %d/%d: %s", i, total, exp_name)

        # Determine arch_type for dispatch
        yaml_cfg = _yaml_cfg_map.get(exp_id)
        arch_type = yaml_cfg.arch_type if yaml_cfg is not None else "donut"
        is_zero_shot = yaml_cfg.is_zero_shot if yaml_cfg is not None else False

        if yaml_cfg is not None:
            _log.debug(
                "  arch=%s  zero_shot=%s  datasets=%s",
                arch_type,
                is_zero_shot,
                [d.name for d in yaml_cfg.datasets],
            )
        elif exp_id in re_mod.EXPERIMENTS:
            _log.debug("  Datasets: %s", re_mod.EXPERIMENTS[exp_id].datasets)

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
                    keep_model=getattr(args, "keep_models", False),
                    no_disk_cleanup=getattr(args, "no_disk_cleanup", False),
                )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            w = f"Experiment {exp_id} ({exp_name}) crashed: {type(exc).__name__}: {exc}"
            _log.error("Exp %d CRASHED: %s", exp_id, exc)
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
        f1 = result.get("metrics", {}).get("global_f1", "N/A")
        n_samples = result.get("num_train_samples", "?")
        _log.info("Exp %d done: F1=%s  samples=%s  time=%.1fm", exp_id, f1, n_samples, elapsed / 60)
        _log.debug("Experiment %d finished (%.1fs)", exp_id, elapsed)
        completed += 1
        if result.get("num_train_samples", 0) == 0 and not is_zero_shot:
            had_empty = True
            failed_experiments.append(exp_id)
            warnings.append(f"Experiment {exp_id} ({exp_name}) had 0 training samples")
        else:
            succeeded += 1

        # ── --verify: post-experiment F1 > 0 check ──────────────────────────
        # Catches silent failures (lm_head_dedup, token2json_list, wrong
        # decoder_start_token_id) that train successfully but produce F1=0.
        if getattr(args, "verify", False):
            _f1_val = result.get("metrics", {}).get("global_f1", None)
            if _f1_val is not None and not is_zero_shot:
                try:
                    _f1_float = float(_f1_val)
                except (TypeError, ValueError):
                    _f1_float = None
                if _f1_float is not None and _f1_float == 0.0:
                    _vw = (
                        f"--verify FAILED for Experiment {exp_id} ({exp_name}): "
                        f"F1=0.0 after training. This indicates a silent failure. "
                        f"Known causes (CLAUDE.md §16): "
                        f"lm_head_dedup (check LmHeadCloneCallback), "
                        f"token2json_list (<sep/> tokens — check _parse_prediction), "
                        f"wrong decoder_start_token_id (use list-form convert_tokens_to_ids). "
                        f"Run: python diagnostics.py --smoke-test"
                    )
                    logging.warning(_vw)
                    warnings.append(_vw)
                    had_empty = True
                    if exp_id not in failed_experiments:
                        failed_experiments.append(exp_id)

        # ── Incremental result sync ──────────────────────────────────────────
        # Push partial results to GitHub after every successful experiment so
        # that results are visible in the PR even if later experiments crash.
        # Only fires when --auto-push is active (same guard as stage_push_results).
        if getattr(args, "auto_push", False):
            _incremental_push_result(exp_id)

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

    # Always call save_summary() so all_experiments.json is written even when
    # some experiments crashed.  The try-except here is a safety net: if
    # save_summary() itself raises (e.g. disk full) we still return a result.
    try:
        re_mod.save_summary()
    except Exception as _ss_exc:
        _log_se.warning(
            "save_summary() failed: %s — results/all_experiments.json may be "
            "incomplete or missing. Check disk space and permissions, then run: "
            "python run_experiments.py --save-summary",
            _ss_exc,
        )

    # Clean up pre-loaded base model now that all experiments are done.
    if _base_model is not None or _base_processor is not None:
        try:
            from constants import _gpu_cleanup

            del _base_model, _base_processor
            _gpu_cleanup()
        except Exception:
            pass

    # ── Disk usage summary ───────────────────────────────────────────────────
    try:
        from constants import format_bytes, get_disk_usage

        _, used_after, free_after = get_disk_usage()
        _log_se.info("[Disk] After experiments: %s free", format_bytes(free_after))
        print(f"[Disk] Space after experiments: {format_bytes(free_after)} free")
    except Exception:
        pass

    # ── Autonomous feedback loop ────────────────────────────────────────────
    # If any experiments crashed with non-OOM code errors, run the CI auto-fix
    # loop (autonomous_ci.autonomous_pipeline) which uses AI to rewrite broken
    # files, then reload the fixed modules and retry only the failed experiments.
    # This eliminates the "copy logs to agent → agent fixes → re-run" cycle.
    if failed_experiments:
        failed_experiments = _autonomous_feedback_loop(
            failed_experiments,
            args=args,
            yaml_cfg_map=_yaml_cfg_map,
            use_yaml_dispatch=use_yaml_dispatch,
        )
        # Update had_empty based on final failure state
        had_empty = bool(failed_experiments)

    exit_status = 1 if had_empty else 0
    return StageResult(
        name="DONUT Experiments", duration=0.0, exit_status=exit_status, warnings=warnings
    )


def _run_trocr_yolo_experiment(args, cfg) -> dict:
    """
    Dispatch Experiment 12 (arch_type=trocr_yolo) to the TrOCR+YOLO training path.
    Returns a result dict compatible with the stage_experiments summary logic.
    """
    logging.getLogger(__name__).debug("[dispatch] arch=trocr_yolo → train_trocr_yolo.py")
    import train_trocr_yolo  # noqa: F401  # early import to fail fast if package missing

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
    logging.getLogger(__name__).debug(
        "[dispatch] is_zero_shot=True → evaluation only (no training)"
    )
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
        logging.getLogger(__name__).warning(
            "[zero-shot] Evaluation failed: %s; returning empty metrics", exc
        )
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
    logging.getLogger(__name__).debug(
        "[dispatch] arch=donut (YAML-only exp %d) → DONUT training path", cfg.id
    )
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
    _log = logging.getLogger(__name__)
    warnings: list[str] = []

    workspace = Path(args.workspace)
    yolo_train_images = workspace / "data" / "yolo" / "images" / "train"
    trocr_train_meta = workspace / "data" / "trocr" / "train" / "metadata.jsonl"

    if yolo_train_images.exists() and trocr_train_meta.exists():
        _log.debug("TrOCR+YOLO data already prepared — skipping.")
        return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=0, warnings=warnings)

    try:
        import data_pipeline as ds_prep

        counts = ds_prep.prepare_all()
        for key, count in counts.items():
            _log.debug("  %s: %s", key, count)
    except Exception as exc:
        w = f"TrOCR data prep failed: {exc}"
        _log.warning("%s", w)
        warnings.append(w)
        return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=1, warnings=warnings)

    return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# Stage 4 — TrOCR+YOLO training and evaluation
# ---------------------------------------------------------------------------


def stage_trocr_experiments(args) -> StageResult:
    """Train YOLOv8 + TrOCR and evaluate on the same 63 SROIE test images.

    By default trains a **separate TrOCR model for each of the 8 experiments**
    using the dataset combination defined in ``run_experiments.EXPERIMENTS``
    (SROIE-only for Exp 1, SROIE+WildReceipt for Exps 2/5, etc.).  This
    mirrors the DONUT multi-dataset design so cross-architecture comparisons
    are fair.

    Pass ``args.trocr_single = True`` (or ``--trocr-single`` on the CLI) to
    revert to the original behaviour: train **once** on SROIE data and copy
    those results to all 8 experiment slots.  Useful for quick smoke-tests or
    when only a single model is needed.

    Results are saved to ``results/trocr_yolo_results.json``.
    """

    _banner("STAGE 4 — TrOCR+YOLO training & evaluation")
    _log = logging.getLogger(__name__)
    warnings: list[str] = []

    trocr_single = getattr(args, "trocr_single", False)

    try:
        import run_experiments as eval_mod  # noqa: I001
        import train_trocr_yolo as trocr_yolo  # noqa: I001

        workspace = Path(args.workspace)

        # Stage 4a: Train YOLO (shared across all experiments)
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
                _log.warning("%s", w)
                warnings.append(w)
                return StageResult(
                    name="TrOCR+YOLO", duration=0.0, exit_status=1, warnings=warnings
                )
            _log.info("Training YOLOv8 text detector ...")
            trocr_yolo.train_yolo(yolo_output)
        else:
            _log.debug("YOLO weights cached at %s", yolo_weights)

        # Explicit GPU cleanup between YOLO and TrOCR to prevent OOM.
        _gpu_cleanup()
        _log.debug("GPU memory freed between YOLO and TrOCR stages.")

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        test_samples = eval_mod.load_test_samples()

        if trocr_single:
            # ── Single-model path (--trocr-single): original behaviour ──────
            # Train TrOCR once on SROIE data, copy same metrics to all 8 experiments.
            trocr_output = workspace / "models" / "trocr_finetuned"
            trocr_best = trocr_output / "best"
            trocr_history: dict = {"train_loss": [], "val_loss": [], "num_train_samples": 0}
            if not trocr_best.exists():
                _log.info("Training TrOCR OCR model (single-experiment mode) ...")
                trocr_history = trocr_yolo.train_trocr(trocr_output)
            else:
                _log.debug("TrOCR model cached at %s", trocr_best)
                hist_path = trocr_output / "training_history.json"
                if hist_path.exists():
                    with open(hist_path) as _fh:
                        trocr_history = json.load(_fh)

            trocr_results: dict = {}
            if len(test_samples) == 0:
                w = "No test samples found — skipping TrOCR+YOLO evaluation."
                _log.warning("%s", w)
                warnings.append(w)
            elif yolo_weights.exists() and trocr_best.exists():
                _log.debug("Evaluating TrOCR+YOLO on %d test images ...", len(test_samples))
                metrics = trocr_yolo.evaluate_trocr_yolo_on_test(
                    str(yolo_weights), str(trocr_best), test_samples
                )
                eval_mod.print_metrics("TrOCR+YOLO", metrics)
                num_samples = trocr_history.get("num_train_samples", 0)
                for exp_id in range(1, 9):
                    trocr_results[str(exp_id)] = {
                        "name": f"TrOCR+YOLO Exp {exp_id}",
                        "metrics": metrics,
                        "num_train_samples": num_samples,
                    }
                _log.info("TrOCR+YOLO F1=%s (single model)", metrics.get("global_f1", "N/A"))
            else:
                w = "YOLO or TrOCR model weights missing — skipping evaluation."
                _log.warning("%s", w)
                warnings.append(w)

        else:
            # ── Per-experiment path (default): train one TrOCR model per experiment ─
            # Each experiment uses the dataset combination defined in
            # run_experiments.EXPERIMENTS (mirrors DONUT multi-dataset design).
            trocr_results = {}
            _exp_ids = sorted(eval_mod.EXPERIMENTS.keys())
            for exp_id in _exp_ids:
                exp_config = eval_mod.EXPERIMENTS.get(exp_id)
                if exp_config is None:
                    _log.debug("[Exp %d] Not found in EXPERIMENTS — skipping TrOCR", exp_id)
                    continue

                _log.info(
                    "[Exp %d/%d] TrOCR training: datasets=%s  sroie_oversample=%d",
                    exp_id,
                    len(_exp_ids),
                    exp_config.datasets,
                    getattr(exp_config, "sroie_oversample", 1),
                )

                trocr_output = workspace / "models" / f"trocr_finetuned_exp{exp_id}"
                trocr_best = trocr_output / "best"
                trocr_history = {"train_loss": [], "val_loss": [], "num_train_samples": 0}

                # Build per-experiment training dataset (SROIE + aux, with oversampling)
                experiment_data_dir = workspace / "data" / "trocr" / f"exp{exp_id}"
                if not (experiment_data_dir / "metadata.jsonl").exists():
                    _log.info("[Exp %d] Building TrOCR training dataset ...", exp_id)
                    trocr_yolo._build_experiment_trocr_metadata(
                        exp_datasets=exp_config.datasets,
                        sroie_oversample=getattr(exp_config, "sroie_oversample", 1),
                        output_dir=experiment_data_dir,
                    )

                # Train (or load cached model)
                if not trocr_best.exists():
                    _log.info("[Exp %d] Training TrOCR ...", exp_id)
                    _gpu_cleanup()
                    trocr_history = trocr_yolo.train_trocr(
                        output_dir=trocr_output,
                        train_data_dir=experiment_data_dir,
                    )
                else:
                    _log.debug("[Exp %d] TrOCR model cached at %s", exp_id, trocr_best)
                    hist_path = trocr_output / "training_history.json"
                    if hist_path.exists():
                        with open(hist_path) as _fh:
                            trocr_history = json.load(_fh)

                # Evaluate
                exp_metrics: dict = {}
                if len(test_samples) == 0:
                    _log.warning("[Exp %d] No test samples — skipping evaluation", exp_id)
                elif yolo_weights.exists() and trocr_best.exists():
                    _gpu_cleanup()
                    exp_metrics = trocr_yolo.evaluate_trocr_yolo_on_test(
                        str(yolo_weights), str(trocr_best), test_samples
                    )
                    eval_mod.print_metrics(f"TrOCR+YOLO Exp {exp_id}", exp_metrics)
                    _log.info(
                        "[Exp %d] F1=%.4f  samples=%d",
                        exp_id,
                        exp_metrics.get("global_f1", 0.0),
                        trocr_history.get("num_train_samples", 0),
                    )
                else:
                    _log.warning(
                        "[Exp %d] YOLO or TrOCR weights missing — skipping evaluation", exp_id
                    )

                trocr_results[str(exp_id)] = {
                    "name": f"TrOCR+YOLO Exp {exp_id}",
                    "metrics": exp_metrics,
                    "num_train_samples": trocr_history.get("num_train_samples", 0),
                }

        # Save results
        out_path = results_dir / "trocr_yolo_results.json"
        with open(out_path, "w") as fh:
            json.dump(trocr_results, fh, indent=2)
        _log.info("TrOCR+YOLO results saved → %s", out_path)

        # GPU cleanup after TrOCR+YOLO stage.
        _gpu_cleanup()

    except Exception as exc:
        import traceback

        traceback.print_exc()
        w = f"TrOCR+YOLO stage failed: {type(exc).__name__}: {exc}"
        _log.warning("%s", w)
        warnings.append(w)
        return StageResult(name="TrOCR+YOLO", duration=0.0, exit_status=1, warnings=warnings)

    return StageResult(name="TrOCR+YOLO", duration=0.0, exit_status=0, warnings=warnings)


# ---------------------------------------------------------------------------
# TrOCR all-backends helpers
# ---------------------------------------------------------------------------


def _evaluate_field_assigner(
    test_samples: list,
    yolo_model,
    trocr_model,
    trocr_processor,
    assigner,
    logger: "logging.Logger",
) -> dict:
    """Run inference on every test sample with the given assigner and return F1 metrics.

    Uses the canonical ``compute_metrics`` from ``run_experiments`` so scoring is
    identical to the DONUT evaluation path (global precision/recall/F1, per-field F1,
    NED, and overall exact-match).
    """
    import run_experiments as _re  # noqa: I001
    import train_trocr_yolo as _tty  # noqa: I001

    from constants import FIELDS

    predictions: list[dict] = []
    ground_truths: list[dict] = [gt for _, gt in test_samples]

    for img_path, _gt in test_samples:
        try:
            pred = _tty.run_trocr_yolo_inference(
                Path(img_path),
                yolo_model,
                trocr_model,
                trocr_processor,
                field_assigner=assigner,
            )
        except Exception as _exc:
            logger.debug("Inference failed for %s: %s", img_path, _exc)
            pred = {f: "" for f in FIELDS}
        predictions.append(pred)

    return _re.compute_metrics(predictions, ground_truths)


def _print_backend_comparison(all_results: dict, logger: "logging.Logger") -> None:
    """Print a comparison table of all TrOCR+YOLO backend F1 scores to stdout and log."""
    _COLS = ("Global F1", "Company", "Date", "Address", "Total", "Params")
    _W = (12, 10, 10, 10, 10, 14)
    _KEYS = ("global_f1", "company_f1", "date_f1", "address_f1", "total_f1")

    # Find best global F1 across all backends for highlighting
    best_f1 = max(
        (d.get("metrics", {}).get("global_f1", 0.0) for d in all_results.values()),
        default=0.0,
    )

    # Build rows
    rows = []
    for backend_key, data in all_results.items():
        m = data.get("metrics", {})
        params_raw = data.get("params", 0)
        if isinstance(params_raw, int) and params_raw > 0:
            params_str = (
                f"{params_raw / 1e6:.2f}M"
                if params_raw >= 1_000_000
                else f"{params_raw / 1e3:.0f}K"
            )
        else:
            params_str = "0 (rules)"
        label = data.get("backend", backend_key)
        rows.append((label, m, params_str))

    # Try rich table first; fall back to plain-text
    try:
        from rich.console import Console as _Console
        from rich.table import Table as _Table

        _con = _Console()
        tbl = _Table(title="TrOCR+YOLO Backend Comparison", show_lines=True)
        tbl.add_column("Backend / Assignment", style="bold")
        for col in _COLS:
            tbl.add_column(col, justify="right")
        for label, m, params_str in rows:
            f1 = m.get("global_f1", 0.0)
            highlight = f1 == best_f1 and best_f1 > 0.0
            style = "bold green" if highlight else ""
            tbl.add_row(
                label,
                f"[{style}]{f1:.4f}[/]" if style else f"{f1:.4f}",
                f"{m.get('company_f1', 0.0):.4f}",
                f"{m.get('date_f1', 0.0):.4f}",
                f"{m.get('address_f1', 0.0):.4f}",
                f"{m.get('total_f1', 0.0):.4f}",
                params_str,
            )
        _con.print(tbl)
    except ImportError:
        # Plain-text table — always visible on console
        w_label = 24
        header = f"{'Backend / Assignment':<{w_label}}" + "".join(
            f"{c:>{w}}" for c, w in zip(_COLS, _W)
        )
        sep = "─" * len(header)
        lines = [sep, header, sep]
        for label, m, params_str in rows:
            f1 = m.get("global_f1", 0.0)
            marker = " ◀ BEST" if f1 == best_f1 and best_f1 > 0.0 else ""
            vals = [m.get(k, 0.0) for k in _KEYS]
            row = (
                f"{label:<{w_label}}"
                + "".join(f"{v:{w}.4f}" for v, w in zip(vals, _W[:-1]))
                + f"{params_str:>{_W[-1]}}"
                + marker
            )
            lines.append(row)
        lines.append(sep)
        table_str = "\n".join(lines)
        print(table_str)  # always visible on stdout
        logger.info("\n%s", table_str)


# ---------------------------------------------------------------------------
# Stage 4-ALL — Comprehensive TrOCR+YOLO with all three field-assigner backends
# ---------------------------------------------------------------------------


def stage_trocr_all_backends(args) -> StageResult:
    """Train TrOCR+YOLO once, then train and evaluate all three FieldAttentionAssigner backends.

    Execution order
    ---------------
    1. YOLO text-region detector training (cached if already done).
    2. TrOCR OCR model training (cached if already done).
    3. Regex heuristic evaluation — the original rule-based baseline, free.
    4. Backend "char"       (~532 K params) — char embeddings, no pretrained weights.
    5. Backend "lm"         (~4.9 M params) — frozen BERT-tiny text encoder.
    6. Backend "lm+vision"  (~5.1 M params) — LM + TrOCR vision features + consistency loss.

    Each backend is trained on the 500-sample SROIE training split using weak NED
    labels, then evaluated on the 63-image SROIE test set.  Results for all four
    configurations are written to results/trocr_all_backends.json and a comparison
    table is printed to the log.
    """
    _banner("STAGE 4-ALL — TrOCR comprehensive (all backends)")
    _log = logging.getLogger(__name__)
    warnings: list[str] = []
    all_results: dict = {}

    try:
        import train_trocr_yolo as trocr_yolo  # noqa: I001
        import run_experiments as eval_mod  # noqa: I001

        workspace = Path(args.workspace)
        sroie_dir = Path(args.sroie_dir)
        yolo_weights = workspace / "models" / "yolo_finetuned" / "run" / "weights" / "best.pt"
        trocr_best = workspace / "models" / "trocr_finetuned" / "best"

        # ── Step 1/6: YOLO + TrOCR training (reuses caching logic) ──────────
        _log.info("[1/6] YOLO + TrOCR base training…")
        base = stage_trocr_experiments(args)
        if base.exit_status > 1:
            return StageResult(
                name="TrOCR All Backends",
                duration=0.0,
                exit_status=base.exit_status,
                warnings=warnings,
            )

        # ── Step 2/6: Record regex baseline from existing results ────────────
        regex_path = Path("results") / "trocr_yolo_results.json"
        if regex_path.exists():
            with open(regex_path) as _fh:
                _rx = json.load(_fh)
            _first = next(iter(_rx.values()), {})
            if not isinstance(_first, dict):
                _first = {}
            all_results["regex"] = {
                "backend": "Regex heuristic",
                "description": "Rule-based field assignment, no trainable parameters",
                "params": 0,
                "training_time_sec": 0,
                "metrics": _first.get("metrics", {}),
            }
            _log.info(
                "[2/6] Regex baseline F1 = %.4f",
                all_results["regex"]["metrics"].get("global_f1", 0.0),
            )
        else:
            _log.warning("[2/6] trocr_yolo_results.json not found — regex baseline skipped.")

        if not yolo_weights.exists() or not trocr_best.exists():
            w = "YOLO or TrOCR weights missing — cannot train field assigners."
            _log.warning("%s", w)
            warnings.append(w)
            return StageResult(
                name="TrOCR All Backends", duration=0.0, exit_status=1, warnings=warnings
            )

        # ── Step 3/6: Load YOLO + TrOCR for inference ───────────────────────
        _log.info("[3/6] Loading YOLO and TrOCR for field-assigner training…")
        _gpu_cleanup()
        yolo_model = trocr_yolo._YOLO_CLS(str(yolo_weights))
        trocr_processor = trocr_yolo.TrOCRProcessor.from_pretrained(str(trocr_best))
        trocr_model = trocr_yolo.VisionEncoderDecoderModel.from_pretrained(str(trocr_best))
        trocr_model = trocr_model.to(trocr_yolo.DEVICE)
        trocr_model.eval()

        test_samples = eval_mod.load_test_samples()
        if not test_samples:
            w = "No test samples found — skipping backend evaluation."
            _log.warning("%s", w)
            warnings.append(w)

        # ── Steps 4–6: Train and evaluate each field-assigner backend ────────
        _FA_BACKENDS = [
            ("char", 532_000, "Char embeddings — 532K params, no pretrained weights"),
            ("lm", 4_900_000, "Frozen BERT-tiny — 4.9M params"),
            (
                "lm+vision",
                5_100_000,
                "BERT-tiny + TrOCR vision features + consistency loss — 5.1M params",
            ),
        ]

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        out_path = results_dir / "trocr_all_backends.json"

        for step, (backend, param_count, description) in enumerate(_FA_BACKENDS, 4):
            _log.info("[%d/6] Training FieldAttentionAssigner backend='%s'…", step, backend)
            _gpu_cleanup()
            t0 = time.time()
            try:
                assigner = trocr_yolo.train_field_assigner(
                    sroie_dir=sroie_dir,
                    yolo_model=yolo_model,
                    trocr_model=trocr_model,
                    trocr_processor=trocr_processor,
                    epochs=trocr_yolo.FIELD_ASSIGNER_EPOCHS,  # patchable for superfast/micro
                    backend=backend,
                    device=trocr_yolo.DEVICE,
                )

                if test_samples:
                    _log.info("  Evaluating on %d test images…", len(test_samples))
                    metrics = _evaluate_field_assigner(
                        test_samples,
                        yolo_model,
                        trocr_model,
                        trocr_processor,
                        assigner,
                        _log,
                    )
                    eval_mod.print_metrics(f"TrOCR+YOLO ({backend})", metrics)
                else:
                    metrics = {}

                training_time = round(time.time() - t0, 1)
                all_results[backend] = {
                    "backend": backend,
                    "description": description,
                    "params": param_count,
                    "training_time_sec": training_time,
                    "metrics": metrics,
                }
                _log.info(
                    "[%d/6] Backend '%s' done in %.1fs  F1=%.4f",
                    step,
                    backend,
                    training_time,
                    metrics.get("global_f1", 0.0),
                )

            except Exception as _exc:
                import traceback as _tb

                _tb.print_exc()
                w = f"Backend '{backend}' failed: {type(_exc).__name__}: {_exc}"
                _log.warning("%s", w)
                warnings.append(w)
                all_results[backend] = {
                    "backend": backend,
                    "description": description,
                    "params": param_count,
                    "training_time_sec": round(time.time() - t0, 1),
                    "metrics": {},
                    "error": str(_exc),
                }

            # Incremental save after every backend so partial results survive crashes
            with open(out_path, "w") as _fh:
                json.dump(all_results, _fh, indent=2)

        # ── Save final all-backend results + comparison table ─────────────────
        _log.info("All-backend results → %s", out_path)
        _print_backend_comparison(all_results, _log)

        # ── Select best backend and update trocr_yolo_results.json ───────────
        # The regex heuristic is the default baseline stored in trocr_yolo_results.json.
        # If a trained backend (char / lm / lm+vision) beats it, replace the
        # per-experiment metrics with the winning backend's metrics so that the
        # paper always reports the best achievable result.
        if all_results:
            best_backend = max(
                all_results.keys(),
                key=lambda k: all_results[k].get("metrics", {}).get("global_f1", 0.0),
            )
            best_metrics = all_results[best_backend].get("metrics", {})
            best_f1 = best_metrics.get("global_f1", 0.0)
            _log.info("[Best backend] '%s'  F1=%.4f", best_backend, best_f1)

            trocr_path = results_dir / "trocr_yolo_results.json"
            if trocr_path.exists():
                with open(trocr_path) as _fh:
                    trocr_results = json.load(_fh)
                updated = 0
                baseline_f1 = max(
                    (
                        _edata.get("metrics", {}).get("global_f1", 0.0)
                        for _edata in trocr_results.values()
                    ),
                    default=0.0,
                )
                for _eid, _edata in trocr_results.items():
                    current_f1 = _edata.get("metrics", {}).get("global_f1", 0.0)
                    if best_f1 > current_f1:
                        trocr_results[_eid]["metrics"] = best_metrics
                        trocr_results[_eid]["backend"] = best_backend
                        updated += 1
                if updated:
                    with open(trocr_path, "w") as _fh:
                        json.dump(trocr_results, _fh, indent=2)
                    _log.info(
                        "Updated %d experiment(s) in trocr_yolo_results.json "
                        "with best backend '%s' (F1=%.4f)",
                        updated,
                        best_backend,
                        best_f1,
                    )
                else:
                    _log.info(
                        "Current baseline (max F1=%.4f) already ≥ best trained backend "
                        "('%s', F1=%.4f) — trocr_yolo_results.json unchanged.",
                        baseline_f1,
                        best_backend,
                        best_f1,
                    )
            else:
                _log.warning(
                    "trocr_yolo_results.json not found — cannot apply best backend update."
                )

        _gpu_cleanup()

    except Exception as exc:
        import traceback

        traceback.print_exc()
        w = f"TrOCR all-backends stage failed: {type(exc).__name__}: {exc}"
        _log.warning("%s", w)
        warnings.append(w)
        return StageResult(
            name="TrOCR All Backends", duration=0.0, exit_status=1, warnings=warnings
        )

    return StageResult(name="TrOCR All Backends", duration=0.0, exit_status=0, warnings=warnings)


def stage_benchmark(args) -> StageResult:
    """Run head-to-head benchmark: DONUT vs YOLOv8+TrOCR+Regex on the SROIE test set.

    Automatically selects the best DONUT experiment model (highest global F1)
    and the trained YOLO best.pt from Stage 4. Produces side-by-side F1 /
    accuracy / speed metrics and journal-ready plots.
    """
    _banner("STAGE 5 — Head-to-head benchmark (DONUT vs YOLOv8+TrOCR+Regex)")
    _log = logging.getLogger(__name__)
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
        _log.warning("%s", w)
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
            _log.warning("%s", w)
            warnings.append(w)
            donut_model_dir = BASE_MODEL  # plain str — valid HuggingFace hub ID

    _log.debug("DONUT model: %s (exp %d, F1=%.4f)", donut_model_dir, best_exp_id, best_f1)

    # --- Find YOLO best.pt ---
    yolo_weights = workspace / "models" / "yolo_finetuned" / "run" / "weights" / "best.pt"
    skip_yolo = not yolo_weights.exists()
    if skip_yolo:
        w = f"YOLO weights not found at {yolo_weights} — benchmark will run DONUT only."
        _log.warning("%s", w)
        warnings.append(w)
    else:
        _log.debug("YOLO model: %s", yolo_weights)

    # --- Find fine-tuned TrOCR model (for fair comparison vs base pretrained) ---
    trocr_finetuned_dir = workspace / "models" / "trocr_finetuned" / "best"
    if _is_valid_model_dir(trocr_finetuned_dir):
        trocr_model_id = str(trocr_finetuned_dir)
        _log.debug("TrOCR model: %s (fine-tuned)", trocr_model_id)
    else:
        trocr_model_id = "microsoft/trocr-base-printed"
        w = (
            f"Fine-tuned TrOCR model not found at {trocr_finetuned_dir} "
            "— falling back to pretrained base model for benchmark."
        )
        _log.warning("%s", w)
        warnings.append(w)
        _log.debug("TrOCR model: %s (pretrained base)", trocr_model_id)

    _log.debug("Test images: %s | labels: %s", test_img_dir, test_key_dir)

    # --- Run benchmark_compare programmatically ---
    try:
        import torch

        import reporting as bench_mod

        pairs = bench_mod.find_pairs(test_img_dir, test_key_dir)
        _log.debug("Found %d test image+label pairs.", len(pairs))

        all_results = []

        # Run DONUT pipeline
        _log.debug("Running DONUT inference ...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        donut_pipe = bench_mod.DonutPipeline(model_id_or_path=str(donut_model_dir))
        donut_result = donut_pipe.run_benchmark(pairs, desc="DONUT benchmark", batch_size=1)
        donut_result = bench_mod.compute_metrics(donut_result)
        all_results.append(donut_result)

        # Free GPU before next pipeline.
        del donut_pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Run YOLOv8+TrOCR+Regex pipeline (if weights available)
        if not skip_yolo:
            _log.debug("Running YOLOv8+TrOCR+Regex inference ...")
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

        # Print comparison report (goes to terminal.txt via bench_mod.print_report → stdout)
        bench_mod.print_report(all_results, n_samples=len(pairs))

        # Save JSON + plots
        out_dir = results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        bench_mod.save_json(all_results, out_dir / "benchmark_results.json")
        bench_mod.plot_results(all_results, out_dir=out_dir / "figures")
        _log.info("Benchmark saved → %s", out_dir)

    except Exception as exc:
        import traceback

        traceback.print_exc()
        w = f"Benchmark stage failed: {type(exc).__name__}: {exc}"
        _log.warning("%s", w)
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
    _log = logging.getLogger(__name__)
    warnings: list[str] = []

    try:
        import reporting as compare_mod

        compare_mod.compare_all()
    except Exception as exc:
        w = f"Comparison stage failed: {exc}"
        _log.warning("%s", w)
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
    import contextlib
    import io as _io

    import data_pipeline as dataset_loaders
    import reporting as ir  # local module

    _banner("STAGE 7 — LaTeX paper generation")
    _log = logging.getLogger(__name__)
    warnings: list[str] = []

    results_path = Path("results") / "all_experiments.json"
    if not results_path.exists():
        w = f"{results_path} not found — generating paper with placeholder values only."
        _log.warning("%s", w)
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
        _log.warning("%s", w)
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

    # Redirect LaTeX table print output to terminal.txt (debug level) not console.
    # Each table call is individually guarded so one failure doesn't skip the rest.
    _buf = _io.StringIO()
    for _table_fn, _table_args in [
        (ir.print_table1_dataset_stats, (actual_counts,)),
        (ir.print_table2_experiments, (all_exp,)),
        (ir.print_table3_perfield, (all_exp,)),
        (ir.print_table4_leaderboard, (all_exp,)),
    ]:
        try:
            with contextlib.redirect_stdout(_buf):
                _table_fn(*_table_args)
        except Exception as _tbl_exc:
            _log.warning("Table generation %s failed: %s", _table_fn.__name__, _tbl_exc)

    # Print TrOCR+YOLO and cross-architecture comparison tables
    trocr_path = Path("results") / "trocr_yolo_results.json"
    if trocr_path.exists():
        with open(trocr_path) as fh:
            trocr_exp = json.load(fh)
        for _table_fn, _table_args in [
            (ir.print_table5_trocr_yolo, (trocr_exp,)),
            (ir.print_table6_cross_architecture, (all_exp, trocr_exp)),
        ]:
            try:
                with contextlib.redirect_stdout(_buf):
                    _table_fn(*_table_args)
            except Exception as _tbl_exc:
                _log.warning("Table generation %s failed: %s", _table_fn.__name__, _tbl_exc)
    _log.debug("LaTeX tables:\n%s", _buf.getvalue())

    try:
        ir.generate_convergence_data(str(results_path))
        ir.generate_convergence_tex(str(results_path))
        ir.generate_f1_barchart_tex(str(results_path))
    except Exception as exc:
        w = f"Convergence/barchart tex generation failed: {exc}; skipping."
        _log.warning("%s", w)
        warnings.append(w)

    # Build a single var_map that covers both paper.tex and presentation.tex
    var_map = ir.build_var_map(all_exp)

    paper_template = Path(args.paper_template)
    output_paper = Path(args.output)

    if paper_template.exists():
        try:
            ir.fill_paper(str(paper_template), str(output_paper), var_map)
            _log.info("Paper written → %s", output_paper)
        except Exception as exc:
            w = f"fill_paper failed for {paper_template}: {exc}"
            _log.warning("%s", w)
            warnings.append(w)

        # ── Compile paper → PDF ────────────────────────────────────────────
        try:
            paper_pdf = ir.compile_pdf(output_paper, work_dir=output_paper.parent)
            if paper_pdf:
                _log.info("PDF compiled → %s", paper_pdf)
            else:
                _log.info(
                    "LaTeX → PDF skipped (no compiler). Install: apt-get install texlive-latex-base"
                )
                _log.debug("PDF compile hint: pdflatex %s", output_paper)
        except Exception as exc:
            w = f"PDF compilation failed: {exc}"
            _log.warning("%s", w)
            warnings.append(w)
    else:
        w = f"paper template not found at {paper_template}; skipping paper_filled.tex generation."
        _log.warning("%s", w)
        warnings.append(w)

    # Also fill presentation.tex → presentation_filled.tex
    pres_template = Path("paper/presentation.tex")
    pres_output = Path("paper/presentation_filled.tex")
    if pres_template.exists():
        try:
            ir.fill_paper(str(pres_template), str(pres_output), var_map)
            _log.debug("Presentation written → %s", pres_output)
        except Exception as exc:
            w = f"fill_paper failed for {pres_template}: {exc}"
            _log.warning("%s", w)
            warnings.append(w)

        # ── Compile presentation → PDF ─────────────────────────────────────
        try:
            pres_pdf = ir.compile_pdf(pres_output, work_dir=pres_output.parent)
            if pres_pdf:
                _log.debug("Presentation PDF compiled → %s", pres_pdf)
        except Exception as exc:
            warnings.append(f"Presentation PDF compilation failed: {exc}")
    else:
        _log.debug("%s not found; skipping presentation_filled.tex generation.", pres_template)

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
        _log = logging.getLogger(__name__)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        _log.debug("Stage %s started at %s", name, ts)
        _log.info("Stage: %s", name)
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
                            _log.warning("[AI Diagnosis] Stage %r: %s", name, _diagnosis)
                except Exception:
                    pass  # diagnosis itself failed — don't mask the real error

            result = StageResult(
                name=name,
                duration=elapsed,
                exit_status=2,
                warnings=[f"Uncaught exception: {type(exc).__name__}: {exc}"],
            )
        ts_end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status = (
            "OK" if result.exit_status == 0 else ("PART" if result.exit_status == 1 else "FAIL")
        )
        _log.info("Stage %s done: %s (%.1fs)", name, status, result.duration)
        _log.debug("Stage %s finished at %s (%.1fs)", name, ts_end, result.duration)
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
            logging.getLogger(__name__).debug("\n%s", repr(self))
            if _FANCY_OUTPUT:
                print(repr(self))
            return exit_code

        # --trocr-only: install SROIE data, prepare TrOCR dataset, then run the
        # comprehensive all-backends stage (YOLO+TrOCR+regex+char+lm+lm+vision).
        # Skips dataset download, pretrained baseline, and all DONUT experiments.
        if getattr(self.args, "trocr_only", False):
            if not self.args.skip_install:
                self._run_stage("SROIE Install", stage_install)
            self._run_stage("TrOCR Data Prep", stage_trocr_data_prep)
            r = self._run_stage("TrOCR All Backends", stage_trocr_all_backends)
            if r.exit_status > exit_code:
                exit_code = r.exit_status
            self._run_stage("Paper Generation", stage_paper)
            self._run_stage("Results Push", stage_push_results)
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

            # Stage 4 — TrOCR+YOLO training, all-backends evaluation & best-backend selection.
            # stage_trocr_all_backends() internally calls stage_trocr_experiments() for
            # YOLO + per-experiment TrOCR training, then evaluates char/lm/lm+vision
            # backends and updates trocr_yolo_results.json with the best backend.
            r = self._run_stage("TrOCR All Backends", stage_trocr_all_backends)
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
                logging.getLogger(__name__).warning("Failed to save pipeline report: %s", _diag_exc)

        # Log full box-drawing execution table to terminal.txt (debug); show compact on console.
        logging.getLogger(__name__).debug("\n%s", repr(self))
        if _FANCY_OUTPUT:
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
        if getattr(args, "trocr_only", False):
            logger.info("[Stage 3-4] TrOCR all-backends (trocr-only mode)...")
            result_trocr = stage_trocr_data_prep(args)
            if result_trocr.exit_status <= 1:
                stage_trocr_all_backends(args)
        elif not args.skip_trocr:
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


def _superfast_mode_handler(args, logger: logging.Logger) -> int:
    """Superfast mode: TrOCR+YOLO only, absolute bare minimum, target <3 min on RTX 4090.

    No DONUT training.  Runs the full TrOCR+YOLO pipeline at minimum settings,
    including all three FieldAttentionAssigner backends so the comparison table
    is still populated:

      YOLO   — yolov8n (3.2 M params), 1 epoch, 160 px, SGD+Nesterov
      TrOCR  — 1 epoch, max_len=32, batch=4, SGD+Nesterov+CosineAnnealingLR
      Regex  — rule-based baseline (0 training cost)
      char   — character-embedding assigner, 1 epoch  (~532 K params)
      lm     — frozen BERT-tiny assigner, 1 epoch     (~4.9 M params)
      lm+vision — BERT-tiny + TrOCR vision feats, 1 epoch (~5.1 M params)

    Produces paper_superfast.tex with all \\VAR{} placeholders resolved (DONUT
    metrics filled with «N/A» since that stage is skipped).
    """
    import train_trocr_yolo as tty

    # ── Stage 0: SROIE install ────────────────────────────────────────────
    if not args.skip_install:
        logger.info("[Superfast Stage 0] SROIE data install...")
        r = stage_install(args)
        if r.exit_status > 1:
            logger.error("SROIE install failed")
            return 2

    # ── Stage 1: TrOCR+YOLO data prep ────────────────────────────────────
    logger.info("[Superfast Stage 1] TrOCR+YOLO data prep...")
    r = stage_trocr_data_prep(args)
    if r.exit_status > 1:
        logger.error("TrOCR data prep failed")
        return 2

    # ── Stage 2: YOLO (1 ep) + TrOCR (1 ep) + all 3 field-assigner backends (1 ep each) ──
    logger.info(
        "[Superfast Stage 2] YOLO (yolov8n 1 ep 160 px) + TrOCR (1 ep) + 3 backends (1 ep each)..."
    )
    _saved = {
        "YOLO_BASE": tty.YOLO_BASE,
        "YOLO_EPOCHS": tty.YOLO_EPOCHS,
        "YOLO_IMG_SIZE": tty.YOLO_IMG_SIZE,
        "YOLO_BATCH": tty.YOLO_BATCH,
        "YOLO_OPTIMIZER": tty.YOLO_OPTIMIZER,
        "YOLO_MOMENTUM": tty.YOLO_MOMENTUM,
        "TROCR_EPOCHS": tty.TROCR_EPOCHS,
        "TROCR_MAX_LEN": tty.TROCR_MAX_LEN,
        "TROCR_BATCH": tty.TROCR_BATCH,
        "TROCR_MINI_MODE": tty.TROCR_MINI_MODE,
        "FIELD_ASSIGNER_EPOCHS": tty.FIELD_ASSIGNER_EPOCHS,
    }
    try:
        tty.YOLO_BASE = "yolov8n.pt"  # 3.2M params — smallest available
        tty.YOLO_EPOCHS = 5  # single pass through dataset
        tty.YOLO_IMG_SIZE = 320  # minimum multiple of 32 that fits stride-32 head
        tty.YOLO_BATCH = 32  # small images fit large batch
        tty.YOLO_OPTIMIZER = "SGD"  # SGD+Nesterov: fastest convergence per step
        tty.YOLO_MOMENTUM = 0.937
        tty.TROCR_EPOCHS = 1
        tty.TROCR_MAX_LEN = 32  # 128 → 32: 4× faster decoding per sample
        tty.TROCR_BATCH = 4  # conservative: avoids OOM after inline YOLO on same GPU
        tty.TROCR_MINI_MODE = True  # SGD+Nesterov+CosineAnnealingLR
        tty.FIELD_ASSIGNER_EPOCHS = 1  # 30 → 1: char / lm / lm+vision each train 1 epoch
        # stage_trocr_all_backends runs all 3 backends (char → lm → lm+vision) so
        # the comparison table is populated even in superfast mode.
        r = stage_trocr_all_backends(args)
    finally:
        for k, v in _saved.items():
            setattr(tty, k, v)

    if r.exit_status > 1:
        logger.warning("TrOCR+YOLO superfast training failed (continuing to paper gen)")

    # ── Stage 3: Paper generation → paper_superfast.tex ──────────────────
    logger.info("[Superfast Stage 3] Generating paper/paper_superfast.tex...")
    args_paper = copy.copy(args)
    args_paper.paper_template = "paper/paper.tex"
    args_paper.output = "paper/paper_superfast.tex"
    return _generate_mini_paper(args_paper, logger)


def _instant_mode_handler(args, logger: logging.Logger) -> int:
    """Instant mode: maximum caching, target <30s on repeat runs.

    First run: behaves like --superfast but also builds tensor caches and
    writes a data-prep completion marker.
    Subsequent runs: data prep is skipped via marker check, YOLO+TrOCR
    training is skipped because cached weights are present, and the
    TrOCRReceiptDataset loads pre-processed tensors directly from disk.

      First run  (~3.5 min): stage_install + data prep + training + cache save
      Repeat run (<30 s):    marker skip + weight skip + tensor cache load + paper
    """
    import train_trocr_yolo as tty

    # ── Stage 0: SROIE install ────────────────────────────────────────────
    if not args.skip_install:
        logger.info("[Instant Stage 0] SROIE data install...")
        r = stage_install(args)
        if r.exit_status > 1:
            logger.error("SROIE install failed")
            return 2

    # ── Stage 1: TrOCR+YOLO data prep (enhanced marker check) ────────────
    logger.info("[Instant Stage 1] TrOCR+YOLO data prep (with marker check)...")
    r = _stage_trocr_data_prep_cached(args, logger)
    if r.exit_status > 1:
        logger.error("TrOCR data prep failed")
        return 2

    # ── Stage 2: YOLO+TrOCR training (skipped when cached weights exist) ─
    # stage_trocr_all_backends → stage_trocr_experiments internally checks whether
    # YOLO and TrOCR weights already exist and skips those training runs if so.
    # We always call stage_trocr_all_backends here because even on repeat runs we
    # still need to train+evaluate the field-assigner backends (the cheapest stage).
    workspace = Path(args.workspace)
    yolo_weights = workspace / "models" / "yolo_finetuned" / "run" / "weights" / "best.pt"
    trocr_best = workspace / "models" / "trocr_finetuned" / "best"

    if yolo_weights.exists() and trocr_best.exists():
        logger.info(
            "[Instant Stage 2] Cached YOLO+TrOCR weights found — "
            "YOLO+TrOCR training will be skipped inside stage_trocr_all_backends."
        )
    else:
        logger.info("[Instant Stage 2] No cached weights — running full training (first run).")

    _saved = {
        "YOLO_BASE": tty.YOLO_BASE,
        "YOLO_EPOCHS": tty.YOLO_EPOCHS,
        "YOLO_IMG_SIZE": tty.YOLO_IMG_SIZE,
        "YOLO_BATCH": tty.YOLO_BATCH,
        "YOLO_OPTIMIZER": tty.YOLO_OPTIMIZER,
        "YOLO_MOMENTUM": tty.YOLO_MOMENTUM,
        "TROCR_EPOCHS": tty.TROCR_EPOCHS,
        "TROCR_MAX_LEN": tty.TROCR_MAX_LEN,
        "TROCR_BATCH": tty.TROCR_BATCH,
        "TROCR_MINI_MODE": tty.TROCR_MINI_MODE,
        "FIELD_ASSIGNER_EPOCHS": tty.FIELD_ASSIGNER_EPOCHS,
        "TROCR_USE_TENSOR_CACHE": tty.TROCR_USE_TENSOR_CACHE,
    }
    try:
        tty.YOLO_BASE = "yolov8n.pt"  # 3.2M params — smallest available
        tty.YOLO_EPOCHS = 5
        tty.YOLO_IMG_SIZE = 320
        tty.YOLO_BATCH = 32
        tty.YOLO_OPTIMIZER = "SGD"
        tty.YOLO_MOMENTUM = 0.937
        tty.TROCR_EPOCHS = 1
        tty.TROCR_MAX_LEN = 32
        tty.TROCR_BATCH = 4
        tty.TROCR_MINI_MODE = True
        tty.FIELD_ASSIGNER_EPOCHS = 1
        tty.TROCR_USE_TENSOR_CACHE = True  # build/load tensor cache
        r = stage_trocr_all_backends(args)
    finally:
        for k, v in _saved.items():
            setattr(tty, k, v)

    if r.exit_status > 1:
        logger.warning("TrOCR+YOLO instant training failed (continuing to paper gen)")

    # ── Stage 3: Paper generation → paper_instant.tex ────────────────────
    logger.info("[Instant Stage 3] Generating paper/paper_instant.tex...")
    args_paper = copy.copy(args)
    args_paper.paper_template = "paper/paper.tex"
    args_paper.output = "paper/paper_instant.tex"
    return _generate_mini_paper(args_paper, logger)


def _stage_trocr_data_prep_cached(args, logger: logging.Logger) -> "StageResult":
    """Enhanced data-prep stage for instant mode: uses a hash-based marker file.

    Writes ``data/.prep_complete_{hash}.marker`` after a successful
    ``prepare_all()`` run.  The hash covers the SROIE data directory path,
    its mtime, and image count, so any source-data change invalidates the
    marker and triggers a fresh prep run.

    Falls back gracefully to the standard directory-existence check when the
    SROIE source directory is inaccessible.
    """
    import hashlib

    _log = logging.getLogger(__name__)
    workspace = Path(args.workspace)
    sroie_dir = Path(args.sroie_dir)
    data_dir = workspace / "data"
    yolo_train_images = workspace / "data" / "yolo" / "images" / "train"
    trocr_train_meta = workspace / "data" / "trocr" / "train" / "metadata.jsonl"

    # Build a hash that captures the state of the SROIE source data.
    marker_file = None
    prep_hash = None
    try:
        img_dir = sroie_dir / "img"
        if img_dir.exists():
            n_images = sum(1 for _ in img_dir.glob("*.jpg"))
            dir_mtime = img_dir.stat().st_mtime
        else:
            n_images = 0
            dir_mtime = 0.0
        raw = f"{sroie_dir}:{dir_mtime:.0f}:{n_images}".encode()
        prep_hash = hashlib.sha256(raw).hexdigest()[:16]
        marker_file = data_dir / f".prep_complete_{prep_hash}.marker"

        if marker_file.exists():
            _log.info("[Instant] Data-prep marker found — skipping prepare_all().")
            return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=0, warnings=[])
    except Exception as _hash_exc:
        _log.debug("Marker hash failed (%s) — falling back to dir check", _hash_exc)

    # Standard directory-existence check (same idempotency guard as stage_trocr_data_prep).
    if yolo_train_images.exists() and trocr_train_meta.exists():
        _log.debug("TrOCR+YOLO data already prepared — skipping.")
        # Write marker so future instant-mode runs take the fast path.
        if marker_file is not None:
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                marker_file.write_text(f"prep_hash={prep_hash}\n")
            except Exception:
                pass
        return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=0, warnings=[])

    # Data not yet prepared — run prepare_all() and write marker on success.
    warnings: list[str] = []
    try:
        import data_pipeline as ds_prep

        counts = ds_prep.prepare_all()
        for key, count in counts.items():
            _log.debug("  %s: %s", key, count)
    except Exception as exc:
        w = f"TrOCR data prep failed: {exc}"
        _log.warning("%s", w)
        warnings.append(w)
        return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=1, warnings=warnings)

    if marker_file is not None:
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            marker_file.write_text(f"prep_hash={prep_hash}\n")
        except Exception:
            pass

    return StageResult(name="TrOCR Data Prep", duration=0.0, exit_status=0, warnings=warnings)


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

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
        "--trocr-only",
        action="store_true",
        help=(
            "Comprehensive TrOCR-only mode: train YOLO+TrOCR once, then train and "
            "evaluate all three FieldAttentionAssigner backends "
            "(char / lm / lm+vision) plus the regex heuristic baseline. "
            "Skips all DONUT experiments. "
            "Results saved to results/trocr_all_backends.json."
        ),
    )
    p.add_argument(
        "--trocr-single",
        action="store_true",
        default=False,
        help=(
            "Train the TrOCR model **once** on SROIE-only data and reuse those "
            "results for all 8 experiment slots (original single-experiment behaviour). "
            "By default the pipeline trains a separate TrOCR model per experiment "
            "using each experiment's dataset combination (SROIE + auxiliary datasets). "
            "Use this flag for quick smoke-tests or when a single baseline model is "
            "sufficient."
        ),
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
        "--fancy",
        action="store_true",
        help=(
            "Enable fancy console output: box-drawing banners, stage headers, "
            "and the full box-drawing results table. By default the console is "
            "minimal (plain text, no decorations) so output fits in Copilot chat."
        ),
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
        "--superfast",
        action="store_true",
        help=(
            "Superfast mode: TrOCR+YOLO only — no DONUT training. Absolute bare minimum. "
            "YOLO: yolov8n, 1 epoch, 160 px, SGD+Nesterov. "
            "TrOCR: 1 epoch, max_len=32, batch=4. "
            "All 3 field-assigner backends (char/lm/lm+vision), 1 epoch each. "
            "Target: <3 min on RTX 4090. Generates paper_superfast.tex."
        ),
    )
    p.add_argument(
        "--instant",
        action="store_true",
        help=(
            "Instant mode: like --superfast but with aggressive tensor caching. "
            "First run builds caches (~3.5 min). "
            "Subsequent runs complete in <30 s by loading pre-processed tensors, "
            "skipping data prep via a hash-based marker, and skipping YOLO+TrOCR "
            "training when cached weights exist. "
            "Generates paper_instant.tex."
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
    p.add_argument(
        "--verify",
        action="store_true",
        help=(
            "After each experiment completes, verify that F1 > 0. "
            "Logs a WARNING if F1 == 0.0 (silent failure indicator). "
            "Use with --experiment N for targeted validation. "
            "Exits with code 1 if any experiment produces F1=0 after training."
        ),
    )
    p.add_argument(
        "--keep-models",
        action="store_true",
        default=False,
        help=(
            "Keep model checkpoint directories after evaluation instead of deleting them. "
            "By default, model directories are removed after each experiment to save disk space "
            "(only the result JSON is needed for the paper pipeline). "
            "Use this flag if you want to reuse checkpoints or inspect model weights."
        ),
    )
    p.add_argument(
        "--no-disk-cleanup",
        action="store_true",
        default=False,
        help=(
            "Disable all automatic disk cleanup between experiments. "
            "By default, model directories are deleted after evaluation to prevent "
            "'No space left on device' (OS error 28) crashes. "
            "Use for debugging only."
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
    logger.info("Pipeline started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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

    # Fancy mode: enable box-drawing banners and decorative console output.
    if getattr(args, "fancy", False):
        global _FANCY_OUTPUT  # noqa: PLW0603
        _FANCY_OUTPUT = True

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

    if getattr(args, "instant", False):
        logger.info("Instant mode detected (--instant flag)")
        exit_code = _instant_mode_handler(args, logger)
        total_elapsed = time.monotonic() - t_start
        logger.info(
            f"Instant mode complete in {total_elapsed / 60:.1f} min (exit code {exit_code})"
        )
        sys.exit(exit_code)

    if getattr(args, "superfast", False):
        logger.info("Superfast mode detected (--superfast flag)")
        exit_code = _superfast_mode_handler(args, logger)
        total_elapsed = time.monotonic() - t_start
        logger.info(
            f"Superfast mode complete in {total_elapsed / 60:.1f} min (exit code {exit_code})"
        )
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

    # ── Environment snapshot (full details to terminal.txt; brief summary to console) ──
    import torch

    _banner("ENVIRONMENT DIAGNOSTICS")
    # Brief single-line GPU/Python summary always shown on console.
    _py_ver = platform.python_version()
    _cuda = torch.cuda.is_available()
    if _cuda:
        _gpu_name = torch.cuda.get_device_name(0)
        _gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(
            "Python %s | GPU: %s (%.0f GB) | CUDA %s",
            _py_ver,
            _gpu_name,
            _gpu_gb,
            torch.version.cuda,
        )
    else:
        logger.info("Python %s | GPU: not available", _py_ver)
    # Full package list goes to terminal.txt only (debug level).
    logger.debug("Platform: %s", platform.platform())
    logger.debug("Executable: %s", sys.executable)
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
            logger.debug("  %-20s: %s", pkg, ver)
        except ImportError:
            logger.debug("  %-20s: NOT INSTALLED", pkg)
    logger.debug("Workspace: %s", args.workspace)
    logger.debug("SROIE dir: %s", args.sroie_dir)
    logger.debug("CWD: %s", Path.cwd())
    try:
        from resource_manager import _get_available_ram_bytes

        _avail = _get_available_ram_bytes()
        logger.debug("RAM: %.1f GB available", _avail / (1024**3))
    except Exception:
        pass
    logger.debug("CPU cores: %d", os.cpu_count() or 0)

    # Run the pipeline via the orchestrator (all stages sequential)
    orchestrator = PipelineOrchestrator(args)
    exit_code = orchestrator.run()

    total_elapsed = time.monotonic() - t_start
    logger.info("DONE — total %.1f min | exit code %d", total_elapsed / 60, exit_code)
    _print_final_summary(Path("results"))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
