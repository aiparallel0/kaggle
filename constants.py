# =============================================================================
# constants.py
# Purpose: Shared project constants + project metadata (merged from pyproject.toml)
# Merged from: pyproject.toml
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-07
# =============================================================================
# constants.py — Single source of truth for all shared constants.
#
# Previously, FIELDS, IMAGE_EXTS, MAX_LENGTH, BASE_MODEL, SEED, and NEW_TOKENS
# were duplicated independently in 5+ files (run_experiments.py, train.py,
# evaluate.py, inject_results.py, dataset_loaders.py).  Any change had to be
# replicated manually, risking silent drift.  All files now import from here.

import logging
import multiprocessing
import os
import threading
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "FIELDS",
    "IMAGE_EXTS",
    "MAX_LENGTH",
    "BASE_MODEL",
    "SEED",
    "NEW_TOKENS",
    "EMPTY_GT",
    "MAX_CONSECUTIVE_BATCH_FAILURES",
    "DEVICE",
    "WORKSPACE",
    # Named constants (OCP-1)
    "LABEL_IGNORE_INDEX",
    "BYTE_UNIT_SIZE",
    "MAX_DATALOADER_WORKERS",
    "MIN_DATALOADER_WORKERS",
    "DONUT_IMAGE_SIZE",
    # Disk utilities
    "format_bytes",
    "get_disk_usage",
    # Project metadata
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "PROJECT_DESCRIPTION",
    "PROJECT_ENTRY_POINT",
]
# _get_sroie_dir, _optimal_num_workers, _gpu_cleanup are intentionally
# NOT in __all__ (underscore-prefixed internal helpers)

# SROIE Task-3 target fields — the four key-value pairs extracted from receipts.
FIELDS: list[str] = ["company", "date", "address", "total"]

# Accepted image file extensions for dataset loading.
IMAGE_EXTS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
    }
)

# MAX_LENGTH=768 is generous for SROIE (output rarely >100 tokens).
# For 4090 iteration speed, operators may override to 256 in ExperimentConfig
# with zero F1 impact on SROIE. Do not lower the default here — other
# experiments (high-res, Exp 14/16/17/18) may need the headroom.
# (Increased from 512 to 768 to reduce truncation of long address fields,
# which was the weakest-performing SROIE field.)
MAX_LENGTH: int = 768

# Base model checkpoint — clean donut-base with no task-specific fine-tuning.
# Using donut-base (not donut-base-finetuned-cord-v2) avoids CORD decoder priors
# that compete with SROIE tokens and cause F1 collapse when fine-tuning on SROIE.
BASE_MODEL: str = "naver-clova-ix/donut-base"

# Global random seed for reproducibility across all experiments.
SEED: int = 42

# Special tokens added to the tokenizer for SROIE structured output.
NEW_TOKENS: list[str] = [
    "<s_sroie>",
    "</s_sroie>",
    "<s_company>",
    "</s_company>",
    "<s_date>",
    "</s_date>",
    "<s_address>",
    "</s_address>",
    "<s_total>",
    "</s_total>",
]

# Empty ground-truth template matching the SROIE schema.
EMPTY_GT: dict[str, str] = {"company": "", "date": "", "address": "", "total": ""}

# ---------------------------------------------------------------------------
# Named constants extracted from magic numbers (OCP-1, CROSS-4)
# ---------------------------------------------------------------------------

# Label ID ignored by CrossEntropyLoss — used to mask empty-field spans and padding.
LABEL_IGNORE_INDEX: int = -100

# Binary byte-unit multiplier for human-readable formatting.
BYTE_UNIT_SIZE: int = 1024

# DataLoader worker count bounds.
MAX_DATALOADER_WORKERS: int = 8
MIN_DATALOADER_WORKERS: int = 4

# DONUT native image resolution (height, width).  All files should reference this
# constant instead of hardcoding (1280, 960).
DONUT_IMAGE_SIZE: tuple[int, int] = (1280, 960)

# Maximum number of consecutive batch-level failures (CUDA OOM, corrupted batch, etc.)
# before the training loop aborts with a RuntimeError.  A training run that silently
# skips more than this many batches in a row has almost certainly stalled — aborting
# is safer than producing a broken model.  Used by train_trocr_yolo.py.
# Fix: issue_report_summary critical #3.
MAX_CONSECUTIVE_BATCH_FAILURES: int = 10

# ---------------------------------------------------------------------------
# Device — GPU if available, else CPU.
# Uses try/except so constants.py can be imported in torch-free test envs.
# ---------------------------------------------------------------------------

try:
    import torch as _torch

    DEVICE: str = "cuda" if _torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE: str = "cpu"

# ---------------------------------------------------------------------------
# Workspace and path helpers
# ---------------------------------------------------------------------------

# Default workspace path — overridden by DONUT_WORKSPACE env var at import time.
# run_all.py sets the env var before lazily importing sub-modules so this value
# is correct by the time the sub-modules are first imported.


def _resolve_workspace() -> Path:
    """Return a writable workspace directory.

    Priority:
    1. ``DONUT_WORKSPACE`` env var path (if it can be created / already exists).
    2. ``<repo_dir>/workspace`` — fallback when the env-var path cannot be
       created due to a permission error, e.g. Git Bash on Windows where
       ``/workspace`` is translated by MSYS2 to
       ``C:\\Program Files\\Git\\workspace`` (a system-protected directory).

    This function is stdlib-only; it must never import torch or transformers.
    """
    _env_val = os.environ.get("DONUT_WORKSPACE", "/workspace")
    _candidate = Path(_env_val)
    try:
        _candidate.mkdir(parents=True, exist_ok=True)
        return _candidate
    except OSError:
        _fallback = Path(__file__).resolve().parent / "workspace"
        try:
            _fallback.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create workspace at {_candidate!r} (primary) or "
                f"{_fallback!r} (fallback). "
                "Set the DONUT_WORKSPACE environment variable to a writable path."
            ) from exc
        return _fallback


WORKSPACE: Path = _resolve_workspace()


def _get_sroie_dir() -> Path:
    """Return SROIE data root; re-reads SROIE_DATA_DIR env var at call time.

    Implemented as a function (not a constant) so that callers who set the
    env var after import time (e.g. run_all.py) get the correct path.
    """
    return Path(os.environ.get("SROIE_DATA_DIR", "/workspace/ICDAR-2019-SROIE/data"))


def _optimal_num_workers() -> int:
    """Return the optimal DataLoader num_workers based on CPU core count."""
    return min(
        MAX_DATALOADER_WORKERS, max(MIN_DATALOADER_WORKERS, multiprocessing.cpu_count() // 2)
    )


def _gpu_cleanup(*objects) -> None:
    """Run garbage collection and empty the CUDA cache.

    Use after a training/evaluation stage to free GPU memory before the
    next stage.  Safe to call with no arguments.

    IMPORTANT: Callers MUST ``del`` their own local references to large
    objects (models, trainers, datasets, processors) *before* calling
    this function.  Passing objects as arguments does NOT free them —
    ``del obj`` inside this function only removes the local parameter
    binding, leaving the caller's references (and the underlying GPU
    tensors) alive.  The correct pattern is::

        del model, processor, trainer, train_ds
        if val_ds is not None:
            del val_ds
        _gpu_cleanup()
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all CUDA ops complete before freeing
        torch.cuda.empty_cache()


def _mask_empty_field_labels(
    labels: "torch.Tensor",  # noqa: F821
    gt: dict[str, str],
    tokenizer: "PreTrainedTokenizerBase",  # noqa: F821
) -> "torch.Tensor":  # type: ignore[name-defined]  # noqa: F821
    """Set label token IDs for empty-field spans to ``LABEL_IGNORE_INDEX``.

    When a ground-truth field value is empty (e.g. address=""), including the
    open/close tag pair in the label sequence teaches the model to output
    ``<s_address></s_address>`` — a negative training signal.  Setting those
    positions to ``LABEL_IGNORE_INDEX`` prevents any gradient from flowing for
    empty fields.  Uses the same convention as padding masks
    (``CrossEntropyLoss`` ignores ``LABEL_IGNORE_INDEX``).

    Parameters
    ----------
    labels:
        1-D label tensor from ``tokenizer(...).input_ids.squeeze()``.
    gt:
        Ground-truth dict with keys matching ``FIELDS``.
    tokenizer:
        HuggingFace tokenizer that has the SROIE special tokens registered via
        ``add_special_tokens``.  Each ``<s_{field}>`` / ``</s_{field}>`` encodes
        to exactly one token.

    Returns
    -------
    torch.Tensor
        The labels tensor with empty-field span positions set to
        ``LABEL_IGNORE_INDEX``.
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for f in FIELDS:
        if gt.get(f, "").strip():
            continue  # field has content — do not mask
        open_id = tokenizer.convert_tokens_to_ids([f"<s_{f}>"])[0]
        close_id = tokenizer.convert_tokens_to_ids([f"</s_{f}>"])[0]
        # Skip if the special tokens are not registered in the vocabulary.
        if unk_id is not None and (open_id == unk_id or close_id == unk_id):
            continue
        open_pos = (labels == open_id).nonzero(as_tuple=True)[0]
        close_pos = (labels == close_id).nonzero(as_tuple=True)[0]
        if len(open_pos) > 0 and len(close_pos) > 0:
            start = int(open_pos[0])
            end = int(close_pos[0])
            if end >= start:
                labels[start : end + 1] = LABEL_IGNORE_INDEX
    return labels


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for fully reproducible training runs.

    Covers Python random, NumPy, PyTorch (CPU + all CUDA devices), and
    cuDNN deterministic mode.  This is the canonical seed-setting function
    for the entire pipeline — import and call it instead of using the
    lighter transformers.set_seed() which only seeds the transformers RNG.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging utilities (from logging_utils.py)
# ---------------------------------------------------------------------------
_NOISY_THIRD_PARTY_LOGGERS = [
    "PIL",
    "PIL.PngImagePlugin",
    "PIL.TiffImagePlugin",
    "PIL.Image",
    "PIL.JpegImagePlugin",
    "PIL.WebPImagePlugin",
    "urllib3",
    "urllib3.connectionpool",
    "filelock",
    "huggingface_hub",
    "huggingface_hub.utils._validators",
    "transformers.tokenization_utils_base",
    "fsspec",
    "fsspec.local",
]


class DeduplicatingHandler(logging.Handler):
    """Wraps another handler; collapses consecutive identical log records.

    When the same (logger-name, level, message) tuple is emitted N times in a
    row the output becomes a single line ending with ``[×N]``.  Different
    messages are emitted immediately, flushing any pending count first.

    Example output::

        2026-03-11 08:19:24 | PIL.PngImagePlugin | DEBUG | STREAM b'IHDR' 16 13  [×47]

    This is thread-safe: all mutable state is protected by a
    ``threading.Lock``.
    """

    def __init__(self, target: logging.Handler) -> None:
        super().__init__()
        self._target = target
        self._lock = threading.Lock()
        self._last_key: tuple | None = None
        self._last_record: logging.LogRecord | None = None
        self._count: int = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record, collapsing consecutive identical messages."""
        key = (record.name, record.levelno, record.getMessage())
        with self._lock:
            if key == self._last_key:
                self._count += 1
            else:
                self._flush_last()
                self._last_key = key
                self._last_record = record
                self._count = 1

    def _flush_last(self) -> None:
        """Emit the pending record (with count suffix if repeated). NOT thread-safe — caller holds lock."""
        if self._last_record is None:
            return
        if self._count > 1:
            self._last_record.msg = f"{self._last_record.getMessage()}  [\u00d7{self._count}]"
            self._last_record.args = ()
        try:
            self._target.emit(self._last_record)
        except Exception:
            self.handleError(self._last_record)
        self._last_record = None
        self._last_key = None
        self._count = 0

    def flush(self) -> None:
        """Flush any pending deduplicated record, then flush the target handler."""
        with self._lock:
            self._flush_last()
        self._target.flush()

    def close(self) -> None:
        """Flush pending records, close the target handler, then close self."""
        with self._lock:
            self._flush_last()
        self._target.close()
        super().close()


def suppress_noisy_loggers(level: int = logging.WARNING) -> None:
    """Set all known noisy third-party loggers to *level* (default: WARNING).

    Call this immediately after ``logging.basicConfig`` / after setting up the
    root logger so that subsequent third-party imports respect the level.
    """
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(level)


# ---------------------------------------------------------------------------
# Shared utility functions (used by both evaluation.py and reporting.py)
# ---------------------------------------------------------------------------


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance — replaces the editdistance package."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
    return dp[n]


def format_bytes(n_bytes: int) -> str:
    """Return a human-readable byte count string (e.g. '1.2 GB', '340 MB').

    Uses binary (1024-based) units: KB = 1024 B, MB = 1024 KB, GB = 1024 MB.
    Safe to call with 0 or negative values.
    """
    n = max(0, int(n_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < BYTE_UNIT_SIZE or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n //= BYTE_UNIT_SIZE
    return f"{n} TB"  # unreachable but satisfies type checker


def get_disk_usage(path: str | Path | None = None) -> tuple[int, int, int]:
    """Return (total, used, free) disk space in bytes for the partition containing *path*.

    Parameters
    ----------
    path:
        Path on the partition to query.  Defaults to the workspace root
        (``WORKSPACE``), or ``/`` if the workspace does not exist.

    Returns
    -------
    tuple[int, int, int]
        ``(total_bytes, used_bytes, free_bytes)``.
        Returns ``(0, 0, 0)`` if ``shutil.disk_usage`` fails (e.g. unsupported OS).
    """
    import shutil

    if path is None:
        path = WORKSPACE if WORKSPACE.exists() else Path("/")
    try:
        usage = shutil.disk_usage(str(path))
        return usage.total, usage.used, usage.free
    except OSError:
        return 0, 0, 0


def _progress(iterable: Iterable, desc: str = "", total: int | None = None) -> Generator:
    """Progress iterator: rich progress bar when available, else single \\r console line.

    When ``rich`` is installed the iterator renders a proper progress bar with
    ETA that updates in-place without cluttering the terminal.  Falls back
    gracefully to the original ``\\r`` overwrite behaviour when ``rich`` is absent.

    File (terminal.txt): emits one DEBUG log line at each 10 % milestone so the
    file retains a human-readable audit trail without flooding the console.
    """
    import logging as _logging
    import sys as _sys

    _log = _logging.getLogger(__name__)
    items = list(iterable) if not hasattr(iterable, "__len__") and total is None else iterable
    n = total if total is not None else len(items)  # type: ignore[arg-type]
    step = max(1, -(-n // 10))  # ceiling division by 10
    label = desc if desc else "Progress"

    # Try rich progress bar first
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(label, total=n)
            for idx, item in enumerate(items):
                # File: only at 10 % milestones
                if idx % step == 0 or idx == n - 1:
                    pct = int(100 * (idx + 1) / max(n, 1))
                    if desc:
                        _log.debug("[progress] %s %3d%%", desc, pct)
                    else:
                        _log.debug("[progress] %3d%%", pct)
                yield item
                progress.advance(task)
        return
    except (ImportError, RuntimeError):
        pass  # rich not available or failed — fall through to plain \r

    for idx, item in enumerate(items):
        pct = int(100 * (idx + 1) / max(n, 1))
        # Console: single overwriting line (bypasses logging system entirely).
        _sys.stdout.write(f"\r{label} {idx + 1}/{n} [{pct}%]   ")
        _sys.stdout.flush()
        # File: only at 10 % milestones (DEBUG → goes to terminal.txt, not console).
        if idx % step == 0 or idx == n - 1:
            if desc:
                _log.debug("[progress] %s %3d%%", desc, pct)
            else:
                _log.debug("[progress] %3d%%", pct)
        yield item
    # Finalize: newline so subsequent log output starts on a fresh line.
    _sys.stdout.write("\n")
    _sys.stdout.flush()


# ---------------------------------------------------------------------------
# Project metadata (merged from pyproject.toml)
# ---------------------------------------------------------------------------
# Tool config (ruff): target-version=py310, line-length=100
# ruff.lint: select E,W,F,I,UP,B,SIM; ignore E501,E741,B905,SIM108,SIM105,UP015
# ruff.format: quote-style=double

PROJECT_NAME: str = "donut-kiedata"
PROJECT_VERSION: str = "1.0.0"
PROJECT_DESCRIPTION: str = (
    "DONUT + TrOCR + YOLO multi-dataset KIE pipeline for receipt understanding"
)
PROJECT_ENTRY_POINT: str = "run_all:main"  # console_scripts entry point


# ---------------------------------------------------------------------------
# Pipeline readiness validation (stdlib-only — safe before torch import)
# ---------------------------------------------------------------------------


def _check_constants_integrity() -> None:
    """Verify constants.py exports have expected values.

    Raises
    ------
    AssertionError
        If any core constant has an unexpected value.
    """
    assert FIELDS == ["company", "date", "address", "total"], (
        f"FIELDS={FIELDS!r} — expected ['company', 'date', 'address', 'total']"
    )
    assert len(NEW_TOKENS) >= 10, f"NEW_TOKENS has {len(NEW_TOKENS)} entries (expected ≥10)"
    assert EMPTY_GT == {"company": "", "date": "", "address": "", "total": ""}, (
        f"EMPTY_GT={EMPTY_GT!r}"
    )
    assert MAX_LENGTH > 0, f"MAX_LENGTH={MAX_LENGTH}"
    assert BASE_MODEL, "BASE_MODEL is empty"
    assert SEED == 42, f"SEED={SEED}"


def _check_new_tokens_completeness() -> None:
    """Verify all SROIE special tokens are present in NEW_TOKENS.

    Raises
    ------
    AssertionError
        If any expected SROIE token is missing from *NEW_TOKENS*.
    """
    expected = {
        "<s_sroie>",
        "</s_sroie>",
        "<s_company>",
        "</s_company>",
        "<s_date>",
        "</s_date>",
        "<s_address>",
        "</s_address>",
        "<s_total>",
        "</s_total>",
    }
    missing = expected - set(NEW_TOKENS)
    assert not missing, f"NEW_TOKENS is missing: {missing}"


def _check_data_pipeline_importable() -> None:
    """Verify data_pipeline module is importable.

    Raises
    ------
    AssertionError
        If the *data_pipeline* module cannot be imported or lacks *SROIELoader*.
    """
    import importlib

    mod = importlib.import_module("data_pipeline")
    assert hasattr(mod, "SROIELoader"), "data_pipeline.SROIELoader not found"


def _check_empty_gt_alignment() -> None:
    """Verify EMPTY_GT keys match FIELDS.

    Raises
    ------
    AssertionError
        If the keys of *EMPTY_GT* do not exactly match *FIELDS*.
    """
    assert set(EMPTY_GT.keys()) == set(FIELDS), (
        f"EMPTY_GT keys {set(EMPTY_GT.keys())} != FIELDS {set(FIELDS)}"
    )


def validate_pipeline_readiness() -> dict[str, Any]:
    """Run lightweight stdlib-only checks that the import chain is intact.

    This function is intentionally free of torch / transformers imports so it
    can be called before any heavy dependency is loaded (e.g. in CI, in the
    startup-diagnostics phase of run_all.py, or from a pre-commit hook).

    Individual checks are available as module-level functions
    (``_check_constants_integrity``, ``_check_new_tokens_completeness``,
    ``_check_data_pipeline_importable``, ``_check_empty_gt_alignment``)
    for callers that need to run a single check selectively.

    Returns
    -------
    dict
        ``{"passed": bool, "checks": [{"name": str, "passed": bool, "error": str|None}]}``

    Examples
    --------
    >>> result = validate_pipeline_readiness()
    >>> assert result["passed"], result["checks"]
    """
    checks: list[dict] = []

    def _run(name: str, fn):
        try:
            fn()
            checks.append({"name": name, "passed": True, "error": None})
        except Exception as exc:
            checks.append({"name": name, "passed": False, "error": str(exc)})

    _run("constants integrity", _check_constants_integrity)
    _run("NEW_TOKENS completeness", _check_new_tokens_completeness)
    _run("data_pipeline import", _check_data_pipeline_importable)
    _run("EMPTY_GT/FIELDS alignment", _check_empty_gt_alignment)

    all_passed = all(c["passed"] for c in checks)
    return {"passed": all_passed, "checks": checks}
