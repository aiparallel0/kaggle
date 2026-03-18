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
from pathlib import Path

__all__ = [
    "FIELDS",
    "IMAGE_EXTS",
    "MAX_LENGTH",
    "BASE_MODEL",
    "SEED",
    "NEW_TOKENS",
    "EMPTY_GT",
    "DEVICE",
    "WORKSPACE",
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
WORKSPACE: Path = Path(os.environ.get("DONUT_WORKSPACE", "/workspace"))


def _get_sroie_dir() -> Path:
    """Return SROIE data root; re-reads SROIE_DATA_DIR env var at call time.

    Implemented as a function (not a constant) so that callers who set the
    env var after import time (e.g. run_all.py) get the correct path.
    """
    return Path(os.environ.get("SROIE_DATA_DIR", "/workspace/ICDAR-2019-SROIE/data"))


def _optimal_num_workers() -> int:
    """Return the optimal DataLoader num_workers based on CPU core count."""
    return min(8, max(4, multiprocessing.cpu_count() // 2))


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


def _mask_empty_field_labels(labels, gt: dict, tokenizer) -> "torch.Tensor":  # type: ignore[name-defined]  # noqa: F821
    """Set label token IDs for empty-field spans to -100.

    When a ground-truth field value is empty (e.g. address=""), including the
    open/close tag pair in the label sequence teaches the model to output
    ``<s_address></s_address>`` — a negative training signal.  Setting those
    positions to -100 prevents any gradient from flowing for empty fields.
    Uses the same -100 convention as padding masks (CrossEntropyLoss ignores
    index -100).

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
        The labels tensor with empty-field span positions set to -100.
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
                labels[start : end + 1] = -100
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
        with self._lock:
            self._flush_last()
        self._target.flush()

    def close(self) -> None:
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


def _progress(iterable, desc: str = "", total: int | None = None):
    """Logging-based progress — replaces tqdm. Emits at 0 %, 10 %, … 100 %."""
    import logging as _logging

    _log = _logging.getLogger(__name__)
    items = list(iterable) if not hasattr(iterable, "__len__") and total is None else iterable
    n = total if total is not None else len(items)  # type: ignore[arg-type]
    step = max(1, -(-n // 10))  # ceiling division by 10
    for idx, item in enumerate(items):
        if idx % step == 0 or idx == n - 1:
            pct = int(100 * idx / max(n - 1, 1))
            if desc:
                _log.info("%s %3d%%", desc, pct)
            else:
                _log.info("%3d%%", pct)
        yield item


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
