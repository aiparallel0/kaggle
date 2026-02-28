# constants.py — Single source of truth for all shared constants.
#
# Previously, FIELDS, IMAGE_EXTS, MAX_LENGTH, BASE_MODEL, SEED, and NEW_TOKENS
# were duplicated independently in 5+ files (run_experiments.py, train.py,
# evaluate.py, inject_results.py, dataset_loaders.py).  Any change had to be
# replicated manually, risking silent drift.  All files now import from here.

import multiprocessing
import os
from pathlib import Path

# SROIE Task-3 target fields — the four key-value pairs extracted from receipts.
FIELDS: list[str] = ["company", "date", "address", "total"]

# Accepted image file extensions for dataset loading.
IMAGE_EXTS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp",
})

# Maximum decoder token length for DONUT fine-tuning and inference.
# Increased from 512 to 768 to reduce truncation of long address fields,
# which was the weakest-performing SROIE field.
MAX_LENGTH: int = 768

# Base model checkpoint — CORD-pretrained DONUT used as starting point.
BASE_MODEL: str = "naver-clova-ix/donut-base-finetuned-cord-v2"

# Global random seed for reproducibility across all experiments.
SEED: int = 42

# Special tokens added to the tokenizer for SROIE structured output.
NEW_TOKENS: list[str] = [
    "<s_sroie>", "</s_sroie>",
    "<s_company>", "</s_company>",
    "<s_date>",    "</s_date>",
    "<s_address>", "</s_address>",
    "<s_total>",   "</s_total>",
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
    """Delete objects, run garbage collection, and empty CUDA cache.

    Use after a training/evaluation stage to free GPU memory before the
    next stage.  Accepts any number of objects to delete; safe to call
    with no arguments (just runs GC + empty_cache).
    """
    import gc

    import torch
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
