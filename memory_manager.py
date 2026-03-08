# =============================================================================
# memory_manager.py
# Purpose: Central memory budget authority — RAM, GPU VRAM, DataLoader workers
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-08
# =============================================================================
"""
memory_manager.py — Single authority for all memory allocation decisions.

WHY THIS MODULE EXISTS
----------------------
Over 20 sequential OOM-fixing PRs failed because memory budget decisions were
scattered across train.py (hardcoded `* 3` per-sample estimate), run_experiments.py
(ad-hoc _gpu_cleanup calls), and dataset_loaders.py (no HF Arrow cache release).
Each PR fixed one symptom, broke another.

This module centralises every memory decision:
  - compute_pil_mb_per_sample()    : correct per-sample RAM estimate at any resolution
  - ram_cache_is_safe()            : replaces every hardcoded `* 3` or `* 14.2` constant
  - release_hf_dataset()           : safe teardown of HuggingFace Arrow datasets
  - flush_hf_arrow_cache()         : disables HF datasets caching to stop accumulation
  - shutdown_dataloader_workers()  : explicit worker termination before del trainer

ADDING A NEW DATASET
--------------------
When adding a new dataset:
  1. Add an entry to datasets_registry.json in the repository root.
  2. Implement a loader class in dataset_loaders.py following the BaseDatasetLoader ABC.
  3. Call release_hf_dataset(ds) + flush_hf_arrow_cache() after sample extraction
     if your loader uses HuggingFace load_from_disk() or load_dataset().
  4. No changes to this file are required.

MEMORY MAP (current, 2026-03-08, RTX 4090 24GB VRAM, 192GB RAM)
-----------------------------------------------------------------
At 1280x960 (correct resolution):
  PIL image per sample     :  3.516 MB  (3 x 1280 x 960 / 1_048_576)
  Float32 tensor per sample: 14.064 MB  (3 x 1280 x 960 x 4 / 1_048_576)
  DONUT model weights      :    ~800 MB on GPU VRAM
  AdamW optimizer states   :  ~1,600 MB on GPU VRAM (2x model = m + v moments)
  Gradient activations     :  ~4,000 MB on GPU VRAM at batch_size=8 (RTX 4090)
  DataLoader worker (x8)   :    ~230 MB each = ~1,840 MB total prefetch (prefetch_factor=2)

At 2560x1920 (WRONG — DO NOT USE):
  PIL image per sample     : 14.064 MB  (4.9x reference)
  Float32 tensor per sample: 56.250 MB  (4x reference)
  DataLoader worker (x8)   :    ~920 MB each = ~7,360 MB total (prefetch_factor=4 old)
  -> Exp 8 (3940 samples): PIL cache alone = 3940 x 14.06 MB = 55.4 GB -> OOM
"""

from __future__ import annotations

import gc
import logging

__all__ = [
    "compute_pil_mb_per_sample",
    "ram_cache_is_safe",
    "ram_headroom_mb",
    "release_hf_dataset",
    "flush_hf_arrow_cache",
    "shutdown_dataloader_workers",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference image dimensions — DONUT native pretrained resolution.
# All arithmetic in this module is parameterised by actual H×W, but these
# serve as the documented reference point.
# ---------------------------------------------------------------------------
_REF_H: int = 1280
_REF_W: int = 960

# Safety fraction of available RAM that the PIL image cache is allowed to use.
# 0.06 = 6%: tighter than the previous 15% to account for sequential train→val
# allocations within the same experiment. With 15%, a 4.6 GB train cache could
# consume most of the RAM before the val PIL check ran, causing SIGKILL at Exp 6.
# At 6%, the Exp 6 train PIL check (4683 MB vs 6% × 37 GB = 2258 MB) is blocked
# before it can drain the available RAM for the val dataset init.
_RAM_SAFETY_FRACTION: float = 0.06


def compute_pil_mb_per_sample(height: int, width: int) -> float:
    """Return the RAM in MB required to store one PIL RGB image at height x width.

    Formula: 3 channels x H x W pixels x 1 byte per channel / 1_048_576 bytes per MB.
    This is the decompressed in-memory size, NOT the compressed JPEG/PNG size on disk.

    Examples
    --------
    >>> compute_pil_mb_per_sample(1280, 960)   # DONUT native
    3.516
    >>> compute_pil_mb_per_sample(2560, 1920)  # WRONG resolution -- 4.9x larger
    14.06
    """
    return (3 * height * width) / (1024 * 1024)


def ram_cache_is_safe(
    n_samples: int,
    height: int,
    width: int,
    safety_fraction: float = _RAM_SAFETY_FRACTION,
) -> bool:
    """Return True only if caching n_samples PIL images will not exhaust RAM.

    This replaces every ``estimated_mb = len(samples) * 3`` pattern that was
    hardcoded throughout train.py. The old constant (3 MB/sample) was only
    approximately correct at 1280x960 and catastrophically wrong at 2560x1920
    (actual: 14.06 MB/sample).

    The cache is allowed only if:
        n_samples x mb_per_sample  <  available_ram_mb x safety_fraction

    Parameters
    ----------
    n_samples : int
        Number of samples to be cached.
    height, width : int
        Image dimensions in pixels (read from processor_config.json).
    safety_fraction : float
        Maximum fraction of available RAM the cache may use. Default 0.15 (15%).
        This leaves 85% for: model weights, optimizer states, activations,
        DataLoader worker prefetch buffers, and OS overhead.

    Returns
    -------
    bool
        True if caching is safe, False if it would risk OOM.
    """
    mb_per_sample = compute_pil_mb_per_sample(height, width)
    estimated_mb = n_samples * mb_per_sample

    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        logger.warning(
            "[memory_manager] psutil not available; disabling PIL image cache "
            "(install psutil to enable adaptive caching)."
        )
        return False

    if available_mb <= 0:
        return False

    threshold_mb = available_mb * safety_fraction
    is_safe = estimated_mb < threshold_mb

    logger.info(
        "[memory_manager] PIL cache check: %d samples x %.2f MB = %.0f MB estimated; "
        "%.0f MB available x %.0f%% safety = %.0f MB threshold -> %s",
        n_samples,
        mb_per_sample,
        estimated_mb,
        available_mb,
        safety_fraction * 100,
        threshold_mb,
        "ALLOW" if is_safe else "SKIP",
    )
    return is_safe


def ram_headroom_mb() -> float:
    """Return the current available system RAM in MB.

    This is a live reading (not cached) suitable for point-in-time checks
    immediately before an allocation, e.g., the dual-budget guard in
    MultiDataset pixel tensor precompute.

    Returns a conservative 0.0 on any error so callers treat an unknown
    RAM state as "no headroom available".
    """
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return 0.0


def release_hf_dataset(ds) -> None:
    """Explicitly release a HuggingFace Dataset or DatasetDict from memory.

    Call this after you have finished iterating a dataset loaded with
    load_from_disk() or load_dataset(). Deletes the passed reference,
    runs garbage collection, and attempts to free Arrow memory maps.

    Parameters
    ----------
    ds : datasets.Dataset | datasets.DatasetDict | None
        The dataset object to release. Passing None is safe (no-op).

    Usage
    -----
    After sample-building in any loader that calls load_from_disk():

        ds = load_from_disk(str(hf_cache))
        for item in ds["train"]:
            samples.append(...)
        release_hf_dataset(ds)   # <- add this line
        del ds                    # <- and this

    """
    if ds is None:
        return
    try:
        # Some dataset types expose cleanup_cache_files()
        if hasattr(ds, "cleanup_cache_files"):
            ds.cleanup_cache_files()
        elif hasattr(ds, "values"):
            # DatasetDict: clean each split
            for split_ds in ds.values():
                if hasattr(split_ds, "cleanup_cache_files"):
                    split_ds.cleanup_cache_files()
    except Exception as exc:
        logger.debug("[memory_manager] release_hf_dataset cleanup_cache_files: %s", exc)
    finally:
        del ds
        gc.collect()


def flush_hf_arrow_cache() -> None:
    """Disable HuggingFace datasets Arrow caching to stop module-level accumulation.

    The `datasets` library maintains a module-level cache of Arrow memory-mapped
    tables. Every call to load_from_disk() registers a new entry. Across 8
    sequential experiments that each call load_funsd() and load_invoices_donut(),
    this cache accumulates without bound.

    This function:
      1. Calls datasets.disable_caching() to prevent new cache entries.
      2. Runs gc.collect() to allow Python to release any Arrow mmaps
         whose refcount has dropped to zero.

    Call this after release_hf_dataset() in any loader that uses HuggingFace.
    It is safe to call multiple times (idempotent).

    IMPORTANT: This does NOT delete files from disk. It only stops new in-memory
    Arrow tables from being cached between calls within this process.
    """
    try:
        import datasets as _ds_lib
        if hasattr(_ds_lib, "disable_caching"):
            _ds_lib.disable_caching()
            logger.debug("[memory_manager] HuggingFace datasets caching disabled.")
    except ImportError:
        pass  # datasets not installed — nothing to flush
    except Exception as exc:
        logger.debug("[memory_manager] flush_hf_arrow_cache: %s", exc)
    finally:
        gc.collect()


def shutdown_dataloader_workers(trainer) -> None:
    """Shut down persistent DataLoader worker subprocesses before deleting the trainer.

    WHY THIS IS NECESSARY
    ----------------------
    Seq2SeqTrainer with dataloader_persistent_workers=True keeps worker subprocesses
    alive after trainer.train() returns. Each worker holds prefetch_factor batches
    in shared memory. At batch_size=2, 1280x960, float32, prefetch_factor=2:
      each worker holds: 2 x 2 x 3 x 1280 x 960 x 4 bytes = 56.6 MB
      8 workers total: 8 x 56.6 MB = 453 MB

    Without explicit shutdown, these workers accumulate across 8 sequential
    experiments: up to 8 x 453 MB = 3.6 GB of zombie worker RAM by Exp 8.

    This function shuts down the DataLoader iterator's worker pool using the
    PyTorch internal `_shutdown_workers()` method. This is best-effort: if the
    internal API changes in a future PyTorch version, it silently skips shutdown
    (the workers will eventually be killed when the trainer is GC'd, just later).

    WHEN TO CALL
    ------------
    Call BEFORE `del trainer`. After `del trainer` the iterator reference is gone
    and you cannot call shutdown on it.

    Parameters
    ----------
    trainer : Seq2SeqTrainer
        The active trainer whose DataLoader workers should be shut down.
    """
    if trainer is None:
        return
    try:
        # Try the train DataLoader first
        train_dl = trainer.get_train_dataloader()
        if hasattr(train_dl, "_iterator") and train_dl._iterator is not None:
            _iter = train_dl._iterator
            if hasattr(_iter, "_shutdown_workers"):
                _iter._shutdown_workers()
                logger.debug("[memory_manager] Train DataLoader workers shut down.")
        del train_dl
    except Exception as exc:
        logger.debug("[memory_manager] shutdown train dl workers: %s", exc)

    try:
        # Also try the eval DataLoader if it exists and has workers
        eval_dl = trainer.get_eval_dataloader()
        if eval_dl is not None and hasattr(eval_dl, "_iterator") and eval_dl._iterator is not None:
            _iter = eval_dl._iterator
            if hasattr(_iter, "_shutdown_workers"):
                _iter._shutdown_workers()
                logger.debug("[memory_manager] Eval DataLoader workers shut down.")
        del eval_dl
    except Exception:
        pass  # eval dl may not exist; always best-effort

    gc.collect()
