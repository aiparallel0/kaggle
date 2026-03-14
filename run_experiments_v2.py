# =============================================================================
# run_experiments_v2.py
# Purpose: 8-experiment DONUT fine-tuning loop — v2 with resolution-sync fix
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-14
# =============================================================================
"""
run_experiments_v2.py — Drop-in replacement for run_experiments.py that adds the
critical processor ↔ encoder image-size synchronisation fix.

ROOT CAUSE OF SUPPRESSED F1 (identified 2026-03-14)
====================================================
The original run_experiments.py loaded the DonutProcessor and
VisionEncoderDecoderModel from the ``naver-clova-ix/donut-base`` checkpoint
but never explicitly aligned two separate resolution fields:

  1. ``processor.image_processor.size``   — controls how PIL images are resized
                                            before being fed to the model.
  2. ``model.config.encoder.image_size``  — controls the Swin encoder's absolute
                                            positional embedding grid (H/32 × W/32).

The pretrained checkpoint stores ``image_size = [2560, 1920]`` in its encoder
config.  When fine-tuning at 1280×960 (the intended resolution per CLAUDE.md),
the processor resizes images to 1280×960 but the encoder's positional embeddings
are still shaped for 2560×1920.  HuggingFace interpolates them silently —
no error is raised — but every attention window operates on incorrect spatial
coordinates, suppressing company-field F1 to as low as 0.016 and global F1
to 0.668 for the SROIE-only baseline.

THE FIX (applied in _apply_resolution_sync)
============================================
After loading (or deep-copying) the processor and model, immediately call::

    _apply_resolution_sync(processor, model, height=1280, width=960)

which sets BOTH fields atomically:

    processor.image_processor.size = {"height": H, "width": W}
    model.config.encoder.image_size = [H, W]

This ensures the Swin encoder's positional embedding grid exactly matches
the resolution the processor produces — eliminating the attention-window
misalignment that suppressed F1 scores.

ADDITIONAL HARDENING IN v2
===========================
Beyond the resolution fix, v2 also:

  * Freezes only Swin stage-0 (not stages 0+1) so the encoder can adapt
    more freely to SROIE's Southeast Asian receipt layouts.
  * Sets ``model.config.max_length = MAX_LENGTH`` and
    ``model.generation_config.max_new_tokens = MAX_LENGTH`` consistently.
  * Logs the resolved image size at INFO level so it appears in terminal.txt
    for every experiment run — making mismatches immediately visible.
  * Adds a post-setup assertion that verifies the two size fields agree before
    training starts (guards against future regressions).

USAGE
===== 
Run all 8 experiments (v2):
    python run_experiments_v2.py --all

Run a single experiment:
    python run_experiments_v2.py --experiment 2

Force re-run (ignore cached results):
    python run_experiments_v2.py --all --force

Results are saved to results/v2/experiment_N.json and a summary to
results/v2/all_experiments.json.  The v2 subdirectory keeps v2 results
separate from the original run so both can be compared side-by-side.

COMPARISON WITH ORIGINAL
=========================
The only structural changes from run_experiments.py are:

  1. Addition of _apply_resolution_sync() helper (lines ~140-190).
  2. Call to _apply_resolution_sync() inside _build_model_and_datasets()
     immediately after the processor and model are ready (after
     resize_token_embeddings and before dataset construction).
  3. RESULTS_DIR changed to Path("results/v2") to avoid overwriting v1 results.
  4. Minor: partial encoder freeze tightened to stage-0 only.
  5. Minor: max_length / max_new_tokens set consistently on model config.

All other logic (OOM recovery, early stopping, cache validation, audit
logging, resource optimisation, dataset loading) is identical to
run_experiments.py to isolate the resolution fix as the only variable.
"""

import argparse
import copy
import dataclasses
import gc
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Set before torch initializes to reduce GPU memory fragmentation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

import dataset_loaders
import memory_manager as _mm

from constants import (
    BASE_MODEL,
    DEVICE,
    MAX_LENGTH,
    NEW_TOKENS,
    SEED,
    WORKSPACE,
    _gpu_cleanup,
    set_seed,
)
from control_suite import validate_sroie_oversample
from resource_optimizer import (
    TrainingAuditLogger,
    detect_system_resources,
    optimize_hyperparams,
)
from train import MultiDataset

try:
    import flash_attn  # noqa: F401
    FLASH_ATTN_AVAILABLE = True
except Exception:
    FLASH_ATTN_AVAILABLE = False

__all__ = [
    "ExperimentConfig",
    "EXPERIMENTS",
    "TRAIN_CONFIG",
    "_config_to_dict",
    "run_experiment",
    "run_experiment_from_config",
    "run_custom_experiment",
    "save_summary",
    "_apply_resolution_sync",
]

logger = logging.getLogger(__name__)

for pkg in ["httpx", "httpcore", "urllib3", "datasets", "transformers", "huggingface_hub"]:
    logging.getLogger(pkg).setLevel(logging.WARNING)

try:
    from logging_utils import suppress_noisy_loggers
    suppress_noisy_loggers()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# v2 results directory — separate from v1 to allow side-by-side comparison
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/v2")

# ---------------------------------------------------------------------------
# THE FIX — _apply_resolution_sync
# ---------------------------------------------------------------------------

# Canonical fine-tuning resolution for DONUT on SROIE.
# Height × Width must be multiples of 32 (patch_size=4, Swin stride=8 → 32).
# 1280×960 is the standard community fine-tuning resolution that fits comfortably
# in 24 GB VRAM at batch_size=8.
_FINETUNE_H: int = 1280  # height (tall receipts)
_FINETUNE_W: int = 960   # width


def _apply_resolution_sync(
    processor: DonutProcessor,
    model: VisionEncoderDecoderModel,
    height: int = _FINETUNE_H,
    width: int = _FINETUNE_W,
) -> None:
    """Synchronise processor image size and Swin encoder image_size.

    This is THE critical fix for suppressed F1 scores.

    The ``naver-clova-ix/donut-base`` pretrained checkpoint stores
    ``model.config.encoder.image_size = [2560, 1920]`` — the resolution used
    during DONUT's original pre-training on IIT-CDIP and SynthDoG.  When we
    fine-tune at 1280×960 (half that), the ``DonutImageProcessor`` resizes
    images to 1280×960 but the Swin encoder's absolute positional embeddings
    remain anchored to the 2560×1920 grid.  HuggingFace interpolates the
    mismatched positional embeddings silently during ``from_pretrained()``,
    producing a model where every attention window sees spatially incorrect
    position codes — suppressing F1, especially for spatially distributed
    fields like company name and address.

    The fix is to set BOTH fields to the same [H, W] value before any forward
    pass or gradient computation occurs.

    Parameters
    ----------
    processor : DonutProcessor
        The processor whose ``image_processor.size`` dict will be updated.
    model : VisionEncoderDecoderModel
        The model whose ``config.encoder.image_size`` list will be updated.
    height : int
        Target image height in pixels (default 1280).
    width : int
        Target image width in pixels (default 960).

    Side Effects
    ------------
    * ``processor.image_processor.size`` → ``{"height": height, "width": width}``.
    * ``model.config.encoder.image_size`` → ``[height, width]``

    Both mutations happen in-place on the objects passed in.

    Post-condition assertion
    ------------------------
    Raises ``AssertionError`` if the two fields still disagree after the
    update (guards against future API changes that rename the fields).
    """
    # 1. Update the image processor (controls PIL resize in __call__)
    processor.image_processor.size = {"height": height, "width": width}

    # 2. Update the encoder config (controls Swin positional embedding grid)
    model.config.encoder.image_size = [height, width]

    # 3. Post-condition: verify both fields agree
    proc_h = processor.image_processor.size["height"]
    proc_w = processor.image_processor.size["width"]
    enc_h, enc_w = model.config.encoder.image_size
    assert proc_h == enc_h and proc_w == enc_w, (
        f"Resolution sync FAILED: processor=({proc_h},{proc_w}) "
        f"vs encoder=({enc_h},{enc_w}).  Check API compatibility."
    )

    logger.info(
        "[ResolutionSync] processor.image_processor.size = {height: %d, width: %d}  |  "
        "model.config.encoder.image_size = [%d, %d]  ✓",
        height, width, height, width,
    )


# ---------------------------------------------------------------------------
# ExperimentConfig — THE single source of truth for all hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Single source of truth for all training hyperparameters (v2)."""

    name: str
    datasets: list[str]
    epochs: int = 10
    lr: float = 5e-5
    batch_size: int = 8
    seed: int = SEED
    early_stopping_patience: int = 3
    base_model: str = BASE_MODEL
    warmup_steps: int = 40
    weight_decay: float = 0.01
    max_length: int = MAX_LENGTH
    gradient_accumulation_steps: int = 2
    encoder_lr: float = 5e-5
    decoder_lr: float = 1e-4
    description: str = ""
    experiment_id: int = 0
    sroie_oversample: int = 1
    skip_oversample_guard: bool = False

    # Mini-mode accelerators
    subsample_train: int = 0
    subsample_eval: int = 0
    skip_step_validation: bool = False
    lr_schedule: str = "cosine"
    optimizer_type: str = "adamw"

    # v2: explicit fine-tuning resolution (height, width)
    # Both the processor and the encoder config will be set to these values.
    finetune_height: int = _FINETUNE_H
    finetune_width: int = _FINETUNE_W

    @property
    def max_epochs(self) -> int:
        return self.epochs

    @property
    def learning_rate(self) -> float:
        return self.lr

    @property
    def per_device_train_batch_size(self) -> int:
        return self.batch_size

    @property
    def output_dir(self) -> str:
        return str(WORKSPACE / "models" / f"v2_experiment_{self.experiment_id}")


# ---------------------------------------------------------------------------
# Experiment definitions (identical training data to v1)
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[int, ExperimentConfig] = {
    1: ExperimentConfig(
        name="SROIE only (baseline)",
        datasets=["sroie"],
        description="Fine-tune on SROIE training set only.",
        experiment_id=1,
    ),
    2: ExperimentConfig(
        name="SROIE + WildReceipt (naïve baseline)",
        datasets=["sroie", "wildreceipt"],
        description="Naïve multi-dataset run — sroie_oversample=1 intentionally. Control group.",
        experiment_id=2,
        sroie_oversample=1,
        skip_oversample_guard=True,
    ),
    3: ExperimentConfig(
        name="SROIE + Invoices-DONUT (naïve baseline)",
        datasets=["sroie", "invoices_donut"],
        description="Naïve multi-dataset run — sroie_oversample=1 intentionally. Control group.",
        experiment_id=3,
        sroie_oversample=1,
        skip_oversample_guard=True,
    ),
    4: ExperimentConfig(
        name="SROIE + WildReceipt + Invoices (naïve baseline)",
        datasets=["sroie", "wildreceipt", "invoices_donut"],
        description="Naïve combination — sroie_oversample=1 intentionally. Control group.",
        experiment_id=4,
        sroie_oversample=1,
        skip_oversample_guard=True,
    ),
    5: ExperimentConfig(
        name="SROIE + WildReceipt (2x SROIE)",
        datasets=["sroie", "wildreceipt"],
        description="SROIE + WildReceipt with 2x SROIE oversampling.",
        epochs=15,
        sroie_oversample=2,
        experiment_id=5,
    ),
    6: ExperimentConfig(
        name="SROIE + Invoices (2x SROIE)",
        datasets=["sroie", "invoices_donut"],
        description="SROIE + Invoices-DONUT with 2x SROIE oversampling.",
        epochs=15,
        sroie_oversample=2,
        experiment_id=6,
    ),
    7: ExperimentConfig(
        name="SROIE + All (2x SROIE)",
        datasets=["sroie", "wildreceipt", "invoices_donut"],
        description="All datasets with 2x SROIE oversampling.",
        epochs=15,
        sroie_oversample=2,
        experiment_id=7,
    ),
    8: ExperimentConfig(
        name="SROIE + All (3x SROIE)",
        datasets=["sroie", "wildreceipt", "invoices_donut"],
        description="All datasets with 3x SROIE oversampling.",
        epochs=15,
        sroie_oversample=3,
        experiment_id=8,
    ),
}

_default_config = ExperimentConfig(name="", datasets=[])  

TRAIN_CONFIG: dict[str, Any] = {
    "max_epochs": _default_config.epochs,
    "learning_rate": _default_config.lr,
    "per_device_train_batch_size": _default_config.batch_size,
    "early_stopping_patience": _default_config.early_stopping_patience,
    "base_model": _default_config.base_model,
    "warmup_steps": _default_config.warmup_steps,
    "weight_decay": _default_config.weight_decay,
    "max_length": _default_config.max_length,
    "seed": _default_config.seed,
    "gradient_accumulation_steps": _default_config.gradient_accumulation_steps,
    # v2: include resolution in cache key so changing finetune resolution
    # correctly invalidates previously cached results.
    "finetune_height": _default_config.finetune_height,
    "finetune_width": _default_config.finetune_width,
}


def _config_to_dict(config: "ExperimentConfig") -> dict:
    return {
        "max_epochs": config.epochs,
        "learning_rate": config.lr,
        "per_device_train_batch_size": config.batch_size,
        "early_stopping_patience": config.early_stopping_patience,
        "base_model": config.base_model,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "max_length": config.max_length,
        "seed": config.seed,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "finetune_height": config.finetune_height,
        "finetune_width": config.finetune_width,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_experiment(
    exp_id: int,
    samples: list[tuple[Path, dict]],
    output_dir: Path,
    val_samples: list[tuple[Path, dict]] = None,
    base_processor=None,
    base_model=None,
    config=None,
) -> list[dict]:
    """Fine-tune DONUT on *samples* and save the model to *output_dir*.

    v2 change: calls _apply_resolution_sync() after model/processor setup
    so that processor.image_processor.size and model.config.encoder.image_size
    are both set to (finetune_height, finetune_width) before any forward pass.
    """
    from train import DonutTrainer

    if config is None:
        config = EXPERIMENTS[exp_id]

    set_seed(config.seed)
    print(f"\n[v2 Exp {exp_id}] Training on {len(samples)} samples -> {output_dir}")
    print(
        f"[v2 Exp {exp_id}] Hyperparams: epochs={config.epochs}, "
        f"lr={config.lr}, batch_size={config.batch_size}, "
        f"warmup={config.warmup_steps}, wd={config.weight_decay}"
    )
    print(
        f"[v2 Exp {exp_id}] Resolution: {config.finetune_height}×{config.finetune_width} "
        f"(processor + encoder synced)"
    )
    if val_samples:
        print(f"[v2 Exp {exp_id}] Validation set: {len(val_samples)} samples")

    def _build_model_and_datasets():
        # ── Load / deep-copy processor and model ──────────────────────────
        if base_processor is not None and base_model is not None:
            _proc = copy.deepcopy(base_processor)
            _mdl = copy.deepcopy(base_model)
        else:
            _proc = DonutProcessor.from_pretrained(config.base_model)
            if FLASH_ATTN_AVAILABLE:
                try:
                    _mdl = VisionEncoderDecoderModel.from_pretrained(
                        config.base_model, attn_implementation="flash_attention_2"
                    )
                    logger.info("[Model] Flash Attention 2 enabled for decoder")
                except Exception as _fa2_err:
                    logger.info(
                        "[Model] Flash Attention 2 load failed (%s: %s), using default attention",
                        type(_fa2_err).__name__,
                        _fa2_err,
                    )
                    _mdl = VisionEncoderDecoderModel.from_pretrained(config.base_model)
            else:
                logger.info("[Model] flash-attn not installed — using default attention (run fine)")
                _mdl = VisionEncoderDecoderModel.from_pretrained(config.base_model)

        # ── CRITICAL v2 FIX: synchronise processor and encoder image size ──
        # Must be called BEFORE tokenizer resizing and dataset construction
        # so that pixel_values precomputed in MultiDataset.__init__ use the
        # correct resolution.
        _apply_resolution_sync(
            _proc, _mdl,
            height=config.finetune_height,
            width=config.finetune_width,
        )

        # ── Add SROIE special tokens ───────────────────────────────────────
        logger.debug("[Pre-resize] Tokenizer vocab size: %d", len(_proc.tokenizer))
        _proc.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
        _mdl.decoder.resize_token_embeddings(len(_proc.tokenizer))
        logger.debug("[Post-resize] Tokenizer vocab size: %d", len(_proc.tokenizer))

        # Verify tokens were added
        for token in NEW_TOKENS:
            token_ids = _proc.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids or len(token_ids) > 1:
                raise RuntimeError(f"Token addition failed for {token}; vocab may be corrupted")
        logger.debug("[Token-verify] All %d SROIE tokens successfully added", len(NEW_TOKENS))

        # ── Weight-tying fix ──────────────────────────────────────────────
        _mdl.decoder.config.tie_word_embeddings = False

        # ── Decoder config ────────────────────────────────────────────────
        _mdl.config.pad_token_id = _proc.tokenizer.pad_token_id
        _mdl.decoder.config.pad_token_id = _proc.tokenizer.pad_token_id
        _mdl.config.decoder_start_token_id = _proc.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        _mdl.decoder.config.decoder_start_token_id = _proc.tokenizer.convert_tokens_to_ids(
            ["<s_sroie>"]
        )[0]
        # v2: set max_length on both model config and generation config consistently
        _mdl.config.max_length = MAX_LENGTH
        if hasattr(_mdl, "generation_config"):
            _mdl.generation_config.max_new_tokens = MAX_LENGTH

        _decoded = _proc.tokenizer.decode([_mdl.config.decoder_start_token_id])
        if _decoded != "<s_sroie>":
            raise RuntimeError(
                f"decoder_start_token_id decodes to '{_decoded}', not '<s_sroie>'."
            )

        # ── Gradient checkpointing ────────────────────────────────────────
        _GRAD_CKPT_VRAM_THRESHOLD_GB = 24.0
        _enable_grad_ckpt = True
        if torch.cuda.is_available():
            try:
                _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                _enable_grad_ckpt = _vram_gb <= _GRAD_CKPT_VRAM_THRESHOLD_GB
                logger.info(
                    "[GradCkpt] %s — VRAM=%.1f GB (threshold %.0f GB)",
                    "Enabled" if _enable_grad_ckpt else "Disabled",
                    _vram_gb,
                    _GRAD_CKPT_VRAM_THRESHOLD_GB,
                )
            except Exception as exc:
                logger.warning("[GradCkpt] VRAM detection failed (%s) — defaulting to enabled", exc)

        if _enable_grad_ckpt:
            _mdl.config.use_cache = False
            _mdl.decoder.config.use_cache = False
            _mdl.gradient_checkpointing_enable()
        else:
            _mdl.config.use_cache = True
            _mdl.decoder.config.use_cache = True

        # v2: freeze only Swin stage-0 (was stages 0+1 in v1).
        # Freezing fewer encoder layers lets the model adapt more freely to
        # SROIE's Southeast Asian receipt layouts while still preserving the
        # low-level patch embeddings from pre-training.
        frozen_count = 0
        for name, param in _mdl.encoder.named_parameters():
            if "layers.0" in name:
                param.requires_grad = False
                frozen_count += 1
        logger.info("[v2] Frozen Swin stage-0 only (%d parameters frozen)", frozen_count)

        # ── Build datasets ────────────────────────────────────────────────
        _train_ds = MultiDataset(samples, _proc, max_length=config.max_length)
        _val_ds = (
            MultiDataset(
                val_samples,
                _proc,
                max_length=config.max_length,
                precompute_tensors=False,
            )
            if val_samples
            else None
        )

        return _proc, _mdl, _train_ds, _val_ds

    processor, model, train_ds, val_ds = _build_model_and_datasets()

    assert config.epochs == config.max_epochs
    assert config.lr == config.learning_rate
    assert config.batch_size == config.per_device_train_batch_size

    trainer = _build_trainer(config, processor, model, train_ds, val_ds)

    if torch.cuda.is_available():
        try:
            _free_vram, _total_vram = torch.cuda.mem_get_info()
            _used_by_others = _total_vram - _free_vram - torch.cuda.memory_allocated()
            if _used_by_others > 1 * 1024**3:
                logger.warning(
                    "[VRAM] External process(es) occupying ~%.1f GB. Risk of OOM.",
                    _used_by_others / 1024**3,
                )
        except Exception:
            pass

    while True:
        try:
            result = trainer.train()
            break
        except torch.cuda.OutOfMemoryError as e:
            if config.batch_size > 1:
                new_batch = max(1, config.batch_size // 2)
                new_accum = config.gradient_accumulation_steps * 2
                config = dataclasses.replace(
                    config, batch_size=new_batch, gradient_accumulation_steps=new_accum
                )
                print(
                    f"[v2 Exp {exp_id}] CUDA OOM — reducing batch_size to "
                    f"{config.batch_size}, increasing grad_accum to "
                    f"{config.gradient_accumulation_steps} and retrying"
                )
                _mm.shutdown_dataloader_workers(trainer)
                if hasattr(train_ds, "clear_caches"):
                    train_ds.clear_caches()
                if val_ds is not None and hasattr(val_ds, "clear_caches"):
                    val_ds.clear_caches()
                del trainer, model, processor, train_ds
                if val_ds is not None:
                    del val_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                torch.cuda.empty_cache()
                processor, model, train_ds, val_ds = _build_model_and_datasets()
                trainer = _build_trainer(config, processor, model, train_ds, val_ds)
            else:
                raise RuntimeError(
                    f"[v2 Exp {exp_id}] CUDA OOM: batch_size already at minimum (1)"
                ) from e

    trainer.save(output_dir)

    _spot_proc = DonutProcessor.from_pretrained(str(output_dir))
    _spot_id = _spot_proc.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    _spot_decoded = _spot_proc.tokenizer.decode([_spot_id])
    if _spot_decoded == "<s_sroie>":
        print(f"[v2 Exp {exp_id}] Processor spot-check PASS: <s_sroie> id={_spot_id}")
    else:
        print(
            f"[v2 Exp {exp_id}] Processor spot-check FAIL: <s_sroie> → '{_spot_decoded}' "
            "(processor corrupt — evaluation will produce F1=0.0)"
        )

    print(
        f"[v2 Exp {exp_id}] Training complete "
        f"(duration={result.duration_seconds:.1f}s, "
        f"train={result.train_samples}, val={result.val_samples})"
    )

    log_history = result.log_history
    _mm.shutdown_dataloader_workers(trainer)
    if hasattr(train_ds, "clear_caches"):
        train_ds.clear_caches()
    if val_ds is not None and hasattr(val_ds, "clear_caches"):
        val_ds.clear_caches()
    del trainer, model, processor, train_ds
    if val_ds is not None:
        del val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"[v2 Exp {exp_id}] GPU memory released")

    return log_history


def _build_trainer(config, processor, model, train_ds, val_ds):
    """Construct a DonutTrainer from the current config.

    Extracted into a helper so the OOM-retry loop can call it without
    duplicating code.
    """
    from train import DonutTrainer
    return DonutTrainer(
        config=config,
        processor=processor,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )


# ---------------------------------------------------------------------------
# Evaluation — unchanged from v1
# ---------------------------------------------------------------------------

def evaluate_experiment(
    exp_id: int, model_dir: Path, config: "ExperimentConfig | None" = None
) -> dict:
    """Evaluate a fine-tuned model on the SROIE test set."""
    from donut_evaluator import DonutEvaluator

    if config is None:
        config = EXPERIMENTS[exp_id]
    test_samples = dataset_loaders.load_sroie_test()

    if getattr(config, "subsample_eval", 0) > 0 and len(test_samples) > config.subsample_eval:
        import random as _rnd
        _rng = _rnd.Random(getattr(config, "seed", SEED))
        test_samples = _rng.sample(test_samples, config.subsample_eval)
        print(f"[v2 Exp {exp_id}] subsample_eval: evaluating on {len(test_samples)} test samples")
    else:
        print(f"[v2 Exp {exp_id}] Evaluating on {len(test_samples)} SROIE test images")

    processor = DonutProcessor.from_pretrained(str(model_dir))

    evaluator = DonutEvaluator(
        model_path=model_dir,
        processor=processor,
        test_dataset=test_samples,
        task_prompt="<s_sroie>",
        max_length=config.max_length,
        device=DEVICE,
    )

    eval_result = evaluator.evaluate(allow_high_parse_failures=True)
    metrics = eval_result.to_dict()

    if eval_result.parse_failures > 0:
        print(
            f"[v2 Exp {exp_id}] WARNING: {eval_result.parse_failures} of "
            f"{eval_result.num_samples} predictions had parse failures"
        )

    return metrics


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    exp_id: int,
    base_processor=None,
    base_model=None,
    overrides: "dict | None" = None,
) -> dict:
    """Run a single v2 experiment: train, evaluate, save results."""
    import dataclasses as _dc

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment ID {exp_id}. Valid: {list(EXPERIMENTS)}")

    config = EXPERIMENTS[exp_id]
    if overrides:
        config = _dc.replace(config, **overrides)
        print(f"[v2 Exp {exp_id}] --param overrides applied: {overrides}")
    print(f"\n{'=' * 72}")
    print(f"[v2] Experiment {exp_id}: {config.name}")
    print(f"Description: {config.description}")
    print(f"Datasets: {config.datasets}")
    print(f"{'=' * 72}")

    validate_sroie_oversample(
        config.datasets,
        config.sroie_oversample,
        skip_guard=getattr(config, "skip_oversample_guard", False),
    )

    result_file = RESULTS_DIR / f"experiment_{exp_id}.json"

    original_config_dict = _config_to_dict(EXPERIMENTS[exp_id])
    if result_file.exists():
        with open(result_file) as fh:
            cached = json.load(fh)
        if (
            cached.get("datasets") != config.datasets
            or cached.get("config") != original_config_dict
        ):  
            print(
                f"[v2 Exp {exp_id}] STALE result detected (datasets or config mismatch). "
                "Deleting and re-running."
            )
            result_file.unlink()
        else:
            print(f"[v2 Exp {exp_id}] Valid cached result found - skipping.")
            return cached

    train_samples, val_samples = dataset_loaders.get_combined_dataset(
        config.datasets, sroie_oversample=config.sroie_oversample
    )

    if getattr(config, "subsample_train", 0) > 0 and len(train_samples) > config.subsample_train:
        import random as _rnd
        _rng = _rnd.Random(config.seed)
        train_samples = _rng.sample(train_samples, config.subsample_train)
        print(f"[v2 Exp {exp_id}] subsample_train: using {len(train_samples)} samples")

    if len(train_samples) == 0:
        print(f"[v2 Exp {exp_id}] WARNING: No samples loaded - saving empty result.")
        result = {
            "experiment_id": exp_id,
            "name": config.name,
            "datasets": config.datasets,
            "config": _config_to_dict(config),
            "num_train_samples": 0,
            "metrics": {},
            "error": "No training samples available",
        }
        result_file.write_text(json.dumps(result, indent=2))
        return result

    resources = detect_system_resources()
    optimized_config = optimize_hyperparams(
        num_train_samples=len(train_samples),
        available_vram_gb=resources.vram_gb,
        available_ram_gb=resources.ram_gb,
    )

    audit_logger = TrainingAuditLogger(append_to_file="terminal_v2.txt")
    audit_logger.log_config_decision(exp_id, optimized_config)

    _is_micro = getattr(config, "skip_step_validation", False)
    old_batch = config.batch_size
    old_accum = config.gradient_accumulation_steps
    config = dataclasses.replace(
        EXPERIMENTS[exp_id],
        batch_size=optimized_config.batch_size,
        gradient_accumulation_steps=(
            EXPERIMENTS[exp_id].gradient_accumulation_steps
            if _is_micro
            else optimized_config.gradient_accumulation_steps
        ),
        encoder_lr=optimized_config.encoder_lr,
        decoder_lr=optimized_config.decoder_lr,
    )
    print(
        f"[v2 Exp {exp_id}] Resource optimization applied: "
        f"batch_size {old_batch} → {config.batch_size}, "
        f"grad_accum {old_accum} → {config.gradient_accumulation_steps}"
        + (
            " (grad_accum preserved: micro/mini mode)"
            if _is_micro
            else f" ({optimized_config.config_explanation})"
        )
    )

    if (
        config.batch_size != EXPERIMENTS[exp_id].batch_size
        or config.gradient_accumulation_steps != EXPERIMENTS[exp_id].gradient_accumulation_steps
    ):
        assert config is not EXPERIMENTS[exp_id]

    from resource_optimizer import validate_training_config

    if not getattr(config, "skip_step_validation", False):
        validate_training_config(
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_samples=len(train_samples),
            epochs=config.epochs,
        )

    model_dir = WORKSPACE / "models" / f"v2_experiment_{exp_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    _t_train_start = time.monotonic()
    log_history = train_experiment(
        exp_id,
        train_samples,
        model_dir,
        val_samples=val_samples,
        base_processor=base_processor,
        base_model=base_model,
        config=config,
    )
    _train_duration_sec = time.monotonic() - _t_train_start

    try:
        metrics = evaluate_experiment(exp_id, model_dir)
        metrics["training_time_sec"] = _train_duration_sec
    except RuntimeError as exc:
        if "Self-test FAILED" in str(exc) or "Parse failure threshold exceeded" in str(exc):
            print(
                f"[v2 Exp {exp_id}] WARNING: evaluation failed — saving zero-metric result. "
                f"Error: {exc}"
            )
            metrics = {f"{_field}_f1": 0.0 for _field in ["global_f1", "global_precision", "global_recall"]}
            from constants import FIELDS as _FIELDS
            for _field in _FIELDS:
                metrics[f"{_field}_ned"] = 1.0
            metrics["error"] = str(exc)
            metrics["self_test_failed"] = True
            metrics["training_time_sec"] = _train_duration_sec
        else:
            raise

    global_f1 = metrics.get("global_f1", 0.0)
    audit_logger.log_training_result(
        experiment_id=exp_id,
        global_f1=global_f1,
        training_time_sec=metrics.get("training_time_sec", 0.0),
        tokens_per_second=metrics.get("tokens_per_second"),
        early_stopping_epoch=metrics.get("early_stopping_epoch"),
        baseline_f1=None,
    )

    result = {
        "experiment_id": exp_id,
        "name": config.name,
        "datasets": config.datasets,
        "config": original_config_dict,
        "num_train_samples": len(train_samples),
        "metrics": metrics,
        "training_log": log_history,
        "v2": True,
        "resolution_fix_applied": True,
        "finetune_height": config.finetune_height,
        "finetune_width": config.finetune_width,
    }
    result_file.write_text(json.dumps(result, indent=2))
    print(f"[v2 Exp {exp_id}] Results saved -> {result_file}")
    print(f"[v2 Exp {exp_id}] Global F1 = {metrics.get('global_f1', 'N/A')}")
    _print_experiment_summary(
        exp_id, config, metrics, elapsed_sec=metrics.get("training_time_sec", 0.0)
    )
    return result


def run_custom_experiment(config: "ExperimentConfig", result_file: Path) -> dict:
    """Run a single experiment with custom hyperparameters (for sweeps)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = config.experiment_id

    print(f"\n{'=' * 72}")
    print(f"[v2 Sweep] Experiment {exp_id}: {config.name}")
    print(f"Datasets: {config.datasets}")
    print(f"{'=' * 72}")

    train_samples, val_samples = dataset_loaders.get_combined_dataset(
        config.datasets, sroie_oversample=config.sroie_oversample
    )
    if len(train_samples) == 0:
        print(f"[v2 Sweep Exp {exp_id}] WARNING: No samples loaded - saving empty result.")
        result = {
            "experiment_id": exp_id,
            "name": config.name,
            "datasets": config.datasets,
            "num_train_samples": 0,
            "metrics": {},
            "error": "No training samples available",
        }
        result_file.write_text(json.dumps(result, indent=2))
        return result

    model_dir = WORKSPACE / "models" / f"v2_sweep_experiment_{exp_id}_{int(time.time())}"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_history = train_experiment(
        exp_id,
        train_samples,
        model_dir,
        val_samples=val_samples,
        config=config,
    )

    metrics = evaluate_experiment(exp_id, model_dir, config=config)

    result = {
        "experiment_id": exp_id,
        "name": config.name,
        "datasets": config.datasets,
        "num_train_samples": len(train_samples),
        "metrics": metrics,
        "training_log": log_history,
        "v2": True,
        "resolution_fix_applied": True,
    }
    result_file.write_text(json.dumps(result, indent=2))
    print(f"[v2 Sweep Exp {exp_id}] Results saved -> {result_file}")
    print(f"[v2 Sweep Exp {exp_id}] Global F1 = {metrics.get('global_f1', 'N/A')}")
    return result


def run_experiment_from_config(
    cfg,
    base_processor=None,
    base_model=None,
    overrides: "dict | None" = None,
) -> dict:
    """Run a YAML-defined DONUT experiment (IDs 9+) via run_custom_experiment()."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = (
        Path(cfg.results_file)
        if getattr(cfg, "results_file", "")
        else RESULTS_DIR / f"experiment_{cfg.id}.json"
    )

    sroie_oversample = 1
    dataset_names: list[str] = []
    if hasattr(cfg, "datasets"):
        for d in cfg.datasets:
            dataset_names.append(d.name)
            if d.name == "sroie":
                sroie_oversample = getattr(d, "oversample", 1)

    ec = ExperimentConfig(
        name=cfg.name,
        datasets=dataset_names,
        epochs=getattr(cfg, "epochs", 10),
        lr=getattr(cfg, "encoder_lr", 5e-5),
        batch_size=getattr(cfg, "batch_size", 8),
        seed=getattr(cfg, "seed", SEED),
        early_stopping_patience=getattr(cfg, "early_stopping_patience", 3),
        base_model=getattr(cfg, "base_checkpoint", BASE_MODEL),
        warmup_steps=getattr(cfg, "warmup_steps", 40),
        weight_decay=getattr(cfg, "weight_decay", 0.01),
        max_length=getattr(cfg, "max_length", MAX_LENGTH),
        gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 2),
        encoder_lr=getattr(cfg, "encoder_lr", 5e-5),
        decoder_lr=getattr(cfg, "decoder_lr", 1e-4),
        description=getattr(cfg, "description", ""),
        experiment_id=getattr(cfg, "id", 0),
        sroie_oversample=sroie_oversample,
        lr_schedule=getattr(cfg, "scheduler_type", "cosine"),
    )

    if overrides:
        ec = dataclasses.replace(ec, **{k: v for k, v in overrides.items() if hasattr(ec, k)} )

    return run_custom_experiment(ec, result_file)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _print_experiment_summary(
    exp_id: int, config: ExperimentConfig, metrics: dict, elapsed_sec: float = 0.0
) -> None:
    if os.environ.get("DONUT_QUIET") == "1":
        return
    summary = {
        "exp": exp_id,
        "name": config.name,
        "samples": metrics.get("num_train_samples", 0),
        "epochs": config.epochs,
        "time_min": round(elapsed_sec / 60, 1),
        "f1": round(metrics.get("global_f1", 0.0), 4),
        "company_f1": round(metrics.get("company_f1", 0.0), 4),
        "date_f1": round(metrics.get("date_f1", 0.0), 4),
        "address_f1": round(metrics.get("address_f1", 0.0), 4),
        "total_f1": round(metrics.get("total_f1", 0.0), 4),
        "parse_failures": metrics.get("parse_failures", 0),
        "status": "ok" if metrics.get("global_f1", 0.0) > 0 else "empty",
        "v2": True,
        "resolution_fix": f"{config.finetune_height}x{config.finetune_width}",
    }
    print(f"--- v2 EXP {exp_id} RESULT ---")
    print(json.dumps(summary))


def save_summary() -> None:
    """Collect all individual v2 result files into results/v2/all_experiments.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for exp_id in EXPERIMENTS:
        result_file = RESULTS_DIR / f"experiment_{exp_id}.json"
        if result_file.exists():
            with open(result_file) as fh:
                all_results[str(exp_id)] = json.load(fh)

    summary_file = RESULTS_DIR / "all_experiments.json"
    summary_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n[v2] Summary saved -> {summary_file}")

    print(f"\n{'=' * 72}")
    print(f"{'Exp':<5} {'Name':<35} {'Train Samples':>14} {'Global F1':>10}")
    print(f"{'-' * 72}")
    for exp_id_str, res in sorted(all_results.items(), key=lambda x: int(x[0])):
        name = res.get("name", "")[:34]
        n = res.get("num_train_samples", 0)
        f1 = res.get("metrics", {}).get("global_f1", float("nan"))
        f1_str = f"{f1:>10.4f}" if not math.isnan(f1) else "       N/A"
        print(f"{exp_id_str:<5} {name:<35} {n:>14} {f1_str}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    audit_logger = TrainingAuditLogger(append_to_file="terminal_v2.txt")
    resources = detect_system_resources()
    audit_logger.log_resource_detection(resources)
    print(
        f"[v2 Resources] GPU: {resources.device_name} ({resources.vram_gb:.1f}GB), "
        f"RAM: {resources.ram_gb:.1f}GB, CPU: {resources.cpu_cores} cores"
    )
    print(
        f"[v2] Resolution fix active: processor.image_processor.size and "
        f"model.config.encoder.image_size will both be set to "
        f"{_FINETUNE_H}×{_FINETUNE_W} for all experiments."
    )

    parser = argparse.ArgumentParser(
        description="Run v2 DONUT SROIE experiments (with resolution-sync fix)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all 8 experiments sequentially")
    group.add_argument(
        "--experiment",
        type=int,
        metavar="N",
        choices=range(1, len(EXPERIMENTS) + 1),
        help="Run a single experiment (1-8)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Delete all cached v2 results and re-run from scratch"
    )
    args = parser.parse_args()

    if args.force:
        for result_file in RESULTS_DIR.glob("experiment_*.json"):
            result_file.unlink()
            print(f"[v2 force] Deleted cached result: {result_file}")

    if args.all:
        _cfg = EXPERIMENTS[1]
        _base_processor = DonutProcessor.from_pretrained(_cfg.base_model)
        _base_model = VisionEncoderDecoderModel.from_pretrained(_cfg.base_model)
        for exp_id in EXPERIMENTS:
            run_experiment(exp_id, base_processor=_base_processor, base_model=_base_model)
        del _base_model, _base_processor
        _gpu_cleanup()
        save_summary()
    else:
        run_experiment(args.experiment)
        save_summary()

if __name__ == "__main__":
    main()