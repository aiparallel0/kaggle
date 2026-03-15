# =============================================================================
# run_experiments.py
# Purpose: 8-experiment DONUT fine-tuning loop with OOM recovery and JSON result serialization
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-07
# =============================================================================
"""
run_experiments.py — Experiment orchestrator for 8 DONUT fine-tuning experiments.

This module is imported by run_all.py as part of the full dual-architecture
pipeline.  It can also be run standalone for DONUT-only experiment runs.

See also: run_all.py — the canonical single entry point that calls this module
          plus the TrOCR+YOLO stages, benchmark comparison, and paper generation.

Usage
-----
Run all experiments:
    python run_experiments.py --all

Run a single experiment (e.g., experiment 2):
    python run_experiments.py --experiment 2

Force re-run (ignore cached results):
    python run_experiments.py --all --force

Each experiment trains DONUT from the base checkpoint (donut-base) and evaluates on the
SROIE test set (63 images in test_img / test_key — custom 80/10/10 split
from 626 labeled training images; the official 347-image test set has no
public ground truth).  Results are saved to
results/experiment_N.json and a summary to results/all_experiments.json.

Architecture
------------
ExperimentConfig is THE single source of truth for all training hyperparameters.
No hardcoded epoch values, learning rates, or batch sizes exist outside of it.
DonutTrainer (from train.py) receives an ExperimentConfig and reads all
hyperparameters from it via duck-typed attribute access.
DonutEvaluator (from donut_evaluator.py) handles model loading with weight re-tying
and evaluation with self-test and parse-failure thresholds.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Set before torch initializes to reduce GPU memory fragmentation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

import dataset_loaders
import memory_manager as _mm

# FIX: Import shared constants from single source of truth (constants.py)
# instead of duplicating FIELDS/IMAGE_EXTS/etc. independently in this file.
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

# Phase 3-5: Dynamic resource optimization and audit logging
from resource_optimizer import (
    TrainingAuditLogger,
    detect_system_resources,
    optimize_hyperparams,
)
from train import MultiDataset  # moved to train.py

# --- Optional flash-attn probe (at module top, after imports) ---
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

# ---------------------------------------------------------------------------
# Logging Configuration — MUST be set before any third-party imports
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Suppress verbose third-party HTTP loggers to keep output clean
for pkg in ["httpx", "httpcore", "urllib3", "datasets", "transformers", "huggingface_hub"]:
    logging.getLogger(pkg).setLevel(logging.WARNING)

# Suppress PIL chunk-level DEBUG flood and other noisy third-party loggers
try:
    from logging_utils import suppress_noisy_loggers

    suppress_noisy_loggers()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")

# ---------------------------------------------------------------------------
# v2 resolution-sync constants and helper
# ---------------------------------------------------------------------------

# Canonical fine-tuning resolution for DONUT on SROIE.
# Height × Width must be multiples of 32 (patch_size=4, Swin stride=8 → 32).
# 1280×960 is the standard community fine-tuning resolution that fits
# comfortably in 24 GB VRAM at batch_size=8.
_FINETUNE_H: int = 1280  # height (tall receipts)
_FINETUNE_W: int = 960   # width


def _apply_resolution_sync(
    processor,
    model,
    height: int = _FINETUNE_H,
    width: int = _FINETUNE_W,
) -> None:
    """Synchronise processor image size and Swin encoder image_size.

    The ``naver-clova-ix/donut-base`` pretrained checkpoint stores
    ``model.config.encoder.image_size = [2560, 1920]``.  When fine-tuning
    at 1280×960, the processor resizes images to 1280×960 but the encoder's
    positional embeddings remain anchored to the 2560×1920 grid.  HuggingFace
    interpolates the mismatched embeddings silently, suppressing F1 for
    spatially distributed fields (company, address).

    This function sets BOTH fields atomically so they agree before any forward
    pass or gradient computation occurs.

    Side Effects
    ------------
    * ``processor.image_processor.size`` → ``{"height": height, "width": width}``.
    * ``model.config.encoder.image_size`` → ``[height, width]``

    Both mutations happen in-place.  Raises ``AssertionError`` if the fields
    still disagree after the update.
    """
    processor.image_processor.size = {"height": height, "width": width}
    model.config.encoder.image_size = [height, width]

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
    """Single source of truth for all training hyperparameters.

    Every training parameter (epochs, lr, batch_size, etc.) lives here.
    DonutTrainer reads them via duck-typed attribute access using the
    property aliases below (max_epochs, learning_rate, etc.).
    """

    name: str
    datasets: list[str]
    epochs: int = (
        10  # Per CLAUDE.md Section 3: optimal is 10 epochs. >12 causes overfit on 63-sample val set
    )
    lr: float = 5e-5
    batch_size: int = 8
    seed: int = SEED
    early_stopping_patience: int = 3  # Per CLAUDE.md: patience=3 is optimal. Prevents overfitting.
    base_model: str = BASE_MODEL
    warmup_steps: int = 40  # Fix: 500 exceeded total steps for small datasets (~312 for Exp 1); 40 is safe across all 8 experiments
    weight_decay: float = 0.01
    max_length: int = MAX_LENGTH
    gradient_accumulation_steps: int = 2
    encoder_lr: float = 5e-5
    decoder_lr: float = 1e-4
    description: str = ""
    experiment_id: int = 0
    sroie_oversample: int = 1  # Number of times to duplicate SROIE training samples (1–3 typical)
    skip_oversample_guard: bool = (
        False  # Set True for intentional naïve control experiments (Exps 2–4).
    )
    # These exist to prove that sroie_oversample=1 hurts F1 vs sroie_oversample>=2.
    # The guard is still enforced for all other paths (custom runs, CLI, etc.).

    # -- Mini-mode accelerators (all default to off so normal runs are unaffected) --
    subsample_train: int = 0  # >0: cap training set to this many samples (mini only)
    subsample_eval: int = 0  # >0: cap test set to this many samples (mini only)
    skip_step_validation: bool = False  # bypass 200-step minimum guard (mini only)
    lr_schedule: str = "cosine"  # "cosine" | "one_cycle" | "linear"
    optimizer_type: str = "adamw"  # "adamw" | "sgd" (sgd = SGD + Nesterov)

    # -- v2: explicit fine-tuning resolution (height × width) -------------
    # When non-default, _apply_resolution_sync() is called before training
    # to align processor.image_processor.size and model.config.encoder.image_size.
    # Both must be multiples of 32 (patch_size=4, Swin stride=8 → 32).
    finetune_height: int = _FINETUNE_H
    finetune_width: int = _FINETUNE_W

    # -- YAML-sourced config compatibility fields -------------------------
    # These fields are present in experiment_config_loader.ExperimentConfig and
    # needed by hparam_search.py, multi_seed_runner.py, and dag_scheduler.py
    # when they call dataclasses.replace() on configs from either class.
    arch_type: str = "donut"
    is_zero_shot: bool = False
    depends_on: list = field(default_factory=list)
    full_parameter_finetuning: bool = True
    image_height: int = 1280
    image_width: int = 960
    allow_high_res: bool = False

    # -- Duck-typed aliases for DonutTrainer compatibility ----------------
    # DonutTrainer reads config.max_epochs, config.learning_rate, etc.
    # These properties ensure a single source of truth (no duplication).

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
        """Default output directory; overridden at call site when needed."""
        return str(WORKSPACE / "models" / f"experiment_{self.experiment_id}")

    @property
    def id(self) -> int:
        """Alias for experiment_id — used by dag_scheduler.py and run_all.py dispatch."""
        return self.experiment_id

    @property
    def dataset_names(self) -> list[str]:
        """Names of all datasets in this experiment.

        Returns datasets directly since they are already strings in this class.
        Provides a consistent interface with experiment_config_loader.ExperimentConfig
        which stores DatasetEntry objects instead.
        """
        return self.datasets

    @property
    def base_checkpoint(self) -> str:
        """Alias for base_model — used by experiment_config_loader code paths."""
        return self.base_model


# ---------------------------------------------------------------------------
# Experiment definitions
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
        description="Naïve multi-dataset run — sroie_oversample=1 intentionally. "
        "Control group: proves auxiliary data dilutes F1 without oversampling.",
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
        description="SROIE + WildReceipt with 2x SROIE oversampling to preserve field coverage.",
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
        description="All datasets with 2x SROIE oversampling to counteract dilution.",
        epochs=15,
        sroie_oversample=2,
        experiment_id=7,
    ),
    8: ExperimentConfig(
        name="SROIE + All (3x SROIE)",
        datasets=["sroie", "wildreceipt", "invoices_donut"],
        description="All datasets with 3x SROIE oversampling for maximum field coverage.",
        epochs=15,
        sroie_oversample=3,
        experiment_id=8,
    ),
}

# ---------------------------------------------------------------------------
# TRAIN_CONFIG — cache validation for result JSON files
# ---------------------------------------------------------------------------
# FIX: Previously only included 5 of 9 hyperparameters (max_epochs, lr,
# batch_size, early_stopping_patience, base_model).  If warmup_steps,
# weight_decay, max_length, or seed changed, the cache validation at
# cached.get("config") != TRAIN_CONFIG would NOT detect stale results.
# Now includes ALL hyperparameters from ExperimentConfig for complete
# staleness detection.

_default_config = ExperimentConfig(name="", datasets=[])

TRAIN_CONFIG: dict[str, Any] = {
    "max_epochs": _default_config.epochs,
    "learning_rate": _default_config.lr,
    "per_device_train_batch_size": _default_config.batch_size,
    "early_stopping_patience": _default_config.early_stopping_patience,
    "base_model": _default_config.base_model,
    # FIX: These were previously missing, causing stale cache hits when
    # warmup_steps/weight_decay/max_length/seed changed.
    "warmup_steps": _default_config.warmup_steps,
    "weight_decay": _default_config.weight_decay,
    "max_length": _default_config.max_length,
    "seed": _default_config.seed,
    "gradient_accumulation_steps": _default_config.gradient_accumulation_steps,
    # v2: include resolution so changing finetune resolution correctly
    # invalidates previously cached results.
    "finetune_height": _default_config.finetune_height,
    "finetune_width": _default_config.finetune_width,
}


def _config_to_dict(config: "ExperimentConfig") -> dict:
    """Serialize an ExperimentConfig to the TRAIN_CONFIG dict format.

    Used to record the *actual* training hyperparameters in the result JSON,
    so cache validation compares against what was truly used.
    """
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
# Training — delegates to DonutTrainer from train.py
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

    All hyperparameters come from *config* if supplied, otherwise from
    ``EXPERIMENTS[exp_id]``.  Pass a custom ExperimentConfig for sweeps.
    Training is delegated to ``DonutTrainer`` from ``train.py``, which reads
    hyperparameters from the config via duck-typed attributes.

    When *base_processor* and *base_model* are supplied (pre-loaded by the
    caller), they are deep-copied from RAM instead of re-deserializing 800 MB
    of safetensors from disk — reducing ~4–6 s of ``from_pretrained`` overhead
    per experiment to a fast in-memory copy.

    Returns the trainer log history (list of per-step dicts) for convergence
    plot generation.
    """
    from train import DonutTrainer

    if config is None:
        config = EXPERIMENTS[exp_id]

    set_seed(config.seed)
    print(f"\n[Exp {exp_id}] Training on {len(samples)} samples -> {output_dir}")
    print(
        f"[Exp {exp_id}] Hyperparams: epochs={config.epochs}, "
        f"lr={config.lr}, batch_size={config.batch_size}, "
        f"warmup={config.warmup_steps}, wd={config.weight_decay}"
    )
    if val_samples:
        print(f"[Exp {exp_id}] Validation set: {len(val_samples)} samples")

    # ---------------------------------------------------------------------------
    # Inner helper: build a fresh model, processor, and datasets.
    # Called once initially and again on each OOM retry so that GPU-resident
    # tensors from the failed attempt are never referenced on the retry.
    # ---------------------------------------------------------------------------
    def _build_model_and_datasets():
        if base_processor is not None and base_model is not None:
            _proc = copy.deepcopy(base_processor)
            _mdl = copy.deepcopy(base_model)
        else:
            _proc = DonutProcessor.from_pretrained(config.base_model)
            # Attempt Flash Attention 2 (requires flash-attn package; speeds up
            # decoder attention and reduces VRAM, enabling larger batch sizes).
            # Falls back silently to eager attention if unavailable or unsupported.
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

        # v2: Synchronise processor and encoder image sizes when resolution fields
        # are set (non-default values trigger the sync; default _FINETUNE_H/_FINETUNE_W
        # values always apply the sync for correctness).
        _apply_resolution_sync(
            _proc, _mdl,
            height=config.finetune_height,
            width=config.finetune_width,
        )

        # Add SROIE special tokens with diagnostic logging (Phase 0a)
        logger.debug("[Pre-resize] Tokenizer vocab size: %d", len(_proc.tokenizer))
        logger.debug(
            "[Pre-resize] Decoder embed_tokens shape: %s",
            _mdl.decoder.model.decoder.embed_tokens.weight.shape,
        )

        _proc.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
        _mdl.decoder.resize_token_embeddings(len(_proc.tokenizer))

        logger.debug("[Post-resize] Tokenizer vocab size: %d", len(_proc.tokenizer))
        logger.debug(
            "[Post-resize] Decoder embed_tokens shape: %s",
            _mdl.decoder.model.decoder.embed_tokens.weight.shape,
        )
        logger.debug("[Post-resize] Decoder lm_head shape: %s", _mdl.decoder.lm_head.weight.shape)

        # Verify NEW_TOKENS were added to tokenizer (Phase 0a diagnostic)
        for token in NEW_TOKENS:
            token_ids = _proc.tokenizer.encode(token, add_special_tokens=False)
            if not token_ids or len(token_ids) > 1:
                logger.error(
                    "CRITICAL: Token %s not in vocab or tokenizes to multiple IDs: %s",
                    token,
                    token_ids,
                )
                raise RuntimeError(f"Token addition failed for {token}; vocab may be corrupted")
        logger.debug(
            "[Token-verify] All %d SROIE tokens successfully added to vocab", len(NEW_TOKENS)
        )

        # After resize, embed_tokens and lm_head are separate tensors with
        # independent random init for the new tokens.  Set tie_word_embeddings=False
        # so save_pretrained() saves BOTH weights independently.  Without this,
        # the saved checkpoint omits lm_head (or tie_weights() overwrites the
        # learned lm_head with embed_tokens), causing F1=0 on reload.
        _mdl.decoder.config.tie_word_embeddings = False

        _mdl.config.pad_token_id = _proc.tokenizer.pad_token_id
        _mdl.decoder.config.pad_token_id = _proc.tokenizer.pad_token_id
        _mdl.config.decoder_start_token_id = _proc.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        _mdl.decoder.config.decoder_start_token_id = _proc.tokenizer.convert_tokens_to_ids(
            ["<s_sroie>"]
        )[0]
        # ── Guardrail: verify decoder_start_token_id decodes back to the task token ──
        _decoded = _proc.tokenizer.decode([_mdl.config.decoder_start_token_id])
        if _decoded != "<s_sroie>":
            raise RuntimeError(
                f"decoder_start_token_id={_mdl.config.decoder_start_token_id} decodes to "
                f"'{_decoded}', not '<s_sroie>'. Token was not added to vocab before "
                f"convert_tokens_to_ids was called, or the list-wrapping syntax is missing. "
                f"Use: tokenizer.convert_tokens_to_ids(['<s_sroie>'])[0]"
            )
        # v2: set max_length on both model config and generation config consistently.
        _mdl.config.max_length = MAX_LENGTH
        if hasattr(_mdl, "generation_config"):
            _mdl.generation_config.max_new_tokens = MAX_LENGTH

        # Only enable gradient checkpointing when VRAM is constrained (< 24 GB).
        # 24 GB covers RTX 3090/4090 (24 GB) and below, where activation memory
        # during DONUT's backward pass (~3.5 GB at batch_size=8) is a real constraint.
        # On high-VRAM cards (RTX PRO 6000 = 95.6 GB, A100 = 80 GB, RTX 4090 = 24 GB
        # boundary) gradient checkpointing adds ~30-40% backward-pass overhead for
        # zero memory benefit — so disable it there.
        _GRAD_CKPT_VRAM_THRESHOLD_GB = 24.0
        _enable_grad_ckpt = True  # default: enable for safety on unknown hardware
        if torch.cuda.is_available():
            try:
                _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if _vram_gb > _GRAD_CKPT_VRAM_THRESHOLD_GB:
                    _enable_grad_ckpt = False
                    logger.info(
                        "[GradCkpt] Disabled — VRAM=%.1f GB > %.0f GB threshold "
                        "(saves ~35%% backward time at no memory cost)",
                        _vram_gb,
                        _GRAD_CKPT_VRAM_THRESHOLD_GB,
                    )
                else:
                    logger.info(
                        "[GradCkpt] Enabled — VRAM=%.1f GB <= %.0f GB threshold",
                        _vram_gb,
                        _GRAD_CKPT_VRAM_THRESHOLD_GB,
                    )
            except Exception as exc:
                logger.warning("[GradCkpt] VRAM detection failed (%s) — defaulting to enabled", exc)

        if _enable_grad_ckpt:
            _mdl.config.use_cache = False  # Required with gradient_checkpointing
            _mdl.decoder.config.use_cache = False
            _mdl.gradient_checkpointing_enable()
        else:
            # use_cache=True is the default and correct when not using grad checkpointing
            _mdl.config.use_cache = True
            _mdl.decoder.config.use_cache = True

        # v2: freeze only Swin stage-0 (v1 froze stages 0+1).
        # Freezing fewer encoder layers lets the model adapt more freely to
        # SROIE's Southeast Asian receipt layouts while still preserving the
        # low-level patch embeddings from pre-training.
        frozen_count = 0
        for name, param in _mdl.encoder.named_parameters():
            if "layers.0" in name:
                param.requires_grad = False
                frozen_count += 1
        if frozen_count:
            logger.info("[FreezeEncoder] Frozen Swin stage-0 only (%d parameters)", frozen_count)

        # Build PyTorch datasets
        _train_ds = MultiDataset(samples, _proc, max_length=config.max_length)
        _val_ds = (
            MultiDataset(
                val_samples,
                _proc,
                max_length=config.max_length,
                precompute_tensors=False,  # val set never needs pixel tensor precompute
            )
            if val_samples
            else None
        )

        return _proc, _mdl, _train_ds, _val_ds

    processor, model, train_ds, val_ds = _build_model_and_datasets()

    # Verify single source of truth: ExperimentConfig properties map correctly
    assert config.epochs == config.max_epochs, (
        f"Single source of truth violation: epochs={config.epochs} "
        f"!= max_epochs={config.max_epochs}"
    )
    assert config.lr == config.learning_rate, (
        f"Single source of truth violation: lr={config.lr} != learning_rate={config.learning_rate}"
    )
    assert config.batch_size == config.per_device_train_batch_size, (
        f"Single source of truth violation: batch_size={config.batch_size} "
        f"!= per_device_train_batch_size={config.per_device_train_batch_size}"
    )

    # Create DonutTrainer — it reads all hyperparams from config
    trainer = DonutTrainer(
        config=config,
        processor=processor,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    # Warn early if another GPU process is consuming VRAM — this can cause OOM
    # mid-epoch, wasting all setup time. Firing the warning before training
    # allows the user to kill the rogue process before the experiment starts.
    if torch.cuda.is_available():
        try:
            _free_vram, _total_vram = torch.cuda.mem_get_info()
            _used_by_others = _total_vram - _free_vram - torch.cuda.memory_allocated()
            if _used_by_others > 1 * 1024**3:  # > 1 GB held by external processes
                logger.warning(
                    "[VRAM] External process(es) occupying ~%.1f GB of GPU memory "
                    "(%s free of %s total). Risk of OOM during training. "
                    "Run `nvidia-smi` and kill any unnecessary GPU processes.",
                    _used_by_others / 1024**3,
                    f"{_free_vram / 1024**3:.1f} GB",
                    f"{_total_vram / 1024**3:.1f} GB",
                )
        except Exception:
            pass

    # Train with progressive OOM recovery: halve batch_size (8→4→2→1) on each
    # CUDA OOM, doubling gradient_accumulation_steps to keep the effective batch
    # the same, then fully rebuilding the model and datasets from scratch each
    # time so the retry starts on a clean, defragmented GPU.
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
                    f"[Exp {exp_id}] CUDA OOM — reducing batch_size to "
                    f"{config.batch_size}, increasing grad_accum to "
                    f"{config.gradient_accumulation_steps} and retrying"
                )
                # Delete ALL GPU-resident objects so the retry starts on a
                # clean, defragmented GPU — not just the trainer wrapper.
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
                    torch.cuda.synchronize()  # Ensure all CUDA ops complete before freeing
                torch.cuda.empty_cache()
                # Re-create everything from the base model to avoid inheriting
                # any gradient state from the failed attempt.
                processor, model, train_ds, val_ds = _build_model_and_datasets()
                trainer = DonutTrainer(
                    config=config,
                    processor=processor,
                    model=model,
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                )
            else:
                raise RuntimeError(
                    f"[Exp {exp_id}] CUDA OOM: batch_size already at minimum (1); "
                    "cannot recover without further hardware constraints"
                ) from e

    # Save model with tied weights
    trainer.save(output_dir)

    # Post-save spot-check: verify <s_sroie> token survives processor serialization
    _spot_proc = DonutProcessor.from_pretrained(str(output_dir))
    _spot_id = _spot_proc.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    _spot_decoded = _spot_proc.tokenizer.decode([_spot_id])
    if _spot_decoded == "<s_sroie>":
        print(
            f"[Exp {exp_id}] Processor spot-check PASS: "
            f"<s_sroie> → id={_spot_id} → '{_spot_decoded}'"
        )
    else:
        print(
            f"[Exp {exp_id}] Processor spot-check FAIL: "
            f"<s_sroie> → id={_spot_id} → '{_spot_decoded}' "
            f"(unk_token_id={_spot_proc.tokenizer.unk_token_id}) — "
            "processor is corrupt, evaluation will produce F1=0.0"
        )

    print(
        f"[Exp {exp_id}] Training complete "
        f"(duration={result.duration_seconds:.1f}s, "
        f"train={result.train_samples}, val={result.val_samples})"
    )

    # FIX: Explicit GPU cleanup between experiments to prevent OOM on GPUs
    # with limited VRAM.  The RTX 4090 has 24GB — sufficient for DONUT but
    # running 8+ experiments sequentially without cleanup risks fragmentation.
    # NOTE: local references MUST be deleted before _gpu_cleanup() is called;
    # passing them as arguments to _gpu_cleanup() is a no-op for freeing memory.
    # log_history is a plain list of dicts — a detached copy not tied to the
    # trainer's internals, so it's safe to capture before deleting the trainer.
    log_history = result.log_history
    # Explicitly clear dataset caches before deleting references — prevents
    # _pixel_cache tensors (up to 7.1 GB) from staying pinned until GC decides to run.
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
        torch.cuda.synchronize()  # Ensure all CUDA ops complete before freeing
    torch.cuda.empty_cache()
    print(f"[Exp {exp_id}] GPU memory released")

    return log_history


# ---------------------------------------------------------------------------
# Evaluation — delegates to DonutEvaluator from donut_evaluator.py
# ---------------------------------------------------------------------------


def evaluate_experiment(
    exp_id: int, model_dir: Path, config: "ExperimentConfig | None" = None
) -> dict:
    """Evaluate a fine-tuned model (at *model_dir*) on the SROIE test set.

    Uses DonutEvaluator from donut_evaluator.py which handles:
      - Weight re-tying via load_model_with_tied_weights (fixes lm_head bug)
      - Self-test before full evaluation
      - Parse failure threshold checking

    Args:
        exp_id: Experiment ID (for logging; only used if config is None)
        model_dir: Path to fine-tuned model directory
        config: Optional ExperimentConfig; if None, loads from EXPERIMENTS[exp_id]
    """
    from donut_evaluator import DonutEvaluator

    if config is None:
        config = EXPERIMENTS[exp_id]
    test_samples = dataset_loaders.load_sroie_test()

    # Micro subsample evaluation — reduce test-set size for fast smoke tests
    if getattr(config, "subsample_eval", 0) > 0 and len(test_samples) > config.subsample_eval:
        import random as _rnd

        _rng = _rnd.Random(getattr(config, "seed", SEED))
        test_samples = _rng.sample(test_samples, config.subsample_eval)
        print(f"[Exp {exp_id}] subsample_eval: evaluating on {len(test_samples)} test samples")
    else:
        print(f"[Exp {exp_id}] Evaluating on {len(test_samples)} SROIE test images")

    # Load processor from the fine-tuned model directory
    processor = DonutProcessor.from_pretrained(str(model_dir))

    # Create evaluator — handles model loading with weight re-tying
    evaluator = DonutEvaluator(
        model_path=model_dir,
        processor=processor,
        test_dataset=test_samples,
        task_prompt="<s_sroie>",
        max_length=config.max_length,
        device=DEVICE,
    )

    # Run evaluation (includes self-test + parse failure threshold).
    # allow_high_parse_failures=True so undertrained models that haven't
    # converged to the SROIE tag format record F1=0.0 instead of crashing.
    eval_result = evaluator.evaluate(allow_high_parse_failures=True)
    metrics = eval_result.to_dict()

    if eval_result.parse_failures > 0:
        print(
            f"[Exp {exp_id}] WARNING: {eval_result.parse_failures} of "
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
    """Run a single experiment: train, evaluate, save results.

    Checks cache validity (datasets AND hyperparams must match) before
    reusing a previous result.  Loads data via dataset_loaders, delegates
    training to train_experiment and evaluation to evaluate_experiment,
    then saves the result JSON.

    When *base_processor* and *base_model* are supplied, they are passed to
    train_experiment() which deep-copies them instead of re-loading from disk.

    When *overrides* is a non-empty dict, those key/value pairs are applied
    to the base ExperimentConfig via dataclasses.replace() before training.
    This allows ``--param`` CLI overrides without mutating the global EXPERIMENTS
    dict (GP-1).
    """
    import dataclasses as _dc

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if exp_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment ID {exp_id}. Valid: {list(EXPERIMENTS)}")

    config = EXPERIMENTS[exp_id]
    if overrides:
        config = _dc.replace(config, **overrides)
        print(f"[Exp {exp_id}] --param overrides applied: {overrides}")
    print(f"\n{'=' * 72}")
    print(f"Experiment {exp_id}: {config.name}")
    print(f"Description: {config.description}")
    print(f"Datasets: {config.datasets}")
    print(f"{'=' * 72}")

    # Validate that multi-dataset runs use sroie_oversample >= 2.
    # Without 2× oversampling, auxiliary data dilutes the SROIE training signal
    # and causes F1 to fall at or below the single-dataset baseline (CLAUDE.md §8).
    # skip_oversample_guard=True is set only for intentional naïve control experiments
    # (Exps 2–4) that deliberately use sroie_oversample=1 to prove dilution.
    validate_sroie_oversample(
        config.datasets,
        config.sroie_oversample,
        skip_guard=getattr(config, "skip_oversample_guard", False),
    )

    result_file = RESULTS_DIR / f"experiment_{exp_id}.json"

    # Check if already done — validate cached result matches current experiment
    # definition (datasets AND hyperparameters) before reusing.
    # Compare against the original (unoptimized) experiment config so that
    # cache hits are hardware-independent: resource optimization is deterministic
    # for the same hardware, so if the experiment definition hasn't changed the
    # cached result is still valid.
    original_config_dict = _config_to_dict(EXPERIMENTS[exp_id])
    if result_file.exists():
        with open(result_file) as fh:
            cached = json.load(fh)
        if (
            cached.get("datasets") != config.datasets
            or cached.get("config") != original_config_dict
        ):
            # NOTE: JSON round-trip preserves numeric equality for floats like 5e-5,
            # so this comparison is safe (5e-5 == 5e-05 after json.load).
            print(
                f"[Exp {exp_id}] STALE result detected (datasets or config mismatch). "
                "Deleting and re-running."
            )
            result_file.unlink()
        else:
            print(f"[Exp {exp_id}] Valid cached result found - skipping.")
            return cached

    # Load data
    train_samples, val_samples = dataset_loaders.get_combined_dataset(
        config.datasets, sroie_oversample=config.sroie_oversample
    )

    # Micro/mini subsample — deterministic RNG so repeated runs give the same split
    if getattr(config, "subsample_train", 0) > 0 and len(train_samples) > config.subsample_train:
        import random as _rnd

        _rng = _rnd.Random(config.seed)
        train_samples = _rng.sample(train_samples, config.subsample_train)
        print(f"[Exp {exp_id}] subsample_train: using {len(train_samples)} samples")

    if len(train_samples) == 0:
        print(f"[Exp {exp_id}] WARNING: No samples loaded - saving empty result.")
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

    # Phase 5: Dynamic resource optimization (Phase 3-5)
    # Detect available hardware and optimize hyperparameters accordingly
    resources = detect_system_resources()
    optimized_config = optimize_hyperparams(
        num_train_samples=len(train_samples),
        available_vram_gb=resources.vram_gb,
        available_ram_gb=resources.ram_gb,
    )

    # Log the config decision to terminal.txt for audit trail
    audit_logger = TrainingAuditLogger(append_to_file="terminal.txt")
    audit_logger.log_config_decision(exp_id, optimized_config)

    # Apply the optimized values to an isolated copy of the experiment config using
    # dataclasses.replace() so the global EXPERIMENTS dict is never mutated.
    # Only batch_size and gradient_accumulation_steps are overridden; epochs and
    # warmup_steps remain fixed per CLAUDE.md experiment design.
    #
    # Bug #2 fix: micro/mini mode sets gradient_accumulation_steps deliberately
    # (e.g. accum=1 for OneCycleLR "every batch → immediate optimizer step").
    # The resource optimizer must NOT override it — doing so cuts optimizer steps
    # in half (76 instead of 150 for micro), preventing any structural learning.
    # skip_step_validation=True is the canonical micro/mini mode signal.
    _is_micro = getattr(config, "skip_step_validation", False)
    old_batch = config.batch_size
    old_accum = config.gradient_accumulation_steps
    config = dataclasses.replace(
        EXPERIMENTS[exp_id],
        batch_size=optimized_config.batch_size,
        gradient_accumulation_steps=(
            EXPERIMENTS[exp_id].gradient_accumulation_steps  # preserve micro/mini accum
            if _is_micro
            else optimized_config.gradient_accumulation_steps
        ),
        encoder_lr=optimized_config.encoder_lr,
        decoder_lr=optimized_config.decoder_lr,
    )
    print(
        f"[Exp {exp_id}] Resource optimization applied: "
        f"batch_size {old_batch} → {config.batch_size}, "
        f"grad_accum {old_accum} → {config.gradient_accumulation_steps}"
        + (
            " (grad_accum preserved: micro/mini mode)"
            if _is_micro
            else f" ({optimized_config.config_explanation})"
        )
    )

    # ── Guardrail: verify global EXPERIMENTS dict was NOT mutated ──────────
    if (
        config.batch_size != EXPERIMENTS[exp_id].batch_size
        or config.gradient_accumulation_steps != EXPERIMENTS[exp_id].gradient_accumulation_steps
    ):
        assert config is not EXPERIMENTS[exp_id], (
            "INVARIANT VIOLATION: config is the same object as EXPERIMENTS[exp_id] "
            "after the dataclasses.replace() call. This indicates a logic error in the "
            "override block — the replace() result was not assigned back to config."
        )

    # ── Guardrail: validate optimizer step count ───────────────────────────
    # skip_step_validation=True is set only by micro/mini modes where a small
    # dataset + few epochs is intentional (smoke-test, not full convergence).
    from resource_optimizer import validate_training_config

    if not getattr(config, "skip_step_validation", False):
        validate_training_config(
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_samples=len(train_samples),
            epochs=config.epochs,
        )
    else:
        print(f"[Exp {exp_id}] step-count validation skipped (micro/mini mode)")

    # Train — pass config explicitly so train_experiment uses the optimized values
    model_dir = WORKSPACE / "models" / f"experiment_{exp_id}"
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

    # Evaluate
    try:
        metrics = evaluate_experiment(exp_id, model_dir)
        metrics["training_time_sec"] = _train_duration_sec
    except RuntimeError as exc:
        # Catch self-test failures and parse-failure-threshold errors for any
        # run type (not just micro/mini).  An undertrained full-run model that
        # hasn't converged to the SROIE tag format should record F1=0.0 and
        # let the remaining experiments continue, not crash the pipeline.
        if "Self-test FAILED" in str(exc) or "Parse failure threshold exceeded" in str(exc):
            print(
                f"[Exp {exp_id}] WARNING: evaluation failed (undertrained model) — "
                f"saving zero-metric result. Error: {exc}"
            )
            metrics = {f: 0.0 for f in ["global_f1", "global_precision", "global_recall"]}
            # Include per-field zeros so downstream paper-generation code doesn't KeyError
            from constants import FIELDS as _FIELDS

            for _field in _FIELDS:
                metrics[f"{_field}_f1"] = 0.0
                metrics[f"{_field}_ned"] = 1.0  # NED=1.0 means maximum edit distance
            metrics["error"] = str(exc)
            metrics["self_test_failed"] = True
            metrics["training_time_sec"] = _train_duration_sec
        else:
            raise

    # Phase 5: Log training result to audit trail
    global_f1 = metrics.get("global_f1", 0.0)
    audit_logger.log_training_result(
        experiment_id=exp_id,
        global_f1=global_f1,
        training_time_sec=metrics.get("training_time_sec", 0.0),
        tokens_per_second=metrics.get("tokens_per_second"),
        early_stopping_epoch=metrics.get("early_stopping_epoch"),
        baseline_f1=None,  # Could set to pretrained F1 for comparison
    )

    # Save result — use the *original* (unoptimized) experiment config dict for cache
    # staleness comparison so future runs can correctly detect stale results.
    # The resource-optimized values (batch_size, encoder_lr, etc.) are hardware-dependent
    # and would cause spurious cache misses on different hardware.
    result = {
        "experiment_id": exp_id,
        "name": config.name,
        "datasets": config.datasets,
        "config": original_config_dict,
        "num_train_samples": len(train_samples),
        "metrics": metrics,
        "training_log": log_history,
    }
    result_file.write_text(json.dumps(result, indent=2))
    print(f"[Exp {exp_id}] Results saved -> {result_file}")
    print(f"[Exp {exp_id}] Global F1 = {metrics.get('global_f1', 'N/A')}")
    _print_experiment_summary(
        exp_id, config, metrics, elapsed_sec=metrics.get("training_time_sec", 0.0)
    )
    return result


def run_custom_experiment(config: "ExperimentConfig", result_file: Path) -> dict:
    """Run a single experiment with custom hyperparameters (for sweeps).

    Similar to run_experiment but:
    - Takes a custom ExperimentConfig instead of looking up EXPERIMENTS[exp_id]
    - Saves result to custom result_file instead of results/experiment_N.json
    - Does not use caching (always re-runs)
    - Does not check EXPERIMENTS for validity

    Args:
        config: Custom ExperimentConfig with all hyperparameters
        result_file: Path to save results JSON

    Returns:
        Result dict with metrics, training_log, etc.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = config.experiment_id

    print(f"\n{'=' * 72}")
    print(f"Sweep Experiment {exp_id}: {config.name}")
    print(f"Datasets: {config.datasets}")
    print(f"Batch size={config.batch_size}, epochs={config.epochs}, lr={config.lr:.0e}")
    print(f"{'=' * 72}")

    # Load data
    train_samples, val_samples = dataset_loaders.get_combined_dataset(
        config.datasets, sroie_oversample=config.sroie_oversample
    )
    if len(train_samples) == 0:
        print(f"[Sweep Exp {exp_id}] WARNING: No samples loaded - saving empty result.")
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

    # Train
    model_dir = WORKSPACE / "models" / f"sweep_experiment_{exp_id}_{int(time.time())}"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_history = train_experiment(
        exp_id,
        train_samples,
        model_dir,
        val_samples=val_samples,
        config=config,
    )

    # Evaluate (pass custom config to avoid EXPERIMENTS lookup)
    metrics = evaluate_experiment(exp_id, model_dir, config=config)

    # Save result
    result = {
        "experiment_id": exp_id,
        "name": config.name,
        "datasets": config.datasets,
        "num_train_samples": len(train_samples),
        "metrics": metrics,
        "training_log": log_history,
    }
    result_file.write_text(json.dumps(result, indent=2))
    print(f"[Sweep Exp {exp_id}] Results saved -> {result_file}")
    print(f"[Sweep Exp {exp_id}] Global F1 = {metrics.get('global_f1', 'N/A')}")
    return result


def run_experiment_from_config(
    cfg,
    base_processor=None,
    base_model=None,
    overrides: "dict | None" = None,
) -> dict:
    """Run a YAML-defined DONUT experiment (IDs 9+) via run_custom_experiment().

    Bridges the ``experiment_config_loader.ExperimentConfig`` object (YAML-sourced)
    to the ``run_experiments.ExperimentConfig`` dataclass expected by
    ``run_custom_experiment()``.  Training logic is NOT duplicated — all training
    is handled by the existing ``run_custom_experiment`` / ``train_experiment`` pair.

    Args:
        cfg: An ``experiment_config_loader.ExperimentConfig`` instance (YAML-loaded).
        base_processor: Optional pre-loaded DonutProcessor for reuse across experiments.
            NOTE: run_custom_experiment does not currently accept base_processor;
            the arg is accepted here for API compatibility but ignored.
        base_model: Optional pre-loaded base model.  Same caveat as base_processor.
        overrides: Optional dict of ExperimentConfig field overrides (e.g.
            ``{"epochs": 5}``).  Applied via ``dataclasses.replace()`` so the
            input cfg is never mutated.

    Returns:
        Result dict with experiment_id, name, datasets, num_train_samples, metrics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = (
        Path(cfg.results_file)
        if getattr(cfg, "results_file", "")
        else RESULTS_DIR / f"experiment_{cfg.id}.json"
    )

    # Compute sroie_oversample from per-dataset oversampling in YAML config.
    # The YAML specifies oversampling per DatasetEntry (.oversample field).
    # The ExperimentConfig uses a single sroie_oversample int for SROIE.
    sroie_oversample = 1
    dataset_names: list[str] = []
    if hasattr(cfg, "datasets"):
        for d in cfg.datasets:
            dataset_names.append(d.name)
            if d.name == "sroie":
                sroie_oversample = getattr(d, "oversample", 1)

    # Build a run_experiments.ExperimentConfig from the YAML cfg fields
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
        ec = dataclasses.replace(ec, **{k: v for k, v in overrides.items() if hasattr(ec, k)})

    return run_custom_experiment(ec, result_file)


# ---------------------------------------------------------------------------
# Compact structured summary helpers
# ---------------------------------------------------------------------------


def _print_experiment_summary(
    exp_id: int, config: ExperimentConfig, metrics: dict, elapsed_sec: float = 0.0
) -> None:
    """Print a compact one-line JSON summary for the experiment result.

    Format is AI-agent-friendly: machine-readable, minimal tokens, single line.
    """
    # Respect DONUT_QUIET env var — suppress if quiet mode is active
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
    }
    print(f"--- EXP {exp_id} RESULT ---")
    print(json.dumps(summary))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def save_summary() -> None:
    """Collect all individual result files into results/all_experiments.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for exp_id in EXPERIMENTS:
        result_file = RESULTS_DIR / f"experiment_{exp_id}.json"
        if result_file.exists():
            with open(result_file) as fh:
                all_results[str(exp_id)] = json.load(fh)

    summary_file = RESULTS_DIR / "all_experiments.json"
    summary_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nSummary saved -> {summary_file}")

    # Pretty-print leaderboard
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
    # Phase 4-5: Initialize audit logger for persistent resource/config tracking
    audit_logger = TrainingAuditLogger(append_to_file="terminal.txt")
    resources = detect_system_resources()
    audit_logger.log_resource_detection(resources)
    print(
        f"[Resources] GPU: {resources.device_name} ({resources.vram_gb:.1f}GB), "
        f"RAM: {resources.ram_gb:.1f}GB, CPU: {resources.cpu_cores} cores"
    )

    parser = argparse.ArgumentParser(description="Run DONUT SROIE experiments")
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
        "--force", action="store_true", help="Delete all cached results and re-run from scratch"
    )
    args = parser.parse_args()

    if args.force:
        for result_file in RESULTS_DIR.glob("experiment_*.json"):
            result_file.unlink()
            print(f"[force] Deleted cached result: {result_file}")

    if args.all:
        # Load base model once and pass to each experiment via deep-copy.
        # This replaces 8 × from_pretrained() disk reads with 8 in-RAM deep
        # copies — saving ~4–6 s of safetensors deserialization per experiment.
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
