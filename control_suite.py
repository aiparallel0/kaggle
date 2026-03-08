# =============================================================================
# control_suite.py
# Purpose: Comprehensive parameter control hub for DONUT, TrOCR, and YOLO
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-08
# =============================================================================
"""
control_suite.py — Single file inventory of every tuneable parameter across all
three model architectures: DONUT, TrOCR, and YOLOv8.

Background
----------
Parameters for this project are scattered across three files:
  - run_experiments.py  → ExperimentConfig (DONUT, 16 fields, well-structured)
  - train_trocr_yolo.py → 9 module-level constants (no dataclass, no ranges)
  - resource_optimizer.py → ResourceOptimizedConfig, TrainingConfig

Two reference files (finetuning_params.json / finetuning_params.md) document
the full parameter space including a ``commonly_underdocumented[]`` section with
parameters that have outsized impact but rarely appear in tutorials.

This module:
  1. Mirrors all DONUT parameters from ExperimentConfig (no duplication —
     ExperimentConfig remains the authoritative training config per CLAUDE.md).
  2. Adds the DONUT parameters that were only implicit (input_size, align_long_axis,
     sort_json_key, gradient_clip_val, swin_window_size).
  3. Provides TrOCRControlConfig — a proper dataclass for all TrOCR parameters,
     replacing the scattered module constants in train_trocr_yolo.py.
  4. Provides YOLOControlConfig — a proper dataclass exposing all Ultralytics
     defaults explicitly, including the critical ``freeze`` parameter that was
     previously absent from the training call.
  5. Bundles everything into a single ``ControlSuite`` with inspection helpers.

Impact legend (from finetuning_params.md):
    CRITICAL  — Wrong value causes silent model failure or full weight re-init
    HIGH      — Wrong value causes ≥5% F1 regression or severe training instability
    MEDIUM    — Affects training efficiency or generalisation noticeably
    LOW       — Marginal effect; safe to leave at default

Commonly-underdocumented parameters are flagged with ⚠️ in comments.

Usage
-----
    from control_suite import CONTROL_SUITE

    # Inspect all parameters
    CONTROL_SUITE.print_summary()

    # Access per-model configs
    CONTROL_SUITE.donut.gradient_clip_val   # → 1.0
    CONTROL_SUITE.trocr.weight_decay        # → 1e-4  (was 0 before this fix)
    CONTROL_SUITE.yolo.freeze               # → None  (explicitly documented)

    # Get only critical parameters
    crits = CONTROL_SUITE.critical_params()

    # Get list of commonly-underdocumented parameter names
    underdoc = CONTROL_SUITE.underdocumented_params()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

# Import shared constants — single source of truth per CLAUDE.md §7
from constants import BASE_MODEL, MAX_LENGTH, SEED

__all__ = [
    "DonutControlConfig",
    "TrOCRControlConfig",
    "YOLOControlConfig",
    "ControlSuite",
    "CONTROL_SUITE",
    "validate_sroie_oversample",
    "get_augmentation_transforms",
]

# ---------------------------------------------------------------------------
# Impact metadata — not enforced at runtime, used by print_summary()
# ---------------------------------------------------------------------------

# Maps param name → (impact_level, is_underdocumented)
# Populated by _register() calls at module load time.
_IMPACT_REGISTRY: dict[str, tuple[str, bool]] = {}


def _register(name: str, impact: str, underdoc: bool = False) -> None:
    """Register a parameter's impact level and underdocumented flag."""
    _IMPACT_REGISTRY[name] = (impact.upper(), underdoc)


# ---------------------------------------------------------------------------
# DONUT — Document Understanding Transformer
# ---------------------------------------------------------------------------

# Register DONUT parameter metadata
_register("donut.input_size", "CRITICAL")
_register("donut.swin_window_size", "CRITICAL", underdoc=True)
_register("donut.base_model", "CRITICAL")
_register("donut.max_length", "HIGH")
_register("donut.sort_json_key", "HIGH", underdoc=True)
_register("donut.epochs", "HIGH")
_register("donut.encoder_lr", "CRITICAL")
_register("donut.decoder_lr", "CRITICAL")
_register("donut.batch_size", "HIGH")
_register("donut.gradient_accumulation_steps", "HIGH")
_register("donut.precision", "HIGH")
_register("donut.early_stopping_patience", "HIGH")
_register("donut.warmup_steps", "MEDIUM")
_register("donut.weight_decay", "LOW")
_register("donut.gradient_clip_val", "MEDIUM")
_register("donut.align_long_axis", "MEDIUM", underdoc=True)
_register("donut.val_check_interval", "LOW", underdoc=True)
_register("donut.lr_schedule", "MEDIUM")
_register("donut.optimizer_type", "MEDIUM")
_register("donut.seed", "LOW")
_register("donut.sroie_oversample", "HIGH")
_register("donut.num_workers", "MEDIUM")
_register("donut.dataloader_pin_memory", "MEDIUM")
_register("donut.dataloader_prefetch_factor", "MEDIUM")
_register("donut.save_total_limit", "LOW")
_register("donut.predict_with_generate", "MEDIUM")
_register("donut.tie_word_embeddings", "CRITICAL")


@dataclass
class DonutControlConfig:
    """All DONUT fine-tuning parameters in one place.

    ExperimentConfig (run_experiments.py) remains the authoritative config for
    training runs (per CLAUDE.md GP-1).  This dataclass adds the parameters
    that are implicit in the codebase (e.g. input_size from the processor,
    gradient_clip_val from the HF Trainer default) and serves as a reference.

    Fields marked ``# ⚠️`` are from the commonly_underdocumented[] section of
    finetuning_params.md — they have outsized impact but rarely appear in tutorials.
    """

    # ── Image Resolution ────────────────────────────────────────────────────
    # impact: CRITICAL
    # Must be multiples of patch_size (32). Pretrain used [2560, 1920].
    # Finetuning: [1280, 960] (width × height). Larger = more VRAM, slower.
    # Common values: [640,480], [960,720], [1280,960], [1920,1440], [2560,1920]
    input_size: list[int] = field(default_factory=lambda: [1280, 960])

    # ── Architecture (READ-ONLY — do not change from pretrain value) ────────
    # impact: CRITICAL ⚠️ UNDERDOCUMENTED
    # donut-base uses window_size=10. Changing this forces full weight re-init
    # for all attention layers — effectively trains from scratch.
    # NEVER change this when fine-tuning from a pretrained checkpoint.
    swin_window_size: int = 10  # READ-ONLY — matches donut-base pretrain config

    # ── Base Model ──────────────────────────────────────────────────────────
    # impact: CRITICAL
    # Clean base checkpoint with no CORD task-specific priors.
    base_model: str = BASE_MODEL  # "naver-clova-ix/donut-base"

    # ── Sequence Length ─────────────────────────────────────────────────────
    # impact: HIGH
    # Caps the generated JSON sequence length. Increased from 512 to 768 to
    # reduce truncation of long address fields (weakest SROIE field).
    max_length: int = MAX_LENGTH  # 768

    # ── Training Core ───────────────────────────────────────────────────────
    # impact: HIGH — optimal per convergence analysis is 10 (CLAUDE.md §3)
    epochs: int = 10

    # impact: CRITICAL — encoder gets lower LR than decoder (layerwise LR)
    encoder_lr: float = 5e-5

    # impact: CRITICAL — decoder LR is 2× encoder LR (faster adaptation)
    decoder_lr: float = 1e-4

    # impact: HIGH — batch=8 is optimal for 500–3940 samples per CLAUDE.md §3
    batch_size: int = 8

    # impact: HIGH — effective batch = batch_size × gradient_accumulation_steps
    # HuggingFace alias: gradient_accumulation_steps
    # fairseq alias: update_freq
    gradient_accumulation_steps: int = 2

    # ── Mixed Precision ─────────────────────────────────────────────────────
    # impact: HIGH
    # Auto-detected at runtime: bf16 on Ampere+, fp16 otherwise.
    # PyTorch Lightning alias: precision=16
    # HuggingFace alias: fp16=True / bf16=True in Seq2SeqTrainingArguments
    precision: str = "auto"  # "auto" | "bf16" | "fp16" | "fp32"

    # ── Early Stopping ──────────────────────────────────────────────────────
    # impact: HIGH — patience=3 prevents overfitting on 63-sample val set
    early_stopping_patience: int = 3

    # ── LR Warmup ───────────────────────────────────────────────────────────
    # impact: MEDIUM
    # Capped at runtime to ≤10% of total optimizer steps.
    # warmup=500 would exceed total steps for small datasets (Exp 1: ~312 steps).
    # Fixed at 40 so it is safe across all 8 experiments.
    warmup_steps: int = 40

    # ── Regularisation ──────────────────────────────────────────────────────
    # impact: LOW
    weight_decay: float = 0.01

    # impact: MEDIUM — prevents exploding gradients in the transformer decoder.
    # 1.0 is the universal default. Not currently passed explicitly to
    # Seq2SeqTrainingArguments (uses HF default which is also 1.0).
    gradient_clip_val: float = 1.0

    # ── Image Preprocessing ─────────────────────────────────────────────────
    # impact: MEDIUM ⚠️ UNDERDOCUMENTED
    # Whether to rotate portrait images to landscape. Usually False for
    # structured documents like receipts (text runs top-to-bottom naturally).
    # DonutProcessor alias: do_align_long_axis
    align_long_axis: bool = False

    # ── Data ────────────────────────────────────────────────────────────────
    # impact: HIGH ⚠️ UNDERDOCUMENTED
    # MUST be False for preprocessed datasets like CORD-v2 and SROIE.
    # Setting True silently corrupts token ordering in ground-truth sequences.
    sort_json_key: bool = False

    # ── Validation ──────────────────────────────────────────────────────────
    # impact: LOW ⚠️ UNDERDOCUMENTED
    # PyTorch Lightning float form: 0.2 = validate 5× per epoch.
    # HuggingFace: eval_strategy="epoch" (always validates once per epoch).
    # Can be useful for fast-converging small datasets.
    val_check_interval: float = 1.0

    # ── LR Schedule ─────────────────────────────────────────────────────────
    # impact: MEDIUM
    # "cosine" | "one_cycle" | "linear"
    # one_cycle: aggressive warmup + cosine decay — faster for short micro runs.
    lr_schedule: str = "cosine"

    # ── Optimizer ───────────────────────────────────────────────────────────
    # impact: MEDIUM
    # "adamw" | "sgd" (sgd = SGD + Nesterov, used in micro/mini mode)
    optimizer_type: str = "adamw"

    # ── Reproducibility ─────────────────────────────────────────────────────
    # impact: LOW
    seed: int = SEED  # 42

    # ── Dataset Balancing ───────────────────────────────────────────────────
    # impact: HIGH — SROIE oversampling is a prerequisite for auxiliary data to help.
    # Without 2× SROIE, Exps 2–4 score at or below the baseline (CLAUDE.md §8).
    sroie_oversample: int = 1

    # ── DataLoader Performance ──────────────────────────────────────────────
    # impact: MEDIUM — auto-computed by _optimal_num_workers() in constants.py
    num_workers: int = 8  # min(8, max(4, cpu_count // 2))

    # impact: MEDIUM — set False when num_workers=0 (incompatible with multiprocessing)
    dataloader_pin_memory: bool = True

    # impact: MEDIUM — prefetch factor for DataLoader workers (set None when workers=0)
    dataloader_prefetch_factor: int = 4

    # ── Checkpointing ───────────────────────────────────────────────────────
    # impact: LOW — keep 3 best checkpoints; older are auto-deleted
    save_total_limit: int = 3

    # ── Generation / Evaluation ─────────────────────────────────────────────
    # impact: MEDIUM — must be True for F1 eval during training in Seq2SeqTrainer
    predict_with_generate: bool = True

    # ── Weight Tying (CRITICAL BUG GUARD) ───────────────────────────────────
    # impact: CRITICAL
    # Must ALWAYS be False after resize_token_embeddings().
    # If True, tie_weights() on checkpoint reload destroys the learned lm_head
    # → F1 = 0.0 on every prediction.  See CLAUDE.md §2 and §16.
    tie_word_embeddings: bool = False  # NEVER change to True


# ---------------------------------------------------------------------------
# TrOCR — Transformer-based OCR
# ---------------------------------------------------------------------------

# Register TrOCR parameter metadata
_register("trocr.model_id", "CRITICAL")
_register("trocr.arch", "CRITICAL")
_register("trocr.input_size", "CRITICAL")
_register("trocr.patch_size", "CRITICAL")
_register("trocr.epochs", "HIGH")
_register("trocr.batch_size", "HIGH")
_register("trocr.learning_rate", "CRITICAL")
_register("trocr.max_length", "MEDIUM")
_register("trocr.gradient_accumulation_steps", "HIGH")
_register("trocr.mini_mode", "MEDIUM")
_register("trocr.lr_scheduler", "MEDIUM")
_register("trocr.warmup_ratio", "MEDIUM")
_register("trocr.warmup_init_lr", "MEDIUM", underdoc=True)
_register("trocr.weight_decay", "LOW")
_register("trocr.adam_beta1", "LOW")
_register("trocr.adam_beta2", "LOW")
_register("trocr.adam_epsilon", "LOW")
_register("trocr.gradient_clip_val", "MEDIUM")
_register("trocr.gradient_checkpointing", "HIGH")
_register("trocr.grad_ckpt_vram_threshold_gb", "HIGH")
_register("trocr.use_cache", "MEDIUM")
_register("trocr.predict_with_generate", "MEDIUM")
_register("trocr.num_beams", "MEDIUM")
_register("trocr.no_repeat_ngram_size", "MEDIUM")
_register("trocr.length_penalty", "LOW")
_register("trocr.patience", "HIGH", underdoc=True)
_register("trocr.augmentation_preset", "HIGH", underdoc=True)
_register("trocr.lora_rank", "HIGH")
_register("trocr.use_dora_encoder", "HIGH")
_register("trocr.num_workers", "MEDIUM")
_register("trocr.seed", "LOW")


@dataclass
class TrOCRControlConfig:
    """All TrOCR fine-tuning parameters.

    Replaces the 9 scattered module-level constants in train_trocr_yolo.py
    with a single structured dataclass. train_trocr_yolo.py keeps the
    module constants for backward compatibility with micro-mode patching
    (``TROCR_EPOCHS = 2`` etc.) but imports the new params from here.

    Architecture note: TrOCR uses a BEiT/DeiT encoder + RoBERTa/UniLM decoder.
    The encoder input is ALWAYS 384×384 — not configurable without full re-train.
    fairseq (official) uses update_freq for gradient accumulation;
    HuggingFace uses gradient_accumulation_steps. Both are aliased here.
    """

    # ── Model Identity ──────────────────────────────────────────────────────
    # impact: CRITICAL
    # "microsoft/trocr-base-printed" for printed text (SROIE receipts).
    # Alternatives: trocr-base-handwritten, trocr-large-printed
    model_id: str = "microsoft/trocr-base-printed"

    # impact: CRITICAL
    # trocr_small (62M) | trocr_base (334M) | trocr_large (558M)
    # base is the standard community choice for receipt OCR
    arch: str = "trocr_base"

    # ── Image Resolution ────────────────────────────────────────────────────
    # impact: CRITICAL ⚠️ UNDERDOCUMENTED (in a different way: it's FIXED)
    # ALL input images are ALWAYS resized to 384×384. Not configurable without
    # re-training the encoder. Known limitation — discussed in microsoft/unilm #674.
    # Workaround: pad image to square before resizing, or use sliding window.
    input_size: int = 384  # FIXED — do not change

    # impact: CRITICAL — 384/16 = 24 patches per side → 576 total patch tokens
    patch_size: int = 16  # FIXED — architectural constant

    # ── Training Core ───────────────────────────────────────────────────────
    # impact: HIGH
    # Official Microsoft training used 300 with patience=20.
    # Community finetuning typically 5–30 epochs on small datasets.
    epochs: int = 10

    # impact: HIGH
    # Official: BSZ=8 across 8 GPUs = 64 effective batch.
    # Community: 4–16 per GPU. VRAM-aware auto-scaling halves this when needed.
    # HuggingFace alias: per_device_train_batch_size
    # fairseq alias: BSZ
    batch_size: int = 16

    # impact: CRITICAL
    # Official training: 2e-5. Community HF finetuning: 4e-5–5e-5 for
    # smaller datasets. Lower LR reduces catastrophic forgetting.
    learning_rate: float = 5e-5

    # impact: MEDIUM
    # TrOCR processes line-cropped images; 64 tokens covers most text lines.
    # Use 512 for full-document inputs.
    # HuggingFace alias: generation_max_length | max_new_tokens
    max_length: int = 128

    # impact: HIGH
    # Gradient accumulation factor. update_freq=4 with BSZ=4 = effective BSZ 16.
    # fairseq alias: update_freq
    # HuggingFace alias: gradient_accumulation_steps
    gradient_accumulation_steps: int = 4

    # impact: MEDIUM
    # True → SGD+Nesterov+CosineAnnealingLR — faster convergence for micro runs.
    # False → AdamW+linear warmup (standard for full training runs).
    mini_mode: bool = False

    # ── LR Schedule ─────────────────────────────────────────────────────────
    # impact: MEDIUM
    # Official uses "inverse_sqrt" warmup decay (fairseq).
    # HuggingFace default is "linear". "cosine" recommended for longer finetune runs.
    # fairseq alias: lr-scheduler
    # HuggingFace alias: lr_scheduler_type
    lr_scheduler: str = "linear"

    # impact: MEDIUM
    # Warmup steps = warmup_ratio × total_training_steps.
    # 10% warmup (ratio=0.1) is the project default, computed inline.
    # fairseq alias: warmup_updates
    # HuggingFace alias: warmup_steps | num_warmup_steps
    warmup_ratio: float = 0.1

    # impact: MEDIUM ⚠️ UNDERDOCUMENTED
    # Starting LR at the very first warmup step. Prevents unstable initial
    # gradient updates. Present in fairseq config but absent from all blog posts.
    # Too-high value causes early divergence (loss spike at step 0).
    warmup_init_lr: float = 1e-8

    # ── Regularisation ──────────────────────────────────────────────────────
    # impact: LOW
    # Reference: 1e-4. NOTE: Before control_suite, AdamW was called without
    # weight_decay → PyTorch default of 0. This was a silent bug.
    weight_decay: float = 1e-4

    # impact: LOW — Adam first moment decay rate
    adam_beta1: float = 0.9

    # impact: LOW — Adam second moment decay rate
    adam_beta2: float = 0.999

    # impact: LOW — Adam numerical stability epsilon term
    adam_epsilon: float = 1e-8

    # impact: MEDIUM — global gradient norm clipping (applied explicitly via
    # clip_grad_norm_ at each accumulation step in train_trocr_yolo.py)
    gradient_clip_val: float = 1.0

    # ── Memory Optimisation ──────────────────────────────────────────────────
    # impact: HIGH — trades ~30-40% extra compute for activation memory savings
    # Required for TrOCR-base (246M params) to fit on lower-VRAM GPUs during backward.
    # use_cache MUST be False when gradient_checkpointing is True (incompatible).
    gradient_checkpointing: bool = True

    # impact: HIGH — VRAM threshold (GB) above which gradient checkpointing is
    # disabled.  Cards with VRAM > threshold have enough headroom that the
    # ~30-40% backward overhead is wasted.  Uses strict greater-than so that a
    # card reporting exactly 24.0 GB (e.g. RTX 4090) keeps checkpointing ON.
    # Mirrors DonutControlConfig / run_experiments._GRAD_CKPT_VRAM_THRESHOLD_GB.
    grad_ckpt_vram_threshold_gb: float = 24.0

    # impact: MEDIUM — must be False when gradient_checkpointing is True
    use_cache: bool = False

    # ── Generation / Evaluation ─────────────────────────────────────────────
    # impact: MEDIUM — must be True for CER/WER evaluation in Seq2SeqTrainer.
    # False = teacher-forcing logits for loss only (no beam search at eval time).
    # HuggingFace alias: predict_with_generate in Seq2SeqTrainingArguments
    predict_with_generate: bool = True

    # impact: MEDIUM — beam size for generation. 4 beams is the project default.
    num_beams: int = 4

    # impact: MEDIUM — set 0 to disable (n-gram blocking harmful for short OCR text)
    no_repeat_ngram_size: int = 0

    # impact: LOW — 1.0 = neutral (do not penalise short outputs)
    length_penalty: float = 1.0

    # ── Early Stopping ──────────────────────────────────────────────────────
    # impact: HIGH ⚠️ UNDERDOCUMENTED
    # Official used patience=20 with 300 epochs.
    # Community finetuning: 3–5 epochs.
    # None = not implemented (current project state for TrOCR).
    patience: int | None = None

    # ── Data Augmentation ───────────────────────────────────────────────────
    # impact: HIGH ⚠️ UNDERDOCUMENTED
    # Official fairseq config: DA2 (stronger augmentation used in IAM training).
    # Microsoft TrOCR paper: DA2 augmentation gives significantly better CER.
    # None = no augmentation (current project state — only basic resize).
    # Options: None | "DA1" | "DA2"
    # DA1 (light):   RandomRotation(±5°) + ColorJitter(brightness=0.2, contrast=0.2)
    # DA2 (strong):  RandomPerspective(distortion=0.2, p=0.5) +
    #                ElasticTransform(alpha=50.0) +
    #                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
    # Use get_augmentation_transforms(preset) to obtain the pipeline.
    augmentation_preset: str | None = None

    # ── PEFT (Parameter-Efficient Fine-Tuning) ───────────────────────────────
    # impact: HIGH
    # LoRA rank for DLoRA-TrOCR (Chang et al. 2024). Reduces trainable params
    # from 334M to <10M with comparable accuracy.
    # None = full fine-tuning (current project state).
    lora_rank: int | None = None

    # impact: HIGH — Apply DoRA to encoder instead of standard LoRA.
    # DLoRA-TrOCR paper: encoder uses DoRA, decoder uses LoRA.
    use_dora_encoder: bool = False

    # ── DataLoader ──────────────────────────────────────────────────────────
    # impact: MEDIUM — auto-computed by _optimal_num_workers() in constants.py
    num_workers: int = 8

    # ── Reproducibility ─────────────────────────────────────────────────────
    # impact: LOW
    seed: int = SEED  # 42

    # ── VRAM Calibration Constants (read-only, not tuneable) ─────────────────
    # Used by effective_batch_size() to compute the safe batch for available VRAM.
    # Empirically calibrated for TrOCR-base with AMP enabled.
    #   _TROCR_RESERVED_GB: total overhead (weights + gradients + AdamW states + system)
    #   _TROCR_PER_ITEM_GB: activation cost per batch item with AMP (bf16/fp16)
    _TROCR_RESERVED_GB: ClassVar[float] = 6.0
    _TROCR_PER_ITEM_GB: ClassVar[float] = 0.3

    def effective_batch_size(self, vram_gb: float) -> int:
        """Return the largest safe batch size for the given available VRAM (GB).

        Uses empirically calibrated constants:
          _TROCR_RESERVED_GB = 6.0  (model weights + grads + AdamW states + overhead)
          _TROCR_PER_ITEM_GB  = 0.3  (activation cost per item with AMP, TrOCR-base)

        Mirrors the inline VRAM check in train_trocr_yolo.train_trocr() so that
        the reported batch size in CONTROL_SUITE always matches the actual training
        batch after scaling.

        Returns at most self.batch_size and at least 1.
        """
        usable = max(vram_gb - self._TROCR_RESERVED_GB, 1.0)
        safe = max(1, int(usable / self._TROCR_PER_ITEM_GB))
        return min(self.batch_size, safe)


# ---------------------------------------------------------------------------
# YOLO — YOLOv8 Object Detection
# ---------------------------------------------------------------------------

# Register YOLO parameter metadata
_register("yolo.base_model", "CRITICAL")
_register("yolo.imgsz", "CRITICAL")
_register("yolo.mosaic", "CRITICAL")
_register("yolo.freeze", "CRITICAL", underdoc=True)
_register("yolo.lr0", "CRITICAL")
_register("yolo.epochs", "HIGH")
_register("yolo.batch", "HIGH")
_register("yolo.amp", "HIGH")
_register("yolo.optimizer", "HIGH")
_register("yolo.patience", "HIGH")
_register("yolo.scale", "HIGH")
_register("yolo.close_mosaic", "HIGH", underdoc=True)
_register("yolo.copy_paste", "HIGH")
_register("yolo.cache", "HIGH")
_register("yolo.fraction", "HIGH")
_register("yolo.box", "HIGH")
_register("yolo.cls", "HIGH")
_register("yolo.momentum", "MEDIUM")
_register("yolo.lrf", "MEDIUM")
_register("yolo.weight_decay", "MEDIUM")
_register("yolo.warmup_epochs", "MEDIUM")
_register("yolo.cos_lr", "MEDIUM")
_register("yolo.fliplr", "MEDIUM")
_register("yolo.degrees", "MEDIUM")
_register("yolo.translate", "MEDIUM")
_register("yolo.mixup", "MEDIUM")
_register("yolo.hsv_h", "MEDIUM")
_register("yolo.hsv_s", "MEDIUM")
_register("yolo.hsv_v", "MEDIUM")
_register("yolo.multi_scale", "MEDIUM")
_register("yolo.rect", "MEDIUM", underdoc=True)
_register("yolo.conf", "MEDIUM")
_register("yolo.iou", "MEDIUM")
_register("yolo.dropout", "MEDIUM")
_register("yolo.single_cls", "MEDIUM")
_register("yolo.workers", "MEDIUM")
_register("yolo.dfl", "MEDIUM")
_register("yolo.warmup_momentum", "LOW")
_register("yolo.warmup_bias_lr", "LOW")
_register("yolo.flipud", "LOW")
_register("yolo.shear", "LOW")
_register("yolo.perspective", "LOW")
_register("yolo.bgr", "LOW")
_register("yolo.max_det", "LOW")
_register("yolo.save_period", "LOW")
_register("yolo.seed", "LOW")
_register("yolo.deterministic", "LOW")
_register("yolo.amp_oom_auto_retry", "MEDIUM", underdoc=True)


@dataclass
class YOLOControlConfig:
    """All YOLOv8 fine-tuning parameters.

    Exposes all Ultralytics defaults explicitly so that any change to the
    training call is intentional and visible, not an accidental reliance on
    a library default that may change across Ultralytics versions.

    Parameters that were previously absent from the train_yolo() call are
    marked MISSING in the comments — they are now explicitly passed.

    Fields marked ``# ⚠️`` are from the commonly_underdocumented[] section.
    """

    # ── Model ────────────────────────────────────────────────────────────────
    # impact: CRITICAL
    # yolov8x.pt: extra-large (~68M params). Batch and imgsz kept low for VRAM.
    # Alternatives: yolov8n.pt (3M), yolov8s.pt (11M), yolov8m.pt (26M),
    #               yolov8l.pt (44M), yolov8x.pt (68M), yolo11n.pt, ...
    base_model: str = "yolov8x.pt"

    # ── Image Resolution ────────────────────────────────────────────────────
    # impact: CRITICAL
    # Square input resolution. Images auto-resized and letterboxed.
    # Must match between training and inference.
    # Reduced from 640 to 512 to lower VRAM usage.
    # Common values: [320, 416, 512, 640, 832, 1024, 1280]
    imgsz: int = 512

    # ── Training Core ───────────────────────────────────────────────────────
    # impact: HIGH
    batch: int = 8

    # impact: HIGH — default 100 for finetuning; 300 for training from scratch
    epochs: int = 50

    # impact: HIGH — halt if no mAP50-95 improvement for N epochs.
    # Ultralytics default: 50. For finetuning: 10–20.
    patience: int = 15

    # ── Optimizer ───────────────────────────────────────────────────────────
    # impact: HIGH
    # "auto" selects AdamW for ≤10 warmup epochs, SGD otherwise.
    # Explicit "AdamW" recommended for finetuning.
    # Options: auto | SGD | Adam | AdamW | NAdam | RAdam | RMSProp
    optimizer: str = "AdamW"

    # impact: MEDIUM — SGD momentum / Adam beta1 (used when optimizer="SGD")
    momentum: float = 0.9

    # impact: CRITICAL — initial (peak) LR. AdamW default ~0.001.
    # Lower values recommended for finetuning to avoid overwriting pretrained features.
    lr0: float = 1e-3

    # impact: MEDIUM — final LR as fraction of lr0 (cosine/linear decay endpoint)
    # final_lr = lr0 × lrf
    lrf: float = 0.01

    # impact: MEDIUM — L2 regularisation. Ultralytics default: 0.0005.
    # PREVIOUSLY MISSING from train_yolo() call.
    weight_decay: float = 0.0005

    # impact: MEDIUM — use cosine annealing LR schedule (smoother decay).
    # When False, uses linear decay. PREVIOUSLY MISSING.
    cos_lr: bool = False

    # ── LR Warmup ───────────────────────────────────────────────────────────
    # impact: MEDIUM — linear warmup from near-zero to lr0 over N epochs.
    # PREVIOUSLY MISSING from train_yolo() call.
    warmup_epochs: float = 3.0

    # impact: LOW — starting SGD momentum during warmup phase.
    # PREVIOUSLY MISSING from train_yolo() call.
    warmup_momentum: float = 0.8

    # impact: LOW — starting LR specifically for bias parameters during warmup.
    # PREVIOUSLY MISSING from train_yolo() call.
    warmup_bias_lr: float = 0.1

    # ── Mixed Precision ─────────────────────────────────────────────────────
    # impact: HIGH — Automatic Mixed Precision (FP16/FP32). On by default.
    # Reduces VRAM ~50%, speeds training ~1.5–2×.
    # Disable (amp=False) if NaN losses appear in early training.
    # ⚠️ UNDERDOCUMENTED: YOLO silently halves batch size on CUDA OOM when amp=True.
    amp: bool = True

    # impact: MEDIUM ⚠️ UNDERDOCUMENTED (consequence of amp=True)
    # YOLO silently halves batch size on CUDA OOM — can cause inconsistent
    # effective batch sizes across experiment runs.
    amp_oom_auto_retry: bool = True  # INFORMATIONAL — cannot be disabled, documented here

    # ── Finetuning ──────────────────────────────────────────────────────────
    # impact: CRITICAL ⚠️ UNDERDOCUMENTED
    # THE most impactful underdocumented parameter for domain finetuning on
    # small datasets. Freezing backbone prevents overfitting and dramatically
    # reduces training time.
    # None = full training (all layers updated) — CURRENT PROJECT STATE
    # 0    = freeze nothing (same as None)
    # 3    = freeze first 3 backbone layers
    # 10   = freeze entire backbone, train only detection head
    # list = freeze specific layer indices
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default None).
    freeze: int | list[int] | None = None

    # ── Augmentation ────────────────────────────────────────────────────────
    # impact: CRITICAL — combines 4 training images into one 2×2 grid.
    # Massively improves detection robustness. Disabled for last close_mosaic epochs.
    # Set to 0.0 if document structure must be preserved exactly.
    # Current project: 0.5 (reduced from default 1.0 for receipt domain)
    mosaic: float = 0.5

    # impact: HIGH ⚠️ UNDERDOCUMENTED
    # Disable mosaic augmentation for the last N epochs.
    # CRITICAL for final accuracy. Ultralytics default: 10.
    # PREVIOUSLY MISSING from train_yolo() call (was relying on default).
    close_mosaic: int = 10

    # impact: MEDIUM — horizontal flip probability.
    # Set 0 for text/receipt detection (left-right orientation matters for text).
    # Current project: 0.0 (already set correctly for text domain)
    fliplr: float = 0.0

    # impact: LOW — vertical flip probability (usually 0 for receipts)
    flipud: float = 0.0

    # impact: MEDIUM — random rotation range in degrees.
    # Small non-zero values help with tilted receipts.
    degrees: float = 5.0

    # impact: MEDIUM — random translation as fraction of image size
    translate: float = 0.1

    # impact: HIGH — random scale (zoom) augmentation range.
    # Critical for detecting text at varying distances.
    scale: float = 0.3

    # impact: LOW — shear transformation in degrees
    shear: float = 0.0

    # impact: LOW — random perspective distortion coefficient
    perspective: float = 0.0

    # impact: MEDIUM — probability of mixup blending. Use 0–0.2 max.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 0).
    mixup: float = 0.0

    # impact: HIGH — probability of copy-paste augmentation.
    # Very effective for rare-class augmentation in imbalanced datasets.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 0).
    copy_paste: float = 0.0

    # impact: MEDIUM — hue jitter magnitude
    hsv_h: float = 0.015

    # impact: MEDIUM — saturation jitter magnitude
    hsv_s: float = 0.7

    # impact: MEDIUM — brightness/value jitter magnitude
    hsv_v: float = 0.4

    # impact: LOW — probability of random BGR channel swap
    bgr: float = 0.0

    # impact: MEDIUM — vary imgsz by ±50% each batch (multi-scale robustness)
    multi_scale: bool = False

    # ── Data ────────────────────────────────────────────────────────────────
    # impact: MEDIUM ⚠️ UNDERDOCUMENTED
    # rect=True batches images by aspect ratio to reduce letterbox padding waste.
    # WARNING: silently disables DataLoader shuffle → can bias training.
    rect: bool = False

    # impact: HIGH — fraction of dataset to use for training.
    # Reduce to 0.1 for fast hyperparameter sweep experiments.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 1.0).
    fraction: float = 1.0

    # ── Performance ─────────────────────────────────────────────────────────
    # impact: HIGH — 'ram' caches entire dataset in memory for maximum I/O speed.
    # 'disk' caches preprocessed images. False = load from disk each epoch.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default False).
    cache: bool | str = False

    # impact: MEDIUM — DataLoader worker threads per GPU rank
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 8).
    workers: int = 8

    # ── Loss Weights ────────────────────────────────────────────────────────
    # impact: HIGH — bounding box regression loss (GIoU/CIoU) coefficient.
    # Increase for tight localisation tasks (e.g. text-region detection).
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 7.5).
    box: float = 7.5

    # impact: HIGH — classification loss weight.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 0.5).
    cls: float = 0.5

    # impact: MEDIUM — Distribution Focal Loss weight.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 1.5).
    dfl: float = 1.5

    # ── Inference / NMS ─────────────────────────────────────────────────────
    # impact: MEDIUM — IoU threshold for NMS. Higher = fewer, more-overlapping boxes.
    conf: float = 0.25

    # impact: MEDIUM — minimum confidence threshold. Lower = more false positives.
    iou: float = 0.7

    # impact: LOW — maximum detections per image after NMS
    max_det: int = 300

    # ── Regularisation ──────────────────────────────────────────────────────
    # impact: MEDIUM — dropout in classification head. 0.1–0.2 for small datasets.
    # PREVIOUSLY MISSING from train_yolo() call (using Ultralytics default 0).
    dropout: float = 0.0

    # ── Task ────────────────────────────────────────────────────────────────
    # impact: MEDIUM — treat all classes as a single class (binary detection).
    single_cls: bool = False

    # ── Checkpointing ───────────────────────────────────────────────────────
    # impact: LOW — save checkpoint every N epochs. -1 = best.pt + last.pt only.
    save_period: int = -1

    # ── Reproducibility ─────────────────────────────────────────────────────
    # impact: LOW
    seed: int = SEED  # 42

    # impact: LOW — forces CUDA deterministic algorithms (slight speed penalty)
    deterministic: bool = True

    def recommended_freeze(self, num_train_samples: int) -> int | None:
        """Return the recommended freeze depth for the given training dataset size.

        Provides evidence-based guidance for domain finetuning on small receipt
        datasets, where freezing backbone layers prevents overfitting:

          >= 2000 samples  → None  (full training — enough data for all layers)
          500–1999 samples → 10   (freeze backbone, train detection head only)
          < 500 samples    → 3    (freeze first 3 backbone layers only)

        Call this when self.freeze is None (not explicitly overridden) to get a
        safe starting point. The recommendation is advisory — the caller decides
        whether to apply it.
        """
        if num_train_samples >= 2000:
            return None
        elif num_train_samples >= 500:
            return 10
        else:
            return 3


# ---------------------------------------------------------------------------
# ControlSuite — top-level bundle
# ---------------------------------------------------------------------------


@dataclass
class ControlSuite:
    """Comprehensive parameter hub for all three model architectures.

    Bundles DonutControlConfig, TrOCRControlConfig, and YOLOControlConfig
    with cross-cutting inspection helpers.

    Usage
    -----
        from control_suite import CONTROL_SUITE

        CONTROL_SUITE.print_summary()
        CONTROL_SUITE.critical_params()
        CONTROL_SUITE.underdocumented_params()
    """

    donut: DonutControlConfig
    trocr: TrOCRControlConfig
    yolo: YOLOControlConfig

    @classmethod
    def default(cls) -> "ControlSuite":
        """Return a ControlSuite with project-default configurations."""
        return cls(
            donut=DonutControlConfig(),
            trocr=TrOCRControlConfig(),
            yolo=YOLOControlConfig(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full control suite to a nested dict."""
        return {
            "donut": asdict(self.donut),
            "trocr": asdict(self.trocr),
            "yolo": asdict(self.yolo),
        }

    def critical_params(self) -> dict[str, Any]:
        """Return only parameters with impact=CRITICAL as a flat dict.

        Keys are prefixed with model name (e.g. "donut.input_size").
        """
        flat = _flatten_suite(self)
        return {
            k: v
            for k, v in flat.items()
            if _IMPACT_REGISTRY.get(k, ("", False))[0] == "CRITICAL"
        }

    def underdocumented_params(self) -> list[str]:
        """Return names of all commonly-underdocumented parameters.

        These are the parameters from the commonly_underdocumented[] section
        of finetuning_params.md that have outsized impact but rarely appear
        in tutorials.
        """
        return [k for k, (_, underdoc) in _IMPACT_REGISTRY.items() if underdoc]

    def print_summary(self) -> None:
        """Print all parameters with impact ratings and current values.

        Impact legend: CRITICAL > HIGH > MEDIUM > LOW
        ⚠️  = commonly underdocumented (see finetuning_params.md)
        """
        _impact_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        flat = _flatten_suite(self)

        # Group by model
        by_model: dict[str, list[tuple[str, Any, str, bool]]] = {
            "donut": [],
            "trocr": [],
            "yolo": [],
        }
        for key, value in flat.items():
            model = key.split(".")[0]
            impact, underdoc = _IMPACT_REGISTRY.get(key, ("LOW", False))
            by_model[model].append((key, value, impact, underdoc))

        _model_labels = {
            "donut": "DONUT — Document Understanding Transformer",
            "trocr": "TrOCR — Transformer OCR",
            "yolo": "YOLOv8 — Object Detection",
        }

        print("\n" + "=" * 80)
        print("CONTROL SUITE — Parameter Inventory")
        print("Impact: CRITICAL > HIGH > MEDIUM > LOW  |  ⚠️  = underdocumented")
        print("=" * 80)

        for model_name, params in by_model.items():
            params.sort(key=lambda x: (_impact_order.get(x[2], 99), x[0]))
            print(f"\n{'─' * 80}")
            print(f"  {_model_labels[model_name]}")
            print(f"{'─' * 80}")
            print(f"  {'Parameter':<45} {'Impact':<10} {'Value'}")
            print(f"  {'─'*44} {'─'*9} {'─'*20}")
            for key, value, impact, underdoc in params:
                param_name = key.split(".", 1)[1]
                flag = " ⚠️" if underdoc else ""
                print(f"  {param_name + flag:<45} {impact:<10} {value!r}")

        print("\n" + "=" * 80)
        total = len(flat)
        n_critical = sum(1 for k in flat if _IMPACT_REGISTRY.get(k, ("",))[0] == "CRITICAL")
        n_underdoc = len(self.underdocumented_params())
        print(f"  Total parameters: {total}  |  Critical: {n_critical}  |  Underdocumented: {n_underdoc}")
        print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_suite(suite: ControlSuite) -> dict[str, Any]:
    """Flatten a ControlSuite to a dot-notation dict keyed by 'model.param'."""
    result: dict[str, Any] = {}
    for model_name, config in [("donut", suite.donut), ("trocr", suite.trocr), ("yolo", suite.yolo)]:
        for k, v in asdict(config).items():
            result[f"{model_name}.{k}"] = v
    return result


def validate_sroie_oversample(
    datasets: list[str],
    sroie_oversample: int,
    skip_guard: bool = False,
) -> None:
    """Raise ValueError if a multi-dataset run uses sroie_oversample < 2.

    Without 2× SROIE oversampling, auxiliary datasets dilute the SROIE training
    signal and cause Experiments 2–4 to score at or below the baseline (see
    CLAUDE.md §8).  This validator enforces the rule at the start of
    run_experiment() so misconfigured runs fail fast rather than silently
    producing suboptimal results.

    Parameters
    ----------
    datasets:
        The list of dataset names for the experiment (e.g. ["sroie", "wildreceipt"]).
    sroie_oversample:
        The SROIE oversampling factor (must be >= 2 when len(datasets) > 1).
    skip_guard:
        When True, bypass the validation entirely.  Use only for intentional
        naïve control experiments (Exps 2–4) that deliberately use
        sroie_oversample=1 to prove that auxiliary data hurts without
        oversampling.  The guard is still enforced for all other callers
        (default False).

    Raises
    ------
    ValueError
        When more than one dataset is combined and sroie_oversample < 2,
        unless skip_guard is True.
    """
    if skip_guard:
        return  # intentional naïve control group — guard explicitly waived
    if len(datasets) > 1 and sroie_oversample < 2:
        raise ValueError(
            f"sroie_oversample={sroie_oversample} is too low for a multi-dataset run "
            f"(datasets={datasets!r}). "
            "Without 2× SROIE oversampling, auxiliary data dilutes the SROIE training "
            "signal and causes F1 to fall at or below the single-dataset baseline. "
            "Set sroie_oversample >= 2 when combining datasets."
        )


def get_augmentation_transforms(preset: str | None):
    """Return a torchvision transforms pipeline for the given preset, or None.

    Preset specifications
    ---------------------
    None
        No augmentation (pass-through) — current project default.
    "DA1" (light)
        RandomRotation(degrees=5) +
        ColorJitter(brightness=0.2, contrast=0.2)
    "DA2" (strong, matches TrOCR paper DA2 config)
        RandomPerspective(distortion_scale=0.2, p=0.5) +
        ElasticTransform(alpha=50.0) +
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)

    CI-safe: torchvision is an optional dependency.  If torchvision is not
    installed, this function returns None instead of raising ImportError, so
    control_suite.py remains importable in any environment.

    Parameters
    ----------
    preset:
        One of None, "DA1", or "DA2".

    Returns
    -------
    torchvision.transforms.Compose | None
        A pipeline that accepts a PIL image and returns a PIL image, or None
        when preset is None or torchvision is unavailable.

    Raises
    ------
    ValueError
        When preset is an unrecognised non-None string.
    """
    if preset is None:
        return None

    # Validate preset name before attempting the optional torchvision import
    # so that misconfigured presets raise immediately in all environments.
    if preset not in ("DA1", "DA2"):
        raise ValueError(
            f"Unknown augmentation preset: {preset!r}. Valid values: None, 'DA1', 'DA2'."
        )

    try:
        from torchvision import transforms
    except ImportError:
        return None

    if preset == "DA1":
        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )
    else:  # "DA2"
        return transforms.Compose(
            [
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ElasticTransform(alpha=50.0),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            ]
        )


# ---------------------------------------------------------------------------
# Module-level singleton — import this
# ---------------------------------------------------------------------------

#: Singleton with project-default configurations for all three models.
#: Import this in train_trocr_yolo.py to access TrOCR and YOLO params.
CONTROL_SUITE: ControlSuite = ControlSuite.default()
