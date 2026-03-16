# =============================================================================
# resource_optimizer.py
# Purpose: Hardware-adaptive batch-size and epoch optimizer for GPU VRAM constraints
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-07
# =============================================================================
"""
resource_optimizer.py — Dynamic hardware detection and hyperparameter optimization.

Detects available GPU VRAM, RAM, and CPU cores, then recommends optimal training
hyperparameters (batch size, gradient accumulation, learning rates, epochs) based
on system capabilities and dataset size.

Also provides TrainingAuditLogger to log configuration decisions and results to
a persistent terminal.txt file, enabling data-driven refactoring decisions.

Key public functions:
  get_image_size_from_processor_config() — reads (height, width) from processor_config.json
      with a safe fallback to the reference resolution (1280, 960).  Used by
      optimize_hyperparams() to scale VRAM estimates for non-reference image sizes.
  optimize_hyperparams()     — returns ResourceOptimizedConfig for given VRAM/dataset
  validate_training_config() — raises ValueError if total optimizer steps < minimum

CLAUDE.md Reference:
  - Optimal batch size: 8 (from testing); fallback 4 (low VRAM) or 16 (high VRAM)
  - Epochs: 10 (fixed per CLAUDE.md convergence analysis, NOT 30)
  - Warmup steps: 40 (fixed per CLAUDE.md LR schedule; see ExperimentConfig.warmup_steps)
  - Learning rates: encoder=5e-5, decoder=1e-4 (fixed, layerwise LR)
  - Early stopping patience: 3 (fixed; 5 for small datasets <1000 samples)
"""

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = [
    "ResourceOptimizedConfig",
    "SystemResources",
    "detect_system_resources",
    "get_image_size_from_processor_config",
    "optimize_hyperparams",
    "validate_training_config",
    "TrainingAuditLogger",
    # Absorbed from training_config.py
    "TrainingConfig",
    "PARAM_GRIDS_DEFAULT",
]

try:
    import torch
except ImportError:
    torch = None


def _get_total_ram_bytes() -> int:
    """Total system RAM without psutil — uses /proc/meminfo (Linux) or os.sysconf."""
    try:
        with open("/proc/meminfo", encoding="ascii") as _f:
            for _line in _f:
                if _line.startswith("MemTotal:"):
                    return int(_line.split()[1]) * 1024
    except OSError:
        pass
    try:
        _page = os.sysconf("SC_PAGE_SIZE")
        _pages = os.sysconf("SC_PHYS_PAGES")
        if _page > 0 and _pages > 0:
            return _page * _pages
    except (AttributeError, ValueError, OSError):
        pass
    return 0  # caller falls back to 16 GB assumption


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Minimum total optimizer steps for DONUT convergence (empirical, per CLAUDE.md § 3).
# Fewer than this produces well-structured XML output but empty field content.
_MIN_OPTIMIZER_STEPS = 200

# Dataset size threshold below which small-dataset optimisations kick in.
# Above this threshold, effective-batch-16 configs produce sufficient steps
# even at 5 epochs (mini-mode): ceil(2000/16)*5 = 625 ≥ 200.
_SMALL_DATASET_THRESHOLD = 2000

# VRAM consumed per training sample at the reference image resolution (1280×960),
# measured empirically on a 24 GB RTX 4090: 22.78 GB / 8 samples = 2.8475 GB/sample.
# This constant is used to scale safe batch-size estimates when the actual image
# resolution differs from the reference (e.g. processor_config.json 2560×1920 = 4× pixels).
# With gradient_checkpointing_enable(), activation VRAM is ~halved.
# Empirically allows batch=4 on 24 GB. The optimizer will still cap at
# batch=2 conservatively — operator may override via ExperimentConfig.
_VRAM_PER_SAMPLE_AT_REF_GB = 2.848

# Reference image size (height, width) for VRAM calibration.  DONUT canonical input
# per CLAUDE.md § 2.  All VRAM estimates are anchored to this resolution.
_REF_IMAGE_SIZE = (1280, 960)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ResourceOptimizedConfig:
    """Recommended training configuration based on system resources."""

    batch_size: int
    gradient_accumulation_steps: int
    encoder_lr: float
    decoder_lr: float
    epochs: int
    warmup_steps: int
    early_stopping_patience: int
    config_explanation: str
    detected_vram_gb: float
    detected_ram_gb: float
    detected_cpu_cores: int


@dataclass
class SystemResources:
    """Detected system hardware capabilities."""

    vram_gb: float
    ram_gb: float
    cpu_cores: int
    cuda_available: bool
    device_name: str


# ---------------------------------------------------------------------------
# Resource Detection
# ---------------------------------------------------------------------------


def detect_system_resources() -> SystemResources:
    """Detect available GPU VRAM, RAM, and CPU cores.

    Returns:
        SystemResources with detected capabilities. Defaults to conservative
        values if detection fails.
    """
    vram_gb = 0.0
    cuda_available = False
    device_name = "CPU (CUDA unavailable)"

    if torch is not None:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Get VRAM for current device
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024**3)
            device_name = torch.cuda.get_device_name(0)

    # Detect RAM
    ram_bytes = _get_total_ram_bytes()
    ram_gb = ram_bytes / (1024**3) if ram_bytes > 0 else 16.0  # fallback: 16 GB

    # Detect CPU cores
    cpu_cores = os.cpu_count() or 4  # Fallback to 4 if detection fails

    return SystemResources(
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        cuda_available=cuda_available,
        device_name=device_name,
    )


def get_image_size_from_processor_config(
    config_path: str = "processor_config.json",
) -> tuple[int, int]:
    """Read image (height, width) from processor_config.json.

    Falls back to the reference resolution (1280, 960) if the file is absent
    or cannot be parsed.  A warning is logged when the fallback is triggered so
    that misconfigured paths are visible in the audit log.  Pass an explicit
    path when calling from a working directory other than the project root.

    Args:
        config_path: Path to processor_config.json (default: "processor_config.json").

    Returns:
        (height, width) tuple, e.g. (2560, 1920) for the current config.
    """
    import json

    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        size = cfg.get("image_processor", {}).get("size", {})
        h = size.get("height", _REF_IMAGE_SIZE[0])
        w = size.get("width", _REF_IMAGE_SIZE[1])
        return (int(h), int(w))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "get_image_size_from_processor_config: could not read %s (%s). "
            "Falling back to reference resolution %s.",
            config_path,
            exc,
            _REF_IMAGE_SIZE,
        )
        return _REF_IMAGE_SIZE


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optimize_hyperparams(
    num_train_samples: int,
    available_vram_gb: float | None = None,
    available_ram_gb: float | None = None,
    image_size: tuple[int, int] | None = None,
) -> ResourceOptimizedConfig:
    """Recommend optimal training hyperparameters based on system resources.

    Args:
        num_train_samples: Total number of training samples across all datasets
        available_vram_gb: GPU VRAM (GB). If None, auto-detect.
        available_ram_gb: System RAM (GB). If None, auto-detect.
        image_size: (height, width) of training images.  If None, reads from
            processor_config.json in the current working directory, falling
            back to the reference resolution (1280, 960) if absent.

    Returns:
        ResourceOptimizedConfig with all recommended hyperparameters and explanation.

    Reference:
        CLAUDE.md § 3. Fine-Tuning Method & Hyperparameter Optimization
        - Batch size 8 is optimal for 500-3940 samples per DONUT testing
        - Epochs = 10 (fixed per convergence analysis)
        - Warmup = 500 (fixed per CLAUDE.md LR schedule)
        - Learning rates: encoder 5e-5 / decoder 1e-4 (fixed, layerwise)
    """
    resources = detect_system_resources()

    # Use provided values or detected ones
    vram_gb = available_vram_gb if available_vram_gb is not None else resources.vram_gb
    ram_gb = available_ram_gb if available_ram_gb is not None else resources.ram_gb

    # Resolve actual image dimensions for VRAM scaling
    img_h, img_w = image_size if image_size is not None else get_image_size_from_processor_config()
    pixels_scale = (img_h * img_w) / (_REF_IMAGE_SIZE[0] * _REF_IMAGE_SIZE[1])

    explanation_parts = []

    # ───────────────────────────────────────────────────────────────────
    # Batch Size Optimization
    # ───────────────────────────────────────────────────────────────────
    #
    # Heuristic: Start with 8 (optimal per testing), then adjust for VRAM
    # and dataset size to avoid overfitting.

    if vram_gb < 8.0:
        batch_size = 4
        explanation_parts.append("batch_size=4: VRAM<8GB (low VRAM fallback)")
    elif vram_gb < 16.0:
        batch_size = 8
        explanation_parts.append("batch_size=8: VRAM 8-16GB (optimal per CLAUDE.md testing)")
    elif vram_gb <= 24.0:
        # 16-24 GB: DONUT at 960×1280 uses ~22.78 GB with batch=8, leaving
        # essentially zero headroom on 24 GB cards (e.g. RTX 4090).  Use
        # batch=2 with accumulation=8 to reach effective batch=16 safely
        # and prevent OOM kills observed at 40% through epoch 2.
        batch_size = 2
        explanation_parts.append(
            f"batch_size=2: VRAM {vram_gb:.1f}GB (≤24 GB); DONUT at 960×1280 "
            "requires ~22.78 GB at batch=8 — using batch=2 with "
            "accumulation_steps=8 (effective batch=16) to avoid OOM"
        )
    else:
        # >24 GB: Image-size-aware safe batch calculation.
        #
        # Root cause of Exp 8 OOM: processor_config.json specifies 2560×1920
        # (4× reference pixels), so hardcoded batch=16 required ~182 GB — far
        # beyond the 96 GB Blackwell card's capacity.
        #
        # Formula: vram_needed = _VRAM_PER_SAMPLE_AT_REF_GB × batch × pixels_scale
        # We require vram_needed ≤ vram_gb × 0.90 (10 % safety headroom).
        max_safe_batch = 1
        for b in (16, 8, 4, 2, 1):
            if _VRAM_PER_SAMPLE_AT_REF_GB * b * pixels_scale <= vram_gb * 0.90:
                max_safe_batch = b
                break

        if num_train_samples < _SMALL_DATASET_THRESHOLD:
            # Small dataset: cap at 4 to ensure enough optimizer steps.
            # batch=8 + accum=2 yields too few optimizer steps (~160 total) for
            # DONUT to converge on experiments with ~500 samples (Exp 1-4).
            batch_size = min(4, max_safe_batch)
            explanation_parts.append(
                f"batch_size={batch_size}: VRAM {vram_gb:.1f} GB, "
                f"image {img_h}×{img_w} (scale={pixels_scale:.2f}×), "
                f"small dataset ({num_train_samples} samples); "
                f"max_safe_batch={max_safe_batch} — capped at 4 for step count"
            )
        else:
            batch_size = min(16, max_safe_batch)
            explanation_parts.append(
                f"batch_size={batch_size}: VRAM {vram_gb:.1f} GB, "
                f"image {img_h}×{img_w} (scale={pixels_scale:.2f}×), "
                f"large dataset ({num_train_samples} samples); "
                f"max_safe_batch={max_safe_batch}"
            )

    # ───────────────────────────────────────────────────────────────────
    # Gradient Accumulation Steps
    # ───────────────────────────────────────────────────────────────────
    #
    # Heuristic: Effective batch = batch_size * accumulation_steps
    # If physical batch is small, accumulate more steps to reach effective batch~16

    if batch_size <= 2 and num_train_samples < _SMALL_DATASET_THRESHOLD and vram_gb <= 24.0:
        # Small dataset on 24 GB card: use accum=4 instead of accum=8.
        # With accum=8 (effective batch=16), mini-mode (epochs=5) produces only
        # ceil(500/16) × 5 = 160 optimizer steps — below _MIN_OPTIMIZER_STEPS.
        # Reducing to accum=4 (effective batch=8): ceil(500/8) × 5 = 315 ≥ 200 ✓
        # Memory is safe: DONUT at 960×1280 with batch=2 uses ~6 GB on 24 GB cards.
        accumulation_steps = 4
        explanation_parts.append(
            f"accumulation_steps=4: physical batch=2, small dataset ({num_train_samples} samples) "
            "on ≤24 GB card — effective batch=8 ensures ≥200 optimizer steps even at 5 epochs"
        )
    elif batch_size <= 2:
        accumulation_steps = 8
        explanation_parts.append(
            "accumulation_steps=8: physical batch=2, reaching effective batch=16"
        )
    elif batch_size <= 4:
        accumulation_steps = 4
        explanation_parts.append(
            "accumulation_steps=4: physical batch=4, reaching effective batch=16"
        )
    else:
        accumulation_steps = 2
        explanation_parts.append(
            f"accumulation_steps=2: physical batch={batch_size}, effective batch={batch_size * 2}"
        )

    # ───────────────────────────────────────────────────────────────────
    # Self-healing: ensure ≥ _MIN_OPTIMIZER_STEPS for DONUT convergence
    # ───────────────────────────────────────────────────────────────────
    #
    # Acts as a general backstop for any VRAM tier / dataset size combination
    # not already handled by the branches above.  Halves accumulation_steps
    # until total_steps ≥ _MIN_OPTIMIZER_STEPS or accum reaches 1.

    # epochs is the default fixed value (per CLAUDE.md § 3).  The self-healing
    # loop uses this as its reference epoch count — callers such as mini-mode
    # that override epochs at training time are protected by the explicit
    # small-dataset condition above (accum=4 on ≤24 GB cards).
    epochs = 10

    while accumulation_steps > 1:
        steps_per_epoch = math.ceil(num_train_samples / (batch_size * accumulation_steps))
        total_steps = steps_per_epoch * epochs
        if total_steps >= _MIN_OPTIMIZER_STEPS:
            break
        old_accum = accumulation_steps
        accumulation_steps //= 2
        explanation_parts.append(
            f"accumulation_steps reduced {old_accum}→{accumulation_steps} "
            f"(self-heal: {total_steps} steps < {_MIN_OPTIMIZER_STEPS} minimum with accum={old_accum})"
        )

    # ───────────────────────────────────────────────────────────────────
    # Fixed Hyperparameters (per CLAUDE.md § 3)
    # ───────────────────────────────────────────────────────────────────

    encoder_lr = 5e-5
    decoder_lr = 1e-4
    explanation_parts.append("encoder_lr=5e-5, decoder_lr=1e-4 (fixed per CLAUDE.md)")

    explanation_parts.append("epochs=10 (fixed per CLAUDE.md § 3 convergence analysis)")

    warmup_steps = 40
    explanation_parts.append(
        "warmup_steps=40 (fixed: 500 overflowed total steps on small datasets)"
    )

    # ───────────────────────────────────────────────────────────────────
    # Early Stopping Patience
    # ───────────────────────────────────────────────────────────────────
    #
    # For small datasets (<1000 samples), patience=3 is too tight: val loss
    # can plateau for 3 consecutive epochs in the early training phase (before
    # the model exits the XML-scaffolding phase) and then resume improving.
    # Example from Exp 1 with patience=3: training would stop at epoch 5 even
    # though val loss continues improving all the way to epoch 10 (0.5220 at
    # epoch 3 → 0.4353 at epoch 10). Using patience=5 gives the model enough
    # runway to pass early plateau phases on these small datasets.
    # For larger datasets (≥1000 samples), patience=3 is retained to prevent
    # overfitting.

    if num_train_samples < 1000:
        early_stopping_patience = 5
        explanation_parts.append(
            "early_stopping_patience=5 (small dataset <1000 samples; patience=3 fires "
            "too early when val loss plateaus and then resumes improving)"
        )
    else:
        early_stopping_patience = 3
        explanation_parts.append(
            "early_stopping_patience=3 (fixed to prevent overfitting on small val set)"
        )

    config_explanation = "; ".join(explanation_parts)

    return ResourceOptimizedConfig(
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        epochs=epochs,
        warmup_steps=warmup_steps,
        early_stopping_patience=early_stopping_patience,
        config_explanation=config_explanation,
        detected_vram_gb=vram_gb,
        detected_ram_gb=ram_gb,
        detected_cpu_cores=resources.cpu_cores,
    )


def validate_training_config(
    batch_size: int,
    gradient_accumulation_steps: int,
    num_train_samples: int,
    epochs: int,
    min_optimizer_steps: int = 200,
) -> None:
    """Validate that a training configuration produces enough optimizer steps.

    Raises ValueError if the effective number of optimizer steps is below
    *min_optimizer_steps* (default 200 — DONUT's empirical minimum to exit
    the XML scaffolding phase and learn field content).

    This catches the 'step starvation' bug where a large batch size on a
    high-VRAM GPU leaves too few gradient updates for the model to converge.

    Args:
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Number of accumulation steps.
        num_train_samples: Total training samples.
        epochs: Number of training epochs.
        min_optimizer_steps: Minimum acceptable total optimizer steps.

    Raises:
        ValueError: If total optimizer steps < min_optimizer_steps.
    """
    import math

    steps_per_epoch = math.ceil(num_train_samples / (batch_size * gradient_accumulation_steps))
    total_steps = steps_per_epoch * epochs
    effective_batch = batch_size * gradient_accumulation_steps
    if total_steps < min_optimizer_steps:
        raise ValueError(
            f"Training config produces only {total_steps} optimizer steps "
            f"({num_train_samples} samples / effective_batch={effective_batch} × {epochs} epochs). "
            f"Minimum required: {min_optimizer_steps}. "
            f"Reduce batch_size or increase gradient_accumulation_steps. "
            f"Current: batch_size={batch_size}, grad_accum={gradient_accumulation_steps}."
        )
    logger.info(
        "[validate_training_config] OK: %d optimizer steps "
        "(batch=%d, accum=%d, samples=%d, epochs=%d)",
        total_steps,
        batch_size,
        gradient_accumulation_steps,
        num_train_samples,
        epochs,
    )


# ---------------------------------------------------------------------------
# Audit Logging
# ---------------------------------------------------------------------------


class TrainingAuditLogger:
    """Persistent audit trail of training decisions and results.

    Appends structured entries to terminal.txt on each run, enabling historical
    comparison and data-driven refactoring decisions.

    Schema:
        [TIMESTAMP] RESOURCE_DETECTION
        GPU: <name> (<vram_gb> GB)
        RAM: <total_gb> GB total, <available_gb> GB available
        CPU: <cores> cores

        [TIMESTAMP] CONFIG_OPTIMIZATION (Exp <id>)
        - batch_size: <val> (<reason>)
        - accumulation_steps: <val> (<reason>)
        ...

        [TIMESTAMP] TRAINING_RESULT (Exp <id>)
        - F1: <val>
        - training_time: <sec>s (<min> min)
        - tokens_per_second: <val>
        - early_stopping_epoch: <val>
        - vs_baseline: <+/- percent>% F1
    """

    def __init__(self, append_to_file: str = "terminal.txt"):
        """Initialize audit logger.

        Args:
            append_to_file: Path to terminal.txt (appended, never overwritten)
        """
        self.log_path = Path(append_to_file)
        self._ensure_parent_dir()

    def _ensure_parent_dir(self) -> None:
        """Ensure parent directory exists."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        """Return current timestamp in ISO format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_resource_detection(self, resources: SystemResources) -> None:
        """Log detected hardware resources.

        Args:
            resources: SystemResources from detect_system_resources()
        """
        lines = [
            f"[{self._timestamp()}] RESOURCE_DETECTION",
            f"GPU: {resources.device_name} ({resources.vram_gb:.1f} GB)"
            if resources.cuda_available
            else "GPU: CPU-only (CUDA unavailable)",
            f"RAM: {resources.ram_gb:.1f} GB total",
            f"CPU: {resources.cpu_cores} cores",
            "",
        ]

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def log_config_decision(self, experiment_id: int, config: ResourceOptimizedConfig) -> None:
        """Log hyperparameter optimization decision.

        Args:
            experiment_id: Experiment number (1-8)
            config: ResourceOptimizedConfig with recommended hyperparameters
        """
        lines = [
            f"[{self._timestamp()}] CONFIG_OPTIMIZATION (Exp {experiment_id})",
            f"- batch_size: {config.batch_size} (effective={config.batch_size * config.gradient_accumulation_steps})",
            f"- accumulation_steps: {config.gradient_accumulation_steps}",
            f"- encoder_lr: {config.encoder_lr:.2e}",
            f"- decoder_lr: {config.decoder_lr:.2e}",
            f"- epochs: {config.epochs}",
            f"- warmup_steps: {config.warmup_steps}",
            f"- early_stopping_patience: {config.early_stopping_patience}",
            f"- reason: {config.config_explanation}",
            "",
        ]

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def log_training_result(
        self,
        experiment_id: int,
        global_f1: float,
        training_time_sec: float,
        tokens_per_second: float | None = None,
        early_stopping_epoch: int | None = None,
        baseline_f1: float | None = None,
    ) -> None:
        """Log training result after experiment completes.

        Args:
            experiment_id: Experiment number (1-8)
            global_f1: Global F1 score on test set
            training_time_sec: Total training time in seconds
            tokens_per_second: Optional throughput metric
            early_stopping_epoch: Epoch at which early stopping triggered
            baseline_f1: Optional baseline F1 for comparison (e.g., pretrained)
        """
        training_time_min = training_time_sec / 60.0

        lines = [
            f"[{self._timestamp()}] TRAINING_RESULT (Exp {experiment_id})",
            f"- F1: {global_f1:.4f}",
            f"- training_time: {training_time_sec:.0f}s ({training_time_min:.1f} min)",
        ]

        if tokens_per_second is not None:
            lines.append(f"- tokens_per_second: {tokens_per_second:.0f}")

        if early_stopping_epoch is not None:
            lines.append(f"- early_stopping_epoch: {early_stopping_epoch}")

        if baseline_f1 is not None and baseline_f1 > 0:
            improvement_pct = ((global_f1 - baseline_f1) / baseline_f1) * 100
            lines.append(f"- vs_baseline: {improvement_pct:+.1f}% F1 (baseline={baseline_f1:.4f})")

        lines.append("")

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def log_pipeline_summary(self, summary: str) -> None:
        """Log a summary message (free-form).

        Args:
            summary: Any string message to append
        """
        lines = [
            f"[{self._timestamp()}] SUMMARY",
            summary,
            "",
        ]

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test resource detection and optimization
    resources = detect_system_resources()
    print(
        f"Detected: {resources.vram_gb:.1f}GB VRAM, {resources.ram_gb:.1f}GB RAM, {resources.cpu_cores} CPU cores"
    )

    config = optimize_hyperparams(num_train_samples=500)
    print("\nOptimized config for 500 samples:")
    print(f"  batch_size={config.batch_size}, accumulation={config.gradient_accumulation_steps}")
    print(f"  epochs={config.epochs}, warmup={config.warmup_steps}")
    print(f"  encoder_lr={config.encoder_lr:.2e}, decoder_lr={config.decoder_lr:.2e}")
    print(f"\nReasoning: {config.config_explanation}")

    # Test audit logger
    logger_obj = TrainingAuditLogger("test_terminal.txt")
    logger_obj.log_resource_detection(resources)
    logger_obj.log_config_decision(1, config)
    logger_obj.log_training_result(
        experiment_id=1,
        global_f1=0.871,
        training_time_sec=1850,
        tokens_per_second=284,
        early_stopping_epoch=8,
        baseline_f1=0.82,
    )
    print("\nAudit log written to test_terminal.txt")


# ---------------------------------------------------------------------------
# TrainingConfig and PARAM_GRIDS_DEFAULT
# Absorbed from training_config.py — hyperparameter definitions for quick
# mode and sweep runs.  Kept in resource_optimizer because both classes
# concern themselves with training resource/parameter choices.
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Single hyperparameter configuration for DONUT training.

    These values are applied during training and appear in results.tex.

    See also: ResourceOptimizedConfig for hardware-adaptive overrides.
    """

    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"  # "cosine", "linear", or "constant"

    def validate(self) -> None:
        """Sanity-check hyperparameters before training starts."""
        if not (1 <= self.batch_size <= 64):
            raise ValueError(f"batch_size must be in range [1, 64], got {self.batch_size}")
        if not (1 <= self.epochs <= 100):
            raise ValueError(f"epochs must be in range [1, 100], got {self.epochs}")
        if not (1e-6 <= self.learning_rate <= 1e-2):
            raise ValueError(
                f"learning_rate must be in range [1e-6, 1e-2], got {self.learning_rate}"
            )
        if self.lr_scheduler_type not in ["linear", "cosine", "constant"]:
            raise ValueError(
                f"lr_scheduler_type must be one of ['linear', 'cosine', 'constant'], "
                f"got '{self.lr_scheduler_type}'"
            )

    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
        }


# Default parameter grids for hyperparameter sweep (--quick --all mode).
# Customise these values before running quick mode with a parameter sweep.
PARAM_GRIDS_DEFAULT: dict[str, list] = {
    "batch_sizes": [4, 8, 16],
    "epochs_list": [5, 10, 15],
    "learning_rates": [1e-5, 5e-5, 1e-4],
    "schedulers": ["linear", "cosine"],  # omit "constant" for speed
}
