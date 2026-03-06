"""
resource_optimizer.py — Dynamic hardware detection and hyperparameter optimization.

Detects available GPU VRAM, RAM, and CPU cores, then recommends optimal training
hyperparameters (batch size, gradient accumulation, learning rates, epochs) based
on system capabilities and dataset size.

Also provides TrainingAuditLogger to log configuration decisions and results to
a persistent terminal.txt file, enabling data-driven refactoring decisions.

CLAUDE.md Reference:
  - Optimal batch size: 8 (from testing); fallback 4 (low VRAM) or 16 (high VRAM)
  - Epochs: 10 (fixed per CLAUDE.md convergence analysis, NOT 30)
  - Warmup steps: 40 (fixed per CLAUDE.md LR schedule; see ExperimentConfig.warmup_steps)
  - Learning rates: encoder=5e-5, decoder=1e-4 (fixed, layerwise LR)
  - Early stopping patience: 3 (fixed)
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = [
    "ResourceOptimizedConfig",
    "SystemResources",
    "detect_system_resources",
    "optimize_hyperparams",
    "TrainingAuditLogger",
]

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)

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
    ram_gb = 0.0
    if psutil is not None:
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024**3)
    else:
        # Fallback: assume 16GB if psutil unavailable
        ram_gb = 16.0

    # Detect CPU cores
    cpu_cores = os.cpu_count() or 4  # Fallback to 4 if detection fails

    return SystemResources(
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        cuda_available=cuda_available,
        device_name=device_name,
    )


# ---------------------------------------------------------------------------
# Hyperparameter Optimization
# ---------------------------------------------------------------------------


def optimize_hyperparams(
    num_train_samples: int,
    available_vram_gb: float | None = None,
    available_ram_gb: float | None = None,
) -> ResourceOptimizedConfig:
    """Recommend optimal training hyperparameters based on system resources.

    Args:
        num_train_samples: Total number of training samples across all datasets
        available_vram_gb: GPU VRAM (GB). If None, auto-detect.
        available_ram_gb: System RAM (GB). If None, auto-detect.

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
        # >24 GB: Still cap at 16 for safety (Exp 8 uses ~3940 samples; batch=32 overfits)
        if num_train_samples < 2000:
            # Small dataset: use batch=4 + accum=4 (same as ≤24 GB path).
            # batch=8 + accum=2 yields too few optimizer steps (~160 total) for
            # DONUT to converge on experiments with ~500 samples (Exp 1-4).
            batch_size = 4
            explanation_parts.append(
                f"batch_size=4: VRAM {vram_gb:.1f}GB but small dataset "
                f"({num_train_samples} samples); using batch=4+accum=4 "
                "(same as ≤24GB path) to ensure sufficient optimizer steps"
            )
        else:
            batch_size = 16
            explanation_parts.append(
                f"batch_size=16: VRAM {vram_gb:.1f}GB with large dataset ({num_train_samples} samples)"
            )

    # ───────────────────────────────────────────────────────────────────
    # Gradient Accumulation Steps
    # ───────────────────────────────────────────────────────────────────
    #
    # Heuristic: Effective batch = batch_size * accumulation_steps
    # If physical batch is small, accumulate more steps to reach effective batch~16

    if batch_size <= 2:
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
    # Fixed Hyperparameters (per CLAUDE.md § 3)
    # ───────────────────────────────────────────────────────────────────

    encoder_lr = 5e-5
    decoder_lr = 1e-4
    explanation_parts.append("encoder_lr=5e-5, decoder_lr=1e-4 (fixed per CLAUDE.md)")

    epochs = 10
    explanation_parts.append("epochs=10 (fixed per CLAUDE.md § 3 convergence analysis)")

    warmup_steps = 40
    explanation_parts.append(
        "warmup_steps=40 (fixed: 500 overflowed total steps on small datasets)"
    )

    # ───────────────────────────────────────────────────────────────────
    # Early Stopping Patience
    # ───────────────────────────────────────────────────────────────────
    #
    # Conservative: patience=3 (default). Can increase on very high VRAM
    # to allow more val-loss monitoring, but Exp 1 shows overfitting at >8 epochs anyway.

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
