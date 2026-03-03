"""
training_config.py — Hyperparameter definitions for quick mode and sweeps.

This module defines the TrainingConfig dataclass and default parameter grids
used by quick mode and hyperparameter sweeps. Users can customize these values
to test different training configurations.

See also: pipeline_config.py — CloudConfig for cloud/orchestration settings
          (API keys, Ollama URL, git branch, etc.).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingConfig:
    """Single hyperparameter configuration for DONUT training.

    These values are applied during training and appear in results.tex.
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

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "lr_scheduler_type": self.lr_scheduler_type,
        }


# Default parameter grids for hyperparameter sweep (--quick -all mode)
# Customize these values before running quick mode with parameter sweep
PARAM_GRIDS_DEFAULT = {
    "batch_sizes": [4, 8, 16],  # Test: 4, 8, 16
    "epochs_list": [5, 10, 15],  # Test: 5, 10, 15
    "learning_rates": [1e-5, 5e-5, 1e-4],  # Test: 1e-5, 5e-5, 1e-4
    "schedulers": ["linear", "cosine"],  # Test: linear, cosine (skip constant for speed)
}
