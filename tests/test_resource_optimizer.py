"""Unit tests for resource_optimizer.py guardrails.

Tests that the optimizer step validation and hyperparameter logic
prevent the 'step starvation' bug that caused DONUT to produce
empty predictions on high-VRAM GPUs.

All tests are CPU-only (no GPU required).
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resource_optimizer import (
    ResourceOptimizedConfig,
    optimize_hyperparams,
    validate_training_config,
)


class TestValidateTrainingConfig:
    """Tests for validate_training_config()."""

    def test_sufficient_steps_passes(self):
        """500 samples, batch=4, accum=4, epochs=10 → 312 steps — should pass."""
        validate_training_config(
            batch_size=4,
            gradient_accumulation_steps=4,
            num_train_samples=500,
            epochs=10,
        )

    def test_step_starvation_raises(self):
        """500 samples, batch=16, accum=2, epochs=10 → 160 steps — should raise."""
        with pytest.raises(ValueError, match="optimizer steps"):
            validate_training_config(
                batch_size=16,
                gradient_accumulation_steps=2,
                num_train_samples=500,
                epochs=10,
            )

    def test_large_dataset_high_batch_passes(self):
        """3940 samples, batch=8, accum=2, epochs=10 → 2462 steps — should pass."""
        validate_training_config(
            batch_size=8,
            gradient_accumulation_steps=2,
            num_train_samples=3940,
            epochs=10,
        )

    def test_custom_min_steps(self):
        """Custom min_optimizer_steps respected."""
        # 312 steps passes default min=200 but fails min=400
        with pytest.raises(ValueError):
            validate_training_config(
                batch_size=4,
                gradient_accumulation_steps=4,
                num_train_samples=500,
                epochs=10,
                min_optimizer_steps=400,
            )

    def test_error_message_contains_batch_info(self):
        """Error message must contain actionable batch/accum info."""
        with pytest.raises(ValueError) as exc_info:
            validate_training_config(
                batch_size=16,
                gradient_accumulation_steps=2,
                num_train_samples=500,
                epochs=10,
            )
        msg = str(exc_info.value)
        assert "batch_size=16" in msg
        assert "grad_accum=2" in msg
        assert "optimizer steps" in msg


class TestOptimizeHyperparamsHighVRAM:
    """Tests for optimize_hyperparams() on high-VRAM GPUs (>24 GB).

    This pins the fix for the step starvation bug on A100/H100 class GPUs.
    """

    def _call(self, num_train_samples: int, vram_gb: float = 40.0) -> ResourceOptimizedConfig:
        return optimize_hyperparams(
            num_train_samples=num_train_samples,
            available_vram_gb=vram_gb,
            available_ram_gb=64.0,
        )

    def _optimizer_steps(
        self, cfg: ResourceOptimizedConfig, num_samples: int, epochs: int = 10
    ) -> int:
        return math.ceil(num_samples / (cfg.batch_size * cfg.gradient_accumulation_steps)) * epochs

    def test_small_dataset_high_vram_produces_enough_steps(self):
        """Exp 1-4 scenario: ~500 samples on A100 must yield >= 200 optimizer steps."""
        cfg = self._call(num_train_samples=500, vram_gb=40.0)
        steps = self._optimizer_steps(cfg, num_samples=500)
        assert steps >= 200, (
            f"Step starvation bug: only {steps} optimizer steps for 500 samples on 40 GB GPU. "
            f"batch_size={cfg.batch_size}, grad_accum={cfg.gradient_accumulation_steps}"
        )

    def test_small_dataset_high_vram_does_not_use_batch16_accum2(self):
        """The step-starvation config (batch=16, accum=2) must NOT be returned for small datasets."""
        cfg = self._call(num_train_samples=500, vram_gb=40.0)
        regressed = cfg.batch_size == 16 and cfg.gradient_accumulation_steps == 2
        assert not regressed, (
            "Regressed to batch_size=16, gradient_accumulation_steps=2 for small dataset on "
            "high-VRAM GPU. This produces ~160 optimizer steps — too few for DONUT to converge."
        )

    def test_large_dataset_high_vram_can_use_larger_batch(self):
        """Exp 8 scenario: ~3940 samples on A100 should use batch >= 4."""
        cfg = self._call(num_train_samples=3940, vram_gb=40.0)
        assert cfg.batch_size >= 4

    def test_80gb_vram_small_dataset(self):
        """H100 (80 GB) with small dataset must still avoid step starvation."""
        cfg = self._call(num_train_samples=500, vram_gb=80.0)
        steps = self._optimizer_steps(cfg, num_samples=500)
        assert steps >= 200, f"Step starvation on H100: only {steps} steps for 500 samples"

    def test_returns_resource_optimized_config(self):
        """Return type must be ResourceOptimizedConfig."""
        cfg = self._call(num_train_samples=500)
        assert isinstance(cfg, ResourceOptimizedConfig)

    def test_all_four_fields_nonzero(self):
        """All four training-relevant fields must be populated."""
        cfg = self._call(num_train_samples=500)
        assert cfg.batch_size > 0
        assert cfg.gradient_accumulation_steps > 0
        assert cfg.encoder_lr > 0.0
        assert cfg.decoder_lr > 0.0


class TestOptimizeHyperparamsLowVRAM:
    """Tests for low-VRAM paths (RTX 4090 and below)."""

    def test_rtx4090_batch_is_2(self):
        """RTX 4090 (24 GB) must use batch_size=2 to prevent OOM."""
        cfg = optimize_hyperparams(
            num_train_samples=500,
            available_vram_gb=24.0,
            available_ram_gb=32.0,
        )
        assert cfg.batch_size == 2, (
            f"RTX 4090 path returned batch_size={cfg.batch_size}, expected 2"
        )

    def test_low_vram_8gb(self):
        """8 GB GPU must use batch_size=4."""
        cfg = optimize_hyperparams(
            num_train_samples=500,
            available_vram_gb=7.0,
            available_ram_gb=16.0,
        )
        assert cfg.batch_size == 4


class TestExperimentConfigImmutability:
    """Tests that ExperimentConfig global state is never mutated."""

    def test_dataclasses_replace_does_not_mutate_original(self):
        """dataclasses.replace() must not change the original config."""
        import dataclasses

        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        from run_experiments import EXPERIMENTS  # noqa: E402, I001

        original_exp1 = EXPERIMENTS[1]
        original_batch = original_exp1.batch_size

        # Simulate what run_experiments.py does
        new_config = dataclasses.replace(original_exp1, batch_size=999)

        assert EXPERIMENTS[1].batch_size == original_batch, (
            f"INVARIANT VIOLATION: EXPERIMENTS[1].batch_size changed from "
            f"{original_batch} to {EXPERIMENTS[1].batch_size} after dataclasses.replace(). "
            "Direct attribute mutation detected."
        )
        assert new_config.batch_size == 999

    def test_direct_mutation_is_possible_without_replace(self):
        """Confirm Python allows direct mutation (so the guardrail is necessary)."""
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        from run_experiments import ExperimentConfig  # noqa: E402, I001

        cfg = ExperimentConfig(name="test", datasets=["sroie"], batch_size=8)
        cfg.batch_size = 99  # This is allowed by Python — the guardrail prevents it in practice
        assert cfg.batch_size == 99  # Confirms the danger is real
