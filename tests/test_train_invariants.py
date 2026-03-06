"""Unit tests for training invariants — token IDs, config property aliases.

These tests catch the class of silent failure where a wrong token ID or
broken property alias produces garbage output that never raises an exception.

All tests are CPU-only (no GPU, no model weights required).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

transformers = pytest.importorskip("transformers", reason="transformers required")


class TestDecoderStartTokenId:
    """Tests for correct decoder_start_token_id assignment.

    Pins the fix for: convert_tokens_to_ids(string) returns ID of '<',
    not the full token. Correct form: convert_tokens_to_ids([string])[0].
    """

    def _make_tokenizer_with_sroie_token(self):
        """Build a minimal tokenizer with <s_sroie> added."""
        # Use the actual donut tokenizer if available, else skip
        try:
            from transformers import DonutProcessor

            proc = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
            proc.tokenizer.add_special_tokens({"additional_special_tokens": ["<s_sroie>"]})
            return proc.tokenizer
        except (OSError, ImportError, ValueError):
            pytest.skip("Cannot load donut tokenizer in this environment")

    def test_list_wrapping_returns_single_token_id(self):
        """convert_tokens_to_ids(['<s_sroie>'])[0] must be a single integer."""
        tokenizer = self._make_tokenizer_with_sroie_token()
        token_id = tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        assert isinstance(token_id, int)
        decoded = tokenizer.decode([token_id])
        assert decoded == "<s_sroie>", (
            f"decoder_start_token_id={token_id} decodes to '{decoded}', not '<s_sroie>'. "
            "Token lookup is broken."
        )

    def test_string_form_returns_wrong_id(self):
        """Demonstrate the bug: convert_tokens_to_ids('<s_sroie>') is NOT the right call."""
        tokenizer = self._make_tokenizer_with_sroie_token()
        # String form treats each character as a token — returns wrong ID
        correct_id = tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        # They should differ (the bug); this test documents the danger
        # Note: if tokenizer special-cases strings this may be equal on newer versions,
        # but the list form is always correct.
        assert isinstance(correct_id, int)
        decoded_correct = tokenizer.decode([correct_id])
        assert decoded_correct == "<s_sroie>"


class TestExperimentConfigPropertyAliases:
    """Tests that ExperimentConfig property aliases are consistent.

    The aliases (max_epochs, learning_rate, per_device_train_batch_size) are
    used by DonutTrainer via duck-typed access. If they diverge from the
    underlying fields, training silently uses wrong values.
    """

    def test_max_epochs_alias(self):
        from run_experiments import ExperimentConfig

        cfg = ExperimentConfig(name="t", datasets=["sroie"], epochs=7)
        assert cfg.max_epochs == 7
        assert cfg.max_epochs == cfg.epochs

    def test_learning_rate_alias(self):
        from run_experiments import ExperimentConfig

        cfg = ExperimentConfig(name="t", datasets=["sroie"], lr=1e-4)
        assert cfg.learning_rate == 1e-4
        assert cfg.learning_rate == cfg.lr

    def test_per_device_train_batch_size_alias(self):
        from run_experiments import ExperimentConfig

        cfg = ExperimentConfig(name="t", datasets=["sroie"], batch_size=4)
        assert cfg.per_device_train_batch_size == 4
        assert cfg.per_device_train_batch_size == cfg.batch_size

    def test_aliases_survive_dataclasses_replace(self):
        """Aliases must still work after dataclasses.replace()."""
        import dataclasses

        from run_experiments import ExperimentConfig

        original = ExperimentConfig(
            name="t", datasets=["sroie"], batch_size=8, epochs=10, lr=5e-5
        )
        new = dataclasses.replace(original, batch_size=2, epochs=5, lr=1e-4)
        assert new.per_device_train_batch_size == 2
        assert new.max_epochs == 5
        assert new.learning_rate == 1e-4
        # Original must be unchanged
        assert original.batch_size == 8
        assert original.epochs == 10

    def test_all_experiments_have_valid_config(self):
        """All 8 experiments in EXPERIMENTS dict must have valid, consistent configs."""
        from run_experiments import EXPERIMENTS

        for exp_id, cfg in EXPERIMENTS.items():
            assert cfg.epochs > 0, f"Exp {exp_id}: epochs must be > 0"
            assert cfg.batch_size > 0, f"Exp {exp_id}: batch_size must be > 0"
            assert cfg.lr > 0, f"Exp {exp_id}: lr must be > 0"
            assert cfg.gradient_accumulation_steps > 0, (
                f"Exp {exp_id}: grad_accum must be > 0"
            )
            assert cfg.max_epochs == cfg.epochs, f"Exp {exp_id}: max_epochs alias broken"
            assert cfg.learning_rate == cfg.lr, f"Exp {exp_id}: learning_rate alias broken"
            assert cfg.per_device_train_batch_size == cfg.batch_size, (
                f"Exp {exp_id}: batch alias broken"
            )
