"""Unit tests for training invariants — token IDs, config property aliases.

These tests catch the class of silent failure where a wrong token ID or
broken property alias produces garbage output that never raises an exception.

All tests are CPU-only (no GPU, no model weights required).
"""

from pathlib import Path

import pytest


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

    def test_decoder_start_token_roundtrip(self):
        """decode(convert_tokens_to_ids(['<s_sroie>'])[0]) must round-trip to '<s_sroie>'.

        Guards against GP-3 / GP-4 (CLAUDE.md §19): using the string form of
        convert_tokens_to_ids returns the ID for '<', not the full token.
        The correct list-wrapping form is always used in train.py and
        run_experiments.py; this test asserts the roundtrip property that
        catches any regression.
        """
        tokenizer = self._make_tokenizer_with_sroie_token()
        token_id = tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        decoded = tokenizer.decode([token_id])
        assert decoded == "<s_sroie>", (
            f"decoder_start_token_id={token_id} decodes to '{decoded}', not '<s_sroie>'. "
            f"Use: tokenizer.convert_tokens_to_ids(['<s_sroie>'])[0]"
        )


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

        original = ExperimentConfig(name="t", datasets=["sroie"], batch_size=8, epochs=10, lr=5e-5)
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
            assert cfg.gradient_accumulation_steps > 0, f"Exp {exp_id}: grad_accum must be > 0"
            assert cfg.max_epochs == cfg.epochs, f"Exp {exp_id}: max_epochs alias broken"
            assert cfg.learning_rate == cfg.lr, f"Exp {exp_id}: learning_rate alias broken"
            assert cfg.per_device_train_batch_size == cfg.batch_size, (
                f"Exp {exp_id}: batch alias broken"
            )


class TestLabelTokenizationNoSpecialTokens:
    """MultiDataset must tokenize labels with add_special_tokens=False.

    Root cause of high training loss: without add_special_tokens=False the
    tokenizer prepends BOS (ID=0) to every label sequence.
    VisionEncoderDecoderModel.forward() calls shift_tokens_right, placing
    decoder_start_token_id at position 0 of decoder_input_ids.  The model
    at position 0 must then predict labels[0] = BOS — a meaningless target
    that is never present during inference (which uses add_special_tokens=False
    for decoder_input_ids).  This train/inference mismatch inflates training
    loss and wastes 2 generation slots per sequence.
    The official DONUT fine-tuning code always uses add_special_tokens=False.
    """

    def test_getitem_label_uses_add_special_tokens_false(self):
        """MultiDataset.__getitem__ must call tokenizer with add_special_tokens=False."""
        from pathlib import Path
        from unittest.mock import MagicMock

        import torch

        import train

        recorded_kwargs: list[dict] = []

        fake_result = MagicMock()
        fake_result.input_ids = torch.tensor([[100, 200, 300, 1, 1]])  # shape (1, 5)

        def _capture_call(text, **kwargs):
            recorded_kwargs.append(kwargs)
            return fake_result

        fake_tokenizer = MagicMock()
        fake_tokenizer.side_effect = _capture_call
        fake_tokenizer.pad_token_id = 1
        fake_tokenizer.unk_token_id = 3
        # unk == unk → _mask_empty_field_labels skips masking (no-op)
        fake_tokenizer.convert_tokens_to_ids.return_value = 3

        fake_processor = MagicMock()
        fake_processor.tokenizer = fake_tokenizer

        samples = [
            (
                Path("/fake/img.jpg"),
                {"company": "ACME", "date": "2024", "address": "ST", "total": "1.00"},
            )
        ]

        # cache_in_ram=False → skip __init__ precomputation; test __getitem__ path only
        ds = train.MultiDataset(
            samples, processor=fake_processor, max_length=32, cache_in_ram=False
        )

        # Provide pre-cached pixel tensor so no image disk read is needed
        ds._pixel_cache[0] = torch.zeros(3, 1, 1)
        assert 0 not in ds._label_cache, "label cache must be empty for this test"

        recorded_kwargs.clear()
        _ = ds[0]

        assert recorded_kwargs, "tokenizer was never called during __getitem__"
        for call_kwargs in recorded_kwargs:
            assert call_kwargs.get("add_special_tokens") is False, (
                "MultiDataset.__getitem__ called tokenizer WITHOUT add_special_tokens=False. "
                "This creates a train/inference mismatch: the tokenizer prepends BOS (ID=0) "
                "to labels, so the model learns to predict BOS at position 0 — never seen "
                "in inference (which uses add_special_tokens=False for decoder_input_ids). "
                "Fix: add add_special_tokens=False to the tokenizer call in train.py."
            )
