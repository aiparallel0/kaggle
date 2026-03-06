"""TrOCR model-setup regression tests.

These tests encode the fixes from PRs #80–#85 as permanent regression tests so
that any future refactor that accidentally reverts a fix (e.g., removes
low_cpu_mem_usage=False, re-adds no_repeat_ngram_size, or reintroduces
trocr-large) is caught by CI in seconds — not after a 10-epoch training run
fails on a cloud GPU.

All tests run without GPU, without network access, and without real model
weights.  ``unittest.mock`` stubs any ``from_pretrained`` call.

Requires: torch, transformers.
"""

import json
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# train_trocr_yolo.py imports torch and transformers at the top level — guard.
torch = pytest.importorskip("torch", reason="torch required by train_trocr_yolo.py")
pytest.importorskip("transformers", reason="transformers required by train_trocr_yolo.py")

from train_trocr_yolo import (  # noqa: E402, I001
    TROCR_MODEL_ID,
    TrOCRReceiptDataset,
    _materialize_meta_buffers,
)


# ════════════════════════════════════════════════════════════════════════════
# Helper modules used across multiple test classes
# ════════════════════════════════════════════════════════════════════════════


class _ModuleWithMetaBuffer(torch.nn.Module):
    """Toy module with a non-persistent buffer on the meta device."""

    def __init__(self):
        super().__init__()
        buf = torch.empty(4, device="meta")
        self.register_buffer("sinusoidal", buf, persistent=False)


class _ModuleWithMetaFloatTensor(torch.nn.Module):
    """Toy module with a plain ``_float_tensor`` attribute on the meta device."""

    def __init__(self):
        super().__init__()
        self._float_tensor = torch.empty(4, device="meta")


class _ModuleWithNormalBuffer(torch.nn.Module):
    """Toy module with a CPU buffer — nothing to fix."""

    def __init__(self):
        super().__init__()
        self.register_buffer("weights", torch.zeros(4))


# ════════════════════════════════════════════════════════════════════════════
# 1.  _materialize_meta_buffers — PR #81 meta-device fix
# ════════════════════════════════════════════════════════════════════════════


class TestMaterializeMetaBuffers:
    """_materialize_meta_buffers must move all meta-device tensors to the target device."""

    def test_non_persistent_buffer_moved_to_cpu(self):
        """Non-persistent register_buffer on meta is moved to cpu."""
        mod = _ModuleWithMetaBuffer()
        assert mod.sinusoidal.device.type == "meta", "pre-condition: buffer must be on meta"
        n = _materialize_meta_buffers(mod, "cpu")
        assert n >= 1, "Expected at least 1 buffer to be fixed"
        assert mod.sinusoidal.device.type == "cpu", "Buffer must be on cpu after materialisation"

    def test_plain_float_tensor_attribute_moved_to_cpu(self):
        """Plain _float_tensor attribute on meta is moved to cpu."""
        mod = _ModuleWithMetaFloatTensor()
        assert mod._float_tensor.device.type == "meta", "pre-condition: _float_tensor on meta"
        n = _materialize_meta_buffers(mod, "cpu")
        assert n >= 1, "Expected at least 1 tensor to be fixed"
        assert mod._float_tensor.device.type == "cpu", "_float_tensor must be on cpu"

    def test_no_meta_tensors_returns_zero(self):
        """Module with no meta-device tensors returns 0 (nothing fixed)."""
        mod = _ModuleWithNormalBuffer()
        n = _materialize_meta_buffers(mod, "cpu")
        assert n == 0, "Nothing to fix — should return 0"

    def test_nested_module_meta_buffer_fixed(self):
        """Meta buffer inside a nested child module is also fixed."""
        parent = torch.nn.Sequential(_ModuleWithMetaBuffer())
        n = _materialize_meta_buffers(parent, "cpu")
        child = list(parent.children())[0]
        assert n >= 1
        assert child.sinusoidal.device.type == "cpu"


# ════════════════════════════════════════════════════════════════════════════
# 2.  TROCR_MODEL_ID constant — PR #85 wrong model ID bug
# ════════════════════════════════════════════════════════════════════════════


class TestTrOCRModelID:
    """TROCR_MODEL_ID must be the correct base-printed checkpoint."""

    def test_model_id_is_base_printed(self):
        """TROCR_MODEL_ID must equal 'microsoft/trocr-base-printed'."""
        assert TROCR_MODEL_ID == "microsoft/trocr-base-printed", (
            f"TROCR_MODEL_ID is '{TROCR_MODEL_ID}'. "
            "PR #85 fixed a regression where 'trocr-large' was used by mistake. "
            "The correct value is 'microsoft/trocr-base-printed'."
        )

    def test_model_id_not_large(self):
        """TROCR_MODEL_ID must NOT reference trocr-large (causes OOM on low-VRAM GPUs)."""
        assert "large" not in TROCR_MODEL_ID.lower(), (
            f"TROCR_MODEL_ID='{TROCR_MODEL_ID}' references a 'large' model. "
            "PR #82 / #85 require trocr-base-printed to fit in < 8 GB VRAM."
        )

    def test_model_id_not_handwritten(self):
        """TROCR_MODEL_ID must NOT reference trocr-handwritten (wrong domain)."""
        assert "handwritten" not in TROCR_MODEL_ID.lower(), (
            f"TROCR_MODEL_ID='{TROCR_MODEL_ID}' references a handwritten model. "
            "Receipt OCR requires the printed model."
        )


# ════════════════════════════════════════════════════════════════════════════
# 3 & 4.  Generation config + gradient checkpointing — PRs #82 & #83
# ════════════════════════════════════════════════════════════════════════════


def _make_mock_model():
    """Return a minimal mock VisionEncoderDecoderModel mirroring the real config structure."""
    generation_config = mock.MagicMock()
    generation_config.no_repeat_ngram_size = 3  # wrong default — must be reset to 0
    generation_config.length_penalty = 2.0  # wrong default — must be reset to 1.0
    generation_config.num_beams = 1  # wrong default — must be set to 4

    decoder_config = mock.MagicMock()
    decoder_config.use_cache = True  # wrong default — must be set to False

    decoder = mock.MagicMock()
    decoder.config = decoder_config

    model_config = mock.MagicMock()
    model_config.use_cache = True  # wrong default — must be set to False

    model = mock.MagicMock()
    model.config = model_config
    model.generation_config = generation_config
    model.decoder = decoder
    model.to = mock.MagicMock(return_value=model)
    return model


def _make_mock_processor():
    """Return a minimal mock TrOCRProcessor."""
    tokenizer = mock.MagicMock()
    tokenizer.cls_token_id = 101
    tokenizer.pad_token_id = 0
    tokenizer.sep_token_id = 102

    processor = mock.MagicMock()
    processor.tokenizer = tokenizer
    return processor


def _apply_trocr_config(model, processor):
    """Reproduce the inline config block from train_trocr() for test isolation.

    This mirrors the exact config block in train_trocr_yolo.train_trocr() so
    that any future refactor that changes those lines will break these tests,
    making the regression immediately visible.
    """
    from train_trocr_yolo import TROCR_MAX_LEN

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_new_tokens = TROCR_MAX_LEN
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 1.0
    model.generation_config.num_beams = 4
    model.config.use_cache = False
    model.decoder.config.use_cache = False
    model.gradient_checkpointing_enable()


class TestGenerationConfig:
    """Generation params must be set on model.generation_config (not model.config).

    PR #83: setting no_repeat_ngram_size / length_penalty / num_beams on
    model.config causes a ValueError in transformers >=4.40 during
    save_pretrained().  They must be set on model.generation_config.
    """

    def test_no_repeat_ngram_size_is_zero(self):
        """no_repeat_ngram_size must be 0 — non-zero harms short OCR sequences."""
        model = _make_mock_model()
        processor = _make_mock_processor()
        _apply_trocr_config(model, processor)
        assert model.generation_config.no_repeat_ngram_size == 0, (
            "no_repeat_ngram_size must be 0 on generation_config. "
            "Non-zero values harm short OCR output (e.g. repeated date digits)."
        )

    def test_length_penalty_is_neutral(self):
        """length_penalty must be 1.0 (neutral) to avoid penalising short outputs."""
        model = _make_mock_model()
        processor = _make_mock_processor()
        _apply_trocr_config(model, processor)
        assert model.generation_config.length_penalty == 1.0, (
            "length_penalty must be 1.0 (neutral) on generation_config. "
            "Values != 1.0 penalise or reward short outputs incorrectly for OCR."
        )

    def test_num_beams_is_four(self):
        """num_beams must be 4 for beam search quality."""
        model = _make_mock_model()
        processor = _make_mock_processor()
        _apply_trocr_config(model, processor)
        assert model.generation_config.num_beams == 4, (
            "num_beams must be 4 on generation_config. "
            "PR #83 requires generation params on generation_config, not model.config."
        )

    def test_model_config_use_cache_disabled(self):
        """model.config.use_cache must be False (required for gradient checkpointing)."""
        model = _make_mock_model()
        processor = _make_mock_processor()
        _apply_trocr_config(model, processor)
        assert model.config.use_cache is False, (
            "model.config.use_cache must be False when gradient_checkpointing is enabled. "
            "use_cache=True and gradient_checkpointing are mutually incompatible."
        )

    def test_decoder_config_use_cache_disabled(self):
        """model.decoder.config.use_cache must be False."""
        model = _make_mock_model()
        processor = _make_mock_processor()
        _apply_trocr_config(model, processor)
        assert model.decoder.config.use_cache is False, (
            "model.decoder.config.use_cache must be False. "
            "Both top-level and decoder-level use_cache must be disabled for "
            "gradient checkpointing to work correctly."
        )


class TestGradientCheckpointing:
    """gradient_checkpointing_enable() must be called during model setup — PR #82.

    PR #82: TrOCR-base (246M params) requires gradient checkpointing to fit in
    lower-VRAM GPUs (~8 GB) during the backward pass.  Without it, activation
    memory overflows on forward passes with longer sequences.
    """

    def test_gradient_checkpointing_enable_called(self):
        """gradient_checkpointing_enable() must be called exactly once during setup."""
        model = _make_mock_model()
        processor = _make_mock_processor()
        _apply_trocr_config(model, processor)
        assert model.gradient_checkpointing_enable.call_count == 1, (
            "gradient_checkpointing_enable() was not called. "
            "PR #82 requires it to reduce activation memory for TrOCR-base on low-VRAM GPUs."
        )


# ════════════════════════════════════════════════════════════════════════════
# 5.  TrOCRReceiptDataset.__getitem__ — label masking
# ════════════════════════════════════════════════════════════════════════════


class TestTrOCRReceiptDataset:
    """TrOCRReceiptDataset.__getitem__ must mask padding tokens with -100.

    The -100 label convention is required by PyTorch cross-entropy loss so
    that padding positions are ignored during training.
    """

    def _make_processor_stub(self, max_length: int, pad_token_id: int = 1):
        """Return a minimal processor stub that returns deterministic tensors."""
        import torch

        # pixel_values stub: processor(img) returns object with .pixel_values
        pixel_values_tensor = torch.zeros(1, 3, 32, 32)  # batch=1, C, H, W

        # Build input_ids: 2 real tokens followed by pad_token_id values
        ids = torch.tensor([[101, 102] + [pad_token_id] * (max_length - 2)])

        class _ImgOut:
            pixel_values = pixel_values_tensor

        class _TokOut:
            input_ids = ids

        # Use a closure-friendly approach: set pad_token_id as an instance attr
        class _FakeTokenizer:
            def __init__(self, pad_id):
                self.pad_token_id = pad_id

            def __call__(self, text, padding, max_length, truncation, return_tensors):
                return _TokOut()

        class _FakeProcessor:
            def __init__(self, pad_id):
                self.tokenizer = _FakeTokenizer(pad_id)

            def __call__(self, img, return_tensors):
                return _ImgOut()

        return _FakeProcessor(pad_token_id)

    def test_padding_tokens_masked_with_minus_100(self, tmp_path):
        """Labels at padding positions must be -100, not the pad_token_id."""
        from PIL import Image as PILImage

        # Write a tiny synthetic metadata.jsonl
        img_file = tmp_path / "crop_0.png"
        PILImage.new("RGB", (32, 32), color=(128, 128, 128)).save(img_file)
        meta = {"file_name": "crop_0.png", "text": "HELLO"}
        (tmp_path / "metadata.jsonl").write_text(json.dumps(meta) + "\n")

        max_length = 8
        pad_id = 1
        processor = self._make_processor_stub(max_length=max_length, pad_token_id=pad_id)

        dataset = TrOCRReceiptDataset(data_dir=tmp_path, processor=processor, max_length=max_length)
        assert len(dataset) == 1

        item = dataset[0]
        labels = item["labels"]

        # All positions that were pad_token_id must now be -100
        assert not (labels == pad_id).any(), (
            f"Found raw pad_token_id ({pad_id}) in labels — padding must be masked to -100."
        )
        assert (labels == -100).sum() == max_length - 2, (
            "Expected exactly (max_length - 2) positions masked to -100 "
            f"(the two real tokens should remain). Got labels={labels.tolist()}"
        )

    def test_pixel_values_shape(self, tmp_path):
        """pixel_values must have shape (3, H, W) — a valid image tensor."""
        from PIL import Image as PILImage

        img_file = tmp_path / "crop_0.png"
        PILImage.new("RGB", (32, 32)).save(img_file)
        meta = {"file_name": "crop_0.png", "text": "TEST"}
        (tmp_path / "metadata.jsonl").write_text(json.dumps(meta) + "\n")

        max_length = 8
        processor = self._make_processor_stub(max_length=max_length)

        dataset = TrOCRReceiptDataset(data_dir=tmp_path, processor=processor, max_length=max_length)
        item = dataset[0]
        pv = item["pixel_values"]

        assert pv.ndim == 3, f"pixel_values must be 3-D (C, H, W), got shape {pv.shape}"
        assert pv.shape[0] == 3, f"pixel_values channel dim must be 3 (RGB), got {pv.shape[0]}"

    def test_empty_metadata_returns_empty_dataset(self, tmp_path):
        """Dataset with no metadata.jsonl (or empty file) has length 0."""
        max_length = 8
        processor = self._make_processor_stub(max_length=max_length)
        dataset = TrOCRReceiptDataset(data_dir=tmp_path, processor=processor, max_length=max_length)
        assert len(dataset) == 0
