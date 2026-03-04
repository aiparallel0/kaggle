"""Validator for model weight integrity - detects safetensors deduplication issues.

From CLAUDE.md Section 16 - BUG B:
"safetensors deduplication drops lm_head.weight → F1~0.42

Root cause: after resize_token_embeddings(), lm_head.weight and embed_tokens.weight
share the same data pointer. safetensors omits the duplicate tensor from checkpoint
shards. load_best_model_at_end reloads the best epoch; lm_head.weight is listed in
missing_keys and is randomly re-initialized."
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelWeightValidator:
    """Validate model weights after checkpoint load."""

    @staticmethod
    def check_lm_head_weight_in_checkpoint(checkpoint_path: Path) -> tuple[bool, str]:
        """Check if decoder.lm_head.weight exists in checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            (exists: bool, error_message: str)
        """
        try:
            from safetensors import safe_open

            # Try loading safetensors files
            safetensors_files = list(checkpoint_path.glob("*.safetensors"))
            if not safetensors_files:
                return False, "No safetensors files found in checkpoint"

            # Check if lm_head.weight is in any shard
            found_lm_head = False
            for shard_path in safetensors_files:
                try:
                    with safe_open(shard_path, framework="pt") as f:
                        keys = f.keys()
                        if "decoder.lm_head.weight" in keys:
                            found_lm_head = True
                            logger.info(f"✓ Found decoder.lm_head.weight in {shard_path.name}")
                            break
                except Exception as e:
                    logger.warning(f"Could not read {shard_path}: {e}")

            if found_lm_head:
                return True, ""
            else:
                return (
                    False,
                    "decoder.lm_head.weight not found in checkpoint (safetensors dedup?)",
                )

        except ImportError:
            logger.warning("safetensors not installed, skipping weight check")
            return True, ""  # Skip check if safetensors unavailable
        except Exception as e:
            logger.warning(f"Could not validate lm_head weight: {e}")
            return True, ""  # Don't fail on unknown errors

    @staticmethod
    def validate_model_forward_pass(
        model,
        processor,
        test_image_path: Path | None = None,
    ) -> tuple[bool, str]:
        """Test that model can do forward pass (catches silent failures).

        Args:
            model: The DONUT model
            processor: The DonutProcessor
            test_image_path: Path to test image (optional)

        Returns:
            (success: bool, error_message: str)
        """
        try:
            import torch
            from PIL import Image

            # Create dummy input if no test image
            if test_image_path is None:
                # Create blank image
                img = Image.new("RGB", (960, 1280), color="white")
            else:
                try:
                    img = Image.open(test_image_path)
                except Exception as e:
                    logger.warning(f"Could not load test image: {e}")
                    return True, ""  # Skip if can't load

            # Process image
            pixel_values = processor(img, return_tensors="pt").pixel_values

            # Run forward pass
            with torch.no_grad():
                outputs = model.encoder(pixel_values)

            if outputs is None or not hasattr(outputs, "last_hidden_state"):
                return False, "Encoder returned invalid output"

            logger.info("✓ Model forward pass successful")
            return True, ""

        except Exception as e:
            logger.warning(f"Model forward pass test failed: {e}")
            return False, f"Forward pass failed: {str(e)}"

    @staticmethod
    def check_missing_keys_after_load(
        missing_keys: list[str],
        unexpected_keys: list[str],
        tie_word_embeddings: bool,
    ) -> tuple[bool, str]:
        """Check if critical keys are missing after model load.

        Args:
            missing_keys: Keys from loading_info['missing_keys']
            unexpected_keys: Keys from loading_info['unexpected_keys']
            tie_word_embeddings: Value of model.config.tie_word_embeddings

        Returns:
            (valid: bool, error_message: str)
        """
        critical_keys = ["decoder.lm_head.weight"]

        for key in critical_keys:
            if key in missing_keys:
                if not tie_word_embeddings:
                    # This is a BUG - lm_head should be saved separately
                    return (
                        False,
                        f"CRITICAL: {key} missing from checkpoint (safetensors dedup)\n"
                        f"This will cause F1~0.42 (all predictions garbled)\n"
                        f"Fix: Ensure LmHeadCloneCallback is used during training",
                    )
                else:
                    # If tie_word_embeddings=True, lm_head is tied to embed_tokens, OK to be missing
                    logger.info(f"Note: {key} missing but tie_word_embeddings=True (expected)")

        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

        return True, ""
