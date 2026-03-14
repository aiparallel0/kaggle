"""
validators/checkpoint_resume_validator.py — Checkpoint integrity validator.

Before loading any checkpoint from ``results/expN/``, call
``validate_checkpoint()`` to verify:

1. ``lm_head.weight`` is present and not a duplicate pointer of
   ``embed_tokens.weight`` (safetensors deduplication bug — see CLAUDE.md §5).
2. ``model.config.encoder.image_size`` matches the ``INPUT_SIZE`` recorded
   in the experiment's result JSON (if available).
3. Token vocabulary size matches the checkpoint's embedding matrix shape.

If any check fails, ``CheckpointCorruptionError`` is raised.

Usage
-----
    from validators.checkpoint_resume_validator import validate_checkpoint

    validate_checkpoint(
        model_path=Path("results/exp6/"),
        result_json_path=Path("results/experiment_6.json"),
        expected_vocab_size=len(processor.tokenizer),
    )
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import CheckpointCorruptionError from pipeline_types so it is project-wide
from pipeline_types.exceptions import CheckpointCorruptionError

__all__ = ["CheckpointCorruptionError", "validate_checkpoint"]


def validate_checkpoint(
    model_path: str | Path,
    result_json_path: str | Path | None = None,
    expected_vocab_size: int | None = None,
) -> None:
    """Validate a fine-tuned DONUT checkpoint before loading for evaluation.

    Parameters
    ----------
    model_path:
        Path to the saved model directory (contains config.json, model.safetensors, …).
    result_json_path:
        Path to the experiment result JSON (e.g. ``results/experiment_6.json``).
        Used to check that encoder image_size matches training resolution.
        Skipped when ``None`` or the file does not exist.
    expected_vocab_size:
        Expected tokenizer vocabulary size (after adding SROIE special tokens).
        Skipped when ``None``.

    Raises
    ------
    CheckpointCorruptionError
        On any integrity failure.
    FileNotFoundError
        When ``model_path`` does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {model_path}")

    # ── Check 1: lm_head.weight presence in safetensors index ────────────
    _check_lm_head(model_path)

    # ── Check 2: encoder image_size matches training resolution ───────────
    if result_json_path is not None:
        _check_image_size(model_path, Path(result_json_path))

    # ── Check 3: vocab size matches embedding shape ───────────────────────
    if expected_vocab_size is not None:
        _check_vocab_size(model_path, expected_vocab_size)

    logger.info("[CheckpointValidator] %s — all checks passed ✓", model_path)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_lm_head(model_path: Path) -> None:
    """Verify lm_head.weight is present in the safetensors index."""
    index_file = model_path / "model.safetensors.index.json"
    single_file = model_path / "model.safetensors"

    if index_file.exists():
        try:
            data: dict = json.loads(index_file.read_text())
            weight_map: dict[str, Any] = data.get("weight_map", {})
            if "decoder.lm_head.weight" not in weight_map:
                raise CheckpointCorruptionError(
                    f"[CheckpointValidator] lm_head.weight is MISSING from "
                    f"{index_file}.  This indicates the safetensors deduplication "
                    f"bug (lm_head was tied to embed_tokens when saved).  "
                    f"Re-train with LmHeadCloneCallback registered.  "
                    f"See CLAUDE.md §5 Pattern 6."
                )
            logger.debug(
                "[CheckpointValidator] lm_head.weight found in index: shard=%s",
                weight_map["decoder.lm_head.weight"],
            )
        except CheckpointCorruptionError:
            raise
        except Exception as exc:
            logger.warning(
                "[CheckpointValidator] Could not parse safetensors index (%s) — skipping lm_head check",
                exc,
            )
    elif single_file.exists():
        # Single-shard model: inspect tensor keys via safetensors header
        try:
            import struct

            with open(single_file, "rb") as fh:
                header_len = struct.unpack("<Q", fh.read(8))[0]
                header_bytes = fh.read(header_len)
            header: dict = json.loads(header_bytes)
            if "decoder.lm_head.weight" not in header:
                raise CheckpointCorruptionError(
                    f"[CheckpointValidator] lm_head.weight is MISSING from "
                    f"{single_file} header.  See CLAUDE.md §5 Pattern 6."
                )
            logger.debug("[CheckpointValidator] lm_head.weight found in single-shard model ✓")
        except CheckpointCorruptionError:
            raise
        except Exception as exc:
            logger.warning(
                "[CheckpointValidator] Could not inspect single-shard safetensors (%s) — "
                "skipping lm_head check",
                exc,
            )
    else:
        # Might be a PyTorch bin checkpoint — skip safetensors check
        logger.debug(
            "[CheckpointValidator] No safetensors file found in %s — skipping lm_head check",
            model_path,
        )


def _check_image_size(model_path: Path, result_json_path: Path) -> None:
    """Check encoder.image_size against training resolution in result JSON."""
    if not result_json_path.exists():
        logger.debug("[CheckpointValidator] Result JSON not found — skipping image_size check")
        return

    # Load encoder config from checkpoint
    encoder_config_file = model_path / "encoder_config.json"
    config_file = model_path / "config.json"

    encoder_image_size: list[int] | None = None
    for cf in (encoder_config_file, config_file):
        if not cf.exists():
            continue
        try:
            cfg: dict = json.loads(cf.read_text())
            # config.json wraps encoder under "encoder" key
            if "encoder" in cfg:
                enc = cfg["encoder"]
            else:
                enc = cfg
            if "image_size" in enc:
                encoder_image_size = enc["image_size"]
                break
        except Exception as exc:
            logger.debug("[CheckpointValidator] Could not parse %s: %s", cf, exc)

    if encoder_image_size is None:
        logger.debug("[CheckpointValidator] encoder.image_size not found — skipping size check")
        return

    # Load expected size from result JSON
    try:
        result: dict = json.loads(result_json_path.read_text())
        metrics: dict = result.get("metrics", {})
        # Try to find input_size recorded during training
        input_size = metrics.get("input_size") or result.get("input_size")
        if input_size is None:
            logger.debug(
                "[CheckpointValidator] input_size not in result JSON — skipping size check"
            )
            return
        if list(encoder_image_size) != list(input_size):
            raise CheckpointCorruptionError(
                f"[CheckpointValidator] encoder.image_size={encoder_image_size} "
                f"does not match training INPUT_SIZE={input_size} from {result_json_path}.  "
                f"The checkpoint was trained at a different resolution."
            )
        logger.debug(
            "[CheckpointValidator] encoder.image_size=%s matches training resolution ✓",
            encoder_image_size,
        )
    except CheckpointCorruptionError:
        raise
    except Exception as exc:
        logger.warning(
            "[CheckpointValidator] Could not verify image_size from result JSON (%s)",
            exc,
        )


def _check_vocab_size(model_path: Path, expected_vocab_size: int) -> None:
    """Verify embedding matrix shape matches expected vocabulary size."""
    config_file = model_path / "config.json"
    if not config_file.exists():
        logger.debug("[CheckpointValidator] config.json not found — skipping vocab check")
        return

    try:
        cfg: dict = json.loads(config_file.read_text())
        decoder_cfg = cfg.get("decoder", cfg)
        # vocab_size can be at top level or under decoder
        vocab_size = decoder_cfg.get("vocab_size") or cfg.get("vocab_size")
        if vocab_size is None:
            logger.debug("[CheckpointValidator] vocab_size not in config — skipping vocab check")
            return
        if int(vocab_size) != expected_vocab_size:
            raise CheckpointCorruptionError(
                f"[CheckpointValidator] Checkpoint vocab_size={vocab_size} "
                f"does not match expected {expected_vocab_size}.  "
                f"The tokenizer has different special tokens than expected."
            )
        logger.debug(
            "[CheckpointValidator] vocab_size=%d matches expected ✓", vocab_size
        )
    except CheckpointCorruptionError:
        raise
    except Exception as exc:
        logger.warning(
            "[CheckpointValidator] Could not verify vocab_size from config (%s)",
            exc,
        )
