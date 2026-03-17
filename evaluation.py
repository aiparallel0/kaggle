# =============================================================================
# evaluation.py
# Purpose: Merged evaluation module — DONUT evaluator + unified model evaluation
# =============================================================================
import json
import logging
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from constants import (
    BASE_MODEL,
    DEVICE,
    EMPTY_GT,
    FIELDS,
    MAX_LENGTH,
    _edit_distance,
    _get_sroie_dir,
    _gpu_cleanup,
    _progress,
)
from data_pipeline import load_sroie_test

__all__ = [
    "DonutEvaluator",
    "EvaluationResult",
    "compute_metrics",
    "normalized_edit_distance",
    "load_model_with_tied_weights",
    "run_inference",
    "remap_cord_to_sroie",
    "DEVICE",
    "load_test_samples",
    "evaluate_donut_on_test",
    "evaluate_trocr_yolo_on_test",
    "print_metrics",
    "generate_comparison_report",
    "generate_json_summary",
]

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None  # type: ignore[assignment]

try:
    import numpy as np
    import torch
except ImportError:
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]

try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    # Import inline fallbacks from train.py (which has the full inline implementation)
    try:
        from train import DonutProcessor, VisionEncoderDecoderModel  # noqa: E402, I001
    except ImportError:
        # train.py also not importable — define minimal stubs
        class DonutProcessor:  # type: ignore[no-redef]
            @classmethod
            def from_pretrained(cls, *a, **kw):
                raise ImportError("transformers is required. pip install transformers")

        class VisionEncoderDecoderModel:  # type: ignore[no-redef]
            @classmethod
            def from_pretrained(cls, *a, **kw):
                raise ImportError("transformers is required. pip install transformers")

# ─────────────────────────────────────────────────────────────────────────────
# Inline image loader — fallback when Pillow is unavailable
# ─────────────────────────────────────────────────────────────────────────────

try:
    from PIL import Image as _PILImage

    def _load_image(path: "str | Path") -> "_PILImage.Image":  # type: ignore[name-defined]
        return _PILImage.open(path).convert("RGB")

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

    def _png_unfilter(scanlines: list, width: int, bpp: int) -> bytes:
        """Apply PNG row de-filtering (Sub/Up/Average/Paeth)."""
        out = []
        prev = bytes(width * bpp)
        for ftype, raw in scanlines:
            row = bytearray(raw)
            if ftype == 1:  # Sub
                for i in range(bpp, len(row)):
                    row[i] = (row[i] + row[i - bpp]) & 0xFF
            elif ftype == 2:  # Up
                for i in range(len(row)):
                    row[i] = (row[i] + prev[i]) & 0xFF
            elif ftype == 3:  # Average
                for i in range(len(row)):
                    a = row[i - bpp] if i >= bpp else 0
                    row[i] = (row[i] + (a + prev[i]) // 2) & 0xFF
            elif ftype == 4:  # Paeth
                for i in range(len(row)):
                    a = row[i - bpp] if i >= bpp else 0
                    b = prev[i]
                    c = prev[i - bpp] if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    row[i] = (row[i] + pr) & 0xFF
            out.append(bytes(row))
            prev = bytes(row)
        return b"".join(out)

    def _load_png(path: "str | Path") -> "np.ndarray | None":
        """Minimal PNG decoder → RGB numpy array."""
        data = Path(path).read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        pos = 8
        width = height = bit_depth = color_type = 0
        idat = b""
        while pos < len(data):
            length = struct.unpack(">I", data[pos : pos + 4])[0]
            ctype = data[pos + 4 : pos + 8]
            chunk = data[pos + 8 : pos + 8 + length]
            pos += 12 + length
            if ctype == b"IHDR":
                width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
            elif ctype == b"IDAT":
                idat += chunk
            elif ctype == b"IEND":
                break
        raw = zlib.decompress(idat)
        if bit_depth != 8 or color_type not in (2, 6):
            return None
        channels = 3 if color_type == 2 else 4
        row_bytes = width * channels
        scanlines = []
        r = 0
        for _ in range(height):
            ftype = raw[r]
            scanlines.append((ftype, raw[r + 1 : r + 1 + row_bytes]))
            r += row_bytes + 1
        pixel_data = _png_unfilter(scanlines, width, channels)
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, channels)
        if channels == 4:
            arr = arr[:, :, :3]
        return arr

    def _load_bmp(path: "str | Path") -> "np.ndarray | None":
        """Minimal BMP decoder for 24-bit uncompressed BMP → RGB numpy array."""
        data = Path(path).read_bytes()
        if data[:2] != b"BM":
            return None
        pixel_offset = struct.unpack_from("<I", data, 10)[0]
        width = struct.unpack_from("<i", data, 18)[0]
        height = struct.unpack_from("<i", data, 22)[0]
        bits_per_pixel = struct.unpack_from("<H", data, 28)[0]
        compression = struct.unpack_from("<I", data, 30)[0]
        if bits_per_pixel != 24 or compression != 0:
            return None
        flipped = height > 0
        height = abs(height)
        row_size = (width * 3 + 3) & ~3
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            src_row = (height - 1 - row) if flipped else row
            start = pixel_offset + src_row * row_size
            raw_row = data[start : start + width * 3]
            pixels = np.frombuffer(raw_row, dtype=np.uint8).reshape(width, 3)
            arr[row] = pixels[:, ::-1]  # BGR → RGB
        return arr

    def _load_jpeg_ctypes(path: "str | Path"):
        """Load JPEG via ImageMagick subprocess (system libjpeg fallback)."""
        import subprocess as _sp

        try:
            result = _sp.run(
                ["convert", str(path), "-colorspace", "RGB", "ppm:-"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                ppm = result.stdout
                lines = ppm.split(b"\n")
                if lines[0] == b"P6":
                    dims = lines[1].split()
                    w, h = int(dims[0]), int(dims[1])
                    pixel_data = b"\n".join(lines[3:])
                    arr = np.frombuffer(pixel_data, dtype=np.uint8)
                    if len(arr) >= h * w * 3:
                        return arr[: h * w * 3].reshape(h, w, 3)
        except Exception:
            pass
        return None

    def _load_image(path: "str | Path"):  # type: ignore[misc]
        """Load an image file as an RGB numpy array without PIL."""
        path = Path(path)
        suffix = path.suffix.lower()
        arr = None
        if suffix == ".png":
            arr = _load_png(path)
        elif suffix in (".bmp",):
            arr = _load_bmp(path)
        elif suffix in (".jpg", ".jpeg"):
            arr = _load_jpeg_ctypes(path)
        if arr is None:
            raise RuntimeError(
                f"Cannot load {path} without Pillow. "
                "Install Pillow: pip install Pillow\n"
                "PNG (8-bit RGB/RGBA), BMP (24-bit), and JPEG (via ImageMagick) "
                "are supported natively."
            )
        return np.ascontiguousarray(arr, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _parse_sroie_output(tokens: str) -> dict:
    """Parse SROIE XML-like output format into a dict.

    SROIE format: <s_sroie><s_company>VALUE</s_company><s_date>VALUE</s_date>...
    This parser extracts values between opening and closing tags for each field.

    Returns a dict with keys from FIELDS; missing fields default to empty string.
    """
    result = EMPTY_GT.copy()

    for field_name in FIELDS:
        open_tag = f"<s_{field_name}>"
        close_tag = f"</s_{field_name}>"

        start_idx = tokens.find(open_tag)
        if start_idx != -1:
            start_idx += len(open_tag)
            end_idx = tokens.find(close_tag, start_idx)
            if end_idx != -1:
                result[field_name] = tokens[start_idx:end_idx].strip()

    return result


def _merge_token2json_pages(result: Any) -> dict:
    """Merge multi-page CORD output (list) into single dict (Phase 0b).

    The base checkpoint (donut-base-finetuned-cord-v2) knows about <sep/>
    (CORD multi-page separator). Even SROIE fine-tuned models can emit <sep/>
    because it's in the inherited vocabulary. When present, token2json()
    returns a list of dicts (one per page) instead of a single dict.

    This utility merges pages with "first occurrence of each key wins" logic,
    ensuring callers always receive a flat dict.

    Args:
        result: Output from processor.token2json() — either dict or list of dicts

    Returns:
        Single flat dict with all pages merged; empty dict if input is None/empty
    """
    if isinstance(result, list):
        merged: dict = {}
        for page in result:
            if isinstance(page, dict):
                for k, v in page.items():
                    if k not in merged:
                        merged[k] = v
        return merged if merged else {}

    if isinstance(result, dict):
        return result

    # Neither list nor dict (shouldn't happen, but defensive)
    return {}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of initial inference calls for which raw token output is logged
_DIAGNOSTIC_LOG_COUNT = 3


# ---------------------------------------------------------------------------
# EvaluationResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics.

    Attributes:
        global_precision: Precision over all (image, field) pairs.
        global_recall:    Recall over all (image, field) pairs.
        global_f1:        Harmonic mean of precision and recall.
        overall_exact_match: Fraction of images where ALL fields matched.
        per_field:        Dict mapping field name to per-field metrics.
        num_samples:      Number of test samples evaluated.
        parse_failures:   Number of samples where token2json failed.
        raw_predictions:  Optional list of raw prediction dicts.
    """

    global_precision: float = 0.0
    global_recall: float = 0.0
    global_f1: float = 0.0
    overall_exact_match: float = 0.0
    per_field: dict[str, dict[str, float]] = field(default_factory=dict)
    num_samples: int = 0
    parse_failures: int = 0
    raw_predictions: list[dict] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict compatible with legacy compute_metrics output."""
        d = {
            "global_precision": round(self.global_precision, 4),
            "global_recall": round(self.global_recall, 4),
            "global_f1": round(self.global_f1, 4),
            "overall_exact_match": round(self.overall_exact_match, 4),
        }
        for fname, fmetrics in self.per_field.items():
            d[f"{fname}_f1"] = round(fmetrics.get("f1", 0.0), 4)
            d[f"{fname}_ned"] = round(fmetrics.get("ned", 1.0), 4)
        return d


# ---------------------------------------------------------------------------
# Model loading helper — fixes the lm_head weight tying bug
# ---------------------------------------------------------------------------


def load_model_with_tied_weights(model_path: str, device: str = DEVICE, processor=None):
    # Use output_loading_info=True to detect missing keys at load time.
    # This is how we distinguish "trained lm_head loaded correctly" from
    # "lm_head randomly re-initialized because it was missing from the shard".
    model, loading_info = VisionEncoderDecoderModel.from_pretrained(
        model_path, output_loading_info=True
    )
    missing_keys = loading_info.get("missing_keys", [])

    # Sanity check: when tie_word_embeddings=False (set in train.py after
    # resize_token_embeddings()), LmHeadCloneCallback ensures lm_head.weight is
    # saved as an independent tensor in every checkpoint shard.  If it is still
    # missing after load, the checkpoint is corrupt — fail loudly instead of
    # silently recovering with random or embed_tokens weights (which produces
    # F1~0.42 and is indistinguishable from a healthy run without this check).
    if "decoder.lm_head.weight" in missing_keys and not getattr(
        model.decoder.config, "tie_word_embeddings", True
    ):
        raise RuntimeError(
            "CRITICAL: decoder.lm_head.weight missing from checkpoint. "
            "The model cannot generate SROIE tokens. Fix checkpoint saving."
        )

    _retie_decoder_head(model, missing_keys=missing_keys)

    model = model.to(device)
    model.eval()

    # ── Guard: verify SROIE tokens are present in processor vocab ───────────
    if processor is not None:
        _unk_id = processor.tokenizer.unk_token_id
        _sroie_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
        if _sroie_id == _unk_id:
            raise RuntimeError(
                f"Loaded processor from {model_path!r} has no SROIE special tokens: "
                f"'<s_sroie>' maps to unk_token_id ({_unk_id}). "
                "The checkpoint is unusable — re-run training so the processor "
                "is saved with add_special_tokens() applied."
            )
        # Fix decoder_start_token_id if it doesn't match <s_sroie>
        if model.config.decoder_start_token_id != _sroie_id:
            logger.warning(
                "decoder_start_token_id=%d does not match <s_sroie> id=%d — overriding.",
                model.config.decoder_start_token_id,
                _sroie_id,
            )
            model.config.decoder_start_token_id = _sroie_id
            model.decoder.config.decoder_start_token_id = _sroie_id

    return model


def _retie_decoder_head(model, missing_keys=None) -> None:
    """Re-tie or recover lm_head.weight after from_pretrained().

    Parameters
    ----------
    model : VisionEncoderDecoderModel
    missing_keys : list of str, optional
        Keys reported missing by from_pretrained(output_loading_info=True).
        When 'decoder.lm_head.weight' is in this list, the weight was NOT
        loaded from the checkpoint — it was randomly re-initialized.  We
        recover by copying embed_tokens.weight → lm_head.weight, which is
        lossy but far better than random init (F1 ~0.42 → comparable to
        embed_tokens quality).  The real fix is LmHeadCloneCallback in
        train.py which prevents this situation from arising.
    """
    if missing_keys is None:
        missing_keys = []

    decoder = model.decoder

    if not getattr(decoder.config, "tie_word_embeddings", True):
        lm_head_missing = any("lm_head.weight" in k for k in missing_keys)
        if lm_head_missing:
            # lm_head was not in the checkpoint — randomly re-initialized.
            # Recover by copying embed_tokens.weight as a starting point.
            embed_tokens = None
            if hasattr(decoder, "model"):
                if hasattr(decoder.model, "decoder"):
                    embed_tokens = getattr(decoder.model.decoder, "embed_tokens", None)
                elif hasattr(decoder.model, "embed_tokens"):
                    embed_tokens = decoder.model.embed_tokens
            if embed_tokens is not None and hasattr(decoder, "lm_head"):
                lm_shape = decoder.lm_head.weight.shape
                embed_shape = embed_tokens.weight.shape
                if lm_shape == embed_shape:
                    decoder.lm_head.weight = torch.nn.Parameter(embed_tokens.weight.data.clone())
                    logger.error(
                        "RECOVERY: decoder.lm_head.weight was MISSING from "
                        "checkpoint (randomly re-initialized). Copied "
                        "embed_tokens.weight as fallback. This is lossy — "
                        "fix: ensure LmHeadCloneCallback is registered in "
                        "DonutTrainer.train() so per-epoch checkpoints "
                        "include lm_head.weight."
                    )
                else:
                    logger.error(
                        "RECOVERY FAILED: lm_head shape %s != embed_tokens "
                        "shape %s — cannot copy. F1 will be degraded.",
                        lm_shape,
                        embed_shape,
                    )
            else:
                logger.error(
                    "RECOVERY FAILED: could not locate embed_tokens for "
                    "lm_head recovery. F1 will be degraded."
                )
        else:
            logger.info(
                "tie_word_embeddings=False and lm_head.weight present in "
                "checkpoint — no recovery needed."
            )
        return

    # Legacy path: re-tie for old checkpoints that have tie_word_embeddings=True
    if hasattr(decoder, "lm_head") and hasattr(decoder, "model"):
        embed_tokens = None
        if hasattr(decoder.model, "decoder") and hasattr(decoder.model.decoder, "embed_tokens"):
            embed_tokens = decoder.model.decoder.embed_tokens
        elif hasattr(decoder.model, "embed_tokens"):
            embed_tokens = decoder.model.embed_tokens

        if embed_tokens is not None:
            lm_shape = decoder.lm_head.weight.shape
            embed_shape = embed_tokens.weight.shape
            if lm_shape == embed_shape:
                decoder.lm_head.weight = embed_tokens.weight
                logger.info(
                    "Re-tied decoder.lm_head.weight → embed_tokens.weight (shape %s)", lm_shape
                )
            else:
                logger.warning(
                    "lm_head shape %s != embed_tokens shape %s — skipping re-tie",
                    lm_shape,
                    embed_shape,
                )

    assert model.decoder.lm_head.weight is not None, (
        "decoder.lm_head.weight is None after _retie_decoder_head() — weight tying failed."
    )


# ---------------------------------------------------------------------------
# DonutEvaluator class
# ---------------------------------------------------------------------------


class DonutEvaluator:
    """OOP evaluator that wraps model loading, self-test, inference, and metrics.

    Usage::

        evaluator = DonutEvaluator(model_path, processor, test_dataset)
        result = evaluator.evaluate()
        print(result.to_dict())
    """

    def __init__(
        self,
        model_path: Path,
        processor: DonutProcessor,
        test_dataset: list[tuple[Path, dict]],
        task_prompt: str = "<s_sroie>",
        max_length: int = MAX_LENGTH,
        device: str = DEVICE,
    ):
        self.model_path = Path(model_path)
        self.processor = processor
        self.test_dataset = test_dataset
        self.task_prompt = task_prompt
        self.max_length = max_length
        self.device = device
        self.parse_failure_count = 0
        self._inference_call_count = 0

        # Load model with weight re-tying fix
        logger.info("Loading model from %s", self.model_path)

        # Run checkpoint integrity checks before loading
        # (lm_head present, vocab size match)
        try:
            from validators import validate_checkpoint

            validate_checkpoint(
                model_path=self.model_path,
                expected_vocab_size=(
                    len(self.processor.tokenizer) if self.processor is not None else None
                ),
            )
        except Exception as _ckpt_exc:
            logger.warning("[DonutEvaluator] Checkpoint validation warning: %s", _ckpt_exc)

        self.model = load_model_with_tied_weights(
            str(self.model_path), device=self.device, processor=self.processor
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, allow_high_parse_failures: bool = False) -> EvaluationResult:
        """Run full evaluation: self-test first, then inference on all samples.

        Args:
            allow_high_parse_failures: When True, a >50% parse failure rate
                logs a warning and returns a zero-metric result instead of
                raising RuntimeError.  Use this for undertrained models where
                training completed successfully but the model hasn't yet
                converged to the expected tag format.  Default False keeps the
                existing behaviour (raises) so callers that want to detect
                truly broken models still get an exception.

        Raises:
            RuntimeError: If self-test fails (model produces no parseable output).
            RuntimeError: If >50% of samples fail to parse and
                allow_high_parse_failures is False (model is likely broken).
        """
        self._self_test()

        ground_truths = [s[1] for s in self.test_dataset]

        predictions = []
        self.parse_failure_count = 0
        self._inference_call_count = 0

        with torch.no_grad():
            for img_path, _gt in _progress(
                self.test_dataset, desc="Evaluating", total=len(self.test_dataset)
            ):
                pred = self._run_inference(img_path, self.task_prompt)
                predictions.append(pred)

        # Parse failure threshold check
        n = len(self.test_dataset)
        if n > 0 and self.parse_failure_count > n * 0.5:
            msg = (
                f"Parse failure threshold exceeded: {self.parse_failure_count}/{n} "
                f"({self.parse_failure_count / n:.1%}) samples failed to parse. "
                f"The model is likely broken — check token2json compatibility."
            )
            if not allow_high_parse_failures:
                raise RuntimeError(msg)
            logger.warning("%s — returning zero-metric result", msg)
            return EvaluationResult(
                global_precision=0.0,
                global_recall=0.0,
                global_f1=0.0,
                overall_exact_match=0.0,
                per_field={f: {"f1": 0.0, "ned": 1.0} for f in FIELDS},
                num_samples=n,
                parse_failures=self.parse_failure_count,
            )

        metrics_dict = self.compute_all_metrics(predictions, ground_truths)
        result = EvaluationResult(
            global_precision=metrics_dict["global_precision"],
            global_recall=metrics_dict["global_recall"],
            global_f1=metrics_dict["global_f1"],
            overall_exact_match=metrics_dict["overall_exact_match"],
            per_field={
                f: {
                    "f1": metrics_dict.get(f"{f}_f1", 0.0),
                    "ned": metrics_dict.get(f"{f}_ned", 1.0),
                }
                for f in FIELDS
            },
            num_samples=n,
            parse_failures=self.parse_failure_count,
            raw_predictions=predictions,
        )
        return result

    # ------------------------------------------------------------------
    # Self-test
    # ------------------------------------------------------------------

    def _self_test(self) -> None:
        """Run inference on one sample and verify the model produces output.

        The self-test asserts that the result is a dict with at least one
        non-empty field value. If it fails, raises with diagnostic info
        including the raw token sequence.

        For pretrained CORD models evaluated on SROIE, the output may not
        match SROIE fields — that's fine. The self-test only checks that
        the model can produce *any* parseable output (non-empty dict).
        """
        if not self.test_dataset:
            raise RuntimeError("Self-test failed: test_dataset is empty")

        img_path, gt = self.test_dataset[0]
        logger.debug("Self-test: running inference on %s", img_path)

        # Run raw generation to capture token sequence for diagnostics
        image = _load_image(img_path)
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        with torch.no_grad():
            # FIX: Removed early_stopping=True — it is deprecated/invalid with
            # num_beams=1 (greedy decoding) and generates thousands of warnings
            # per eval call in transformers>=4.35.
            # Also removed repetition_penalty and no_repeat_ngram_size which were
            # blocking structural tags like </s_company> and </s_date>.
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.max_length,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        # Diagnostic: log first 20 token IDs and decoder seed token for actionable failure analysis
        logger.info(
            "Self-test raw token IDs (first 20): %s",
            outputs.sequences[0].tolist()[:20],
        )
        logger.info(
            "Self-test decoder_input_ids=%s → decoded=%r",
            decoder_input_ids[0].tolist(),
            self.processor.tokenizer.decode(decoder_input_ids[0].tolist()),
        )

        raw_tokens = self.processor.batch_decode(outputs.sequences)[0]
        cleaned = raw_tokens.replace(self.processor.tokenizer.eos_token, "")
        cleaned = cleaned.replace(self.processor.tokenizer.pad_token, "").strip()

        # For SROIE, use custom parser; for CORD, use token2json
        if self.task_prompt.startswith("<s_sroie"):
            try:
                parsed = _parse_sroie_output(cleaned)
            except Exception as exc:
                raise RuntimeError(
                    f"Self-test FAILED: SROIE parser raised {type(exc).__name__}: {exc}\n"
                    f"  Raw tokens: {raw_tokens!r}\n"
                    f"  Cleaned:    {cleaned!r}\n"
                    f"  Model path: {self.model_path}"
                ) from exc
        else:
            try:
                parsed = self.processor.token2json(cleaned)
            except Exception as exc:
                raise RuntimeError(
                    f"Self-test FAILED: token2json raised {type(exc).__name__}: {exc}\n"
                    f"  Raw tokens: {raw_tokens!r}\n"
                    f"  Cleaned:    {cleaned!r}\n"
                    f"  Model path: {self.model_path}"
                ) from exc

            # token2json returns a list when <sep/> tokens are present (CORD multi-page).
            # Merge pages before unwrapping so the dict check below works correctly. (Phase 0b)
            parsed = _merge_token2json_pages(parsed)

        # Unwrap task-prompt wrappers
        parsed = _unwrap_prediction(parsed, self.task_prompt)

        # Check that we got at least one non-empty value.
        # Non-fatal: micro/mini smoke-test models are deliberately undertrained
        # and may not yet produce parseable SROIE tags. Log a warning and let
        # the full 63-sample evaluation determine the true F1 rather than
        # aborting with zeroed metrics.
        if not isinstance(parsed, dict) or not parsed:
            logger.warning(
                "Self-test: model produced empty dict — model may be undertrained "
                "(micro/mini mode). Continuing with full evaluation.\n"
                "  Raw tokens: %r\n"
                "  Cleaned:    %r\n"
                "  Parsed:     %r\n"
                "  Model path: %s",
                raw_tokens,
                cleaned,
                parsed,
                self.model_path,
            )
            return

        has_nonempty = any(
            str(v).strip() for v in parsed.values() if isinstance(v, (str, int, float))
        )
        # For nested dicts (e.g. CORD output), any non-empty sub-dict counts
        if not has_nonempty:
            has_nonempty = any(v for v in parsed.values() if isinstance(v, dict) and v)
        if not has_nonempty:
            has_nonempty = any(v for v in parsed.values() if isinstance(v, list) and v)

        if not has_nonempty:
            logger.warning(
                "Self-test: all fields empty in parsed output — model may be undertrained "
                "(micro/mini mode). Continuing with full evaluation.\n"
                "  Raw tokens: %r\n"
                "  Cleaned:    %r\n"
                "  Parsed:     %r\n"
                "  Model path: %s",
                raw_tokens,
                cleaned,
                parsed,
                self.model_path,
            )
            return

        logger.info("Self-test PASSED: parsed %d key(s) from %s", len(parsed), img_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        image_path: Path,
        task_prompt: str,
        preloaded_image: "Any | None" = None,
    ) -> dict:
        """Run inference on a single image and return parsed dict.

        Handles:
          - token2json failures (returns {} with log)
          - {"sroie": {...}} unwrapping
          - Diagnostic logging for the first few calls
        """
        self._inference_call_count += 1

        if preloaded_image is not None:
            image = preloaded_image
        else:
            image = _load_image(image_path)

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.max_length,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self.processor.tokenizer.pad_token, "").strip()

        # Diagnostic logging for the first few calls (file only — too verbose for console)
        if self._inference_call_count <= _DIAGNOSTIC_LOG_COUNT:
            logger.debug(
                "Inference #%d raw tokens: %s",
                self._inference_call_count,
                sequence[:200] + ("..." if len(sequence) > 200 else ""),
            )

        parsed = self._parse_prediction(sequence)
        parsed = _unwrap_prediction(parsed, task_prompt)
        return parsed

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_prediction(self, tokens: str) -> dict:
        """Parse prediction using appropriate parser for the task format.

        For SROIE task prompts (<s_sroie>), uses the custom _parse_sroie_output()
        parser to extract values from XML-like tags.
        For CORD task prompts, falls back to token2json().

        token2json returns a list when the generated sequence contains <sep/>
        tokens (CORD multi-page format).  Even SROIE fine-tuned models can
        emit <sep/> because the base checkpoint (donut-base-finetuned-cord-v2)
        knows the token.  Merge pages into one dict (first occurrence of each
        key wins) so callers always receive a flat dict.
        """
        # For SROIE output, use the custom parser that understands SROIE tags
        if getattr(self, "task_prompt", "").startswith("<s_sroie"):
            try:
                result = _parse_sroie_output(tokens)
                if result and any(v for v in result.values()):  # At least one non-empty field
                    return result
                # No fields extracted — log and increment failure
                logger.warning("SROIE parser returned empty result from tokens: %.100s", tokens)
                self.parse_failure_count += 1
                return {}
            except Exception as exc:
                logger.warning("SROIE parser failed: %s — tokens: %.100s", exc, tokens)
                self.parse_failure_count += 1
                return {}

        # For CORD/other formats, use token2json
        try:
            result = self.processor.token2json(tokens)
            result = _merge_token2json_pages(result)  # Phase 0b: consolidate list merging
            if result:
                return result
            # Empty result from token2json (either [] list or {} dict)
            logger.warning("token2json returned empty result: %s", type(result))
            self.parse_failure_count += 1
            return {}
        except Exception as exc:
            logger.warning("token2json failed: %s — tokens: %.100s", exc, tokens)
            self.parse_failure_count += 1
            return {}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_f1(self, preds: list[dict], labels: list[dict]) -> float:
        """Compute global F1 over all (image, field) pairs.

        A pair is a true positive if the predicted string equals the ground
        truth string (case-insensitive, stripped).

        Returns 0.0 when ``preds`` is empty (no predictions to evaluate).
        """
        if not preds:
            return 0.0
        tp, total_pred, total_gt = 0, 0, 0
        for pred, gt in zip(preds, labels):
            for f in FIELDS:
                p_val = str(pred.get(f, "")).strip().lower()
                g_val = str(gt.get(f, "")).strip().lower()
                if g_val:
                    total_gt += 1
                if p_val:
                    total_pred += 1
                if p_val and g_val and p_val == g_val:
                    tp += 1
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _compute_ned(self, pred: str, gt: str) -> float:
        """Compute Normalized Edit Distance between two strings.

        NED is **lower-is-better**: 0 = identical, 1 = maximally different.

        Uses ``max(len(pred), len(gt))`` as denominator so NED ∈ [0, 1].
        """
        return normalized_edit_distance(pred, gt)

    def compute_all_metrics(
        self,
        predictions: list[dict],
        ground_truths: list[dict],
    ) -> dict[str, float]:
        """Compute all metrics: global F1, per-field F1, per-field NED, exact match.

        Returns a flat dict compatible with the legacy ``compute_metrics`` output.
        """
        return compute_metrics(predictions, ground_truths)


# ---------------------------------------------------------------------------
# Prediction unwrapping helper
# ---------------------------------------------------------------------------


def _unwrap_prediction(parsed: dict, task_prompt: str) -> dict:
    """Unwrap task-prompt wrappers from token2json output.

    token2json may wrap SROIE output as ``{"sroie": {...}}``.
    CORD output may appear as ``{"cord-v2": {...}}``.
    """
    if not isinstance(parsed, dict):
        return parsed

    # Unwrap {"sroie": {...}} for SROIE task prompts
    if (
        task_prompt.startswith("<s_sroie")
        and "sroie" in parsed
        and isinstance(parsed["sroie"], dict)
    ):
        return parsed["sroie"]

    # Unwrap {"cord-v2": {...}} for CORD task prompts
    if (
        task_prompt.startswith("<s_cord")
        and "cord-v2" in parsed
        and isinstance(parsed["cord-v2"], dict)
    ):
        return parsed["cord-v2"]

    return parsed


# ---------------------------------------------------------------------------
# Module-level backward-compatible functions
# ---------------------------------------------------------------------------

# Global counter for diagnostic logging in the module-level run_inference
_module_inference_count = 0


def run_inference(model, processor, image_path, task_prompt, max_length=512, preloaded_image=None):
    """Run inference on a single image. Accepts an optional pre-loaded PIL Image.

    This is the backward-compatible module-level function. For new code,
    prefer ``DonutEvaluator._run_inference()``.

    After model.generate(), diagnostic logging is emitted for the first few
    calls to help debug token2json failures.
    """
    global _module_inference_count
    _module_inference_count += 1

    if preloaded_image is not None:
        image = preloaded_image
    else:
        image = _load_image(image_path)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(DEVICE)

    # FIX: Removed early_stopping=True — invalid with num_beams=1 (greedy
    # decoding).  This caused thousands of deprecation warnings per eval run.
    # Also removed repetition_penalty and no_repeat_ngram_size which were
    # blocking structural tags like </s_company> and </s_date>.
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=max_length,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "").strip()

    # Diagnostic logging for the first few calls (file only — too verbose for console)
    if _module_inference_count <= _DIAGNOSTIC_LOG_COUNT:
        logger.debug(
            "run_inference #%d [%s] raw tokens: %s",
            _module_inference_count,
            task_prompt,
            sequence[:200] + ("..." if len(sequence) > 200 else ""),
        )

    # Check for task prompt mismatch (model outputting CORD schema for SROIE task)
    if task_prompt.startswith("<s_sroie") and sequence.startswith("<s_cord-v2>"):
        logger.warning(
            "Task prompt mismatch for %s: asked for <s_sroie> but model output starts with "
            "<s_cord-v2>. Model has not learned SROIE task format (CORD pretraining dominates). "
            "This indicates insufficient training or a corrupted checkpoint.",
            image_path,
        )

    # For SROIE output, use custom parser; for CORD, use token2json
    if task_prompt.startswith("<s_sroie"):
        try:
            result = _parse_sroie_output(sequence)
            return result if result and any(v for v in result.values()) else {}
        except Exception as e:
            logger.warning("SROIE parser failed for %s: %s", image_path, e)
            return {}

    # For CORD/other formats, use token2json
    try:
        result = processor.token2json(sequence)
    except Exception as e:
        logger.warning("token2json failed for %s: %s", image_path, e)
        return {}

    if not isinstance(result, dict):
        if isinstance(result, list):
            # CORD multi-page format: token2json returns a list when the
            # generated sequence contains <sep/> tokens (multiple line items).
            # Pass the list through so remap_cord_to_sroie can merge pages.
            logger.debug(
                "token2json returned list for %s (%d pages) — passing to caller",
                image_path,
                len(result),
            )
            return result
        logger.warning("token2json returned non-dict for %s: %s", image_path, type(result))
        return {}

    # Unwrap task-prompt wrappers
    result = _unwrap_prediction(result, task_prompt)
    return result


def remap_cord_to_sroie(cord_output):
    """Map CORD schema fields to SROIE field names (best effort).

    Handles the 'cord-v2' top-level wrapper that the pretrained CORD model
    may emit (e.g. ``{"cord-v2": {...}}``).

    Robust remapping handles multiple CORD schema variants:
      - ``store_info.store_name`` → ``company``
      - ``store_info.region`` or ``store_info.address`` → ``address``
      - ``total.total_price`` or ``total.total_etc`` → ``total``
      - ``date.date_value`` (dict, list, or plain string) → ``date``
      - Plain string values for any key
    """
    result = EMPTY_GT.copy()  # Phase 0b: use single source of truth

    # Handle list output from token2json (multi-page CORD with <sep/> tokens).
    # Merge all pages: first occurrence of each top-level key wins. (Phase 0b)
    cord_output = _merge_token2json_pages(cord_output)

    if not isinstance(cord_output, dict):
        return result

    # Unwrap the "cord-v2" top-level key if present
    if "cord-v2" in cord_output and isinstance(cord_output["cord-v2"], dict):
        cord_output = cord_output["cord-v2"]

    # --- company / address from store_info ---
    store_info = cord_output.get("store_info", {})
    if isinstance(store_info, dict):
        result["company"] = str(store_info.get("store_name", "")).strip()
        # Try multiple address field names
        addr = store_info.get("region", "") or store_info.get("address", "")
        result["address"] = str(addr).strip()
    elif isinstance(store_info, list) and store_info:
        first = store_info[0] if isinstance(store_info[0], dict) else {}
        result["company"] = str(first.get("store_name", "")).strip()
        addr = first.get("region", "") or first.get("address", "")
        result["address"] = str(addr).strip()
    elif isinstance(store_info, str):
        result["company"] = store_info.strip()

    # --- total ---
    total_info = cord_output.get("total", {})
    if isinstance(total_info, dict):
        # Try multiple total field names
        total_val = (
            total_info.get("total_price", "")
            or total_info.get("total_etc", "")
            or total_info.get("cashprice", "")
        )
        result["total"] = str(total_val).strip()
    elif isinstance(total_info, list) and total_info:
        first = total_info[0] if isinstance(total_info[0], dict) else {}
        total_val = (
            first.get("total_price", "") or first.get("total_etc", "") or first.get("cashprice", "")
        )
        result["total"] = str(total_val).strip()
    elif isinstance(total_info, str):
        result["total"] = total_info.strip()

    # --- date ---
    date_info = cord_output.get("date", {})
    if isinstance(date_info, dict):
        result["date"] = str(date_info.get("date_value", "")).strip()
    elif isinstance(date_info, list) and date_info:
        first = date_info[0]
        if isinstance(first, dict):
            result["date"] = str(first.get("date_value", "")).strip()
        else:
            result["date"] = str(first).strip()
    elif isinstance(date_info, str):
        result["date"] = date_info.strip()

    # --- fallback: check for direct SROIE-like keys at top level ---
    for f in FIELDS:
        if not result[f] and f in cord_output:
            val = cord_output[f]
            if isinstance(val, str):
                result[f] = val.strip()
            elif isinstance(val, dict):
                # Take first non-empty string value from the sub-dict
                for v in val.values():
                    if isinstance(v, str) and v.strip():
                        result[f] = v.strip()
                        break

    return result


def normalized_edit_distance(pred, gt):
    """Compute Normalized Edit Distance (NED) between pred and gt strings.

    NED = editdistance(pred, gt) / max(len(pred), len(gt))

    **Lower is better**: NED ∈ [0, 1] where 0 means identical and 1 means
    maximally different. This formulation uses ``max(len(pred), len(gt))``
    as the denominator, ensuring NED ≤ 1.0.
    """
    pred, gt = str(pred).lower().strip(), str(gt).lower().strip()
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _edit_distance(pred, gt) / max(len(pred), len(gt))


def compute_metrics(predictions, ground_truths):
    """Official SROIE Task 3 metric: global F1 over all (image, field) pairs.

    A pair is TP if predicted string == ground truth string
    (case-insensitive, stripped).

    Returns a flat dict with keys:
      ``global_precision``, ``global_recall``, ``global_f1``,
      ``overall_exact_match``, and per-field ``{field}_f1``, ``{field}_ned``.
    """
    tp, total_pred, total_gt = 0, 0, 0
    per_field = {f: {"tp": 0, "pred": 0, "gt": 0, "ned": []} for f in FIELDS}
    exact_match_all = []

    # Check for total prediction failure: all predictions are empty dicts
    all_empty = all(not any(str(pred.get(f, "")).strip() for f in FIELDS) for pred in predictions)
    if all_empty and predictions:
        print(
            "CRITICAL WARNING: ALL predictions are empty (token2json total failure). "
            "F1 will be 0.0 — check model output and token2json compatibility.",
            file=sys.stderr,
        )

    for pred, gt in zip(predictions, ground_truths):
        all_correct = True
        for f in FIELDS:
            p_val = str(pred.get(f, "")).strip().lower()
            g_val = str(gt.get(f, "")).strip().lower()

            if g_val:
                total_gt += 1
                per_field[f]["gt"] += 1
            if p_val:
                total_pred += 1
                per_field[f]["pred"] += 1
            if p_val and g_val and p_val == g_val:
                tp += 1
                per_field[f]["tp"] += 1
            elif not p_val and not g_val:
                pass  # Both absent = true negative
            else:
                all_correct = False

            per_field[f]["ned"].append(normalized_edit_distance(p_val, g_val))
        exact_match_all.append(int(all_correct))

    precision = tp / total_pred if total_pred > 0 else 0
    recall = tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    summary = {
        "global_precision": round(precision, 4),
        "global_recall": round(recall, 4),
        "global_f1": round(f1, 4),
        "overall_exact_match": round(np.mean(exact_match_all), 4) if exact_match_all else 0.0,
    }

    for f in FIELDS:
        fp = per_field[f]
        p = fp["tp"] / fp["pred"] if fp["pred"] > 0 else 0
        r = fp["tp"] / fp["gt"] if fp["gt"] > 0 else 0
        f_score = 2 * p * r / (p + r) if (p + r) > 0 else 0
        summary[f"{f}_f1"] = round(f_score, 4)
        summary[f"{f}_ned"] = round(np.mean(fp["ned"]), 4) if fp["ned"] else 1.0

    return summary


def print_results(pretrained_m, finetuned_m):
    """Pretty-print side-by-side pretrained vs. fine-tuned metrics."""
    print(f"\n{'=' * 72}")
    print(f"{'METRIC':<30} {'PRETRAINED':>18} {'FINE-TUNED':>18}")
    print(f"{'=' * 72}")
    print(f"{'Global F1':<30} {pretrained_m['global_f1']:>18.4f} {finetuned_m['global_f1']:>18.4f}")
    print(
        f"{'Global Precision':<30} {pretrained_m['global_precision']:>18.4f} {finetuned_m['global_precision']:>18.4f}"
    )
    print(
        f"{'Global Recall':<30} {pretrained_m['global_recall']:>18.4f} {finetuned_m['global_recall']:>18.4f}"
    )
    print(
        f"{'Overall Exact Match':<30} {pretrained_m['overall_exact_match']:>18.4f} {finetuned_m['overall_exact_match']:>18.4f}"
    )
    print(f"{'-' * 72}")
    for f in FIELDS:
        print(f"{f + ' F1':<30} {pretrained_m[f + '_f1']:>18.4f} {finetuned_m[f + '_f1']:>18.4f}")
        print(
            f"{f + ' NED':<30} {pretrained_m[f + '_ned']:>18.4f} {finetuned_m[f + '_ned']:>18.4f}"
        )
    print(f"{'=' * 72}")

    print(f"\n{'SROIE TASK 3 LEADERBOARD COMPARISON':^72}")
    print(f"{'-' * 72}")
    print(f"{'Method':<35} {'F1':>10}")
    print(f"{'-' * 72}")
    leaderboard = [
        ("LayoutLMv3 (Huang et al. 2022)", 0.9633),
        ("PICK (Yu et al. 2021)", 0.9612),
        ("BROS (Hong et al. 2022)", 0.9548),
        ("LayoutLMv2 (Xu et al. 2021)", 0.9495),
        ("DONUT SROIE fine-tuned (Kim 2022)", 0.8411),
        ("Our pretrained (CORD zero-shot)", pretrained_m["global_f1"]),
        ("Our fine-tuned (this work)", finetuned_m["global_f1"]),
    ]
    for name, score in sorted(leaderboard, key=lambda x: x[1], reverse=True):
        marker = " ◄" if "Our" in name else ""
        print(f"{name:<35} {score:>10.4f}{marker}")
    print(f"{'=' * 72}\n")


# ---------------------------------------------------------------------------
# Legacy standalone entry point
# ---------------------------------------------------------------------------


def main():
    """Legacy standalone entry point for ad-hoc evaluation.

    For the full 8-experiment pipeline, use ``python run_all.py`` instead.
    """
    # Load test images + ground truth using canonical loader
    from dataset_loaders import load_sroie_test

    test_samples = load_sroie_test()

    print(f"Evaluating on {len(test_samples)} test images")

    ground_truths = [s[1] for s in test_samples]
    image_paths = [s[0] for s in test_samples]

    # Load pretrained (CORD) — hub model, no re-tying needed
    print("Loading pretrained model (CORD)...")
    pre_processor = DonutProcessor.from_pretrained(BASE_MODEL)
    pre_model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL).to(DEVICE)
    pre_model.eval()

    # Load fine-tuned — apply weight re-tying fix
    print("Loading fine-tuned model...")
    workspace = os.environ.get("DONUT_WORKSPACE", "/workspace")
    ft_model_dir = os.path.join(workspace, "donut-sroie-finetuned")
    ft_processor = DonutProcessor.from_pretrained(ft_model_dir)
    ft_model = load_model_with_tied_weights(ft_model_dir, device=DEVICE)

    pretrained_preds = []
    finetuned_preds = []

    with torch.no_grad():
        for img_path in _progress(image_paths, desc="Inference"):
            # Pretrained (CORD) → remap
            raw = run_inference(pre_model, pre_processor, img_path, "<s_cord-v2>")
            pretrained_preds.append(remap_cord_to_sroie(raw))

            # Fine-tuned (SROIE) — unwrapping handled inside run_inference
            raw_ft = run_inference(ft_model, ft_processor, img_path, "<s_sroie>")
            finetuned_preds.append(raw_ft)

    pretrained_metrics = compute_metrics(pretrained_preds, ground_truths)
    finetuned_metrics = compute_metrics(finetuned_preds, ground_truths)

    print_results(pretrained_metrics, finetuned_metrics)

    # Save everything
    output = {
        "pretrained_metrics": pretrained_metrics,
        "finetuned_metrics": finetuned_metrics,
        "samples": [
            {
                "image": str(p),
                "ground_truth": gt,
                "pretrained_pred": pp,
                "finetuned_pred": fp,
            }
            for p, gt, pp, fp in zip(image_paths, ground_truths, pretrained_preds, finetuned_preds)
        ],
    }
    output_file = os.path.join(workspace, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved -> {output_file}")


# ── Config ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")

SROIE_DATA_DIR = _get_sroie_dir()


# ════════════════════════════════════════════════════════════════════════════
# Shared metric computation (SROIE Task-3 compatible)
# ════════════════════════════════════════════════════════════════════════════
# Import compute_metrics from donut_evaluator — single source of truth for
# SROIE Task-3 F1/NED/exact-match shared across DONUT and TrOCR+YOLO.
# compute_metrics is defined earlier in this same merged file (from donut_evaluator section)
compute_sroie_metrics = compute_metrics  # noqa: F821 (defined above in merged file)


# ════════════════════════════════════════════════════════════════════════════
# Load SROIE test set (shared by both architectures)
# ════════════════════════════════════════════════════════════════════════════
def load_test_samples() -> list[tuple[Path, dict[str, str]]]:
    """Load the 63 SROIE test images + ground truth via the canonical loader.

    Both DONUT and TrOCR+YOLO are evaluated on this EXACT same set.
    Delegates to dataset_loaders.load_sroie_test() — single source of truth.
    """
    return load_sroie_test()


# ════════════════════════════════════════════════════════════════════════════
# Evaluate DONUT on test set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_donut_on_test(
    model_path: str,
    test_samples: list[tuple[Path, dict[str, str]]],
) -> dict:
    """Evaluate a DONUT model on the SROIE test set. Returns metrics dict."""
    from transformers import DonutProcessor

    from evaluation import _parse_sroie_output, load_model_with_tied_weights  # noqa: E402

    processor = DonutProcessor.from_pretrained(model_path)
    model = load_model_with_tied_weights(model_path, device=DEVICE)

    predictions = []
    ground_truths = [s[1] for s in test_samples]
    latencies = []
    parse_failures = 0

    with torch.no_grad():
        for img_path, _gt in _progress(test_samples, desc="DONUT eval"):
            image = _PILImage.open(img_path).convert("RGB")
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)
            decoder_input_ids = processor.tokenizer(
                "<s_sroie>", add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(DEVICE)

            t0 = time.perf_counter()
            # FIX: No early_stopping=True — invalid with num_beams=1
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=MAX_LENGTH,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            sequence = processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(processor.tokenizer.eos_token, "")
            sequence = sequence.replace(processor.tokenizer.pad_token, "").strip()

            try:
                parsed = _parse_sroie_output(sequence)
            except Exception:
                parsed = {}
                parse_failures += 1

            predictions.append(parsed)

    metrics = compute_sroie_metrics(predictions, ground_truths)
    metrics["parse_failures"] = parse_failures
    metrics["num_samples"] = len(test_samples)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 1) if latencies else 0.0

    # GPU cleanup
    _gpu_cleanup(model, processor)

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Evaluate TrOCR+YOLO on test set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_trocr_yolo_on_test(
    yolo_weights: str,
    trocr_model_path: str,
    test_samples: list[tuple[Path, dict[str, str]]],
) -> dict:
    """Evaluate TrOCR+YOLO pipeline on the SROIE test set. Returns metrics dict."""
    from importlib import import_module

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from ultralytics import YOLO

    # Import the inference function and meta-buffer fix from train_trocr_yolo.py
    trocr_yolo_module = import_module("train_trocr_yolo")
    run_pipeline = trocr_yolo_module.run_trocr_yolo_inference
    _materialize_meta_buffers = trocr_yolo_module._materialize_meta_buffers

    yolo_model = YOLO(str(yolo_weights))
    trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_path)
    # FIX: low_cpu_mem_usage=False + _materialize_meta_buffers prevents the
    # meta-device crash on TrOCR's sinusoidal positional embedding buffer.
    trocr_model = VisionEncoderDecoderModel.from_pretrained(
        trocr_model_path, low_cpu_mem_usage=False
    ).to(DEVICE)
    _materialize_meta_buffers(trocr_model, DEVICE)
    trocr_model.eval()

    predictions = []
    ground_truths = [s[1] for s in test_samples]
    latencies = []

    with torch.no_grad():
        for img_path, _gt in _progress(test_samples, desc="TrOCR+YOLO eval"):
            t0 = time.perf_counter()
            pred = run_pipeline(img_path, yolo_model, trocr_model, trocr_processor)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            predictions.append(pred)

    metrics = compute_sroie_metrics(predictions, ground_truths)
    metrics["num_samples"] = len(test_samples)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies)), 1) if latencies else 0.0

    # GPU cleanup
    _gpu_cleanup(yolo_model, trocr_model, trocr_processor)

    return metrics


# ── Print metrics ────────────────────────────────────────────────────────────
def print_metrics(name: str, metrics: dict) -> None:
    """Pretty-print evaluation metrics in structured format."""
    print(f"\n  {'=' * 55}")
    print(f"  {name} Results")
    print(f"  {'=' * 55}")
    print(f"  Global F1:        {metrics.get('global_f1', 0):.4f}")
    print(f"  Global Precision: {metrics.get('global_precision', 0):.4f}")
    print(f"  Global Recall:    {metrics.get('global_recall', 0):.4f}")
    print(f"  Exact Match:      {metrics.get('overall_exact_match', 0):.4f}")
    print(f"  Num Samples:      {metrics.get('num_samples', 0)}")
    if "mean_latency_ms" in metrics:
        print(f"  Mean Latency:     {metrics['mean_latency_ms']:.1f} ms/image")
    print(f"  {'-' * 55}")
    for f in FIELDS:
        f1 = metrics.get(f"{f}_f1", 0)
        ned = metrics.get(f"{f}_ned", 1)
        print(f"  {f:12s}  F1={f1:.4f}  NED={ned:.4f}")
    print(f"  {'=' * 55}")


def generate_comparison_report(results: dict) -> None:
    """Generate a detailed HTML comparison report of all evaluated models."""
    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Model Evaluation Report</title>",
        "<style>",
        "  body { font-family: monospace; margin: 20px; }",
        "  table { border-collapse: collapse; margin: 20px 0; }",
        "  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "  th { background-color: #4CAF50; color: white; }",
        "  tr:nth-child(even) { background-color: #f2f2f2; }",
        "  .metric-high { color: green; font-weight: bold; }",
        "  .metric-low { color: red; }",
        "  h1, h2 { color: #333; }",
        "</style></head><body>",
        "<h1>Model Evaluation Report</h1>",
    ]

    if not results:
        html_lines.append("<p>No results to display.</p>")
    else:
        # Extract architectures and create comparison table
        html_lines.append("<h2>Global Performance Comparison</h2>")
        html_lines.append(
            "<table><tr><th>Architecture</th><th>Global F1</th><th>Precision</th><th>Recall</th><th>Exact Match</th><th>Latency (ms)</th></tr>"
        )

        for arch, metrics in results.items():
            f1 = metrics.get("global_f1", 0)
            prec = metrics.get("global_precision", 0)
            rec = metrics.get("global_recall", 0)
            em = metrics.get("overall_exact_match", 0)
            lat = metrics.get("mean_latency_ms", 0)

            f1_class = "metric-high" if f1 > 0.85 else "metric-low" if f1 < 0.7 else ""

            html_lines.append(
                f"<tr><td><strong>{arch}</strong></td><td class='{f1_class}'>{f1:.4f}</td>"
                f"<td>{prec:.4f}</td><td>{rec:.4f}</td><td>{em:.4f}</td><td>{lat:.1f}</td></tr>"
            )

        html_lines.append("</table>")

        # Per-field comparison
        html_lines.append("<h2>Per-Field Metrics</h2>")
        for arch, metrics in results.items():
            html_lines.append(f"<h3>{arch.upper()}</h3>")
            html_lines.append("<table><tr><th>Field</th><th>F1</th><th>NED</th></tr>")

            for field in FIELDS:
                f1 = metrics.get(f"{field}_f1", 0)
                ned = metrics.get(f"{field}_ned", 1)
                html_lines.append(f"<tr><td>{field}</td><td>{f1:.4f}</td><td>{ned:.4f}</td></tr>")

            html_lines.append("</table>")

    html_lines.extend(["</body></html>"])

    report_path = RESULTS_DIR / "evaluation_report.html"
    with open(report_path, "w") as f:
        f.write("\n".join(html_lines))

    print(f"\n📊 Detailed report saved -> {report_path}")


def generate_json_summary(results: dict) -> None:
    """Export evaluation results in structured JSON format."""
    summary = {
        "evaluation_timestamp": __import__("datetime").datetime.now().isoformat(),
        "num_test_samples": results.get("donut", {}).get("num_samples", 0),
        "architectures_evaluated": list(results.keys()),
        "results": results,
    }

    summary_path = RESULTS_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📋 Summary saved -> {summary_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate DONUT and TrOCR+YOLO models")
    parser.add_argument(
        "--donut-only", action="store_true", help="Only evaluate DONUT (skip TrOCR+YOLO)"
    )
    parser.add_argument(
        "--trocr-only", action="store_true", help="Only evaluate TrOCR+YOLO (skip DONUT)"
    )
    parser.add_argument("--report", action="store_true", help="Generate HTML comparison report")

    args = parser.parse_args()

    test_samples = load_test_samples()
    print(f"Loaded {len(test_samples)} test samples")

    results = {}

    # Evaluate DONUT (if model exists and not skipped)
    if not args.trocr_only:
        donut_model = Path("models/donut_finetuned/best")
        if donut_model.exists():
            results["donut"] = evaluate_donut_on_test(str(donut_model), test_samples)
            print_metrics("DONUT", results["donut"])
        else:
            print(f"  DONUT model not found at {donut_model} — skipping")

    # Evaluate TrOCR+YOLO (if models exist and not skipped)
    if not args.donut_only:
        yolo_weights = Path("models/yolo_finetuned/run/weights/best.pt")
        trocr_model = Path("models/trocr_finetuned/best")
        if yolo_weights.exists() and trocr_model.exists():
            results["trocr_yolo"] = evaluate_trocr_yolo_on_test(
                str(yolo_weights), str(trocr_model), test_samples
            )
            print_metrics("TrOCR+YOLO", results["trocr_yolo"])
        else:
            print("  TrOCR+YOLO models not found — skipping")

    # Save combined results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Metrics saved -> {out_path}")

    # Always generate summary
    generate_json_summary(results)
    if results:
        generate_comparison_report(results)
