"""
donut_evaluator.py — DONUT SROIE evaluation with OOP interface and legacy compat.

Renamed from evaluate.py to avoid shadowing the HuggingFace ``evaluate``
library (pip install evaluate).  evaluate.py is kept as a backward-compat
shim that re-exports everything from this module.

WARNING: For the full 8-experiment pipeline, use: python run_all.py

Critical bug fixes in this version:
  - Re-ties decoder.lm_head.weight after from_pretrained() to fix missing
    keys warning that causes garbage generation (F1=0).
  - Detects missing lm_head via output_loading_info=True and recovers by
    copying embed_tokens.weight (avoids the F1~0.42 collapse).
  - Unwraps {"sroie": {...}} wrapper from token2json output.
  - Adds self-test before full evaluation to catch broken models early.
  - Parse failure threshold: raises if >50% of samples fail to parse.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import editdistance
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DonutProcessor, VisionEncoderDecoderModel

# FIX: Import shared constants from single source of truth (constants.py)
# instead of duplicating FIELDS/IMAGE_EXTS independently in this file.
from constants import BASE_MODEL, DEVICE, EMPTY_GT, FIELDS, MAX_LENGTH

__all__ = ["DonutEvaluator", "EvaluationResult", "compute_metrics", "normalized_edit_distance"]

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

    for field in FIELDS:
        open_tag = f"<s_{field}>"
        close_tag = f"</s_{field}>"

        start_idx = tokens.find(open_tag)
        if start_idx != -1:
            start_idx += len(open_tag)
            end_idx = tokens.find(close_tag, start_idx)
            if end_idx != -1:
                result[field] = tokens[start_idx:end_idx].strip()

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


def load_model_with_tied_weights(model_path: str, device: str = DEVICE):
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
        self.model = load_model_with_tied_weights(str(self.model_path), device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self) -> EvaluationResult:
        """Run full evaluation: self-test first, then inference on all samples.

        Raises:
            RuntimeError: If self-test fails (model produces no parseable output).
            RuntimeError: If >50% of samples fail to parse (model is broken).
        """
        self._self_test()

        image_paths = [s[0] for s in self.test_dataset]
        ground_truths = [s[1] for s in self.test_dataset]

        predictions = []
        self.parse_failure_count = 0
        self._inference_call_count = 0

        with torch.no_grad():
            for img_path, _gt in tqdm(self.test_dataset, desc="Evaluating"):
                pred = self._run_inference(img_path, self.task_prompt)
                predictions.append(pred)

        # Parse failure threshold check
        n = len(self.test_dataset)
        if n > 0 and self.parse_failure_count > n * 0.5:
            raise RuntimeError(
                f"Parse failure threshold exceeded: {self.parse_failure_count}/{n} "
                f"({self.parse_failure_count / n:.1%}) samples failed to parse. "
                f"The model is likely broken — check token2json compatibility."
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
        image = Image.open(img_path).convert("RGB")
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

        # Check that we got at least one non-empty value
        if not isinstance(parsed, dict) or not parsed:
            raise RuntimeError(
                f"Self-test FAILED: model produced empty dict\n"
                f"  Raw tokens: {raw_tokens!r}\n"
                f"  Cleaned:    {cleaned!r}\n"
                f"  Parsed:     {parsed!r}\n"
                f"  Model path: {self.model_path}"
            )

        has_nonempty = any(
            str(v).strip() for v in parsed.values() if isinstance(v, (str, int, float))
        )
        # For nested dicts (e.g. CORD output), any non-empty sub-dict counts
        if not has_nonempty:
            has_nonempty = any(v for v in parsed.values() if isinstance(v, dict) and v)
        if not has_nonempty:
            has_nonempty = any(v for v in parsed.values() if isinstance(v, list) and v)

        if not has_nonempty:
            raise RuntimeError(
                f"Self-test FAILED: all fields empty in parsed output\n"
                f"  Raw tokens: {raw_tokens!r}\n"
                f"  Cleaned:    {cleaned!r}\n"
                f"  Parsed:     {parsed!r}\n"
                f"  Model path: {self.model_path}"
            )

        logger.info("Self-test PASSED: parsed %d key(s) from %s", len(parsed), img_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        image_path: Path,
        task_prompt: str,
        preloaded_image: Image.Image | None = None,
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
            image = Image.open(image_path).convert("RGB")

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
        if self.task_prompt.startswith("<s_sroie"):
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
        image = Image.open(image_path).convert("RGB")
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
    return editdistance.eval(pred, gt) / max(len(pred), len(gt))


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
        for img_path in tqdm(image_paths, desc="Inference"):
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


if __name__ == "__main__":
    main()
