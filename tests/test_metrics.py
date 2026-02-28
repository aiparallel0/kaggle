"""Tests for evaluation metrics — compute_metrics, normalized_edit_distance, _unwrap_prediction.

These tests exercise the core metric logic without requiring GPU or model weights.
Requires: torch, transformers (evaluate.py imports them at module level).
"""

import sys
import unittest.mock as mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch", reason="torch required by donut_evaluator.py")
pytest.importorskip("transformers", reason="transformers required by donut_evaluator.py")

from donut_evaluator import (  # noqa: E402
    _unwrap_prediction,
    compute_metrics,
    normalized_edit_distance,
)

# ---------------------------------------------------------------------------
# normalized_edit_distance
# ---------------------------------------------------------------------------

class TestNED:
    def test_identical_strings(self):
        assert normalized_edit_distance("hello", "hello") == 0.0

    def test_completely_different(self):
        ned = normalized_edit_distance("abc", "xyz")
        assert ned == 1.0  # 3 edits / max(3, 3)

    def test_empty_both(self):
        assert normalized_edit_distance("", "") == 0.0

    def test_empty_gt(self):
        assert normalized_edit_distance("something", "") == 1.0

    def test_empty_pred(self):
        assert normalized_edit_distance("", "something") == 1.0

    def test_case_insensitive(self):
        assert normalized_edit_distance("HELLO", "hello") == 0.0

    def test_whitespace_stripping(self):
        assert normalized_edit_distance("  hello  ", "hello") == 0.0

    def test_partial_match(self):
        ned = normalized_edit_distance("hello", "hallo")
        assert 0 < ned < 1.0  # 1 edit / 5 chars

    def test_ned_bounded(self):
        ned = normalized_edit_distance("abcdef", "xyz")
        assert 0.0 <= ned <= 1.0


# ---------------------------------------------------------------------------
# _parse_prediction list-merge behaviour (exercised via DonutEvaluator)
# ---------------------------------------------------------------------------

class TestParsePredictionListMerge:
    """Verify token2json list output is merged rather than discarded.

    Root cause of F1=0.0078: _parse_prediction() returned {} when token2json()
    returned a list (CORD <sep/> multi-page format), collapsing all predictions
    to empty dicts.  Fix: merge list pages into a single flat dict.
    """

    def _make_evaluator_stub(self):
        """Return a minimal DonutEvaluator-like object with just _parse_prediction."""
        pytest.importorskip("torch")
        from donut_evaluator import DonutEvaluator

        # Build the smallest possible evaluator without hitting from_pretrained
        evaluator = object.__new__(DonutEvaluator)
        evaluator.parse_failure_count = 0
        evaluator._inference_call_count = 0

        class _FakeProcessor:
            def token2json(self, tokens):
                # Simulate multi-page list output
                return [
                    {"company": "MYDIN MALL", "date": "25/12/2023"},
                    {"address": "NO 1 JALAN", "total": "47.80"},
                ]

        evaluator.processor = _FakeProcessor()
        return evaluator

    def test_list_pages_merged_to_dict(self):
        """Multi-page list from token2json is merged into a single flat dict."""
        evaluator = self._make_evaluator_stub()
        result = evaluator._parse_prediction("<irrelevant tokens>")
        assert isinstance(result, dict), "Expected dict, got list (merge failed)"
        assert result["company"] == "MYDIN MALL"
        assert result["date"] == "25/12/2023"
        assert result["address"] == "NO 1 JALAN"
        assert result["total"] == "47.80"

    def test_first_occurrence_wins_on_duplicate_keys(self):
        """When multiple pages share a key, the first page's value wins."""
        pytest.importorskip("torch")
        from donut_evaluator import DonutEvaluator

        evaluator = object.__new__(DonutEvaluator)
        evaluator.parse_failure_count = 0
        evaluator._inference_call_count = 0

        class _FakeProcessor:
            def token2json(self, tokens):
                return [
                    {"company": "FIRST"},
                    {"company": "SECOND", "total": "10.00"},
                ]

        evaluator.processor = _FakeProcessor()
        result = evaluator._parse_prediction("<tokens>")
        assert result["company"] == "FIRST", "First page's value should win"
        assert result["total"] == "10.00"

    def test_no_parse_failure_counted_for_list(self):
        """List output is NOT a parse failure — it contains valid data."""
        evaluator = self._make_evaluator_stub()
        evaluator._parse_prediction("<tokens>")
        assert evaluator.parse_failure_count == 0

    def test_empty_list_counts_as_failure(self):
        """Fully empty list (no dict pages) is a parse failure."""
        pytest.importorskip("torch")
        from donut_evaluator import DonutEvaluator

        evaluator = object.__new__(DonutEvaluator)
        evaluator.parse_failure_count = 0
        evaluator._inference_call_count = 0

        class _FakeProcessor:
            def token2json(self, tokens):
                return []

        evaluator.processor = _FakeProcessor()
        result = evaluator._parse_prediction("<tokens>")
        assert result == {}
        assert evaluator.parse_failure_count == 1


# ---------------------------------------------------------------------------
# _unwrap_prediction
# ---------------------------------------------------------------------------

class TestUnwrapPrediction:
    def test_sroie_unwrap(self):
        parsed = {"sroie": {"company": "ACME", "total": "10.00"}}
        result = _unwrap_prediction(parsed, "<s_sroie>")
        assert result == {"company": "ACME", "total": "10.00"}

    def test_cord_unwrap(self):
        parsed = {"cord-v2": {"menu": "item1"}}
        result = _unwrap_prediction(parsed, "<s_cord>")
        assert result == {"menu": "item1"}

    def test_no_wrapper_passthrough(self):
        parsed = {"company": "ACME", "total": "10.00"}
        result = _unwrap_prediction(parsed, "<s_sroie>")
        assert result == {"company": "ACME", "total": "10.00"}

    def test_non_dict_passthrough(self):
        result = _unwrap_prediction("not a dict", "<s_sroie>")
        assert result == "not a dict"

    def test_wrong_wrapper_key(self):
        parsed = {"other": {"company": "ACME"}}
        result = _unwrap_prediction(parsed, "<s_sroie>")
        assert result == {"other": {"company": "ACME"}}


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        gt = [{"company": "ACME", "date": "01/01/2024", "address": "123 Main St", "total": "10.00"}]
        preds = [{"company": "ACME", "date": "01/01/2024", "address": "123 Main St", "total": "10.00"}]
        m = compute_metrics(preds, gt)
        assert m["global_f1"] == 1.0
        assert m["global_precision"] == 1.0
        assert m["global_recall"] == 1.0
        assert m["overall_exact_match"] == 1.0

    def test_all_wrong_predictions(self):
        gt = [{"company": "ACME", "date": "01/01/2024", "address": "123 Main St", "total": "10.00"}]
        preds = [{"company": "WRONG", "date": "WRONG", "address": "WRONG", "total": "WRONG"}]
        m = compute_metrics(preds, gt)
        assert m["global_f1"] == 0.0

    def test_empty_predictions(self):
        gt = [{"company": "ACME", "date": "01/01/2024", "address": "123 Main St", "total": "10.00"}]
        preds = [{"company": "", "date": "", "address": "", "total": ""}]
        m = compute_metrics(preds, gt)
        assert m["global_f1"] == 0.0
        assert m["global_precision"] == 0

    def test_partial_match(self):
        gt = [{"company": "ACME", "date": "01/01/2024", "address": "123 Main St", "total": "10.00"}]
        preds = [{"company": "ACME", "date": "01/01/2024", "address": "WRONG", "total": "WRONG"}]
        m = compute_metrics(preds, gt)
        assert 0 < m["global_f1"] < 1.0
        assert m["company_f1"] == 1.0
        assert m["date_f1"] == 1.0
        assert m["address_f1"] == 0.0
        assert m["total_f1"] == 0.0

    def test_case_insensitive(self):
        gt = [{"company": "ACME Corp", "date": "", "address": "", "total": ""}]
        preds = [{"company": "acme corp", "date": "", "address": "", "total": ""}]
        m = compute_metrics(preds, gt)
        assert m["company_f1"] == 1.0

    def test_whitespace_stripped(self):
        gt = [{"company": "  ACME  ", "date": "", "address": "", "total": ""}]
        preds = [{"company": "ACME", "date": "", "address": "", "total": ""}]
        m = compute_metrics(preds, gt)
        assert m["company_f1"] == 1.0

    def test_multiple_samples(self):
        gt = [
            {"company": "A", "date": "1", "address": "X", "total": "10"},
            {"company": "B", "date": "2", "address": "Y", "total": "20"},
        ]
        preds = [
            {"company": "A", "date": "1", "address": "X", "total": "10"},
            {"company": "B", "date": "2", "address": "Z", "total": "99"},
        ]
        m = compute_metrics(preds, gt)
        assert m["global_precision"] == 0.75
        assert m["global_recall"] == 0.75
        assert m["overall_exact_match"] == 0.5

    def test_per_field_ned(self):
        gt = [{"company": "ACME", "date": "01/01", "address": "", "total": ""}]
        preds = [{"company": "ACME", "date": "01/01", "address": "", "total": ""}]
        m = compute_metrics(preds, gt)
        assert m["company_ned"] == 0.0
        assert m["date_ned"] == 0.0

    def test_missing_field_in_pred(self):
        gt = [{"company": "ACME", "date": "01/01", "address": "123 St", "total": "10"}]
        preds = [{"company": "ACME"}]
        m = compute_metrics(preds, gt)
        assert m["company_f1"] == 1.0
        assert m["date_f1"] == 0.0

    def test_returns_all_expected_keys(self):
        gt = [{"company": "A", "date": "B", "address": "C", "total": "D"}]
        preds = [{"company": "A", "date": "B", "address": "C", "total": "D"}]
        m = compute_metrics(preds, gt)
        expected_keys = {
            "global_precision", "global_recall", "global_f1", "overall_exact_match",
            "company_f1", "company_ned", "date_f1", "date_ned",
            "address_f1", "address_ned", "total_f1", "total_ned",
        }
        assert expected_keys.issubset(set(m.keys()))


# ---------------------------------------------------------------------------
# load_model_with_tied_weights — checkpoint sanity check
# ---------------------------------------------------------------------------

class TestLoadModelWithTiedWeights:
    """Verify load_model_with_tied_weights raises loudly when lm_head is missing.

    Root cause of F1~0.42: safetensors deduplicates lm_head.weight when it
    shares a data pointer with embed_tokens.weight, so per-epoch checkpoints
    omit lm_head.  When load_best_model_at_end reloads the best epoch,
    lm_head is randomly re-initialized.  The fix (LmHeadCloneCallback) forces
    a deep clone before every save.  This sanity check ensures the pipeline
    fails loudly if lm_head is still missing despite the callback.
    """

    def _make_mock_model(self, tie_word_embeddings: bool):
        """Return a minimal mock VisionEncoderDecoderModel."""
        decoder_config = mock.MagicMock()
        decoder_config.tie_word_embeddings = tie_word_embeddings
        decoder = mock.MagicMock()
        decoder.config = decoder_config
        model = mock.MagicMock()
        model.decoder = decoder
        model.to = mock.MagicMock(return_value=model)
        model.eval = mock.MagicMock(return_value=None)
        return model

    def test_raises_when_lm_head_missing_and_tie_false(self):
        """RuntimeError raised when lm_head.weight absent and tie_word_embeddings=False."""
        from donut_evaluator import load_model_with_tied_weights

        mock_model = self._make_mock_model(tie_word_embeddings=False)
        loading_info = {"missing_keys": ["decoder.lm_head.weight"], "unexpected_keys": []}

        with mock.patch(
            "donut_evaluator.VisionEncoderDecoderModel.from_pretrained",
            return_value=(mock_model, loading_info),
        ):
            with pytest.raises(RuntimeError, match="CRITICAL"):
                load_model_with_tied_weights("/fake/checkpoint")

    def test_no_raise_when_lm_head_present(self):
        """No RuntimeError when lm_head.weight is present in the checkpoint."""
        from donut_evaluator import load_model_with_tied_weights

        mock_model = self._make_mock_model(tie_word_embeddings=False)
        loading_info = {"missing_keys": [], "unexpected_keys": []}

        with mock.patch(
            "donut_evaluator.VisionEncoderDecoderModel.from_pretrained",
            return_value=(mock_model, loading_info),
        ):
            # Should not raise — lm_head is present
            result = load_model_with_tied_weights("/fake/checkpoint")
            assert result is mock_model

    def test_no_raise_for_legacy_tied_checkpoint(self):
        """No RuntimeError for old-style checkpoints with tie_word_embeddings=True.

        Legacy checkpoints tie lm_head to embed_tokens, so lm_head.weight is
        legitimately absent from the shard — _retie_decoder_head re-ties it.
        """
        from donut_evaluator import load_model_with_tied_weights

        mock_model = self._make_mock_model(tie_word_embeddings=True)
        loading_info = {"missing_keys": ["decoder.lm_head.weight"], "unexpected_keys": []}

        with mock.patch(
            "donut_evaluator.VisionEncoderDecoderModel.from_pretrained",
            return_value=(mock_model, loading_info),
        ):
            # Should not raise — legacy tied checkpoint, _retie_decoder_head handles it
            load_model_with_tied_weights("/fake/checkpoint")
