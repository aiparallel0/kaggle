"""Tests for evaluation metrics — compute_metrics, normalized_edit_distance, _unwrap_prediction.

These tests exercise the core metric logic without requiring GPU or model weights.
Requires: torch, transformers (evaluate.py imports them at module level).
"""

import sys
from pathlib import Path

import pytest
from donut_evaluator import compute_metrics, normalized_edit_distance, _unwrap_prediction

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch", reason="torch required by donut_evaluator.py")
pytest.importorskip("transformers", reason="transformers required by donut_evaluator.py")



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
