"""Tests for evaluation metrics — compute_metrics, normalized_edit_distance, _unwrap_prediction.

These tests exercise the core metric logic without requiring GPU or model weights.
Requires: torch, transformers (donut_evaluator.py imports them at module level).
"""

import unittest.mock as mock
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch required by donut_evaluator.py")
pytest.importorskip("transformers", reason="transformers required by donut_evaluator.py")

from donut_evaluator import _unwrap_prediction, compute_metrics, normalized_edit_distance  # noqa: E402, I001


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
        preds = [
            {"company": "ACME", "date": "01/01/2024", "address": "123 Main St", "total": "10.00"}
        ]
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
            "global_precision",
            "global_recall",
            "global_f1",
            "overall_exact_match",
            "company_f1",
            "company_ned",
            "date_f1",
            "date_ned",
            "address_f1",
            "address_ned",
            "total_f1",
            "total_ned",
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


# ---------------------------------------------------------------------------
# DonutEvaluator.evaluate() — allow_high_parse_failures parameter
# ---------------------------------------------------------------------------


class TestAllowHighParseFailures:
    """Regression tests for the allow_high_parse_failures fix.

    Root cause of the pipeline crash: an undertrained full-run model (not
    mini/micro) that produces 100% parse failures raised RuntimeError from
    evaluate().  The _is_undertrained guard in run_experiment() only caught
    this error for skip_step_validation=True (mini/micro) runs, so the
    exception propagated all the way up and killed Experiments 2–8.

    Fix: evaluate(allow_high_parse_failures=True) returns a zero-metric
    EvaluationResult instead of raising, and run_experiments.py passes
    allow_high_parse_failures=True unconditionally.
    """

    def _make_evaluator_with_all_failures(self, n_samples: int = 10):
        """Return a DonutEvaluator stub that simulates 100% parse failures."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from donut_evaluator import DonutEvaluator, EvaluationResult

        evaluator = object.__new__(DonutEvaluator)
        evaluator.parse_failure_count = 0
        evaluator._inference_call_count = 0
        evaluator.max_length = 768
        evaluator.task_prompt = "<s_sroie>"
        evaluator.device = "cpu"
        # Use non-empty ground truth for all fields so failures produce F1=0.0
        # even when the model could theoretically have predicted correctly.
        evaluator.test_dataset = [
            (
                Path("/fake/img.jpg"),
                {
                    "company": "MYDIN MALL",
                    "date": "25/12/2023",
                    "address": "123 ST",
                    "total": "9.90",
                },
            )
        ] * n_samples

        class _FakeProcessor:
            def token2json(self, tokens):
                return {}

        evaluator.processor = _FakeProcessor()

        # Patch _self_test to be a no-op.
        evaluator._self_test = lambda: None

        # Patch _run_inference to return {} AND increment parse_failure_count,
        # mimicking what the real _parse_prediction does on a bad token sequence.
        def _failing_inference(img_path, task_prompt, preloaded_image=None):
            evaluator.parse_failure_count += 1
            return {}

        evaluator._run_inference = _failing_inference

        return evaluator, EvaluationResult

    def test_raises_by_default_when_threshold_exceeded(self):
        """evaluate() raises RuntimeError when >50% parse failures and flag is False."""
        evaluator, _ = self._make_evaluator_with_all_failures(n_samples=10)
        with pytest.raises(RuntimeError, match="Parse failure threshold exceeded"):
            evaluator.evaluate(allow_high_parse_failures=False)

    def test_returns_zero_metrics_when_flag_true(self):
        """evaluate(allow_high_parse_failures=True) returns zero-metric EvaluationResult."""
        evaluator, EvaluationResult = self._make_evaluator_with_all_failures(n_samples=10)
        result = evaluator.evaluate(allow_high_parse_failures=True)
        assert result.global_f1 == 0.0
        assert result.global_precision == 0.0
        assert result.global_recall == 0.0
        assert result.overall_exact_match == 0.0
        assert result.parse_failures == 10

    def test_per_field_zeros_when_flag_true(self):
        """Per-field metrics are all zero / NED=1.0 when flag is True."""
        evaluator, _ = self._make_evaluator_with_all_failures(n_samples=4)
        result = evaluator.evaluate(allow_high_parse_failures=True)
        for field_name in ["company", "date", "address", "total"]:
            assert result.per_field[field_name]["f1"] == 0.0, (
                f"Expected {field_name}_f1=0.0 but got {result.per_field[field_name]['f1']}"
            )
            assert result.per_field[field_name]["ned"] == 1.0, (
                f"Expected {field_name}_ned=1.0 but got {result.per_field[field_name]['ned']}"
            )

    def test_default_false_keeps_existing_behaviour(self):
        """Calling evaluate() without the argument still raises (backward compat)."""
        evaluator, _ = self._make_evaluator_with_all_failures(n_samples=6)
        with pytest.raises(RuntimeError, match="Parse failure threshold exceeded"):
            evaluator.evaluate()

    def test_no_raise_below_threshold(self):
        """evaluate(allow_high_parse_failures=True) does not change low-failure behaviour."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from donut_evaluator import DonutEvaluator

        evaluator = object.__new__(DonutEvaluator)
        evaluator.parse_failure_count = 0  # zero failures
        evaluator._inference_call_count = 0
        evaluator.max_length = 768
        evaluator.task_prompt = "<s_sroie>"
        evaluator.device = "cpu"
        evaluator.test_dataset = [
            (
                Path("/fake/img.jpg"),
                {"company": "ACME", "date": "01/01", "address": "123 St", "total": "10"},
            )
        ]

        class _FakeProcessor:
            def token2json(self, tokens):
                return {"company": "ACME", "date": "01/01", "address": "123 St", "total": "10"}

        evaluator.processor = _FakeProcessor()
        evaluator._self_test = lambda: None
        evaluator._run_inference = lambda *a, **kw: {
            "company": "ACME",
            "date": "01/01",
            "address": "123 St",
            "total": "10",
        }

        # Should return normal (non-zero) metrics — flag has no effect below threshold
        result = evaluator.evaluate(allow_high_parse_failures=True)
        assert result.global_f1 == 1.0


# ---------------------------------------------------------------------------
# run_experiments.py — parse failure catch block covers full (non-mini) runs
# ---------------------------------------------------------------------------


class TestRunExperimentsParseFailureCatch:
    """Regression test: 'Parse failure threshold exceeded' must be caught for
    full (non-mini/micro) runs, not just skip_step_validation=True runs.

    Before the fix, the _is_undertrained guard meant that a full-run Exp 1
    crashing with 100% parse failures would re-raise and kill Experiments 2–8.
    """

    def test_evaluate_experiment_passes_allow_high_parse_failures(self):
        """evaluate_experiment() must call evaluator.evaluate(allow_high_parse_failures=True).

        Uses AST inspection so the check is immune to code reformatting: we
        look for a keyword node `allow_high_parse_failures=True` inside any
        Call node within the function body.
        """
        import ast
        import inspect

        # Guard torch/transformers per Pattern 7
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from run_experiments import evaluate_experiment  # noqa: E402

        assert callable(evaluate_experiment), "evaluate_experiment must be callable"

        src = inspect.getsource(evaluate_experiment)
        tree = ast.parse(src)

        # Collect all keyword arguments named 'allow_high_parse_failures' that
        # are set to the constant True anywhere inside the function.
        found = any(
            isinstance(node, ast.keyword)
            and node.arg == "allow_high_parse_failures"
            and isinstance(node.value, ast.Constant)
            and node.value.value is True
            for node in ast.walk(tree)
        )
        assert found, (
            "evaluate_experiment() must pass allow_high_parse_failures=True to "
            "evaluator.evaluate(). Without this, undertrained full-run models "
            "that produce 100% parse failures will crash the entire pipeline."
        )

    def test_run_experiment_catch_block_covers_full_runs(self):
        """The except RuntimeError block in run_experiment() must not use
        _is_undertrained to gate 'Parse failure threshold exceeded' handling.

        Uses AST inspection to check for Name nodes (variable references), so
        the check is not confused by comments or docstrings.
        """
        import ast
        import inspect

        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        from run_experiments import run_experiment  # noqa: E402

        src = inspect.getsource(run_experiment)
        tree = ast.parse(src)

        # Check that no Name node in the AST refers to the removed _is_undertrained
        # variable.  ast.Name nodes are variable references, not comments/strings.
        names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
        assert "_is_undertrained" not in names, (
            "run_experiment() still references _is_undertrained. "
            "Remove this guard so 'Parse failure threshold exceeded' is caught "
            "for all run types (full, mini, micro)."
        )


# ---------------------------------------------------------------------------
# Required named tests from CLAUDE.md §16
# ---------------------------------------------------------------------------


def test_lm_head_not_missing_after_reload():
    """Checkpoint reload must not drop lm_head.weight (safetensors dedup guard).

    Root cause of F1~0.42: safetensors deduplicates lm_head.weight when it
    shares a data pointer with embed_tokens.weight after resize_token_embeddings().
    LmHeadCloneCallback breaks the aliasing before each save so the weight is
    written to the shard.  load_model_with_tied_weights() raises immediately if
    lm_head is still absent (tie_word_embeddings=False checkpoints only).

    This test asserts the success path: when lm_head IS present in the
    checkpoint, load_model_with_tied_weights returns the model without raising.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import unittest.mock as _mock

    from donut_evaluator import load_model_with_tied_weights  # noqa: E402

    decoder_config = _mock.MagicMock()
    decoder_config.tie_word_embeddings = False
    decoder = _mock.MagicMock()
    decoder.config = decoder_config
    mock_model = _mock.MagicMock()
    mock_model.decoder = decoder
    mock_model.to = _mock.MagicMock(return_value=mock_model)
    mock_model.eval = _mock.MagicMock(return_value=None)

    # lm_head is present — missing_keys is empty
    loading_info = {"missing_keys": [], "unexpected_keys": []}

    with _mock.patch(
        "donut_evaluator.VisionEncoderDecoderModel.from_pretrained",
        return_value=(mock_model, loading_info),
    ):
        result = load_model_with_tied_weights("/fake/checkpoint")

    assert result is mock_model, (
        "load_model_with_tied_weights must return the model when lm_head is present"
    )


def test_token2json_list_output_merged():
    """token2json list output (CORD <sep/> pages) must be merged into a flat dict.

    Root cause of F1=0.0078: _parse_prediction() returned {} when token2json()
    returned a list (CORD multi-page format), collapsing all predictions to
    empty dicts.  Fix: merge list pages into a single flat dict (first value wins).
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from donut_evaluator import DonutEvaluator  # noqa: E402

    evaluator = object.__new__(DonutEvaluator)
    evaluator.parse_failure_count = 0
    evaluator._inference_call_count = 0

    class _FakeProcessor:
        def token2json(self, tokens):
            # Simulate multi-page list (CORD <sep/> behaviour leaking into SROIE)
            return [
                {"company": "MYDIN MALL", "date": "25/12/2023"},
                {"address": "NO 1 JALAN PUCHONG", "total": "47.80"},
            ]

    evaluator.processor = _FakeProcessor()

    result = evaluator._parse_prediction("<irrelevant tokens>")

    assert isinstance(result, dict), f"Expected dict after merge, got {type(result)}"
    assert result.get("company") == "MYDIN MALL"
    assert result.get("date") == "25/12/2023"
    assert result.get("address") == "NO 1 JALAN PUCHONG"
    assert result.get("total") == "47.80"
    assert evaluator.parse_failure_count == 0, "List merge must NOT count as a parse failure"


# ---------------------------------------------------------------------------
# Bug 1: Unified exact-match F1 in benchmark_compare.py
# ---------------------------------------------------------------------------


class TestBenchmarkCompareMetricUnification:
    """Assert that benchmark_compare._token_f1 is exact-match and
    _token_f1_squad is the SQuAD bag-of-words variant.

    The point: cross-architecture comparison (DONUT vs TrOCR+YOLO) must use
    the same metric protocol so numbers are comparable.  Before the fix,
    _token_f1 was the SQuAD partial-credit function, inflating TrOCR+YOLO
    scores relative to DONUT's exact-match scores.
    """

    def _get_fns(self):
        """Import both metric functions from benchmark_compare without torch."""
        import sys
        # benchmark_compare imports torch/PIL at module level; mock them
        mods = {
            "torch": mock.MagicMock(),
            "PIL": mock.MagicMock(),
            "PIL.Image": mock.MagicMock(),
            "tqdm": mock.MagicMock(),
            "matplotlib": mock.MagicMock(),
            "matplotlib.pyplot": mock.MagicMock(),
            "editdistance": mock.MagicMock(),
            "ultralytics": mock.MagicMock(),
        }
        # Avoid re-importing if already present (torch might be available)
        saved = {}
        for k, v in mods.items():
            if k not in sys.modules:
                saved[k] = v
        with mock.patch.dict(sys.modules, saved):
            import benchmark_compare as bc  # noqa: I001
            import importlib

            importlib.reload(bc)
            return bc._token_f1, bc._token_f1_squad

    def test_token_f1_is_exact_match(self):
        """_token_f1 must return 1.0 only on exact match, 0.0 on partial match."""
        try:
            import benchmark_compare as bc
        except ImportError:
            pytest.skip("benchmark_compare unavailable")
        # Exact match → 1.0
        assert bc._token_f1("WATSON SODA", "watson soda") == 1.0
        # Partial word overlap → must be 0.0 (exact-match protocol)
        assert bc._token_f1("WATSON SODA SNACKS", "watson soda") == 0.0

    def test_token_f1_squad_is_partial_credit(self):
        """_token_f1_squad must give partial credit for overlapping tokens."""
        try:
            import benchmark_compare as bc
        except ImportError:
            pytest.skip("benchmark_compare unavailable")
        # "WATSON SODA SNACKS" vs "watson soda" — 2 tokens in common
        score = bc._token_f1_squad("WATSON SODA SNACKS", "watson soda")
        assert 0.0 < score < 1.0, (
            f"_token_f1_squad should give partial credit, got {score}"
        )

    def test_exact_match_and_squad_differ_on_partial(self):
        """Confirm the two functions produce different scores on a partial match
        so that the change from _token_f1_squad to _token_f1 actually matters."""
        try:
            import benchmark_compare as bc
        except ImportError:
            pytest.skip("benchmark_compare unavailable")
        pred = "123 MAIN STREET SINGAPORE"
        gold = "123 MAIN STREET"
        exact = bc._token_f1(pred, gold)
        squad = bc._token_f1_squad(pred, gold)
        assert exact != squad, (
            "Exact-match and SQuAD scores must differ on a partial-match example; "
            f"both returned {exact}"
        )
        assert exact == 0.0, f"Exact-match should be 0.0 for partial, got {exact}"
        assert squad > 0.0, f"SQuAD partial credit should be > 0.0, got {squad}"

    def test_both_empty_returns_one(self):
        """Both functions must return 1.0 when both pred and gold are empty."""
        try:
            import benchmark_compare as bc
        except ImportError:
            pytest.skip("benchmark_compare unavailable")
        assert bc._token_f1("", "") == 1.0
        assert bc._token_f1_squad("", "") == 1.0


# ---------------------------------------------------------------------------
# Bug 4: Interactive selection must not be silently bypassed
# ---------------------------------------------------------------------------


class TestInteractiveSelectionFallback:
    """Assert _load_experiment_configs_for_run() always opens the interactive
    selection screen when --interactive is set, even without experiments/ dir."""

    def test_interactive_flag_triggers_prompt_without_experiments_dir(self, tmp_path, monkeypatch):
        """When experiments/ does not exist and --interactive is True, the
        function must call _interactive_experiment_selection (not return [])."""
        import sys
        import types

        monkeypatch.chdir(tmp_path)  # clean dir — no experiments/ subdir

        # Minimal args namespace with interactive=True
        args = types.SimpleNamespace(interactive=True, experiments=None)

        # Stub run_experiments.EXPERIMENTS so the legacy fallback works
        stub_config = types.SimpleNamespace(id=1, name="SROIE only", arch_type="donut")
        fake_re_mod = types.MagicMock()
        fake_re_mod.EXPERIMENTS = {1: stub_config}
        monkeypatch.setitem(sys.modules, "run_experiments", fake_re_mod)

        # Capture whether _interactive_experiment_selection is called
        called_with = []

        def fake_interactive(all_configs):
            called_with.extend(all_configs)
            return list(all_configs)

        # Import and monkeypatch _load_experiment_configs_for_run
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import run_all
        monkeypatch.setattr(run_all, "_interactive_experiment_selection", fake_interactive)

        result = run_all._load_experiment_configs_for_run(args)

        assert called_with, (
            "_interactive_experiment_selection was never called — "
            "interactive flag was silently bypassed"
        )
        assert result == [stub_config], f"Expected [stub_config], got {result}"
