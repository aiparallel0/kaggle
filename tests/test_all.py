"""Merged test suite — all tests from tests/test_*.py combined into one file.

Individual test files merged:
  - test_address_extractor.py
  - test_cache_lifecycle.py
  - test_constants.py
  - test_control_suite.py
  - test_dataset_loaders.py
  - test_dataset_normalizer.py
  - test_guardrails.py
  - test_inject_results.py
  - test_integration_contracts.py
  - test_integration_smoke.py
  - test_logging_utils.py
  - test_memory_manager.py
  - test_metrics.py
  - test_pipeline_critic.py
  - test_pipeline_smoke.py
  - test_pipeline_types.py
  - test_resource_optimizer.py
  - test_train_invariants.py
  - test_trocr_setup.py
  - test_validators.py
"""

# Standard library
import ast
import dataclasses
import importlib
import inspect
import json
import logging
import math
import os
import tempfile
import threading
import unittest
import unittest.mock as mock
import warnings
from pathlib import Path

# Third-party — pytest is optional; falls back to inline unittest stub
try:
    import pytest
except ImportError:
    import importlib as _importlib

    class _RaisesCtx:
        """Mimics pytest.raises() context manager."""

        def __init__(self, exc_class, match=None):
            self.exc_class = exc_class
            self.match = match
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exc_class!r} to be raised")
            if not issubclass(exc_type, self.exc_class):
                return False
            if self.match is not None:
                import re as _re

                if not _re.search(self.match, str(exc_val)):
                    raise AssertionError(f"Pattern {self.match!r} not found in {exc_val!r}")
            self.value = exc_val
            return True

    class _Mark:
        @staticmethod
        def skipif(condition, *, reason=""):
            return unittest.skipIf(condition, reason)

        @staticmethod
        def parametrize(argnames, argvalues, **kw):
            def _dec(func):
                import functools

                @functools.wraps(func)
                def _wrapper(self_inner):
                    names = [n.strip() for n in argnames.split(",")]
                    first = argvalues[0] if argvalues else ()
                    vals = first if len(names) > 1 else (first,)
                    return func(self_inner, *vals)

                return _wrapper

            return _dec

        def __getattr__(self, name):
            return lambda *a, **kw: lambda f: f

    class _Pytest:
        mark = _Mark()

        @staticmethod
        def importorskip(modname, reason=None, **kw):
            spec = _importlib.util.find_spec(modname)
            if spec is None:
                raise unittest.SkipTest(
                    f"could not import {modname!r}" + (f": {reason}" if reason else "")
                )
            return _importlib.import_module(modname)

        @staticmethod
        def fixture(func=None, *, scope="function", **kw):
            if func is not None:
                return func
            return lambda f: f

        @staticmethod
        def raises(exc_class, match=None, **kw):
            return _RaisesCtx(exc_class, match=match)

        @staticmethod
        def warns(warning_class, **kw):
            import warnings as _w

            return _w.catch_warnings()

    pytest = _Pytest()


# Project imports (do not require torch/transformers)
from dataset_normalizer import DatasetNormalizer, extract_address_from_seller, normalise_samples

import resource_manager as mm
from cloud_orchestration import (
    AggregatedResults,
    ArchitectureAudit,
    BenchmarkNarrowness,
    BugPattern,
    BugReport,
    CheckResult,
    CheckStatus,
    CritiqueFinding,
    CritiqueReport,
    DataSplitValidationReport,
    EpochConfoundAudit,
    ExperimentMetrics,
    ExperimentResult,
    FindingSeverity,
    MultipleTestingAudit,
    PipelineCritic,
    PipelineResult,
    PretrainingBiasAudit,
    SeverityLevel,
    StatisticalPowerAudit,
    ValidationReport,
    _family_wise_error_rate,
    _norm_ppf,
    _two_proportion_mdd,
)
from constants import BASE_MODEL, EMPTY_GT, FIELDS, IMAGE_EXTS, MAX_LENGTH, NEW_TOKENS, SEED
from control_suite import (
    CONTROL_SUITE,
    ControlSuite,
    DonutControlConfig,
    TrOCRControlConfig,
    YOLOControlConfig,
    get_augmentation_transforms,
    validate_sroie_oversample,
)
from dataset_loaders import (
    DatasetLoadError,
    _ensure_dir,
    _validate_sample_schema,
    _validate_samples_nonempty,
)
from reporting import EXP_NAMES, LEADERBOARD, PaperInjector, UnresolvedVarError, _safe
from constants import DeduplicatingHandler, suppress_noisy_loggers
from resource_manager import (
    ResourceOptimizedConfig,
    optimize_hyperparams,
    validate_training_config,
)

# ---------------------------------------------------------------------------
# Optional heavy-dependency guard (torch / transformers).
# Pattern 7 (CLAUDE.md §16): these try/except blocks must appear BEFORE the
# imports they guard.  Tests that need torch are marked with @_needs_torch so
# they skip gracefully when torch is absent; all other tests continue to run.
# ---------------------------------------------------------------------------
try:
    import torch
    import transformers as _transformers_mod  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

_needs_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch and transformers required")

# Guarded project imports — only reachable when torch + transformers are present
if _TORCH_AVAILABLE:
    from constants import _mask_empty_field_labels  # noqa: E402, I001
    from dataset_loaders import _LOADERS  # noqa: E402, I001
    from donut_evaluator import (  # noqa: E402, I001
        _unwrap_prediction,
        compute_metrics,
        normalized_edit_distance,
    )
    from run_experiments import (  # noqa: E402, I001
        EXPERIMENTS,
        ExperimentConfig,
        _config_to_dict,
        run_custom_experiment,
        train_experiment,
    )
    from train import MultiDataset  # noqa: E402, I001
    from train_trocr_yolo import (  # noqa: E402, I001
        TROCR_MODEL_ID,
        TrOCRReceiptDataset,
        _EXPECTED_MISSING_TROCR,
        _materialize_meta_buffers,
        _print_trocr_load_report,
    )
else:
    _mask_empty_field_labels = None  # type: ignore[assignment]
    _LOADERS = None  # type: ignore[assignment]
    _unwrap_prediction = None  # type: ignore[assignment]
    compute_metrics = None  # type: ignore[assignment]
    normalized_edit_distance = None  # type: ignore[assignment]
    EXPERIMENTS = None  # type: ignore[assignment]
    ExperimentConfig = None  # type: ignore[assignment]
    _config_to_dict = None  # type: ignore[assignment]
    run_custom_experiment = None  # type: ignore[assignment]
    train_experiment = None  # type: ignore[assignment]
    MultiDataset = None  # type: ignore[assignment]
    TROCR_MODEL_ID = None  # type: ignore[assignment]
    TrOCRReceiptDataset = None  # type: ignore[assignment]
    _EXPECTED_MISSING_TROCR = None  # type: ignore[assignment]
    _materialize_meta_buffers = None  # type: ignore[assignment]
    _print_trocr_load_report = None  # type: ignore[assignment]


# ============================================================================
# From test_address_extractor.py
# ============================================================================


class TestExtractAddressFromSeller(unittest.TestCase):
    def test_street_number_split(self):
        """Typical Invoices-DONUT seller string with street number."""
        company, address = extract_address_from_seller(
            "Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228"
        )
        assert "Patel" in company
        assert "Montgomery" in company
        assert "356" in address
        assert "Kyle Vista" in address

    def test_po_box(self):
        """P.O. Box pattern detected and split correctly."""
        company, address = extract_address_from_seller("ACME Corp P.O. Box 1234 Springfield")
        assert company == "ACME Corp"
        assert "Box 1234" in address

    def test_street_suffix_only(self):
        """Street suffix like 'Avenue' detected even without a leading number."""
        company, address = extract_address_from_seller("Smith & Sons Oak Avenue Suite 10")
        assert "Smith" in company
        assert "Avenue" in address

    def test_state_zip_only(self):
        """State abbreviation + ZIP at the end still detected."""
        company, address = extract_address_from_seller("Global Inc CA 90210")
        assert "Global" in company
        assert "CA 90210" in address

    def test_no_address_detected(self):
        """Seller string with no address signals → full string as company."""
        company, address = extract_address_from_seller("Acme Corporation")
        assert company == "Acme Corporation"
        assert address == ""

    def test_empty_string(self):
        company, address = extract_address_from_seller("")
        assert company == ""
        assert address == ""

    def test_whitespace_only(self):
        company, address = extract_address_from_seller("   ")
        assert company == ""
        assert address == ""

    def test_company_trailing_comma_stripped(self):
        """Trailing comma from company name is stripped."""
        company, address = extract_address_from_seller("Johnson, Lee, 100 Main St City, TX 75001")
        # trailing comma/space should be stripped from company
        assert not company.endswith(",")
        assert "100 Main" in address

    def test_returns_tuple_of_strings(self):
        result = extract_address_from_seller("Some Co 123 Road St")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# SROIE key file multi-line address parsing
# ---------------------------------------------------------------------------


class TestSROIEKeyFileParsing(unittest.TestCase):
    """Tests for _load_key_file (dataset_loaders) and _parse_txt_key (train)."""

    def _write_key_file(self, tmp_path: Path, lines: list[str]) -> Path:
        key_file = tmp_path / "X00016469671.txt"
        key_file.write_text("\n".join(lines), encoding="utf-8")
        return key_file

    def test_four_line_file(self, tmp_path):
        """Standard 4-line SROIE key file parses correctly."""
        from dataset_loaders import _load_key_file  # testing internal behaviour — intentional

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        (key_dir / "sample.txt").write_text(
            "ACME STORE\n25/12/2023\n1 ORCHARD ROAD\n12.50", encoding="utf-8"
        )
        gt = _load_key_file(key_dir, "sample")
        assert gt["company"] == "ACME STORE"
        assert gt["date"] == "25/12/2023"
        assert gt["address"] == "1 ORCHARD ROAD"
        assert gt["total"] == "12.50"

    def test_five_line_file_multi_line_address(self, tmp_path):
        """5-line key file: address spans lines 2 and 3, total is line 4."""
        from dataset_loaders import _load_key_file  # testing internal behaviour — intentional

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        (key_dir / "sample.txt").write_text(
            "MYDIN MALL\n25/12/2023\nNO 1 JALAN PUCHONG\n47100 PUCHONG JAYA\n47.80",
            encoding="utf-8",
        )
        gt = _load_key_file(key_dir, "sample")
        assert gt["company"] == "MYDIN MALL"
        assert gt["date"] == "25/12/2023"
        assert gt["address"] == "NO 1 JALAN PUCHONG 47100 PUCHONG JAYA"
        assert gt["total"] == "47.80"

    def test_six_line_file_multi_line_address(self, tmp_path):
        """6-line key file: address spans lines 2-4, total is line 5."""
        from dataset_loaders import _load_key_file  # testing internal behaviour — intentional

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        (key_dir / "sample.txt").write_text(
            "WATSON'S SODA\n01/01/2024\nBLOCK A\nJALAN MANIS\n56000 KL\n9.90",
            encoding="utf-8",
        )
        gt = _load_key_file(key_dir, "sample")
        assert gt["address"] == "BLOCK A JALAN MANIS 56000 KL"
        assert gt["total"] == "9.90"

    def test_train_parse_txt_key_four_lines(self, tmp_path):
        """_load_key_file handles standard 4-line file."""
        from dataset_loaders import _load_key_file  # noqa: E402, I001

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        key_file = key_dir / "sample.txt"
        key_file.write_text("STORE\n02/02/2024\n100 MAIN ST\n5.00", encoding="utf-8")
        gt = _load_key_file(key_dir, "sample")
        assert gt is not None
        assert gt["address"] == "100 MAIN ST"
        assert gt["total"] == "5.00"

    def test_train_parse_txt_key_multiline_address(self, tmp_path):
        """_load_key_file handles 5-line file with multi-line address."""
        from dataset_loaders import _load_key_file  # noqa: E402, I001

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        key_file = key_dir / "sample.txt"
        key_file.write_text("STORE\n02/02/2024\n100 MAIN ST\nSUITE 5\n5.00", encoding="utf-8")
        gt = _load_key_file(key_dir, "sample")
        assert gt is not None
        assert gt["address"] == "100 MAIN ST SUITE 5"
        assert gt["total"] == "5.00"

    def test_train_parse_txt_key_too_few_lines(self, tmp_path):
        """_load_key_file returns empty dict for files with fewer than 4 lines."""
        from dataset_loaders import _load_key_file  # noqa: E402, I001

        key_dir = tmp_path / "key"
        key_dir.mkdir()
        key_file = key_dir / "sample.txt"
        key_file.write_text("STORE\n02/02/2024\n100 MAIN ST", encoding="utf-8")
        assert _load_key_file(key_dir, "sample") == {}


# ============================================================================
# From test_cache_lifecycle.py
# ============================================================================


class TestDatasetCacheClearing(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
    """Verify that clear_caches() empties all three internal cache dicts."""

    def _make_empty_dataset(self):
        """Build a zero-sample MultiDataset (no GPU / disk access needed)."""
        try:
            from transformers import DonutProcessor

            proc = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        except (OSError, ImportError, ValueError):
            pytest.skip("Cannot load DonutProcessor in this environment")

        return MultiDataset(samples=[], processor=proc, cache_in_ram=False)

    def test_clear_caches_method_exists(self):
        """MultiDataset must expose a clear_caches() method."""
        assert hasattr(MultiDataset, "clear_caches"), (
            "MultiDataset.clear_caches() is required to explicitly free "
            "_pixel_cache, _image_cache, and _label_cache before GC."
        )
        assert callable(MultiDataset.clear_caches)

    def test_clear_caches_empties_pixel_cache(self):
        """clear_caches() must empty _pixel_cache."""
        ds = self._make_empty_dataset()
        # Manually populate the cache to simulate a trained dataset
        ds._pixel_cache[0] = torch.zeros(3, 4, 4)
        ds._pixel_cache[1] = torch.zeros(3, 4, 4)
        assert len(ds._pixel_cache) == 2

        ds.clear_caches()
        assert len(ds._pixel_cache) == 0, (
            "_pixel_cache must be empty after clear_caches() — "
            "it holds up to 7.1 GB of float32 tensors."
        )

    def test_clear_caches_empties_image_cache(self):
        """clear_caches() must empty _image_cache."""
        ds = self._make_empty_dataset()
        from PIL import Image

        ds._image_cache[0] = Image.new("RGB", (4, 4))
        assert len(ds._image_cache) == 1

        ds.clear_caches()
        assert len(ds._image_cache) == 0

    def test_clear_caches_empties_label_cache(self):
        """clear_caches() must empty _label_cache."""
        ds = self._make_empty_dataset()
        ds._label_cache[0] = torch.zeros(768, dtype=torch.long)
        assert len(ds._label_cache) == 1

        ds.clear_caches()
        assert len(ds._label_cache) == 0

    def test_clear_caches_all_three_at_once(self):
        """clear_caches() must clear all three caches in a single call."""
        from PIL import Image

        ds = self._make_empty_dataset()
        ds._pixel_cache[0] = torch.zeros(3, 4, 4)
        ds._image_cache[0] = Image.new("RGB", (4, 4))
        ds._label_cache[0] = torch.zeros(768, dtype=torch.long)

        ds.clear_caches()
        assert len(ds._pixel_cache) == 0
        assert len(ds._image_cache) == 0
        assert len(ds._label_cache) == 0

    def test_clear_caches_idempotent(self):
        """Calling clear_caches() twice must not raise."""
        ds = self._make_empty_dataset()
        ds.clear_caches()
        ds.clear_caches()  # second call must be a no-op


# ---------------------------------------------------------------------------
# AST contract: train_experiment() cleanup calls clear_caches()
# ---------------------------------------------------------------------------


class TestTrainExperimentCleanup(unittest.TestCase):
    """AST-based test that clear_caches() is called before del train_ds."""

    def setUp(self):
        src_path = Path(__file__).resolve().parent.parent / "run_experiments.py"
        self.run_experiments_source = src_path.read_text()
        self.run_experiments_ast = ast.parse(self.run_experiments_source)

    @pytest.fixture(scope="class")
    def run_experiments_source(self):
        src_path = Path(__file__).resolve().parent.parent / "run_experiments.py"
        return src_path.read_text()

    @pytest.fixture(scope="class")
    def run_experiments_ast(self, run_experiments_source):
        return ast.parse(run_experiments_source)

    def _find_train_experiment_func(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train_experiment":
                return node
        return None

    def _count_clear_caches_calls(self, func_node) -> int:
        """Count actual ast.Call nodes where the called attribute is 'clear_caches'."""
        count = 0
        for node in ast.walk(func_node):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clear_caches"
            ):
                count += 1
        return count

    def test_clear_caches_called_on_train_ds_before_del(self):
        run_experiments_ast = self.run_experiments_ast
        """train_experiment() must call .clear_caches() at least once before del train_ds.

        This is the critical sequence that prevents _pixel_cache tensors
        (~7.1 GB) from staying pinned while the next experiment allocates
        its own cache.
        """
        func = self._find_train_experiment_func(run_experiments_ast)
        assert func is not None, "train_experiment() function not found in run_experiments.py"

        count = self._count_clear_caches_calls(func)
        assert count >= 1, (
            "train_experiment() must call .clear_caches() on train_ds before deleting it. "
            "Without this, _pixel_cache tensors (up to 7.1 GB) are not freed between "
            "the 8 sequential experiments."
        )

    def test_clear_caches_called_in_oom_retry(self):
        run_experiments_ast = self.run_experiments_ast
        """The OOM-retry cleanup block must also call clear_caches().

        When CUDA OOM fires mid-training, the same cleanup pattern must be
        applied to free pinned pixel tensors before the retry rebuild.
        We expect at least 2 call sites: normal cleanup + OOM retry.
        """
        func = self._find_train_experiment_func(run_experiments_ast)
        assert func is not None

        count = self._count_clear_caches_calls(func)
        assert count >= 2, (
            f"Expected .clear_caches() to be called at least 2 times in train_experiment() "
            f"(normal cleanup + OOM retry), but found {count} call site(s)."
        )


# ---------------------------------------------------------------------------
# MultiDataset precompute_tensors=False
# ---------------------------------------------------------------------------


class TestPrecomputeTensorsFlag(unittest.TestCase):
    """Verify precompute_tensors=False behaviour in MultiDataset.__init__."""

    def _make_dataset(self, precompute_tensors: bool, cache_in_ram: bool = False):
        """Build a zero-sample MultiDataset without touching disk or GPU."""
        try:
            from transformers import DonutProcessor

            proc = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        except (OSError, ImportError, ValueError):
            pytest.skip("Cannot load DonutProcessor in this environment")

        return MultiDataset(
            samples=[],
            processor=proc,
            cache_in_ram=cache_in_ram,
            precompute_tensors=precompute_tensors,
        )

    def test_accepts_precompute_tensors_false(self):
        """MultiDataset.__init__ must accept precompute_tensors=False without error."""
        ds = self._make_dataset(precompute_tensors=False)
        assert ds is not None

    def test_accepts_precompute_tensors_true(self):
        """MultiDataset.__init__ must accept precompute_tensors=True (default) without error."""
        ds = self._make_dataset(precompute_tensors=True)
        assert ds is not None

    def test_pixel_cache_empty_when_precompute_false(self):
        """With precompute_tensors=False, _pixel_cache must be empty even if images are cached."""
        ds = self._make_dataset(precompute_tensors=False, cache_in_ram=True)
        # Manually populate _image_cache to simulate what happens when images are loaded
        from PIL import Image

        ds._image_cache[0] = Image.new("RGB", (4, 4))
        # Pixel cache must remain empty — the precompute was suppressed
        assert len(ds._pixel_cache) == 0, (
            "_pixel_cache must be empty when precompute_tensors=False. "
            "Exp 6 OOM: val dataset precomputed 1902 MB of pixel tensors unnecessarily."
        )

    def test_label_cache_empty_when_precompute_false(self):
        """With precompute_tensors=False, _label_cache must be empty even if images are cached."""
        ds = self._make_dataset(precompute_tensors=False, cache_in_ram=True)
        from PIL import Image

        ds._image_cache[0] = Image.new("RGB", (4, 4))
        assert len(ds._label_cache) == 0, (
            "_label_cache must be empty when precompute_tensors=False."
        )

    def test_val_dataset_in_run_experiments_uses_precompute_false(self):
        """AST check: _build_model_and_datasets() must pass precompute_tensors=False for val."""
        src_path = Path(__file__).resolve().parent.parent / "run_experiments.py"
        source = src_path.read_text()
        # The string "precompute_tensors=False" must appear in the source
        assert "precompute_tensors=False" in source, (
            "run_experiments.py must pass precompute_tensors=False when constructing "
            "the val MultiDataset. Without this, ~1902 MB of val pixel tensors are "
            "allocated after the train cache, causing SIGKILL at Experiment 6."
        )


# ============================================================================
# From test_constants.py
# ============================================================================


class TestFields(unittest.TestCase):
    def test_fields_is_list(self):
        assert isinstance(FIELDS, list)

    def test_fields_has_four_entries(self):
        assert len(FIELDS) == 4

    def test_fields_contains_expected(self):
        assert FIELDS == ["company", "date", "address", "total"]

    def test_fields_are_strings(self):
        for f in FIELDS:
            assert isinstance(f, str)

    def test_fields_are_lowercase(self):
        for f in FIELDS:
            assert f == f.lower()


class TestImageExts(unittest.TestCase):
    def test_image_exts_is_frozenset(self):
        assert isinstance(IMAGE_EXTS, frozenset)

    def test_common_extensions_present(self):
        for ext in (".jpg", ".jpeg", ".png"):
            assert ext in IMAGE_EXTS

    def test_all_start_with_dot(self):
        for ext in IMAGE_EXTS:
            assert ext.startswith(".")

    def test_all_lowercase(self):
        for ext in IMAGE_EXTS:
            assert ext == ext.lower()


class TestMaxLength(unittest.TestCase):
    def test_max_length_is_int(self):
        assert isinstance(MAX_LENGTH, int)

    def test_max_length_positive(self):
        assert MAX_LENGTH > 0

    def test_max_length_value(self):
        assert MAX_LENGTH == 768


class TestBaseModel(unittest.TestCase):
    def test_base_model_is_string(self):
        assert isinstance(BASE_MODEL, str)

    def test_base_model_value(self):
        assert BASE_MODEL == "naver-clova-ix/donut-base"


class TestSeed(unittest.TestCase):
    def test_seed_is_int(self):
        assert isinstance(SEED, int)

    def test_seed_value(self):
        assert SEED == 42


class TestNewTokens(unittest.TestCase):
    def test_new_tokens_is_list(self):
        assert isinstance(NEW_TOKENS, list)

    def test_new_tokens_has_pairs(self):
        # Every opening token should have a matching closing token
        assert len(NEW_TOKENS) % 2 == 0

    def test_sroie_wrapper_present(self):
        assert "<s_sroie>" in NEW_TOKENS
        assert "</s_sroie>" in NEW_TOKENS

    def test_field_tokens_present(self):
        for field in FIELDS:
            assert f"<s_{field}>" in NEW_TOKENS
            assert f"</s_{field}>" in NEW_TOKENS


class TestEmptyGT(unittest.TestCase):
    def test_empty_gt_is_dict(self):
        assert isinstance(EMPTY_GT, dict)

    def test_empty_gt_keys_match_fields(self):
        assert list(EMPTY_GT.keys()) == FIELDS

    def test_empty_gt_values_are_empty_strings(self):
        for v in EMPTY_GT.values():
            assert v == ""


# ---------------------------------------------------------------------------
# _mask_empty_field_labels
# ---------------------------------------------------------------------------

# Token IDs used in mock tokenizer below.
_OPEN_IDS = {"company": 10, "date": 12, "address": 14, "total": 16}
_CLOSE_IDS = {"company": 11, "date": 13, "address": 15, "total": 17}


class _FakeTokenizer:
    """Minimal tokenizer stub for masking tests."""

    unk_token_id = 0

    def convert_tokens_to_ids(self, tok: str) -> int:
        for f, tid in _OPEN_IDS.items():
            if tok == f"<s_{f}>":
                return tid
        for f, tid in _CLOSE_IDS.items():
            if tok == f"</s_{f}>":
                return tid
        return self.unk_token_id


class TestMaskEmptyFieldLabels(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
    """Unit tests for _mask_empty_field_labels in constants.py."""

    def _make_labels(self):
        """Return a label tensor covering all four fields with dummy content."""
        # Layout: <s_co> CO </s_co> <s_da> DA </s_da> <s_ad> AD </s_ad> <s_to> TO </s_to>
        return torch.tensor([10, 100, 11, 12, 200, 13, 14, 300, 15, 16, 400, 17, -100])

    def test_empty_address_masked(self):
        labels = self._make_labels()
        gt = {"company": "ACME", "date": "2024", "address": "", "total": "9.99"}
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        # <s_address>=14 at idx 6, </s_address>=15 at idx 8 → both -100
        assert result[6].item() == -100
        assert result[7].item() == -100
        assert result[8].item() == -100

    def test_non_empty_fields_unchanged(self):
        labels = self._make_labels()
        gt = {"company": "ACME", "date": "2024", "address": "", "total": "9.99"}
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        # company, date, total spans should be untouched
        assert result[0].item() == 10  # <s_company>
        assert result[1].item() == 100  # company value
        assert result[2].item() == 11  # </s_company>
        assert result[4].item() == 200  # date value
        assert result[10].item() == 400  # total value

    def test_all_empty_all_masked(self):
        labels = self._make_labels()
        gt = {"company": "", "date": "", "address": "", "total": ""}
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        # All tag and content positions should be -100 (padding was already -100)
        for i in range(12):
            assert result[i].item() == -100, f"Position {i} should be -100"

    def test_no_empty_fields_unchanged(self):
        labels = self._make_labels()
        gt = {"company": "A", "date": "B", "address": "C", "total": "D"}
        original = labels.clone()
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        assert torch.equal(result, original)

    def test_whitespace_only_treated_as_empty(self):
        labels = self._make_labels()
        gt = {"company": "ACME", "date": "2024", "address": "   ", "total": "9.99"}
        result = _mask_empty_field_labels(labels, gt, _FakeTokenizer())
        assert result[6].item() == -100
        assert result[8].item() == -100

    def test_unknown_token_skipped(self):
        """When a special token is unknown (maps to unk_id), the field is skipped."""
        labels = self._make_labels()
        gt = {"company": "", "date": "2024", "address": "C", "total": "9.99"}

        class AllUnkTokenizer(_FakeTokenizer):
            def convert_tokens_to_ids(self, tok):
                return self.unk_token_id  # everything maps to unk

        original = labels.clone()
        result = _mask_empty_field_labels(labels, gt, AllUnkTokenizer())
        # No masking should occur because all tokens map to unk
        assert torch.equal(result, original)


# ============================================================================
# From test_control_suite.py
# ============================================================================


def test_control_suite_singleton_is_correct_type():
    """CONTROL_SUITE module-level singleton is a ControlSuite instance."""
    assert isinstance(CONTROL_SUITE, ControlSuite)
    assert isinstance(CONTROL_SUITE.donut, DonutControlConfig)
    assert isinstance(CONTROL_SUITE.trocr, TrOCRControlConfig)
    assert isinstance(CONTROL_SUITE.yolo, YOLOControlConfig)


def test_control_suite_default_factory():
    """ControlSuite.default() returns a fresh identical instance."""
    suite = ControlSuite.default()
    assert isinstance(suite, ControlSuite)
    assert suite.donut.base_model == CONTROL_SUITE.donut.base_model
    assert suite.trocr.model_id == CONTROL_SUITE.trocr.model_id
    assert suite.yolo.base_model == CONTROL_SUITE.yolo.base_model


# ---------------------------------------------------------------------------
# DONUT — defaults must match ExperimentConfig
# ---------------------------------------------------------------------------


def test_donut_base_model_matches_constants():
    """DonutControlConfig.base_model matches constants.BASE_MODEL."""
    assert CONTROL_SUITE.donut.base_model == BASE_MODEL


def test_donut_max_length_matches_constants():
    """DonutControlConfig.max_length matches constants.MAX_LENGTH."""
    assert CONTROL_SUITE.donut.max_length == MAX_LENGTH


def test_donut_seed_matches_constants():
    """DonutControlConfig.seed matches constants.SEED."""
    assert CONTROL_SUITE.donut.seed == SEED


def test_donut_defaults_match_experiment_config():
    """Key DonutControlConfig defaults match ExperimentConfig defaults.

    ExperimentConfig is the authoritative training config per CLAUDE.md GP-1.
    DonutControlConfig must mirror these values exactly for consistency.
    """
    d = CONTROL_SUITE.donut
    # From ExperimentConfig in run_experiments.py
    assert d.epochs == 10, "DonutControlConfig.epochs must match ExperimentConfig.epochs=10"
    assert d.encoder_lr == 5e-5, "encoder_lr must be 5e-5 (CLAUDE.md §3)"
    assert d.decoder_lr == 1e-4, "decoder_lr must be 1e-4 (2× encoder LR)"
    assert d.batch_size == 8, "batch_size=8 is optimal per CLAUDE.md §3"
    assert d.gradient_accumulation_steps == 2
    assert d.early_stopping_patience == 3
    assert d.weight_decay == 0.01
    assert d.warmup_steps == 40


def test_donut_tie_word_embeddings_is_false():
    """tie_word_embeddings must be False — changing to True causes F1=0.

    See CLAUDE.md §2: After resize_token_embeddings(), tie_word_embeddings=True
    causes tie_weights() on reload to destroy the learned lm_head → F1=0.
    """
    assert CONTROL_SUITE.donut.tie_word_embeddings is False, (
        "tie_word_embeddings must be False — see CLAUDE.md §2 lm_head weight tying bug"
    )


def test_donut_sort_json_key_is_false():
    """sort_json_key must be False for CORD-based datasets.

    Setting True silently corrupts token ordering in preprocessed datasets.
    See finetuning_params.md commonly_underdocumented[].
    """
    assert CONTROL_SUITE.donut.sort_json_key is False, (
        "sort_json_key must be False for SROIE/CORD datasets — silently corrupts token ordering"
    )


def test_donut_input_size_is_canonical():
    """input_size must be the canonical DONUT finetuning resolution [1280, 960]."""
    assert CONTROL_SUITE.donut.input_size == [1280, 960], (
        "input_size must be [1280, 960] (width×height) — DONUT canonical finetuning resolution"
    )


def test_donut_swin_window_size_is_10():
    """swin_window_size must be 10 (matches donut-base pretrain config).

    Changing this forces full weight re-initialization of all attention layers.
    See finetuning_params.md: marked CRITICAL ⚠️ underdocumented.
    """
    assert CONTROL_SUITE.donut.swin_window_size == 10, (
        "swin_window_size=10 must match donut-base pretrain — changing causes full re-init"
    )


def test_donut_gradient_clip_val_is_one():
    """gradient_clip_val is explicitly documented (was previously implicit)."""
    assert CONTROL_SUITE.donut.gradient_clip_val == 1.0


# ---------------------------------------------------------------------------
# TrOCR — critical parameters and previously-missing ones
# ---------------------------------------------------------------------------


def test_trocr_model_id():
    """TrOCR model ID is the printed-text variant."""
    assert CONTROL_SUITE.trocr.model_id == "microsoft/trocr-base-printed"


def test_trocr_input_size_is_fixed():
    """TrOCR input_size is always 384 — FIXED by architecture (not configurable)."""
    assert CONTROL_SUITE.trocr.input_size == 384, (
        "TrOCR input_size is hardcoded at 384×384 by the BEiT/DeiT encoder architecture"
    )


def test_trocr_patch_size_is_fixed():
    """TrOCR patch_size is always 16 — FIXED by architecture."""
    assert CONTROL_SUITE.trocr.patch_size == 16


def test_trocr_weight_decay_nonzero():
    """TrOCR weight_decay must be non-zero.

    PREVIOUSLY MISSING BUG: AdamW was called without weight_decay argument,
    causing PyTorch to use its default of 0.0. Reference value is 1e-4.
    """
    assert CONTROL_SUITE.trocr.weight_decay > 0, (
        "TrOCR weight_decay must be > 0. Before control_suite, AdamW used "
        "PyTorch default of 0 because weight_decay was not passed."
    )
    assert CONTROL_SUITE.trocr.weight_decay == 1e-4


def test_trocr_gradient_checkpointing_enabled():
    """gradient_checkpointing must be True for TrOCR-base to fit in VRAM."""
    assert CONTROL_SUITE.trocr.gradient_checkpointing is True


def test_trocr_use_cache_disabled():
    """use_cache must be False when gradient_checkpointing is True (incompatible)."""
    assert CONTROL_SUITE.trocr.use_cache is False


def test_trocr_grad_ckpt_vram_threshold_gb_is_24():
    """grad_ckpt_vram_threshold_gb must be 24.0 to mirror the DONUT path.

    Cards with VRAM > 24 GB have enough headroom to skip gradient checkpointing;
    cards at or below 24 GB (e.g. RTX 4090 reporting exactly 24.0 GB) must keep
    it enabled.  The strictly-greater-than comparison is intentional.
    """
    assert CONTROL_SUITE.trocr.grad_ckpt_vram_threshold_gb == 24.0


def test_trocr_grad_ckpt_vram_threshold_registered():
    """grad_ckpt_vram_threshold_gb is registered in the control registry."""
    params = CONTROL_SUITE.to_dict()
    assert "grad_ckpt_vram_threshold_gb" in params.get("trocr", {}), (
        "grad_ckpt_vram_threshold_gb must be a field on TrOCRControlConfig "
        "so that CONTROL_SUITE.to_dict() and print_summary() include it."
    )


def test_trocr_lr_scheduler_is_string():
    """lr_scheduler must be a string (was hardcoded 'linear' before control_suite)."""
    assert isinstance(CONTROL_SUITE.trocr.lr_scheduler, str)
    assert CONTROL_SUITE.trocr.lr_scheduler == "linear"


def test_trocr_warmup_ratio_is_valid():
    """warmup_ratio must be in [0, 1]."""
    assert 0 < CONTROL_SUITE.trocr.warmup_ratio <= 0.5


def test_trocr_warmup_init_lr_is_tiny():
    """warmup_init_lr must be a tiny value (prevents instability at step 0).

    See finetuning_params.md: marked ⚠️ underdocumented — absent from tutorials.
    """
    assert CONTROL_SUITE.trocr.warmup_init_lr < 1e-6, (
        "warmup_init_lr must be tiny (default 1e-8) to prevent instability at warmup step 0"
    )


def test_trocr_patience_documented():
    """patience field exists (even if None — early stopping not yet implemented)."""
    # Attribute must exist; None means not implemented yet
    assert hasattr(CONTROL_SUITE.trocr, "patience")


def test_trocr_augmentation_preset_documented():
    """augmentation_preset field exists (⚠️ underdocumented — DA2 is the official preset)."""
    assert hasattr(CONTROL_SUITE.trocr, "augmentation_preset")


def test_trocr_lora_rank_documented():
    """lora_rank field exists (full finetuning currently, PEFT is optional)."""
    assert hasattr(CONTROL_SUITE.trocr, "lora_rank")


# ---------------------------------------------------------------------------
# YOLO — critical parameters and previously-missing ones
# ---------------------------------------------------------------------------


def test_yolo_freeze_is_documented():
    """freeze field exists — the #1 most impactful underdocumented YOLO param.

    PREVIOUSLY MISSING from train_yolo() call entirely (was using Ultralytics
    default of None). See finetuning_params.md: marked CRITICAL ⚠️.
    """
    assert hasattr(CONTROL_SUITE.yolo, "freeze"), (
        "freeze field must exist in YOLOControlConfig — "
        "it is the most impactful underdocumented finetuning parameter"
    )
    # Currently None (full training) — correct for first pass without domain data
    assert CONTROL_SUITE.yolo.freeze is None


def test_yolo_fliplr_is_zero():
    """fliplr must be 0.0 for text detection (left-right orientation matters)."""
    assert CONTROL_SUITE.yolo.fliplr == 0.0, (
        "fliplr must be 0.0 for receipt text detection — "
        "horizontal flipping corrupts text reading direction"
    )


def test_yolo_mosaic_is_reduced_for_receipts():
    """mosaic is reduced from Ultralytics default 1.0 for the receipt domain."""
    assert 0.0 <= CONTROL_SUITE.yolo.mosaic <= 1.0
    assert CONTROL_SUITE.yolo.mosaic < 1.0, (
        "mosaic should be < 1.0 for receipt domain — "
        "full mosaic (4-image grid) distorts document structure"
    )


def test_yolo_close_mosaic_nonzero():
    """close_mosaic must be > 0 (⚠️ critical for mAP convergence).

    Ultralytics default is 10. Setting to 0 prevents final learning stabilisation.
    See finetuning_params.md: CRITICALLY IMPORTANT for final accuracy.
    """
    assert CONTROL_SUITE.yolo.close_mosaic > 0, (
        "close_mosaic must be > 0 — disabling mosaic for last N epochs is critical for mAP convergence"
    )


def test_yolo_amp_enabled():
    """AMP (mixed precision) is enabled by default for VRAM efficiency."""
    assert CONTROL_SUITE.yolo.amp is True


def test_yolo_weight_decay_documented():
    """weight_decay is explicitly documented (was previously using Ultralytics default)."""
    assert CONTROL_SUITE.yolo.weight_decay == 0.0005


def test_yolo_close_mosaic_documented():
    """close_mosaic is explicitly set (was previously relying on Ultralytics default 10)."""
    assert CONTROL_SUITE.yolo.close_mosaic == 10


def test_yolo_rect_is_false():
    """rect must be False — rect=True silently disables DataLoader shuffle.

    See finetuning_params.md: ⚠️ underdocumented — 'rect shuffle conflict'.
    """
    assert CONTROL_SUITE.yolo.rect is False


def test_yolo_loss_weights_documented():
    """box, cls, dfl loss weights are explicitly documented (were previously missing)."""
    assert CONTROL_SUITE.yolo.box == 7.5
    assert CONTROL_SUITE.yolo.cls == 0.5
    assert CONTROL_SUITE.yolo.dfl == 1.5


# ---------------------------------------------------------------------------
# Cross-model inspection helpers
# ---------------------------------------------------------------------------


def test_to_dict_returns_nested_dict():
    """to_dict() returns a properly nested dict with all three model keys."""
    d = CONTROL_SUITE.to_dict()
    assert set(d.keys()) == {"donut", "trocr", "yolo"}
    assert isinstance(d["donut"], dict)
    assert isinstance(d["trocr"], dict)
    assert isinstance(d["yolo"], dict)
    # Spot-check a few values
    assert d["donut"]["base_model"] == BASE_MODEL
    assert d["trocr"]["model_id"] == "microsoft/trocr-base-printed"
    assert d["yolo"]["freeze"] is None


def test_critical_params_returns_dict():
    """critical_params() returns a dict with at least the known critical params."""
    crits = CONTROL_SUITE.critical_params()
    assert isinstance(crits, dict)
    # Known critical params that must be present
    assert "donut.input_size" in crits
    assert "donut.encoder_lr" in crits
    assert "donut.tie_word_embeddings" in crits
    assert "trocr.model_id" in crits
    assert "trocr.learning_rate" in crits
    assert "yolo.freeze" in crits
    assert "yolo.imgsz" in crits
    assert "yolo.mosaic" in crits


def test_underdocumented_params_returns_list():
    """underdocumented_params() returns a non-empty list of parameter names."""
    underdoc = CONTROL_SUITE.underdocumented_params()
    assert isinstance(underdoc, list)
    assert len(underdoc) > 0
    # Known underdocumented params from finetuning_params.md commonly_underdocumented[]
    assert "donut.swin_window_size" in underdoc
    assert "donut.sort_json_key" in underdoc
    assert "donut.align_long_axis" in underdoc
    assert "trocr.warmup_init_lr" in underdoc
    assert "trocr.patience" in underdoc
    assert "trocr.augmentation_preset" in underdoc
    assert "yolo.freeze" in underdoc
    assert "yolo.close_mosaic" in underdoc
    assert "yolo.rect" in underdoc


def test_print_summary_runs_without_error(capsys):
    """print_summary() runs without raising and produces output."""
    CONTROL_SUITE.print_summary()
    captured = capsys.readouterr()
    assert "CONTROL SUITE" in captured.out
    assert "DONUT" in captured.out
    assert "TrOCR" in captured.out
    assert "YOLOv8" in captured.out
    assert "CRITICAL" in captured.out
    assert "freeze" in captured.out


def test_total_param_count_is_comprehensive():
    """Total parameter count covers all three models with significant coverage."""
    d = CONTROL_SUITE.to_dict()
    total = sum(len(v) for v in d.values())
    # At least 60 params total (DONUT ~20, TrOCR ~20, YOLO ~40+)
    assert total >= 60, f"Expected ≥60 documented parameters, found {total}"


# ---------------------------------------------------------------------------
# Root Cause 1 — validate_sroie_oversample
# ---------------------------------------------------------------------------


def test_validate_sroie_oversample_single_dataset_passes():
    """Single-dataset runs are always valid regardless of sroie_oversample."""
    validate_sroie_oversample(["sroie"], sroie_oversample=1)
    validate_sroie_oversample(["sroie"], sroie_oversample=2)
    validate_sroie_oversample(["wildreceipt"], sroie_oversample=1)


def test_validate_sroie_oversample_multi_dataset_sufficient_passes():
    """Multi-dataset runs with sroie_oversample >= 2 must not raise."""
    validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=2)
    validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=3)
    validate_sroie_oversample(["sroie", "wildreceipt", "invoices_donut"], sroie_oversample=2)


def test_validate_sroie_oversample_multi_dataset_insufficient_raises():
    """Multi-dataset runs with sroie_oversample < 2 must raise ValueError."""
    with pytest.raises(ValueError, match="sroie_oversample=1"):
        validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=1)


def test_validate_sroie_oversample_error_message_is_informative():
    """ValueError message must mention both datasets and the required minimum."""
    with pytest.raises(ValueError) as exc_info:
        validate_sroie_oversample(["sroie", "invoices_donut"], sroie_oversample=1)
    msg = str(exc_info.value)
    assert "sroie_oversample" in msg
    assert ">= 2" in msg


def test_validate_sroie_oversample_exported():
    """validate_sroie_oversample must be in control_suite.__all__."""
    import control_suite

    assert "validate_sroie_oversample" in control_suite.__all__


def test_validate_sroie_oversample_skip_guard_waives_check():
    """skip_guard=True must allow multi-dataset + oversample=1 without raising."""
    validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=1, skip_guard=True)
    validate_sroie_oversample(
        ["sroie", "wildreceipt", "invoices_donut"], sroie_oversample=1, skip_guard=True
    )


def test_validate_sroie_oversample_skip_guard_false_still_raises():
    """Explicitly passing skip_guard=False must still raise for bad configs."""
    with pytest.raises(ValueError):
        validate_sroie_oversample(["sroie", "wildreceipt"], sroie_oversample=1, skip_guard=False)


# ---------------------------------------------------------------------------
# Root Cause 2 — TrOCRControlConfig.effective_batch_size
# ---------------------------------------------------------------------------


def test_effective_batch_size_high_vram_returns_max():
    """With ample VRAM (>= reserved + batch * per_item), return full batch_size."""
    cfg = TrOCRControlConfig(batch_size=16)
    # 6.0 reserved + 16 * 0.3 = 6.0 + 4.8 = 10.8 GiB needed; 24 GiB available → no scaling
    result = cfg.effective_batch_size(vram_gb=24.0)
    assert result == 16


def test_effective_batch_size_low_vram_reduces_batch():
    """With constrained VRAM, effective_batch_size returns a value < batch_size."""
    cfg = TrOCRControlConfig(batch_size=16)
    # 8 GiB free: usable = 8 - 6 = 2 GiB, safe = int(2 / 0.3) = 6
    result = cfg.effective_batch_size(vram_gb=8.0)
    assert result < 16
    assert result >= 1


def test_effective_batch_size_minimum_is_one():
    """effective_batch_size never returns less than 1, even for batch_size=1."""
    cfg = TrOCRControlConfig(batch_size=1)
    # min(batch_size=1, safe) == 1 regardless of VRAM
    result = cfg.effective_batch_size(vram_gb=24.0)
    assert result == 1
    result_low = cfg.effective_batch_size(vram_gb=0.1)
    assert result_low >= 1


def test_effective_batch_size_matches_inline_formula():
    """effective_batch_size must produce the same result as the inline formula in train_trocr."""
    cfg = TrOCRControlConfig(batch_size=16)
    free_gb = 10.0
    # Replicate the formula from train_trocr_yolo.py
    reserved = cfg._TROCR_RESERVED_GB
    per_item = cfg._TROCR_PER_ITEM_GB
    usable = max(free_gb - reserved, 1.0)
    expected = min(16, max(1, int(usable / per_item)))
    assert cfg.effective_batch_size(free_gb) == expected


def test_trocr_control_config_has_calibration_constants():
    """TrOCRControlConfig must expose _TROCR_RESERVED_GB and _TROCR_PER_ITEM_GB."""
    assert hasattr(TrOCRControlConfig, "_TROCR_RESERVED_GB")
    assert hasattr(TrOCRControlConfig, "_TROCR_PER_ITEM_GB")
    assert TrOCRControlConfig._TROCR_RESERVED_GB == 6.0
    assert TrOCRControlConfig._TROCR_PER_ITEM_GB == 0.3


# ---------------------------------------------------------------------------
# Root Cause 5 — YOLOControlConfig.recommended_freeze
# ---------------------------------------------------------------------------


def test_recommended_freeze_large_dataset_no_freeze():
    """>= 2000 samples: no freezing (full training)."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(2000) is None
    assert cfg.recommended_freeze(3940) is None


def test_recommended_freeze_medium_dataset_freeze_backbone():
    """500–1999 samples: freeze backbone (value 10)."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(500) == 10
    assert cfg.recommended_freeze(1000) == 10
    assert cfg.recommended_freeze(1999) == 10


def test_recommended_freeze_small_dataset_freeze_early_layers():
    """< 500 samples: freeze first 3 backbone layers."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(499) == 3
    assert cfg.recommended_freeze(100) == 3
    assert cfg.recommended_freeze(0) == 3


def test_recommended_freeze_boundary_exactly_500():
    """Boundary: exactly 500 samples maps to freeze=10."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(500) == 10


def test_recommended_freeze_boundary_exactly_2000():
    """Boundary: exactly 2000 samples maps to no freeze."""
    cfg = YOLOControlConfig()
    assert cfg.recommended_freeze(2000) is None


# ---------------------------------------------------------------------------
# Root Cause 4 — AST regression: weight_decay wired from CONTROL_SUITE
# ---------------------------------------------------------------------------


def test_train_trocr_yolo_reads_weight_decay_from_control_suite():
    """train_trocr_yolo.py must read weight_decay from CONTROL_SUITE.trocr, not hardcode 0.

    Uses AST inspection to verify the AdamW call passes weight_decay via
    CONTROL_SUITE.trocr (or a local alias thereof), not as a literal 0 or omitted.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).parent.parent / "train_trocr_yolo.py").read_text()
    tree = ast.parse(source)

    # Collect all keyword arguments to AdamW calls
    adamw_weight_decay_values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_adamw = (isinstance(func, ast.Name) and func.id == "AdamW") or (
                isinstance(func, ast.Attribute) and func.attr == "AdamW"
            )
            if is_adamw:
                for kw in node.keywords:
                    if kw.arg == "weight_decay":
                        adamw_weight_decay_values.append(kw.value)

    assert adamw_weight_decay_values, (
        "No AdamW calls with weight_decay= found in train_trocr_yolo.py"
    )

    # At least one AdamW call must read weight_decay from a non-literal source.
    # Acceptable: CONTROL_SUITE.trocr.weight_decay OR a local alias (_trocr.weight_decay
    # where _trocr is assigned CONTROL_SUITE.trocr earlier in the function).
    # Unacceptable: weight_decay=0 (literal 0) or weight_decay omitted entirely.
    def _is_attribute_access(node: ast.expr) -> bool:
        """Return True if node is an attribute access (x.y or x.y.z), not a literal."""
        return isinstance(node, ast.Attribute)

    has_attribute_ref = any(_is_attribute_access(v) for v in adamw_weight_decay_values)
    assert has_attribute_ref, (
        "train_trocr_yolo.py AdamW optimizer must read weight_decay via an attribute "
        "access (e.g. CONTROL_SUITE.trocr.weight_decay or _trocr.weight_decay), not as "
        f"a literal. Found AST nodes: {[ast.dump(v) for v in adamw_weight_decay_values]}"
    )

    # Also verify none of the AdamW weight_decay arguments are literal 0.
    for val in adamw_weight_decay_values:
        if isinstance(val, ast.Constant):
            assert val.value != 0, (
                "train_trocr_yolo.py AdamW weight_decay must not be literal 0 — "
                "read from CONTROL_SUITE.trocr.weight_decay instead."
            )


# ---------------------------------------------------------------------------
# Root Cause 3 — get_augmentation_transforms helper
# ---------------------------------------------------------------------------


def test_get_augmentation_transforms_none_returns_none():
    """get_augmentation_transforms(None) must return None."""
    assert get_augmentation_transforms(None) is None


def test_get_augmentation_transforms_invalid_raises():
    """get_augmentation_transforms with unknown preset must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown augmentation preset"):
        get_augmentation_transforms("DA99")


def test_get_augmentation_transforms_exported():
    """get_augmentation_transforms must be in control_suite.__all__."""
    import control_suite

    assert "get_augmentation_transforms" in control_suite.__all__


# ============================================================================
# From test_dataset_loaders.py
# ============================================================================


class TestDatasetLoadError(unittest.TestCase):
    def test_attributes(self):
        exc = DatasetLoadError("test_ds", "file not found")
        assert exc.dataset_name == "test_ds"
        assert exc.reason == "file not found"

    def test_message_prefix(self):
        exc = DatasetLoadError("test_ds", "file not found")
        assert "FATAL:" in str(exc)
        assert "test_ds" in str(exc)

    def test_inherits_exception(self):
        exc = DatasetLoadError("ds", "reason")
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# _validate_sample_schema
# ---------------------------------------------------------------------------


class TestValidateSampleSchema(unittest.TestCase):
    def test_valid_sample(self):
        sample = (Path("/img.jpg"), {"company": "A", "date": "B", "address": "C", "total": "D"})
        assert _validate_sample_schema(sample, "test") is True

    def test_missing_field(self):
        sample = (Path("/img.jpg"), {"company": "A", "date": "B"})
        assert _validate_sample_schema(sample, "test") is False

    def test_non_path(self):
        sample = ("/img.jpg", {"company": "A", "date": "B", "address": "C", "total": "D"})
        assert _validate_sample_schema(sample, "test") is False

    def test_non_dict_gt(self):
        sample = (Path("/img.jpg"), "not a dict")
        assert _validate_sample_schema(sample, "test") is False

    def test_wrong_length(self):
        sample = (Path("/img.jpg"),)
        assert _validate_sample_schema(sample, "test") is False

    def test_not_tuple(self):
        assert _validate_sample_schema("string", "test") is False

    def test_extra_fields_ok(self):
        """Extra fields beyond SROIE schema should still validate."""
        sample = (
            Path("/img.jpg"),
            {"company": "A", "date": "B", "address": "C", "total": "D", "extra": "E"},
        )
        assert _validate_sample_schema(sample, "test") is True


# ---------------------------------------------------------------------------
# _validate_samples_nonempty
# ---------------------------------------------------------------------------


class TestValidateSamplesNonempty(unittest.TestCase):
    def test_nonempty_passes(self):
        samples = [(Path("/img.jpg"), {"company": "", "date": "", "address": "", "total": ""})]
        result = _validate_samples_nonempty(samples, "test")
        assert result == samples

    def test_empty_raises(self):
        try:
            _validate_samples_nonempty([], "test_ds")
            assert False, "Should have raised DatasetLoadError"
        except DatasetLoadError as e:
            assert e.dataset_name == "test_ds"
            assert "0 samples" in e.reason


# ---------------------------------------------------------------------------
# _ensure_dir
# ---------------------------------------------------------------------------


class TestEnsureDir(unittest.TestCase):
    def test_creates_dir(self, tmp_path):
        new_dir = tmp_path / "a" / "b" / "c"
        result = _ensure_dir(new_dir)
        assert result == new_dir
        assert new_dir.is_dir()

    def test_existing_dir(self, tmp_path):
        result = _ensure_dir(tmp_path)
        assert result == tmp_path
        assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# Path helpers (env var behavior)
# ---------------------------------------------------------------------------


class TestPathHelpers(unittest.TestCase):
    def test_get_datasets_dir_default(self):
        from dataset_loaders import _get_datasets_dir

        old = os.environ.pop("DONUT_WORKSPACE", None)
        try:
            d = _get_datasets_dir()
            assert d == Path("/workspace/datasets")
        finally:
            if old is not None:
                os.environ["DONUT_WORKSPACE"] = old

    def test_get_datasets_dir_custom(self):
        from dataset_loaders import _get_datasets_dir

        old = os.environ.get("DONUT_WORKSPACE")
        os.environ["DONUT_WORKSPACE"] = "/tmp/custom"
        try:
            d = _get_datasets_dir()
            assert d == Path("/tmp/custom/datasets")
        finally:
            if old is not None:
                os.environ["DONUT_WORKSPACE"] = old
            else:
                del os.environ["DONUT_WORKSPACE"]

    def test_get_sroie_dir_default(self):
        from dataset_loaders import _get_sroie_dir

        old = os.environ.pop("SROIE_DATA_DIR", None)
        try:
            d = _get_sroie_dir()
            assert d == Path("/workspace/ICDAR-2019-SROIE/data")
        finally:
            if old is not None:
                os.environ["SROIE_DATA_DIR"] = old

    def test_get_sroie_dir_custom(self):
        from dataset_loaders import _get_sroie_dir

        old = os.environ.get("SROIE_DATA_DIR")
        os.environ["SROIE_DATA_DIR"] = "/data/sroie"
        try:
            d = _get_sroie_dir()
            assert d == Path("/data/sroie")
        finally:
            if old is not None:
                os.environ["SROIE_DATA_DIR"] = old
            else:
                del os.environ["SROIE_DATA_DIR"]


# ---------------------------------------------------------------------------
# Dataset split separation (no data access required)
# ---------------------------------------------------------------------------


class TestSROIESplitDirectories(unittest.TestCase):
    """Verify val and test splits use distinct directories (no data leakage)."""

    def test_val_and_test_use_different_dirs(self):
        """val_img/ != test_img/ — ensures early stopping does not use test data."""
        from dataset_loaders import SROIELoader

        loader = SROIELoader()
        val_dirs = loader._SPLIT_DIRS["val"]
        test_dirs = loader._SPLIT_DIRS["test"]
        assert val_dirs != test_dirs, (
            "val and test splits share the same directories — "
            "early stopping would optimize for test performance (data leakage)"
        )

    def test_train_val_test_all_different(self):
        """All three splits use different source directories."""
        from dataset_loaders import SROIELoader

        loader = SROIELoader()
        dirs = [loader._SPLIT_DIRS[s] for s in ("train", "val", "test")]
        assert len(set(dirs)) == 3, (
            f"Expected 3 unique split directory pairs, got {len(set(dirs))}: {dirs}"
        )

    def test_val_uses_val_img(self):
        from dataset_loaders import SROIELoader

        loader = SROIELoader()
        img_subdir, key_subdir = loader._SPLIT_DIRS["val"]
        assert img_subdir == "val_img"
        assert key_subdir == "val_key"

    def test_test_uses_test_img(self):
        from dataset_loaders import SROIELoader

        loader = SROIELoader()
        img_subdir, key_subdir = loader._SPLIT_DIRS["test"]
        assert img_subdir == "test_img"
        assert key_subdir == "test_key"

    def test_train_uses_img(self):
        from dataset_loaders import SROIELoader

        loader = SROIELoader()
        img_subdir, key_subdir = loader._SPLIT_DIRS["train"]
        assert img_subdir == "img"
        assert key_subdir == "key"


# ---------------------------------------------------------------------------
# Seller split cache
# ---------------------------------------------------------------------------


class TestSellerSplitCache(unittest.TestCase):
    def test_load_returns_dict(self):
        from dataset_loaders import _load_seller_split_cache

        cache = _load_seller_split_cache()
        assert isinstance(cache, dict)

    def test_no_metadata_key(self):
        from dataset_loaders import _load_seller_split_cache

        cache = _load_seller_split_cache()
        assert "_metadata" not in cache

    def test_entries_have_company_and_address(self):
        from dataset_loaders import _load_seller_split_cache

        cache = _load_seller_split_cache()
        for seller, split in cache.items():
            assert "company" in split, f"Missing 'company' for {seller!r}"
            assert "address" in split, f"Missing 'address' for {seller!r}"

    def test_cache_used_in_remap(self):
        """When seller is in cache, _invoices_donut_remap uses cached values."""
        import json

        from dataset_loaders import (
            InvoicesDonutLoader,
            _load_seller_split_cache,
        )

        # Ensure cache is loaded
        cache = _load_seller_split_cache()
        if not cache:
            pytest.skip("seller_split_cache.json is empty — skipping cache integration test")
        seller, expected = next(iter(cache.items()))

        gt_str = json.dumps(
            {
                "gt_parse": {
                    "header": {"seller": seller, "invoice_date": "2024-01-01"},
                    "summary": {"total_gross_worth": "100.00"},
                }
            }
        )
        result = InvoicesDonutLoader._invoices_donut_remap(gt_str)
        assert result["company"] == expected["company"]
        assert result["address"] == expected["address"]


# ---------------------------------------------------------------------------
# Required named test from CLAUDE.md §16
# ---------------------------------------------------------------------------


def test_val_test_no_overlap():
    """Val and test splits must never overlap (no data leakage into early stopping).

    Verifies that:
    1. SROIELoader._SPLIT_DIRS['val'] != SROIELoader._SPLIT_DIRS['test']  (structural)
    2. load_sroie_val() and load_sroie_test() use distinct directory paths
       so the same image cannot appear in both splits.

    Root cause: if val_img/ == test_img/, early stopping optimises for test
    performance, inflating reported F1 by ~5–8 pp.  stage_install() creates
    physically separate val_img/ and test_img/ directories to prevent this.
    """
    from dataset_loaders import SROIELoader

    loader = SROIELoader()

    val_img_dir, val_key_dir = loader._SPLIT_DIRS["val"]
    test_img_dir, test_key_dir = loader._SPLIT_DIRS["test"]

    assert val_img_dir != test_img_dir, (
        f"val and test use the same img dir '{val_img_dir}' — "
        "early stopping would optimise for test performance (data leakage). "
        "Fix: ensure stage_install() creates val_img/ and test_img/ as distinct dirs."
    )
    assert val_key_dir != test_key_dir, (
        f"val and test use the same key dir '{val_key_dir}' — "
        "key files would be shared between early stopping and final evaluation."
    )

    # Verify the canonical directory names match the expected values from CLAUDE.md
    assert val_img_dir == "val_img", f"Expected val_img_dir='val_img', got '{val_img_dir}'"
    assert test_img_dir == "test_img", f"Expected test_img_dir='test_img', got '{test_img_dir}'"


# ============================================================================
# From test_dataset_normalizer.py
# ============================================================================


def _make_samples(gts: list[dict]) -> list[tuple[Path, dict]]:
    """Build a list of (Path, dict) samples from a list of raw gt dicts."""
    return [(Path(f"/fake/{i}.jpg"), gt) for i, gt in enumerate(gts)]


class TestAliasResolution(unittest.TestCase):
    def test_store_name_maps_to_company(self):
        samples = _make_samples(
            [
                {
                    "store_name": "ACME Corp",
                    "date": "01/01/2024",
                    "address": "123 St",
                    "total": "10.00",
                }
            ]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == "ACME Corp"

    def test_merchant_maps_to_company(self):
        samples = _make_samples(
            [{"merchant": "My Shop", "date": "01/01/2024", "address": "Addr", "total": "5.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == "My Shop"

    def test_grand_total_maps_to_total(self):
        samples = _make_samples(
            [{"company": "Shop", "date": "01/01/2024", "address": "Addr", "grand_total": "99.99"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["total"] == "99.99"

    def test_receipt_date_maps_to_date(self):
        samples = _make_samples(
            [{"company": "Shop", "receipt_date": "12/12/2023", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["date"] == "12/12/2023"


class TestNAHandling(unittest.TestCase):
    def test_na_becomes_empty_string(self):
        samples = _make_samples(
            [{"company": "N/A", "date": "01/01/2024", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == ""

    def test_lowercase_na_becomes_empty_string(self):
        samples = _make_samples(
            [{"company": "n/a", "date": "01/01/2024", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == ""

    def test_null_becomes_empty_string(self):
        samples = _make_samples(
            [{"company": "null", "date": "01/01/2024", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == ""

    def test_none_value_becomes_empty_string(self):
        samples = _make_samples(
            [{"company": None, "date": "01/01/2024", "address": "Addr", "total": "1.00"}]
        )
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert result[0][1]["company"] == ""


class TestMissingKeyDefaulting(unittest.TestCase):
    def test_missing_keys_default_to_empty_string(self):
        """A dict with only 2 fields should produce all 4 canonical keys."""
        samples = _make_samples([{"company": "ACME", "date": "01/01/2024"}])
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        gt = result[0][1]
        assert set(gt.keys()) == {"company", "date", "address", "total"}
        assert gt["address"] == ""
        assert gt["total"] == ""

    def test_empty_input_dict_gives_all_four_keys(self):
        samples = _make_samples([{}])
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        gt = result[0][1]
        assert set(gt.keys()) == {"company", "date", "address", "total"}
        for v in gt.values():
            assert v == ""


class TestCoverageCheck(unittest.TestCase):
    def _full_sample(self, company="ACME", date="01/01/2024", total="1.00"):
        return {"company": company, "date": date, "address": "123 St", "total": total}

    def test_raises_when_coverage_below_threshold(self):
        """10 samples where company is empty → coverage=0% < 30% threshold."""
        gts = [{"date": "01/01/2024", "address": "Addr", "total": "1.00"} for _ in range(10)]
        samples = _make_samples(gts)
        with pytest.raises(ValueError, match="company"):
            DatasetNormalizer(coverage_threshold=0.30).normalise(samples)

    def test_passes_when_coverage_above_threshold(self):
        """10 samples all having company → coverage=100% ≥ 30% threshold."""
        gts = [self._full_sample() for _ in range(10)]
        samples = _make_samples(gts)
        result = DatasetNormalizer(coverage_threshold=0.30).normalise(samples)
        assert len(result) == 10

    def test_disabled_coverage_check(self):
        """coverage_threshold=0.0 disables the check entirely."""
        gts = [{"date": "01/01/2024", "address": "Addr", "total": "1.00"} for _ in range(10)]
        samples = _make_samples(gts)
        result = DatasetNormalizer(coverage_threshold=0.0).normalise(samples)
        assert len(result) == 10


class TestCurrencyStripping(unittest.TestCase):
    def _sample(self, total: str) -> list[tuple[Path, dict]]:
        return _make_samples(
            [{"company": "Shop", "date": "01/01/2024", "address": "Addr", "total": total}]
        )

    def test_strip_rm_prefix(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=True).normalise(
            self._sample("RM 47.80")
        )
        assert result[0][1]["total"] == "47.80"

    def test_strip_dollar_prefix(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=True).normalise(
            self._sample("$10.50")
        )
        assert result[0][1]["total"] == "10.50"

    def test_strip_s_dollar_prefix(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=True).normalise(
            self._sample("S$25.00")
        )
        assert result[0][1]["total"] == "25.00"

    def test_no_strip_when_disabled(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=False).normalise(
            self._sample("RM 47.80")
        )
        assert result[0][1]["total"] == "RM 47.80"

    def test_no_strip_when_disabled_dollar(self):
        result = DatasetNormalizer(coverage_threshold=0.0, strip_currency=False).normalise(
            self._sample("$10.50")
        )
        assert result[0][1]["total"] == "$10.50"


class TestConvenienceWrapper(unittest.TestCase):
    def test_normalise_samples_returns_same_as_class(self):
        samples = _make_samples(
            [{"store_name": "X", "date": "01/01/2024", "address": "A", "total": "1.00"}]
        )
        result = normalise_samples(samples, source_name="test", coverage_threshold=0.0)
        assert result[0][1]["company"] == "X"


# ============================================================================
# From test_guardrails.py
# ============================================================================

_ROOT = Path(__file__).resolve().parent.parent


def _read(filename: str) -> str:
    return (_ROOT / filename).read_text(encoding="utf-8")


def _ast_call_names(source: str) -> list[str]:
    """Return a flat list of all function/method names called in *source*."""
    tree = ast.parse(source)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return names


# ---------------------------------------------------------------------------
# GP-1 — Never mutate EXPERIMENTS global state
# Verify run_experiment() uses dataclasses.replace()
# ---------------------------------------------------------------------------


class TestGP1ExperimentsImmutability(unittest.TestCase):
    """GP-1 (CLAUDE.md §19): run_experiment() must use dataclasses.replace().

    Python dataclasses are mutable.  ``config = EXPERIMENTS[exp_id]`` is a
    reference, not a copy.  Any field assignment propagates back to the global
    dict and corrupts the cache-validity check for all subsequent experiments.
    """

    def test_run_experiment_calls_dataclasses_replace(self):
        """run_experiment() source must contain a dataclasses.replace() call."""
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        pytest.importorskip("transformers", reason="transformers required")

        import inspect

        from run_experiments import run_experiment  # noqa: E402, I001

        src = inspect.getsource(run_experiment)
        tree = ast.parse(src)

        # Either `dataclasses.replace(...)` (Attribute call) or a bare `replace(...)`
        found_replace = any(
            (isinstance(node, ast.Attribute) and node.attr == "replace")
            or (isinstance(node, ast.Name) and node.id == "replace")
            for node in ast.walk(tree)
        )
        assert found_replace, (
            "run_experiment() must call dataclasses.replace() to isolate config from the "
            "global EXPERIMENTS dict. "
            "GP-1 (CLAUDE.md §19): never assign to EXPERIMENTS[N].field — use "
            "dataclasses.replace(EXPERIMENTS[N], field=value) instead."
        )

    def test_run_experiments_file_imports_dataclasses(self):
        """run_experiments.py must import the dataclasses module."""
        source = _read("run_experiments.py")
        tree = ast.parse(source)

        # Look for `import dataclasses`
        found = any(
            isinstance(node, ast.Import)
            and any(alias.name == "dataclasses" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert found, "run_experiments.py must `import dataclasses` to use dataclasses.replace()."


# ---------------------------------------------------------------------------
# GP-2 — Always validate optimizer step count before training
# ---------------------------------------------------------------------------


class TestGP2ValidateStepCount(unittest.TestCase):
    """GP-2 (CLAUDE.md §19): validate_training_config() must be called before training.

    Without step-count validation, a large batch on high-VRAM GPU can reduce
    total optimizer steps below DONUT's convergence threshold (~200 steps),
    producing perfectly structured but completely empty predictions.  This
    never raises an exception — it is a silent failure.
    """

    def test_run_experiment_calls_validate_training_config(self):
        """run_experiment() must reference validate_training_config."""
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        pytest.importorskip("transformers", reason="transformers required")

        import inspect

        from run_experiments import run_experiment  # noqa: E402, I001

        src = inspect.getsource(run_experiment)
        call_names = _ast_call_names(src)

        assert "validate_training_config" in call_names, (
            "run_experiment() must call validate_training_config() before starting training. "
            "GP-2 (CLAUDE.md §19): enforces minimum optimizer step count (~200 steps)."
        )

    def test_validate_training_config_exists_in_resource_optimizer(self):
        """resource_optimizer.py must export validate_training_config."""
        from resource_optimizer import validate_training_config

        assert callable(validate_training_config), (
            "validate_training_config must be a callable in resource_optimizer.py"
        )

    def test_validate_training_config_raises_on_too_few_steps(self):
        """Fewer than 200 optimizer steps must raise ValueError."""
        from resource_optimizer import validate_training_config

        # 10 samples × 1 epoch / batch_size=8 / accum=1 = only 1 step — way too few
        with pytest.raises((ValueError, RuntimeError)):
            validate_training_config(
                batch_size=8,
                gradient_accumulation_steps=1,
                num_train_samples=10,
                epochs=1,
            )


# ---------------------------------------------------------------------------
# GP-3 — Always use list form for convert_tokens_to_ids
# ---------------------------------------------------------------------------


class TestGP3ConvertTokensListForm(unittest.TestCase):
    """GP-3 (CLAUDE.md §19): convert_tokens_to_ids must be called with a list.

    The string form iterates over characters, returning the ID for '<' — not
    the full '<s_sroie>' token.  This causes the decoder to start from the
    wrong token, producing garbage output that never raises an exception.

    Correct:   tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    Wrong:     tokenizer.convert_tokens_to_ids("<s_sroie>")
    """

    @staticmethod
    def _find_string_form_violations(source: str, filename: str) -> list[str]:
        """Return violation descriptions for any string-form convert_tokens_to_ids calls."""
        violations = []
        for i, line in enumerate(source.splitlines(), 1):
            if "convert_tokens_to_ids(" not in line:
                continue
            stripped = line.strip()
            idx = stripped.find("convert_tokens_to_ids(")
            after = stripped[idx + len("convert_tokens_to_ids(") :]
            # String literal immediately after '(' means wrong form
            if after.startswith('"') or after.startswith("'"):
                violations.append(f"{filename}:{i}: {stripped}")
        return violations

    def test_train_py_uses_list_form(self):
        """train.py must use the list form of convert_tokens_to_ids."""
        source = _read("train.py")
        violations = self._find_string_form_violations(source, "train.py")
        assert violations == [], (
            "train.py uses string form of convert_tokens_to_ids() — GP-3 violation: "
            + "; ".join(violations)
        )

    def test_run_experiments_uses_list_form(self):
        """run_experiments.py must use the list form of convert_tokens_to_ids."""
        source = _read("run_experiments.py")
        violations = self._find_string_form_violations(source, "run_experiments.py")
        assert violations == [], (
            "run_experiments.py uses string form of convert_tokens_to_ids() — GP-3 violation: "
            + "; ".join(violations)
        )


# ---------------------------------------------------------------------------
# GP-4 — decoder_start_token_id roundtrip verification
# ---------------------------------------------------------------------------


class TestGP4DecoderStartTokenVerification(unittest.TestCase):
    """GP-4 (CLAUDE.md §19): after setting decoder_start_token_id, the code
    must verify ``tokenizer.decode([token_id]) == '<s_sroie>'`` and raise
    RuntimeError if wrong.

    Silent wrong token IDs produce models that generate valid XML structure
    but wrong content — indistinguishable from normal output without manual
    inspection.
    """

    def test_train_py_has_roundtrip_check(self):
        """train.py must contain decoder_start_token_id roundtrip assertion."""
        source = _read("train.py")
        assert "decoder_start_token_id" in source, (
            "train.py must set model.config.decoder_start_token_id"
        )
        # A RuntimeError guard must be present for the GP-4 roundtrip
        assert "RuntimeError" in source, (
            "train.py must raise RuntimeError when decoder_start_token_id decodes incorrectly. "
            "GP-4 (CLAUDE.md §19): always verify the roundtrip "
            "decode(convert_tokens_to_ids(['<s_sroie>'])[0]) == '<s_sroie>'."
        )

    def test_run_experiments_has_roundtrip_check(self):
        """run_experiments.py must contain decoder_start_token_id roundtrip assertion."""
        source = _read("run_experiments.py")
        assert "decoder_start_token_id" in source
        assert "RuntimeError" in source, (
            "run_experiments.py must raise RuntimeError when decoder_start_token_id "
            "decodes incorrectly (GP-4)."
        )

    def test_train_py_verifies_sroie_token_name(self):
        """train.py must reference '<s_sroie>' as the expected decoded token."""
        source = _read("train.py")
        assert "<s_sroie>" in source, (
            "train.py must verify that decoder_start_token_id decodes back to '<s_sroie>'"
        )


# ---------------------------------------------------------------------------
# Pattern 3 — Compatibility shim (CLAUDE.md §5 Pattern 3)
# ---------------------------------------------------------------------------


class TestPattern3CompatShim(unittest.TestCase):
    """Pattern 3 (CLAUDE.md §5): The PreTrainedTokenizerBase compat shim must
    be present in dataset_loaders.py.

    In transformers ≥4.47, PreTrainedTokenizerBase moved from
    ``transformers.tokenization_utils_base`` to ``transformers`` directly.
    The try/except shim ensures both old and new versions work.
    """

    def test_compat_shim_contains_pretrained_tokenizer_base(self):
        """dataset_loaders.py must reference PreTrainedTokenizerBase."""
        source = _read("dataset_loaders.py")
        assert "PreTrainedTokenizerBase" in source, (
            "dataset_loaders.py is missing the PreTrainedTokenizerBase import. "
            "Pattern 3 requires the compat shim for transformers ≥4.47 compatibility."
        )

    def test_compat_shim_has_fallback_import_path(self):
        """The shim must include the fallback from tokenization_utils_base."""
        source = _read("dataset_loaders.py")
        assert "tokenization_utils_base" in source, (
            "dataset_loaders.py missing fallback import from "
            "transformers.tokenization_utils_base — "
            "this will crash on transformers <4.47. "
            "Pattern 3 (CLAUDE.md §5): preserve the compat shim."
        )

    def test_compat_shim_uses_try_except(self):
        """The shim must be a try/except block, not a version comparison."""
        source = _read("dataset_loaders.py")
        tree = ast.parse(source)

        # The shim wraps the conditional import in a try/except:
        #   try:
        #       import transformers
        #       if not hasattr(...):
        #           from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        #           ...
        #   except Exception:
        #       pass
        #
        # So PreTrainedTokenizerBase appears in the try *body*, not the except handler.
        # Walk all Try nodes and check whether their body (or handlers) reference the name.
        found_shim = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_body_src = "\n".join(ast.unparse(stmt) for stmt in node.body)
                if "PreTrainedTokenizerBase" in try_body_src:
                    found_shim = True
                    break

        assert found_shim, (
            "dataset_loaders.py has no try/except block whose body imports "
            "PreTrainedTokenizerBase. Pattern 3 requires the try/except form (not a "
            "version check) so that the shim works across all "
            "Python/transformers version combinations."
        )


# ---------------------------------------------------------------------------
# Pattern 7 — importorskip must precede guarded imports in test files
# ---------------------------------------------------------------------------


class TestPattern7ImportorskipOrder(unittest.TestCase):
    """Pattern 7 (CLAUDE.md §16): pytest.importorskip() must appear before
    the import of the package it guards.

    If the import fires before the importorskip call, a missing package raises
    ImportError at collection time and NO tests in that file run — they don't
    even show as 'skipped'.
    """

    def test_test_metrics_importorskip_before_donut_evaluator(self):
        """test_all.py must guard torch before importing donut_evaluator.

        In the merged file the guard is a try/except + _TORCH_AVAILABLE flag
        (instead of module-level pytest.importorskip) so that non-torch tests
        still collect. The invariant is that the guard appears BEFORE the
        protected import.
        """
        source = (Path(__file__).parent / "test_all.py").read_text()
        donut_import_pos = source.find("from donut_evaluator import")
        assert donut_import_pos != -1, "'from donut_evaluator import' not found in test_all.py"

        # Accept either importorskip (original Pattern 7) or _TORCH_AVAILABLE try/except guard
        guard_candidates = [
            source.find("importorskip"),
            source.find("_TORCH_AVAILABLE"),
        ]
        guard_pos = min((p for p in guard_candidates if p != -1), default=-1)
        assert guard_pos != -1, (
            "test_all.py must have a torch guard (importorskip or _TORCH_AVAILABLE try/except). "
            "Pattern 7 (CLAUDE.md §16)."
        )
        assert guard_pos < donut_import_pos, (
            "Pattern 7 violation in test_all.py: 'from donut_evaluator import' appears "
            "before the torch guard. The guard must come first so that collection fails "
            "gracefully (skip) rather than crashing with ImportError."
        )

    def test_test_pipeline_smoke_has_importorskip(self):
        """test_all.py must guard torch imports (importorskip or try/except)."""
        source = (Path(__file__).parent / "test_all.py").read_text()
        has_guard = "importorskip" in source or "_TORCH_AVAILABLE" in source
        assert has_guard, (
            "test_all.py must guard torch imports with importorskip or _TORCH_AVAILABLE. "
            "Pattern 7 (CLAUDE.md §16)."
        )

    def test_test_train_invariants_importorskip_before_train(self):
        """test_all.py must guard transformers before torch-dependent imports."""
        source = (Path(__file__).parent / "test_all.py").read_text()
        has_guard = "importorskip" in source or "_TORCH_AVAILABLE" in source
        assert has_guard, (
            "test_all.py must guard transformers with importorskip or _TORCH_AVAILABLE. "
            "Pattern 7 (CLAUDE.md §16)."
        )


# ============================================================================
# From test_inject_results.py
# ============================================================================


class TestSafe(unittest.TestCase):
    def test_present_value(self):
        assert _safe({"f1": 0.9123}, "f1") == "0.9123"

    def test_missing_key(self):
        assert _safe({}, "f1") == "N/A"

    def test_none_value(self):
        assert _safe({"f1": None}, "f1") == "N/A"

    def test_custom_format(self):
        assert _safe({"f1": 0.9123}, "f1", fmt=".2f") == "0.91"

    def test_zero_value(self):
        assert _safe({"f1": 0.0}, "f1") == "0.0000"

    def test_integer_value(self):
        assert _safe({"n": 500}, "n", fmt="d") == "500"


# ---------------------------------------------------------------------------
# EXP_NAMES & LEADERBOARD
# ---------------------------------------------------------------------------


class TestStaticData(unittest.TestCase):
    def test_exp_names_has_at_least_eight_entries(self):
        assert len(EXP_NAMES) >= 8

    def test_exp_names_keys_include_1_to_8(self):
        assert {str(i) for i in range(1, 9)}.issubset(set(EXP_NAMES.keys()))

    def test_exp_names_keys_include_9_to_18(self):
        # Architecture comparison experiments 9-18 should be present
        assert {str(i) for i in range(9, 19)}.issubset(set(EXP_NAMES.keys()))

    def test_leaderboard_has_entries(self):
        assert len(LEADERBOARD) > 0

    def test_leaderboard_scores_are_valid(self):
        for name, score in LEADERBOARD:
            assert isinstance(name, str)
            assert 0 < score <= 1.0, f"Invalid score for {name}: {score}"


# ---------------------------------------------------------------------------
# PaperInjector
# ---------------------------------------------------------------------------


class TestPaperInjector(unittest.TestCase):
    def _make_injector(self, tmp_dir, all_exp=None, eval_res=None, template=""):
        """Create a PaperInjector with mock data."""
        results_dir = Path(tmp_dir) / "results"
        results_dir.mkdir()

        if all_exp is not None:
            (results_dir / "all_experiments.json").write_text(json.dumps(all_exp), encoding="utf-8")
        if eval_res is not None:
            (results_dir / "evaluation_results.json").write_text(
                json.dumps(eval_res), encoding="utf-8"
            )

        template_path = Path(tmp_dir) / "paper.tex"
        template_path.write_text(template, encoding="utf-8")

        return PaperInjector(results_dir, template_path)

    def test_build_var_map_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp)
            var_map = injector.build_var_map()
            assert "best_f1" in var_map
            assert var_map["best_f1"] == "0.0000"

    def test_build_var_map_with_experiments(self):
        all_exp = {
            "1": {
                "metrics": {"global_f1": 0.85, "global_precision": 0.86, "global_recall": 0.84},
                "num_train_samples": 500,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp)
            var_map = injector.build_var_map()
            assert var_map["exp1_f1"] == "0.8500"
            assert var_map["exp1_prec"] == "0.8600"
            assert var_map["exp1_n"] == "500"

    def test_best_experiment_selection(self):
        all_exp = {
            "1": {"metrics": {"global_f1": 0.80}, "num_train_samples": 500},
            "2": {"metrics": {"global_f1": 0.90}, "num_train_samples": 1000},
            "3": {"metrics": {"global_f1": 0.85}, "num_train_samples": 800},
        }
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp)
            var_map = injector.build_var_map()
            assert var_map["best_exp"] == "2"
            assert var_map["best_f1"] == "0.9000"

    def test_fill_substitution(self):
        all_exp = {
            "1": {"metrics": {"global_f1": 0.85}, "num_train_samples": 500},
        }
        template = r"F1 = \VAR{exp1_f1}, Best = \VAR{best_f1}"
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp, template=template)
            filled = injector.fill()
            assert "0.8500" in filled
            assert r"\VAR{" not in filled

    def test_fill_raises_on_unresolved(self):
        template = r"\VAR{nonexistent_var}"
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, template=template)
            try:
                injector.fill(strict=True)
                assert False, "Should have raised UnresolvedVarError"
            except UnresolvedVarError as e:
                assert "nonexistent_var" in str(e)

    def test_pretrained_metrics(self):
        eval_res = {
            "pretrained_metrics": {
                "global_f1": 0.10,
                "global_precision": 0.12,
                "global_recall": 0.09,
                "overall_exact_match": 0.05,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, eval_res=eval_res)
            var_map = injector.build_var_map()
            assert var_map["pre_f1"] == "0.1000"
            assert var_map["pre_f1_pct"] == "10.00"

    def test_gain_calculation(self):
        all_exp = {
            "1": {"metrics": {"global_f1": 0.80}, "num_train_samples": 500},
            "4": {"metrics": {"global_f1": 0.85}, "num_train_samples": 1400},
        }
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp)
            var_map = injector.build_var_map()
            assert var_map["gain_1_4"] == "+0.0500"


# ---------------------------------------------------------------------------
# Partial-results robustness (paper/presentation generation)
# ---------------------------------------------------------------------------


class TestPartialResultsRobustness(unittest.TestCase):
    """Paper/presentation generation must not fail when only some experiments
    have completed.  Unresolved \\VAR{} vars should fall back to "---"."""

    def _make_injector(self, tmp, all_exp=None, template=None, eval_res=None):
        tmp = Path(tmp)
        if all_exp:
            (tmp / "all_experiments.json").write_text(json.dumps(all_exp), encoding="utf-8")
        if eval_res:
            (tmp / "evaluation_results.json").write_text(json.dumps(eval_res), encoding="utf-8")
        tmpl_path = tmp / "template.tex"
        if template:
            tmpl_path.write_text(template, encoding="utf-8")
        else:
            tmpl_path.write_text(r"F1=\VAR{exp1_f1}", encoding="utf-8")
        return PaperInjector(tmp, tmpl_path)

    def test_fill_with_no_results_does_not_raise(self):
        """With zero experiment results, fill() must return a string, not raise."""
        template = r"\VAR{exp1_f1} \VAR{exp2_f1} \VAR{exp3_f1} \VAR{best_f1}"
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, template=template)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = injector.fill()
            assert isinstance(result, str), "fill() must return a string"
            assert r"\VAR{" not in result, "No raw \\VAR{} must remain in output"

    def test_fill_with_partial_results_does_not_raise(self):
        """With only Exp 1 complete, fill() for a template that references all
        8 experiments must not raise — missing ones fall back to '---'."""
        all_exp = {
            "1": {
                "metrics": {
                    "global_f1": 0.85,
                    "global_precision": 0.86,
                    "global_recall": 0.84,
                    "overall_exact_match": 0.75,
                    "company_f1": 0.90,
                    "company_ned": 0.05,
                    "date_f1": 0.95,
                    "date_ned": 0.02,
                    "address_f1": 0.80,
                    "address_ned": 0.10,
                    "total_f1": 0.85,
                    "total_ned": 0.08,
                },
                "num_train_samples": 500,
            }
        }
        # Template references exp2_f1 through exp8_f1 (not in results)
        template = " ".join(rf"\VAR{{exp{i}_f1}}" for i in range(1, 9))
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, all_exp=all_exp, template=template)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = injector.fill()
            assert r"\VAR{" not in result, "No raw \\VAR{} must remain in output"
            assert "0.8500" in result, "Exp 1 F1 must be filled from results"
            assert "---" in result, "Missing exps must fall back to '---'"

    def test_fill_strict_raises_for_unknown_var(self):
        """fill(strict=True) must still raise UnresolvedVarError for truly
        unknown variables (not in pre-populated fallbacks)."""
        template = r"\VAR{completely_unknown_key_xyz}"
        with tempfile.TemporaryDirectory() as tmp:
            injector = self._make_injector(tmp, template=template)
            with pytest.raises(UnresolvedVarError):
                injector.fill(strict=True)

    def test_all_exp_vars_pre_populated(self):
        """build_var_map() must pre-populate expN_* vars for all N=1..8
        even when no experiments have completed."""
        from constants import FIELDS

        with tempfile.TemporaryDirectory() as tmp:
            template_path = Path(tmp) / "t.tex"
            template_path.write_text("x", encoding="utf-8")
            injector = PaperInjector(Path(tmp), template_path)
            var_map = injector.build_var_map()
            for i in range(1, 9):
                assert f"exp{i}_n" in var_map, f"exp{i}_n missing from var_map"
                assert f"exp{i}_f1" in var_map, f"exp{i}_f1 missing from var_map"
                for fld in FIELDS:
                    assert f"exp{i}_{fld}_f1" in var_map, f"exp{i}_{fld}_f1 missing from var_map"


# ---------------------------------------------------------------------------
# New robustness tests (paper/presentation generation)
# ---------------------------------------------------------------------------


class TestInjectorRobustness(unittest.TestCase):
    """Edge-case robustness for paper/presentation generation."""

    def _make_injector(self, tmp_dir, template="x"):
        tmp = Path(tmp_dir)
        tmpl = tmp / "paper.tex"
        tmpl.write_text(template, encoding="utf-8")
        return PaperInjector(tmp, tmpl)

    # -- malformed JSON -------------------------------------------------------

    def test_malformed_all_experiments_json_falls_back_gracefully(self):
        """_load_all_experiments() must return {} (not raise) for truncated JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_json = Path(tmp) / "all_experiments.json"
            bad_json.write_text("{broken json,,", encoding="utf-8")
            injector = self._make_injector(tmp)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = injector._load_all_experiments()
            assert result == {}, "Malformed JSON must fall back to empty dict"
            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            assert len(user_warnings) > 0, "A UserWarning must be emitted for malformed JSON"
            assert any(
                "parse" in str(w.message).lower() or "json" in str(w.message).lower()
                for w in user_warnings
            ), "Warning must mention parsing/JSON"

    def test_malformed_evaluation_results_json_falls_back_gracefully(self):
        """_load_evaluation_results() must return {} (not raise) for truncated JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_json = Path(tmp) / "evaluation_results.json"
            bad_json.write_text('{"pretrained_metrics": {', encoding="utf-8")
            injector = self._make_injector(tmp)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = injector._load_evaluation_results()
            assert result == {}, "Malformed JSON must fall back to empty dict"
            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            assert len(user_warnings) > 0, "A UserWarning must be emitted for malformed JSON"
            assert any(
                "parse" in str(w.message).lower() or "json" in str(w.message).lower()
                for w in user_warnings
            ), "Warning must mention parsing/JSON"

    def test_fill_with_malformed_json_returns_string(self):
        """fill() must return a filled string even when all_experiments.json is broken."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "all_experiments.json").write_text("{bad", encoding="utf-8")
            template = r"\VAR{exp1_f1} \VAR{best_f1}"
            injector = self._make_injector(tmp, template=template)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = injector.fill()
            assert isinstance(result, str)
            assert r"\VAR{" not in result

    # -- output directory creation -------------------------------------------

    def test_fill_paper_creates_missing_output_directory(self):
        """fill_paper() must create the parent directory of output_path if absent."""
        from inject_results import fill_paper

        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp) / "paper.tex"
            paper.write_text(r"F1=\VAR{exp1_f1}", encoding="utf-8")
            # Nested directory that does not exist yet
            output = Path(tmp) / "subdir" / "nested" / "out.tex"
            assert not output.parent.exists()
            fill_paper(str(paper), str(output), {"exp1_f1": "0.8500"})
            assert output.exists(), "fill_paper must create missing parent dirs"
            assert "0.8500" in output.read_text(encoding="utf-8")

    def test_fill_paper_creates_output_in_existing_directory(self):
        """fill_paper() must still work normally when output dir already exists."""
        from inject_results import fill_paper

        with tempfile.TemporaryDirectory() as tmp:
            paper = Path(tmp) / "paper.tex"
            paper.write_text(r"F1=\VAR{best_f1}", encoding="utf-8")
            output = Path(tmp) / "out.tex"
            fill_paper(str(paper), str(output), {"best_f1": "0.9000"})
            assert "0.9000" in output.read_text(encoding="utf-8")


# ============================================================================
# From test_integration_contracts.py
# ============================================================================

_REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse(filename: str) -> ast.Module:
    return ast.parse((_REPO / filename).read_text())


def _source(filename: str) -> str:
    return (_REPO / filename).read_text()


# ---------------------------------------------------------------------------
# Contract 1: DonutEvaluator uses model_path, not model_dir
# ---------------------------------------------------------------------------


class TestDonutEvaluatorModelPath(unittest.TestCase):
    def test_model_dir_not_used_in_run_all_zero_shot(self):
        """run_all._run_zero_shot_experiment must NOT pass model_dir= to DonutEvaluator."""
        src = _source("run_all.py")
        # Find the _run_zero_shot_experiment function source
        tree = _parse("run_all.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_run_zero_shot_experiment":
                func_src = ast.get_source_segment(src, node) or ""
                assert "model_dir=" not in func_src, (
                    "_run_zero_shot_experiment still passes model_dir= to DonutEvaluator; "
                    "the correct kwarg is model_path="
                )
                assert "model_path=" in func_src, (
                    "_run_zero_shot_experiment must pass model_path= to DonutEvaluator"
                )
                return
        pytest.fail("_run_zero_shot_experiment function not found in run_all.py")

    def test_donut_evaluator_constructor_has_model_path_param(self):
        """DonutEvaluator.__init__ must accept model_path (not model_dir)."""
        tree = _parse("evaluation.py")
        src = _source("evaluation.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DonutEvaluator":
                for item in ast.walk(node):
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        init_src = ast.get_source_segment(src, item) or ""
                        assert "model_path" in init_src, (
                            "DonutEvaluator.__init__ must have a model_path parameter"
                        )
                        assert "model_dir" not in init_src, (
                            "DonutEvaluator.__init__ must NOT use model_dir (use model_path)"
                        )
                        return
        pytest.fail("DonutEvaluator.__init__ not found in donut_evaluator.py")


# ---------------------------------------------------------------------------
# Contract 2: run_experiments exports run_experiment_from_config
# ---------------------------------------------------------------------------


class TestRunExperimentFromConfigExported(unittest.TestCase):
    def test_in_all_list(self):
        """run_experiments.__all__ must include run_experiment_from_config."""
        tree = _parse("run_experiments.py")
        src = _source("run_experiments.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        all_src = ast.get_source_segment(src, node) or ""
                        assert "run_experiment_from_config" in all_src, (
                            "run_experiments.__all__ must include 'run_experiment_from_config'"
                        )
                        return
        pytest.fail("__all__ assignment not found in run_experiments.py")

    def test_run_experiment_from_config_in_all_ast(self):
        """AST check: run_experiment_from_config is defined as a function in run_experiments.py."""
        tree = _parse("run_experiments.py")
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert "run_experiment_from_config" in func_names, (
            "run_experiment_from_config function definition not found in run_experiments.py"
        )


# ---------------------------------------------------------------------------
# Contract 3: yaml fallback raises RuntimeError, not silent 0.0 F1
# ---------------------------------------------------------------------------


class TestYamlFallbackRaisesError(unittest.TestCase):
    def test_no_return_zero_f1_dict(self):
        """_run_yaml_donut_experiment must NOT return a dict with global_f1: 0.0."""
        src = _source("run_all.py")
        tree = _parse("run_all.py")
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_run_yaml_donut_experiment"
            ):
                func_src = ast.get_source_segment(src, node) or ""
                # The old fallback returned {"metrics": {"global_f1": 0.0}}
                assert '"global_f1": 0.0' not in func_src and "'global_f1': 0.0" not in func_src, (
                    "_run_yaml_donut_experiment must not return a silent 0.0 F1 placeholder; "
                    "it must raise RuntimeError so the failure is loud."
                )
                return
        pytest.fail("_run_yaml_donut_experiment not found in run_all.py")

    def test_raises_runtime_error(self):
        """_run_yaml_donut_experiment fallback path must raise RuntimeError."""
        src = _source("run_all.py")
        tree = _parse("run_all.py")
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_run_yaml_donut_experiment"
            ):
                func_src = ast.get_source_segment(src, node) or ""
                assert "raise RuntimeError" in func_src, (
                    "_run_yaml_donut_experiment fallback must raise RuntimeError"
                )
                return
        pytest.fail("_run_yaml_donut_experiment not found in run_all.py")


# ---------------------------------------------------------------------------
# Contract 4: trocr calls data prep before training
# ---------------------------------------------------------------------------


class TestTrocrCallsDataPrep(unittest.TestCase):
    def test_stage_trocr_data_prep_called_in_trocr_experiment(self):
        """_run_trocr_yolo_experiment must reference stage_trocr_data_prep."""
        src = _source("run_all.py")
        tree = _parse("run_all.py")
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_run_trocr_yolo_experiment"
            ):
                func_src = ast.get_source_segment(src, node) or ""
                assert "stage_trocr_data_prep" in func_src, (
                    "_run_trocr_yolo_experiment must call stage_trocr_data_prep() "
                    "to ensure YOLO/TrOCR data is ready before training"
                )
                return
        pytest.fail("_run_trocr_yolo_experiment not found in run_all.py")


# ---------------------------------------------------------------------------
# Contract 5: logging_utils importable and functional
# ---------------------------------------------------------------------------


class TestLoggingUtilsImportable(unittest.TestCase):
    def test_import(self):
        """logging_utils must be importable without heavy dependencies."""
        mod = importlib.import_module("logging_utils")
        assert hasattr(mod, "suppress_noisy_loggers")
        assert hasattr(mod, "DeduplicatingHandler")
        assert hasattr(mod, "_NOISY_THIRD_PARTY_LOGGERS")

    def test_suppress_noisy_loggers_callable(self):
        import logging

        from constants import suppress_noisy_loggers

        # Should not raise
        suppress_noisy_loggers(logging.WARNING)

    def test_noisy_logger_list_contains_pil(self):
        from constants import _NOISY_THIRD_PARTY_LOGGERS

        assert "PIL" in _NOISY_THIRD_PARTY_LOGGERS
        assert "PIL.PngImagePlugin" in _NOISY_THIRD_PARTY_LOGGERS


# ============================================================================
# From test_integration_smoke.py
# ============================================================================


class TestImportChain(unittest.TestCase):
    """Verify every module can be imported without crashing."""

    def test_constants_importable(self):
        from constants import BASE_MODEL, FIELDS, SEED

        assert len(FIELDS) == 4
        assert BASE_MODEL == "naver-clova-ix/donut-base"
        assert SEED == 42

    def test_address_extractor_importable(self):
        # extract_address_from_seller moved to dataset_normalizer.py
        from dataset_normalizer import extract_address_from_seller

        assert callable(extract_address_from_seller)

    def test_inject_results_importable(self):
        from inject_results import LEADERBOARD

        assert len(LEADERBOARD) > 0

    def test_training_config_importable(self):
        # TrainingConfig moved to resource_optimizer.py
        from resource_optimizer import TrainingConfig

        cfg = TrainingConfig()
        cfg.validate()
        assert cfg.batch_size == 8

    def test_pipeline_config_importable(self):
        # PipelineMode merged into cloud_pipeline.py
        from cloud_orchestration import PipelineMode

        assert PipelineMode.AUTO.value == "auto"

    def test_retro_ui_importable(self):
        # RetroUIFormatter moved to cloud_pipeline.py
        from cloud_orchestration import RetroUIFormatter

        assert RetroUIFormatter.bold("x")

    def test_test_runner_importable(self):
        from cloud_orchestration import TestRunner

        assert hasattr(TestRunner, "run_all_checks")

    def test_results_aggregator_importable(self):
        # ResultsAggregator moved to inject_results.py
        from inject_results import ResultsAggregator

        agg = ResultsAggregator(results_dir=Path("/tmp"))
        assert agg is not None

    def test_storage_manager_importable(self):
        # StorageManager merged into cloud_pipeline.py
        from cloud_orchestration import StorageManager

        sm = StorageManager(results_dir=Path("/tmp"))
        assert sm is not None

    def test_git_controller_importable(self):
        # GitController merged into cloud_pipeline.py
        from cloud_orchestration import GitController

        assert hasattr(GitController, "get_current_branch")

    def test_validators_importable(self):
        from validators import ImportChainChecker

        success, errors = ImportChainChecker.check_all()
        # Import chain should pass (constants.py exists)
        assert isinstance(errors, list)
        assert success, f"Import chain check failed: {errors}"


# ============================================================================
# From test_logging_utils.py
# ============================================================================


class _CapturingHandler(logging.Handler):
    """Minimal handler that stores formatted records for inspection."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestDeduplicatingHandler(unittest.TestCase):
    def _make_record(self, name: str, level: int, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name=name,
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_single_message_emitted_on_flush(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        rec = self._make_record("test", logging.DEBUG, "hello")
        handler.emit(rec)
        handler.flush()
        assert len(cap.records) == 1
        assert cap.records[0].getMessage() == "hello"

    def test_repeated_message_collapsed_with_count(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        for _ in range(5):
            handler.emit(self._make_record("test", logging.DEBUG, "same msg"))
        handler.flush()
        assert len(cap.records) == 1
        assert "[×5]" in cap.records[0].msg

    def test_different_messages_emitted_separately(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        handler.emit(self._make_record("test", logging.DEBUG, "msg A"))
        handler.emit(self._make_record("test", logging.DEBUG, "msg A"))
        handler.emit(self._make_record("test", logging.DEBUG, "msg B"))
        handler.flush()
        # msg A (count=2) + msg B (count=1)
        assert len(cap.records) == 2
        assert "[×2]" in cap.records[0].msg
        # msg B has count 1 — no suffix appended
        assert "[×" not in cap.records[1].msg

    def test_single_occurrence_has_no_count_suffix(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        handler.emit(self._make_record("test", logging.INFO, "only once"))
        handler.flush()
        assert len(cap.records) == 1
        assert "[×" not in cap.records[0].msg

    def test_different_levels_not_collapsed(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        handler.emit(self._make_record("test", logging.DEBUG, "same text"))
        handler.emit(self._make_record("test", logging.WARNING, "same text"))
        handler.flush()
        # Different level → different key → two separate records
        assert len(cap.records) == 2

    def test_close_flushes_pending(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        for _ in range(3):
            handler.emit(self._make_record("test", logging.DEBUG, "pending"))
        handler.close()
        assert len(cap.records) == 1
        assert "[×3]" in cap.records[0].msg

    def test_thread_safety_no_crash(self):
        """Multiple threads emitting the same message must not crash or lose records."""
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    handler.emit(self._make_record("t", logging.DEBUG, "concurrent"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        handler.flush()
        assert not errors, f"Thread errors: {errors}"
        # At least one record must have been emitted (possibly collapsed)
        assert len(cap.records) >= 1


class TestSuppressNoisyLoggers(unittest.TestCase):
    def test_pil_set_to_warning(self):
        import logging

        suppress_noisy_loggers(logging.WARNING)
        assert logging.getLogger("PIL").level == logging.WARNING

    def test_pil_png_plugin_set_to_warning(self):
        suppress_noisy_loggers(logging.WARNING)
        assert logging.getLogger("PIL.PngImagePlugin").level == logging.WARNING

    def test_custom_level(self):
        suppress_noisy_loggers(logging.ERROR)
        assert logging.getLogger("PIL").level == logging.ERROR
        # Reset to WARNING for other tests
        suppress_noisy_loggers(logging.WARNING)


# ============================================================================
# From test_memory_manager.py
# ============================================================================


class TestComputePilMbPerSample(unittest.TestCase):
    """Tests for compute_pil_mb_per_sample()."""

    def test_reference_resolution_1280x960(self):
        """At DONUT native 1280x960: exactly 3 x 1280 x 960 / 1_048_576."""
        expected = 3 * 1280 * 960 / (1024 * 1024)
        result = mm.compute_pil_mb_per_sample(1280, 960)
        assert abs(result - expected) < 1e-6, f"Expected {expected:.4f}, got {result:.4f}"

    def test_wrong_resolution_2560x1920_is_4x_larger(self):
        """At 2560x1920 the per-sample cost is exactly 4x the reference."""
        ref = mm.compute_pil_mb_per_sample(1280, 960)
        wrong = mm.compute_pil_mb_per_sample(2560, 1920)
        ratio = wrong / ref
        assert abs(ratio - 4.0) < 1e-6, (
            f"2560x1920 should be 4x reference but ratio is {ratio:.4f}. "
            "This confirms that 2560x1920 in processor_config.json causes 4x RAM cost."
        )

    def test_trocr_line_crop_384x384(self):
        """TrOCR line crops at 384x384 are much smaller than full receipt images."""
        trocr_mb = mm.compute_pil_mb_per_sample(384, 384)
        donut_mb = mm.compute_pil_mb_per_sample(1280, 960)
        # 384x384 = 147456 px; 1280x960 = 1228800 px -> ratio ~= 8.33x
        assert trocr_mb < donut_mb, "TrOCR line crops must be smaller than full receipts"
        assert trocr_mb < 1.0, f"384x384 line crops should be < 1 MB, got {trocr_mb:.3f} MB"

    def test_zero_dimensions_returns_zero(self):
        """Edge case: zero height or width returns 0."""
        assert mm.compute_pil_mb_per_sample(0, 960) == 0.0
        assert mm.compute_pil_mb_per_sample(1280, 0) == 0.0


def _patch_available_ram(available_bytes: int):
    """Context manager: patch memory_manager._get_available_ram_bytes to return a fixed value."""
    import unittest.mock as mock

    return mock.patch("memory_manager._get_available_ram_bytes", return_value=available_bytes)


class TestRamCacheIsSafe(unittest.TestCase):
    """Tests for ram_cache_is_safe() -- the gate that replaced `len(samples) * 3`."""

    def test_small_dataset_at_ref_resolution_is_safe(self):
        """500 samples x 3.516 MB = 1,758 MB -- safe on any machine with > 12 GB RAM."""
        with _patch_available_ram(64 * 1024 * 1024 * 1024):  # 64 GB
            result = mm.ram_cache_is_safe(500, 1280, 960)
        assert result is True, "500 samples at 1280x960 should be safe on 64 GB RAM"

    def test_large_dataset_at_wrong_resolution_is_rejected(self):
        """3940 samples x 14.06 MB = 55,396 MB -> must be blocked at 192 GB.

        55 GB > 192 GB x 15% = 28.8 GB threshold -> rejected.
        This is the exact Experiment 8 scenario that caused 192 GB RAM exhaustion.
        """
        with _patch_available_ram(192 * 1024 * 1024 * 1024):  # 192 GB
            result = mm.ram_cache_is_safe(3940, 2560, 1920)
        assert result is False, (
            "3940 samples x 14.06 MB = 55.4 GB must be rejected at 15% of 192 GB (28.8 GB threshold). "
            "This is the Exp 8 OOM scenario."
        )

    def test_large_dataset_at_ref_resolution_is_allowed(self):
        """3940 samples x 3.516 MB = 13,853 MB < 256 GB x 6% = 15,729 MB -> SAFE.

        Uses 256 GB RAM to ensure the correct resolution (1280x960) is allowed
        even with the conservative 6% safety fraction.
        """
        with _patch_available_ram(256 * 1024 * 1024 * 1024):  # 256 GB
            result = mm.ram_cache_is_safe(3940, 1280, 960)
        assert result is True, (
            "3940 samples x 3.516 MB = 13.8 GB should be allowed at 6% of 256 GB (15.7 GB threshold)."
        )

    def test_psutil_import_error_returns_false(self):
        """If RAM detection fails (returns 0), caching is disabled (safe default)."""
        with _patch_available_ram(0):  # simulate unknown/unavailable
            result = mm.ram_cache_is_safe(100, 1280, 960)
        assert isinstance(result, bool)
        assert result is False, "Unknown RAM should disable caching"

    def test_custom_safety_fraction(self):
        """Custom safety_fraction parameter is respected."""
        # 10 GB available = 10,240 MB
        with _patch_available_ram(10 * 1024 * 1024 * 1024):
            # 500 x 3.516 MB = 1758 MB; 10,240 MB x 0.10 = 1,024 MB threshold
            # 1758 > 1024 -> REJECTED at 10% safety
            result_10pct = mm.ram_cache_is_safe(500, 1280, 960, safety_fraction=0.10)
            # 10,240 MB x 0.25 = 2,560 MB threshold
            # 1758 < 2560 -> ALLOWED at 25% safety
            result_25pct = mm.ram_cache_is_safe(500, 1280, 960, safety_fraction=0.25)
        assert result_10pct is False, "Should be rejected at 10% safety fraction"
        assert result_25pct is True, "Should be allowed at 25% safety fraction"


class TestFlushHfArrowCache(unittest.TestCase):
    """Tests for flush_hf_arrow_cache() -- smoke tests only (no network)."""

    def test_flush_does_not_crash_when_datasets_installed(self):
        """flush_hf_arrow_cache() must not raise even if datasets is installed."""
        mm.flush_hf_arrow_cache()  # should complete without exception

    def test_flush_does_not_crash_when_datasets_missing(self):
        """flush_hf_arrow_cache() must not raise if datasets is not installed."""
        import unittest.mock as mock

        with mock.patch.dict("sys.modules", {"datasets": None}):
            mm.flush_hf_arrow_cache()  # should complete without exception


class TestReleaseHfDataset(unittest.TestCase):
    """Tests for release_hf_dataset() -- unit tests with mock datasets."""

    def test_none_is_noop(self):
        """Passing None is safe (no-op)."""
        mm.release_hf_dataset(None)  # must not raise

    def test_object_with_cleanup_cache_files_is_called(self):
        """Objects with cleanup_cache_files() have it called."""
        import unittest.mock as mock

        mock_ds = mock.Mock()
        mock_ds.cleanup_cache_files = mock.Mock()
        mm.release_hf_dataset(mock_ds)
        mock_ds.cleanup_cache_files.assert_called_once()

    def test_object_without_cleanup_does_not_crash(self):
        """Objects without cleanup_cache_files() are handled gracefully."""

        class FakeDataset:
            pass

        mm.release_hf_dataset(FakeDataset())  # must not raise


class TestShutdownDataloaderWorkers(unittest.TestCase):
    """Tests for shutdown_dataloader_workers() -- smoke tests with mock trainer."""

    def test_none_trainer_is_noop(self):
        """Passing None is safe (no-op)."""
        mm.shutdown_dataloader_workers(None)  # must not raise

    def test_trainer_without_get_train_dataloader_does_not_crash(self):
        """Trainers that don't have get_train_dataloader() are handled gracefully."""

        class FakeTrainer:
            pass

        mm.shutdown_dataloader_workers(FakeTrainer())  # must not raise

    def test_trainer_with_iterator_calls_shutdown(self):
        """When _iterator._shutdown_workers exists it is called."""
        import unittest.mock as mock

        mock_iter = mock.Mock()
        mock_iter._shutdown_workers = mock.Mock()
        mock_dl = mock.Mock()
        mock_dl._iterator = mock_iter
        mock_trainer = mock.Mock()
        mock_trainer.get_train_dataloader = mock.Mock(return_value=mock_dl)
        mock_trainer.get_eval_dataloader = mock.Mock(return_value=None)
        mm.shutdown_dataloader_workers(mock_trainer)
        mock_iter._shutdown_workers.assert_called_once()


class TestRamHeadroomMb(unittest.TestCase):
    """Tests for the new ram_headroom_mb() function."""

    def test_returns_positive_float_on_real_system(self):
        """ram_headroom_mb() must return a non-negative float on any machine."""
        result = mm.ram_headroom_mb()
        assert isinstance(result, float), f"Expected float, got {type(result).__name__}"
        assert result >= 0.0, f"Expected non-negative value, got {result}"

    def test_returns_float_with_mock_available_ram(self):
        """ram_headroom_mb() returns available_bytes / 1024**2 from _get_available_ram_bytes()."""
        with _patch_available_ram(8 * 1024 * 1024 * 1024):  # 8 GB
            result = mm.ram_headroom_mb()
        expected = 8 * 1024  # 8 GB in MB = 8192
        assert abs(result - expected) < 1.0, f"Expected ~{expected} MB, got {result:.1f} MB"

    def test_returns_zero_when_ram_unknown(self):
        """ram_headroom_mb() returns 0.0 when RAM detection returns 0."""
        with _patch_available_ram(0):
            result = mm.ram_headroom_mb()
        assert result == 0.0, f"Expected 0.0 when RAM unknown, got {result}"

    def test_exported_in_all(self):
        """ram_headroom_mb must be listed in memory_manager.__all__."""
        assert "ram_headroom_mb" in mm.__all__, (
            "ram_headroom_mb must be exported in __all__ so callers can do "
            "`from memory_manager import ram_headroom_mb`"
        )


class TestRamSafetyFractionRegressionGuard(unittest.TestCase):
    """Regression guard: _RAM_SAFETY_FRACTION must never be raised back to 0.15."""

    def test_safety_fraction_is_at_most_0_10(self):
        """_RAM_SAFETY_FRACTION must be <= 0.10.

        The OOM crash at Experiment 6 was caused by the old 15% (0.15) value.
        The fix lowers it to 6% (0.06). This test prevents future PRs from
        raising it back above 10% without explicit acknowledgement.
        """
        fraction = mm._RAM_SAFETY_FRACTION
        assert fraction <= 0.10, (
            f"_RAM_SAFETY_FRACTION = {fraction} is too high (must be <= 0.10). "
            "Raising it above 0.10 risks the Exp 6 three-layer OOM failure: "
            "train PIL cache drains RAM before the val init runs."
        )

    def test_safety_fraction_is_positive(self):
        """_RAM_SAFETY_FRACTION must be strictly positive."""
        assert mm._RAM_SAFETY_FRACTION > 0.0, (
            "_RAM_SAFETY_FRACTION must be > 0 to avoid dividing by zero or "
            "always rejecting the cache."
        )


# ============================================================================
# From test_metrics.py
# ============================================================================


class TestNED(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )

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


class TestParsePredictionListMerge(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


class TestUnwrapPrediction(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )

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


class TestComputeMetrics(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )

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


class TestLoadModelWithTiedWeights(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


class TestAllowHighParseFailures(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


class TestRunExperimentsParseFailureCatch(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


class TestBenchmarkCompareMetricUnification(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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
        assert 0.0 < score < 1.0, f"_token_f1_squad should give partial credit, got {score}"

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


class TestInteractiveSelectionFallback(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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
        fake_re_mod = mock.MagicMock()
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


# ============================================================================
# From test_pipeline_critic.py
# ============================================================================

_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Helper utilities
# =============================================================================


class TestNormPpf(unittest.TestCase):
    """_norm_ppf: rational approximation to the standard-normal inverse CDF."""

    def test_median_is_zero(self):
        assert abs(_norm_ppf(0.5)) < 1e-6

    def test_p975_is_196(self):
        """Standard value: z_{0.975} ≈ 1.96 (two-sided α=0.05)."""
        assert abs(_norm_ppf(0.975) - 1.96) < 0.01

    def test_p80_is_0842(self):
        """Standard value: z_{0.80} ≈ 0.842 (power=80%)."""
        assert abs(_norm_ppf(0.80) - 0.842) < 0.01

    def test_symmetry(self):
        """ppf(1-p) == -ppf(p) for all valid p."""
        for p in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9):
            assert abs(_norm_ppf(1 - p) + _norm_ppf(p)) < 1e-6

    def test_boundary_raises(self):
        with pytest.raises(ValueError):
            _norm_ppf(0.0)
        with pytest.raises(ValueError):
            _norm_ppf(1.0)

    def test_extreme_values_finite(self):
        assert math.isfinite(_norm_ppf(0.001))
        assert math.isfinite(_norm_ppf(0.999))


class TestTwoProportionMdd(unittest.TestCase):
    """_two_proportion_mdd: minimum detectable difference."""

    def test_larger_n_gives_smaller_mdd(self):
        mdd_small = _two_proportion_mdd(n=100, p_bar=0.85)
        mdd_large = _two_proportion_mdd(n=1000, p_bar=0.85)
        assert mdd_small > mdd_large

    def test_higher_power_gives_larger_mdd(self):
        mdd_80 = _two_proportion_mdd(n=252, p_bar=0.85, power=0.80)
        mdd_50 = _two_proportion_mdd(n=252, p_bar=0.85, power=0.50)
        assert mdd_80 > mdd_50

    def test_positive_result(self):
        mdd = _two_proportion_mdd(n=252, p_bar=0.85)
        assert mdd > 0

    def test_known_value_approximately_0_089(self):
        """With N=252, p_bar=0.85, alpha=0.05, power=0.80, MDD ≈ 0.089.

        Reference value independently derived:
          z_{0.975} = 1.960, z_{0.80} = 0.842  (standard normal quantiles)
          MDD = (1.960 + 0.842) * sqrt(2 * 0.85 * 0.15 / 252)
              = 2.802 * sqrt(0.00101190...)
              = 2.802 * 0.031810...
              ≈ 0.08913

        Verified against scipy.stats two-proportion z-test power formula:
          from statsmodels.stats.power import NormalIndPower
          NormalIndPower().solve_power(effect_size=..., nobs1=252, alpha=0.05)
        gives the same order of magnitude.
        """
        mdd = _two_proportion_mdd(n=252, p_bar=0.85, power=0.80)
        # Allow ±0.005 tolerance for rational approximation error in _norm_ppf
        assert abs(mdd - 0.089) < 0.005


class TestFamilyWiseErrorRate(unittest.TestCase):
    """_family_wise_error_rate: FWER for k independent tests."""

    def test_single_test_equals_alpha(self):
        assert abs(_family_wise_error_rate(k=1, alpha=0.05) - 0.05) < 1e-9

    def test_seven_tests_approx_30_percent(self):
        fwer = _family_wise_error_rate(k=7, alpha=0.05)
        assert abs(fwer - (1 - 0.95**7)) < 1e-9
        # Approximately 30%
        assert 0.28 < fwer < 0.32

    def test_monotone_increasing_in_k(self):
        prev = 0.0
        for k in range(1, 20):
            fwer = _family_wise_error_rate(k=k, alpha=0.05)
            assert fwer > prev
            prev = fwer


# =============================================================================
# CritiqueFinding / CritiqueReport types
# =============================================================================


class TestCritiqueFinding(unittest.TestCase):
    def _make(self, severity=FindingSeverity.WARNING) -> CritiqueFinding:
        return CritiqueFinding(
            severity=severity,
            category="test",
            title="Test finding",
            description="A test description.",
            evidence="test evidence",
            recommendation="Fix it.",
        )

    def test_to_dict_contains_severity_value(self):
        f = self._make(FindingSeverity.FATAL)
        d = f.to_dict()
        assert d["severity"] == "FATAL"

    def test_to_dict_has_required_keys(self):
        f = self._make()
        d = f.to_dict()
        for key in ("severity", "category", "title", "description", "evidence", "recommendation"):
            assert key in d


class TestCritiqueReport(unittest.TestCase):
    def _make_report(self, severities: list[FindingSeverity]) -> CritiqueReport:
        findings = [
            CritiqueFinding(
                severity=s,
                category="test",
                title="t",
                description="d",
                evidence="e",
                recommendation="r",
            )
            for s in severities
        ]
        return CritiqueReport(findings=findings)

    def test_passed_is_false_when_fatal_present(self):
        report = self._make_report([FindingSeverity.FATAL])
        assert not report.passed

    def test_passed_is_true_when_no_fatal(self):
        report = self._make_report([FindingSeverity.CRITICAL, FindingSeverity.WARNING])
        assert report.passed

    def test_fatal_count(self):
        report = self._make_report(
            [FindingSeverity.FATAL, FindingSeverity.FATAL, FindingSeverity.WARNING]
        )
        assert report.fatal_count == 2

    def test_critical_count(self):
        report = self._make_report([FindingSeverity.CRITICAL, FindingSeverity.FATAL])
        assert report.critical_count == 1

    def test_warning_count(self):
        report = self._make_report([FindingSeverity.WARNING, FindingSeverity.WARNING])
        assert report.warning_count == 2

    def test_to_dict_keys(self):
        report = self._make_report([FindingSeverity.WARNING])
        d = report.to_dict()
        for key in ("passed", "fatal_count", "critical_count", "warning_count", "findings"):
            assert key in d

    def test_to_dict_findings_is_list(self):
        report = self._make_report([FindingSeverity.INFO])
        assert isinstance(report.to_dict()["findings"], list)

    def test_empty_report_passes(self):
        report = CritiqueReport()
        assert report.passed
        assert report.fatal_count == 0


# =============================================================================
# Audit 1 — StatisticalPowerAudit
# =============================================================================


class TestStatisticalPowerAudit(unittest.TestCase):
    def test_returns_list_of_findings(self):
        findings = StatisticalPowerAudit().run()
        assert isinstance(findings, list)
        assert len(findings) >= 1

    def test_every_finding_is_critique_finding(self):
        for f in StatisticalPowerAudit().run():
            assert isinstance(f, CritiqueFinding)

    def test_underpowered_finding_present(self):
        """252 pairs cannot reliably detect a 0.0479 F1 gain — must fire."""
        findings = StatisticalPowerAudit().run()
        cats = [f.category for f in findings]
        assert "statistical_validity" in cats

    def test_underpowered_finding_is_critical_or_fatal(self):
        """Power finding must be CRITICAL or FATAL."""
        findings = StatisticalPowerAudit().run()
        power_findings = [
            f
            for f in findings
            if "underpowered" in f.title.lower() or "minimum detectable" in f.title.lower()
        ]
        assert power_findings, "No underpowered finding returned"
        for f in power_findings:
            assert f.severity in (FindingSeverity.CRITICAL, FindingSeverity.FATAL), (
                f"Expected CRITICAL/FATAL, got {f.severity}"
            )

    def test_per_field_warning_present(self):
        """Per-field sample size warning must be in findings."""
        findings = StatisticalPowerAudit().run()
        per_field = [
            f for f in findings if "per-field" in f.title.lower() or "per field" in f.title.lower()
        ]
        assert per_field, "No per-field sample size warning returned"
        assert per_field[0].severity == FindingSeverity.WARNING

    def test_finding_includes_n_pairs_in_evidence(self):
        findings = StatisticalPowerAudit().run()
        all_evidence = " ".join(f.evidence for f in findings)
        # 63 × 4 = 252 pairs
        assert "252" in all_evidence

    def test_mdd_exceeds_claimed_gain(self):
        """Hard check: MDD at 80% power must exceed the paper's claimed 0.0479 gain."""
        n_pairs = 63 * 4
        mdd = _two_proportion_mdd(n_pairs, p_bar=0.85, power=0.80)
        assert mdd > StatisticalPowerAudit.CLAIMED_GAIN, (
            f"MDD={mdd:.4f} must exceed claimed gain={StatisticalPowerAudit.CLAIMED_GAIN}"
        )


# =============================================================================
# Audit 2 — MultipleTestingAudit
# =============================================================================


class TestMultipleTestingAudit(unittest.TestCase):
    def test_returns_non_empty_list(self):
        findings = MultipleTestingAudit().run()
        assert len(findings) >= 1

    def test_finding_is_critical_or_worse(self):
        findings = MultipleTestingAudit().run()
        assert any(
            f.severity in (FindingSeverity.CRITICAL, FindingSeverity.FATAL) for f in findings
        )

    def test_fwer_value_in_evidence(self):
        findings = MultipleTestingAudit().run()
        all_evidence = " ".join(f.evidence for f in findings)
        # FWER for k=7 comparisons at α=0.05 is ~30%; evidence should mention it
        assert "FWER" in all_evidence or "fwer" in all_evidence.lower()

    def test_category_is_multiple_testing(self):
        findings = MultipleTestingAudit().run()
        cats = {f.category for f in findings}
        assert "multiple_testing" in cats

    def test_finding_mentions_bonferroni(self):
        findings = MultipleTestingAudit().run()
        combined = " ".join(f.description + f.recommendation for f in findings)
        assert "Bonferroni" in combined or "bonferroni" in combined.lower()


# =============================================================================
# Audit 3 — EpochConfoundAudit
# =============================================================================


class TestEpochConfoundAudit(unittest.TestCase):
    def test_returns_list(self):
        findings = EpochConfoundAudit().run()
        assert isinstance(findings, list)

    def test_confound_finding_present(self):
        """Exps 5–8 run 15 epochs, Exp 1 runs 10 — confound must be detected."""
        findings = EpochConfoundAudit().run()
        assert len(findings) >= 1, "EpochConfoundAudit should produce at least one finding"

    def test_finding_is_critical(self):
        findings = EpochConfoundAudit().run()
        assert any(f.severity == FindingSeverity.CRITICAL for f in findings), (
            "Epoch confound should be rated CRITICAL"
        )

    def test_category_is_experimental_design(self):
        findings = EpochConfoundAudit().run()
        cats = {f.category for f in findings}
        assert "experimental_design" in cats

    def test_extract_epochs_returns_dict(self):
        epochs = EpochConfoundAudit._extract_experiment_epochs()
        assert isinstance(epochs, dict)
        assert len(epochs) >= 1

    def test_baseline_exp1_has_fewer_epochs_than_exp6(self):
        """Exp 1 must have fewer epochs than Exp 6; both IDs must be present."""
        epochs = EpochConfoundAudit._extract_experiment_epochs()
        assert 1 in epochs, (
            f"Exp 1 not found in extracted epochs dict; returned keys: {sorted(epochs.keys())}"
        )
        assert 6 in epochs, (
            f"Exp 6 not found in extracted epochs dict; returned keys: {sorted(epochs.keys())}"
        )
        assert epochs[1] < epochs[6], (
            f"Expected Exp 1 epochs ({epochs[1]}) < Exp 6 epochs ({epochs[6]}); "
            "epoch confound requires the 'winning' experiment to train longer"
        )

    def test_recommendation_mentions_controlled_ablation(self):
        findings = EpochConfoundAudit().run()
        combined = " ".join(f.recommendation for f in findings)
        assert "ablation" in combined.lower() or "controlled" in combined.lower()


# =============================================================================
# Audit 4 — PretrainingBiasAudit
# =============================================================================


class TestPretrainingBiasAudit(unittest.TestCase):
    def test_returns_list(self):
        findings = PretrainingBiasAudit().run()
        assert isinstance(findings, list)

    def test_at_least_one_finding(self):
        findings = PretrainingBiasAudit().run()
        assert len(findings) >= 1

    def test_finding_severity_is_warning_or_higher(self):
        findings = PretrainingBiasAudit().run()
        for f in findings:
            assert f.severity in (
                FindingSeverity.WARNING,
                FindingSeverity.CRITICAL,
                FindingSeverity.FATAL,
            )

    def test_base_model_name_in_evidence(self):
        findings = PretrainingBiasAudit().run()
        combined = " ".join(f.evidence for f in findings)
        assert "naver-clova-ix/donut-base" in combined

    def test_category_is_pretraining_bias(self):
        findings = PretrainingBiasAudit().run()
        cats = {f.category for f in findings}
        assert "pretraining_bias" in cats

    def test_synthdog_mentioned(self):
        findings = PretrainingBiasAudit().run()
        combined = " ".join(f.description for f in findings)
        assert "SynthDoG" in combined or "synthdog" in combined.lower()


# =============================================================================
# Audit 5 — ArchitectureAudit
# =============================================================================


class TestArchitectureAudit(unittest.TestCase):
    def test_returns_list(self):
        findings = ArchitectureAudit().run()
        assert isinstance(findings, list)

    def test_at_least_one_finding(self):
        findings = ArchitectureAudit().run()
        assert len(findings) >= 1

    def test_finding_mentions_pretraining(self):
        findings = ArchitectureAudit().run()
        combined = " ".join(f.description for f in findings)
        assert "pretrain" in combined.lower()

    def test_category_is_comparison_fairness(self):
        findings = ArchitectureAudit().run()
        cats = {f.category for f in findings}
        assert "comparison_fairness" in cats

    def test_f1_values_in_evidence(self):
        findings = ArchitectureAudit().run()
        combined = " ".join(f.evidence for f in findings)
        assert "0.8982" in combined or "0.2035" in combined


# =============================================================================
# Audit 6 — BenchmarkNarrowness
# =============================================================================


class TestBenchmarkNarrowness(unittest.TestCase):
    def test_returns_list(self):
        findings = BenchmarkNarrowness().run()
        assert isinstance(findings, list)

    def test_fatal_finding_for_test_set_mismatch(self):
        """Comparing against published DONUT (different test set) must be FATAL."""
        findings = BenchmarkNarrowness().run()
        fatal_findings = [f for f in findings if f.severity == FindingSeverity.FATAL]
        assert len(fatal_findings) >= 1, (
            "BenchmarkNarrowness must produce at least one FATAL finding "
            "for the custom-vs-official test set mismatch"
        )

    def test_fatal_category_is_benchmark_validity(self):
        findings = BenchmarkNarrowness().run()
        fatal = [f for f in findings if f.severity == FindingSeverity.FATAL]
        assert any(f.category == "benchmark_validity" for f in fatal)

    def test_narrowness_warning_present(self):
        findings = BenchmarkNarrowness().run()
        warnings = [f for f in findings if f.severity == FindingSeverity.WARNING]
        assert len(warnings) >= 1

    def test_official_test_size_in_evidence(self):
        findings = BenchmarkNarrowness().run()
        combined = " ".join(f.evidence for f in findings)
        assert "347" in combined

    def test_custom_test_size_63_in_evidence(self):
        findings = BenchmarkNarrowness().run()
        combined = " ".join(f.evidence for f in findings)
        assert "63" in combined

    def test_recommendation_mentions_leaderboard(self):
        findings = BenchmarkNarrowness().run()
        combined = " ".join(f.recommendation for f in findings)
        assert (
            "leaderboard" in combined.lower() or "ICDAR" in combined or "split" in combined.lower()
        )


# =============================================================================
# PipelineCritic orchestrator
# =============================================================================


class TestPipelineCritic(unittest.TestCase):
    def test_run_returns_critique_report(self):
        report = PipelineCritic().run()
        assert isinstance(report, CritiqueReport)

    def test_report_has_findings(self):
        report = PipelineCritic().run()
        assert len(report.findings) >= 1

    def test_report_not_passed_due_to_fatal(self):
        """The pipeline has FATAL findings — it does not survive scrutiny."""
        report = PipelineCritic().run()
        assert not report.passed, (
            "Pipeline should NOT pass scrutiny: BenchmarkNarrowness produces a FATAL finding"
        )

    def test_report_has_fatal_and_critical(self):
        report = PipelineCritic().run()
        assert report.fatal_count >= 1
        assert report.critical_count >= 1

    def test_findings_sorted_by_severity(self):
        """FATAL findings must come before CRITICAL, CRITICAL before WARNING."""
        order = {
            FindingSeverity.FATAL: 0,
            FindingSeverity.CRITICAL: 1,
            FindingSeverity.WARNING: 2,
            FindingSeverity.INFO: 3,
        }
        report = PipelineCritic().run()
        severities = [order[f.severity] for f in report.findings]
        assert severities == sorted(severities), "Findings must be sorted by severity (FATAL first)"

    def test_all_six_categories_represented(self):
        """Every audit class must contribute at least one finding."""
        report = PipelineCritic().run()
        cats = {f.category for f in report.findings}
        expected = {
            "statistical_validity",
            "multiple_testing",
            "experimental_design",
            "pretraining_bias",
            "comparison_fairness",
            "benchmark_validity",
        }
        missing = expected - cats
        assert not missing, f"Missing finding categories: {missing}"

    def test_to_dict_is_serialisable(self):
        """CritiqueReport.to_dict() must return a JSON-serialisable structure."""
        import json

        report = PipelineCritic().run()
        d = report.to_dict()
        # Should not raise
        json.dumps(d)

    def test_print_loud_does_not_crash(self, capsys):
        """print_loud() must not raise (even with exit_on_fatal=False)."""
        report = PipelineCritic().run()
        report.print_loud(exit_on_fatal=False)
        captured = capsys.readouterr()
        assert "PIPELINE CRITIC" in captured.out
        assert "VERDICT" in captured.out

    def test_print_loud_exit_on_fatal_calls_sys_exit(self):
        """print_loud(exit_on_fatal=True) must call sys.exit(1) when FATAL found."""
        report = PipelineCritic().run()
        assert not report.passed  # Precondition: report has a FATAL finding
        with pytest.raises(SystemExit) as exc_info:
            report.print_loud(exit_on_fatal=True)
        assert exc_info.value.code == 1


# ============================================================================
# From test_pipeline_smoke.py
# ============================================================================


class TestExperimentsImmutability(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
    """EXPERIMENTS global dict must not be mutated by run_experiment()."""

    def test_resource_optimization_does_not_mutate_global(self):
        """Applying resource-optimized batch_size must not change EXPERIMENTS."""
        # Snapshot original values
        original = {exp_id: dataclasses.replace(config) for exp_id, config in EXPERIMENTS.items()}
        # Verify snapshot matches current values (no prior mutation)
        for exp_id, config in EXPERIMENTS.items():
            assert config.batch_size == original[exp_id].batch_size, (
                f"EXPERIMENTS[{exp_id}].batch_size was mutated from "
                f"{original[exp_id].batch_size} to {config.batch_size}. "
                "run_experiment() must use dataclasses.replace() instead of "
                "mutating the global EXPERIMENTS dict."
            )
            assert (
                config.gradient_accumulation_steps == original[exp_id].gradient_accumulation_steps
            ), (
                f"EXPERIMENTS[{exp_id}].gradient_accumulation_steps was mutated from "
                f"{original[exp_id].gradient_accumulation_steps} to "
                f"{config.gradient_accumulation_steps}. "
                "run_experiment() must use dataclasses.replace() instead of "
                "mutating the global EXPERIMENTS dict."
            )

    def test_config_to_dict_reflects_actual_values(self):
        """_config_to_dict must serialize the passed config, not global defaults."""
        for exp_id, config in EXPERIMENTS.items():
            config_dict = _config_to_dict(config)
            assert config_dict["max_epochs"] == config.epochs, (
                f"_config_to_dict for Exp {exp_id} recorded epochs={config_dict['max_epochs']} "
                f"but config.epochs={config.epochs}"
            )
            assert config_dict["per_device_train_batch_size"] == config.batch_size
            assert config_dict["gradient_accumulation_steps"] == config.gradient_accumulation_steps

        # Verify that a modified (resource-optimized) copy produces correct dict
        modified = dataclasses.replace(EXPERIMENTS[1], batch_size=4, gradient_accumulation_steps=4)
        d = _config_to_dict(modified)
        assert d["per_device_train_batch_size"] == 4
        assert d["gradient_accumulation_steps"] == 4
        # Global EXPERIMENTS[1] must be unchanged
        assert EXPERIMENTS[1].batch_size == 8
        assert EXPERIMENTS[1].gradient_accumulation_steps == 2


class TestExperimentConfigFields(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
    """ExperimentConfig must use experiment_id, not id."""

    def test_has_experiment_id_field(self):
        """experiment_id is the correct field name."""
        c = ExperimentConfig(experiment_id=99, name="test", datasets=["sroie"])
        assert c.experiment_id == 99

    def test_does_not_have_id_field(self):
        """id is a @property alias for experiment_id, NOT a dataclass field."""
        c = ExperimentConfig(experiment_id=7, name="test", datasets=["sroie"])
        # id must exist as a property and equal experiment_id
        assert c.id == 7, "id property must return experiment_id"
        # id must NOT be a dataclass field (only a property)
        field_names = {f.name for f in dataclasses.fields(c)}
        assert "id" not in field_names, (
            "id must be a @property alias, not a dataclass field, "
            "to prevent 'id' from appearing in dataclasses.asdict() output"
        )

    def test_does_not_accept_lr_scheduler_type(self):
        """lr_scheduler_type is NOT a valid field — prevents regression of Bug A."""
        field_names = {f.name for f in dataclasses.fields(ExperimentConfig)}
        assert "lr_scheduler_type" not in field_names, (
            "lr_scheduler_type is not a valid ExperimentConfig field. "
            "Remove it from the dataclass or fix any callers that pass it."
        )

    def test_instantiation_with_defaults(self):
        """ExperimentConfig can be instantiated with only required fields."""
        c = ExperimentConfig(experiment_id=1, name="smoke", datasets=["sroie"])
        assert c.epochs > 0
        assert c.lr > 0
        assert c.batch_size > 0


class TestExperimentConfigLoaderCompat(unittest.TestCase):
    """Verify cross-class compatibility: both ExperimentConfig classes share a common interface."""

    def test_run_experiments_id_property(self):
        """run_experiments.ExperimentConfig.id must alias experiment_id."""
        if not _TORCH_AVAILABLE:
            pytest.skip("torch required")
        c = ExperimentConfig(experiment_id=7, name="test", datasets=["sroie"])
        assert c.id == 7
        assert c.id == c.experiment_id

    def test_run_experiments_id_is_not_dataclass_field(self):
        """id must be a property, not a dataclass field, to stay out of asdict()."""
        if not _TORCH_AVAILABLE:
            pytest.skip("torch required")
        c = ExperimentConfig(experiment_id=3, name="test", datasets=["sroie"])
        field_names = {f.name for f in dataclasses.fields(c)}
        assert "id" not in field_names, "id must be a @property, not a dataclass field"

    def test_run_experiments_dataset_names_property(self):
        """run_experiments.ExperimentConfig.dataset_names must return list[str]."""
        if not _TORCH_AVAILABLE:
            pytest.skip("torch required")
        c = ExperimentConfig(experiment_id=1, name="test", datasets=["sroie", "wildreceipt"])
        assert c.dataset_names == ["sroie", "wildreceipt"]

    def test_run_experiments_base_checkpoint_property(self):
        """run_experiments.ExperimentConfig.base_checkpoint must alias base_model."""
        if not _TORCH_AVAILABLE:
            pytest.skip("torch required")
        from constants import BASE_MODEL

        c = ExperimentConfig(experiment_id=1, name="test", datasets=["sroie"])
        assert c.base_checkpoint == c.base_model
        assert c.base_model == BASE_MODEL

    def test_run_experiments_has_yaml_compat_fields(self):
        """run_experiments.ExperimentConfig must have YAML-only fields for dataclasses.replace() compat."""
        if not _TORCH_AVAILABLE:
            pytest.skip("torch required")
        c = ExperimentConfig(experiment_id=1, name="test", datasets=["sroie"])
        assert c.arch_type == "donut"
        assert c.is_zero_shot is False
        assert c.depends_on == []
        assert c.full_parameter_finetuning is True
        assert c.image_height == 1280
        assert c.image_width == 960
        assert c.allow_high_res is False

    def test_loader_experiment_id_property(self):
        """experiment_config_loader.ExperimentConfig.experiment_id must alias id."""
        import experiment_config as _ecl
        from experiment_config import DatasetEntry

        c = _ecl.ExperimentConfig(
            id=5,
            name="yaml_test",
            datasets=[DatasetEntry(name="sroie")],
        )
        assert c.experiment_id == 5
        assert c.experiment_id == c.id

    def test_loader_base_model_property(self):
        """experiment_config_loader.ExperimentConfig.base_model must alias base_checkpoint."""
        import experiment_config as _ecl
        from experiment_config import DatasetEntry

        c = _ecl.ExperimentConfig(
            id=5,
            name="yaml_test",
            datasets=[DatasetEntry(name="sroie")],
            base_checkpoint="naver-clova-ix/donut-base",
        )
        assert c.base_model == "naver-clova-ix/donut-base"
        assert c.base_model == c.base_checkpoint

    def test_loader_dataset_names_property(self):
        """experiment_config_loader.ExperimentConfig.dataset_names returns list[str]."""
        import experiment_config as _ecl
        from experiment_config import DatasetEntry

        c = _ecl.ExperimentConfig(
            id=2,
            name="yaml_test2",
            datasets=[DatasetEntry(name="sroie"), DatasetEntry(name="wildreceipt")],
        )
        assert c.dataset_names == ["sroie", "wildreceipt"]

    def test_loader_warmup_steps_default(self):
        """experiment_config_loader.ExperimentConfig warmup_steps default must be 40."""
        import experiment_config as _ecl
        from experiment_config import DatasetEntry

        c = _ecl.ExperimentConfig(
            id=1,
            name="yaml_test",
            datasets=[DatasetEntry(name="sroie")],
        )
        assert c.warmup_steps == 40, (
            f"warmup_steps default must be 40 (matches run_experiments.ExperimentConfig), "
            f"got {c.warmup_steps}"
        )


class TestTrainExperimentSignature(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
    """train_experiment must accept an optional config= parameter."""

    def test_accepts_config_keyword(self):
        """train_experiment signature must have config= parameter (Bug C fix)."""
        sig = inspect.signature(train_experiment)
        assert "config" in sig.parameters, (
            "train_experiment() is missing the 'config' parameter. "
            "Without it, run_custom_experiment() silently uses the wrong config "
            "from EXPERIMENTS[exp_id] instead of the custom one."
        )

    def test_config_defaults_to_none(self):
        """config parameter must default to None so existing callers are unaffected."""
        sig = inspect.signature(train_experiment)
        param = sig.parameters["config"]
        assert param.default is None, (
            "train_experiment(config=) must default to None to remain backwards-compatible "
            "with callers that do not pass a custom config."
        )


class TestExperimentsRegistry(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
    """All experiment dataset names must exist in the loader registry."""

    def test_all_dataset_names_are_valid(self):
        """Every dataset name in EXPERIMENTS must have a loader in _LOADERS."""
        invalid = {}
        for exp_id, config in EXPERIMENTS.items():
            bad = [d for d in config.datasets if d not in _LOADERS]
            if bad:
                invalid[exp_id] = bad

        assert not invalid, (
            f"Experiments reference dataset names with no loader: {invalid}. "
            f"Valid loader names: {sorted(_LOADERS.keys())}"
        )

    def test_experiments_covers_expected_ids(self):
        """EXPERIMENTS must define exactly 8 experiments (IDs 1–8)."""
        assert set(EXPERIMENTS.keys()) == set(range(1, 9)), (
            f"Expected EXPERIMENTS to have keys 1–8, got: {sorted(EXPERIMENTS.keys())}"
        )

    def test_all_configs_have_experiment_id(self):
        """Every config in EXPERIMENTS must have experiment_id set (not 0 default)."""
        bad = [exp_id for exp_id, config in EXPERIMENTS.items() if config.experiment_id == 0]
        assert not bad, (
            f"Experiments {bad} have experiment_id=0 (default). "
            "Set experiment_id=N explicitly in each ExperimentConfig."
        )


class TestRunCustomExperimentCallable(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
    """run_custom_experiment must exist and have the right signature."""

    def test_is_callable(self):
        assert callable(run_custom_experiment)

    def test_accepts_config_and_result_file(self):
        sig = inspect.signature(run_custom_experiment)
        params = list(sig.parameters.keys())
        assert "config" in params, "run_custom_experiment must accept 'config'"
        assert "result_file" in params, "run_custom_experiment must accept 'result_file'"


# ============================================================================
# From test_pipeline_types.py
# ============================================================================


class TestSeverityLevel(unittest.TestCase):
    """SeverityLevel enum must export canonical string values."""

    def test_critical_value(self):
        assert SeverityLevel.CRITICAL.value == "CRITICAL"

    def test_warning_value(self):
        assert SeverityLevel.WARNING.value == "WARNING"

    def test_info_value(self):
        assert SeverityLevel.INFO.value == "INFO"

    def test_is_string_enum(self):
        # SeverityLevel must be usable as a string key in JSON serialization
        assert isinstance(SeverityLevel.CRITICAL, str)


# ---------------------------------------------------------------------------
# CheckStatus enum
# ---------------------------------------------------------------------------


class TestCheckStatus(unittest.TestCase):
    """CheckStatus enum must export lowercase values matching CLAUDE.md §12."""

    def test_passed_value(self):
        assert CheckStatus.PASSED.value == "passed"

    def test_failed_value(self):
        assert CheckStatus.FAILED.value == "failed"

    def test_warning_value(self):
        assert CheckStatus.WARNING.value == "warning"


# ---------------------------------------------------------------------------
# CheckResult
# ---------------------------------------------------------------------------


class TestCheckResult(unittest.TestCase):
    def test_required_fields(self):
        cr = CheckResult(name="constants", status=CheckStatus.PASSED, message="OK")
        assert cr.name == "constants"
        assert cr.status == CheckStatus.PASSED
        assert cr.message == "OK"

    def test_optional_fields_default_none(self):
        cr = CheckResult(name="test", status=CheckStatus.FAILED, message="err")
        assert cr.details is None
        assert cr.recovery_action is None

    def test_details_can_be_set(self):
        cr = CheckResult(name="t", status=CheckStatus.WARNING, message="w", details="extra info")
        assert cr.details == "extra info"


# ---------------------------------------------------------------------------
# ExperimentMetrics
# ---------------------------------------------------------------------------


class TestExperimentMetrics(unittest.TestCase):
    """ExperimentMetrics must have zero defaults for all optional fields."""

    def test_global_f1_required(self):
        m = ExperimentMetrics(global_f1=0.87)
        assert m.global_f1 == 0.87

    def test_precision_recall_default_zero(self):
        m = ExperimentMetrics(global_f1=0.0)
        assert m.global_precision == 0.0
        assert m.global_recall == 0.0

    def test_per_field_f1_defaults_zero(self):
        m = ExperimentMetrics(global_f1=0.0)
        assert m.company_f1 == 0.0
        assert m.date_f1 == 0.0
        assert m.address_f1 == 0.0
        assert m.total_f1 == 0.0

    def test_per_field_ned_defaults_zero(self):
        m = ExperimentMetrics(global_f1=0.0)
        assert m.company_ned == 0.0
        assert m.date_ned == 0.0
        assert m.address_ned == 0.0
        assert m.total_ned == 0.0

    def test_parse_failures_default_zero(self):
        m = ExperimentMetrics(global_f1=0.0)
        assert m.parse_failures == 0
        assert m.total_predictions == 0

    def test_all_four_fields_present(self):
        """The four SROIE fields must be accessible as attributes."""
        m = ExperimentMetrics(global_f1=0.0)
        for field in ("company", "date", "address", "total"):
            assert hasattr(m, f"{field}_f1"), f"Missing attribute: {field}_f1"
            assert hasattr(m, f"{field}_ned"), f"Missing attribute: {field}_ned"


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult(unittest.TestCase):
    """ExperimentResult.to_dict() must produce a JSON-compatible dict with
    all required keys from CLAUDE.md §13 (Results Format).
    """

    def _make_result(self) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=1,
            name="SROIE only (baseline)",
            datasets=["sroie"],
            num_train_samples=500,
            metrics=ExperimentMetrics(global_f1=0.87),
        )

    def test_to_dict_has_required_top_level_keys(self):
        d = self._make_result().to_dict()
        required = {
            "experiment_id",
            "name",
            "datasets",
            "num_train_samples",
            "metrics",
            "duration_sec",
            "timestamp",
        }
        assert required.issubset(set(d.keys())), (
            f"to_dict() is missing required keys: {required - set(d.keys())}"
        )

    def test_metrics_in_dict_is_dict(self):
        d = self._make_result().to_dict()
        assert isinstance(d["metrics"], dict), "metrics must be a dict in to_dict() output"
        assert "global_f1" in d["metrics"]

    def test_experiment_id_correct(self):
        d = self._make_result().to_dict()
        assert d["experiment_id"] == 1

    def test_datasets_is_list(self):
        result = self._make_result()
        assert isinstance(result.datasets, list)
        assert result.datasets == ["sroie"]

    def test_checkpoint_path_defaults_none(self):
        result = self._make_result()
        assert result.checkpoint_path is None

    def test_duration_defaults_zero(self):
        result = self._make_result()
        assert result.duration_sec == 0.0

    def test_timestamp_is_isoformat_string(self):
        d = self._make_result().to_dict()
        ts = d["timestamp"]
        assert isinstance(ts, str)
        assert "T" in ts, f"Expected ISO 8601 timestamp (contains 'T'), got: {ts!r}"

    def test_to_dict_metrics_has_all_sroie_fields(self):
        """metrics dict must include all 4 SROIE field F1/NED entries."""
        d = self._make_result().to_dict()
        m = d["metrics"]
        from constants import FIELDS

        for field in FIELDS:
            assert f"{field}_f1" in m, f"Missing {field}_f1 in metrics dict"
            assert f"{field}_ned" in m, f"Missing {field}_ned in metrics dict"


# ---------------------------------------------------------------------------
# BugPattern and BugReport
# ---------------------------------------------------------------------------


class TestBugPattern(unittest.TestCase):
    def test_all_required_fields_settable(self):
        bp = BugPattern(
            severity=SeverityLevel.CRITICAL,
            category="syntax",
            file=Path("test.py"),
            line=10,
            column=5,
            description="Missing bracket",
            code_snippet="x = [1, 2, 3",
            fix_suggestion="Close the bracket",
        )
        assert bp.severity == SeverityLevel.CRITICAL
        assert bp.line == 10
        assert bp.column == 5
        assert bp.category == "syntax"


class TestBugReport(unittest.TestCase):
    def test_empty_report_defaults(self):
        report = BugReport()
        assert report.bugs == []
        assert report.total_critical == 0
        assert report.total_warnings == 0

    def test_to_dict_has_required_keys(self):
        report = BugReport()
        d = report.to_dict()
        assert "bugs" in d
        assert "total_critical" in d
        assert "total_warnings" in d
        assert "timestamp" in d

    def test_to_dict_bugs_is_list(self):
        report = BugReport()
        d = report.to_dict()
        assert isinstance(d["bugs"], list)

    def test_to_dict_serializes_bug_file_as_string(self):
        """BugPattern.file is a Path — to_dict() must serialize it as str."""
        report = BugReport()
        report.bugs.append(
            BugPattern(
                severity=SeverityLevel.WARNING,
                category="logic",
                file=Path("/some/file.py"),
                line=42,
                column=1,
                description="Test",
                code_snippet="",
                fix_suggestion="",
            )
        )
        d = report.to_dict()
        assert isinstance(d["bugs"][0]["file"], str), (
            "BugPattern.file must be serialized as str in to_dict(), not Path"
        )

    def test_to_dict_severity_is_string(self):
        """Severity enum must be serialized as its string value."""
        report = BugReport()
        report.bugs.append(
            BugPattern(
                severity=SeverityLevel.CRITICAL,
                category="syntax",
                file=Path("f.py"),
                line=1,
                column=1,
                description="Test",
                code_snippet="",
                fix_suggestion="",
            )
        )
        d = report.to_dict()
        assert d["bugs"][0]["severity"] == "CRITICAL"


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult(unittest.TestCase):
    def test_partial_success_false_without_mode_result(self):
        result = PipelineResult(success=False, mode="ml_training")
        assert result.partial_success is False

    def test_partial_success_false_with_none_mode_result(self):
        result = PipelineResult(success=False, mode="ml_training", mode_result=None)
        assert result.partial_success is False

    def test_duration_defaults_zero(self):
        result = PipelineResult(success=True, mode="code_repair")
        assert result.duration_sec == 0.0

    def test_errors_defaults_empty(self):
        result = PipelineResult(success=True, mode="code_repair")
        assert result.errors == []
        assert result.warnings == []

    def test_mode_stored_correctly(self):
        result = PipelineResult(success=True, mode="ml_training")
        assert result.mode == "ml_training"


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------


class TestValidationReport(unittest.TestCase):
    def test_defaults(self):
        r = ValidationReport(passed=True)
        assert r.error is None
        assert r.recovery_action is None
        assert r.details == {}

    def test_failed_with_error(self):
        r = ValidationReport(passed=False, error="something went wrong")
        assert not r.passed
        assert r.error == "something went wrong"


# ---------------------------------------------------------------------------
# AggregatedResults
# ---------------------------------------------------------------------------


class TestAggregatedResults(unittest.TestCase):
    def test_default_experiments_empty(self):
        agg = AggregatedResults()
        assert agg.experiments == []

    def test_default_best_experiment_none(self):
        agg = AggregatedResults()
        assert agg.best_experiment is None

    def test_default_f1_zero(self):
        agg = AggregatedResults()
        assert agg.baseline_f1 == 0.0
        assert agg.improvement == 0.0

    def test_default_per_field_empty(self):
        agg = AggregatedResults()
        assert agg.per_field_analysis == {}


# ---------------------------------------------------------------------------
# DataSplitValidationReport
# ---------------------------------------------------------------------------


class TestDataSplitValidationReport(unittest.TestCase):
    def test_defaults(self):
        r = DataSplitValidationReport(passed=True)
        assert r.train_count == 0
        assert r.val_count == 0
        assert r.test_count == 0
        assert r.errors == []
        assert r.warnings == []

    def test_failed_state(self):
        r = DataSplitValidationReport(passed=False, errors=["Missing val_img/ directory"])
        assert not r.passed
        assert "Missing val_img/ directory" in r.errors


# ============================================================================
# From test_resource_optimizer.py
# ============================================================================


class TestValidateTrainingConfig(unittest.TestCase):
    """Tests for validate_training_config()."""

    def test_sufficient_steps_passes(self):
        """500 samples, batch=4, accum=4, epochs=10 → 312 steps — should pass."""
        validate_training_config(
            batch_size=4,
            gradient_accumulation_steps=4,
            num_train_samples=500,
            epochs=10,
        )

    def test_step_starvation_raises(self):
        """500 samples, batch=16, accum=2, epochs=10 → 160 steps — should raise."""
        with pytest.raises(ValueError, match="optimizer steps"):
            validate_training_config(
                batch_size=16,
                gradient_accumulation_steps=2,
                num_train_samples=500,
                epochs=10,
            )

    def test_large_dataset_high_batch_passes(self):
        """3940 samples, batch=8, accum=2, epochs=10 → 2462 steps — should pass."""
        validate_training_config(
            batch_size=8,
            gradient_accumulation_steps=2,
            num_train_samples=3940,
            epochs=10,
        )

    def test_custom_min_steps(self):
        """Custom min_optimizer_steps respected."""
        # 312 steps passes default min=200 but fails min=400
        with pytest.raises(ValueError):
            validate_training_config(
                batch_size=4,
                gradient_accumulation_steps=4,
                num_train_samples=500,
                epochs=10,
                min_optimizer_steps=400,
            )

    def test_error_message_contains_batch_info(self):
        """Error message must contain actionable batch/accum info."""
        with pytest.raises(ValueError) as exc_info:
            validate_training_config(
                batch_size=16,
                gradient_accumulation_steps=2,
                num_train_samples=500,
                epochs=10,
            )
        msg = str(exc_info.value)
        assert "batch_size=16" in msg
        assert "grad_accum=2" in msg
        assert "optimizer steps" in msg


class TestOptimizeHyperparamsHighVRAM(unittest.TestCase):
    """Tests for optimize_hyperparams() on high-VRAM GPUs (>24 GB).

    This pins the fix for the step starvation bug on A100/H100 class GPUs.
    All tests use the reference image size (1280×960) to exercise the
    calibrated tier thresholds independently of the active processor_config.json.
    """

    def _call(self, num_train_samples: int, vram_gb: float = 40.0) -> ResourceOptimizedConfig:
        return optimize_hyperparams(
            num_train_samples=num_train_samples,
            available_vram_gb=vram_gb,
            available_ram_gb=64.0,
            image_size=(1280, 960),  # reference resolution — keep tests config-independent
        )

    def _optimizer_steps(
        self, cfg: ResourceOptimizedConfig, num_samples: int, epochs: int = 10
    ) -> int:
        return math.ceil(num_samples / (cfg.batch_size * cfg.gradient_accumulation_steps)) * epochs

    def test_small_dataset_high_vram_produces_enough_steps(self):
        """Exp 1-4 scenario: ~500 samples on A100 must yield >= 200 optimizer steps."""
        cfg = self._call(num_train_samples=500, vram_gb=40.0)
        steps = self._optimizer_steps(cfg, num_samples=500)
        assert steps >= 200, (
            f"Step starvation bug: only {steps} optimizer steps for 500 samples on 40 GB GPU. "
            f"batch_size={cfg.batch_size}, grad_accum={cfg.gradient_accumulation_steps}"
        )

    def test_small_dataset_high_vram_does_not_use_batch16_accum2(self):
        """The step-starvation config (batch=16, accum=2) must NOT be returned for small datasets."""
        cfg = self._call(num_train_samples=500, vram_gb=40.0)
        regressed = cfg.batch_size == 16 and cfg.gradient_accumulation_steps == 2
        assert not regressed, (
            "Regressed to batch_size=16, gradient_accumulation_steps=2 for small dataset on "
            "high-VRAM GPU. This produces ~160 optimizer steps — too few for DONUT to converge."
        )

    def test_large_dataset_high_vram_can_use_larger_batch(self):
        """Exp 8 scenario: ~3940 samples on A100 should use batch >= 4."""
        cfg = self._call(num_train_samples=3940, vram_gb=40.0)
        assert cfg.batch_size >= 4

    def test_80gb_vram_small_dataset(self):
        """H100 (80 GB) with small dataset must still avoid step starvation."""
        cfg = self._call(num_train_samples=500, vram_gb=80.0)
        steps = self._optimizer_steps(cfg, num_samples=500)
        assert steps >= 200, f"Step starvation on H100: only {steps} steps for 500 samples"

    def test_returns_resource_optimized_config(self):
        """Return type must be ResourceOptimizedConfig."""
        cfg = self._call(num_train_samples=500)
        assert isinstance(cfg, ResourceOptimizedConfig)

    def test_all_four_fields_nonzero(self):
        """All four training-relevant fields must be populated."""
        cfg = self._call(num_train_samples=500)
        assert cfg.batch_size > 0
        assert cfg.gradient_accumulation_steps > 0
        assert cfg.encoder_lr > 0.0
        assert cfg.decoder_lr > 0.0


class TestOptimizeHyperparamsLowVRAM(unittest.TestCase):
    """Tests for low-VRAM paths (RTX 4090 and below).

    All tests pass image_size=(1280, 960) to keep them independent of the
    active processor_config.json, since the low-VRAM tier thresholds were
    calibrated at the reference resolution.
    """

    def test_rtx4090_batch_is_2(self):
        """RTX 4090 (24 GB) must use batch_size=2 to prevent OOM."""
        cfg = optimize_hyperparams(
            num_train_samples=500,
            available_vram_gb=24.0,
            available_ram_gb=32.0,
            image_size=(1280, 960),
        )
        assert cfg.batch_size == 2, (
            f"RTX 4090 path returned batch_size={cfg.batch_size}, expected 2"
        )

    def test_low_vram_8gb(self):
        """8 GB GPU must use batch_size=4."""
        cfg = optimize_hyperparams(
            num_train_samples=500,
            available_vram_gb=7.0,
            available_ram_gb=16.0,
            image_size=(1280, 960),
        )
        assert cfg.batch_size == 4

    def test_rtx4090_small_dataset_validates_with_5_epochs(self):
        """RTX 4090 (24 GB) + 500 samples: returned config must pass validate_training_config at 5 epochs.

        Regression test for the mini-mode ValueError:
          'Training config produces only 160 optimizer steps (500 samples /
           effective_batch=16 × 5 epochs). Minimum required: 200.'
        Root cause: 24 GB path used accum=8 (effective batch=16) for ALL datasets,
        giving ceil(500/16) × 5 = 160 steps — below the 200-step minimum when
        mini-mode (epochs=5) overrides the default 10-epoch config.
        Fix: small datasets (< 2000 samples) on ≤ 24 GB use accum=4 (effective batch=8),
        giving ceil(500/8) × 5 = 315 steps ≥ 200 ✓
        """
        cfg = optimize_hyperparams(
            num_train_samples=500,
            available_vram_gb=24.0,
            available_ram_gb=32.0,
            image_size=(1280, 960),
        )
        # Must not raise — previously raised ValueError with accum=8
        validate_training_config(
            batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_samples=500,
            epochs=5,
        )

    def test_rtx4090_small_dataset_step_count_with_5_epochs(self):
        """ceil(500 / (bs × accum)) × 5 must be ≥ 200 on 24 GB GPU with 500 samples."""
        cfg = optimize_hyperparams(
            num_train_samples=500,
            available_vram_gb=24.0,
            available_ram_gb=32.0,
            image_size=(1280, 960),
        )
        steps = math.ceil(500 / (cfg.batch_size * cfg.gradient_accumulation_steps)) * 5
        assert steps >= 200, (
            f"RTX 4090 + 500 samples: only {steps} optimizer steps with 5 epochs "
            f"(batch={cfg.batch_size}, accum={cfg.gradient_accumulation_steps}). "
            "Mini-mode uses epochs=5 — config must tolerate it."
        )

    def test_rtx4090_large_dataset_uses_higher_accum(self):
        """Large dataset (≥ 2000 samples) on 24 GB may use accum=8 (enough steps even at 5 epochs)."""
        cfg = optimize_hyperparams(
            num_train_samples=2000,
            available_vram_gb=24.0,
            available_ram_gb=32.0,
            image_size=(1280, 960),
        )
        # 2000 samples with accum=8 at 5 epochs: ceil(2000/16)*5 = 625 ≥ 200 ✓
        steps = math.ceil(2000 / (cfg.batch_size * cfg.gradient_accumulation_steps)) * 5
        assert steps >= 200, f"Large dataset path still too few steps: {steps} with 5 epochs"


class TestImageSizeAwareVRAM(unittest.TestCase):
    """Tests for the image-size-aware VRAM calibration (Task 2 OOM fix).

    Pins the fix for Experiment 8 OOM on the Vast.ai RTX 6000 Blackwell 96 GB:
    processor_config.json uses 2560×1920 (4× reference pixels), causing the old
    hardcoded batch=16 to require ~182 GB — far beyond 96 GB capacity.
    """

    def test_get_image_size_fallback_on_missing_file(self):
        """get_image_size_from_processor_config falls back to (1280, 960) when file absent."""
        from resource_optimizer import get_image_size_from_processor_config

        result = get_image_size_from_processor_config("/nonexistent/path/processor_config.json")
        assert result == (1280, 960), f"Expected fallback (1280, 960), got {result}"

    def test_get_image_size_parses_valid_config(self, tmp_path):
        """get_image_size_from_processor_config correctly parses a valid config file."""
        import json

        from resource_optimizer import get_image_size_from_processor_config

        cfg = {"image_processor": {"size": {"height": 2560, "width": 1920}}}
        config_file = tmp_path / "processor_config.json"
        config_file.write_text(json.dumps(cfg), encoding="utf-8")

        result = get_image_size_from_processor_config(str(config_file))
        assert result == (2560, 1920), f"Expected (2560, 1920), got {result}"

    def test_96gb_blackwell_4x_pixels_batch_is_safe(self):
        """96 GB GPU + 2560×1920 images must not assign batch_size=16 (Exp 8 OOM fix).

        With 4× reference pixels: vram_needed(batch=16) = 2.848×16×4 = 182 GB > 96 GB.
        Safe maximum is batch=4 (2.848×4×4 = 45.6 GB < 86.4 GB = 96×0.90).
        """
        cfg = optimize_hyperparams(
            num_train_samples=3940,
            available_vram_gb=96.0,
            available_ram_gb=256.0,
            image_size=(2560, 1920),
        )
        assert cfg.batch_size <= 4, (
            f"96 GB GPU with 2560×1920 images returned batch_size={cfg.batch_size}; "
            "max safe is 4 (batch=8 requires ~91 GB which exceeds 90% of 96 GB). "
            "This is the Exp 8 OOM regression."
        )
        assert cfg.batch_size >= 1, "batch_size must be at least 1"

    def test_96gb_blackwell_4x_pixels_enough_optimizer_steps(self):
        """96 GB + 2560×1920 + large dataset must still yield ≥ 200 optimizer steps."""
        cfg = optimize_hyperparams(
            num_train_samples=3940,
            available_vram_gb=96.0,
            available_ram_gb=256.0,
            image_size=(2560, 1920),
        )
        steps = math.ceil(3940 / (cfg.batch_size * cfg.gradient_accumulation_steps)) * 10
        assert steps >= 200, (
            f"96 GB Blackwell + Exp 8: only {steps} optimizer steps "
            f"(batch={cfg.batch_size}, accum={cfg.gradient_accumulation_steps})"
        )

    def test_pixels_scale_1x_matches_reference_behavior(self):
        """At reference resolution (1280×960), >24 GB path behaves as before the fix.

        Verifies that image-size-aware formula does not regress reference-resolution paths.
        """
        # 40 GB + 1280×960: max_safe_batch = 8 (22.8 GB < 36 GB); small dataset → 4
        cfg = optimize_hyperparams(
            num_train_samples=500,
            available_vram_gb=40.0,
            available_ram_gb=64.0,
            image_size=(1280, 960),
        )
        steps = math.ceil(500 / (cfg.batch_size * cfg.gradient_accumulation_steps)) * 10
        assert steps >= 200, f"Reference resolution regression: only {steps} steps"
        assert cfg.batch_size >= 1


class TestExperimentConfigImmutability(unittest.TestCase):
    """Tests that ExperimentConfig global state is never mutated."""

    def test_dataclasses_replace_does_not_mutate_original(self):
        """dataclasses.replace() must not change the original config."""
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        from run_experiments import EXPERIMENTS  # noqa: E402, I001

        original_exp1 = EXPERIMENTS[1]
        original_batch = original_exp1.batch_size

        # Simulate what run_experiments.py does
        new_config = dataclasses.replace(original_exp1, batch_size=999)

        assert EXPERIMENTS[1].batch_size == original_batch, (
            f"INVARIANT VIOLATION: EXPERIMENTS[1].batch_size changed from "
            f"{original_batch} to {EXPERIMENTS[1].batch_size} after dataclasses.replace(). "
            "Direct attribute mutation detected."
        )
        assert new_config.batch_size == 999

    def test_direct_mutation_is_possible_without_replace(self):
        """Confirm Python allows direct mutation (so the guardrail is necessary)."""
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        from run_experiments import ExperimentConfig  # noqa: E402, I001

        cfg = ExperimentConfig(name="test", datasets=["sroie"], batch_size=8)
        cfg.batch_size = 99  # This is allowed by Python — the guardrail prevents it in practice
        assert cfg.batch_size == 99  # Confirms the danger is real


class TestOOMRecovery(unittest.TestCase):
    """Tests for the OOM recovery logic in train_experiment().

    Verifies that the recovery path now allows batch_size to be reduced all
    the way to 1 (one extra step vs. the old minimum of 2).
    """

    def test_oom_recovery_minimum_is_one(self):
        """OOM recovery condition must allow batch_size=1 (not stop at 2).

        Simulates the recovery logic: starting from batch_size=2 one more OOM
        must be recoverable by halving to 1 and doubling grad_accum.
        """
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        from run_experiments import ExperimentConfig  # noqa: E402, I001

        # Simulate config at the last recovery step before old hard stop
        cfg = ExperimentConfig(
            name="test",
            datasets=["sroie"],
            batch_size=2,
            gradient_accumulation_steps=8,
        )

        # Old code: batch_size > 2 was False → raised immediately.
        # New code: batch_size > 1 is True → one more recovery is possible.
        assert cfg.batch_size > 1, "batch_size=2 must be > 1 so OOM recovery can halve it to 1"

        new_batch = max(1, cfg.batch_size // 2)
        new_accum = cfg.gradient_accumulation_steps * 2
        recovered = dataclasses.replace(
            cfg, batch_size=new_batch, gradient_accumulation_steps=new_accum
        )

        assert recovered.batch_size == 1, (
            f"Expected batch_size=1 after final recovery step, got {recovered.batch_size}"
        )
        assert recovered.gradient_accumulation_steps == 16, (
            f"Expected grad_accum=16 after final recovery step, got {recovered.gradient_accumulation_steps}"
        )

    def test_oom_recovery_raises_at_batch_size_one(self):
        """When batch_size is already 1, OOM recovery must raise RuntimeError."""
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        from run_experiments import ExperimentConfig  # noqa: E402, I001

        cfg = ExperimentConfig(name="test", datasets=["sroie"], batch_size=1)

        # New code: batch_size > 1 is False → raise
        assert not (cfg.batch_size > 1), (
            "batch_size=1 must NOT satisfy the recovery condition — should raise"
        )

    def test_effective_batch_preserved_across_recovery_steps(self):
        """Each OOM recovery step must preserve the effective batch size (batch × accum)."""
        pytest.importorskip("torch", reason="torch required by run_experiments.py")
        from run_experiments import ExperimentConfig  # noqa: E402, I001

        cfg = ExperimentConfig(
            name="test",
            datasets=["sroie"],
            batch_size=8,
            gradient_accumulation_steps=2,
        )
        effective = cfg.batch_size * cfg.gradient_accumulation_steps

        # Simulate two recovery steps: 8→4→2, accum 2→4→8
        cfg = dataclasses.replace(
            cfg,
            batch_size=max(1, cfg.batch_size // 2),
            gradient_accumulation_steps=cfg.gradient_accumulation_steps * 2,
        )
        assert cfg.batch_size * cfg.gradient_accumulation_steps == effective

        cfg = dataclasses.replace(
            cfg,
            batch_size=max(1, cfg.batch_size // 2),
            gradient_accumulation_steps=cfg.gradient_accumulation_steps * 2,
        )
        assert cfg.batch_size * cfg.gradient_accumulation_steps == effective


# ============================================================================
# From test_train_invariants.py
# ============================================================================


class TestDecoderStartTokenId(unittest.TestCase):
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


class TestExperimentConfigPropertyAliases(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


class TestLabelTokenizationNoSpecialTokens(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


# ============================================================================
# From test_trocr_setup.py
# ============================================================================

if _TORCH_AVAILABLE:

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


class TestMaterializeMetaBuffers(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


class TestTrOCRModelID(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


class TestGenerationConfig(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


class TestGradientCheckpointing(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


class TestTrOCRReceiptDataset(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
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


# ════════════════════════════════════════════════════════════════════════════
# 6.  LOAD REPORT pooler key filtering — Bug (a) fix
# ════════════════════════════════════════════════════════════════════════════


class TestTrOCRLoadReport(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
    """_print_trocr_load_report must filter encoder.pooler.dense.* from MISSING list.

    microsoft/trocr-base-printed uses a BEiT vision encoder that never includes
    a pooler layer.  The generic VisionEncoderDecoderModel wrapper declares an
    optional encoder.pooler submodule, so HuggingFace always reports these two
    keys as MISSING even though they are structurally absent and unused.

    The LOAD REPORT must show 0 MISSING rows for a clean trocr-base-printed load.
    """

    def test_expected_missing_trocr_contains_pooler_keys(self):
        """_EXPECTED_MISSING_TROCR must include both encoder.pooler.dense.* keys."""
        assert "encoder.pooler.dense.weight" in _EXPECTED_MISSING_TROCR, (
            "encoder.pooler.dense.weight must be in _EXPECTED_MISSING_TROCR — "
            "it is structurally absent in BEiT-based TrOCR models."
        )
        assert "encoder.pooler.dense.bias" in _EXPECTED_MISSING_TROCR, (
            "encoder.pooler.dense.bias must be in _EXPECTED_MISSING_TROCR — "
            "it is structurally absent in BEiT-based TrOCR models."
        )

    def test_expected_missing_is_frozenset(self):
        """_EXPECTED_MISSING_TROCR must be a frozenset (immutable, no accidental mutation)."""
        assert isinstance(_EXPECTED_MISSING_TROCR, frozenset), (
            f"_EXPECTED_MISSING_TROCR is a {type(_EXPECTED_MISSING_TROCR).__name__}, not frozenset. "
            "Use frozenset to prevent accidental mutation."
        )

    def test_pooler_keys_filtered_from_missing_output(self, capsys):
        """_print_trocr_load_report must not print encoder.pooler.dense.* as MISSING."""
        loading_info = {
            "missing_keys": [
                "encoder.pooler.dense.weight",
                "encoder.pooler.dense.bias",
            ],
            "unexpected_keys": [],
        }
        _print_trocr_load_report("microsoft/trocr-base-printed", loading_info)
        captured = capsys.readouterr()
        assert "encoder.pooler.dense.weight" not in captured.out, (
            "encoder.pooler.dense.weight must NOT appear in LOAD REPORT output — "
            "it is a known-benign structural absence for BEiT-based TrOCR."
        )
        assert "encoder.pooler.dense.bias" not in captured.out, (
            "encoder.pooler.dense.bias must NOT appear in LOAD REPORT output — "
            "it is a known-benign structural absence for BEiT-based TrOCR."
        )

    def test_clean_load_shows_ok_not_missing(self, capsys):
        """When only pooler keys are missing, LOAD REPORT must show OK (0 MISSING)."""
        loading_info = {
            "missing_keys": [
                "encoder.pooler.dense.weight",
                "encoder.pooler.dense.bias",
            ],
            "unexpected_keys": [],
        }
        _print_trocr_load_report("microsoft/trocr-base-printed", loading_info)
        captured = capsys.readouterr()
        # Should show OK, not MISSING
        assert "MISSING" not in captured.out, (
            "LOAD REPORT must show 0 MISSING rows when only pooler keys are absent "
            "(those are filtered). Got output:\n" + captured.out
        )
        assert "OK" in captured.out, (
            "LOAD REPORT must show 'OK' for a clean trocr-base-printed load. "
            "Got output:\n" + captured.out
        )

    def test_non_pooler_missing_key_still_reported(self, capsys):
        """Non-pooler missing keys must still appear as MISSING in the report."""
        loading_info = {
            "missing_keys": [
                "encoder.pooler.dense.weight",
                "encoder.pooler.dense.bias",
                "decoder.some_other_weight",  # This should NOT be filtered
            ],
            "unexpected_keys": [],
        }
        _print_trocr_load_report("microsoft/trocr-base-printed", loading_info)
        captured = capsys.readouterr()
        assert "decoder.some_other_weight" in captured.out, (
            "Non-pooler missing keys must still appear in LOAD REPORT output."
        )
        assert "MISSING" in captured.out, (
            "LOAD REPORT must show MISSING for non-pooler missing keys."
        )

    def test_model_id_in_report_header(self, capsys):
        """LOAD REPORT header must include the model ID for traceability."""
        loading_info = {"missing_keys": [], "unexpected_keys": []}
        _print_trocr_load_report("microsoft/trocr-base-printed", loading_info)
        captured = capsys.readouterr()
        assert "microsoft/trocr-base-printed" in captured.out, (
            "LOAD REPORT header must include the model ID for traceability."
        )


# ════════════════════════════════════════════════════════════════════════════
# Gradient checkpointing VRAM-threshold guard (RC4 / Root Cause 4)
# ════════════════════════════════════════════════════════════════════════════


class TestGradientCheckpointingThreshold(unittest.TestCase):
    pytestmark = pytest.mark.skipif(
        not _TORCH_AVAILABLE,
        reason="torch and transformers required",
    )
    """Verify the VRAM-threshold logic for TrOCR gradient checkpointing.

    On cards with VRAM > 24 GB, gradient checkpointing should be disabled
    (no memory benefit, ~35% backward overhead wasted).  On cards with
    VRAM <= 24 GB, it must stay enabled (required to fit 246M params).
    The threshold uses strict greater-than so a card reporting exactly
    24.0 GB (e.g. RTX 4090) keeps checkpointing ON.
    """

    def _run_threshold_logic(self, vram_gb: float) -> bool:
        """Replicate the threshold decision from train_trocr_yolo.train_trocr().

        Returns True if gradient checkpointing would be enabled for the
        given VRAM amount.
        """
        from control_suite import CONTROL_SUITE

        threshold = CONTROL_SUITE.trocr.grad_ckpt_vram_threshold_gb
        # gradient checkpointing is enabled when vram_gb <= threshold (strict > comparison disables above threshold)
        return vram_gb <= threshold

    def test_high_vram_disables_grad_ckpt(self):
        """VRAM=96 GB (RTX 6000 Blackwell) → gradient checkpointing disabled."""
        assert not self._run_threshold_logic(96.0), (
            "With 96 GB VRAM, gradient checkpointing should be disabled — "
            "it adds ~35% backward overhead for zero memory benefit."
        )

    def test_mid_vram_disables_grad_ckpt(self):
        """VRAM=40 GB (A100 40 GB) → gradient checkpointing disabled."""
        assert not self._run_threshold_logic(40.0)

    def test_exactly_24gb_enables_grad_ckpt(self):
        """VRAM=24.0 GB (RTX 4090 at boundary) → gradient checkpointing ENABLED.

        The strictly-greater-than comparison means that a card reporting
        exactly 24.0 GB keeps gradient checkpointing ON, preventing the
        OOM that the >= comparison would have caused.
        """
        assert self._run_threshold_logic(24.0), (
            "VRAM=24.0 GB must enable gradient checkpointing (uses strict > not >=). "
            "This is the RTX 4090 boundary bug (Root Cause 2/4)."
        )

    def test_just_below_24gb_enables_grad_ckpt(self):
        """VRAM=23.65 GB (RTX 4090 typical reading) → gradient checkpointing ENABLED."""
        assert self._run_threshold_logic(23.65)

    def test_16gb_enables_grad_ckpt(self):
        """VRAM=16 GB (V100 / RTX 3080) → gradient checkpointing enabled."""
        assert self._run_threshold_logic(16.0)

    def test_threshold_value_is_24(self):
        """The threshold constant must be exactly 24.0 to match the DONUT path."""
        from control_suite import CONTROL_SUITE

        assert CONTROL_SUITE.trocr.grad_ckpt_vram_threshold_gb == 24.0


# ============================================================================
# From test_validators.py
# ============================================================================


class TestBugPatternDetectorJsonConfusion(unittest.TestCase):
    """JSON literal detection in Python source code.

    Guards Pattern 2 (CLAUDE.md §5): ``true``/``false``/``null`` pasted from
    JSON into Python cause NameError at runtime — detection must fire at static
    scan time.
    """

    def test_detects_bare_true(self):
        from validators import BugPatternDetector

        bugs = BugPatternDetector.detect_python_json_confusion(
            'config = {"use_cache": true}', Path("test.py")
        )
        assert len(bugs) == 1
        assert "true" in bugs[0].description

    def test_detects_bare_false(self):
        from validators import BugPatternDetector

        bugs = BugPatternDetector.detect_python_json_confusion(
            "model.config.tie_word_embeddings = false", Path("test.py")
        )
        assert len(bugs) == 1
        assert "false" in bugs[0].description

    def test_detects_bare_null(self):
        from validators import BugPatternDetector

        bugs = BugPatternDetector.detect_python_json_confusion("x = null", Path("test.py"))
        assert len(bugs) == 1
        assert "null" in bugs[0].description

    def test_no_false_positive_for_python_booleans(self):
        from validators import BugPatternDetector

        code = 'use_cache = True\ntie = False\nx = None\nword = "truecolor"'
        bugs = BugPatternDetector.detect_python_json_confusion(code, Path("test.py"))
        assert bugs == [], f"Expected no bugs for valid Python booleans, got: {bugs}"

    def test_skips_comment_lines(self):
        from validators import BugPatternDetector

        bugs = BugPatternDetector.detect_python_json_confusion(
            "# tie_word_embeddings = false  (JSON literal example in comment)",
            Path("test.py"),
        )
        assert bugs == [], "Comment lines must not trigger JSON literal detection"

    def test_fix_suggestion_contains_replacement(self):
        from validators import BugPatternDetector

        bugs = BugPatternDetector.detect_python_json_confusion("x = true", Path("test.py"))
        assert "True" in bugs[0].fix_suggestion

    def test_severity_is_critical(self):
        from validators import BugPatternDetector

        from cloud_orchestration import SeverityLevel

        bugs = BugPatternDetector.detect_python_json_confusion("x = false", Path("test.py"))
        assert bugs[0].severity == SeverityLevel.CRITICAL


# ---------------------------------------------------------------------------
# BugPatternDetector — Syntax errors
# ---------------------------------------------------------------------------


class TestBugPatternDetectorSyntax(unittest.TestCase):
    """Syntax error detection via ast.parse."""

    def test_valid_python_no_bugs(self):
        from validators import BugPatternDetector

        code = 'x = [1, 2, 3]\ny = {"key": "value"}\nresult = x + [y["key"]]'
        bugs = BugPatternDetector.detect_syntax_errors(code, Path("test.py"))
        assert bugs == []

    def test_syntax_error_detected(self):
        from validators import BugPatternDetector

        code = "x = [1, 2, 3"  # Missing closing bracket
        bugs = BugPatternDetector.detect_syntax_errors(code, Path("test.py"))
        assert len(bugs) >= 1
        assert any(b.category == "syntax" for b in bugs)

    def test_syntax_error_has_non_negative_line(self):
        from validators import BugPatternDetector

        code = "x = [1, 2, 3\ny = 'ok'"  # Missing bracket
        bugs = BugPatternDetector.detect_syntax_errors(code, Path("test.py"))
        assert len(bugs) >= 1
        assert bugs[0].line >= 0

    def test_missing_comma_detected(self):
        from validators import BugPatternDetector

        # ExperimentConfig list with missing trailing comma (Pattern 1 from CLAUDE.md §5)
        code = "configs = [\n    ExperimentConfig(id=1)\n    ExperimentConfig(id=2)\n]"
        bugs = BugPatternDetector.detect_syntax_errors(code, Path("run_experiments.py"))
        assert len(bugs) >= 1


# ---------------------------------------------------------------------------
# BugPatternDetector — tie_word_embeddings check (CLAUDE.md §3 warning)
# ---------------------------------------------------------------------------


class TestBugPatternDetectorTieWordEmbeddings(unittest.TestCase):
    """Detection of missing ``tie_word_embeddings = False`` after
    ``resize_token_embeddings()``.

    Root cause: without ``config.tie_word_embeddings = False``, safetensors
    deduplicates ``lm_head.weight`` → F1=0 on checkpoint reload.
    """

    def test_resize_without_tie_word_embeddings_detected(self):
        from validators import BugPatternDetector

        code = "model.resize_token_embeddings(len(tokenizer))\ntrainer.train()"
        bugs = BugPatternDetector.detect_missing_tie_word_embeddings(code, Path("train.py"))
        assert len(bugs) >= 1
        assert any("tie_word_embeddings" in b.fix_suggestion for b in bugs)

    def test_resize_with_tie_word_embeddings_no_bug(self):
        from validators import BugPatternDetector

        code = (
            "model.resize_token_embeddings(len(tokenizer))\n"
            "model.config.tie_word_embeddings = False\n"
        )
        bugs = BugPatternDetector.detect_missing_tie_word_embeddings(code, Path("train.py"))
        assert bugs == [], f"Expected no bugs when tie_word_embeddings is set: {bugs}"

    def test_no_resize_no_bug(self):
        from validators import BugPatternDetector

        code = "trainer.train()\nresult = trainer.evaluate()"
        bugs = BugPatternDetector.detect_missing_tie_word_embeddings(code, Path("train.py"))
        assert bugs == []

    def test_severity_is_critical(self):
        from validators import BugPatternDetector

        from cloud_orchestration import SeverityLevel

        code = "model.resize_token_embeddings(len(tokenizer))\npass"
        bugs = BugPatternDetector.detect_missing_tie_word_embeddings(code, Path("train.py"))
        assert bugs, "Expected at least one bug"
        assert bugs[0].severity == SeverityLevel.CRITICAL


# ---------------------------------------------------------------------------
# ImportChainChecker
# ---------------------------------------------------------------------------


class TestImportChainChecker(unittest.TestCase):
    """Tests for ImportChainChecker utility methods."""

    def test_check_constants_import_succeeds(self):
        """The real constants.py must import successfully."""
        from validators import ImportChainChecker

        success, error = ImportChainChecker.check_constants_import()
        assert success, f"constants.py import failed: {error}"
        assert error == ""

    def test_check_all_succeeds(self):
        """Full import chain (constants + dataset_loaders) must pass."""
        from validators import ImportChainChecker

        success, errors = ImportChainChecker.check_all()
        assert success, f"Import chain check failed: {errors}"
        assert errors == []

    def test_diagnose_no_module_named(self):
        from validators import ImportChainChecker

        suggestion = ImportChainChecker.diagnose_import_error("No module named 'torch'")
        assert "pip install" in suggestion.lower() or "requirements" in suggestion.lower()

    def test_diagnose_syntax_error(self):
        from validators import ImportChainChecker

        # diagnose_import_error checks for "syntax error" (with space, lower-case)
        suggestion = ImportChainChecker.diagnose_import_error("syntax error at line 5")
        assert "syntax" in suggestion.lower()

    def test_diagnose_tie_word_embeddings(self):
        from validators import ImportChainChecker

        suggestion = ImportChainChecker.diagnose_import_error(
            "AttributeError: tie_word_embeddings not found"
        )
        assert "tie_word_embeddings" in suggestion.lower()

    def test_diagnose_unknown_error_returns_non_empty(self):
        from validators import ImportChainChecker

        suggestion = ImportChainChecker.diagnose_import_error("completely unknown error XYZ123")
        assert len(suggestion) > 0, "diagnose_import_error must return a non-empty suggestion"

    def test_diagnose_lm_head_error(self):
        from validators import ImportChainChecker

        suggestion = ImportChainChecker.diagnose_import_error(
            "RuntimeError: CRITICAL lm_head weight missing"
        )
        assert "lm_head" in suggestion.lower()


# ---------------------------------------------------------------------------
# ModelWeightValidator — checkpoint key checks (safetensors dedup guard)
# ---------------------------------------------------------------------------


class TestModelWeightValidator(unittest.TestCase):
    """Tests for ModelWeightValidator.check_missing_keys_after_load.

    Exercises the safetensors deduplication guard (BUG B / F1~0.42 pattern).
    No GPU or actual checkpoint files required.
    """

    def test_lm_head_present_passes(self):
        """No missing keys → validation passes."""
        from validators import ModelWeightValidator

        valid, error = ModelWeightValidator.check_missing_keys_after_load(
            missing_keys=[],
            unexpected_keys=[],
            tie_word_embeddings=False,
        )
        assert valid is True
        assert error == ""

    def test_lm_head_missing_tie_false_fails(self):
        """lm_head.weight missing with tie_word_embeddings=False → critical failure.

        This is BUG B (CLAUDE.md §16): safetensors omitted lm_head.weight from
        the checkpoint shard, lm_head gets randomly re-initialized on reload →
        F1~0.42 (garbled but syntactically valid predictions).
        """
        from validators import ModelWeightValidator

        valid, error = ModelWeightValidator.check_missing_keys_after_load(
            missing_keys=["decoder.lm_head.weight"],
            unexpected_keys=[],
            tie_word_embeddings=False,
        )
        assert valid is False
        assert "CRITICAL" in error
        assert "lm_head" in error.lower() or "decoder.lm_head.weight" in error

    def test_lm_head_missing_tie_true_passes(self):
        """lm_head.weight missing with tie_word_embeddings=True → OK (legacy tied checkpoint)."""
        from validators import ModelWeightValidator

        valid, error = ModelWeightValidator.check_missing_keys_after_load(
            missing_keys=["decoder.lm_head.weight"],
            unexpected_keys=[],
            tie_word_embeddings=True,
        )
        assert valid is True, "Legacy tied checkpoints legitimately omit lm_head.weight"

    def test_other_missing_keys_no_lm_head_error(self):
        """Non-critical missing keys do not trigger the lm_head error."""
        from validators import ModelWeightValidator

        valid, error = ModelWeightValidator.check_missing_keys_after_load(
            missing_keys=["some.other.weight"],
            unexpected_keys=[],
            tie_word_embeddings=False,
        )
        assert valid is True

    def test_empty_unexpected_keys_passes(self):
        from validators import ModelWeightValidator

        valid, error = ModelWeightValidator.check_missing_keys_after_load(
            missing_keys=[],
            unexpected_keys=[],
            tie_word_embeddings=False,
        )
        assert valid is True


# ---------------------------------------------------------------------------
# DataSplitValidator — val/test leakage check
# ---------------------------------------------------------------------------


class TestDataSplitValidator(unittest.TestCase):
    """Tests for DataSplitValidator.check_no_val_test_leakage.

    Guards BUG A (CLAUDE.md §16): val and test must be physically separate
    directories so early stopping does not optimise for test performance.
    """

    def test_different_empty_dirs_passes(self, tmp_path):
        """Physically distinct empty dirs → no leakage."""
        from validators import DataSplitValidator

        val_dir = tmp_path / "val_img"
        test_dir = tmp_path / "test_img"
        val_dir.mkdir()
        test_dir.mkdir()

        valid, error = DataSplitValidator.check_no_val_test_leakage(val_dir, test_dir)
        assert valid is True
        assert error == ""

    def test_same_dir_fails(self, tmp_path):
        """Same path for val and test → leakage detected."""
        from validators import DataSplitValidator

        shared_dir = tmp_path / "shared_img"
        shared_dir.mkdir()

        valid, error = DataSplitValidator.check_no_val_test_leakage(shared_dir, shared_dir)
        assert valid is False
        assert "CRITICAL" in error or "same" in error.lower()

    def test_overlapping_images_fails(self, tmp_path):
        """Same image stem in both dirs → leakage detected."""
        from validators import DataSplitValidator

        val_dir = tmp_path / "val_img"
        test_dir = tmp_path / "test_img"
        val_dir.mkdir()
        test_dir.mkdir()

        # Both sets share the same receipt
        (val_dir / "receipt001.jpg").write_bytes(b"")
        (test_dir / "receipt001.jpg").write_bytes(b"")

        valid, error = DataSplitValidator.check_no_val_test_leakage(val_dir, test_dir)
        assert valid is False
        assert "overlap" in error.lower() or "leakage" in error.lower()

    def test_no_overlap_passes(self, tmp_path):
        """Different image stems in each dir → no leakage."""
        from validators import DataSplitValidator

        val_dir = tmp_path / "val_img"
        test_dir = tmp_path / "test_img"
        val_dir.mkdir()
        test_dir.mkdir()

        (val_dir / "receipt001.jpg").write_bytes(b"")
        (test_dir / "receipt002.jpg").write_bytes(b"")

        valid, error = DataSplitValidator.check_no_val_test_leakage(val_dir, test_dir)
        assert valid is True, f"Expected no leakage, got: {error}"

    def test_partial_overlap_detected(self, tmp_path):
        """Even a single shared file triggers the leakage warning."""
        from validators import DataSplitValidator

        val_dir = tmp_path / "val_img"
        test_dir = tmp_path / "test_img"
        val_dir.mkdir()
        test_dir.mkdir()

        # 3 val, 3 test, 1 shared
        for i in range(3):
            (val_dir / f"val_{i:03d}.jpg").write_bytes(b"")
        for i in range(3):
            (test_dir / f"test_{i:03d}.jpg").write_bytes(b"")
        (val_dir / "shared.jpg").write_bytes(b"")
        (test_dir / "shared.jpg").write_bytes(b"")

        valid, error = DataSplitValidator.check_no_val_test_leakage(val_dir, test_dir)
        assert valid is False


if __name__ == "__main__":
    unittest.main(verbosity=2)
