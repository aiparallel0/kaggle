"""Tests for the validators package.

Phase 2 — NSA: Signal Intelligence
Tests for BugPatternDetector, ImportChainChecker, ModelWeightValidator, and
DataSplitValidator. All tests run without GPU, model weights, or internet access.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# BugPatternDetector — JSON literal confusion (Pattern 2 from CLAUDE.md §5)
# ---------------------------------------------------------------------------


class TestBugPatternDetectorJsonConfusion:
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
        from pipeline_types import SeverityLevel
        from validators import BugPatternDetector

        bugs = BugPatternDetector.detect_python_json_confusion("x = false", Path("test.py"))
        assert bugs[0].severity == SeverityLevel.CRITICAL


# ---------------------------------------------------------------------------
# BugPatternDetector — Syntax errors
# ---------------------------------------------------------------------------


class TestBugPatternDetectorSyntax:
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


class TestBugPatternDetectorTieWordEmbeddings:
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
        from pipeline_types import SeverityLevel
        from validators import BugPatternDetector

        code = "model.resize_token_embeddings(len(tokenizer))\npass"
        bugs = BugPatternDetector.detect_missing_tie_word_embeddings(code, Path("train.py"))
        assert bugs, "Expected at least one bug"
        assert bugs[0].severity == SeverityLevel.CRITICAL


# ---------------------------------------------------------------------------
# ImportChainChecker
# ---------------------------------------------------------------------------


class TestImportChainChecker:
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


class TestModelWeightValidator:
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


class TestDataSplitValidator:
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
