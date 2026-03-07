"""Tests for pipeline_types dataclasses.

Phase 3 — MI6: Counter-Intelligence
Tests structural correctness of the inter-module dataclass contracts.
These tests catch the class of bug where a refactor renames a field and
the pipeline silently passes None or default values instead of raising.

All tests run without GPU, model weights, or disk access.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_types import (
    AggregatedResults,
    BugPattern,
    BugReport,
    CheckResult,
    CheckStatus,
    DataSplitValidationReport,
    ExperimentMetrics,
    ExperimentResult,
    PipelineResult,
    SeverityLevel,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# SeverityLevel enum
# ---------------------------------------------------------------------------


class TestSeverityLevel:
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


class TestCheckStatus:
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


class TestCheckResult:
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


class TestExperimentMetrics:
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


class TestExperimentResult:
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


class TestBugPattern:
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


class TestBugReport:
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


class TestPipelineResult:
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


class TestValidationReport:
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


class TestAggregatedResults:
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


class TestDataSplitValidationReport:
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
