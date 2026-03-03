"""Type definitions for cloud pipeline."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import json


class SeverityLevel(str, Enum):
    """Bug severity levels."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class CheckStatus(str, Enum):
    """Status of a preflight check."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


class RecoveryAction(str, Enum):
    """Available error recovery actions."""
    FIX_IMPORTS = "fix_imports"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    REINSTALL_DATA = "reinstall_data"
    RELOAD_CHECKPOINT = "reload_checkpoint"
    SKIP_TO_NEXT_STAGE = "skip_to_next_stage"
    COMMIT_PARTIAL = "commit_partial"
    ABORT = "abort"


# ============================================================================
# Preflight & Validation Results
# ============================================================================

@dataclass
class CheckResult:
    """Result of a single preflight check."""
    name: str
    status: CheckStatus
    message: str
    details: Optional[str] = None
    recovery_action: Optional[RecoveryAction] = None


@dataclass
class PreflightReport:
    """Comprehensive preflight validation report."""
    passed: bool
    checks: Dict[str, CheckResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": {k: v.__dict__ for k, v in self.checks.items()},
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# Bug Detection Results
# ============================================================================

@dataclass
class BugPattern:
    """Detected code issue from bug pattern scanner."""
    severity: SeverityLevel
    category: str  # "syntax", "logic", "compatibility"
    file: Path
    line: int
    column: int
    description: str
    code_snippet: str
    fix_suggestion: str


@dataclass
class BugReport:
    """Report of all detected bugs in codebase."""
    bugs: List[BugPattern] = field(default_factory=list)
    total_critical: int = 0
    total_warnings: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "bugs": [
                {
                    "severity": b.severity.value,
                    "category": b.category,
                    "file": str(b.file),
                    "line": b.line,
                    "column": b.column,
                    "description": b.description,
                    "code_snippet": b.code_snippet,
                    "fix_suggestion": b.fix_suggestion,
                }
                for b in self.bugs
            ],
            "total_critical": self.total_critical,
            "total_warnings": self.total_warnings,
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# File Operations Results
# ============================================================================

@dataclass
class FileFix:
    """Result of fixing a single file."""
    file_path: Path
    success: bool
    original_content: str
    fixed_content: Optional[str] = None
    validation_report: Optional[Any] = None
    error: Optional[str] = None


# ============================================================================
# Validation Reports
# ============================================================================

@dataclass
class ValidationReport:
    """Result of validating a code change."""
    passed: bool
    error: Optional[str] = None
    recovery_action: Optional[RecoveryAction] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSplitValidationReport:
    """Result of SROIE data split validation."""
    passed: bool
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Code Repair Results
# ============================================================================

@dataclass
class CodeRepairResult:
    """Result of Mode A (code repair)."""
    success: bool
    files_fixed: int = 0
    files_failed: int = 0
    validation_report: Optional[PreflightReport] = None
    git_commits: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# ML Training & Experiment Results
# ============================================================================

@dataclass
class ExperimentMetrics:
    """Metrics from a single experiment."""
    global_f1: float
    global_precision: float = 0.0
    global_recall: float = 0.0
    overall_exact_match: float = 0.0
    company_f1: float = 0.0
    company_ned: float = 0.0
    date_f1: float = 0.0
    date_ned: float = 0.0
    address_f1: float = 0.0
    address_ned: float = 0.0
    total_f1: float = 0.0
    total_ned: float = 0.0
    parse_failures: int = 0
    total_predictions: int = 0


@dataclass
class ExperimentResult:
    """Result of a single training experiment."""
    experiment_id: int
    name: str
    datasets: List[str]
    num_train_samples: int
    metrics: ExperimentMetrics
    checkpoint_path: Optional[Path] = None
    duration_sec: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "datasets": self.datasets,
            "num_train_samples": self.num_train_samples,
            "metrics": self.metrics.__dict__,
            "duration_sec": self.duration_sec,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AggregatedResults:
    """Aggregation of all experiment results."""
    experiments: List[ExperimentResult] = field(default_factory=list)
    best_experiment: Optional[ExperimentResult] = None
    baseline_f1: float = 0.0
    improvement: float = 0.0
    per_field_analysis: Dict[str, Any] = field(default_factory=dict)
    generated_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExperimentValidationReport:
    """Validation report for experiment results."""
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# Cloud Storage Results
# ============================================================================

@dataclass
class UploadReport:
    """Result of uploading a file to cloud storage."""
    success: bool
    local_path: Path
    remote_path: str
    size_bytes: int = 0
    duration_sec: float = 0.0
    storage_type: str = "local"  # "s3" | "gcs" | "local"
    error: Optional[str] = None


@dataclass
class SyncReport:
    """Result of syncing entire directory to cloud."""
    backend_type: str
    total_files: int = 0
    uploaded: int = 0
    failed: int = 0
    duration_sec: float = 0.0
    details: List[UploadReport] = field(default_factory=list)


# ============================================================================
# Git Results
# ============================================================================

@dataclass
class GitCommitReport:
    """Result of creating a git commit."""
    success: bool
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# ML Training Results
# ============================================================================

@dataclass
class MLTrainingResult:
    """Result of Mode B (ML training)."""
    success: bool
    experiments_run: Dict[int, ExperimentResult] = field(default_factory=dict)
    best_experiment_id: Optional[int] = None
    aggregated_results: Optional[AggregatedResults] = None
    paper_generated: bool = False
    cloud_sync_report: Optional[SyncReport] = None
    git_commits: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_sec: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# Overall Pipeline Results
# ============================================================================

@dataclass
class PipelineResult:
    """Final result of entire pipeline execution."""
    success: bool
    mode: str  # "code_repair" | "ml_training"
    mode_result: Optional[Any] = None  # CodeRepairResult | MLTrainingResult
    duration_sec: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def partial_success(self) -> bool:
        """Check if partial success (some but not all tasks completed)."""
        if isinstance(self.mode_result, MLTrainingResult):
            return len(self.mode_result.experiments_run) > 0
        return False
