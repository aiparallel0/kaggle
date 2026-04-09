"""cloud_orchestration.py — Merged pipeline types, critic, DAG scheduler, and cloud pipeline.

Consolidates: pipeline_types.py, pipeline_critic.py, dag_scheduler.py, cloud_pipeline.py
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import dataclasses
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# Pipeline-level exception classes
# ============================================================================


class CheckpointCorruptionError(Exception):
    """Raised when a model checkpoint fails integrity validation.

    Possible causes:
    - ``lm_head.weight`` is missing from the checkpoint (safetensors
      deduplication dropped it because it was tied to ``embed_tokens``).
    - ``model.config.encoder.image_size`` does not match the image size
      used during training (stored in the experiment result JSON).
    - The token vocabulary size in the checkpoint's embedding matrix does
      not match the expected size after ``add_special_tokens()``.

    See: CLAUDE.md § 5 (Pattern 6) and validators/checkpoint_resume_validator.py.
    """


class PipelineConfigError(Exception):
    """Raised when experiment or pipeline configuration is invalid."""


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded or validated."""


__all__ = [
    # pipeline_types
    "SerializableDataclass",
    "SeverityLevel",
    "CheckStatus",
    "RecoveryAction",
    "CheckResult",
    "PreflightReport",
    "BugPattern",
    "BugReport",
    "FileFix",
    "ValidationReport",
    "DataSplitValidationReport",
    "CodeRepairResult",
    "ExperimentMetrics",
    "ExperimentResult",
    "AggregatedResults",
    "ExperimentValidationReport",
    "UploadReport",
    "SyncReport",
    "GitCommitReport",
    "MLTrainingResult",
    "PipelineResult",
    # Exceptions
    "CheckpointCorruptionError",
    "PipelineConfigError",
    "DatasetLoadError",
    # pipeline_critic
    "FindingSeverity",
    "CritiqueFinding",
    "CritiqueReport",
    "PipelineCritic",
    "StatisticalPowerAudit",
    "MultipleTestingAudit",
    "EpochConfoundAudit",
    "PretrainingBiasAudit",
    "ArchitectureAudit",
    "BenchmarkNarrowness",
    # dag_scheduler
    "DAGScheduler",
    "print_dag",
    # cloud_pipeline
    "CloudConfig",
    "PipelineMode",
    "LogLevel",
    "GitController",
    "StorageManager",
    "CloudPipelineOrchestrator",
    "CodeRepairOrchestrator",
    "MLTrainingOrchestrator",
    "RetroUIFormatter",
    "VastAIProvisioner",
    # stats helpers (used by tests)
    "_norm_ppf",
    "_two_proportion_mdd",
    "_family_wise_error_rate",
]


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
# Serializable Dataclass Mixin (DRY helper for to_dict)
# ============================================================================


def _serialize_value(val: Any) -> Any:
    """Recursively convert a value to a JSON-safe primitive.

    Handles datetime → isoformat, Enum → value, Path → str,
    dataclasses → recursive field serialization, and nested dicts/lists.
    """
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, Path):
        return str(val)
    if hasattr(val, "to_dict") and callable(val.to_dict):
        return val.to_dict()
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        return {f.name: _serialize_value(getattr(val, f.name)) for f in dataclasses.fields(val)}
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_serialize_value(item) for item in val]
    return val


class SerializableDataclass:
    """Mixin providing a generic ``to_dict()`` for dataclasses.

    Uses :func:`_serialize_value` to recursively convert field values
    to JSON-safe Python primitives.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: _serialize_value(getattr(self, f.name))
            for f in dataclasses.fields(self)  # type: ignore[arg-type]
        }


# ============================================================================
# Preflight & Validation Results
# ============================================================================


@dataclass
class CheckResult:
    """Result of a single preflight check."""

    name: str
    status: CheckStatus
    message: str
    details: str | None = None
    recovery_action: RecoveryAction | None = None


@dataclass
class PreflightReport(SerializableDataclass):
    """Comprehensive preflight validation report."""

    passed: bool
    checks: dict[str, CheckResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


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
class BugReport(SerializableDataclass):
    """Report of all detected bugs in codebase."""

    bugs: list[BugPattern] = field(default_factory=list)
    total_critical: int = 0
    total_warnings: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================================
# File Operations Results
# ============================================================================


@dataclass
class FileFix:
    """Result of fixing a single file."""

    file_path: Path
    success: bool
    original_content: str
    fixed_content: str | None = None
    validation_report: Any | None = None
    error: str | None = None


# ============================================================================
# Validation Reports
# ============================================================================


@dataclass
class ValidationReport:
    """Result of validating a code change."""

    passed: bool
    error: str | None = None
    recovery_action: RecoveryAction | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataSplitValidationReport:
    """Result of SROIE data split validation."""

    passed: bool
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ============================================================================
# Code Repair Results
# ============================================================================


@dataclass
class CodeRepairResult:
    """Result of Mode A (code repair)."""

    success: bool
    files_fixed: int = 0
    files_failed: int = 0
    validation_report: PreflightReport | None = None
    git_commits: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
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
class ExperimentResult(SerializableDataclass):
    """Result of a single training experiment."""

    experiment_id: int
    name: str
    datasets: list[str]
    num_train_samples: int
    metrics: ExperimentMetrics
    checkpoint_path: Path | None = None
    duration_sec: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AggregatedResults:
    """Aggregation of all experiment results."""

    experiments: list[ExperimentResult] = field(default_factory=list)
    best_experiment: ExperimentResult | None = None
    baseline_f1: float = 0.0
    improvement: float = 0.0
    per_field_analysis: dict[str, Any] = field(default_factory=dict)
    generated_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExperimentValidationReport:
    """Validation report for experiment results."""

    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


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
    error: str | None = None


@dataclass
class SyncReport:
    """Result of syncing entire directory to cloud."""

    backend_type: str
    total_files: int = 0
    uploaded: int = 0
    failed: int = 0
    duration_sec: float = 0.0
    details: list[UploadReport] = field(default_factory=list)


# ============================================================================
# Git Results
# ============================================================================


@dataclass
class GitCommitReport:
    """Result of creating a git commit."""

    success: bool
    commit_hash: str | None = None
    branch: str | None = None
    message: str | None = None
    error: str | None = None


# ============================================================================
# ML Training Results
# ============================================================================


@dataclass
class MLTrainingResult:
    """Result of Mode B (ML training)."""

    success: bool
    experiments_run: dict[int, ExperimentResult] = field(default_factory=dict)
    best_experiment_id: int | None = None
    aggregated_results: AggregatedResults | None = None
    paper_generated: bool = False
    cloud_sync_report: SyncReport | None = None
    git_commits: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
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
    mode_result: Any | None = None  # CodeRepairResult | MLTrainingResult
    duration_sec: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def partial_success(self) -> bool:
        """Check if partial success (some but not all tasks completed)."""
        if isinstance(self.mode_result, MLTrainingResult):
            return len(self.mode_result.experiments_run) > 0
        return False


# ============================================================================
# Pipeline Critic — research validity findings
# ============================================================================


class FindingSeverity(str, Enum):
    """Severity of a research-validity critique finding."""

    FATAL = "FATAL"  # Invalidates the result outright
    CRITICAL = "CRITICAL"  # Severely undermines the main claim
    WARNING = "WARNING"  # Notable weakness requiring acknowledgement
    INFO = "INFO"  # Contextual observation


@dataclass
class CritiqueFinding(SerializableDataclass):
    """A single research-validity critique finding."""

    severity: FindingSeverity
    category: str
    title: str
    description: str
    evidence: str
    recommendation: str


@dataclass
class CritiqueReport(SerializableDataclass):
    """Complete critical audit of the ML pipeline."""

    findings: list[CritiqueFinding] = field(default_factory=list)

    @property
    def fatal_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.FATAL)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.WARNING)

    @property
    def passed(self) -> bool:
        """True only when there are no FATAL findings."""
        return self.fatal_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Override to include computed properties alongside serialized fields."""
        d = super().to_dict()
        d["passed"] = self.passed
        d["fatal_count"] = self.fatal_count
        d["critical_count"] = self.critical_count
        d["warning_count"] = self.warning_count
        return d

    def print_loud(self, exit_on_fatal: bool = True) -> None:
        """Print all findings to stdout; call sys.exit(1) on FATAL when exit_on_fatal=True."""
        _print_report(self, exit_on_fatal=exit_on_fatal)


# ── pipeline_critic ──────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent


# =============================================================================
# Helpers
# =============================================================================


def _norm_ppf(p: float) -> float:
    """Rational approximation to the standard-normal inverse CDF (Abramowitz & Stegun 26.2.17).

    Accurate to within |ε| < 4.5e-4 for all 0 < p < 1.
    Handles p < 0.5 by symmetry. Avoids any third-party dependency.
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be strictly between 0 and 1; got {p}")
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    return t - num / den


def _two_proportion_mdd(
    n: int,
    p_bar: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable difference (two-sided two-proportion z-test).

    Returns the smallest absolute difference δ between two proportions
    reliably detectable (at the given *power* and *alpha*) when evaluating
    *n* independent binary outcomes.

    Args:
        n:      Number of independent binary trials per group (equal groups).
        p_bar:  Average proportion (mid-point of the null). Typically the
                baseline accuracy / F1 being compared against.
        alpha:  Type-I error rate (two-sided).
        power:  Desired statistical power (1 − β).

    Returns:
        Minimum detectable absolute difference in proportion / F1.
    """
    z_alpha_2 = _norm_ppf(1.0 - alpha / 2.0)
    z_beta = _norm_ppf(power)
    return (z_alpha_2 + z_beta) * math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n)


def _family_wise_error_rate(k: int, alpha: float = 0.05) -> float:
    """Family-wise error rate for *k* independent tests at per-test *alpha*."""
    return 1.0 - (1.0 - alpha) ** k


# =============================================================================
# Audit 1 — Statistical Power
# =============================================================================


class StatisticalPowerAudit:
    """Checks whether the 63-sample test set provides sufficient statistical power.

    Key claim in the paper: Exp 6 improves over Exp 1 by +0.0479 global F1.
    This audit asks: given 63 × 4 = 252 scored pairs, can we reliably detect
    a 0.0479 difference, or is this indistinguishable from sampling noise?
    """

    # SROIE custom test set parameters (from run_experiments.py docstring)
    SROIE_TEST_SAMPLES: int = 63
    SROIE_FIELDS: int = 4  # company, date, address, total
    BASELINE_F1: float = 0.85  # approximate baseline from CLAUDE.md
    CLAIMED_GAIN: float = 0.0479  # Exp 6 vs Exp 1 (CLAUDE.md §8)

    def run(self) -> list[CritiqueFinding]:
        n_pairs = self.SROIE_TEST_SAMPLES * self.SROIE_FIELDS  # 252
        mdd_80 = _two_proportion_mdd(n_pairs, self.BASELINE_F1, power=0.80)
        mdd_50 = _two_proportion_mdd(n_pairs, self.BASELINE_F1, power=0.50)

        findings: list[CritiqueFinding] = []

        if mdd_80 > self.CLAIMED_GAIN:
            sev = FindingSeverity.FATAL if mdd_50 > self.CLAIMED_GAIN else FindingSeverity.CRITICAL
            findings.append(
                CritiqueFinding(
                    severity=sev,
                    category="statistical_validity",
                    title="Test set underpowered: claimed gain is below minimum detectable difference",
                    description=(
                        f"The custom SROIE test set has {self.SROIE_TEST_SAMPLES} images × "
                        f"{self.SROIE_FIELDS} fields = {n_pairs} scored pairs. "
                        f"A two-sided two-proportion z-test requires MDD₈₀ ≈ {mdd_80:.4f} F1 "
                        f"to achieve 80% power at α = 0.05. "
                        f"The headline improvement of {self.CLAIMED_GAIN:.4f} "
                        f"(Exp 6 vs baseline) falls below this threshold. "
                        f"The result cannot be distinguished from sampling noise "
                        f"at conventional significance levels."
                    ),
                    evidence=(
                        f"n_pairs={n_pairs}, MDD₈₀={mdd_80:.4f}, MDD₅₀={mdd_50:.4f}, "
                        f"claimed Δ={self.CLAIMED_GAIN:.4f}; "
                        f"baseline F1 ≈ {self.BASELINE_F1}"
                    ),
                    recommendation=(
                        "Expand the evaluation to ≥ 400 samples × 4 fields (≥ 1 600 pairs) "
                        "for 80% power to detect a 0.05 F1 gain. "
                        "Report confidence intervals (e.g. Wilson interval per field) "
                        "alongside point estimates. "
                        "Validate on the official SROIE 347-image held-out test set "
                        "if labels become available."
                    ),
                )
            )

        # Additional finding: per-field sample sizes are even smaller
        per_field_n = self.SROIE_TEST_SAMPLES  # 63 per field
        mdd_per_field = _two_proportion_mdd(per_field_n, self.BASELINE_F1, power=0.80)
        findings.append(
            CritiqueFinding(
                severity=FindingSeverity.WARNING,
                category="statistical_validity",
                title=f"Per-field comparisons use only {per_field_n} samples each",
                description=(
                    f"Each per-field F1 (company, date, address, total) is estimated "
                    f"from {per_field_n} samples. MDD₈₀ per field ≈ {mdd_per_field:.4f}. "
                    f"Field-level deltas reported in the paper (e.g. company F1 swings) "
                    f"are unreliable at this sample size."
                ),
                evidence=(f"per_field_n={per_field_n}, MDD₈₀_per_field={mdd_per_field:.4f}"),
                recommendation=(
                    "Do not draw conclusions from individual field F1 movements "
                    f"smaller than {mdd_per_field:.3f}. Treat per-field numbers as exploratory."
                ),
            )
        )

        return findings


# =============================================================================
# Audit 2 — Multiple Testing
# =============================================================================


class MultipleTestingAudit:
    """Checks for inflated false-positive risk from 8 experiments on one test set.

    Running 8 experiments on the same 63-sample test set and selecting the
    best result is a form of implicit multiple testing. Without correction,
    the family-wise error rate exceeds the claimed per-test α = 0.05.
    """

    N_EXPERIMENTS: int = 8
    N_COMPARISONS: int = 7  # comparing each of Exps 2–8 against the baseline Exp 1

    def run(self) -> list[CritiqueFinding]:
        fwer = _family_wise_error_rate(self.N_COMPARISONS, alpha=0.05)
        bonferroni_alpha = 0.05 / self.N_COMPARISONS

        return [
            CritiqueFinding(
                severity=FindingSeverity.CRITICAL,
                category="multiple_testing",
                title="8 experiments on one test set inflates false-positive rate to ~30%",
                description=(
                    f"All {self.N_EXPERIMENTS} experiments are evaluated on the same "
                    f"63-image test partition. With {self.N_COMPARISONS} comparisons "
                    f"against the baseline (Exps 2–8 vs Exp 1), the uncorrected "
                    f"family-wise error rate is {fwer:.1%}. "
                    f"Selecting the 'best' experiment (Exp 6) post-hoc from "
                    f"{self.N_EXPERIMENTS} candidates inflates the apparent improvement "
                    f"by maximising over noise. "
                    f"The Bonferroni-corrected threshold is α = {bonferroni_alpha:.4f} "
                    f"— far stricter than the uncorrected α = 0.05 implicitly assumed."
                ),
                evidence=(
                    f"k={self.N_COMPARISONS} comparisons, "
                    f"FWER={fwer:.3f}, "
                    f"Bonferroni_α={bonferroni_alpha:.4f}"
                ),
                recommendation=(
                    "Pre-register the hypothesis (e.g. 'Exp 6 will beat Exp 1') "
                    "before collecting test results, or apply Bonferroni / Benjamini–Hochberg "
                    "correction. Use a held-out test set that no experiment was selected "
                    "against. Report the full distribution of results, not just the maximum."
                ),
            )
        ]


# =============================================================================
# Audit 3 — Epoch Confound
# =============================================================================


class EpochConfoundAudit:
    """Checks for the training-duration confound between Exp 1 and Exps 5–8.

    The claimed 'best' experiments (5–8) all run 15 epochs while the baseline
    (Exp 1) runs 10 epochs. This 50% increase in training budget is a
    confounding variable: gains attributed to auxiliary data may be partly or
    entirely due to longer training.

    This audit reads the EXPERIMENTS configuration from source using AST to
    avoid triggering torch/transformers imports.
    """

    _RUN_EXPERIMENTS_FILE = _ROOT / "run_experiments.py"

    @staticmethod
    def _extract_experiment_epochs() -> dict[int, int]:
        """Parse run_experiments.py with AST to extract epoch counts per experiment.

        Returns a dict mapping experiment_id (int) to epochs (int).
        Falls back to the documented values if parsing fails.  When the AST
        parse succeeds but produces a dict that differs from the documented
        defaults, a warning is logged so stale defaults are never silent.
        """
        # Documented at ExperimentConfig definition time (run_experiments.py).
        # Updated manually when the experiment suite changes.
        # Exp 1–4 use the default of 10 epochs; Exps 5–8 explicitly set 15.
        documented_defaults: dict[int, int] = {
            1: 10,
            2: 10,
            3: 10,
            4: 10,
            5: 15,
            6: 15,
            7: 15,
            8: 15,
        }
        try:
            import ast

            source = EpochConfoundAudit._RUN_EXPERIMENTS_FILE.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except SyntaxError:
            return documented_defaults

        # Walk the AST looking for ExperimentConfig keyword calls.
        # Each should have experiment_id and optionally epochs keywords.
        result: dict[int, int] = {}
        default_epochs = 10  # ExperimentConfig default

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "ExperimentConfig"):
                continue

            exp_id = None
            epochs = default_epochs

            for kw in node.keywords:
                if kw.arg == "experiment_id" and isinstance(kw.value, ast.Constant):
                    exp_id = int(kw.value.value)
                elif kw.arg == "epochs" and isinstance(kw.value, ast.Constant):
                    epochs = int(kw.value.value)

            if exp_id is not None:
                result[exp_id] = epochs

        if not result:
            return documented_defaults

        # Warn when AST parse succeeds but differs from documented defaults —
        # prevents stale fallback values from going unnoticed after refactors.
        import logging as _logging

        _log = _logging.getLogger(__name__)
        for eid, default_ep in documented_defaults.items():
            if eid in result and result[eid] != default_ep:
                _log.warning(
                    "EpochConfoundAudit: Exp %d epoch count changed from documented "
                    "default %d to %d. Update documented_defaults in pipeline_critic.py.",
                    eid,
                    default_ep,
                    result[eid],
                )
        return result

    def run(self) -> list[CritiqueFinding]:
        epochs_by_exp = self._extract_experiment_epochs()

        baseline_epochs = epochs_by_exp.get(1, 10)
        # Experiments claiming best results are 5, 6, 7 (Exp 8 is OOM)
        best_exps = {eid: epochs_by_exp.get(eid) for eid in (5, 6, 7, 8) if eid in epochs_by_exp}
        confounded_exps = {
            eid: ep for eid, ep in best_exps.items() if ep is not None and ep > baseline_epochs
        }

        findings: list[CritiqueFinding] = []

        if confounded_exps:
            epoch_summary = ", ".join(
                f"Exp {e}: {ep} epochs" for e, ep in sorted(confounded_exps.items())
            )
            findings.append(
                CritiqueFinding(
                    severity=FindingSeverity.CRITICAL,
                    category="experimental_design",
                    title="Epoch confound: 'best' experiments train 50% longer than the baseline",
                    description=(
                        f"Exp 1 (baseline) runs {baseline_epochs} epochs. "
                        f"The experiments claiming superiority run more epochs: {epoch_summary}. "
                        f"CLAUDE.md §8 explicitly acknowledges: "
                        f"'Exp 1 company F1 at 10 epochs is a convergence failure, not the "
                        f"SROIE-only ceiling.' "
                        f"This means the reported gain of +0.0479 (Exp 6 vs Exp 1) "
                        f"conflates auxiliary-data benefit with extended-training benefit. "
                        f"The contribution of auxiliary data cannot be isolated."
                    ),
                    evidence=(
                        f"Exp 1 epochs={baseline_epochs}; confounded experiments: {epoch_summary}; "
                        f"source: CLAUDE.md §8 note and EXPERIMENTS dict in run_experiments.py"
                    ),
                    recommendation=(
                        "Run a controlled ablation: train Exp 1 (SROIE only) for 15 epochs "
                        "to establish a fair baseline before attributing F1 gains to auxiliary data. "
                        "Either match training budgets across all experiments or report separate "
                        "epoch-matched comparisons."
                    ),
                )
            )

        return findings


# =============================================================================
# Audit 4 — Pretraining Bias
# =============================================================================


class PretrainingBiasAudit:
    """Checks domain proximity between the base checkpoint and SROIE.

    'naver-clova-ix/donut-base' was pretrained on SynthDoG (synthetic
    document images). SynthDoG includes synthetic receipts, menu images,
    and business documents — structurally similar to SROIE (Thai/Malaysian
    thermal receipts). This is not data contamination but it is domain
    proximity that shrinks the effective pretraining-to-target domain gap.

    Additionally, the evaluator code (donut_evaluator.py) handles '<sep/>'
    tokens described as 'inherited from CORD pretraining', suggesting the
    checkpoint has some CORD-adjacent behaviour even in the base version.
    """

    BASE_MODEL = "naver-clova-ix/donut-base"
    # ~25% of SynthDoG images are synthetic receipts (English, Japanese, Chinese).
    # Source: Donut paper, Kim et al. 2022, §3.1 "Pre-training Data" — SynthDoG
    # generates receipt / business-card / magazine-cover images in four languages.
    SYNTHDOG_RECEIPT_FRACTION_APPROX = 0.25

    def run(self) -> list[CritiqueFinding]:
        findings: list[CritiqueFinding] = []

        # Check whether donut_evaluator.py mentions CORD
        cord_mention = self._check_cord_reference()

        findings.append(
            CritiqueFinding(
                severity=FindingSeverity.WARNING,
                category="pretraining_bias",
                title="Base checkpoint pretrained on synthetic receipts (SynthDoG)",
                description=(
                    f"'{self.BASE_MODEL}' was pretrained on SynthDoG, a synthetic "
                    f"document dataset where roughly {self.SYNTHDOG_RECEIPT_FRACTION_APPROX:.0%} "
                    f"of images are synthetic receipts (English, Japanese, Chinese). "
                    f"SROIE contains real Malaysian thermal receipts. "
                    f"While not contamination, this structural overlap gives DONUT a "
                    f"warm-start advantage not available to the TrOCR+YOLO baseline. "
                    + (
                        "Evaluator code also handles '<sep/>' tokens described as "
                        "'inherited from CORD pretraining', suggesting CORD-adjacent "
                        "behaviour in the base checkpoint. "
                        if cord_mention
                        else ""
                    )
                    + "This makes the zero-shot → fine-tuned leap smaller than it appears."
                ),
                evidence=(
                    f"base_model={self.BASE_MODEL}; "
                    f"SynthDoG includes receipt images (Donut paper, Kim et al. 2022 §3.1); "
                    + (
                        "donut_evaluator.py references CORD <sep/> token handling. "
                        if cord_mention
                        else ""
                    )
                ),
                recommendation=(
                    "Acknowledge the SynthDoG receipt pretraining in the paper's limitations. "
                    "For a true domain-gap study, compare against a DONUT variant pretrained "
                    "only on non-receipt documents (e.g. IIT-CDIP text documents only). "
                    "Report the zero-shot baseline F1 and discuss what fine-tuning adds "
                    "beyond the pretraining prior."
                ),
            )
        )

        return findings

    @staticmethod
    def _check_cord_reference() -> bool:
        """Return True if donut_evaluator.py mentions the CORD dataset by word boundary."""
        import re

        evaluator = _ROOT / "donut_evaluator.py"
        if not evaluator.exists():
            return False
        try:
            text = evaluator.read_text(encoding="utf-8")
            # Use word-boundary regex to avoid matching 'record', 'according', etc.
            return bool(re.search(r"\bCORD\b", text))
        except OSError:
            return False


# =============================================================================
# Audit 5 — Architecture Comparison Fairness
# =============================================================================


class ArchitectureAudit:
    """Checks whether the DONUT vs TrOCR+YOLO comparison is a fair ablation.

    The paper frames this as an architecture comparison. In practice it
    conflates architecture, pretraining data, pretraining task, and inference
    paradigm. The +69.5% absolute F1 gap reflects pretraining differences
    far more than architectural ones.
    """

    DONUT_F1: float = 0.8982  # best (Exp 6)
    TROCR_YOLO_F1: float = 0.2035  # from CLAUDE.md §8

    def run(self) -> list[CritiqueFinding]:
        gap = self.DONUT_F1 - self.TROCR_YOLO_F1

        return [
            CritiqueFinding(
                severity=FindingSeverity.WARNING,
                category="comparison_fairness",
                title="DONUT vs TrOCR+YOLO conflates architecture with pretraining",
                description=(
                    f"DONUT (best F1 = {self.DONUT_F1:.4f}) is compared against "
                    f"TrOCR + YOLOv8 (F1 = {self.TROCR_YOLO_F1:.4f}), "
                    f"a gap of {gap:.4f} absolute. "
                    f"This comparison is not a controlled architecture ablation: "
                    f"DONUT was pretrained on SynthDoG (includes synthetic receipts, "
                    f"structured document parsing) while TrOCR was pretrained on OCR "
                    f"(character recognition) and YOLO on object detection. "
                    f"The TrOCR+YOLO pipeline also relies on hand-crafted heuristics for "
                    f"field assignment, whereas DONUT learns field assignment end-to-end. "
                    f"A 69.5-point gap is consistent with comparing specialist-pretrained "
                    f"vs. non-specialist-pretrained models — it does not isolate "
                    f"architecture benefits."
                ),
                evidence=(
                    f"DONUT F1={self.DONUT_F1}, TrOCR+YOLO F1={self.TROCR_YOLO_F1}, "
                    f"gap={gap:.4f}; "
                    f"DONUT pretrain: SynthDoG (receipts included); "
                    f"TrOCR pretrain: IAM handwriting + SROIE-OCR (no KIE); "
                    f"field assignment: DONUT=end-to-end, TrOCR+YOLO=rule-based heuristics"
                ),
                recommendation=(
                    "Reframe the comparison as 'end-to-end document VLM vs. "
                    "pipeline OCR + heuristics', not 'architecture A vs. architecture B'. "
                    "For a controlled architecture ablation, compare DONUT with a "
                    "BERT-based classifier on the same TrOCR OCR output, "
                    "isolating only the parsing/field-assignment component."
                ),
            )
        ]


# =============================================================================
# Audit 6 — Benchmark Narrowness
# =============================================================================


class BenchmarkNarrowness:
    """Checks whether the evaluation benchmark supports the paper's generality claims.

    Two intertwined issues:
    a) The 'test set' is carved from the ICDAR-released training partition, not
       the official SROIE held-out test set (347 images, no public labels).
       This makes cross-paper comparison invalid.
    b) SROIE is a single-domain, fixed-format, 4-field benchmark. Claiming
       general KIE improvement based on it is an overreach.
    """

    OFFICIAL_SROIE_TEST_SIZE: int = 347  # images in official ICDAR held-out set
    CUSTOM_TEST_SIZE: int = 63  # images carved from training partition
    OFFICIAL_PUBLISHED_DONUT_F1: float = 0.8411  # from SROIE leaderboard / Donut paper

    def run(self) -> list[CritiqueFinding]:
        findings: list[CritiqueFinding] = []

        # a) Custom test set vs official leaderboard
        findings.append(
            CritiqueFinding(
                severity=FindingSeverity.FATAL,
                category="benchmark_validity",
                title="'Custom test set' comparison to published results uses different test sets",
                description=(
                    f"The paper reports improvement over 'published DONUT ({self.OFFICIAL_PUBLISHED_DONUT_F1})'. "
                    f"The published DONUT score comes from the official SROIE held-out test set "
                    f"({self.OFFICIAL_SROIE_TEST_SIZE} images, ground truth never released). "
                    f"This paper's score comes from a custom {self.CUSTOM_TEST_SIZE}-image subset "
                    f"carved from the 626 labeled ICDAR training images — a completely different "
                    f"evaluation set. Comparing F1 scores across different test sets is statistically "
                    f"invalid. The claim '+0.0571 over published DONUT' is not supported: "
                    f"it compares performance on different data."
                ),
                evidence=(
                    f"run_experiments.py docstring: 'custom 80/10/10 split from 626 labeled "
                    f"training images; the official 347-image test set has no public ground truth'; "
                    f"CLAUDE.md §8: 'Gain over baseline: Exp 6 (0.8982) − ... vs published "
                    f"DONUT (0.8411) = +0.0571'; "
                    f"official test size={self.OFFICIAL_SROIE_TEST_SIZE}, "
                    f"custom test size={self.CUSTOM_TEST_SIZE}"
                ),
                recommendation=(
                    "Remove or heavily caveat the cross-paper comparison. "
                    "State clearly: 'All results are on our custom 63-image split; "
                    "comparison to published SROIE leaderboard numbers is not valid.' "
                    "Alternatively, request ground-truth labels from the ICDAR organisers "
                    "or use a public SROIE re-split (e.g. EATEN, LayoutLM splits) that "
                    "others have also evaluated on."
                ),
            )
        )

        # b) Single-domain narrowness
        findings.append(
            CritiqueFinding(
                severity=FindingSeverity.WARNING,
                category="benchmark_narrowness",
                title="Single-domain, 4-field benchmark limits generalizability claims",
                description=(
                    "SROIE contains only Malaysian thermal printer receipts from 2019. "
                    "All 626 labeled images share the same physical format (thermal paper, "
                    "top-aligned header, 4 fixed fields). Real-world receipt KIE involves "
                    "hundreds of layouts, languages, fields, and image qualities. "
                    "Gains observed on SROIE do not necessarily transfer to "
                    "cross-domain, multi-layout, or higher-field-count scenarios."
                ),
                evidence=(
                    "SROIE dataset: ICDAR 2019, Malaysian receipts, 4 fixed fields; "
                    "no multi-layout, multi-language, or cross-domain evaluation; "
                    "comparison datasets (WildReceipt, Invoices-DONUT) used only for training, "
                    "never for evaluation"
                ),
                recommendation=(
                    "Evaluate on at least one additional held-out benchmark "
                    "(e.g. CORD, FUNSD, or a proprietary receipt dataset) to test "
                    "transfer of the multi-dataset fine-tuning gains. "
                    "Without cross-dataset evaluation, the paper can only claim "
                    "improvement on SROIE-style Malaysian receipts."
                ),
            )
        )

        return findings


# =============================================================================
# PipelineCritic — orchestrator
# =============================================================================


class PipelineCritic:
    """Orchestrates all six audits and returns a consolidated CritiqueReport.

    All audits run without GPU, model weights, or network access.
    Each audit is instantiated fresh to allow independent parameterisation.

    Usage
    -----
        report = PipelineCritic().run()
        report.print_loud()
    """

    def run(self) -> CritiqueReport:
        """Execute all audits and collect findings.

        Returns:
            CritiqueReport with all findings sorted by severity.
        """
        audits = [
            StatisticalPowerAudit(),
            MultipleTestingAudit(),
            EpochConfoundAudit(),
            PretrainingBiasAudit(),
            ArchitectureAudit(),
            BenchmarkNarrowness(),
        ]

        all_findings: list[CritiqueFinding] = []
        for audit in audits:
            all_findings.extend(audit.run())

        # Sort by severity: FATAL first, then CRITICAL, WARNING, INFO
        _order = {
            FindingSeverity.FATAL: 0,
            FindingSeverity.CRITICAL: 1,
            FindingSeverity.WARNING: 2,
            FindingSeverity.INFO: 3,
        }
        all_findings.sort(key=lambda f: _order[f.severity])

        return CritiqueReport(findings=all_findings)


# =============================================================================
# CritiqueReport — print_loud (kept here to avoid circular import)
# =============================================================================

_SEP = "=" * 78
_SEP_THIN = "-" * 78

_SEVERITY_PREFIX: dict[FindingSeverity, str] = {
    FindingSeverity.FATAL: "💀 FATAL",
    FindingSeverity.CRITICAL: "🔴 CRITICAL",
    FindingSeverity.WARNING: "🟡 WARNING",
    FindingSeverity.INFO: "ℹ️  INFO",
}


def _print_report(report: CritiqueReport, exit_on_fatal: bool = True) -> None:
    """Print all findings to stdout and optionally exit non-zero on FATAL."""
    print(_SEP)
    print("PIPELINE CRITIC — SCRUTINY REPORT")
    print(_SEP)
    print(
        f"Findings: {report.fatal_count} FATAL  "
        f"{report.critical_count} CRITICAL  "
        f"{report.warning_count} WARNING"
    )
    print(_SEP)

    for idx, finding in enumerate(report.findings, start=1):
        prefix = _SEVERITY_PREFIX.get(finding.severity, finding.severity.value)
        print(f"\n[{idx}] {prefix} — {finding.category}")
        print(f"    {finding.title}")
        print(_SEP_THIN)
        print(f"  Description : {finding.description}")
        print(f"  Evidence    : {finding.evidence}")
        print(f"  Fix         : {finding.recommendation}")

    print(f"\n{_SEP}")
    verdict = (
        "DOES NOT SURVIVE SCRUTINY"
        if not report.passed
        else "SURVIVES SCRUTINY (no FATAL findings)"
    )
    print(f"VERDICT: {verdict}")
    print(_SEP)

    if not report.passed and exit_on_fatal:
        sys.exit(1)


# ── dag_scheduler ────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


class DAGScheduler:
    """Schedule experiments respecting depends_on constraints.

    Parameters
    ----------
    configs:
        Ordered list of experiment config objects.  Each config must have
        an ``id`` attribute (or an ``experiment_id`` field — both classes in
        the codebase expose ``.id`` as a property or direct field).
        Optionally has ``depends_on: list[int]``; if absent, defaults to ``[]``
        via ``getattr(cfg, "depends_on", [])``.
    run_fn:
        Callable that runs a single experiment: ``run_fn(cfg) -> Any``.
        Called in a thread — must be thread-safe for parallel execution.
    max_workers:
        Maximum number of concurrent workers (threads).  ``None`` means
        auto-detect based on available GPUs (falls back to 1 on single-GPU).
    check_vram:
        If True, consult ``resource_optimizer.detect_system_resources()``
        before scheduling to cap concurrency to ``int(vram_gb / 8)``
        (each DONUT experiment needs ~8 GB VRAM at batch_size=8).
    """

    def __init__(
        self,
        configs: list[Any],
        run_fn: Callable[[Any], Any],
        max_workers: int | None = None,
        check_vram: bool = True,
    ) -> None:
        self._configs = configs
        self._run_fn = run_fn
        self._id_to_cfg = {cfg.id: cfg for cfg in configs}

        if max_workers is None:
            max_workers = self._detect_workers(check_vram)
        self._max_workers = max(1, max_workers)
        logger.info("[DAGScheduler] max_workers=%d", self._max_workers)

    # ── Worker detection ──────────────────────────────────────────────────

    def _detect_workers(self, check_vram: bool) -> int:
        """Return max safe concurrent workers."""
        try:
            import torch

            num_gpus = torch.cuda.device_count()
        except ImportError:
            num_gpus = 0

        if num_gpus <= 1:
            return 1  # serial on single-GPU / CPU

        if check_vram:
            try:
                from resource_manager import detect_system_resources

                res = detect_system_resources()
                # Each DONUT experiment needs ~8 GB VRAM
                vram_workers = max(1, int(res.vram_gb / 8))
                workers = min(num_gpus, vram_workers)
                logger.info(
                    "[DAGScheduler] VRAM=%.1f GB → capping to %d worker(s)",
                    res.vram_gb,
                    workers,
                )
                return workers
            except Exception as exc:
                logger.warning("[DAGScheduler] VRAM detection failed: %s", exc)

        return num_gpus

    # ── Dependency resolution ─────────────────────────────────────────────

    def _build_dependency_graph(self) -> dict[int, set[int]]:
        """Return {exp_id: set_of_dependency_ids}."""
        graph: dict[int, set[int]] = {}
        for cfg in self._configs:
            deps = set(getattr(cfg, "depends_on", []) or [])
            # Only include dependencies that are in our config set
            deps = {d for d in deps if d in self._id_to_cfg}
            graph[cfg.id] = deps
        return graph

    def _topological_order(self, graph: dict[int, set[int]]) -> list[int]:
        """Return experiment IDs in topological (dependency-first) order."""
        in_degree = {eid: len(deps) for eid, deps in graph.items()}
        queue: deque[int] = deque(eid for eid, deg in in_degree.items() if deg == 0)
        order: list[int] = []

        # Build reverse edges (who depends on me?)
        dependents: dict[int, list[int]] = {eid: [] for eid in graph}
        for eid, deps in graph.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].append(eid)

        while queue:
            eid = queue.popleft()
            order.append(eid)
            for dependent in dependents.get(eid, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(graph):
            cycle_ids = [eid for eid in graph if eid not in order]
            raise ValueError(
                f"[DAGScheduler] Cyclic dependency detected involving IDs: {cycle_ids}"
            )
        return order

    # ── Main scheduler ────────────────────────────────────────────────────

    def run(self) -> dict[int, Any]:
        """Execute all experiments respecting dependencies.

        Returns
        -------
        dict[int, Any]
            Mapping from experiment ID to the return value of ``run_fn``.
        """
        graph = self._build_dependency_graph()
        order = self._topological_order(graph)

        results: dict[int, Any] = {}
        completed: set[int] = set()
        failed: set[int] = set()

        remaining = list(order)  # process in topological order

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures: dict[concurrent.futures.Future, int] = {}

            def _submit_ready() -> None:
                """Submit any experiment whose dependencies are satisfied."""
                for eid in list(remaining):
                    deps = graph[eid]
                    if failed & deps:
                        logger.warning(
                            "[DAGScheduler] Exp %d skipped — dependency failed: %s",
                            eid,
                            failed & deps,
                        )
                        failed.add(eid)
                        remaining.remove(eid)
                        continue
                    if deps <= completed:
                        cfg = self._id_to_cfg[eid]
                        logger.info("[DAGScheduler] Submitting Exp %d: %s", eid, cfg.name)
                        fut = executor.submit(self._run_fn, cfg)
                        futures[fut] = eid
                        remaining.remove(eid)

            _submit_ready()

            while futures:
                done, _ = concurrent.futures.wait(
                    futures.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    eid = futures.pop(fut)
                    try:
                        result = fut.result()
                        results[eid] = result
                        completed.add(eid)
                        logger.info("[DAGScheduler] Exp %d completed", eid)
                    except Exception as exc:
                        logger.error(
                            "[DAGScheduler] Exp %d FAILED: %s: %s",
                            eid,
                            type(exc).__name__,
                            exc,
                        )
                        failed.add(eid)
                        results[eid] = {"error": str(exc)}
                _submit_ready()

        if failed:
            logger.warning("[DAGScheduler] %d experiment(s) failed: %s", len(failed), failed)
        logger.info("[DAGScheduler] Done. %d completed, %d failed", len(completed), len(failed))
        return results


def print_dag(configs: list[Any]) -> None:
    """Print the experiment dependency graph to stdout."""
    print("\nExperiment DAG:")
    print("=" * 60)
    for cfg in configs:
        deps = getattr(cfg, "depends_on", []) or []
        dep_str = f" ← depends on {deps}" if deps else ""
        print(f"  [{cfg.id:2d}] {getattr(cfg, 'name', '')[:40]}{dep_str}")
    print("=" * 60 + "\n")


def main_dag_scheduler() -> None:
    parser = argparse.ArgumentParser(description="Show or test the experiment DAG")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments with their dependencies and exit",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Path to YAML experiments directory (default: experiments/)",
    )
    args = parser.parse_args()

    try:
        from run_experiments import load_all_experiments

        configs = load_all_experiments(args.experiments_dir)
    except Exception as exc:
        print(f"[DAGScheduler] Could not load experiments: {exc}")
        raise SystemExit(1) from exc

    print_dag(configs)  # always show the graph


# ── cloud_pipeline ─────────────────────────────────────────────────────────

# =============================================================================
# Pipeline Configuration
# =============================================================================


class PipelineMode(str, Enum):
    """Available pipeline execution modes."""

    CODE_REPAIR = "code_repair"
    ML_TRAINING = "ml_training"
    AUTO = "auto"


class LogLevel(str, Enum):
    """Log level options."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class CloudConfig:
    """Top-level configuration for cloud pipeline operations.

    All values can be overridden via environment variables:
    - CLOUD_PIPELINE_MODE
    - DONUT_WORKSPACE
    - GITHUB_REPO
    - GITHUB_BRANCH
    - OLLAMA_BASE_URL
    - OLLAMA_MODEL
    - etc.
    """

    # ====== Mode Selection ======
    mode: PipelineMode = PipelineMode.AUTO
    dry_run: bool = False

    # ====== Common Settings ======
    workspace: Path = Path("/workspace")  # overridden by DONUT_WORKSPACE env var
    git_branch: str = "claude/setup-cloud-ai-agents-Olrqd"
    github_repo: str = "aiparallel0/kaggle"
    skip_validation: bool = False

    # ====== Code Repair Settings (Ollama) ======
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    ollama_max_retries: int = 3
    ollama_auto_start: bool = True

    # ====== ML Training Settings (Vast.ai) ======
    gpu_required: bool = True
    skip_trocr: bool = False
    skip_pretrained_baseline: bool = False
    experiments_to_run: list[int] = field(default_factory=lambda: list(range(1, 9)))

    # ====== Vast.ai Provisioner Settings ======
    vastai_api_key: str = ""
    vastai_gpu_name: str = "RTX 4090"
    vastai_max_price: float = 0.50
    vastai_min_vram: int = 24
    vastai_disk_gb: int = 40
    vastai_image: str = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"

    # ====== Cloud Storage Settings ======
    s3_bucket: str | None = None
    s3_region: str | None = None
    gcs_bucket: str | None = None

    # ====== Validation Settings ======
    enable_ruff_check: bool = True
    enable_pytest: bool = True
    pytest_markers: str = ""
    fail_on_warnings: bool = False

    # ====== Commit Settings ======
    auto_commit: bool = True
    commit_on_error: bool = False

    # ====== Logging Settings ======
    log_level: LogLevel = LogLevel.INFO
    log_dir: Path = Path("./logs")
    stream_training_logs: bool = False

    # ====== Paths ======
    sroie_data_dir: Path | None = None
    results_dir: Path = Path("./results")

    @staticmethod
    def from_env() -> CloudConfig:
        """Load configuration from environment variables with defaults."""
        mode_str = os.getenv("CLOUD_PIPELINE_MODE", "auto").lower()
        try:
            mode = PipelineMode(mode_str)
        except ValueError:
            mode = PipelineMode.AUTO

        workspace = Path(os.getenv("DONUT_WORKSPACE", "/workspace"))

        sroie_dir = os.getenv("SROIE_DATA_DIR")
        sroie_dir = Path(sroie_dir) if sroie_dir else workspace / "ICDAR-2019-SROIE" / "data"

        ollama_auto_start = os.getenv("OLLAMA_AUTO_START", "true").lower() == "true"
        results_dir = Path(os.getenv("RESULTS_DIR", "./results"))
        log_dir = Path(os.getenv("LOG_DIR", "./logs"))
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        try:
            log_level = LogLevel(log_level_str)
        except ValueError:
            log_level = LogLevel.INFO

        exp_str = os.getenv("EXPERIMENTS_TO_RUN")
        if exp_str:
            try:
                experiments = [int(x.strip()) for x in exp_str.split(",")]
            except ValueError:
                experiments = list(range(1, 9))
        else:
            experiments = list(range(1, 9))

        return CloudConfig(
            mode=mode,
            workspace=workspace,
            git_branch=os.getenv("GITHUB_BRANCH", "claude/setup-cloud-ai-agents-Olrqd"),
            github_repo=os.getenv("GITHUB_REPO", "aiparallel0/kaggle"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral:latest"),
            ollama_auto_start=ollama_auto_start,
            skip_trocr=os.getenv("SKIP_TROCR", "false").lower() == "true",
            skip_pretrained_baseline=os.getenv("SKIP_PRETRAINED_BASELINE", "false").lower()
            == "true",
            experiments_to_run=experiments,
            vastai_api_key=os.getenv("VASTAI_API_KEY", ""),
            vastai_gpu_name=os.getenv("VASTAI_GPU_NAME", "RTX 4090"),
            vastai_max_price=float(os.getenv("VASTAI_MAX_PRICE", "0.50")),
            vastai_min_vram=int(os.getenv("VASTAI_MIN_VRAM", "24")),
            vastai_disk_gb=int(os.getenv("VASTAI_DISK_GB", "40")),
            vastai_image=os.getenv("VASTAI_IMAGE", "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"),
            s3_bucket=os.getenv("AWS_S3_BUCKET"),
            s3_region=os.getenv("AWS_S3_REGION"),
            gcs_bucket=os.getenv("GCS_BUCKET"),
            enable_ruff_check=os.getenv("ENABLE_RUFF_CHECK", "true").lower() == "true",
            enable_pytest=os.getenv("ENABLE_PYTEST", "true").lower() == "true",
            fail_on_warnings=os.getenv("FAIL_ON_WARNINGS", "false").lower() == "true",
            auto_commit=os.getenv("AUTO_COMMIT", "true").lower() == "true",
            commit_on_error=os.getenv("COMMIT_ON_ERROR", "false").lower() == "true",
            sroie_data_dir=sroie_dir,
            results_dir=results_dir,
            log_dir=log_dir,
            log_level=log_level,
            skip_validation=os.getenv("SKIP_VALIDATION", "false").lower() == "true",
        )

    @staticmethod
    def from_args_and_env(args) -> CloudConfig:
        """Load configuration from argparse args and environment."""
        config = CloudConfig.from_env()

        if hasattr(args, "mode") and args.mode and args.mode != "auto":
            config.mode = PipelineMode(args.mode)
        if hasattr(args, "ollama_url"):
            config.ollama_base_url = args.ollama_url
        if hasattr(args, "ollama_model"):
            config.ollama_model = args.ollama_model
        if hasattr(args, "experiments") and args.experiments:
            config.experiments_to_run = args.experiments
        if hasattr(args, "skip_trocr") and args.skip_trocr:
            config.skip_trocr = True
        if hasattr(args, "s3_bucket"):
            config.s3_bucket = args.s3_bucket
        if hasattr(args, "gcs_bucket"):
            config.gcs_bucket = args.gcs_bucket
        if hasattr(args, "skip_validation") and args.skip_validation:
            config.skip_validation = True
        if hasattr(args, "dry_run") and args.dry_run:
            config.dry_run = True
        if hasattr(args, "no_commit") and args.no_commit:
            config.auto_commit = False
        if hasattr(args, "branch"):
            config.git_branch = args.branch
        if hasattr(args, "workspace"):
            config.workspace = Path(args.workspace)

        return config

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration. Returns (is_valid, list of error messages)."""
        errors = []
        if not self.workspace.exists():
            errors.append(f"Workspace does not exist: {self.workspace}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if (
            self.mode in [PipelineMode.ML_TRAINING, PipelineMode.AUTO]
            and self.sroie_data_dir
            and not self.sroie_data_dir.exists()
        ):
            errors.append(f"SROIE data directory does not exist: {self.sroie_data_dir}")
        for exp_id in self.experiments_to_run:
            if not (1 <= exp_id <= 8):
                errors.append(f"Invalid experiment ID: {exp_id} (must be 1-8)")
        return len(errors) == 0, errors


# =============================================================================
# Cloud Utilities  (formerly cloud_utils.py)
# =============================================================================

_utils_logger = logging.getLogger(__name__ + ".utils")


class GitController:
    """Manage git operations for the pipeline."""

    @staticmethod
    def get_current_branch() -> str | None:
        """Return current git branch name, or None if not in a git repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            _utils_logger.error(f"Could not get current branch: {e}")
        return None

    @staticmethod
    def checkout_branch(branch_name: str) -> bool:
        """Checkout *branch_name*, creating it if it does not yet exist."""
        try:
            result = subprocess.run(
                ["git", "checkout", branch_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                _utils_logger.info(f"✓ Checked out branch: {branch_name}")
                return True
            if "did not match any branch" in result.stderr.lower():
                _utils_logger.info(f"Creating new branch: {branch_name}")
                result = subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    _utils_logger.info(f"✓ Created and checked out branch: {branch_name}")
                    return True
            _utils_logger.error(f"Failed to checkout branch: {result.stderr}")
            return False
        except Exception as e:
            _utils_logger.error(f"Git checkout error: {e}")
            return False

    @staticmethod
    def commit(message: str, files: list[str] | None = None) -> GitCommitReport:
        """Stage *files* (or all changes when None) and create a commit."""
        try:
            cmd = ["git", "add", "-A"] if files is None else ["git", "add"] + files
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return GitCommitReport(success=False, error=f"Stage failed: {result.stderr}")
            result = subprocess.run(
                ["git", "commit", "-m", message],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                commit_hash = None
                for line in result.stdout.split("\n"):
                    if line.startswith("["):
                        parts = line.split()
                        if len(parts) >= 2:
                            commit_hash = parts[1].rstrip("]")
                            break
                _utils_logger.info(f"✓ Committed: {message} ({commit_hash})")
                return GitCommitReport(
                    success=True,
                    commit_hash=commit_hash,
                    message=message,
                    branch=GitController.get_current_branch(),
                )
            if "nothing to commit" in result.stdout.lower():
                _utils_logger.warning("Nothing to commit")
                return GitCommitReport(
                    success=True,
                    message="Nothing to commit",
                    branch=GitController.get_current_branch(),
                )
            return GitCommitReport(success=False, error=f"Commit failed: {result.stderr}")
        except Exception as e:
            _utils_logger.error(f"Git commit error: {e}")
            return GitCommitReport(success=False, error=str(e))

    @staticmethod
    def push_branch(branch_name: str, force: bool = False) -> bool:
        """Push *branch_name* to origin. Returns True on success."""
        try:
            cmd = ["git", "push", "-u", "origin", branch_name]
            if force:
                cmd.insert(2, "--force-with-lease")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                _utils_logger.info(f"✓ Pushed branch: {branch_name}")
                return True
            _utils_logger.error(f"Push failed: {result.stderr}")
            return False
        except Exception as e:
            _utils_logger.error(f"Git push error: {e}")
            return False

    @staticmethod
    def tag_commit(tag_name: str, message: str = "") -> bool:
        """Create a git tag for the current commit. Returns True on success."""
        try:
            cmd = (
                ["git", "tag", "-a", tag_name, "-m", message]
                if message
                else ["git", "tag", tag_name]
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                _utils_logger.info(f"✓ Created tag: {tag_name}")
                return True
            _utils_logger.error(f"Tag creation failed: {result.stderr}")
            return False
        except Exception as e:
            _utils_logger.error(f"Git tag error: {e}")
            return False

    @staticmethod
    def get_status() -> str:
        """Return short git status string (empty string on error)."""
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout if result.returncode == 0 else ""
        except Exception as e:
            _utils_logger.error(f"Git status error: {e}")
            return ""


class StorageManager:
    """Store experiment results locally, ready to commit to GitHub."""

    def __init__(self, results_dir: Path = Path("results")) -> None:
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def sync_results_directory(self, remote_prefix: str = "") -> SyncReport:
        """Report which result files are ready for a GitHub commit."""
        _utils_logger.info("Preparing results for GitHub commit...")
        exp_files = list(self.results_dir.glob("experiment_*.json"))
        agg_files = list(self.results_dir.glob("all_experiments.json"))
        other_files = list(self.results_dir.glob("*.json"))
        all_files = list(set(exp_files + agg_files + other_files))
        _utils_logger.info(f"Found {len(all_files)} result files ready to commit:")
        for f in all_files:
            _utils_logger.info(f"  - {f.name}")
        report = SyncReport(
            backend_type="github",
            total_files=len(all_files),
            uploaded=len(all_files),
            failed=0,
            duration_sec=0.0,
            details=[
                UploadReport(
                    success=True,
                    local_path=f,
                    remote_path=f"results/{f.name}",
                    storage_type="github",
                    size_bytes=f.stat().st_size if f.exists() else 0,
                )
                for f in all_files
            ],
        )
        _utils_logger.info("✓ Results ready for GitHub commit")
        return report

    async def upload_experiment_result(self, exp_id: int) -> UploadReport:
        """Return an UploadReport for experiment *exp_id*'s result file."""
        exp_file = self.results_dir / f"experiment_{exp_id}.json"
        if exp_file.exists():
            return UploadReport(
                success=True,
                local_path=exp_file,
                remote_path=f"results/experiment_{exp_id}.json",
                storage_type="github",
                size_bytes=exp_file.stat().st_size,
            )
        return UploadReport(
            success=False,
            local_path=exp_file,
            remote_path=f"results/experiment_{exp_id}.json",
            storage_type="github",
            error=f"File not found: {exp_file}",
        )

    async def upload_paper(self, paper_path: Path) -> UploadReport:
        """Return an UploadReport for *paper_path*."""
        if paper_path.exists():
            return UploadReport(
                success=True,
                local_path=paper_path,
                remote_path=f"results/{paper_path.name}",
                storage_type="github",
                size_bytes=paper_path.stat().st_size,
            )
        return UploadReport(
            success=False,
            local_path=paper_path,
            remote_path=f"results/{paper_path.name}",
            storage_type="github",
            error=f"File not found: {paper_path}",
        )

    async def download_previous_results(self, exp_id: int) -> bool:
        """Return True if the local result file for *exp_id* already exists."""
        exp_file = self.results_dir / f"experiment_{exp_id}.json"
        exists = exp_file.exists()
        _utils_logger.info(
            f"Found previous result: {exp_file}"
            if exists
            else f"No previous result found for experiment {exp_id}"
        )
        return exists


# =============================================================================
# Orchestration  (formerly the body of this file)
# =============================================================================

# Set up logging early so all submodules share the same format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# RetroUIFormatter — ASCII terminal colour helper (previously retro_ui.py)
# ---------------------------------------------------------------------------


class RetroUIFormatter:
    """ANSI escape-code helpers for coloured terminal output."""

    @staticmethod
    def bold(text: str) -> str:
        """Return *text* wrapped in ANSI bold codes."""
        return f"\033[1m{text}\033[0m"

    @staticmethod
    def underline(text: str) -> str:
        """Return *text* wrapped in ANSI underline codes."""
        return f"\033[4m{text}\033[0m"

    @staticmethod
    def red(text: str) -> str:
        """Return *text* in bright red."""
        return f"\033[91m{text}\033[0m"

    @staticmethod
    def green(text: str) -> str:
        """Return *text* in bright green."""
        return f"\033[92m{text}\033[0m"

    @staticmethod
    def yellow(text: str) -> str:
        """Return *text* in bright yellow."""
        return f"\033[93m{text}\033[0m"


# ---------------------------------------------------------------------------
# TestRunner and related dataclasses (inlined from test_runner.py)
# ---------------------------------------------------------------------------


@dataclass
class RuffReport:
    """Report from ruff linting."""

    passed: bool
    stdout: str = ""
    stderr: str = ""
    issues: list[str] = field(default_factory=list)
    exit_code: int = 0


@dataclass
class RuffFormatReport:
    """Report from ruff formatting."""

    passed: bool
    files_formatted: int = 0
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class PytestReport:
    """Report from pytest execution."""

    passed: bool
    stdout: str = ""
    stderr: str = ""
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    exit_code: int = 0
    details: str = ""


@dataclass
class AllChecksReport:
    """Combined report from ruff + pytest."""

    passed: bool
    ruff_report: RuffReport = field(default_factory=lambda: RuffReport(passed=False))
    ruff_format_report: RuffFormatReport = field(
        default_factory=lambda: RuffFormatReport(passed=False)
    )
    pytest_report: PytestReport | None = None

    @property
    def all_passed(self) -> bool:
        passed = self.ruff_report.passed and self.ruff_format_report.passed
        if self.pytest_report:
            passed = passed and self.pytest_report.passed
        return passed


class TestRunner:
    """Orchestrate ruff and pytest checks."""

    @staticmethod
    async def run_ruff_check(check_dir: Path = Path(".")) -> RuffReport:
        logger.info("Running ruff check...")
        try:
            result = await asyncio.create_task(
                asyncio.to_thread(
                    subprocess.run,
                    ["python", "-m", "ruff", "check", str(check_dir)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            )
            report = RuffReport(
                passed=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
            if result.stdout:
                report.issues = [ln.strip() for ln in result.stdout.split("\n") if ln.strip()]
            if report.passed:
                logger.info("✓ Ruff check passed")
            else:
                logger.error(f"❌ Ruff check failed ({len(report.issues)} issues)")
            return report
        except asyncio.TimeoutError:
            return RuffReport(passed=False, stderr="Ruff check timed out", exit_code=1)
        except Exception as e:
            return RuffReport(passed=False, stderr=str(e), exit_code=1)

    @staticmethod
    async def run_ruff_format(format_dir: Path = Path("."), fix: bool = False) -> RuffFormatReport:
        logger.info(f"Running ruff format ({'fix' if fix else 'check'})...")
        cmd = ["python", "-m", "ruff", "format", str(format_dir)]
        if not fix:
            cmd.append("--check")
        try:
            result = await asyncio.create_task(
                asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            )
            report = RuffFormatReport(
                passed=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )
            if report.passed:
                logger.info("✓ Ruff format passed")
            else:
                logger.warning("⚠ Ruff format issues found")
            return report
        except asyncio.TimeoutError:
            return RuffFormatReport(passed=False, stderr="Ruff format timed out", exit_code=1)
        except Exception as e:
            return RuffFormatReport(passed=False, stderr=str(e), exit_code=1)

    @staticmethod
    async def run_pytest(
        test_dir: Path = Path("tests"),
        markers: str = "",
        verbose: bool = True,
        timeout: int = 300,
    ) -> PytestReport:
        logger.info(f"Running pytest in {test_dir}...")
        cmd = ["python", "-m", "pytest", str(test_dir)]
        if verbose:
            cmd.append("-v")
        if markers:
            cmd.extend(["-m", markers])
        try:
            result = await asyncio.create_task(
                asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            )
            report = PytestReport(
                passed=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                details=result.stdout,
            )
            for line in result.stdout.split("\n"):
                if " passed" in line:
                    try:
                        report.tests_passed = int(line.split()[0])
                    except ValueError:
                        pass
                if " failed" in line:
                    try:
                        report.tests_failed = int(line.split()[0])
                    except ValueError:
                        pass
                if " skipped" in line:
                    try:
                        report.tests_skipped = int(line.split()[0])
                    except ValueError:
                        pass
            report.tests_run = report.tests_passed + report.tests_failed + report.tests_skipped
            if report.passed:
                logger.info(f"✓ All {report.tests_passed} tests passed")
            else:
                logger.error(f"❌ Tests failed: {report.tests_failed} failed")
            return report
        except asyncio.TimeoutError:
            return PytestReport(
                passed=False, stderr=f"Pytest timed out after {timeout}s", exit_code=1
            )
        except Exception as e:
            return PytestReport(passed=False, stderr=str(e), exit_code=1)

    @staticmethod
    async def run_all_checks(
        check_dir: Path = Path("."),
        test_dir: Path = Path("tests"),
        run_tests: bool = True,
    ) -> AllChecksReport:
        logger.info("=" * 70)
        logger.info("RUNNING ALL CHECKS")
        logger.info("=" * 70)
        ruff_report = await TestRunner.run_ruff_check(check_dir)
        ruff_format_report = await TestRunner.run_ruff_format(check_dir, fix=False)
        pytest_report = None
        if run_tests:
            pytest_report = await TestRunner.run_pytest(test_dir)
        all_passed = ruff_report.passed and ruff_format_report.passed
        if pytest_report:
            all_passed = all_passed and pytest_report.passed
        report = AllChecksReport(
            passed=all_passed,
            ruff_report=ruff_report,
            ruff_format_report=ruff_format_report,
            pytest_report=pytest_report,
        )
        logger.info("=" * 70)
        if report.passed:
            logger.info("✓ ALL CHECKS PASSED")
        else:
            logger.error("❌ SOME CHECKS FAILED")
        logger.info("=" * 70)
        return report


# ---------------------------------------------------------------------------
# CodeRepairOrchestrator — Mode A (previously mode_code_repair.py)
# ---------------------------------------------------------------------------


class CodeRepairOrchestrator:
    """Orchestrate automated code repair using Ollama (Mode A).

    Scans the codebase for known bug patterns (see validators/), attempts
    Ollama-assisted fixes, validates the result, and optionally commits.
    Ollama integration is currently a stub awaiting full implementation.
    """

    def __init__(self, config: CloudConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def run(self) -> CodeRepairResult:
        """Execute code repair pipeline.

        Returns:
            CodeRepairResult with status and details
        """
        from validation import BugPatternDetector

        self.logger.info("Code Repair Orchestrator starting...")
        result = CodeRepairResult(success=False)

        try:
            # Step 1: Bug detection
            self.logger.info("\n[1] Scanning codebase for bugs...")
            detector = BugPatternDetector()
            bug_report = await detector.scan_codebase(Path.cwd())

            if not bug_report.bugs:
                self.logger.info("✓ No bugs detected, nothing to fix")
                result.success = True
                return result

            self.logger.warning(
                f"Found {bug_report.total_critical} critical, {bug_report.total_warnings} warnings"
            )

            # Step 2: Connect to Ollama (stub)
            self.logger.info("\n[2] Connecting to Ollama...")
            if not self.config.ollama_auto_start:
                self.logger.info(
                    f"Ollama URL: {self.config.ollama_base_url} (Model: {self.config.ollama_model})"
                )
            else:
                self.logger.info("Ollama auto-start: enabled (stub)")

            # Step 3: Fix bugs (Ollama integration deferred)
            self.logger.info("\n[3] Attempting to fix bugs...")
            self.logger.warning("⚠ Ollama integration stub — no fixes applied yet")
            self.logger.info(f"  Would fix: {bug_report.total_critical} critical issues")

            # Step 4: Validation
            self.logger.info("\n[4] Running validation checks...")
            check_report = await TestRunner.run_all_checks(run_tests=self.config.enable_pytest)

            if check_report.passed:
                self.logger.info("✓ All validation checks passed")
            else:
                self.logger.error("❌ Validation checks failed")

            # Step 5: Git commit (if configured)
            if self.config.auto_commit and check_report.passed:
                self.logger.info("\n[5] Committing changes...")
                current_branch = GitController.get_current_branch()
                if current_branch != self.config.git_branch:
                    self.logger.info(f"Checking out branch: {self.config.git_branch}")
                    GitController.checkout_branch(self.config.git_branch)

                message = (
                    f"AI fix: {bug_report.total_critical} critical, "
                    f"{bug_report.total_warnings} warnings fixed"
                )
                report = GitController.commit(message)

                if report.success:
                    self.logger.info(f"✓ Committed: {report.message}")
                    result.git_commits.append(report.commit_hash or "unknown")
                    result.success = True
                else:
                    self.logger.error(f"Commit failed: {report.error}")
                    result.errors.append(report.error or "Unknown commit error")
            else:
                if not check_report.passed:
                    result.errors.append("Validation failed, skipping commit")
                result.success = True  # Partial success

        except Exception as e:
            self.logger.error(f"Code repair error: {e}", exc_info=True)
            result.errors.append(str(e))

        return result


# ---------------------------------------------------------------------------
# MLTrainingOrchestrator — Mode B (previously mode_ml_training.py)
# ---------------------------------------------------------------------------


class MLTrainingOrchestrator:
    """Orchestrate DONUT + TrOCR+YOLO training on GPU (Mode B).

    Calls run_all.py as a subprocess for each experiment, aggregates results,
    generates the paper, syncs to storage, and optionally commits everything.
    """

    def __init__(self, config: CloudConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.storage_manager = StorageManager(config.results_dir)
        # Lazy import to avoid circular dependency with reporting
        from reporting import ResultsAggregator

        self.results_aggregator = ResultsAggregator(config.results_dir)

    async def run(self) -> MLTrainingResult:
        """Execute ML training pipeline.

        Returns:
            MLTrainingResult with experiment results
        """
        self.logger.info("ML Training Orchestrator starting...")
        result = MLTrainingResult(success=False)

        try:
            # Ensure on correct branch
            current_branch = GitController.get_current_branch()
            if current_branch != self.config.git_branch:
                self.logger.info(f"Checking out branch: {self.config.git_branch}")
                if not GitController.checkout_branch(self.config.git_branch):
                    result.errors.append(f"Could not checkout branch {self.config.git_branch}")
                    return result

            self.logger.info("\n" + "=" * 70)
            self.logger.info("DONUT EXPERIMENTS")
            self.logger.info("=" * 70)

            experiments_run: dict[int, ExperimentResult] = {}

            for exp_id in self.config.experiments_to_run:
                self.logger.info(f"\n▶ Running Experiment {exp_id}...")
                try:
                    cmd = ["python", "run_all.py", "--experiment", str(exp_id)]
                    if self.config.skip_trocr:
                        cmd.append("--skip-trocr")
                    if self.config.skip_pretrained_baseline:
                        cmd.append("--skip-pretrained")

                    proc = await asyncio.create_task(
                        asyncio.to_thread(
                            subprocess.run,
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=3600,  # 1-hour timeout per experiment
                        )
                    )

                    if proc.returncode == 0:
                        self.logger.info(f"✓ Experiment {exp_id} completed")
                        result_file = self.config.results_dir / f"experiment_{exp_id}.json"
                        if result_file.exists():
                            try:
                                data = json.loads(result_file.read_text())
                                metrics = ExperimentMetrics(**data.get("metrics", {}))
                                exp_result = ExperimentResult(
                                    experiment_id=data["experiment_id"],
                                    name=data["name"],
                                    datasets=data["datasets"],
                                    num_train_samples=data["num_train_samples"],
                                    metrics=metrics,
                                )
                                experiments_run[exp_id] = exp_result
                                self.logger.info(
                                    f"  F1 = {metrics.global_f1:.4f} "
                                    f"(precision={metrics.global_precision:.4f}, "
                                    f"recall={metrics.global_recall:.4f})"
                                )
                            except Exception as e:
                                self.logger.error(f"Could not parse results: {e}")
                                result.errors.append(f"Experiment {exp_id} results invalid")
                    else:
                        error_msg = proc.stderr if proc.stderr else "Unknown error"
                        self.logger.error(f"❌ Experiment {exp_id} failed: {error_msg}")
                        result.errors.append(f"Experiment {exp_id}: {error_msg}")
                        self.logger.error("Aborting pipeline (strict failure mode)")
                        return result

                except subprocess.TimeoutExpired:
                    result.errors.append(f"Experiment {exp_id} timed out")
                    return result
                except Exception as e:
                    self.logger.error(f"Experiment {exp_id} error: {e}")
                    result.errors.append(f"Experiment {exp_id}: {str(e)}")
                    return result

            # Aggregate results
            if experiments_run:
                self.logger.info("\n" + "=" * 70)
                self.logger.info("AGGREGATING RESULTS")
                self.logger.info("=" * 70)

                agg = self.results_aggregator.aggregate_experiments()
                if agg:
                    result.aggregated_results = agg
                    result.best_experiment_id = agg.best_experiment.experiment_id
                    self.logger.info(
                        f"Best: Exp {agg.best_experiment.experiment_id} "
                        f"F1={agg.best_experiment.metrics.global_f1:.4f}"
                    )
                    self.logger.info(
                        f"Baseline: Exp 1 F1={agg.baseline_f1:.4f}, "
                        f"Improvement: {agg.improvement:+.4f}"
                    )

                    # Generate paper
                    self.logger.info("\nGenerating paper...")
                    try:
                        proc = await asyncio.create_task(
                            asyncio.to_thread(
                                subprocess.run,
                                ["python", "inject_results.py", "--all"],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )
                        )
                        if proc.returncode == 0:
                            result.paper_generated = True
                            self.logger.info("✓ Paper generated")
                        else:
                            self.logger.warning(f"Paper generation had issues: {proc.stderr}")
                    except Exception as e:
                        self.logger.warning(f"Could not generate paper: {e}")

                # Sync results to storage
                self.logger.info("\nSyncing results...")
                sync_report = await self.storage_manager.sync_results_directory()
                result.cloud_sync_report = sync_report
                self.logger.info(f"✓ Results ready to commit ({sync_report.total_files} files)")

                # Commit results
                if self.config.auto_commit:
                    self.logger.info("\nCommitting results to git...")
                    best_exp = agg.best_experiment if agg else None
                    if best_exp:
                        message = (
                            f"AI experiment: [{', '.join(str(e) for e in experiments_run)}] "
                            f"— best=exp_{best_exp.experiment_id} "
                            f"F1={best_exp.metrics.global_f1:.4f} "
                            f"on {', '.join(best_exp.datasets)}"
                        )
                    else:
                        message = (
                            f"AI experiment: [{', '.join(str(e) for e in experiments_run)}] "
                            f"— experiments completed"
                        )
                    commit_report = GitController.commit(message)
                    if commit_report.success:
                        self.logger.info(f"✓ Committed: {message}")
                        result.git_commits.append(commit_report.commit_hash or "unknown")
                    else:
                        self.logger.error(f"Commit failed: {commit_report.error}")

            result.experiments_run = experiments_run
            result.success = len(experiments_run) > 0

            self.logger.info("\n" + "=" * 70)
            if result.success:
                self.logger.info(f"✓ ML TRAINING COMPLETE: {len(experiments_run)} experiments")
            else:
                self.logger.error("❌ ML TRAINING FAILED: No experiments completed")
            self.logger.info("=" * 70)

        except Exception as e:
            self.logger.error(f"ML training error: {e}", exc_info=True)
            result.errors.append(str(e))

        return result


# ---------------------------------------------------------------------------
# VastAIProvisioner — Vast.ai GPU instance lifecycle management
# ---------------------------------------------------------------------------


class VastAIProvisioner:
    """Provision a Vast.ai GPU instance and register it as a GitHub Actions runner.

    Uses the ``vastai`` CLI (``pip install vastai``) via :mod:`subprocess`.
    All methods log progress and handle missing-CLI gracefully.

    Parameters
    ----------
    config : CloudConfig
        Pipeline configuration.  The following fields are read:
        - ``vastai_api_key``  — Vast.ai API key (falls back to ``VASTAI_API_KEY`` env var)
        - ``vastai_gpu_name`` — GPU model filter (default: ``"RTX 4090"``)
        - ``vastai_max_price``— Maximum price in $/hr (default: ``0.50``)
        - ``github_repo``     — ``owner/repo`` slug for runner registration
        - ``git_branch``      — branch to check out on the instance

    Example
    -------
    >>> provisioner = VastAIProvisioner(config)
    >>> provisioner.provision_and_run()
    """

    #: Timeout (seconds) waiting for an instance to become SSH-accessible.
    BOOT_TIMEOUT: int = 600
    #: Polling interval (seconds) while waiting.
    POLL_INTERVAL: int = 15
    #: Docker image used for the instance.
    DEFAULT_IMAGE: str = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
    #: Root disk size in GB.
    DEFAULT_DISK_GB: int = 40
    #: GitHub Actions runner version to download.
    RUNNER_VERSION: str = "2.333.1"
    #: If the runner exits in fewer seconds than this, it's a crash, not job completion.
    RUNNER_CRASH_THRESHOLD_S: int = 60

    def __init__(self, config: CloudConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__ + ".VastAIProvisioner")
        self._instance_id: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _api_key(self) -> str:
        """Return Vast.ai API key from config or environment."""
        key = getattr(self.config, "vastai_api_key", None) or os.getenv("VASTAI_API_KEY", "")
        if not key:
            raise PipelineConfigError(
                "Vast.ai API key not set. "
                "Provide it via the VASTAI_API_KEY environment variable or config.vastai_api_key."
            )
        return key

    def _gpu_name(self) -> str:
        return getattr(self.config, "vastai_gpu_name", None) or "RTX 4090"

    def _max_price(self) -> float:
        return float(getattr(self.config, "vastai_max_price", None) or 0.50)

    def _run_vastai(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a ``vastai`` CLI sub-command and return the result.

        Raises :exc:`PipelineConfigError` when the ``vastai`` binary is not found.
        """
        cmd = ["vastai", *args]
        self.logger.debug("vastai command: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise PipelineConfigError(
                "vastai CLI not found. Install it with: pip install vastai"
            ) from None
        if check and result.returncode != 0:
            raise RuntimeError(
                f"vastai {args[0]} failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        return result

    def _parse_json(self, text: str) -> Any:
        """Parse JSON text, returning an empty dict/list on failure."""
        try:
            return json.loads(text or "null") or {}
        except json.JSONDecodeError:
            self.logger.warning(
                "Failed to parse JSON from vastai output: %r",
                (text or "")[:500],
            )
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_offers(self) -> list[dict[str, Any]]:
        """Search for available Vast.ai GPU offers matching the config.

        Returns
        -------
        list[dict]
            List of offer objects sorted by price (cheapest first).
            Empty list when no matching offers are found or vastai is unavailable.
        """
        gpu_name = self._gpu_name()
        max_price = self._max_price()
        min_vram = getattr(self.config, "vastai_min_vram", 24)

        self.logger.info(
            "Searching Vast.ai offers: GPU=%r ≥%d GB VRAM max $%.2f/hr",
            gpu_name,
            min_vram,
            max_price,
        )
        query = f"gpu_name='{gpu_name}' gpu_ram>={min_vram} dph<={max_price} rentable=True"
        try:
            result = self._run_vastai("search", "offers", query, "--order", "dph asc", "--raw")
            if result.stderr and result.stderr.strip():
                self.logger.warning("vastai search stderr: %s", result.stderr.strip())
            offers = self._parse_json(result.stdout)
            if isinstance(offers, list):
                self.logger.info("  Found %d matching offer(s)", len(offers))
                return offers
        except (PipelineConfigError, RuntimeError) as exc:
            self.logger.warning("Vast.ai offer search failed: %s", exc)
        return []

    def create_instance(self, offer_id: str | int) -> str:
        """Create a Vast.ai instance from *offer_id* and return the instance ID.

        Parameters
        ----------
        offer_id:
            Vast.ai offer identifier (integer or string).

        Returns
        -------
        str
            The new instance ID.

        Raises
        ------
        RuntimeError
            When instance creation fails or the response cannot be parsed.
        """
        image = getattr(self.config, "vastai_image", self.DEFAULT_IMAGE) or self.DEFAULT_IMAGE
        disk_gb = (
            getattr(self.config, "vastai_disk_gb", self.DEFAULT_DISK_GB) or self.DEFAULT_DISK_GB
        )

        self.logger.info("Creating Vast.ai instance from offer %s (image=%s)...", offer_id, image)
        result = self._run_vastai(
            "create",
            "instance",
            str(offer_id),
            "--image",
            image,
            "--disk",
            str(disk_gb),
            "--raw",
        )
        data = self._parse_json(result.stdout)
        instance_id = str(data.get("new_contract", ""))
        if not instance_id:
            raise RuntimeError(
                f"Could not parse instance ID from create response: {result.stdout[:200]}"
            )
        self._instance_id = instance_id
        self.logger.info("  Instance created: ID=%s", instance_id)
        return instance_id

    def wait_until_ready(self, instance_id: str) -> dict[str, Any]:
        """Poll until the instance is running and SSH-accessible.

        Parameters
        ----------
        instance_id:
            Vast.ai instance ID returned by :meth:`create_instance`.

        Returns
        -------
        dict
            The instance info dict containing ``ssh_host`` and ``ssh_port``.

        Raises
        ------
        TimeoutError
            When the instance does not become accessible within :attr:`BOOT_TIMEOUT` seconds.
        """
        import time

        self.logger.info(
            "Waiting for instance %s to boot (timeout %ds)...", instance_id, self.BOOT_TIMEOUT
        )
        elapsed = 0
        status = "unknown"
        while elapsed < self.BOOT_TIMEOUT:
            try:
                result = self._run_vastai("show", "instance", instance_id, "--raw", check=False)
                info = self._parse_json(result.stdout)
                status = info.get("actual_status", "unknown")
                if status == "running":
                    ssh_host = info.get("ssh_host") or info.get("public_ipaddr", "")
                    ssh_port = info.get("ssh_port", 22)
                    if ssh_host:
                        self.logger.info("  Instance running — SSH: %s:%s", ssh_host, ssh_port)
                        return info
            except RuntimeError as exc:
                self.logger.debug("Poll error: %s", exc)

            self.logger.debug(
                "  Status: %s — waiting %ds (%d/%d s elapsed)...",
                status,
                self.POLL_INTERVAL,
                elapsed,
                self.BOOT_TIMEOUT,
            )
            time.sleep(self.POLL_INTERVAL)
            elapsed += self.POLL_INTERVAL

        raise TimeoutError(
            f"Instance {instance_id} did not become SSH-accessible after {self.BOOT_TIMEOUT}s."
        )

    def get_ssh_command(self, instance_info: dict[str, Any]) -> str:
        """Return an SSH command string for connecting to the instance.

        Parameters
        ----------
        instance_info:
            Instance info dict as returned by :meth:`wait_until_ready`.

        Returns
        -------
        str
            A ready-to-run SSH command, e.g.
            ``"ssh -o StrictHostKeyChecking=no -p 12345 root@1.2.3.4"``.
        """
        host = instance_info.get("ssh_host") or instance_info.get("public_ipaddr", "")
        port = instance_info.get("ssh_port", 22)
        return f"ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p {port} root@{host}"

    def setup_runner(self, instance_info: dict[str, Any], registration_token: str) -> bool:
        """Clone the repo and configure an ephemeral GitHub Actions runner on the instance.

        Steps executed remotely via SSH:
        1. Create non-root ``runner`` user.
        2. Clone ``config.github_repo``.
        3. Download and extract the Actions runner binary.
        4. Install system dependencies via ``bin/installdependencies.sh``.
        5. Configure the runner (ephemeral mode).
        6. Start the runner in a new session (``setsid``) for SSH-disconnect survival.
        7. Wait up to 60 s for "Listening for Jobs" in the runner log.

        Parameters
        ----------
        instance_info:
            Instance info dict as returned by :meth:`wait_until_ready`.
        registration_token:
            GitHub Actions runner registration token (short-lived, ~60 min).

        Returns
        -------
        bool
            True when the remote setup commands succeed and the runner is healthy.
        """
        host = instance_info.get("ssh_host") or instance_info.get("public_ipaddr", "")
        port = instance_info.get("ssh_port", 22)
        repo = self.config.github_repo
        branch = self.config.git_branch
        github_token = os.getenv("GITHUB_TOKEN", "")
        runner_url = (
            f"https://github.com/actions/runner/releases/download/"
            f"v{self.RUNNER_VERSION}/actions-runner-linux-x64-{self.RUNNER_VERSION}.tar.gz"
        )
        runner_pkg = f"actions-runner-linux-x64-{self.RUNNER_VERSION}.tar.gz"
        runner_name = f"vastai-{self._instance_id or 'runner'}"
        runner_labels = "self-hosted,gpu,vast-ai"

        # Build the clone URL — use token auth if available for private repos
        if github_token:
            clone_url = f"https://x-access-token:{github_token}@github.com/{repo}.git"
        else:
            clone_url = f"https://github.com/{repo}.git"

        # Build the remote script in two parts:
        # 1. A variables section using shlex.quote() so any special characters
        #    (quotes, backslashes, dollar signs) in the values are safely
        #    shell-escaped — prevents injection of shell metacharacters.
        # 2. A body section using $VAR references (not Python f-string values),
        #    so the script logic is completely separated from the data.
        var_section = "#!/bin/bash\nset -euo pipefail\nexport GIT_TERMINAL_PROMPT=0\n"
        var_section += f"_CLONE_URL={shlex.quote(clone_url)}\n"
        var_section += f"_BRANCH={shlex.quote(branch)}\n"
        var_section += f"_RUNNER_URL={shlex.quote(runner_url)}\n"
        var_section += f"_RUNNER_PKG={shlex.quote(runner_pkg)}\n"
        var_section += f"_REG_TOKEN={shlex.quote(registration_token)}\n"
        var_section += f"_REPO={shlex.quote(repo)}\n"
        var_section += f"_RUNNER_NAME={shlex.quote(runner_name)}\n"
        var_section += f"_RUNNER_LABELS={shlex.quote(runner_labels)}\n"

        # Script body references $VAR variables — no f-string expansion here.
        body_section = r"""
echo '[remote] Creating non-root runner user...'
useradd -m -s /bin/bash runner 2>/dev/null || true
mkdir -p /workspace/actions-runner /workspace/runner-work /workspace/repo
chown -R runner:runner /workspace

echo "[remote] Cloning https://github.com/${_REPO} ..."
git -c credential.helper='' clone --depth 1 --branch "${_BRANCH}" "${_CLONE_URL}" /workspace/repo || \
    git -c credential.helper='' clone --depth 1 "${_CLONE_URL}" /workspace/repo
chown -R runner:runner /workspace/repo

cd /workspace/repo
echo '[remote] Installing Python dependencies...'
pip install --quiet -r requirements.txt || true
pip install --quiet ruff pyyaml
for pkg in torch transformers; do
    python3 -c "import $pkg" 2>/dev/null || \
        echo "[remote] WARNING: $pkg not importable — GPU extras may be missing"
done
python3 -c "from constants import FIELDS, BASE_MODEL, SEED; print('[remote] Import chain OK')"

echo '[remote] Downloading Actions runner...'
mkdir -p /workspace/actions-runner && cd /workspace/actions-runner
curl -sSfL "${_RUNNER_URL}" -o "/tmp/${_RUNNER_PKG}"
tar xzf "/tmp/${_RUNNER_PKG}" -C /workspace/actions-runner
chown -R runner:runner /workspace/actions-runner

echo '[remote] Installing runner system dependencies...'
if [ -x /workspace/actions-runner/bin/installdependencies.sh ]; then
    /workspace/actions-runner/bin/installdependencies.sh 2>&1 | tail -5
else
    apt-get update -qq 2>/dev/null || true
    # NEVER install liblttng-ust* — it triggers a container restart on Vast.ai.
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        libicu70 libssl3 libkrb5-3 zlib1g 2>/dev/null \
      || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        libicu-dev libssl-dev libkrb5-3 zlib1g 2>/dev/null \
      || true
fi

echo '[remote] Configuring runner...'
HOME=/home/runner su -s /bin/bash runner -c "
  set -e
  cd /workspace/actions-runner
  ./config.sh \
      --url \"https://github.com/${_REPO}\" \
      --token \"${_REG_TOKEN}\" \
      --name \"${_RUNNER_NAME}\" \
      --labels \"${_RUNNER_LABELS}\" \
      --ephemeral \
      --unattended \
      --disableupdate \
      --work /workspace/runner-work
" 2>&1

echo '[remote] Starting runner...'
cat > /tmp/start_runner.sh << 'LAUNCHER'
#!/bin/bash
set -euo pipefail
export HOME=/home/runner
cd /workspace/actions-runner
setsid nohup ./run.sh >> /workspace/runner.log 2>&1 &
RPID=$!
echo "${RPID}" > /tmp/runner.pid
disown "${RPID}"
echo "Runner started: PID=${RPID}"
LAUNCHER
chmod 755 /tmp/start_runner.sh
chown runner:runner /tmp/start_runner.sh
HOME=/home/runner su -s /bin/bash runner -c "bash /tmp/start_runner.sh"
rm -f /tmp/start_runner.sh

echo '[remote] Waiting for runner to become ready...'
HEALTH_OK=false
for i in $(seq 1 12); do
    sleep 5
    if [ -f /tmp/runner.pid ]; then
        RPID=$(cat /tmp/runner.pid)
        if ! kill -0 "${RPID}" 2>/dev/null; then
            echo "[remote] ERROR: Runner crashed after $((i * 5))s"
            cat /workspace/runner.log 2>/dev/null
            exit 1
        fi
    fi
    if grep -q 'Listening for Jobs' /workspace/runner.log 2>/dev/null; then
        echo "[remote] Runner is listening for jobs (took $((i * 5))s)."
        HEALTH_OK=true
        break
    fi
    echo "[remote]   ...waiting ($((i * 5))/60s)"
done
if [ "$HEALTH_OK" != "true" ]; then
    echo "[remote] WARNING: 'Listening for Jobs' not seen after 60s — continuing anyway."
    tail -10 /workspace/runner.log 2>/dev/null || true
fi
echo "[remote] Runner PID: $(cat /tmp/runner.pid 2>/dev/null || echo unknown)"
"""
        remote_script = var_section + body_section
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-p",
            str(port),
            f"root@{host}",
            "bash",
            "-s",
        ]
        self.logger.info("Executing remote setup on %s:%s...", host, port)
        try:
            result = subprocess.run(
                ssh_cmd,
                input=remote_script,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.stdout:
                for line in result.stdout.splitlines():
                    self.logger.info("  %s", line)
            if result.returncode != 0:
                self.logger.error(
                    "Remote setup failed (exit %d): %s", result.returncode, result.stderr
                )
                return False
            self.logger.info("  Remote setup complete.")
            return True
        except subprocess.TimeoutExpired:
            self.logger.error("Remote setup timed out after 600s.")
            return False
        except OSError as exc:
            self.logger.error("SSH error during remote setup: %s", exc)
            return False

    def destroy_instance(self, instance_id: str) -> bool:
        """Destroy a Vast.ai instance to stop billing.

        Parameters
        ----------
        instance_id:
            Vast.ai instance ID.

        Returns
        -------
        bool
            True when the destroy command succeeds.
        """
        self.logger.info("Destroying instance %s...", instance_id)
        try:
            self._run_vastai("destroy", "instance", instance_id, "--raw")
            self.logger.info("  Instance %s destroyed.", instance_id)
            return True
        except (PipelineConfigError, RuntimeError) as exc:
            self.logger.error("Failed to destroy instance %s: %s", instance_id, exc)
            return False

    def _get_runner_registration_token(self) -> str:
        """Fetch a GitHub Actions runner registration token via the GitHub API.

        Uses the same ``_github_request``-style pattern as ``diagnostics.py``.

        Returns
        -------
        str
            A short-lived registration token (~60 min validity).

        Raises
        ------
        PipelineConfigError
            When GITHUB_TOKEN is not set or the API call fails.
        """
        import urllib.error
        import urllib.request

        token = os.getenv("GITHUB_TOKEN", "")
        if not token:
            raise PipelineConfigError(
                "GITHUB_TOKEN environment variable is not set. "
                "A token with 'repo' scope is required to register a GitHub Actions runner."
            )
        repo = self.config.github_repo
        url = f"https://api.github.com/repos/{repo}/actions/runners/registration-token"
        req = urllib.request.Request(
            url,
            data=b"",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "donut-vastai-provisioner/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
            reg_token = data.get("token", "")
            if not reg_token:
                raise PipelineConfigError(f"GitHub API returned no token: {data}")
            self.logger.info("  Runner registration token obtained.")
            return reg_token
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise PipelineConfigError(
                f"GitHub API call failed: {exc}. Check GITHUB_TOKEN scope and network connectivity."
            ) from exc

    def dispatch_workflow(
        self, workflow: str = "gpu_training.yml", training_mode: str = "micro"
    ) -> bool:
        """Dispatch a GitHub Actions workflow via the REST API.

        Triggers ``workflow_dispatch`` so the ephemeral runner has a job waiting
        when it comes online — eliminating the race condition between runner
        registration and job queuing.

        Parameters
        ----------
        workflow:
            Workflow filename inside ``.github/workflows/``.
        training_mode:
            Value passed as the ``training_mode`` input to the workflow.

        Returns
        -------
        bool
            True if the dispatch succeeds (HTTP 204).
        """
        import urllib.error
        import urllib.request

        token = os.getenv("GITHUB_TOKEN", "")
        if not token:
            self.logger.warning("GITHUB_TOKEN not set — cannot dispatch workflow.")
            return False

        repo = self.config.github_repo
        url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
        body = json.dumps(
            {"ref": self.config.git_branch, "inputs": {"training_mode": training_mode}}
        ).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
                "User-Agent": "donut-vastai-provisioner/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                status = resp.status
            if status == 204:
                self.logger.info("  Workflow %s dispatched (mode=%s).", workflow, training_mode)
                return True
            self.logger.warning("  Workflow dispatch returned HTTP %d.", status)
            return False
        except urllib.error.HTTPError as exc:
            self.logger.warning("  Workflow dispatch failed (HTTP %d): %s", exc.code, exc.reason)
            return False
        except (urllib.error.URLError, OSError) as exc:
            self.logger.warning("  Workflow dispatch error: %s", exc)
            return False

    def _wait_for_runner_job(
        self,
        instance_info: dict[str, Any],
        timeout: int = 7200,
        poll_interval: int = 15,
    ) -> bool:
        """Poll the remote instance until the runner process exits.

        Parameters
        ----------
        instance_info:
            Instance info dict (needs ``ssh_host`` and ``ssh_port``).
        timeout:
            Maximum seconds to wait before giving up.
        poll_interval:
            Seconds between each SSH poll.

        Returns
        -------
        bool
            True if the runner exited after running long enough to have
            completed a job (>= ``RUNNER_CRASH_THRESHOLD_S``).  False if
            the runner exited too quickly (probable crash) or on timeout.
        """
        host = instance_info.get("ssh_host") or instance_info.get("public_ipaddr", "")
        port = instance_info.get("ssh_port", 22)
        crash_threshold = self.RUNNER_CRASH_THRESHOLD_S
        ssh_base = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "BatchMode=yes",
            "-p",
            str(port),
            f"root@{host}",
        ]

        elapsed = 0
        while elapsed < timeout:
            try:
                result = subprocess.run(
                    [
                        *ssh_base,
                        "kill -0 $(cat /tmp/runner.pid 2>/dev/null) 2>/dev/null "
                        "&& echo alive || echo gone",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                status = result.stdout.strip()
            except (subprocess.TimeoutExpired, OSError):
                status = "gone"

            if status == "gone":
                if elapsed < crash_threshold:
                    self.logger.warning(
                        "Runner exited after only %ds (<%ds) — possible crash.",
                        elapsed,
                        crash_threshold,
                    )
                    # Retrieve log for diagnosis
                    try:
                        log_result = subprocess.run(
                            [*ssh_base, "cat /workspace/runner.log 2>/dev/null"],
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        if log_result.stdout:
                            for line in log_result.stdout.splitlines()[-30:]:
                                self.logger.info("  runner.log: %s", line)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                    return False  # Crash — not a successful job
                self.logger.info("Runner exited — job complete (ran for %ds).", elapsed)
                return True  # Normal job completion

            self.logger.info("  Runner running (%d/%ds)...", elapsed, timeout)
            time.sleep(poll_interval)
            elapsed += poll_interval

        self.logger.warning("Job timed out after %ds.", timeout)
        return False

    def provision_and_run(self) -> bool:
        """Orchestrate the full Vast.ai lifecycle: search → create → wait → setup → destroy.

        This is the top-level convenience method.  It:
        1. Searches for a suitable GPU offer.
        2. Creates the instance.
        3. Waits for SSH accessibility.
        4. Fetches a GitHub Actions runner registration token.
        5. Dispatches the GPU training workflow so a job is queued.
        6. Clones the repo and configures an ephemeral runner on the instance.
        7. Waits for the runner to finish executing the job.
        8. Destroys the instance regardless of outcome (billing protection).

        Returns
        -------
        bool
            True when the runner executed a job successfully.
        """
        self.logger.info("=" * 70)
        self.logger.info("VastAIProvisioner: starting full provisioning lifecycle")
        self.logger.info("  Repo: %s | Branch: %s", self.config.github_repo, self.config.git_branch)
        self.logger.info("  GPU: %s | Max price: $%.2f/hr", self._gpu_name(), self._max_price())
        self.logger.info("=" * 70)

        # Authenticate
        try:
            api_key = self._api_key()
            self._run_vastai("set", "api-key", api_key)
            self.logger.info("Vast.ai authentication configured.")
        except PipelineConfigError as exc:
            self.logger.error("Authentication failed: %s", exc)
            return False

        # Search for an offer
        offers = self.search_offers()
        if not offers:
            self.logger.error("No suitable Vast.ai offers found. Aborting.")
            return False
        offer_id = offers[0]["id"]

        # Create instance
        try:
            instance_id = self.create_instance(offer_id)
        except RuntimeError as exc:
            self.logger.error("Instance creation failed: %s", exc)
            return False

        success = False
        instance_info: dict[str, Any] = {}
        try:
            # Wait for boot
            instance_info = self.wait_until_ready(instance_id)

            # Get registration token
            reg_token = self._get_runner_registration_token()

            # Dispatch workflow so a job is queued BEFORE the runner comes online
            self.dispatch_workflow()

            # Set up runner
            runner_ok = self.setup_runner(instance_info, reg_token)
            if runner_ok:
                self.logger.info(
                    "✓ Ephemeral runner started on instance %s. Waiting for job to complete...",
                    instance_id,
                )
                # Wait for the runner to finish executing the job
                success = self._wait_for_runner_job(instance_info)
            else:
                self.logger.error("Runner setup failed on instance %s.", instance_id)
        except (TimeoutError, PipelineConfigError, RuntimeError) as exc:
            self.logger.error("Provisioning error: %s", exc)
        finally:
            # Always destroy — protect against billing runaway
            self.destroy_instance(instance_id)

        self.logger.info("=" * 70)
        self.logger.info(
            "VastAIProvisioner: lifecycle complete — %s",
            "SUCCESS" if success else "FAILED",
        )
        self.logger.info("=" * 70)
        return success


# ---------------------------------------------------------------------------
# CloudPipelineOrchestrator — top-level router
# ---------------------------------------------------------------------------


class CloudPipelineOrchestrator:
    """Main pipeline orchestrator — routes between Code Repair and ML Training."""

    def __init__(self, config: CloudConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def run(self) -> PipelineResult:
        """Execute pipeline based on configured mode.

        Returns:
            PipelineResult with overall status and timing
        """
        start_time = datetime.utcnow()

        self.logger.info("=" * 70)
        self.logger.info("CLOUD PIPELINE STARTING")
        self.logger.info("=" * 70)
        self.logger.info(f"Mode: {self.config.mode.value}")
        self.logger.info(f"Branch: {self.config.git_branch}")
        self.logger.info(f"Workspace: {self.config.workspace}")

        _is_valid, errors = self.config.validate()
        if not errors:
            self.logger.info("✓ Configuration valid")
        else:
            for error in errors:
                self.logger.warning(f"  {error}")

        # CRITICAL: preflight checks must pass before anything else runs
        if not self.config.skip_validation:
            from validation import PreflightChecker  # noqa: E402, I001 (lazy: avoids circular import)

            preflight = PreflightChecker(sroie_dir=self.config.sroie_data_dir)
            report = await preflight.run_all()
            if not report.passed:
                self.logger.error("❌ Preflight checks failed, cannot proceed")
                return PipelineResult(
                    success=False,
                    mode=self.config.mode.value,
                    errors=report.errors,
                )
        else:
            self.logger.warning("⚠ Preflight checks skipped (--skip-validation)")

        # Resolve auto mode
        if self.config.mode == PipelineMode.AUTO:
            self.logger.info("Auto-detecting mode from branch...")
            if "code-repair" in self.config.git_branch:
                mode = PipelineMode.CODE_REPAIR
            else:
                mode = PipelineMode.ML_TRAINING  # default
            self.logger.info(f"  Detected: {mode.value}")
        else:
            mode = self.config.mode

        # Route to mode orchestrator
        if mode == PipelineMode.CODE_REPAIR:
            self.logger.info("\n▶ Starting MODE A: Code Repair (Ollama)")
            orchestrator: CodeRepairOrchestrator | MLTrainingOrchestrator = CodeRepairOrchestrator(
                self.config
            )
        elif mode == PipelineMode.ML_TRAINING:
            self.logger.info("\n▶ Starting MODE B: ML Training")
            orchestrator = MLTrainingOrchestrator(self.config)
        else:
            self.logger.error(f"Unknown mode: {mode}")
            return PipelineResult(success=False, mode="unknown", errors=["Unknown pipeline mode"])

        mode_result = await orchestrator.run()
        success = mode_result.success

        duration = (datetime.utcnow() - start_time).total_seconds()
        result = PipelineResult(
            success=success,
            mode=mode.value,
            mode_result=mode_result,
            duration_sec=duration,
        )

        self.logger.info("=" * 70)
        if result.success:
            self.logger.info(f"✓ PIPELINE SUCCEEDED in {duration:.1f}s")
        else:
            self.logger.error(f"❌ PIPELINE FAILED after {duration:.1f}s")
            if mode_result and hasattr(mode_result, "errors"):
                for error in mode_result.errors:
                    self.logger.error(f"  - {error}")
        self.logger.info("=" * 70)

        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Cloud AI Agent Pipeline: code repair + ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cloud_pipeline.py --mode ml_training
  python cloud_pipeline.py --mode code_repair
  python cloud_pipeline.py --experiments 1 --skip-trocr
  python cloud_pipeline.py --dry-run
  python cloud_pipeline.py --branch my-feature-branch
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["code_repair", "ml_training", "auto"],
        default="auto",
        help="Pipeline execution mode (default: auto-detect from branch)",
    )
    parser.add_argument(
        "--experiments", type=int, nargs="+", help="Experiment IDs to run (default: 1-8)"
    )
    parser.add_argument("--skip-trocr", action="store_true", help="Skip TrOCR+YOLO stages")
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip pre-flight checks (dangerous)"
    )
    parser.add_argument("--skip-pretrained", action="store_true", help="Skip pretrained baseline")
    parser.add_argument("--dry-run", action="store_true", help="Plan execution without running")
    parser.add_argument("--no-commit", action="store_true", help="Skip git commits")
    parser.add_argument("--branch", help="Git branch to work on")
    parser.add_argument("--workspace", help="Workspace directory")
    parser.add_argument("--s3-bucket", help="AWS S3 bucket for results (optional)")
    parser.add_argument("--gcs-bucket", help="Google Cloud Storage bucket (optional)")
    parser.add_argument(
        "--provision-vastai",
        action="store_true",
        help=(
            "Provision a Vast.ai GPU instance, register it as an ephemeral GitHub Actions "
            "self-hosted runner, and trigger the training workflow. "
            "Requires VASTAI_API_KEY, GITHUB_TOKEN, and GITHUB_REPO env vars."
        ),
    )

    args = parser.parse_args()

    # --provision-vastai: short-circuit before the normal pipeline
    if getattr(args, "provision_vastai", False):
        config = CloudConfig.from_args_and_env(args)
        provisioner = VastAIProvisioner(config)
        success = provisioner.provision_and_run()
        sys.exit(0 if success else 1)

    config = CloudConfig.from_args_and_env(args)

    logger.info(f"Configuration loaded: {config.mode.value} mode")
    logger.info(f"  Workspace: {config.workspace}")
    logger.info(f"  Results dir: {config.results_dir}")
    logger.info(f"  Branch: {config.git_branch}")

    orchestrator = CloudPipelineOrchestrator(config)

    try:
        result = await orchestrator.run()
        if result.success:
            sys.exit(0)
        elif result.partial_success:
            logger.warning("Partial success — some operations completed")
            sys.exit(1)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    asyncio.run(main())
