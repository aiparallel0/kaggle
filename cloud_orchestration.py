"""cloud_orchestration.py — Merged pipeline types, critic, DAG scheduler, and cloud pipeline.

Consolidates: pipeline_types.py, pipeline_critic.py, dag_scheduler.py, cloud_pipeline.py
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import math
import os
import subprocess
import sys
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
class PreflightReport:
    """Comprehensive preflight validation report."""

    passed: bool
    checks: dict[str, CheckResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
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

    bugs: list[BugPattern] = field(default_factory=list)
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
class ExperimentResult:
    """Result of a single training experiment."""

    experiment_id: int
    name: str
    datasets: list[str]
    num_train_samples: int
    metrics: ExperimentMetrics
    checkpoint_path: Path | None = None
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
class CritiqueFinding:
    """A single research-validity critique finding."""

    severity: FindingSeverity
    category: str
    title: str
    description: str
    evidence: str
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "recommendation": self.recommendation,
        }


@dataclass
class CritiqueReport:
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

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "fatal_count": self.fatal_count,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "findings": [f.to_dict() for f in self.findings],
        }

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
        configs: list,
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


def print_dag(configs: list) -> None:
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
    def commit(message: str, files: list | None = None) -> GitCommitReport:
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

    args = parser.parse_args()
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
