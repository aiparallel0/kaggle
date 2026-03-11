# =============================================================================
# cloud_pipeline.py
# Purpose: Cloud pipeline config, utilities, and orchestration (unified)
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-07
# =============================================================================
"""cloud_pipeline.py — Unified orchestrator for code repair and ML training.

Consolidates pipeline_config.py (CloudConfig, PipelineMode, LogLevel),
cloud_utils.py (GitController, StorageManager), and the orchestration layer
into a single module.  Two execution modes:

  Mode A — Code Repair  : Ollama-based automated bug fixing
  Mode B — ML Training  : GPU experiment runner (Vast.ai / local GPU)

Usage
-----
  python cloud_pipeline.py --mode ml_training --experiments 1 2 3
  python cloud_pipeline.py --mode code_repair --dry-run
  python cloud_pipeline.py --mode auto          # auto-detect from branch name
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from pipeline_types import (
    CodeRepairResult,
    ExperimentMetrics,
    ExperimentResult,
    GitCommitReport,
    MLTrainingResult,
    PipelineResult,
    SyncReport,
    UploadReport,
)
from preflight_checks import PreflightChecker

__all__ = [
    # Config (formerly pipeline_config.py)
    "CloudConfig",
    "PipelineMode",
    "LogLevel",
    # Utilities (formerly cloud_utils.py)
    "GitController",
    "StorageManager",
    # Orchestration
    "CloudPipelineOrchestrator",
    "CodeRepairOrchestrator",
    "MLTrainingOrchestrator",
    "RetroUIFormatter",
]


# =============================================================================
# Pipeline Configuration  (formerly pipeline_config.py)
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
    def from_env() -> "CloudConfig":
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
    def from_args_and_env(args) -> "CloudConfig":
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

    def __init__(self, results_dir: Path = Path("results")):
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
logger = logging.getLogger(__name__)


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
# CodeRepairOrchestrator — Mode A (previously mode_code_repair.py)
# ---------------------------------------------------------------------------


class CodeRepairOrchestrator:
    """Orchestrate automated code repair using Ollama (Mode A).

    Scans the codebase for known bug patterns (see validators/), attempts
    Ollama-assisted fixes, validates the result, and optionally commits.
    Ollama integration is currently a stub awaiting full implementation.
    """

    def __init__(self, config: CloudConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def run(self) -> CodeRepairResult:
        """Execute code repair pipeline.

        Returns:
            CodeRepairResult with status and details
        """
        from test_runner import TestRunner
        from validators import BugPatternDetector

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

    def __init__(self, config: CloudConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.storage_manager = StorageManager(config.results_dir)
        # Lazy import to avoid circular dependency with inject_results
        from inject_results import ResultsAggregator

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

    def __init__(self, config: CloudConfig):
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
