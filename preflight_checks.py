# =============================================================================
# preflight_checks.py
# Purpose: Pre-flight environment validation (GPU, disk space, datasets, Python dependencies)
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-07
# =============================================================================
"""Orchestrator for all pre-flight validation checks.

From CLAUDE.md Section 5:
"Before running any experiment, always verify import chain is working first.
If it fails, fix the core import chain before touching experiment logic."

This module runs all safety checks before any pipeline stage executes.
"""

import asyncio
import logging
from pathlib import Path

from constants import _get_sroie_dir
from pipeline_types import CheckResult, CheckStatus, PreflightReport
from validators import (
    DataSplitValidator,
    ImportChainChecker,
    SeedValidator,
)

__all__ = ["PreflightChecker", "validate_pipeline"]

logger = logging.getLogger(__name__)


class PreflightChecker:
    """Run comprehensive preflight validation before pipeline execution."""

    def __init__(self, sroie_dir: Path | None = None):
        self.sroie_dir = sroie_dir or _get_sroie_dir()

    async def check_import_chain(self) -> CheckResult:
        """Check if constants.py import works (CRITICAL)."""
        logger.info("Checking import chain...")
        success, errors = ImportChainChecker.check_all()

        if not errors:
            result_status = CheckStatus.PASSED
            message = "All imports working"
        else:
            result_status = CheckStatus.FAILED
            message = "; ".join(errors)

        return CheckResult(
            name="import_chain",
            status=result_status,
            message=message,
        )

    async def check_constants_integrity(self) -> CheckResult:
        """Check if all required constants are defined."""
        logger.info("Checking constants integrity...")

        try:
            from constants import (
                BASE_MODEL,
                FIELDS,
                IMAGE_EXTS,
                SEED,
            )

            errors = []
            if not FIELDS:
                errors.append("FIELDS is empty")
            if not BASE_MODEL:
                errors.append("BASE_MODEL is empty")
            if not IMAGE_EXTS:
                errors.append("IMAGE_EXTS is empty")

            if errors:
                return CheckResult(
                    name="constants",
                    status=CheckStatus.FAILED,
                    message="; ".join(errors),
                )

            return CheckResult(
                name="constants",
                status=CheckStatus.PASSED,
                message=f"All constants valid (FIELDS={FIELDS}, SEED={SEED})",
            )

        except ImportError as e:
            return CheckResult(
                name="constants",
                status=CheckStatus.FAILED,
                message=f"Import error: {e}",
            )

    async def check_data_split_integrity(self) -> CheckResult:
        """Check SROIE data split (val_img != test_img)."""
        logger.info("Checking SROIE data split...")

        if not self.sroie_dir.exists():
            return CheckResult(
                name="data_split",
                status=CheckStatus.FAILED,
                message=f"SROIE directory not found: {self.sroie_dir}",
            )

        report = DataSplitValidator.validate_sroie_split(self.sroie_dir)

        if report.passed:
            return CheckResult(
                name="data_split",
                status=CheckStatus.PASSED,
                message=f"Split valid: {report.train_count}/{report.val_count}/{report.test_count}",
            )
        else:
            return CheckResult(
                name="data_split",
                status=CheckStatus.FAILED,
                message="; ".join(report.errors),
            )

    async def check_seed_consistency(self) -> CheckResult:
        """Check seed reproducibility."""
        logger.info("Checking seed consistency...")

        success, errors = SeedValidator.check_all()

        if success:
            return CheckResult(
                name="seed",
                status=CheckStatus.PASSED,
                message="Seed=42 and RNG consistent",
            )
        else:
            return CheckResult(
                name="seed",
                status=CheckStatus.WARNING,
                message="; ".join(errors),
            )

    async def check_gpu_availability(self) -> CheckResult:
        """Check if GPU is available."""
        logger.info("Checking GPU availability...")

        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                return CheckResult(
                    name="gpu",
                    status=CheckStatus.PASSED,
                    message=f"GPU available: {device_count}x {device_name}",
                )
            else:
                return CheckResult(
                    name="gpu",
                    status=CheckStatus.WARNING,
                    message="No GPU detected (will use CPU, training will be slow)",
                )

        except ImportError:
            return CheckResult(
                name="gpu",
                status=CheckStatus.WARNING,
                message="torch not installed (cannot check GPU)",
            )

    async def check_disk_space(self, min_gb: int = 100) -> CheckResult:
        """Check available disk space."""
        logger.info("Checking disk space...")

        try:
            import shutil

            stat = shutil.disk_usage("/")
            available_gb = stat.free / (1024**3)

            if available_gb >= min_gb:
                return CheckResult(
                    name="disk_space",
                    status=CheckStatus.PASSED,
                    message=f"Disk space OK: {available_gb:.1f} GB available",
                )
            else:
                return CheckResult(
                    name="disk_space",
                    status=CheckStatus.WARNING,
                    message=f"Low disk space: {available_gb:.1f} GB available (need {min_gb} GB)",
                )

        except Exception as e:
            return CheckResult(
                name="disk_space",
                status=CheckStatus.WARNING,
                message=f"Could not check disk space: {e}",
            )

    async def check_git_state(self) -> CheckResult:
        """Check git working directory state."""
        logger.info("Checking git state...")

        try:
            import subprocess

            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return CheckResult(
                    name="git",
                    status=CheckStatus.FAILED,
                    message="Not in a git repository",
                )

            # Check current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                branch = result.stdout.strip()
                return CheckResult(
                    name="git",
                    status=CheckStatus.PASSED,
                    message=f"Git ready on branch: {branch}",
                )
            else:
                return CheckResult(
                    name="git",
                    status=CheckStatus.FAILED,
                    message="Could not determine current branch",
                )

        except Exception as e:
            return CheckResult(
                name="git",
                status=CheckStatus.WARNING,
                message=f"Git check failed: {e}",
            )

    async def check_cloud_credentials(self) -> CheckResult:
        """Check cloud storage credentials."""
        import os

        logger.info("Checking cloud credentials...")

        # Note: Based on user feedback, we commit results to GitHub (via git)
        # No S3/GCS needed, so this is just informational
        has_aws = os.getenv("AWS_ACCESS_KEY_ID") is not None
        has_gcs = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is not None
        has_github = os.getenv("GITHUB_TOKEN") is not None

        status_msg = "GitHub: "
        status_msg += "✓" if has_github else "✗"
        if has_aws:
            status_msg += " AWS: ✓"
        if has_gcs:
            status_msg += " GCS: ✓"

        if has_github:
            return CheckResult(
                name="credentials",
                status=CheckStatus.PASSED,
                message=status_msg,
            )
        else:
            return CheckResult(
                name="credentials",
                status=CheckStatus.WARNING,
                message=status_msg + " (GitHub token recommended for git operations)",
            )

    async def run_all(self) -> PreflightReport:
        """Run all preflight checks.

        Returns:
            PreflightReport with all results
        """
        report = PreflightReport(passed=False)

        logger.info("=" * 70)
        logger.info("PREFLIGHT CHECKS")
        logger.info("=" * 70)

        # CRITICAL: Import chain must work first
        import_result = await self.check_import_chain()
        report.checks["import_chain"] = import_result

        if import_result.status == CheckStatus.FAILED:
            logger.error("❌ CRITICAL: Import chain broken, cannot proceed")
            report.errors.append(f"Import chain: {import_result.message}")
            return report

        # Other checks can run in parallel
        results = await asyncio.gather(
            self.check_constants_integrity(),
            self.check_data_split_integrity(),
            self.check_seed_consistency(),
            self.check_gpu_availability(),
            self.check_disk_space(),
            self.check_git_state(),
            self.check_cloud_credentials(),
        )

        check_names = [
            "constants",
            "data_split",
            "seed",
            "gpu",
            "disk_space",
            "git",
            "credentials",
        ]

        for name, result in zip(check_names, results):
            report.checks[name] = result

            if result.status == CheckStatus.FAILED:
                report.errors.append(f"{name}: {result.message}")
            elif result.status == CheckStatus.WARNING:
                report.warnings.append(f"{name}: {result.message}")

        # Determine overall pass/fail
        # Pass if no FAILED checks, warnings are OK
        critical_failures = [
            c
            for c in report.checks.values()
            if c.status == CheckStatus.FAILED and c.name in ["import_chain", "data_split"]
        ]

        report.passed = len(critical_failures) == 0

        # Log summary
        logger.info("=" * 70)
        if report.passed:
            logger.info("✓ PREFLIGHT CHECKS PASSED")
            if report.warnings:
                logger.warning(f"  Warnings: {len(report.warnings)}")
        else:
            logger.error("❌ PREFLIGHT CHECKS FAILED")
            for error in report.errors:
                logger.error(f"  - {error}")

        logger.info("=" * 70)

        return report


def validate_pipeline() -> bool:
    """Validate that the evaluation pipeline can load and initialize models.

    Migrated from evaluate.py (now deleted) so this logic lives alongside
    the other preflight validators.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("\n🔍 Validating evaluation pipeline...\n")

    checks = {
        "constants": False,
        "donut_evaluator": False,
        "device": False,
        "device_type": False,
    }

    try:
        from constants import BASE_MODEL, FIELDS, MAX_LENGTH  # noqa: F401

        print(f"  ✓ Constants loaded: {len(FIELDS)} fields, max_length={MAX_LENGTH}")
        checks["constants"] = True
    except Exception as e:
        print(f"  ✗ Failed to load constants: {e}")

    try:
        from donut_evaluator import DonutEvaluator  # noqa: F401

        print("  ✓ DonutEvaluator class available")
        checks["donut_evaluator"] = True
    except Exception as e:
        print(f"  ✗ Failed to load DonutEvaluator: {e}")

    try:
        import torch  # noqa: F401

        print(f"  ✓ PyTorch loaded: {torch.__version__}")
        checks["device"] = True
    except Exception as e:
        print(f"  ✗ Failed to load PyTorch: {e}")

    try:
        from donut_evaluator import DEVICE  # noqa: F401

        print(f"  ✓ Detected device: {DEVICE}")
        checks["device_type"] = True
    except Exception as e:
        print(f"  ✗ Failed to detect device: {e}")

    print(f"\n{'=' * 50}")
    if all(checks.values()):
        print("✅ Pipeline validation PASSED")
        return True
    else:
        print("❌ Pipeline validation FAILED:")
        for check, result in checks.items():
            status = "✓" if result else "✗"
            print(f"   {status} {check}")
        return False
