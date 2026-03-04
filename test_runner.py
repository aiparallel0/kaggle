"""Test runner orchestration - ruff checks and pytest execution."""

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["TestRunner", "RuffReport", "RuffFormatReport", "PytestReport", "AllChecksReport"]

logger = logging.getLogger(__name__)


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
        """Check if all checks passed."""
        passed = self.ruff_report.passed and self.ruff_format_report.passed
        if self.pytest_report:
            passed = passed and self.pytest_report.passed
        return passed


class TestRunner:
    """Orchestrate ruff and pytest checks."""

    @staticmethod
    async def run_ruff_check(check_dir: Path = Path(".")) -> RuffReport:
        """Run ruff lint check.

        Args:
            check_dir: Directory to check (default: current directory)

        Returns:
            RuffReport with results
        """
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

            # Parse output for issues
            if result.stdout:
                report.issues = [line.strip() for line in result.stdout.split("\n") if line.strip()]

            if report.passed:
                logger.info("✓ Ruff check passed")
            else:
                logger.error(f"❌ Ruff check failed ({len(report.issues)} issues)")
                for issue in report.issues[:5]:  # Show first 5
                    logger.error(f"  {issue}")

            return report

        except asyncio.TimeoutError:
            return RuffReport(
                passed=False,
                stderr="Ruff check timed out",
                exit_code=1,
            )
        except Exception as e:
            logger.error(f"Ruff check error: {e}")
            return RuffReport(
                passed=False,
                stderr=str(e),
                exit_code=1,
            )

    @staticmethod
    async def run_ruff_format(format_dir: Path = Path("."), fix: bool = False) -> RuffFormatReport:
        """Run ruff format check/fix.

        Args:
            format_dir: Directory to format (default: current directory)
            fix: If True, apply fixes; if False, only check

        Returns:
            RuffFormatReport with results
        """
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
            return RuffFormatReport(
                passed=False,
                stderr="Ruff format timed out",
                exit_code=1,
            )
        except Exception as e:
            logger.error(f"Ruff format error: {e}")
            return RuffFormatReport(
                passed=False,
                stderr=str(e),
                exit_code=1,
            )

    @staticmethod
    async def run_pytest(
        test_dir: Path = Path("tests"),
        markers: str = "",
        verbose: bool = True,
        timeout: int = 300,
    ) -> PytestReport:
        """Run pytest tests.

        Args:
            test_dir: Tests directory
            markers: Pytest markers to filter tests
            verbose: If True, show verbose output
            timeout: Timeout in seconds

        Returns:
            PytestReport with results
        """
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

            # Parse output for counts
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
                logger.error(
                    f"❌ Tests failed: {report.tests_failed} failed, "
                    f"{report.tests_passed} passed, {report.tests_skipped} skipped"
                )

            return report

        except asyncio.TimeoutError:
            logger.error(f"Pytest timed out after {timeout} seconds")
            return PytestReport(
                passed=False,
                stderr=f"Pytest timed out after {timeout} seconds",
                exit_code=1,
            )
        except Exception as e:
            logger.error(f"Pytest error: {e}")
            return PytestReport(
                passed=False,
                stderr=str(e),
                exit_code=1,
            )

    @staticmethod
    async def run_all_checks(
        check_dir: Path = Path("."),
        test_dir: Path = Path("tests"),
        run_tests: bool = True,
    ) -> AllChecksReport:
        """Run all code quality checks in sequence.

        Args:
            check_dir: Directory to lint
            test_dir: Tests directory
            run_tests: If False, skip pytest

        Returns:
            AllChecksReport with all results
        """
        logger.info("=" * 70)
        logger.info("RUNNING ALL CHECKS")
        logger.info("=" * 70)

        # Run ruff checks
        ruff_report = await TestRunner.run_ruff_check(check_dir)
        ruff_format_report = await TestRunner.run_ruff_format(check_dir, fix=False)

        # Run tests if requested
        pytest_report = None
        if run_tests:
            pytest_report = await TestRunner.run_pytest(test_dir)

        # Combine results
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
            if not ruff_report.passed:
                logger.error(f"  Ruff: {len(ruff_report.issues)} issues")
            if not ruff_format_report.passed:
                logger.error("  Ruff format: formatting issues")
            if pytest_report and not pytest_report.passed:
                logger.error(
                    f"  Pytest: {pytest_report.tests_failed} failures, "
                    f"{pytest_report.tests_passed} passed"
                )

        logger.info("=" * 70)

        return report
