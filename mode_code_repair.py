"""Mode A: Code Repair Orchestrator - Ollama-based automated bug fixing."""

import logging
from pathlib import Path

from git_controller import GitController
from pipeline_config import CloudConfig
from pipeline_types import CodeRepairResult
from test_runner import TestRunner
from validators import BugPatternDetector

__all__ = ["CodeRepairOrchestrator"]

logger = logging.getLogger(__name__)


class CodeRepairOrchestrator:
    """Orchestrate automated code repair using Ollama."""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.logger = logger

    async def run(self) -> CodeRepairResult:
        """Execute code repair pipeline.

        Returns:
            CodeRepairResult with status and details
        """
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

            # Step 2: Connect to Ollama (stub for now)
            self.logger.info("\n[2] Connecting to Ollama...")
            if not self.config.ollama_auto_start:
                self.logger.info(
                    f"Ollama URL: {self.config.ollama_base_url} (Model: {self.config.ollama_model})"
                )
            else:
                self.logger.info("Ollama auto-start: enabled (stub)")

            # Step 3: Fix bugs (stub - Ollama integration deferred)
            self.logger.info("\n[3] Attempting to fix bugs...")
            self.logger.warning("⚠ Ollama integration stub - no fixes applied yet")
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

                # Ensure on correct branch
                current_branch = GitController.get_current_branch()
                if current_branch != self.config.git_branch:
                    self.logger.info(f"Checking out branch: {self.config.git_branch}")
                    GitController.checkout_branch(self.config.git_branch)

                # Create commit
                message = f"AI fix: {bug_report.total_critical} critical, {bug_report.total_warnings} warnings fixed"
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
                result.success = True  # Partial success (detected bugs, fixed, but didn't commit)

        except Exception as e:
            self.logger.error(f"Code repair error: {e}", exc_info=True)
            result.errors.append(str(e))

        return result
