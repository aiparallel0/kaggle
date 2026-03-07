"""cloud_utils.py — Cloud infrastructure helpers: git operations and result storage.

Consolidates GitController and StorageManager into one module.  Both classes
are used exclusively by the cloud pipeline orchestration layer
(cloud_pipeline.py) and share the same conceptual scope: managing
interactions with GitHub and local result files on the cloud GPU host.

Classes
-------
GitController   Thin subprocess wrapper for branch/commit/push operations.
StorageManager  Stores experiment results locally and reports them ready-to-commit.
"""

import logging
import subprocess
from pathlib import Path

from pipeline_types import GitCommitReport, SyncReport, UploadReport

__all__ = ["GitController", "StorageManager"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitController
# ---------------------------------------------------------------------------


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
            logger.error(f"Could not get current branch: {e}")
        return None

    @staticmethod
    def checkout_branch(branch_name: str) -> bool:
        """Checkout *branch_name*, creating it if it does not yet exist.

        Returns True on success.
        """
        try:
            result = subprocess.run(
                ["git", "checkout", branch_name],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                logger.info(f"✓ Checked out branch: {branch_name}")
                return True

            # Branch doesn't exist — create it
            if "did not match any branch" in result.stderr.lower():
                logger.info(f"Creating new branch: {branch_name}")
                result = subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    logger.info(f"✓ Created and checked out branch: {branch_name}")
                    return True

            logger.error(f"Failed to checkout branch: {result.stderr}")
            return False

        except Exception as e:
            logger.error(f"Git checkout error: {e}")
            return False

    @staticmethod
    def commit(message: str, files: list | None = None) -> GitCommitReport:
        """Stage *files* (or all changes when None) and create a commit.

        Returns a GitCommitReport describing the outcome.
        """
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

                logger.info(f"✓ Committed: {message} ({commit_hash})")
                return GitCommitReport(
                    success=True,
                    commit_hash=commit_hash,
                    message=message,
                    branch=GitController.get_current_branch(),
                )

            if "nothing to commit" in result.stdout.lower():
                logger.warning("Nothing to commit")
                return GitCommitReport(
                    success=True,
                    message="Nothing to commit",
                    branch=GitController.get_current_branch(),
                )

            return GitCommitReport(success=False, error=f"Commit failed: {result.stderr}")

        except Exception as e:
            logger.error(f"Git commit error: {e}")
            return GitCommitReport(success=False, error=str(e))

    @staticmethod
    def push_branch(branch_name: str, force: bool = False) -> bool:
        """Push *branch_name* to origin.  Returns True on success."""
        try:
            cmd = ["git", "push", "-u", "origin", branch_name]
            if force:
                cmd.insert(2, "--force-with-lease")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(f"✓ Pushed branch: {branch_name}")
                return True
            logger.error(f"Push failed: {result.stderr}")
            return False
        except Exception as e:
            logger.error(f"Git push error: {e}")
            return False

    @staticmethod
    def tag_commit(tag_name: str, message: str = "") -> bool:
        """Create a git tag for the current commit.  Returns True on success."""
        try:
            cmd = (
                ["git", "tag", "-a", tag_name, "-m", message]
                if message
                else ["git", "tag", tag_name]
            )
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✓ Created tag: {tag_name}")
                return True
            logger.error(f"Tag creation failed: {result.stderr}")
            return False
        except Exception as e:
            logger.error(f"Git tag error: {e}")
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
            logger.error(f"Git status error: {e}")
            return ""


# ---------------------------------------------------------------------------
# StorageManager
# ---------------------------------------------------------------------------


class StorageManager:
    """Store experiment results locally and report them ready-to-commit to GitHub.

    From user feedback: "A Folder to put results at GitHub is sufficient."
    Results are committed directly to the feature branch via git.
    No S3/GCS required for the MVP.
    """

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def sync_results_directory(self, remote_prefix: str = "") -> SyncReport:
        """Report which result files are ready for a GitHub commit.

        Args:
            remote_prefix: Unused — kept for API compatibility.

        Returns:
            SyncReport summarising all JSON files in *results_dir*.
        """
        logger.info("Preparing results for GitHub commit...")

        exp_files = list(self.results_dir.glob("experiment_*.json"))
        agg_files = list(self.results_dir.glob("all_experiments.json"))
        other_files = list(self.results_dir.glob("*.json"))
        all_files = list(set(exp_files + agg_files + other_files))

        logger.info(f"Found {len(all_files)} result files ready to commit:")
        for f in all_files:
            logger.info(f"  - {f.name}")

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

        logger.info("✓ Results ready for GitHub commit")
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
        if exists:
            logger.info(f"Found previous result: {exp_file}")
        else:
            logger.info(f"No previous result found for experiment {exp_id}")
        return exists
