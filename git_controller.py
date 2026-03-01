"""Git operations controller - branch/commit management."""

import subprocess
import logging
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

from types import GitCommitReport

logger = logging.getLogger(__name__)


class GitController:
    """Manage git operations for the pipeline."""

    @staticmethod
    def get_current_branch() -> Optional[str]:
        """Get current git branch name.

        Returns:
            Branch name or None if not in git repo
        """
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
        """Checkout a git branch, creating if necessary.

        Args:
            branch_name: Branch name to checkout

        Returns:
            True if successful
        """
        try:
            # Try to checkout existing branch
            result = subprocess.run(
                ["git", "checkout", branch_name],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                logger.info(f"✓ Checked out branch: {branch_name}")
                return True

            # If branch doesn't exist, create it
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
    def commit(message: str, files: Optional[list] = None) -> GitCommitReport:
        """Create a git commit.

        Args:
            message: Commit message
            files: List of files to stage (None = all)

        Returns:
            GitCommitReport with results
        """
        try:
            # Stage files
            if files is None:
                cmd = ["git", "add", "-A"]
            else:
                cmd = ["git", "add"] + files

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return GitCommitReport(
                    success=False, error=f"Stage failed: {result.stderr}"
                )

            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", message],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Extract commit hash
                output_lines = result.stdout.split("\n")
                commit_hash = None
                for line in output_lines:
                    if line.startswith("["):
                        # Format: [branch_name hash] message
                        parts = line.split()
                        if len(parts) >= 2:
                            commit_hash = parts[1].rstrip("]")
                            break

                logger.info(f"✓ Committed: {message}")
                logger.info(f"  Hash: {commit_hash}")

                return GitCommitReport(
                    success=True,
                    commit_hash=commit_hash,
                    message=message,
                    branch=GitController.get_current_branch(),
                )

            elif "nothing to commit" in result.stdout.lower():
                logger.warning("Nothing to commit")
                return GitCommitReport(
                    success=True, message="Nothing to commit", branch=GitController.get_current_branch()
                )
            else:
                return GitCommitReport(
                    success=False, error=f"Commit failed: {result.stderr}"
                )

        except Exception as e:
            logger.error(f"Git commit error: {e}")
            return GitCommitReport(success=False, error=str(e))

    @staticmethod
    def push_branch(branch_name: str, force: bool = False) -> bool:
        """Push branch to origin.

        Args:
            branch_name: Branch to push
            force: If True, use --force-with-lease

        Returns:
            True if successful
        """
        try:
            cmd = ["git", "push", "-u", "origin", branch_name]
            if force:
                cmd.insert(2, "--force-with-lease")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logger.info(f"✓ Pushed branch: {branch_name}")
                return True
            else:
                logger.error(f"Push failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Git push error: {e}")
            return False

    @staticmethod
    def tag_commit(tag_name: str, message: str = "") -> bool:
        """Create a git tag for current commit.

        Args:
            tag_name: Tag name
            message: Tag message (optional)

        Returns:
            True if successful
        """
        try:
            if message:
                cmd = ["git", "tag", "-a", tag_name, "-m", message]
            else:
                cmd = ["git", "tag", tag_name]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                logger.info(f"✓ Created tag: {tag_name}")
                return True
            else:
                logger.error(f"Tag creation failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Git tag error: {e}")
            return False

    @staticmethod
    def get_status() -> str:
        """Get git status.

        Returns:
            Status output
        """
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
