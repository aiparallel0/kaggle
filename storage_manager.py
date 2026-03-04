"""Storage manager - simplified for GitHub commits (user preferred).

From user feedback: "A Folder to put results at GitHub is sufficient."
Results are committed directly to the feature branch via git.
No S3/GCS needed for MVP.
"""

import logging
from pathlib import Path

from pipeline_types import SyncReport, UploadReport

logger = logging.getLogger(__name__)


class StorageManager:
    """Manage result storage via git commits to GitHub."""

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def sync_results_directory(self, remote_prefix: str = "") -> SyncReport:
        """Sync results/ directory to GitHub via git.

        Since user chose GitHub as primary storage, this is just a report
        that results are ready to commit.

        Args:
            remote_prefix: (unused, for API compatibility)

        Returns:
            SyncReport showing what would be committed
        """
        logger.info("Preparing results for GitHub commit...")

        # Find experiment results
        exp_files = list(self.results_dir.glob("experiment_*.json"))
        agg_files = list(self.results_dir.glob("all_experiments.json"))
        other_files = list(self.results_dir.glob("*.json"))

        all_files = exp_files + agg_files + other_files
        all_files = list(set(all_files))  # Remove duplicates

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
        """Prepare experiment result for GitHub commit.

        Args:
            exp_id: Experiment ID

        Returns:
            UploadReport (always successful for GitHub)
        """
        exp_file = self.results_dir / f"experiment_{exp_id}.json"

        if exp_file.exists():
            return UploadReport(
                success=True,
                local_path=exp_file,
                remote_path=f"results/experiment_{exp_id}.json",
                storage_type="github",
                size_bytes=exp_file.stat().st_size,
            )
        else:
            return UploadReport(
                success=False,
                local_path=exp_file,
                remote_path=f"results/experiment_{exp_id}.json",
                storage_type="github",
                error=f"File not found: {exp_file}",
            )

    async def upload_paper(self, paper_path: Path) -> UploadReport:
        """Prepare paper for GitHub commit.

        Args:
            paper_path: Path to generated paper

        Returns:
            UploadReport
        """
        if paper_path.exists():
            return UploadReport(
                success=True,
                local_path=paper_path,
                remote_path=f"results/{paper_path.name}",
                storage_type="github",
                size_bytes=paper_path.stat().st_size,
            )
        else:
            return UploadReport(
                success=False,
                local_path=paper_path,
                remote_path=f"results/{paper_path.name}",
                storage_type="github",
                error=f"File not found: {paper_path}",
            )

    async def download_previous_results(self, exp_id: int) -> bool:
        """Download previous results for comparison (stub - GitHub only).

        Args:
            exp_id: Experiment ID

        Returns:
            True if file exists locally
        """
        exp_file = self.results_dir / f"experiment_{exp_id}.json"
        exists = exp_file.exists()

        if exists:
            logger.info(f"Found previous result: {exp_file}")
        else:
            logger.info(f"No previous result found for experiment {exp_id}")

        return exists
