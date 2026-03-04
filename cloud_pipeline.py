"""Cloud AI Agent Pipeline - unified orchestrator for code repair and ML training.

Main entry point for the entire cloud pipeline system.
Supports two modes:
  - Mode A: Code Repair (Ollama-based bug fixing)
  - Mode B: ML Training (Vast.ai GPU experiments)

Usage:
  python cloud_pipeline.py --mode ml_training --experiments 1 2 3
  python cloud_pipeline.py --mode code_repair --dry-run
  python cloud_pipeline.py --mode auto  (detect from git branch)
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime

# Set up logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from pipeline_config import CloudConfig, PipelineMode
from pipeline_types import PipelineResult
from preflight_checks import PreflightChecker


class CloudPipelineOrchestrator:
    """Main pipeline orchestrator - routes between modes and manages execution."""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def run(self) -> PipelineResult:
        """Execute pipeline based on configured mode.

        Returns:
            PipelineResult with overall status
        """
        start_time = datetime.utcnow()

        self.logger.info("=" * 70)
        self.logger.info("CLOUD PIPELINE STARTING")
        self.logger.info("=" * 70)
        self.logger.info(f"Mode: {self.config.mode.value}")
        self.logger.info(f"Branch: {self.config.git_branch}")
        self.logger.info(f"Workspace: {self.config.workspace}")

        # Validate configuration
        is_valid, errors = self.config.validate()
        if not errors:
            self.logger.info("✓ Configuration valid")
        else:
            for error in errors:
                self.logger.warning(f"  {error}")

        # CRITICAL: Run preflight checks first
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

        # Determine mode
        if self.config.mode == PipelineMode.AUTO:
            # Auto-detect from branch name
            self.logger.info("Auto-detecting mode from branch...")
            if "code-repair" in self.config.git_branch:
                mode = PipelineMode.CODE_REPAIR
            elif "ml-train" in self.config.git_branch:
                mode = PipelineMode.ML_TRAINING
            else:
                mode = PipelineMode.ML_TRAINING  # Default
            self.logger.info(f"  Detected: {mode.value}")
        else:
            mode = self.config.mode

        # Route to appropriate mode
        if mode == PipelineMode.CODE_REPAIR:
            self.logger.info("\n▶ Starting MODE A: Code Repair (Ollama)")
            try:
                from mode_code_repair import CodeRepairOrchestrator

                orchestrator = CodeRepairOrchestrator(self.config)
                mode_result = await orchestrator.run()
                success = mode_result.success
            except ImportError as e:
                self.logger.error(f"Code repair mode not available: {e}")
                mode_result = None
                success = False

        elif mode == PipelineMode.ML_TRAINING:
            self.logger.info("\n▶ Starting MODE B: ML Training (Vast.ai)")
            try:
                from mode_ml_training import MLTrainingOrchestrator

                orchestrator = MLTrainingOrchestrator(self.config)
                mode_result = await orchestrator.run()
                success = mode_result.success
            except ImportError as e:
                self.logger.error(f"ML training mode not available: {e}")
                mode_result = None
                success = False
        else:
            self.logger.error(f"Unknown mode: {mode}")
            return PipelineResult(
                success=False, mode="unknown", errors=["Unknown pipeline mode"]
            )

        # Finalize
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


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cloud AI Agent Pipeline: code repair + ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ML training mode (all experiments)
  python cloud_pipeline.py --mode ml_training

  # Code repair mode
  python cloud_pipeline.py --mode code_repair

  # Single experiment, skip TrOCR
  python cloud_pipeline.py --experiments 1 --skip-trocr

  # Dry-run (no execution)
  python cloud_pipeline.py --dry-run

  # Manual branch selection
  python cloud_pipeline.py --branch my-feature-branch
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["code_repair", "ml_training", "auto"],
        default="auto",
        help="Pipeline execution mode (default: auto-detect from branch)",
    )

    # Experiment selection
    parser.add_argument(
        "--experiments",
        type=int,
        nargs="+",
        help="Experiment IDs to run (default: 1-8)",
    )

    # Skip options
    parser.add_argument(
        "--skip-trocr",
        action="store_true",
        help="Skip TrOCR+YOLO stages in ML training",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip pre-flight checks (dangerous)",
    )
    parser.add_argument(
        "--skip-pretrained",
        action="store_true",
        help="Skip pretrained baseline evaluation",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan execution without running",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Skip git commits",
    )

    # Configuration
    parser.add_argument(
        "--branch",
        help="Git branch to work on",
    )
    parser.add_argument(
        "--workspace",
        help="Workspace directory",
    )

    # Cloud storage (note: user chose GitHub, so these are optional)
    parser.add_argument(
        "--s3-bucket",
        help="AWS S3 bucket for results (optional)",
    )
    parser.add_argument(
        "--gcs-bucket",
        help="Google Cloud Storage bucket (optional)",
    )

    args = parser.parse_args()

    # Load configuration
    config = CloudConfig.from_args_and_env(args)

    # Log configuration
    logger.info(f"Configuration loaded: {config.mode.value} mode")
    logger.info(f"  Workspace: {config.workspace}")
    logger.info(f"  Results dir: {config.results_dir}")
    logger.info(f"  Branch: {config.git_branch}")

    # Run pipeline
    orchestrator = CloudPipelineOrchestrator(config)

    try:
        result = await orchestrator.run()

        if result.success:
            sys.exit(0)
        elif result.partial_success:
            # Partial success (e.g., some experiments completed)
            logger.warning("Partial success - some operations completed")
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
