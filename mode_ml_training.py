"""Mode B: ML Training Orchestrator - GPU experiment runner on Vast.ai."""

import asyncio
import json
import logging
import subprocess

from git_controller import GitController
from pipeline_config import CloudConfig
from pipeline_types import ExperimentResult, MLTrainingResult
from results_aggregator import ResultsAggregator
from storage_manager import StorageManager

__all__ = ["MLTrainingOrchestrator"]

logger = logging.getLogger(__name__)


class MLTrainingOrchestrator:
    """Orchestrate DONUT + TrOCR+YOLO training on GPU."""

    def __init__(self, config: CloudConfig):
        self.config = config
        self.logger = logger
        self.storage_manager = StorageManager(config.results_dir)
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

            # Run DONUT experiments
            self.logger.info("\n" + "=" * 70)
            self.logger.info("DONUT EXPERIMENTS")
            self.logger.info("=" * 70)

            experiments_run: dict[int, ExperimentResult] = {}

            for exp_id in self.config.experiments_to_run:
                self.logger.info(f"\n▶ Running Experiment {exp_id}...")

                # Call run_all.py --experiment N
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
                            timeout=3600,  # 1 hour timeout per experiment
                        )
                    )

                    if proc.returncode == 0:
                        self.logger.info(f"✓ Experiment {exp_id} completed")

                        # Load result
                        result_file = self.config.results_dir / f"experiment_{exp_id}.json"
                        if result_file.exists():
                            try:
                                data = json.loads(result_file.read_text())
                                from pipeline_types import ExperimentMetrics

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

                        # Strict mode: abort on first failure
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
                        cmd = ["python", "inject_results.py", "--all"]
                        proc = await asyncio.create_task(
                            asyncio.to_thread(
                                subprocess.run,
                                cmd,
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
