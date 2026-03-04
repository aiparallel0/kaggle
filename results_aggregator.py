"""Results aggregator - merge and analyze experiment results."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from pipeline_types import AggregatedResults, ExperimentResult

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """Aggregate experiment results for paper generation."""

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_experiments(self) -> Optional[AggregatedResults]:
        """Load and aggregate all experiment results.

        Returns:
            AggregatedResults or None if no experiments found
        """
        logger.info("Aggregating experiment results...")

        # Load all experiment_N.json files
        experiment_files = sorted(self.results_dir.glob("experiment_*.json"))

        if not experiment_files:
            logger.warning("No experiment results found")
            return None

        experiments = []
        for exp_file in experiment_files:
            try:
                data = json.loads(exp_file.read_text())
                # Convert dict to ExperimentResult if needed
                if isinstance(data, dict):
                    from pipeline_types import ExperimentMetrics

                    metrics = ExperimentMetrics(**data.get("metrics", {}))
                    exp = ExperimentResult(
                        experiment_id=data["experiment_id"],
                        name=data["name"],
                        datasets=data["datasets"],
                        num_train_samples=data["num_train_samples"],
                        metrics=metrics,
                    )
                    experiments.append(exp)
            except Exception as e:
                logger.warning(f"Could not load {exp_file}: {e}")

        if not experiments:
            logger.warning("No valid experiments loaded")
            return None

        # Find best experiment by F1
        best_exp = max(experiments, key=lambda e: e.metrics.global_f1)
        baseline_exp = next((e for e in experiments if e.experiment_id == 1), None)
        baseline_f1 = baseline_exp.metrics.global_f1 if baseline_exp else 0.0
        improvement = best_exp.metrics.global_f1 - baseline_f1

        # Create aggregated result
        agg = AggregatedResults(
            experiments=experiments,
            best_experiment=best_exp,
            baseline_f1=baseline_f1,
            improvement=improvement,
            generated_timestamp=datetime.utcnow(),
        )

        logger.info(
            f"✓ Aggregated {len(experiments)} experiments. "
            f"Best: Exp {best_exp.experiment_id} F1={best_exp.metrics.global_f1:.4f}"
        )

        return agg

    def build_paper_metrics(self, agg: AggregatedResults) -> Dict[str, str]:
        """Build \VAR{} key→value map for LaTeX template.

        Args:
            agg: Aggregated results

        Returns:
            Dictionary of template variable → value
        """
        logger.info("Building paper metrics...")

        metrics = {}

        # Best experiment metrics
        if agg.best_experiment:
            metrics["best_exp_id"] = str(agg.best_experiment.experiment_id)
            metrics["best_exp_name"] = agg.best_experiment.name
            metrics["best_exp_f1"] = f"{agg.best_experiment.metrics.global_f1:.4f}"
            metrics["best_exp_precision"] = f"{agg.best_experiment.metrics.global_precision:.4f}"
            metrics["best_exp_recall"] = f"{agg.best_experiment.metrics.global_recall:.4f}"

        # Baseline (Exp 1)
        metrics["baseline_f1"] = f"{agg.baseline_f1:.4f}"
        metrics["improvement"] = f"{agg.improvement:+.4f}"
        metrics["improvement_percent"] = (
            f"{(agg.improvement / agg.baseline_f1 * 100):+.1f}%" if agg.baseline_f1 > 0 else "0%"
        )

        # Individual experiment results
        for exp in agg.experiments:
            base = f"exp{exp.experiment_id}"
            metrics[f"{base}_name"] = exp.name
            metrics[f"{base}_f1"] = f"{exp.metrics.global_f1:.4f}"
            metrics[f"{base}_datasets"] = ", ".join(exp.datasets)

        logger.info(f"Built {len(metrics)} paper metrics")

        return metrics

    def save_aggregated_results(
        self, agg: AggregatedResults, output_file: Optional[Path] = None
    ) -> bool:
        """Save aggregated results to JSON.

        NOTE: all_experiments.json is owned by run_experiments.save_summary() —
        do not write to it from this class.

        Args:
            agg: Aggregated results
            output_file: Output path (default: aggregated_summary.json)

        Returns:
            True if successful
        """
        if output_file is None:
            output_file = self.results_dir / "aggregated_summary.json"

        try:
            data = {
                "generated_timestamp": agg.generated_timestamp.isoformat(),
                "num_experiments": len(agg.experiments),
                "best_experiment": (
                    {
                        "experiment_id": agg.best_experiment.experiment_id,
                        "name": agg.best_experiment.name,
                        "f1": agg.best_experiment.metrics.global_f1,
                    }
                    if agg.best_experiment
                    else None
                ),
                "baseline_f1": agg.baseline_f1,
                "improvement": agg.improvement,
                "experiments": [e.to_dict() for e in agg.experiments],
            }

            output_file.write_text(json.dumps(data, indent=2))
            logger.info(f"✓ Saved aggregated results to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Could not save aggregated results: {e}")
            return False

    @staticmethod
    def load_all_experiments(results_dir: Path) -> List[ExperimentResult]:
        """Load all experiments from results directory.

        Args:
            results_dir: Path to results directory

        Returns:
            List of ExperimentResult objects
        """
        experiments = []
        exp_files = sorted(results_dir.glob("experiment_*.json"))

        for exp_file in exp_files:
            try:
                data = json.loads(exp_file.read_text())
                from pipeline_types import ExperimentMetrics

                metrics = ExperimentMetrics(**data.get("metrics", {}))
                exp = ExperimentResult(
                    experiment_id=data["experiment_id"],
                    name=data["name"],
                    datasets=data["datasets"],
                    num_train_samples=data["num_train_samples"],
                    metrics=metrics,
                )
                experiments.append(exp)
            except Exception as e:
                logger.warning(f"Could not load {exp_file}: {e}")

        return experiments
