"""Results aggregator - merge and analyze experiment results."""

import json
import logging
from datetime import datetime
from pathlib import Path

from pipeline_types import AggregatedResults, ExperimentResult

__all__ = ["ResultsAggregator"]

logger = logging.getLogger(__name__)


class ResultsAggregator:
    """Aggregate experiment results for paper generation."""

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_experiments(self) -> AggregatedResults | None:
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

    def build_paper_metrics(self, agg: "AggregatedResults") -> dict[str, str]:
        r"""Build \VAR{} key→value map for LaTeX template.

        NOTE: build_var_map in inject_results.py is the canonical implementation.
        Delegates to inject_results.build_var_map — single source of truth.

        Args:
            agg: Aggregated results

        Returns:
            Dictionary of template variable → value
        """
        from inject_results import build_var_map  # lazy to avoid circular import

        all_exp = {str(e.experiment_id): e.__dict__ for e in agg.experiments}
        return build_var_map(all_exp)

    def save_aggregated_results(
        self, agg: AggregatedResults, output_file: Path | None = None
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
    def load_all_experiments(results_dir: Path) -> list[ExperimentResult]:
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
