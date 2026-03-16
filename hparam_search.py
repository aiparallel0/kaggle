# =============================================================================
# hparam_search.py
# Purpose: Optuna hyperparameter search for DONUT SROIE fine-tuning
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
hparam_search.py — Optuna-based hyperparameter sweep.

Sweeps over lr, batch_size, warmup_steps, and weight_decay using
Experiment 2 (SROIE + WildReceipt) as the default target.

Each trial calls ``run_custom_experiment()`` from ``run_experiments.py``
and optimises two objectives:
  1. Maximize global F1 (higher is better).
  2. Minimize validation loss (lower is better).

Failed trials (OOM, error) are marked as F1=0.0 / loss=inf.

Results are stored in ``results/optuna/study.db`` (SQLite).

Usage
-----
    python hparam_search.py
    python hparam_search.py --n-trials 30 --experiment 6
    python hparam_search.py --n-trials 20 --study-name my_study

Install optional dependency
---------------------------
    pip install "optuna>=3.0.0"
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


def _ensure_optuna():
    """Import optuna or raise a clear ImportError."""
    try:
        import optuna  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "optuna is not installed. Install it with: pip install 'optuna>=3.0.0'"
        ) from exc
    return optuna


def run_hparam_search(
    experiment_id: int = 2,
    n_trials: int = 20,
    study_name: str = "donut_sroie_sweep",
    results_dir: str | Path = "results/optuna",
    timeout_seconds: int | None = None,
) -> optuna.Study:
    """Run Optuna multi-objective hyperparameter sweep.

    Parameters
    ----------
    experiment_id:
        Which EXPERIMENTS entry to use as the base config (default: 2).
    n_trials:
        Number of Optuna trials to run (default: 20).
    study_name:
        Name of the Optuna study (default: "donut_sroie_sweep").
    results_dir:
        Directory where the SQLite study DB is stored
        (default: "results/optuna").
    timeout_seconds:
        Optional wall-clock timeout in seconds.

    Returns
    -------
    optuna.Study
        The completed multi-objective study.
    """
    import optuna as _optuna

    from run_experiments import EXPERIMENTS, ExperimentConfig, run_custom_experiment

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = results_dir / "study.db"
    storage = f"sqlite:///{db_path}"

    base_config = EXPERIMENTS.get(experiment_id)
    if base_config is None:
        raise ValueError(f"Unknown experiment_id={experiment_id}. Valid: {list(EXPERIMENTS)}")

    # Suppress Optuna's per-trial INFO logs in non-verbose mode
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)

    def objective(trial: Any) -> tuple[float, float]:
        """Return (global_f1, val_loss) — maximise f1, minimise loss."""
        lr = trial.suggest_float("lr", 1e-5, 2e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        warmup_steps = trial.suggest_categorical("warmup_steps", [20, 40, 80])
        weight_decay = trial.suggest_categorical("weight_decay", [0.001, 0.01, 0.1])

        sweep_config: ExperimentConfig = dataclasses.replace(
            base_config,
            lr=lr,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            experiment_id=experiment_id,
        )

        timestamp = int(time.time())
        result_file = results_dir / f"trial_{trial.number}_{timestamp}.json"

        try:
            result = run_custom_experiment(sweep_config, result_file)
        except Exception as exc:
            logger.warning(
                "[HparamSearch] Trial %d failed (%s: %s) — marking as F1=0.0",
                trial.number,
                type(exc).__name__,
                exc,
            )
            return 0.0, float("inf")

        metrics = result.get("metrics", {})
        global_f1 = float(metrics.get("global_f1", 0.0))
        # Use training loss as proxy for val loss when val_loss not available
        val_loss = float(metrics.get("eval_loss", metrics.get("val_loss", float("inf"))))
        logger.info(
            "[HparamSearch] Trial %d: lr=%.2e bs=%d warmup=%d wd=%.3f → F1=%.4f loss=%.4f",
            trial.number,
            lr,
            batch_size,
            warmup_steps,
            weight_decay,
            global_f1,
            val_loss,
        )
        return global_f1, val_loss

    study = _optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=["maximize", "minimize"],
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        catch=(Exception,),
    )

    print(f"\n[HparamSearch] Completed {len(study.trials)} trial(s)")
    print(f"[HparamSearch] Pareto-front size: {len(study.best_trials)}")
    for t in study.best_trials[:5]:
        print(f"  Trial {t.number}: values={t.values}  params={t.params}")
    print(f"[HparamSearch] Study DB: {db_path}")

    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep for DONUT SROIE")
    parser.add_argument(
        "--experiment",
        type=int,
        default=2,
        help="Base experiment ID to sweep over (default: 2)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials (default: 20)",
    )
    parser.add_argument(
        "--study-name",
        default="donut_sroie_sweep",
        help="Optuna study name (default: donut_sroie_sweep)",
    )
    parser.add_argument(
        "--results-dir",
        default="results/optuna",
        help="Directory for study.db and trial JSONs (default: results/optuna)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Wall-clock timeout in seconds (default: unlimited)",
    )
    args = parser.parse_args()

    try:
        _ensure_optuna()
    except ImportError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_hparam_search(
        experiment_id=args.experiment,
        n_trials=args.n_trials,
        study_name=args.study_name,
        results_dir=args.results_dir,
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    main()
