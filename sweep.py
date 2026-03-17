# =============================================================================
# sweep.py
# Purpose: Merged sweep module — hyperparameter search + multi-seed runner
# Merged from: hparam_search.py + multi_seed_runner.py
# =============================================================================
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import statistics
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


# ---------------------------------------------------------------------------
# Multi-seed runner (from multi_seed_runner.py)
# ---------------------------------------------------------------------------

DEFAULT_SEEDS: list[int] = [42, 123, 7, 99, 2026]


def run_multi_seed(
    experiment_id: int,
    seeds: list[int] | None = None,
    results_dir: str | Path = "results/multi_seed",
    force: bool = False,
) -> dict:
    """Run experiment *experiment_id* with each seed in *seeds*.

    Parameters
    ----------
    experiment_id:
        Experiment to run (from EXPERIMENTS dict in run_experiments.py).
    seeds:
        List of integer seeds.  Defaults to DEFAULT_SEEDS if not provided.
    results_dir:
        Directory to write per-seed and aggregated results.
    force:
        If True, delete cached aggregated result and re-run.

    Returns
    -------
    dict
        Aggregated result with per-field mean ± std of F1 and NED.

    Raises
    ------
    RuntimeError
        When any individual seed fails (OOM, error).  No partial results are
        saved.
    """
    from run_experiments import EXPERIMENTS, run_custom_experiment

    if seeds is None:
        seeds = DEFAULT_SEEDS

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"exp_{experiment_id}_seeds.json"

    if out_file.exists() and not force:
        print(f"[MultiSeed] Cached result found: {out_file}  (use --force to re-run)")
        with open(out_file) as fh:
            return json.load(fh)

    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment_id={experiment_id}. Valid: {list(EXPERIMENTS)}")

    base_config = EXPERIMENTS[experiment_id]
    print(
        f"\n[MultiSeed] Experiment {experiment_id}: {base_config.name}\n"
        f"[MultiSeed] Seeds: {seeds}\n" + "=" * 60
    )

    per_seed_results: list[dict] = []
    start_total = time.monotonic()

    for seed in seeds:
        seed_config = dataclasses.replace(base_config, seed=seed)
        seed_result_file = results_dir / f"exp_{experiment_id}_seed_{seed}.json"

        print(f"\n[MultiSeed] Running seed={seed} ...")
        try:
            result = run_custom_experiment(seed_config, seed_result_file)
        except Exception as exc:
            # GP-1: never leave partial results — abort entirely
            msg = (
                f"[MultiSeed] Seed {seed} FAILED ({type(exc).__name__}: {exc}). "
                "Aborting multi-seed run — no partial results saved."
            )
            print(msg)
            raise RuntimeError(msg) from exc

        per_seed_results.append(result)
        f1 = result.get("metrics", {}).get("global_f1", 0.0)
        print(f"[MultiSeed] Seed {seed}: global_f1={f1:.4f}")

    total_sec = time.monotonic() - start_total

    # ── Aggregate metrics ─────────────────────────────────────────────────
    aggregated: dict = {
        "experiment_id": experiment_id,
        "name": base_config.name,
        "datasets": base_config.datasets,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "total_time_sec": total_sec,
        "metrics": {},
        "per_seed_metrics": [],
    }

    for r in per_seed_results:
        aggregated["per_seed_metrics"].append(r.get("metrics", {}))

    # Compute mean ± std for every metric present in all seed results
    all_metric_keys: set[str] = set()
    for r in per_seed_results:
        all_metric_keys.update(r.get("metrics", {}).keys())

    for key in sorted(all_metric_keys):
        vals = [float(r.get("metrics", {}).get(key, math.nan)) for r in per_seed_results]
        vals_ok = [v for v in vals if not math.isnan(v)]
        if not vals_ok:
            continue
        mean = statistics.mean(vals_ok)
        std = statistics.stdev(vals_ok) if len(vals_ok) > 1 else 0.0
        aggregated["metrics"][key] = {"mean": mean, "std": std}

    out_file.write_text(json.dumps(aggregated, indent=2))
    print(f"\n[MultiSeed] Done — {len(seeds)} seeds in {total_sec / 60:.1f} min")
    print(f"[MultiSeed] Saved: {out_file}")

    global_f1_mean = aggregated["metrics"].get("global_f1", {}).get("mean", 0.0)
    global_f1_std = aggregated["metrics"].get("global_f1", {}).get("std", 0.0)
    print(f"[MultiSeed] Global F1 = {global_f1_mean:.4f} ± {global_f1_std:.4f}")

    return aggregated


def print_multi_seed_table(results_dir: str | Path = "results/multi_seed") -> None:
    """Print a leaderboard table of all multi-seed results.

    Callable from inject_results.py to add a "Robustness" table to the paper.
    """
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("exp_*_seeds.json"))
    if not files:
        print("[MultiSeed] No multi-seed results found in", results_dir)
        return

    header = f"{'Exp':<5} {'Name':<40} {'Seeds':<20} {'F1 mean':<10} {'F1 std':<8}"
    print("\n" + "=" * len(header))
    print("MULTI-SEED ROBUSTNESS TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception:
            continue
        exp_id = data.get("experiment_id", "?")
        name = data.get("name", "")[:38]
        seeds = data.get("seeds", [])
        seeds_str = ",".join(str(s) for s in seeds[:4])
        if len(seeds) > 4:
            seeds_str += "…"
        f1_stats = data.get("metrics", {}).get("global_f1", {})
        mean = f1_stats.get("mean", float("nan"))
        std = f1_stats.get("std", float("nan"))
        print(f"{exp_id:<5} {name:<40} {seeds_str:<20} {mean:<10.4f} {std:<8.4f}")
    print("=" * len(header) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DONUT experiment with multiple seeds to measure variance"
    )
    parser.add_argument(
        "--experiment",
        type=int,
        required=True,
        help="Experiment ID to run (1–8)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help=f"Comma-separated seed list (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--results-dir",
        default="results/multi_seed",
        help="Directory for results (default: results/multi_seed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if cached result exists",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print results table and exit (no training)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.table:
        print_multi_seed_table(args.results_dir)
        return

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_multi_seed(
        experiment_id=args.experiment,
        seeds=seeds,
        results_dir=args.results_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
