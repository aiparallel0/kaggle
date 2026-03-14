# =============================================================================
# dag_scheduler.py
# Purpose: DAG-based parallel experiment scheduler for multi-GPU setups
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
dag_scheduler.py — Concurrent experiment scheduler with dependency resolution.

Reads ``depends_on`` fields from experiment YAML configs (loaded via
``experiment_config_loader``) and schedules experiments so that all
dependencies finish before a dependent experiment starts.

On a multi-GPU setup (``torch.cuda.device_count() > 1``), experiments with
no unmet dependencies run concurrently using
``concurrent.futures.ThreadPoolExecutor``.  On a single-GPU or CPU-only
setup, experiments are serialised (max_workers=1) so no performance is lost.

The ``--parallel`` flag in ``run_all.py`` enables this scheduler.  Without
the flag, ``stage_experiments()`` uses the existing serial loop.

DAG Dependency Format (YAML)
-----------------------------
In any experiment YAML under ``experiments/``, add a top-level key::

    depends_on: [1, 2]   # this experiment requires experiments 1 and 2 to finish first

If ``depends_on`` is absent the experiment has no prerequisites.

Usage (from run_all.py)
-----------------------
    from dag_scheduler import DAGScheduler
    scheduler = DAGScheduler(configs, run_fn)
    scheduler.run()

Standalone
----------
    python dag_scheduler.py --list          # print dependency graph
"""
from __future__ import annotations

import concurrent.futures
import logging
from collections import deque
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class DAGScheduler:
    """Schedule experiments respecting depends_on constraints.

    Parameters
    ----------
    configs:
        Ordered list of experiment config objects.  Each config must have
        an ``id`` attribute.  Optionally has ``depends_on: list[int]``.
    run_fn:
        Callable that runs a single experiment: ``run_fn(cfg) -> Any``.
        Called in a thread — must be thread-safe for parallel execution.
    max_workers:
        Maximum number of concurrent workers (threads).  ``None`` means
        auto-detect based on available GPUs (falls back to 1 on single-GPU).
    check_vram:
        If True, consult ``resource_optimizer.detect_system_resources()``
        before scheduling to cap concurrency to ``int(vram_gb / 8)``
        (each DONUT experiment needs ~8 GB VRAM at batch_size=8).
    """

    def __init__(
        self,
        configs: list,
        run_fn: Callable[[Any], Any],
        max_workers: int | None = None,
        check_vram: bool = True,
    ) -> None:
        self._configs = configs
        self._run_fn = run_fn
        self._id_to_cfg = {cfg.id: cfg for cfg in configs}

        if max_workers is None:
            max_workers = self._detect_workers(check_vram)
        self._max_workers = max(1, max_workers)
        logger.info("[DAGScheduler] max_workers=%d", self._max_workers)

    # ── Worker detection ──────────────────────────────────────────────────

    def _detect_workers(self, check_vram: bool) -> int:
        """Return max safe concurrent workers."""
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except ImportError:
            num_gpus = 0

        if num_gpus <= 1:
            return 1  # serial on single-GPU / CPU

        if check_vram:
            try:
                from resource_optimizer import detect_system_resources
                res = detect_system_resources()
                # Each DONUT experiment needs ~8 GB VRAM
                vram_workers = max(1, int(res.vram_gb / 8))
                workers = min(num_gpus, vram_workers)
                logger.info(
                    "[DAGScheduler] VRAM=%.1f GB → capping to %d worker(s)",
                    res.vram_gb, workers,
                )
                return workers
            except Exception as exc:
                logger.warning("[DAGScheduler] VRAM detection failed: %s", exc)

        return num_gpus

    # ── Dependency resolution ─────────────────────────────────────────────

    def _build_dependency_graph(self) -> dict[int, set[int]]:
        """Return {exp_id: set_of_dependency_ids}."""
        graph: dict[int, set[int]] = {}
        for cfg in self._configs:
            deps = set(getattr(cfg, "depends_on", []) or [])
            # Only include dependencies that are in our config set
            deps = {d for d in deps if d in self._id_to_cfg}
            graph[cfg.id] = deps
        return graph

    def _topological_order(self, graph: dict[int, set[int]]) -> list[int]:
        """Return experiment IDs in topological (dependency-first) order."""
        in_degree = {eid: len(deps) for eid, deps in graph.items()}
        queue: deque[int] = deque(eid for eid, deg in in_degree.items() if deg == 0)
        order: list[int] = []

        # Build reverse edges (who depends on me?)
        dependents: dict[int, list[int]] = {eid: [] for eid in graph}
        for eid, deps in graph.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].append(eid)

        while queue:
            eid = queue.popleft()
            order.append(eid)
            for dependent in dependents.get(eid, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(order) != len(graph):
            cycle_ids = [eid for eid in graph if eid not in order]
            raise ValueError(
                f"[DAGScheduler] Cyclic dependency detected involving IDs: {cycle_ids}"
            )
        return order

    # ── Main scheduler ────────────────────────────────────────────────────

    def run(self) -> dict[int, Any]:
        """Execute all experiments respecting dependencies.

        Returns
        -------
        dict[int, Any]
            Mapping from experiment ID to the return value of ``run_fn``.
        """
        graph = self._build_dependency_graph()
        order = self._topological_order(graph)

        results: dict[int, Any] = {}
        completed: set[int] = set()
        failed: set[int] = set()

        remaining = list(order)  # process in topological order

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            futures: dict[concurrent.futures.Future, int] = {}

            def _submit_ready() -> None:
                """Submit any experiment whose dependencies are satisfied."""
                for eid in list(remaining):
                    deps = graph[eid]
                    if failed & deps:
                        logger.warning(
                            "[DAGScheduler] Exp %d skipped — dependency failed: %s",
                            eid, failed & deps,
                        )
                        failed.add(eid)
                        remaining.remove(eid)
                        continue
                    if deps <= completed:
                        cfg = self._id_to_cfg[eid]
                        logger.info("[DAGScheduler] Submitting Exp %d: %s", eid, cfg.name)
                        fut = executor.submit(self._run_fn, cfg)
                        futures[fut] = eid
                        remaining.remove(eid)

            _submit_ready()

            while futures:
                done, _ = concurrent.futures.wait(
                    futures.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    eid = futures.pop(fut)
                    try:
                        result = fut.result()
                        results[eid] = result
                        completed.add(eid)
                        logger.info("[DAGScheduler] Exp %d completed", eid)
                    except Exception as exc:
                        logger.error(
                            "[DAGScheduler] Exp %d FAILED: %s: %s",
                            eid, type(exc).__name__, exc,
                        )
                        failed.add(eid)
                        results[eid] = {"error": str(exc)}
                _submit_ready()

        if failed:
            logger.warning("[DAGScheduler] %d experiment(s) failed: %s", len(failed), failed)
        logger.info("[DAGScheduler] Done. %d completed, %d failed", len(completed), len(failed))
        return results


def print_dag(configs: list) -> None:
    """Print the experiment dependency graph to stdout."""
    print("\nExperiment DAG:")
    print("=" * 60)
    for cfg in configs:
        deps = getattr(cfg, "depends_on", []) or []
        dep_str = f" ← depends on {deps}" if deps else ""
        print(f"  [{cfg.id:2d}] {getattr(cfg, 'name', '')[:40]}{dep_str}")
    print("=" * 60 + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Show or test the experiment DAG")
    parser.add_argument(
        "--list", action="store_true",
        help="List all experiments with their dependencies and exit",
    )
    parser.add_argument(
        "--experiments-dir", default="experiments",
        help="Path to YAML experiments directory (default: experiments/)",
    )
    args = parser.parse_args()

    try:
        from experiment_config_loader import load_all_experiments
        configs = load_all_experiments(args.experiments_dir)
    except Exception as exc:
        print(f"[DAGScheduler] Could not load experiments: {exc}")
        raise SystemExit(1) from exc

    print_dag(configs)  # always show the graph


if __name__ == "__main__":
    main()
