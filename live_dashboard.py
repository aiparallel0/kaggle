# =============================================================================
# live_dashboard.py
# Purpose: Live training dashboard callback using rich
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
live_dashboard.py — ``LiveDashboardCallback`` for DonutTrainer.

After every epoch, appends a row to ``convergence_expN.csv`` and (when
``rich`` is installed) redraws a live table showing:
  - Epoch number
  - Train loss
  - Validation loss
  - Best F1 so far

The callback is always-on when ``rich`` is installed.  Set
``DISABLE_LIVE_DASHBOARD=1`` in the environment to suppress it.

The callback is wired into ``DonutTrainer.__init__`` in ``train.py``
automatically — no explicit registration is required from user code.

Install optional dependency
---------------------------
    pip install "rich>=13.0.0"
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _EpochRow:
    epoch: int
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    best_f1: float = float("nan")


class LiveDashboardCallback:
    """Trainer callback that writes CSV rows and optionally redraws a rich table.

    Compatible with HuggingFace ``TrainerCallback`` interface: the callback
    class inherits from ``transformers.TrainerCallback`` lazily (at __init__
    time) to avoid importing transformers at module level.

    Parameters
    ----------
    csv_path:
        Path to the CSV convergence log file.  Defaults to
        ``convergence_exp{experiment_id}.csv`` in the current directory.
    experiment_id:
        Integer experiment ID used for the default CSV filename.
    use_rich:
        If True and ``rich`` is installed, render a live table.
        Set to False (or set env var DISABLE_LIVE_DASHBOARD=1) to disable.
    """

    def __init__(
        self,
        csv_path: str | Path | None = None,
        experiment_id: int = 0,
        use_rich: bool | None = None,
    ) -> None:
        # Lazy import of TrainerCallback to avoid top-level transformers import
        try:
            from transformers import TrainerCallback
            self.__class__ = type(
                "LiveDashboardCallback",
                (self.__class__, TrainerCallback),
                {},
            )
        except ImportError:
            pass  # standalone usage without transformers (tests etc.)

        if csv_path is None:
            csv_path = f"convergence_exp{experiment_id}.csv"
        self._csv_path = Path(csv_path)
        self._experiment_id = experiment_id
        self._rows: list[_EpochRow] = []
        self._best_f1: float = float("nan")

        # Determine whether to use rich
        if use_rich is None:
            use_rich = os.environ.get("DISABLE_LIVE_DASHBOARD", "0") != "1"

        self._rich_enabled = False
        self._live = None
        if use_rich:
            try:
                import rich  # noqa: F401
                self._rich_enabled = True
            except ImportError:
                pass

        # Write CSV header if file doesn't exist
        if not self._csv_path.exists():
            try:
                self._csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._csv_path, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["epoch", "train_loss", "val_loss", "best_f1"])
            except OSError as exc:
                logger.warning("[LiveDashboard] Could not create CSV %s: %s", self._csv_path, exc)

    # ── HuggingFace TrainerCallback interface ─────────────────────────────

    def on_epoch_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Called at the end of each epoch by HuggingFace Trainer."""
        log = state.log_history if state is not None else []
        train_loss = float("nan")
        val_loss = float("nan")

        # Walk log history in reverse to find the most recent train/val entries
        for entry in reversed(log):
            if "loss" in entry and train_loss != train_loss:  # nan check
                train_loss = float(entry["loss"])
            if "eval_loss" in entry and val_loss != val_loss:  # nan check
                val_loss = float(entry["eval_loss"])
            if train_loss == train_loss and val_loss == val_loss:
                break

        epoch = int(getattr(state, "epoch", len(self._rows) + 1))
        row = _EpochRow(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        self._rows.append(row)

        # Append to CSV
        try:
            with open(self._csv_path, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, train_loss, val_loss, self._best_f1])
        except OSError as exc:
            logger.warning("[LiveDashboard] CSV write failed: %s", exc)

        # Redraw rich table if enabled
        if self._rich_enabled:
            self._redraw()

    def update_best_f1(self, f1: float) -> None:
        """Update the best F1 seen so far (called from the evaluator)."""
        if f1 > self._best_f1 or self._best_f1 != self._best_f1:  # nan check
            self._best_f1 = f1
            if self._rows:
                self._rows[-1].best_f1 = f1

    def _redraw(self) -> None:
        """Redraw the rich live table."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title=f"Experiment {self._experiment_id} — Training Progress",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Epoch", justify="right", style="dim")
            table.add_column("Train Loss", justify="right")
            table.add_column("Val Loss", justify="right")
            table.add_column("Best F1", justify="right", style="green")

            for r in self._rows[-10:]:  # show last 10 epochs
                tl = f"{r.train_loss:.4f}" if r.train_loss == r.train_loss else "—"
                vl = f"{r.val_loss:.4f}" if r.val_loss == r.val_loss else "—"
                bf = f"{r.best_f1:.4f}" if r.best_f1 == r.best_f1 else "—"
                table.add_row(str(r.epoch), tl, vl, bf)

            console.print(table)
        except Exception as exc:
            logger.debug("[LiveDashboard] rich redraw failed: %s", exc)

    def close(self) -> None:
        """Finalize the dashboard (no-op when rich.live.Live is not used)."""
        pass
