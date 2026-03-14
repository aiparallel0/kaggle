# =============================================================================
# tui_console.py
# Purpose: Optional Textual TUI for interactive experiment selection
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
tui_console.py — Textual-based TUI for interactive experiment selection.

When ``textual`` is installed, ``select_experiments_tui()`` presents a
checkable list of all experiments with live VRAM information and returns the
user's selection.  When ``textual`` is not installed, the function raises
``ImportError`` and the caller falls back to the existing plain-text prompt.

Usage (from run_all.py)
-----------------------
    try:
        from tui_console import select_experiments_tui
        selected = select_experiments_tui(all_configs)
    except ImportError:
        selected = _plain_text_selection(all_configs)

Install optional dependency
---------------------------
    pip install "textual>=0.40.0"
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # only used for type annotations


def select_experiments_tui(all_configs: list) -> list:
    """Display a Textual TUI and return the user-selected experiment configs.

    Parameters
    ----------
    all_configs:
        List of experiment config objects (must have .id, .name, and optionally
        .arch_type attributes).

    Returns
    -------
    list
        Filtered list of configs chosen by the user.  Returns *all_configs* if
        the user presses "Run All".  Returns an empty list if the user cancels.

    Raises
    ------
    ImportError
        When ``textual`` is not installed.  The caller should catch this and
        fall back to the plain-text prompt.
    """
    try:
        from textual.app import App, ComposeResult  # noqa: F401
        from textual.widgets import Button, Checkbox, Footer, Header, Label, Static  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "textual is not installed. Install it with: pip install 'textual>=0.40.0'"
        ) from exc

    # ── VRAM info ──────────────────────────────────────────────────────────
    vram_info = ""
    try:
        from resource_optimizer import detect_system_resources
        res = detect_system_resources()
        vram_info = f"  VRAM: {res.vram_gb:.1f} GB free  |  RAM: {res.ram_gb:.1f} GB"
    except Exception:
        pass

    # ── Build app ──────────────────────────────────────────────────────────
    from textual.app import App, ComposeResult
    from textual.widgets import Button, Checkbox, Footer, Header, Label, Static

    class ExperimentSelector(App):
        CSS = """
        Screen { layout: vertical; }
        #vram-bar { background: $panel; padding: 0 2; height: 1; }
        #experiments { height: 1fr; overflow-y: auto; padding: 1 2; }
        .exp-row { height: 3; }
        #buttons { height: 5; layout: horizontal; padding: 1 2; }
        Button { margin: 0 1; }
        """

        BINDINGS = [("q", "quit", "Quit")]

        def __init__(self, configs: list) -> None:
            super().__init__()
            self._configs = configs
            self._selected_ids: set[int] = set()
            self._run_all = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            yield Static(vram_info or "  (VRAM info unavailable)", id="vram-bar")
            exp_list = Static(id="experiments")
            yield exp_list
            yield Static(id="buttons")
            yield Footer()

        def on_mount(self) -> None:
            container = self.query_one("#experiments", Static)
            container.update("")
            # replace the Static with actual Checkboxes via mount
            from textual.widgets import Checkbox

            for cfg in self._configs:
                arch = getattr(cfg, "arch_type", "donut")
                label = f"[{cfg.id:2d}]  {cfg.name[:42]:<42s}  ({arch})"
                cb = Checkbox(label, id=f"exp-{cfg.id}", value=True)
                self.query_one("#experiments").mount(cb)

            btn_container = self.query_one("#buttons", Static)
            btn_container.mount(Button("Run Selected", id="btn-selected", variant="primary"))
            btn_container.mount(Button("Run All", id="btn-all", variant="success"))
            btn_container.mount(Button("Cancel", id="btn-cancel", variant="error"))

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-all":
                self._run_all = True
                self.exit()
            elif event.button.id == "btn-selected":
                self._run_all = False
                for cfg in self._configs:
                    cb = self.query_one(f"#exp-{cfg.id}", Checkbox)
                    if cb.value:
                        self._selected_ids.add(cfg.id)
                self.exit()
            elif event.button.id == "btn-cancel":
                self._selected_ids = set()
                self.exit()

    app = ExperimentSelector(all_configs)
    app.run()

    if app._run_all:
        return list(all_configs)
    if not app._selected_ids:
        return []
    selected = [c for c in all_configs if c.id in app._selected_ids]
    selected.sort(key=lambda c: c.id)
    return selected
