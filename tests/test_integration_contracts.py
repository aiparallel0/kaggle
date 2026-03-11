"""tests/test_integration_contracts.py — Integration contract tests.

Verifies cross-module API contracts using AST inspection (no GPU/torch needed)
and lightweight runtime checks (no model weights or disk access required).

Contracts verified:
  1. DonutEvaluator uses ``model_path`` kwarg, NOT ``model_dir``.
  2. run_experiments exports ``run_experiment_from_config`` in ``__all__``.
  3. _run_yaml_donut_experiment raises RuntimeError instead of returning 0.0 F1.
  4. _run_trocr_yolo_experiment calls stage_trocr_data_prep before training.
  5. logging_utils is importable without heavy deps.
  6. suppress_noisy_loggers is callable and accepts an int level.

All AST-based tests work without torch / transformers installed.
"""

import ast
import importlib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse(filename: str) -> ast.Module:
    return ast.parse((_REPO / filename).read_text())


def _source(filename: str) -> str:
    return (_REPO / filename).read_text()


# ---------------------------------------------------------------------------
# Contract 1: DonutEvaluator uses model_path, not model_dir
# ---------------------------------------------------------------------------


class TestDonutEvaluatorModelPath:
    def test_model_dir_not_used_in_run_all_zero_shot(self):
        """run_all._run_zero_shot_experiment must NOT pass model_dir= to DonutEvaluator."""
        src = _source("run_all.py")
        # Find the _run_zero_shot_experiment function source
        tree = _parse("run_all.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_run_zero_shot_experiment":
                func_src = ast.get_source_segment(src, node) or ""
                assert "model_dir=" not in func_src, (
                    "_run_zero_shot_experiment still passes model_dir= to DonutEvaluator; "
                    "the correct kwarg is model_path="
                )
                assert "model_path=" in func_src, (
                    "_run_zero_shot_experiment must pass model_path= to DonutEvaluator"
                )
                return
        pytest.fail("_run_zero_shot_experiment function not found in run_all.py")

    def test_donut_evaluator_constructor_has_model_path_param(self):
        """DonutEvaluator.__init__ must accept model_path (not model_dir)."""
        tree = _parse("donut_evaluator.py")
        src = _source("donut_evaluator.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "DonutEvaluator":
                for item in ast.walk(node):
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        init_src = ast.get_source_segment(src, item) or ""
                        assert "model_path" in init_src, (
                            "DonutEvaluator.__init__ must have a model_path parameter"
                        )
                        assert "model_dir" not in init_src, (
                            "DonutEvaluator.__init__ must NOT use model_dir (use model_path)"
                        )
                        return
        pytest.fail("DonutEvaluator.__init__ not found in donut_evaluator.py")


# ---------------------------------------------------------------------------
# Contract 2: run_experiments exports run_experiment_from_config
# ---------------------------------------------------------------------------


class TestRunExperimentFromConfigExported:
    def test_in_all_list(self):
        """run_experiments.__all__ must include run_experiment_from_config."""
        tree = _parse("run_experiments.py")
        src = _source("run_experiments.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        all_src = ast.get_source_segment(src, node) or ""
                        assert "run_experiment_from_config" in all_src, (
                            "run_experiments.__all__ must include 'run_experiment_from_config'"
                        )
                        return
        pytest.fail("__all__ assignment not found in run_experiments.py")

    def test_run_experiment_from_config_in_all_ast(self):
        """AST check: run_experiment_from_config is defined as a function in run_experiments.py."""
        tree = _parse("run_experiments.py")
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert "run_experiment_from_config" in func_names, (
            "run_experiment_from_config function definition not found in run_experiments.py"
        )


# ---------------------------------------------------------------------------
# Contract 3: yaml fallback raises RuntimeError, not silent 0.0 F1
# ---------------------------------------------------------------------------


class TestYamlFallbackRaisesError:
    def test_no_return_zero_f1_dict(self):
        """_run_yaml_donut_experiment must NOT return a dict with global_f1: 0.0."""
        src = _source("run_all.py")
        tree = _parse("run_all.py")
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_run_yaml_donut_experiment"
            ):
                func_src = ast.get_source_segment(src, node) or ""
                # The old fallback returned {"metrics": {"global_f1": 0.0}}
                assert '"global_f1": 0.0' not in func_src and "'global_f1': 0.0" not in func_src, (
                    "_run_yaml_donut_experiment must not return a silent 0.0 F1 placeholder; "
                    "it must raise RuntimeError so the failure is loud."
                )
                return
        pytest.fail("_run_yaml_donut_experiment not found in run_all.py")

    def test_raises_runtime_error(self):
        """_run_yaml_donut_experiment fallback path must raise RuntimeError."""
        src = _source("run_all.py")
        tree = _parse("run_all.py")
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_run_yaml_donut_experiment"
            ):
                func_src = ast.get_source_segment(src, node) or ""
                assert "raise RuntimeError" in func_src, (
                    "_run_yaml_donut_experiment fallback must raise RuntimeError"
                )
                return
        pytest.fail("_run_yaml_donut_experiment not found in run_all.py")


# ---------------------------------------------------------------------------
# Contract 4: trocr calls data prep before training
# ---------------------------------------------------------------------------


class TestTrocrCallsDataPrep:
    def test_stage_trocr_data_prep_called_in_trocr_experiment(self):
        """_run_trocr_yolo_experiment must reference stage_trocr_data_prep."""
        src = _source("run_all.py")
        tree = _parse("run_all.py")
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "_run_trocr_yolo_experiment"
            ):
                func_src = ast.get_source_segment(src, node) or ""
                assert "stage_trocr_data_prep" in func_src, (
                    "_run_trocr_yolo_experiment must call stage_trocr_data_prep() "
                    "to ensure YOLO/TrOCR data is ready before training"
                )
                return
        pytest.fail("_run_trocr_yolo_experiment not found in run_all.py")


# ---------------------------------------------------------------------------
# Contract 5: logging_utils importable and functional
# ---------------------------------------------------------------------------


class TestLoggingUtilsImportable:
    def test_import(self):
        """logging_utils must be importable without heavy dependencies."""
        mod = importlib.import_module("logging_utils")
        assert hasattr(mod, "suppress_noisy_loggers")
        assert hasattr(mod, "DeduplicatingHandler")
        assert hasattr(mod, "_NOISY_THIRD_PARTY_LOGGERS")

    def test_suppress_noisy_loggers_callable(self):
        import logging

        from logging_utils import suppress_noisy_loggers

        # Should not raise
        suppress_noisy_loggers(logging.WARNING)

    def test_noisy_logger_list_contains_pil(self):
        from logging_utils import _NOISY_THIRD_PARTY_LOGGERS

        assert "PIL" in _NOISY_THIRD_PARTY_LOGGERS
        assert "PIL.PngImagePlugin" in _NOISY_THIRD_PARTY_LOGGERS
