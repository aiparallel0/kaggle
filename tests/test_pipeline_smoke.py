"""Pipeline integration smoke tests.

These tests validate the integration *contract* between pipeline components —
specifically the field names and call signatures that span multiple files.

They catch the class of bug that recurs every ~2 PRs: a new feature or
refactor references a non-existent attribute on a dataclass (e.g. config.id
instead of config.experiment_id), or passes a non-existent keyword argument
to a function. Python does not catch these at import time — they only surface
at runtime.

All tests here run without GPU/disk access. They only import dataclasses,
inspect signatures, and check dict keys. Torch is required only because
run_experiments.py imports it at the top level.
"""

import inspect
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

# run_experiments.py imports torch at top level — guard the whole module.
torch = pytest.importorskip("torch", reason="torch required by run_experiments.py")
pytest.importorskip("transformers", reason="transformers required by run_experiments.py")
from run_experiments import (  # noqa: E402, I001
    EXPERIMENTS,
    ExperimentConfig,
    run_custom_experiment,
    train_experiment,
)
from dataset_loaders import _LOADERS  # noqa: E402, I001


class TestExperimentConfigFields:
    """ExperimentConfig must use experiment_id, not id."""

    def test_has_experiment_id_field(self):
        """experiment_id is the correct field name."""
        c = ExperimentConfig(experiment_id=99, name="test", datasets=["sroie"])
        assert c.experiment_id == 99

    def test_does_not_have_id_field(self):
        """id is NOT a field — prevents regression of Bug A/B."""
        c = ExperimentConfig(experiment_id=1, name="test", datasets=["sroie"])
        assert not hasattr(c, "id"), (
            "ExperimentConfig must not have an 'id' attribute — use experiment_id. "
            "If 'id' was added, update run_all.py and run_experiments.py to match."
        )

    def test_does_not_accept_lr_scheduler_type(self):
        """lr_scheduler_type is NOT a valid field — prevents regression of Bug A."""
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(ExperimentConfig)}
        assert "lr_scheduler_type" not in field_names, (
            "lr_scheduler_type is not a valid ExperimentConfig field. "
            "Remove it from the dataclass or fix any callers that pass it."
        )

    def test_instantiation_with_defaults(self):
        """ExperimentConfig can be instantiated with only required fields."""
        c = ExperimentConfig(experiment_id=1, name="smoke", datasets=["sroie"])
        assert c.epochs > 0
        assert c.lr > 0
        assert c.batch_size > 0


class TestTrainExperimentSignature:
    """train_experiment must accept an optional config= parameter."""

    def test_accepts_config_keyword(self):
        """train_experiment signature must have config= parameter (Bug C fix)."""
        sig = inspect.signature(train_experiment)
        assert "config" in sig.parameters, (
            "train_experiment() is missing the 'config' parameter. "
            "Without it, run_custom_experiment() silently uses the wrong config "
            "from EXPERIMENTS[exp_id] instead of the custom one."
        )

    def test_config_defaults_to_none(self):
        """config parameter must default to None so existing callers are unaffected."""
        sig = inspect.signature(train_experiment)
        param = sig.parameters["config"]
        assert param.default is None, (
            "train_experiment(config=) must default to None to remain backwards-compatible "
            "with callers that do not pass a custom config."
        )


class TestExperimentsRegistry:
    """All experiment dataset names must exist in the loader registry."""

    def test_all_dataset_names_are_valid(self):
        """Every dataset name in EXPERIMENTS must have a loader in _LOADERS."""
        invalid = {}
        for exp_id, config in EXPERIMENTS.items():
            bad = [d for d in config.datasets if d not in _LOADERS]
            if bad:
                invalid[exp_id] = bad

        assert not invalid, (
            f"Experiments reference dataset names with no loader: {invalid}. "
            f"Valid loader names: {sorted(_LOADERS.keys())}"
        )

    def test_experiments_covers_expected_ids(self):
        """EXPERIMENTS must define exactly 8 experiments (IDs 1–8)."""
        assert set(EXPERIMENTS.keys()) == set(range(1, 9)), (
            f"Expected EXPERIMENTS to have keys 1–8, got: {sorted(EXPERIMENTS.keys())}"
        )

    def test_all_configs_have_experiment_id(self):
        """Every config in EXPERIMENTS must have experiment_id set (not 0 default)."""
        bad = [exp_id for exp_id, config in EXPERIMENTS.items() if config.experiment_id == 0]
        assert not bad, (
            f"Experiments {bad} have experiment_id=0 (default). "
            "Set experiment_id=N explicitly in each ExperimentConfig."
        )


class TestRunCustomExperimentCallable:
    """run_custom_experiment must exist and have the right signature."""

    def test_is_callable(self):
        assert callable(run_custom_experiment)

    def test_accepts_config_and_result_file(self):
        sig = inspect.signature(run_custom_experiment)
        params = list(sig.parameters.keys())
        assert "config" in params, "run_custom_experiment must accept 'config'"
        assert "result_file" in params, "run_custom_experiment must accept 'result_file'"
