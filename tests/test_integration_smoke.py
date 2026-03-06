"""
tests/test_integration_smoke.py — Smoke test: verify the full import chain works
without GPU/model weights.

These tests must pass in CI (no torch, no GPU).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestImportChain:
    """Verify every module can be imported without crashing."""

    def test_constants_importable(self):
        from constants import BASE_MODEL, FIELDS, SEED

        assert len(FIELDS) == 4
        assert BASE_MODEL == "naver-clova-ix/donut-base"
        assert SEED == 42

    def test_address_extractor_importable(self):
        # extract_address_from_seller moved to dataset_normalizer.py
        from dataset_normalizer import extract_address_from_seller

        assert callable(extract_address_from_seller)

    def test_inject_results_importable(self):
        from inject_results import LEADERBOARD

        assert len(LEADERBOARD) > 0

    def test_training_config_importable(self):
        # TrainingConfig moved to resource_optimizer.py
        from resource_optimizer import TrainingConfig

        cfg = TrainingConfig()
        cfg.validate()
        assert cfg.batch_size == 8

    def test_pipeline_config_importable(self):
        from pipeline_config import PipelineMode

        assert PipelineMode.AUTO.value == "auto"

    def test_retro_ui_importable(self):
        # RetroUIFormatter moved to cloud_pipeline.py
        from cloud_pipeline import RetroUIFormatter

        assert RetroUIFormatter.bold("x")

    def test_test_runner_importable(self):
        from test_runner import TestRunner

        assert hasattr(TestRunner, "run_all_checks")

    def test_results_aggregator_importable(self):
        # ResultsAggregator moved to inject_results.py
        from inject_results import ResultsAggregator

        agg = ResultsAggregator(results_dir=Path("/tmp"))
        assert agg is not None

    def test_storage_manager_importable(self):
        # StorageManager moved to cloud_utils.py
        from cloud_utils import StorageManager

        sm = StorageManager(results_dir=Path("/tmp"))
        assert sm is not None

    def test_git_controller_importable(self):
        # GitController moved to cloud_utils.py
        from cloud_utils import GitController

        assert hasattr(GitController, "get_current_branch")

    def test_validators_importable(self):
        from validators import ImportChainChecker

        success, errors = ImportChainChecker.check_all()
        # Import chain should pass (constants.py exists)
        assert isinstance(errors, list)
        assert success, f"Import chain check failed: {errors}"
