"""Unit tests for MultiDataset cache lifecycle and run_experiments cleanup.

These tests verify:
  - MultiDataset.clear_caches() empties all three cache dicts.
  - The train_experiment() cleanup block calls clear_caches() before
    deleting the dataset references (AST-based contract test).
  - The OOM-retry cleanup block in train_experiment() also calls clear_caches().

All tests run without GPU, model weights, or torch.cuda — CPU-only.
"""

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch", reason="torch required by train.py")
pytest.importorskip("transformers", reason="transformers required by train.py")

from train import MultiDataset  # noqa: E402, I001


# ---------------------------------------------------------------------------
# MultiDataset.clear_caches()
# ---------------------------------------------------------------------------


class TestDatasetCacheClearing:
    """Verify that clear_caches() empties all three internal cache dicts."""

    def _make_empty_dataset(self):
        """Build a zero-sample MultiDataset (no GPU / disk access needed)."""
        try:
            from transformers import DonutProcessor

            proc = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        except (OSError, ImportError, ValueError):
            pytest.skip("Cannot load DonutProcessor in this environment")

        return MultiDataset(samples=[], processor=proc, cache_in_ram=False)

    def test_clear_caches_method_exists(self):
        """MultiDataset must expose a clear_caches() method."""
        assert hasattr(MultiDataset, "clear_caches"), (
            "MultiDataset.clear_caches() is required to explicitly free "
            "_pixel_cache, _image_cache, and _label_cache before GC."
        )
        assert callable(MultiDataset.clear_caches)

    def test_clear_caches_empties_pixel_cache(self):
        """clear_caches() must empty _pixel_cache."""
        ds = self._make_empty_dataset()
        # Manually populate the cache to simulate a trained dataset
        ds._pixel_cache[0] = torch.zeros(3, 4, 4)
        ds._pixel_cache[1] = torch.zeros(3, 4, 4)
        assert len(ds._pixel_cache) == 2

        ds.clear_caches()
        assert len(ds._pixel_cache) == 0, (
            "_pixel_cache must be empty after clear_caches() — "
            "it holds up to 7.1 GB of float32 tensors."
        )

    def test_clear_caches_empties_image_cache(self):
        """clear_caches() must empty _image_cache."""
        ds = self._make_empty_dataset()
        from PIL import Image

        ds._image_cache[0] = Image.new("RGB", (4, 4))
        assert len(ds._image_cache) == 1

        ds.clear_caches()
        assert len(ds._image_cache) == 0

    def test_clear_caches_empties_label_cache(self):
        """clear_caches() must empty _label_cache."""
        ds = self._make_empty_dataset()
        ds._label_cache[0] = torch.zeros(768, dtype=torch.long)
        assert len(ds._label_cache) == 1

        ds.clear_caches()
        assert len(ds._label_cache) == 0

    def test_clear_caches_all_three_at_once(self):
        """clear_caches() must clear all three caches in a single call."""
        from PIL import Image

        ds = self._make_empty_dataset()
        ds._pixel_cache[0] = torch.zeros(3, 4, 4)
        ds._image_cache[0] = Image.new("RGB", (4, 4))
        ds._label_cache[0] = torch.zeros(768, dtype=torch.long)

        ds.clear_caches()
        assert len(ds._pixel_cache) == 0
        assert len(ds._image_cache) == 0
        assert len(ds._label_cache) == 0

    def test_clear_caches_idempotent(self):
        """Calling clear_caches() twice must not raise."""
        ds = self._make_empty_dataset()
        ds.clear_caches()
        ds.clear_caches()  # second call must be a no-op


# ---------------------------------------------------------------------------
# AST contract: train_experiment() cleanup calls clear_caches()
# ---------------------------------------------------------------------------


class TestTrainExperimentCleanup:
    """AST-based test that clear_caches() is called before del train_ds."""

    @pytest.fixture(scope="class")
    def run_experiments_source(self):
        src_path = Path(__file__).resolve().parent.parent / "run_experiments.py"
        return src_path.read_text()

    @pytest.fixture(scope="class")
    def run_experiments_ast(self, run_experiments_source):
        return ast.parse(run_experiments_source)

    def _find_train_experiment_func(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "train_experiment":
                return node
        return None

    def _count_clear_caches_calls(self, func_node) -> int:
        """Count actual ast.Call nodes where the called attribute is 'clear_caches'."""
        count = 0
        for node in ast.walk(func_node):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clear_caches"
            ):
                count += 1
        return count

    def test_clear_caches_called_on_train_ds_before_del(self, run_experiments_ast):
        """train_experiment() must call .clear_caches() at least once before del train_ds.

        This is the critical sequence that prevents _pixel_cache tensors
        (~7.1 GB) from staying pinned while the next experiment allocates
        its own cache.
        """
        func = self._find_train_experiment_func(run_experiments_ast)
        assert func is not None, "train_experiment() function not found in run_experiments.py"

        count = self._count_clear_caches_calls(func)
        assert count >= 1, (
            "train_experiment() must call .clear_caches() on train_ds before deleting it. "
            "Without this, _pixel_cache tensors (up to 7.1 GB) are not freed between "
            "the 8 sequential experiments."
        )

    def test_clear_caches_called_in_oom_retry(self, run_experiments_ast):
        """The OOM-retry cleanup block must also call clear_caches().

        When CUDA OOM fires mid-training, the same cleanup pattern must be
        applied to free pinned pixel tensors before the retry rebuild.
        We expect at least 2 call sites: normal cleanup + OOM retry.
        """
        func = self._find_train_experiment_func(run_experiments_ast)
        assert func is not None

        count = self._count_clear_caches_calls(func)
        assert count >= 2, (
            f"Expected .clear_caches() to be called at least 2 times in train_experiment() "
            f"(normal cleanup + OOM retry), but found {count} call site(s)."
        )
