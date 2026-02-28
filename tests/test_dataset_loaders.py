"""Tests for dataset_loaders.py — schema helpers, DatasetLoadError, path helpers."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_loaders import (
    DatasetLoadError,
    _ensure_dir,
    _validate_sample_schema,
    _validate_samples_nonempty,
)

# ---------------------------------------------------------------------------
# DatasetLoadError
# ---------------------------------------------------------------------------

class TestDatasetLoadError:
    def test_attributes(self):
        exc = DatasetLoadError("test_ds", "file not found")
        assert exc.dataset_name == "test_ds"
        assert exc.reason == "file not found"

    def test_message_prefix(self):
        exc = DatasetLoadError("test_ds", "file not found")
        assert "FATAL:" in str(exc)
        assert "test_ds" in str(exc)

    def test_inherits_exception(self):
        exc = DatasetLoadError("ds", "reason")
        assert isinstance(exc, Exception)


# ---------------------------------------------------------------------------
# _validate_sample_schema
# ---------------------------------------------------------------------------

class TestValidateSampleSchema:
    def test_valid_sample(self):
        sample = (Path("/img.jpg"), {"company": "A", "date": "B", "address": "C", "total": "D"})
        assert _validate_sample_schema(sample, "test") is True

    def test_missing_field(self):
        sample = (Path("/img.jpg"), {"company": "A", "date": "B"})
        assert _validate_sample_schema(sample, "test") is False

    def test_non_path(self):
        sample = ("/img.jpg", {"company": "A", "date": "B", "address": "C", "total": "D"})
        assert _validate_sample_schema(sample, "test") is False

    def test_non_dict_gt(self):
        sample = (Path("/img.jpg"), "not a dict")
        assert _validate_sample_schema(sample, "test") is False

    def test_wrong_length(self):
        sample = (Path("/img.jpg"),)
        assert _validate_sample_schema(sample, "test") is False

    def test_not_tuple(self):
        assert _validate_sample_schema("string", "test") is False

    def test_extra_fields_ok(self):
        """Extra fields beyond SROIE schema should still validate."""
        sample = (Path("/img.jpg"), {"company": "A", "date": "B", "address": "C", "total": "D", "extra": "E"})
        assert _validate_sample_schema(sample, "test") is True


# ---------------------------------------------------------------------------
# _validate_samples_nonempty
# ---------------------------------------------------------------------------

class TestValidateSamplesNonempty:
    def test_nonempty_passes(self):
        samples = [(Path("/img.jpg"), {"company": "", "date": "", "address": "", "total": ""})]
        result = _validate_samples_nonempty(samples, "test")
        assert result == samples

    def test_empty_raises(self):
        try:
            _validate_samples_nonempty([], "test_ds")
            assert False, "Should have raised DatasetLoadError"
        except DatasetLoadError as e:
            assert e.dataset_name == "test_ds"
            assert "0 samples" in e.reason


# ---------------------------------------------------------------------------
# _ensure_dir
# ---------------------------------------------------------------------------

class TestEnsureDir:
    def test_creates_dir(self, tmp_path):
        new_dir = tmp_path / "a" / "b" / "c"
        result = _ensure_dir(new_dir)
        assert result == new_dir
        assert new_dir.is_dir()

    def test_existing_dir(self, tmp_path):
        result = _ensure_dir(tmp_path)
        assert result == tmp_path
        assert tmp_path.is_dir()


# ---------------------------------------------------------------------------
# Path helpers (env var behavior)
# ---------------------------------------------------------------------------

class TestPathHelpers:
    def test_get_datasets_dir_default(self):
        from dataset_loaders import _get_datasets_dir
        old = os.environ.pop("DONUT_WORKSPACE", None)
        try:
            d = _get_datasets_dir()
            assert d == Path("/workspace/datasets")
        finally:
            if old is not None:
                os.environ["DONUT_WORKSPACE"] = old

    def test_get_datasets_dir_custom(self):
        from dataset_loaders import _get_datasets_dir
        old = os.environ.get("DONUT_WORKSPACE")
        os.environ["DONUT_WORKSPACE"] = "/tmp/custom"
        try:
            d = _get_datasets_dir()
            assert d == Path("/tmp/custom/datasets")
        finally:
            if old is not None:
                os.environ["DONUT_WORKSPACE"] = old
            else:
                del os.environ["DONUT_WORKSPACE"]

    def test_get_sroie_dir_default(self):
        from dataset_loaders import _get_sroie_dir
        old = os.environ.pop("SROIE_DATA_DIR", None)
        try:
            d = _get_sroie_dir()
            assert d == Path("/workspace/ICDAR-2019-SROIE/data")
        finally:
            if old is not None:
                os.environ["SROIE_DATA_DIR"] = old

    def test_get_sroie_dir_custom(self):
        from dataset_loaders import _get_sroie_dir
        old = os.environ.get("SROIE_DATA_DIR")
        os.environ["SROIE_DATA_DIR"] = "/data/sroie"
        try:
            d = _get_sroie_dir()
            assert d == Path("/data/sroie")
        finally:
            if old is not None:
                os.environ["SROIE_DATA_DIR"] = old
            else:
                del os.environ["SROIE_DATA_DIR"]


# ---------------------------------------------------------------------------
# Dataset split separation (no data access required)
# ---------------------------------------------------------------------------

class TestSROIESplitDirectories:
    """Verify val and test splits use distinct directories (no data leakage)."""

    def test_val_and_test_use_different_dirs(self):
        """val_img/ != test_img/ — ensures early stopping does not use test data."""
        from dataset_loaders import SROIELoader
        loader = SROIELoader()
        val_dirs = loader._SPLIT_DIRS["val"]
        test_dirs = loader._SPLIT_DIRS["test"]
        assert val_dirs != test_dirs, (
            "val and test splits share the same directories — "
            "early stopping would optimize for test performance (data leakage)"
        )

    def test_train_val_test_all_different(self):
        """All three splits use different source directories."""
        from dataset_loaders import SROIELoader
        loader = SROIELoader()
        dirs = [loader._SPLIT_DIRS[s] for s in ("train", "val", "test")]
        assert len(set(dirs)) == 3, (
            f"Expected 3 unique split directory pairs, got {len(set(dirs))}: {dirs}"
        )

    def test_val_uses_val_img(self):
        from dataset_loaders import SROIELoader
        loader = SROIELoader()
        img_subdir, key_subdir = loader._SPLIT_DIRS["val"]
        assert img_subdir == "val_img"
        assert key_subdir == "val_key"

    def test_test_uses_test_img(self):
        from dataset_loaders import SROIELoader
        loader = SROIELoader()
        img_subdir, key_subdir = loader._SPLIT_DIRS["test"]
        assert img_subdir == "test_img"
        assert key_subdir == "test_key"

    def test_train_uses_img(self):
        from dataset_loaders import SROIELoader
        loader = SROIELoader()
        img_subdir, key_subdir = loader._SPLIT_DIRS["train"]
        assert img_subdir == "img"
        assert key_subdir == "key"
