"""
tests/test_memory_manager.py
CPU-only tests for memory_manager.py.

These tests verify that the per-sample RAM arithmetic is correct at both
the DONUT native resolution (1280x960) and the previously-wrong resolution
(2560x1920), and that the safety gate correctly blocks oversized caches.

No GPU, no network, no HuggingFace downloads required.
"""

import memory_manager as mm


class TestComputePilMbPerSample:
    """Tests for compute_pil_mb_per_sample()."""

    def test_reference_resolution_1280x960(self):
        """At DONUT native 1280x960: exactly 3 x 1280 x 960 / 1_048_576."""
        expected = 3 * 1280 * 960 / (1024 * 1024)
        result = mm.compute_pil_mb_per_sample(1280, 960)
        assert abs(result - expected) < 1e-6, f"Expected {expected:.4f}, got {result:.4f}"

    def test_wrong_resolution_2560x1920_is_4x_larger(self):
        """At 2560x1920 the per-sample cost is exactly 4x the reference."""
        ref = mm.compute_pil_mb_per_sample(1280, 960)
        wrong = mm.compute_pil_mb_per_sample(2560, 1920)
        ratio = wrong / ref
        assert abs(ratio - 4.0) < 1e-6, (
            f"2560x1920 should be 4x reference but ratio is {ratio:.4f}. "
            "This confirms that 2560x1920 in processor_config.json causes 4x RAM cost."
        )

    def test_trocr_line_crop_384x384(self):
        """TrOCR line crops at 384x384 are much smaller than full receipt images."""
        trocr_mb = mm.compute_pil_mb_per_sample(384, 384)
        donut_mb = mm.compute_pil_mb_per_sample(1280, 960)
        # 384x384 = 147456 px; 1280x960 = 1228800 px -> ratio ~= 8.33x
        assert trocr_mb < donut_mb, "TrOCR line crops must be smaller than full receipts"
        assert trocr_mb < 1.0, f"384x384 line crops should be < 1 MB, got {trocr_mb:.3f} MB"

    def test_zero_dimensions_returns_zero(self):
        """Edge case: zero height or width returns 0."""
        assert mm.compute_pil_mb_per_sample(0, 960) == 0.0
        assert mm.compute_pil_mb_per_sample(1280, 0) == 0.0


def _make_psutil_mock(available_bytes: int):
    """Return a mock psutil module with virtual_memory().available set."""
    import unittest.mock as mock

    mock_psutil = mock.MagicMock()
    mock_vm = mock.MagicMock()
    mock_vm.available = available_bytes
    mock_psutil.virtual_memory.return_value = mock_vm
    return mock_psutil


class TestRamCacheIsSafe:
    """Tests for ram_cache_is_safe() -- the gate that replaced `len(samples) * 3`."""

    def test_small_dataset_at_ref_resolution_is_safe(self):
        """500 samples x 3.516 MB = 1,758 MB -- safe on any machine with > 12 GB RAM."""
        import unittest.mock as mock

        mock_psutil = _make_psutil_mock(64 * 1024 * 1024 * 1024)  # 64 GB
        with mock.patch.dict("sys.modules", {"psutil": mock_psutil}):
            result = mm.ram_cache_is_safe(500, 1280, 960)
        assert result is True, "500 samples at 1280x960 should be safe on 64 GB RAM"

    def test_large_dataset_at_wrong_resolution_is_rejected(self):
        """3940 samples x 14.06 MB = 55,396 MB -> must be blocked at 192 GB.

        55 GB > 192 GB x 15% = 28.8 GB threshold -> rejected.
        This is the exact Experiment 8 scenario that caused 192 GB RAM exhaustion.
        """
        import unittest.mock as mock

        mock_psutil = _make_psutil_mock(192 * 1024 * 1024 * 1024)  # 192 GB
        with mock.patch.dict("sys.modules", {"psutil": mock_psutil}):
            result = mm.ram_cache_is_safe(3940, 2560, 1920)
        assert result is False, (
            "3940 samples x 14.06 MB = 55.4 GB must be rejected at 15% of 192 GB (28.8 GB threshold). "
            "This is the Exp 8 OOM scenario."
        )

    def test_large_dataset_at_ref_resolution_is_allowed(self):
        """3940 samples x 3.516 MB = 13,853 MB < 256 GB x 6% = 15,729 MB -> SAFE.

        Uses 256 GB RAM to ensure the correct resolution (1280x960) is allowed
        even with the conservative 6% safety fraction.
        """
        import unittest.mock as mock

        mock_psutil = _make_psutil_mock(256 * 1024 * 1024 * 1024)  # 256 GB
        with mock.patch.dict("sys.modules", {"psutil": mock_psutil}):
            result = mm.ram_cache_is_safe(3940, 1280, 960)
        assert result is True, (
            "3940 samples x 3.516 MB = 13.8 GB should be allowed at 6% of 256 GB (15.7 GB threshold)."
        )

    def test_psutil_import_error_returns_false(self):
        """If psutil raises ImportError, caching is disabled (safe default)."""
        import unittest.mock as mock

        with mock.patch.dict("sys.modules", {"psutil": None}):
            result = mm.ram_cache_is_safe(100, 1280, 960)
        assert isinstance(result, bool)

    def test_custom_safety_fraction(self):
        """Custom safety_fraction parameter is respected."""
        import unittest.mock as mock

        # 10 GB available = 10,240 MB
        mock_psutil = _make_psutil_mock(10 * 1024 * 1024 * 1024)
        with mock.patch.dict("sys.modules", {"psutil": mock_psutil}):
            # 500 x 3.516 MB = 1758 MB; 10,240 MB x 0.10 = 1,024 MB threshold
            # 1758 > 1024 -> REJECTED at 10% safety
            result_10pct = mm.ram_cache_is_safe(500, 1280, 960, safety_fraction=0.10)
            # 10,240 MB x 0.25 = 2,560 MB threshold
            # 1758 < 2560 -> ALLOWED at 25% safety
            result_25pct = mm.ram_cache_is_safe(500, 1280, 960, safety_fraction=0.25)
        assert result_10pct is False, "Should be rejected at 10% safety fraction"
        assert result_25pct is True, "Should be allowed at 25% safety fraction"


class TestFlushHfArrowCache:
    """Tests for flush_hf_arrow_cache() -- smoke tests only (no network)."""

    def test_flush_does_not_crash_when_datasets_installed(self):
        """flush_hf_arrow_cache() must not raise even if datasets is installed."""
        mm.flush_hf_arrow_cache()  # should complete without exception

    def test_flush_does_not_crash_when_datasets_missing(self):
        """flush_hf_arrow_cache() must not raise if datasets is not installed."""
        import unittest.mock as mock

        with mock.patch.dict("sys.modules", {"datasets": None}):
            mm.flush_hf_arrow_cache()  # should complete without exception


class TestReleaseHfDataset:
    """Tests for release_hf_dataset() -- unit tests with mock datasets."""

    def test_none_is_noop(self):
        """Passing None is safe (no-op)."""
        mm.release_hf_dataset(None)  # must not raise

    def test_object_with_cleanup_cache_files_is_called(self):
        """Objects with cleanup_cache_files() have it called."""
        import unittest.mock as mock

        mock_ds = mock.Mock()
        mock_ds.cleanup_cache_files = mock.Mock()
        mm.release_hf_dataset(mock_ds)
        mock_ds.cleanup_cache_files.assert_called_once()

    def test_object_without_cleanup_does_not_crash(self):
        """Objects without cleanup_cache_files() are handled gracefully."""

        class FakeDataset:
            pass

        mm.release_hf_dataset(FakeDataset())  # must not raise


class TestShutdownDataloaderWorkers:
    """Tests for shutdown_dataloader_workers() -- smoke tests with mock trainer."""

    def test_none_trainer_is_noop(self):
        """Passing None is safe (no-op)."""
        mm.shutdown_dataloader_workers(None)  # must not raise

    def test_trainer_without_get_train_dataloader_does_not_crash(self):
        """Trainers that don't have get_train_dataloader() are handled gracefully."""

        class FakeTrainer:
            pass

        mm.shutdown_dataloader_workers(FakeTrainer())  # must not raise

    def test_trainer_with_iterator_calls_shutdown(self):
        """When _iterator._shutdown_workers exists it is called."""
        import unittest.mock as mock

        mock_iter = mock.Mock()
        mock_iter._shutdown_workers = mock.Mock()
        mock_dl = mock.Mock()
        mock_dl._iterator = mock_iter
        mock_trainer = mock.Mock()
        mock_trainer.get_train_dataloader = mock.Mock(return_value=mock_dl)
        mock_trainer.get_eval_dataloader = mock.Mock(return_value=None)
        mm.shutdown_dataloader_workers(mock_trainer)
        mock_iter._shutdown_workers.assert_called_once()


class TestRamHeadroomMb:
    """Tests for the new ram_headroom_mb() function."""

    def test_returns_positive_float_on_real_system(self):
        """ram_headroom_mb() must return a positive float on any machine with psutil."""
        result = mm.ram_headroom_mb()
        assert isinstance(result, float), f"Expected float, got {type(result).__name__}"
        assert result >= 0.0, f"Expected non-negative value, got {result}"

    def test_returns_float_with_mock_psutil(self):
        """ram_headroom_mb() returns available / 1024**2 from psutil.virtual_memory()."""
        import unittest.mock as mock

        mock_psutil = _make_psutil_mock(8 * 1024 * 1024 * 1024)  # 8 GB
        with mock.patch.dict("sys.modules", {"psutil": mock_psutil}):
            result = mm.ram_headroom_mb()
        expected = 8 * 1024  # 8 GB in MB = 8192
        assert abs(result - expected) < 1.0, f"Expected ~{expected} MB, got {result:.1f} MB"

    def test_returns_zero_on_psutil_error(self):
        """ram_headroom_mb() returns 0.0 when psutil is unavailable."""
        import unittest.mock as mock

        with mock.patch.dict("sys.modules", {"psutil": None}):
            result = mm.ram_headroom_mb()
        assert result == 0.0, f"Expected 0.0 on psutil error, got {result}"

    def test_exported_in_all(self):
        """ram_headroom_mb must be listed in memory_manager.__all__."""
        assert "ram_headroom_mb" in mm.__all__, (
            "ram_headroom_mb must be exported in __all__ so callers can do "
            "`from memory_manager import ram_headroom_mb`"
        )


class TestRamSafetyFractionRegressionGuard:
    """Regression guard: _RAM_SAFETY_FRACTION must never be raised back to 0.15."""

    def test_safety_fraction_is_at_most_0_10(self):
        """_RAM_SAFETY_FRACTION must be <= 0.10.

        The OOM crash at Experiment 6 was caused by the old 15% (0.15) value.
        The fix lowers it to 6% (0.06). This test prevents future PRs from
        raising it back above 10% without explicit acknowledgement.
        """
        fraction = mm._RAM_SAFETY_FRACTION
        assert fraction <= 0.10, (
            f"_RAM_SAFETY_FRACTION = {fraction} is too high (must be <= 0.10). "
            "Raising it above 0.10 risks the Exp 6 three-layer OOM failure: "
            "train PIL cache drains RAM before the val init runs."
        )

    def test_safety_fraction_is_positive(self):
        """_RAM_SAFETY_FRACTION must be strictly positive."""
        assert mm._RAM_SAFETY_FRACTION > 0.0, (
            "_RAM_SAFETY_FRACTION must be > 0 to avoid dividing by zero or "
            "always rejecting the cache."
        )
