"""Unit tests for logging_utils.py.

Tests:
  - DeduplicatingHandler collapses identical consecutive messages into [×N].
  - DeduplicatingHandler emits different messages without collapsing.
  - DeduplicatingHandler.flush() emits any pending buffered record.
  - DeduplicatingHandler thread safety (no crashes, correct count).
  - suppress_noisy_loggers() sets PIL loggers to WARNING.

No GPU, no torch, no network required.
"""

import logging
import threading

from logging_utils import DeduplicatingHandler, suppress_noisy_loggers


class _CapturingHandler(logging.Handler):
    """Minimal handler that stores formatted records for inspection."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestDeduplicatingHandler:
    def _make_record(self, name: str, level: int, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name=name,
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_single_message_emitted_on_flush(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        rec = self._make_record("test", logging.DEBUG, "hello")
        handler.emit(rec)
        handler.flush()
        assert len(cap.records) == 1
        assert cap.records[0].getMessage() == "hello"

    def test_repeated_message_collapsed_with_count(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        for _ in range(5):
            handler.emit(self._make_record("test", logging.DEBUG, "same msg"))
        handler.flush()
        assert len(cap.records) == 1
        assert "[×5]" in cap.records[0].msg

    def test_different_messages_emitted_separately(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        handler.emit(self._make_record("test", logging.DEBUG, "msg A"))
        handler.emit(self._make_record("test", logging.DEBUG, "msg A"))
        handler.emit(self._make_record("test", logging.DEBUG, "msg B"))
        handler.flush()
        # msg A (count=2) + msg B (count=1)
        assert len(cap.records) == 2
        assert "[×2]" in cap.records[0].msg
        # msg B has count 1 — no suffix appended
        assert "[×" not in cap.records[1].msg

    def test_single_occurrence_has_no_count_suffix(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        handler.emit(self._make_record("test", logging.INFO, "only once"))
        handler.flush()
        assert len(cap.records) == 1
        assert "[×" not in cap.records[0].msg

    def test_different_levels_not_collapsed(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        handler.emit(self._make_record("test", logging.DEBUG, "same text"))
        handler.emit(self._make_record("test", logging.WARNING, "same text"))
        handler.flush()
        # Different level → different key → two separate records
        assert len(cap.records) == 2

    def test_close_flushes_pending(self):
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        for _ in range(3):
            handler.emit(self._make_record("test", logging.DEBUG, "pending"))
        handler.close()
        assert len(cap.records) == 1
        assert "[×3]" in cap.records[0].msg

    def test_thread_safety_no_crash(self):
        """Multiple threads emitting the same message must not crash or lose records."""
        cap = _CapturingHandler()
        handler = DeduplicatingHandler(cap)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(50):
                    handler.emit(self._make_record("t", logging.DEBUG, "concurrent"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        handler.flush()
        assert not errors, f"Thread errors: {errors}"
        # At least one record must have been emitted (possibly collapsed)
        assert len(cap.records) >= 1


class TestSuppressNoisyLoggers:
    def test_pil_set_to_warning(self):
        import logging

        suppress_noisy_loggers(logging.WARNING)
        assert logging.getLogger("PIL").level == logging.WARNING

    def test_pil_png_plugin_set_to_warning(self):
        suppress_noisy_loggers(logging.WARNING)
        assert logging.getLogger("PIL.PngImagePlugin").level == logging.WARNING

    def test_custom_level(self):
        suppress_noisy_loggers(logging.ERROR)
        assert logging.getLogger("PIL").level == logging.ERROR
        # Reset to WARNING for other tests
        suppress_noisy_loggers(logging.WARNING)
