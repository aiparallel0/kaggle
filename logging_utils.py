"""logging_utils.py — Shared logging helpers for the DONUT KIE pipeline.

Provides:
  - DeduplicatingHandler: collapses consecutive identical log records into one
    with a [×N] suffix so that PIL chunk-level DEBUG floods don't fill logs.
  - suppress_noisy_loggers(): sets known verbose third-party loggers to WARNING.
  - _NOISY_THIRD_PARTY_LOGGERS: the canonical list of loggers to suppress.

Thread-safety: DeduplicatingHandler uses a threading.Lock around all mutable
state so it is safe to use from multiple threads.
"""

import logging
import threading

_NOISY_THIRD_PARTY_LOGGERS = [
    "PIL",
    "PIL.PngImagePlugin",
    "PIL.TiffImagePlugin",
    "PIL.Image",
    "PIL.JpegImagePlugin",
    "PIL.WebPImagePlugin",
    "urllib3",
    "urllib3.connectionpool",
    "filelock",
    "huggingface_hub",
    "huggingface_hub.utils._validators",
    "transformers.tokenization_utils_base",
    "fsspec",
    "fsspec.local",
]


class DeduplicatingHandler(logging.Handler):
    """Wraps another handler; collapses consecutive identical log records.

    When the same (logger-name, level, message) tuple is emitted N times in a
    row the output becomes a single line ending with ``[×N]``.  Different
    messages are emitted immediately, flushing any pending count first.

    Example output::

        2026-03-11 08:19:24 | PIL.PngImagePlugin | DEBUG | STREAM b'IHDR' 16 13  [×47]

    This is thread-safe: all mutable state is protected by a
    ``threading.Lock``.
    """

    def __init__(self, target: logging.Handler) -> None:
        super().__init__()
        self._target = target
        self._lock = threading.Lock()
        self._last_key: tuple | None = None
        self._last_record: logging.LogRecord | None = None
        self._count: int = 0

    def emit(self, record: logging.LogRecord) -> None:
        key = (record.name, record.levelno, record.getMessage())
        with self._lock:
            if key == self._last_key:
                self._count += 1
            else:
                self._flush_last()
                self._last_key = key
                self._last_record = record
                self._count = 1

    def _flush_last(self) -> None:
        """Emit the pending record (with count suffix if repeated). NOT thread-safe — caller holds lock."""
        if self._last_record is None:
            return
        if self._count > 1:
            self._last_record.msg = f"{self._last_record.getMessage()}  [×{self._count}]"
            self._last_record.args = ()
        try:
            self._target.emit(self._last_record)
        except Exception:
            self.handleError(self._last_record)
        self._last_record = None
        self._last_key = None
        self._count = 0

    def flush(self) -> None:
        with self._lock:
            self._flush_last()
        self._target.flush()

    def close(self) -> None:
        with self._lock:
            self._flush_last()
        self._target.close()
        super().close()


def suppress_noisy_loggers(level: int = logging.WARNING) -> None:
    """Set all known noisy third-party loggers to *level* (default: WARNING).

    Call this immediately after ``logging.basicConfig`` / after setting up the
    root logger so that subsequent third-party imports respect the level.
    """
    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(level)
