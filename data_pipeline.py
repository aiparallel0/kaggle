# =============================================================================
# data_pipeline.py
# Merged from: dataset_normalizer.py, dataset_loaders.py, preprocess_seller_split.py, dataset_preparation.py
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# =============================================================================
"""
data_pipeline.py — Merged data pipeline module.

Combines dataset_normalizer.py, dataset_loaders.py, preprocess_seller_split.py,
and dataset_preparation.py into a single module for the DONUT Receipt KIE pipeline.

Sections:
  1. DatasetNormalizer and field-value normalisation utilities (from dataset_normalizer.py)
  2. ABC-based dataset loaders for SROIE, WildReceipt, FUNSD, Invoices-DONUT (from dataset_loaders.py)
  3. Seller-string splitting with spaCy NER and heuristic fallback (from preprocess_seller_split.py)
  4. YOLO + TrOCR data preparation utilities (from dataset_preparation.py)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, overload

# Ensure sibling modules are importable regardless of CWD.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── Transformers compat shim ──────────────────────────────────────────
# In transformers ≥4.47 PreTrainedTokenizerBase moved to
# transformers.tokenization_utils_base.  The `datasets` library still
# tries to access it at the old location during load_dataset() which
# causes an AttributeError.  Restore the alias so CORD and
# Invoices-DONUT downloads succeed.
try:
    import transformers

    if not hasattr(transformers, "PreTrainedTokenizerBase"):
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        transformers.PreTrainedTokenizerBase = PreTrainedTokenizerBase
except (ImportError, AttributeError):
    pass

import resource_manager as _mm  # noqa: E402
from constants import EMPTY_GT, FIELDS, SEED, _get_sroie_dir  # noqa: E402
from constants import IMAGE_EXTS as _IMAGE_EXTS_SET  # noqa: E402


# ---------------------------------------------------------------------------
# Inline cross-platform file lock — replaces the 'filelock' PyPI package.
# Fix: issue_report_summary medium #8 — prevent TOCTOU race on .done_ markers.
# ---------------------------------------------------------------------------
@contextmanager
def _file_lock(lock_path: Path, timeout: float = 60.0):
    """Acquire an advisory exclusive lock on *lock_path*, yield, then release.

    On POSIX uses ``fcntl.flock`` (non-blocking poll with backoff).
    On Windows (and if fcntl is unavailable) falls back to a best-effort
    polling loop using ``os.open(O_CREAT|O_EXCL)`` atomicity.

    The lock is advisory only — uncooperative processes are not blocked.
    All callers in this codebase use this helper, so it is sufficient.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout

    try:
        import fcntl as _fcntl  # POSIX only

        lock_file = open(lock_path, "a")  # noqa: SIM115, WPS515
        try:
            while True:
                try:
                    _fcntl.flock(lock_file, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                    break  # acquired
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Could not acquire file lock on {lock_path} within {timeout}s"
                        ) from None
                    time.sleep(0.1)
            try:
                yield
            finally:
                _fcntl.flock(lock_file, _fcntl.LOCK_UN)
        finally:
            lock_file.close()

    except ImportError:
        # Windows / no fcntl: use O_CREAT|O_EXCL atomic create as lock
        fd = None
        try:
            while True:
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    break  # created exclusively → lock acquired
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Could not acquire file lock on {lock_path} within {timeout}s"
                        ) from None
                    time.sleep(0.1)
            yield
        finally:
            if fd is not None:
                os.close(fd)
            try:
                lock_path.unlink()
            except OSError:
                pass


__all__ = [
    "Sample",
    "DatasetLoadError",
    "BaseDatasetLoader",
    "SROIELoader",
    "WildReceiptLoader",
    "FUNSDLoader",
    "InvoicesDonutLoader",
    "load_sroie_train",
    "load_sroie_test",
    "load_sroie_val",
    "load_wildreceipt",
    "load_funsd",
    "load_invoices_donut",
    "get_combined_dataset",
    "get_dataset_loader",
    "split_dataset",
    # from preprocess_seller_split
    "build_cache",
    "main",
    # from dataset_normalizer
    "DatasetNormalizer",
    "normalise_samples",
    "extract_address_from_seller",
    "FIELD_ALIASES",
    # from dataset_preparation
    "group_words_into_lines",
    "build_yolo_split",
    "build_trocr_split",
    "write_yolo_yaml",
    "prepare_all",
    "validate_preparation",
]

# ── dataset_normalizer ───────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Field alias map: non-canonical name → canonical SROIE field name
FIELD_ALIASES: dict[str, str] = {
    # company aliases
    "store_name": "company",
    "merchant": "company",
    "vendor": "company",
    "shop": "company",
    "seller": "company",
    "business_name": "company",
    # date aliases
    "receipt_date": "date",
    "transaction_date": "date",
    "invoice_date": "date",
    # total aliases
    "amount": "total",
    "grand_total": "total",
    "total_amount": "total",
    "sum": "total",
    # address aliases
    "store_address": "address",
    "location": "address",
}

# Currency prefixes to strip during normalisation (only used for YOLO/TrOCR pipeline).
# Order matters: longer prefixes (e.g. "S$") must appear before any prefix they start
# with (e.g. "$") so that the longest match is found first.
_CURRENCY_PREFIXES = ("RM", "S$", "USD", "SGD", "MYR", "$", "£", "€", "¥")

Sample = tuple[Path, dict[str, str]]


class DatasetNormalizer:
    """Normalise a list of (Path, dict) samples to the canonical SROIE schema.

    Parameters
    ----------
    coverage_threshold:
        Minimum fraction of samples that must have a non-empty value for each
        of the *mandatory* fields (company, date, total).  Raises ValueError
        if coverage drops below this threshold after normalisation.
        Set to 0.0 to disable the check.
    strip_currency:
        If True, strip common currency symbols from the 'total' field.
        Use for the YOLO/TrOCR pipeline only — NOT for DONUT (which learns
        the raw format including currency symbols).
    source_name:
        Human-readable dataset name for log messages.
    """

    MANDATORY_FIELDS = ("company", "date", "total")

    def __init__(
        self,
        coverage_threshold: float = 0.30,
        strip_currency: bool = False,
        source_name: str = "unknown",
    ) -> None:
        self.coverage_threshold = coverage_threshold
        self.strip_currency = strip_currency
        self.source_name = source_name

    def normalise(self, samples: list[Sample]) -> list[Sample]:
        """Return a new list of samples normalised to the canonical schema."""
        if not samples:
            return []

        normalised: list[Sample] = []
        for img_path, raw_gt in samples:
            gt = self._normalise_dict(raw_gt)
            normalised.append((img_path, gt))

        self._assert_coverage(normalised)
        return normalised

    def _normalise_dict(self, raw: dict[str, Any]) -> dict[str, str]:
        """Normalise a single ground-truth dict to the canonical schema."""
        # Step 1: resolve aliases
        resolved: dict[str, str] = {}
        for k, v in raw.items():
            normalised_key = k.lower().strip()
            canonical_key = FIELD_ALIASES.get(normalised_key, normalised_key)
            if canonical_key in FIELDS:
                resolved[canonical_key] = str(v).strip() if v is not None else ""

        # Step 2: ensure all 4 canonical keys are present, default to ""
        result: dict[str, str] = {f: "" for f in FIELDS}
        for f in FIELDS:
            val = resolved.get(f, "")
            # Treat "N/A", "n/a", "null", "none", "nan" as empty
            if val.lower() in ("n/a", "null", "none", "nan", "-"):
                val = ""
            result[f] = val

        # Step 3: optional currency strip (YOLO/TrOCR only)
        if self.strip_currency and result["total"]:
            total = result["total"].strip()
            for prefix in _CURRENCY_PREFIXES:
                if total.upper().startswith(prefix.upper()):
                    total = total[len(prefix) :].strip()
                    break
            result["total"] = total

        return result

    def _assert_coverage(self, samples: list[Sample]) -> None:
        """Raise ValueError if coverage for any mandatory field is below threshold."""
        if self.coverage_threshold <= 0.0 or not samples:
            return
        n = len(samples)
        for field in self.MANDATORY_FIELDS:
            count = sum(1 for _, gt in samples if gt.get(field, ""))
            coverage = count / n
            if coverage < self.coverage_threshold:
                raise ValueError(
                    f"[DatasetNormalizer] {self.source_name}: field '{field}' "
                    f"coverage {coverage:.1%} is below threshold {self.coverage_threshold:.1%} "
                    f"({count}/{n} samples have non-empty values). "
                    f"Check the dataset loader or lower coverage_threshold."
                )
            logger.info(
                "[DatasetNormalizer] %s: field '%s' coverage %.1f%% (%d/%d)",
                self.source_name,
                field,
                coverage * 100,
                count,
                n,
            )


def normalise_samples(
    samples: list[Sample],
    source_name: str = "unknown",
    coverage_threshold: float = 0.30,
    strip_currency: bool = False,
    normalizer: DatasetNormalizer | None = None,
) -> list[Sample]:
    """Convenience wrapper around DatasetNormalizer.normalise().

    Parameters
    ----------
    normalizer : DatasetNormalizer | None
        Optional pre-configured normalizer instance.  When *None*
        (default), a new ``DatasetNormalizer`` is constructed from the
        remaining keyword arguments.
    """
    if normalizer is None:
        normalizer = DatasetNormalizer(
            coverage_threshold=coverage_threshold,
            strip_currency=strip_currency,
            source_name=source_name,
        )
    return normalizer.normalise(samples)


# ---------------------------------------------------------------------------
# extract_address_from_seller
# Absorbed from address_extractor.py — regex-based heuristic that splits the
# combined ``seller`` string in the Invoices-DONUT dataset into a
# (company_name, address) pair.  Lives here because it is a dataset
# normalisation utility consumed by InvoicesDonutLoader in dataset_loaders.py.
# ---------------------------------------------------------------------------

# Matches a leading street number followed by a space and at least one letter:
# e.g. "123 Main", "4500 Oak".
_STREET_NUMBER_RE = re.compile(r"\b\d+\s+[A-Za-z]")

# P.O. Box variants
_PO_BOX_RE = re.compile(r"\bP\.?\s*O\.?\s*Box\b", re.IGNORECASE)

# Common street-type suffixes that follow a street name.
_STREET_SUFFIX_RE = re.compile(
    r"\b(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|"
    r"Way|Lane|Ln|Court|Ct|Place|Pl|Terrace|Ter|Circle|Cir|"
    r"Highway|Hwy|Parkway|Pkwy|Trail|Trl|Run|Loop|Row)\b",
    re.IGNORECASE,
)

# US-style state abbreviation + ZIP: e.g. "MA 46228" or "CA 90210-1234".
_STATE_ZIP_RE = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")

# Patterns ordered by decreasing specificity so the earliest match wins.
_ADDRESS_PATTERNS: list[re.Pattern[str]] = [
    _PO_BOX_RE,
    _STREET_NUMBER_RE,
    _STREET_SUFFIX_RE,
    _STATE_ZIP_RE,
]


def extract_address_from_seller(seller_str: str) -> tuple[str, str]:
    """Split a combined Invoices-DONUT seller string into (company_name, address).

    Parameters
    ----------
    seller_str:
        Raw seller string, e.g.
        ``"Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228"``

    Returns
    -------
    tuple[str, str]
        ``(company_name, address)`` — *address* is an empty string when no
        address pattern is detected in *seller_str*.
    """
    seller_str = seller_str.strip()
    if not seller_str:
        return ("", "")

    # Find the earliest position where any address pattern matches.
    earliest_start: int | None = None
    for pattern in _ADDRESS_PATTERNS:
        m = pattern.search(seller_str)
        if m and (earliest_start is None or m.start() < earliest_start):
            earliest_start = m.start()

    if earliest_start is None:
        return (seller_str, "")

    company_name = seller_str[:earliest_start].strip().rstrip(",").strip()
    address = seller_str[earliest_start:].strip()
    return (company_name, address)


# ── dataset_loaders ──────────────────────────────────────────────────────

# ── Shared constants (derived from constants.py) ─────────────────────
_SROIE_FIELDS = frozenset(EMPTY_GT.keys())
_IMAGE_EXTS = _IMAGE_EXTS_SET

# ── Seller split cache (loaded lazily on first use) ───────────────────
_SELLER_SPLIT_CACHE: dict[str, dict[str, str]] | None = None


def _load_seller_split_cache() -> dict[str, dict[str, str]]:
    """Load seller_split_cache.json from the repo root (lazy, cached in module)."""
    global _SELLER_SPLIT_CACHE
    if _SELLER_SPLIT_CACHE is None:
        cache_path = Path(__file__).parent / "seller_split_cache.json"
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                # Strip metadata key so only seller-string entries remain.
                _SELLER_SPLIT_CACHE = {
                    k: v for k, v in raw.items() if k != "_metadata" and isinstance(v, dict)
                }
            except (json.JSONDecodeError, OSError):
                _SELLER_SPLIT_CACHE = {}
        else:
            _SELLER_SPLIT_CACHE = {}
    return _SELLER_SPLIT_CACHE


# ======================================================================
#  Custom exception
# ======================================================================


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded or produces zero samples.

    All loader classes raise this instead of silently returning [].
    The message is prefixed with ``FATAL:`` for grep-ability.
    """

    def __init__(self, dataset_name: str, reason: str) -> None:
        self.dataset_name = dataset_name
        self.reason = reason
        msg = f"FATAL: [{dataset_name}] {reason}"
        # Always echo to stderr so CI logs can grep for FATAL:
        print(msg, file=sys.stderr, flush=True)
        super().__init__(msg)


# ======================================================================
#  Path helpers — no module-level constants (BUG C FIX)
# ======================================================================


def _get_datasets_dir() -> Path:
    """Return auxiliary-dataset root; respects DONUT_WORKSPACE env var.

    Re-reads the environment variable on *every* call so that callers
    who set it after import time (e.g. run_all.py) get the right path.
    """
    return Path(os.environ.get("DONUT_WORKSPACE", "/workspace")) / "datasets"


def _ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it doesn't exist, then return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ======================================================================
#  Download helper
# ======================================================================


def _download_with_progress(url: str, dest_path: Path) -> None:
    """Download *url* to *dest_path* with 60-s timeout, chunked streaming,
    per-chunk progress, and up to 3 retries with exponential back-off."""
    max_retries = 3
    backoff = [5, 15, 45]
    for attempt in range(max_retries):
        try:
            response = urllib.request.urlopen(url, timeout=60)
            total = int(response.headers.get("Content-Length") or 0)
            total_mb = total / 1024 / 1024
            downloaded = 0
            t0 = time.time()
            with open(dest_path, "wb") as fh:
                while True:
                    chunk = response.read(8 * 1024 * 1024)  # 8 MB chunks
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    dl_mb = downloaded / 1024 / 1024
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\r[{pct:5.1f}%] {dl_mb:.1f} / {total_mb:.1f} MB", end="", flush=True
                        )
                    else:
                        print(f"\r[{dl_mb:.1f} MB downloaded]", end="", flush=True)
            elapsed = time.time() - t0
            print(f"\nDownload complete in {elapsed:.1f}s")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            if dest_path.exists():
                dest_path.unlink()
            if attempt < max_retries - 1:
                wait = backoff[attempt]
                print(
                    f"\n[download] Attempt {attempt + 1} failed: {exc}. Retrying in {wait}s ...",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise


# ======================================================================
#  Inline HuggingFace dataset downloader — fallback when `datasets` absent
# ======================================================================


def _read_hf_token() -> str | None:
    """Read HuggingFace token from hf_token.txt (one line, no newline required)."""
    try:
        token_path = Path(__file__).parent / "hf_token.txt"
        if token_path.exists():
            return token_path.read_text().strip() or None
    except OSError:
        pass
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _hf_api_get(url: str, hf_token: str | None = None) -> Any:
    """GET a HuggingFace API URL and return parsed JSON."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "inline-hf-downloader/1.0")
    if hf_token:
        req.add_header("Authorization", f"Bearer {hf_token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _hf_discover_config(repo_id: str, hf_token: str | None = None) -> str:
    """Discover the first available config name for a HuggingFace dataset.

    Queries the datasets-server ``/splits`` endpoint and returns the config
    name from the first split entry.  Falls back to ``"default"`` when the
    API call fails or the response is malformed.
    """
    try:
        splits_info = _hf_api_get(
            f"https://datasets-server.huggingface.co/splits?dataset={repo_id}",
            hf_token,
        )
        return (
            splits_info["splits"][0].get("config", "default")
            if splits_info.get("splits")
            else "default"
        )
    except (KeyError, IndexError, TypeError):
        return "default"


def _hf_fetch_rows_batch(
    repo_id: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    hf_token: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch a single batch of rows from the HuggingFace datasets-server API.

    Returns the list of row dicts from the ``rows`` key, or an empty list
    when the request fails (the caller decides whether to retry or stop).

    Raises
    ------
    Exception
        Propagates any HTTP / JSON error so the caller can log and decide
        whether to continue.
    """
    resp = _hf_api_get(
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset={repo_id}&config={config}&split={split}"
        f"&offset={offset}&length={length}",
        hf_token,
    )
    return resp.get("rows", [])


def _hf_write_and_mark(
    all_rows: list[dict[str, Any]],
    img_dir: Path,
    jsonl_path: Path,
    split: str,
    marker_path: Path,
    hf_token: str | None = None,
) -> None:
    """Write downloaded rows to JSONL, save images to disk, and create the done marker.

    For each row:
    - Image columns (dicts with a ``"src"`` key) are downloaded to *img_dir*
      and their path is stored in the JSONL record.
    - All other JSON-serialisable values are written verbatim.

    After all rows are written, the *marker_path* file is touched to signal
    that the download is complete.
    """
    log = logging.getLogger(__name__)
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for idx, row_wrapper in enumerate(all_rows):
            row = row_wrapper.get("row", row_wrapper)
            record: dict[str, Any] = {}
            for key, val in row.items():
                if isinstance(val, dict) and "src" in val:
                    # Image feature — download the image URL
                    img_path = img_dir / f"{split}_{idx:06d}.jpg"
                    if not img_path.exists():
                        try:
                            img_req = urllib.request.Request(val["src"])
                            if hf_token:
                                img_req.add_header("Authorization", f"Bearer {hf_token}")
                            with urllib.request.urlopen(img_req, timeout=30) as img_resp:
                                img_path.write_bytes(img_resp.read())
                        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as img_exc:
                            log.warning("[inline-hf] Image %d download failed: %s", idx, img_exc)
                    record[key] = str(img_path)
                elif isinstance(val, (str, int, float, list, dict, bool)) or val is None:
                    record[key] = val
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    marker_path.touch()
    log.info("[inline-hf] Download complete: %d rows saved to %s", len(all_rows), jsonl_path.parent)


def _hf_acquire_lock_and_check_cache(cache_dir: Path, marker: Path) -> bool:
    """Acquire the in-process threading lock and check whether *marker* exists.

    Uses a lazily-initialised ``threading.Lock`` stored on the module-level
    ``_hf_download_dataset_inline`` function object to prevent a TOCTOU race
    when two threads both see ``marker.exists() == False`` and both start
    downloading.

    Returns ``True`` when the cache is already populated (caller should
    short-circuit) or ``False`` when the download is still needed.
    """
    import threading as _threading

    _hf_download_lock = getattr(_hf_download_dataset_inline, "_lock", None)
    if _hf_download_lock is None:
        _hf_download_lock = _threading.Lock()
        _hf_download_dataset_inline._lock = _hf_download_lock  # type: ignore[attr-defined]

    with _hf_download_lock:
        # Re-check inside the lock — another thread may have finished while we waited.
        return marker.exists()


def _hf_fetch_row_count(
    repo_id: str, split: str, hf_token: str | None = None, fallback: int = 1000
) -> int:
    """Query the HuggingFace datasets-server ``/size`` endpoint for *split*.

    Returns the number of rows when the API call succeeds, or *fallback*
    when the request fails or the response is malformed.
    """
    try:
        size_info = _hf_api_get(
            f"https://datasets-server.huggingface.co/size?dataset={repo_id}",
            hf_token,
        )
        for s in size_info.get("size", {}).get("splits", []):
            if s.get("split") == split:
                return int(s.get("num_rows", fallback))
    except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError):
        pass
    return fallback


def _hf_download_rows_batch(
    repo_id: str,
    config: str,
    split: str,
    hf_token: str | None,
    total_rows: int,
) -> list[dict]:
    """Download all rows for *split* in batches of 100.

    Uses :func:`_hf_fetch_rows_batch` for each page and stops on the
    first failure or when *total_rows* have been collected.

    Returns the accumulated list of row dicts.
    """
    log = logging.getLogger(__name__)
    batch_size = 100
    all_rows: list[dict] = []
    for offset in range(0, total_rows + batch_size, batch_size):
        try:
            rows = _hf_fetch_rows_batch(repo_id, config, split, offset, batch_size, hf_token)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            log.warning("[inline-hf] Batch offset=%d failed: %s — stopping early.", offset, exc)
            break
        if not rows:
            break
        all_rows.extend(rows)
        log.info("[inline-hf] %d/%d rows fetched …", len(all_rows), total_rows)
        if len(all_rows) >= total_rows:
            break
    return all_rows


def _hf_download_dataset_inline(
    repo_id: str,
    dest_dir: Path,
    hf_token: str | None = None,
    split: str = "train",
) -> Path:
    """Download a HuggingFace dataset via the datasets-server HTTP API.

    Saves to dest_dir/hf_cache_inline/ as:
      images/{split}_{idx:06d}.jpg  — one JPEG per sample
      data_{split}.jsonl            — one JSON record per line (no image bytes)
      .done_{split}                 — marker written on success

    Returns the hf_cache_inline/ directory path.

    This is a zero-external-dependency replacement for:
        ds = load_dataset(repo_id); ds.save_to_disk(...)

    Delegates to:
    - :func:`_hf_acquire_lock_and_check_cache` — in-process threading lock + marker check
    - :func:`_hf_discover_config` — HF API config discovery
    - :func:`_hf_fetch_row_count` — total row count via size endpoint
    - :func:`_hf_download_rows_batch` — paginated batch download loop
    - :func:`_hf_fetch_rows_batch` — single-page row fetching
    - :func:`_hf_write_and_mark` — JSONL + image writing and marker creation
    """
    cache_dir = dest_dir / "hf_cache_inline"
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / f".done_{split}"

    # ROBUSTNESS: in-process lock prevents a TOCTOU race when two threads
    # both see marker.exists() == False and both start downloading.
    if _hf_acquire_lock_and_check_cache(cache_dir, marker):
        return cache_dir

    # Fix: issue_report_summary medium #8 — atomic file lock so two concurrent
    # processes cannot both start downloading and corrupt the cache.
    lock_path = cache_dir / f".lock_{split}"
    with _file_lock(lock_path):
        # Re-check inside the file lock: another process may have completed
        # the download while we were waiting.
        if marker.exists():
            return cache_dir

        img_dir = cache_dir / "images"
        img_dir.mkdir(exist_ok=True)
        jsonl_path = cache_dir / f"data_{split}.jsonl"

        log = logging.getLogger(__name__)
        log.info("[inline-hf] Downloading %s/%s from HuggingFace datasets-server …", repo_id, split)

        config = _hf_discover_config(repo_id, hf_token)
        total_rows = _hf_fetch_row_count(repo_id, split, hf_token)
        all_rows = _hf_download_rows_batch(repo_id, config, split, hf_token, total_rows)
        _hf_write_and_mark(all_rows, img_dir, jsonl_path, split, marker, hf_token)
        return cache_dir


def _hf_load_jsonl_rows(cache_dir: Path, split: str = "train") -> list[dict[str, Any]]:
    """Load the inline-downloaded JSONL cache produced by _hf_download_dataset_inline().

    Each returned dict has the same keys as the original HF dataset row, except:
    - image columns contain a Path string pointing to the saved JPEG file
      (not a PIL Image; callers use PIL.Image.open() or inline image loader)
    """
    jsonl_path = cache_dir / f"data_{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Inline HF cache not found at {jsonl_path}. "
            f"Run _hf_download_dataset_inline() first, or install the `datasets` package."
        )
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ======================================================================
#  Schema validation helpers
# ======================================================================


def _validate_sample_schema(sample: Sample, dataset_name: str) -> bool:
    """Spot-check that *sample* matches the SROIE schema.

    Returns True if the sample is a 2-tuple of (Path, dict) with the
    expected keys.  Does **not** raise — the caller decides what to do.
    """
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        return False
    path, gt = sample
    if not isinstance(path, Path):
        return False
    if not isinstance(gt, dict):
        return False
    return _SROIE_FIELDS.issubset(gt.keys())


def _validate_samples_nonempty(samples: list[Sample], dataset_name: str) -> list[Sample]:
    """Return *samples* unchanged if non-empty; raise DatasetLoadError otherwise."""
    if not samples:
        raise DatasetLoadError(
            dataset_name,
            "Loader returned 0 samples — dataset is MISSING from this run.",
        )
    return samples


def _log_field_coverage(samples: list[tuple[Path, dict[str, str]]], dataset_name: str) -> None:
    """Log per-field fill rates for a loaded dataset.

    Parameters
    ----------
    samples : list of (Path, dict)
        Standard Sample tuples (image_path, ground_truth_dict).
    dataset_name : str
        Human-readable name for log messages.
    """
    if not samples:
        return
    from constants import FIELDS

    n = len(samples)
    for field in FIELDS:
        filled = sum(1 for _, gt in samples if gt.get(field, "").strip())
        pct = filled / n * 100
        logging.getLogger(__name__).info(
            "  %s field coverage: %s = %d/%d (%.1f%%)", dataset_name, field, filled, n, pct
        )


# ======================================================================
#  SROIE key-file reader (BUG A FIX)
# ======================================================================


def _load_key_file(key_dir: Path, stem: str) -> dict[str, str]:
    """Load a SROIE key file for the given image stem.

    BUG A FIX: The SROIE repo stores annotations as plain .txt files with
    4 lines (company / date / address / total), NOT as .json files.  The old
    code only tried .json, so ``key_file.exists()`` was always False and every
    sample was silently skipped.  We now try .txt first (the format the repo
    actually ships), then .json as a fallback for pre-converted data.
    """
    # Try .txt first — the 4-line format that the repo actually provides
    key_file_txt = key_dir / (stem + ".txt")
    if key_file_txt.exists():
        try:
            lines = key_file_txt.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) >= 4:
                return {
                    "company": lines[0].strip(),
                    "date": lines[1].strip(),
                    "address": " ".join(line.strip() for line in lines[2:-1]).strip(),
                    "total": lines[-1].strip(),
                }
        except OSError:
            pass

    # Fallback: .json in case the user has pre-converted the files
    key_file_json = key_dir / (stem + ".json")
    if key_file_json.exists():
        try:
            gt = json.loads(key_file_json.read_text(encoding="utf-8"))
            return {k: str(gt.get(k, "")) for k in EMPTY_GT}
        except (json.JSONDecodeError, OSError):
            pass

    return {}


# ======================================================================
#  Abstract base class
# ======================================================================


class BaseDatasetLoader(ABC):
    """Abstract base class for all dataset loaders.

    Concrete subclasses must implement four methods:

    - ``load(split)``         → list of ``(Path, dict)`` samples
    - ``validate_cache()``    → True if local cache is usable
    - ``clear_cache()``       → delete local cache
    - ``sample_count(split)`` → number of samples in *split*

    ``load()`` must raise :class:`DatasetLoadError` on failure and must
    never return an empty list.
    """

    # Friendly name shown in log messages
    name: str = "base"

    # ── abstract interface ────────────────────────────────────────────

    @abstractmethod
    def load(self, split: str = "train") -> list[Sample]:
        """Load samples for *split*.

        Must raise :class:`DatasetLoadError` on failure.
        Must never return an empty list — raise instead.
        """
        ...

    @abstractmethod
    def validate_cache(self) -> bool:
        """Return True if local cache is present **and** schema-valid.

        Must spot-check the first record, not just test file presence.
        """
        ...

    def clear_cache(self) -> None:  # noqa: B027
        """Delete any local cached data for this dataset.

        Subclasses that download data should override this to remove
        their local cache directory.  The default is a no-op, which is
        correct for datasets whose data is user-provided (e.g. SROIE).
        """

    @abstractmethod
    def sample_count(self, split: str = "train") -> int:
        """Return the number of samples in *split* without fully loading."""
        ...

    @property
    @abstractmethod
    def _subdir_name(self) -> str:
        """Subdirectory name under data/ for this dataset (e.g. 'funsd')."""
        ...

    # ── helpers available to subclasses ───────────────────────────────

    def _log(self, message: str) -> None:
        """Log a bracketed info line (non-blocking, no GIL flush)."""
        logging.getLogger(__name__).info("[%s] %s", self.name, message)

    def _warn(self, message: str) -> None:
        """Log a bracketed warning (non-blocking, no GIL flush)."""
        logging.getLogger(__name__).warning("[%s] %s", self.name, message)

    def _fatal(self, reason: str) -> DatasetLoadError:
        """Build and return a :class:`DatasetLoadError` (also prints FATAL:)."""
        return DatasetLoadError(self.name, reason)

    # ── Phase 1 Consolidation: Shared utilities for all loaders ────────

    def _get_dest_dir(self, subdir: str) -> Path:
        """Get and create the destination directory for this dataset.

        Args:
            subdir: Subdirectory name under data/ (e.g., "wildreceipt", "funsd")

        Returns:
            Path to the destination directory (created if not present)

        Example: self._dest_dir() = self._get_dest_dir("wildreceipt")
        """
        return _ensure_dir(_get_datasets_dir() / subdir)

    def _get_marker_path(self, dest_dir: Path) -> Path:
        """Get the marker file path for a cached dataset.

        Args:
            dest_dir: Destination directory (from _get_dest_dir)

        Returns:
            Path to .downloaded marker file

        Example: self._marker() = self._get_marker_path(self._dest_dir())
        """
        return dest_dir / ".downloaded"

    def _get_hf_cache_dir(self, dest_dir: Path) -> Path:
        """Get the HuggingFace cache subdirectory for a dataset.

        Args:
            dest_dir: Destination directory (from _get_dest_dir)

        Returns:
            Path to hf_cache/ subdirectory

        Example: self._hf_cache() = self._get_hf_cache_dir(self._dest_dir())
        """
        return dest_dir / "hf_cache"

    # ── Template methods using _subdir_name ───────────────────────────

    def _dest_dir(self) -> Path:
        """Get and create the destination directory for this dataset."""
        return self._get_dest_dir(self._subdir_name)

    def _marker(self) -> Path:
        """Get the marker file path for a cached dataset."""
        return self._get_marker_path(self._dest_dir())

    def _hf_cache(self) -> Path:
        """Get the HuggingFace cache subdirectory for a dataset."""
        return self._get_hf_cache_dir(self._dest_dir())

    @staticmethod
    def _empty_gt_dict() -> dict[str, str]:
        """Return a new empty ground-truth dict with all SROIE fields.

        Consolidation: Replaces `{k: "" for k in EMPTY_GT}` pattern throughout.

        Returns:
            {"company": "", "date": "", "address": "", "total": ""}
        """
        return {k: "" for k in EMPTY_GT}

    def _check_and_update_cache(
        self, marker: Path, cache_path: Path, message_if_invalid: str = None
    ) -> bool:
        """Check if cache is valid; warn and delete marker if not.

        Args:
            marker: Path to .downloaded marker file
            cache_path: Path to validate (e.g., hf_cache/ or extract dir)
            message_if_invalid: Optional warning message to log if cache invalid

        Returns:
            True if cache is valid (marker present AND cache_path exists)
            False if cache is invalid (marker will be deleted for re-download)
        """
        if marker.exists():
            if not cache_path.exists():
                if message_if_invalid:
                    self._warn(message_if_invalid)
                marker.unlink(missing_ok=True)
                return False
            return True
        return False


# ======================================================================
#  SROIELoader
# ======================================================================


class SROIELoader(BaseDatasetLoader):
    """Load SROIE from local img/key directories.

    Supports three splits:
    - ``train`` → ``img/`` + ``key/``
    - ``test``  → ``test_img/`` + ``test_key/``
    - ``val``   → ``val_img/`` + ``val_key/``

    For ``test`` and ``val`` the loader falls back to an empty list with
    a warning (instead of raising) because those directories may
    legitimately not exist in some pipeline configurations.
    """

    name = "SROIE"

    @property
    def _subdir_name(self) -> str:
        return "sroie"

    # Map split name → (img_subdir, key_subdir)
    _SPLIT_DIRS = {
        "train": ("img", "key"),
        "test": ("test_img", "test_key"),
        "val": ("val_img", "val_key"),
    }

    # ── public interface ──────────────────────────────────────────────

    def load(self, split: str = "train") -> list[Sample]:
        """Load SROIE samples for *split*.

        Raises DatasetLoadError for the ``train`` split if the directory
        does not exist.  For ``test`` and ``val`` splits, returns ``[]``
        with a warning (matching legacy behaviour expected by callers).
        """
        img_subdir, key_subdir = self._split_dirs(split)
        sroie_dir = _get_sroie_dir()
        img_dir = sroie_dir / img_subdir
        key_dir = sroie_dir / key_subdir

        if not img_dir.exists():
            if split == "train":
                raise self._fatal(
                    f"Training img dir not found: {img_dir}. "
                    "Set SROIE_DATA_DIR to the correct path."
                )
            # test / val may be absent — warn but don't crash
            self._warn(f"{img_subdir}/ directory not found — returning 0 {split} samples.")
            return []

        samples: list[Sample] = []
        for img_path in sorted(
            p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ):
            gt = _load_key_file(key_dir, img_path.stem)
            # Fix: issue_report_summary high #5 — replace `if gt:` truthiness check with
            # explicit validation of all required SROIE fields. A dict like {"company": ""}
            # passes `if gt:` but causes KeyError during training on any access to a missing
            # field. All four SROIE fields must be present (values may be empty strings).
            if gt and all(field in gt for field in FIELDS):
                samples.append((img_path, gt))
            elif gt:
                # gt is non-empty but missing one or more required fields — log and skip.
                missing_fields = [f for f in FIELDS if f not in gt]
                logging.getLogger(__name__).warning(
                    "Skipping %s: ground-truth dict is missing required SROIE field(s) %s. "
                    "Found keys: %s",
                    img_path.name,
                    missing_fields,
                    list(gt.keys()),
                )

        if split == "train" and not samples:
            raise self._fatal(
                f"img dir exists ({img_dir}) but 0 samples matched — "
                "check that key/ files are present."
            )
        _log_field_coverage(samples, self.name)
        return samples

    def validate_cache(self) -> bool:
        """SROIE is local data; 'cache' is valid if train split loads
        at least one sample whose schema is correct."""
        try:
            samples = self.load("train")
        except DatasetLoadError:
            return False
        if not samples:
            return False
        return _validate_sample_schema(samples[0], self.name)

    def clear_cache(self) -> None:
        """SROIE data is user-provided — nothing to clear."""
        self._log("SROIE data is user-provided; clear_cache() is a no-op.")

    def sample_count(self, split: str = "train") -> int:
        """Count images in the split directory without loading key files."""
        img_subdir, _ = self._split_dirs(split)
        img_dir = _get_sroie_dir() / img_subdir
        if not img_dir.exists():
            return 0
        return sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)

    # ── private helpers ───────────────────────────────────────────────

    def _split_dirs(self, split: str) -> tuple[str, str]:
        """Return (img_subdir, key_subdir) for *split*, raising on bad name."""
        if split not in self._SPLIT_DIRS:
            raise ValueError(f"Unknown SROIE split '{split}'. Valid: {list(self._SPLIT_DIRS)}")
        return self._SPLIT_DIRS[split]


# ======================================================================
#  WildReceiptLoader
# ======================================================================


class WildReceiptLoader(BaseDatasetLoader):
    """Download WildReceipt from the OpenMMLab tar and normalize to SROIE schema.

    BUG 1 FIX: Old code used ``load_dataset("Theivaprakasham/wildreceipt")``
    which fails with "Dataset scripts are no longer supported" and silently
    returned [].  New code downloads the official OpenMMLab tar directly.
    """

    name = "WildReceipt"

    _URL = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"

    # WildReceipt label index → SROIE field mapping
    #   1 = Store_name_value  → company
    #   3 = Store_addr_value  → address
    #   7 = Date_value        → date
    #  23 = Total_value       → total
    _IDX_TO_FIELD: dict[int, str] = {
        1: "company",
        3: "address",
        7: "date",
        23: "total",
    }

    # ── download & extraction ─────────────────────────────────────────

    @property
    def _subdir_name(self) -> str:
        return "wildreceipt"

    def _inner_dir(self) -> Path:
        """The extracted ``wildreceipt/`` subdirectory inside _dest_dir()."""
        return self._dest_dir() / "wildreceipt"

    def _download(self) -> Path:
        """Download and extract the tar if not already cached."""
        dest = self._dest_dir()
        marker = self._marker()
        if marker.exists():
            # Validate cache: ensure train.txt exists inside the extracted dir
            if not (self._inner_dir() / "train.txt").exists():
                self._warn("Cache marker present but train.txt missing — re-downloading.")
                marker.unlink(missing_ok=True)
            else:
                return dest

        url = self._URL
        tar_path = dest / "wildreceipt.tar"
        try:
            self._log(f"Downloading from {url} ...")
            _download_with_progress(url, tar_path)
            self._log("Extracting tar ...")
            with tarfile.open(str(tar_path)) as tf:
                # Safe extraction: use filter='data' on Python 3.12+
                if sys.version_info >= (3, 12):
                    tf.extractall(str(dest), filter="data")
                else:
                    tf.extractall(str(dest))
            tar_path.unlink(missing_ok=True)
            marker.touch()
            self._log("Download and extraction complete.")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, tarfile.TarError) as exc:
            raise self._fatal(
                f"Download failed from {url}: {exc}. This experiment will have MISSING DATA."
            ) from exc
        return dest

    # ── public interface ──────────────────────────────────────────────

    def load(self, split: str = "train") -> list[Sample]:
        """Load WildReceipt (OpenMMLab format) and normalize to SROIE schema.

        Only the ``train`` split is used to prevent test-set contamination.
        """
        self._download()
        inner = self._inner_dir()

        # Map split to file name
        split_file = "train.txt" if split == "train" else f"{split}.txt"
        txt_path = inner / split_file
        if not txt_path.exists():
            raise self._fatal(f"'{split_file}' not found in {inner}. Download may be incomplete.")

        samples: list[Sample] = []
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Phase 1: Use shared consolidation utility
            gt = self._empty_gt_dict()
            for ann in obj.get("annotations", []):
                label_idx = int(ann.get("label", -1))
                field = self._IDX_TO_FIELD.get(label_idx)
                text = str(ann.get("text", "")).strip()
                if field and text:
                    gt[field] = (gt[field] + " " + text).strip() if gt[field] else text

            file_name = obj.get("file_name", "")
            img_path = inner / file_name
            if img_path.exists():
                samples.append((img_path, gt))

        _log_field_coverage(samples, self.name)
        return _validate_samples_nonempty(samples, self.name)

    def validate_cache(self) -> bool:
        """Check that the cache directory has train.txt and the first
        record can be parsed and has the right schema."""
        try:
            train_txt = self._inner_dir() / "train.txt"
            if not train_txt.exists():
                return False
            # Spot-check: load just the first record
            for line in train_txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Must have annotations list and file_name
                return not ("annotations" not in obj or "file_name" not in obj)
        except (OSError, json.JSONDecodeError):
            return False

    def clear_cache(self) -> None:
        """Remove the entire WildReceipt cache directory."""
        import shutil

        dest = self._dest_dir()
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
            self._log("Cache cleared.")

    def sample_count(self, split: str = "train") -> int:
        """Count lines in the split file without fully parsing."""
        split_file = "train.txt" if split == "train" else f"{split}.txt"
        txt_path = self._inner_dir() / split_file
        if not txt_path.exists():
            return 0
        return sum(1 for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip())


# ======================================================================
#  FUNSDLoader
# ======================================================================


class FUNSDLoader(BaseDatasetLoader):
    """Download FUNSD (Form Understanding in Noisy Scanned Documents) from
    HuggingFace and normalize to SROIE schema.

    FUNSD (nielsr/funsd) has 149 training + 50 test scanned business forms
    with BIO-tagged named entities typed as HEADER, QUESTION, ANSWER, or
    OTHER.  We reconstruct entity spans from the BIO sequence, then assign
    answer spans to SROIE fields using:
      - company : first HEADER span
      - date    : first ANSWER span matching a date regex
      - total   : first ANSWER span matching a currency/amount regex
      - address : first remaining ANSWER span with length > 5

    No HuggingFace token is required.  Replaces CORD which had tag
    coverage issues causing ~50% of training images to produce empty
    ground-truth fields, halving effective F1.
    """

    name = "FUNSD"

    # BIO tag indices from the ClassLabel in nielsr/funsd:
    # O=0, B-HEADER=1, I-HEADER=2, B-QUESTION=3, I-QUESTION=4,
    # B-ANSWER=5, I-ANSWER=6
    _TAG_TO_TYPE: dict[int, str] = {
        1: "header",
        2: "header",
        3: "question",
        4: "question",
        5: "answer",
        6: "answer",
    }
    _BEGIN_TAGS: frozenset[int] = frozenset({1, 3, 5})

    _DATE_RE = re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"
        r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
        r"\.?\s+\d{1,2},?\s+\d{4}\b",
        re.IGNORECASE,
    )
    _AMOUNT_RE = re.compile(r"\$?\s*\d[\d,]*\.\d{2}\b")

    # ── download ──────────────────────────────────────────────────────

    @property
    def _subdir_name(self) -> str:
        return "funsd"

    def _download(self) -> Path:
        """Download FUNSD from the HuggingFace datasets hub."""
        dest = self._dest_dir()
        marker = self._marker()
        if marker.exists():
            if (
                not self._hf_cache().exists()
                and not (dest / "hf_cache_inline" / ".done_train").exists()
            ):
                self._warn("Cache marker present but cache missing — re-downloading.")
                marker.unlink(missing_ok=True)
            else:
                return dest

        # Pin a stable commit so the dataset schema doesn't drift between runs.
        # Verified good on 2026-03-07 (experiments 1-8 reference run).
        # Override via env var HF_FUNSD_REVISION if upstream moves.
        _funsd_repo = "nielsr/funsd"
        _funsd_rev = os.environ.get("HF_FUNSD_REVISION", None)  # None → latest main

        hf_token = _read_hf_token()
        try:
            from datasets import load_dataset  # type: ignore

            self._log(f"Downloading {_funsd_repo} from HuggingFace ...")
            _kw: dict = {"trust_remote_code": False}
            if _funsd_rev:
                _kw["revision"] = _funsd_rev
            ds = load_dataset(_funsd_repo, token=hf_token or None, **_kw)
            ds.save_to_disk(str(self._hf_cache()))
            marker.touch()
            self._log("Download complete.")
        except ImportError:
            self._log("datasets package absent — using inline HF downloader for FUNSD ...")
            try:
                _hf_download_dataset_inline(_funsd_repo, dest, hf_token, split="train")
                marker.touch()
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                raise self._fatal(
                    f"Inline download failed for {_funsd_repo}: {exc}\n"
                    "  Tip: set HF_FUNSD_REVISION=<commit> or place hf_token.txt for auth."
                ) from exc
        except Exception as exc:  # intentional broad catch: HF datasets library
            raise self._fatal(
                f"Download failed for {_funsd_repo}: {exc}\n"
                "  If the repo requires authentication or has moved, set hf_token.txt "
                "or HF_FUNSD_REVISION=<commit>."
            ) from exc
        return dest

    # ── FUNSD → SROIE remapping ───────────────────────────────────────

    @classmethod
    def _extract_spans(cls, words: list[str], ner_tags: list[int]) -> dict[str, list[str]]:
        """Group consecutive BIO-tagged tokens into typed entity spans.

        Returns a dict with keys 'header', 'question', 'answer', each
        mapping to a list of reconstructed text strings (one per span).
        """
        spans: dict[str, list[str]] = {"header": [], "question": [], "answer": []}
        current_type: str | None = None
        current_words: list[str] = []

        for word, tag in zip(words, ner_tags):
            entity_type = cls._TAG_TO_TYPE.get(tag)
            is_begin = tag in cls._BEGIN_TAGS

            if is_begin:
                if current_words and current_type:
                    spans[current_type].append(" ".join(current_words))
                current_type = entity_type
                current_words = [word]
            elif entity_type is not None and entity_type == current_type:
                current_words.append(word)
            else:
                # O tag or broken I-tag
                if current_words and current_type:
                    spans[current_type].append(" ".join(current_words))
                current_type = entity_type
                current_words = [word] if entity_type else []

        if current_words and current_type:
            spans[current_type].append(" ".join(current_words))

        return spans

    @classmethod
    def _funsd_remap(cls, words: list[str], ner_tags: list[int]) -> dict[str, str]:
        """Map a FUNSD token sequence to the SROIE schema.

        Extraction strategy:
          - company : first HEADER span (fallback: first QUESTION span)
          - date    : first ANSWER span matching _DATE_RE
          - total   : first ANSWER span matching _AMOUNT_RE
          - address : first ANSWER span with len > 5 not claimed by above
        """
        # Phase 1: Use shared consolidation utility
        gt = BaseDatasetLoader._empty_gt_dict()
        try:
            spans = cls._extract_spans(words, ner_tags)

            # Company from first header span
            if spans["header"]:
                gt["company"] = spans["header"][0]
            elif spans["question"]:
                gt["company"] = spans["question"][0]

            # Date / total / address from answer spans (order matters)
            for ans in spans["answer"]:
                ans = ans.strip()
                if not ans:
                    continue
                if not gt["date"] and cls._DATE_RE.search(ans):
                    gt["date"] = ans
                elif not gt["total"] and cls._AMOUNT_RE.search(ans):
                    gt["total"] = ans
                elif not gt["address"] and len(ans) > 5:
                    gt["address"] = ans
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("FUNSD remap failed for sample: %s", exc)
        return gt

    # ── public interface ──────────────────────────────────────────────

    def _load_from_hf_arrow(self, split: str, img_dest_dir: Path) -> list[Sample] | None:
        """Try loading FUNSD from HF Arrow cache.

        Returns a list of samples on success, or ``None`` if the Arrow
        cache is missing or the ``datasets`` package is unavailable.
        """
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            return None

        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
            ds_splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            samples: list[Sample] = []
            for ds_split in ds_splits:
                if ds_split == "test":
                    continue
                if split != "train" and ds_split != split:
                    continue
                split_ds = ds[ds_split] if hasattr(ds, "keys") else ds
                for item in split_ds:
                    words = item.get("words", [])
                    ner_tags = item.get("ner_tags", [])
                    pil_image = item.get("image")
                    if not words or pil_image is None:
                        continue
                    gt = self._funsd_remap(words, ner_tags)
                    rgb = pil_image.convert("RGB")
                    img_hash = hashlib.md5(rgb.tobytes()).hexdigest()[:8]
                    img_path = img_dest_dir / f"{ds_split}_{img_hash}.jpg"
                    if not img_path.exists():
                        rgb.save(img_path, "JPEG")
                    samples.append((img_path, gt))
            _mm.release_hf_dataset(ds)
            _mm.flush_hf_arrow_cache()
            return samples
        except ImportError:
            return None
        except Exception as exc:  # intentional broad catch: HF datasets library
            raise self._fatal(f"Failed to load Arrow cache: {exc}") from exc

    def _load_from_inline_cache(self, split: str, dest: Path) -> list[Sample]:
        """Load FUNSD from the inline JSONL fallback cache."""
        inline_cache = dest / "hf_cache_inline"
        if not inline_cache.exists():
            raise self._fatal(
                "Neither HF Arrow cache nor inline cache found. "
                "Install the `datasets` package or run _hf_download_dataset_inline()."
            )
        samples: list[Sample] = []
        for row in _hf_load_jsonl_rows(inline_cache, split="train"):
            if split == "test":
                continue
            words = row.get("words", [])
            ner_tags = row.get("ner_tags", [])
            img_val = row.get("image")
            if not words or img_val is None:
                continue
            img_path = Path(str(img_val))
            if not img_path.exists():
                continue
            gt = self._funsd_remap(words, ner_tags)
            samples.append((img_path, gt))
        return samples

    def load(self, split: str = "train") -> list[Sample]:
        """Load FUNSD and normalize to SROIE schema.

        Skips the ``test`` split to prevent contamination.  Uses content
        hash for on-disk filenames to avoid collisions.

        Uses HF Arrow cache (datasets library) when available; falls back to
        the inline JSONL cache written by _hf_download_dataset_inline().
        """
        dest = self._download()
        img_dest_dir = _ensure_dir(dest / "images")

        # Try HF Arrow first
        samples = self._load_from_hf_arrow(split, img_dest_dir)
        if samples is None:
            # Fall back to inline cache
            samples = self._load_from_inline_cache(split, dest)

        if not samples:
            self._warn("FUNSD returned 0 samples — check HF cache.")
            return []
        _log_field_coverage(samples, self.name)
        return samples

    def validate_cache(self) -> bool:
        """Check that hf_cache exists and first record has words + ner_tags."""
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            return False
        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
            splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            first_split = ds[splits[0]] if hasattr(ds, "keys") else ds
            if len(first_split) == 0:
                return False
            item = first_split[0]
            return bool(item.get("words")) and "ner_tags" in item
        except Exception:  # intentional broad catch: HF datasets library
            return False

    def clear_cache(self) -> None:
        """Remove the entire FUNSD cache directory."""
        import shutil

        dest = self._dest_dir()
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
            self._log("Cache cleared.")

    def sample_count(self, split: str = "train") -> int:
        """Return approximate sample count from the HF cache (train splits only)."""
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            return 0
        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
            total = 0
            splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            for s in splits:
                if s == "test":
                    continue
                split_ds = ds[s] if hasattr(ds, "keys") else ds
                total += len(split_ds)
            return total
        except Exception:  # intentional broad catch: HF datasets library
            return 0


class InvoicesDonutLoader(BaseDatasetLoader):
    """Download Invoices-DONUT (katanaml-org/invoices-donut-data-v1) from
    HuggingFace and normalize to SROIE schema."""

    name = "Invoices-DONUT"

    # ── download ──────────────────────────────────────────────────────

    @property
    def _subdir_name(self) -> str:
        return "invoices_donut"

    def _download(self) -> Path:
        """Download Invoices-DONUT from the HuggingFace datasets hub."""
        dest = self._dest_dir()
        marker = self._marker()
        if marker.exists():
            if (
                not self._hf_cache().exists()
                and not (dest / "hf_cache_inline" / ".done_train").exists()
            ):
                self._warn("Cache marker present but cache missing — re-downloading.")
                marker.unlink(missing_ok=True)
            else:
                return dest

        # Pin to a stable tag/commit to prevent schema drift between runs.
        # Verified schema (ground_truth JSON field) on 2026-03-07 reference run.
        # Override via env var HF_INVOICES_REVISION if upstream moves.
        _inv_repo = "katanaml-org/invoices-donut-data-v1"
        _inv_rev = os.environ.get("HF_INVOICES_REVISION", None)  # None → latest main

        hf_token = _read_hf_token()
        try:
            from datasets import load_dataset  # type: ignore

            self._log(f"Downloading {_inv_repo} from HuggingFace ...")
            _kw: dict = {"trust_remote_code": False}
            if _inv_rev:
                _kw["revision"] = _inv_rev
            ds = load_dataset(_inv_repo, token=hf_token or None, **_kw)
            ds.save_to_disk(str(self._hf_cache()))
            marker.touch()
            self._log("Download complete.")
        except ImportError:
            self._log("datasets package absent — using inline HF downloader for Invoices-DONUT ...")
            try:
                _hf_download_dataset_inline(_inv_repo, dest, hf_token, split="train")
                marker.touch()
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                raise self._fatal(
                    f"Inline download failed for {_inv_repo}: {exc}\n"
                    "  Tip: set HF_INVOICES_REVISION=<commit> or place hf_token.txt for auth."
                ) from exc
        except Exception as exc:  # intentional broad catch: HF datasets library
            raise self._fatal(
                f"Download failed for {_inv_repo}: {exc}\n"
                "  If the dataset schema changed, set HF_INVOICES_REVISION=<known-good-commit>.\n"
                "  Known-good schemas: ground_truth field must be a JSON string with keys\n"
                "  'gt_parse' → {{company, date, address, total}}."
            ) from exc
        return dest

    # ── Invoices-DONUT → SROIE remapping ─────────────────────────────

    @staticmethod
    def _invoices_donut_remap(ground_truth_str: Any) -> dict[str, str]:
        """Parse Invoices-DONUT ground_truth JSON and remap to SROIE schema.

        Field mapping rationale:
        - company: seller name from gt_parse.header.seller
        - date:    invoice date from gt_parse.header.invoice_date
        - address: parsed from gt_parse.header.seller via ML/cache lookup
        - total:   total_gross_worth with currency symbol stripped
        """
        # Phase 1: Use shared consolidation utility
        gt = BaseDatasetLoader._empty_gt_dict()
        try:
            obj = (
                json.loads(ground_truth_str)
                if isinstance(ground_truth_str, str)
                else ground_truth_str
            )
            gt_parse = obj.get("gt_parse", obj)

            header = gt_parse.get("header", {})
            if isinstance(header, dict):
                seller_raw = str(header.get("seller", "")).strip()
                cache = _load_seller_split_cache()
                if seller_raw in cache:
                    entry = cache[seller_raw]
                    company_name = entry.get("company", "")
                    address = entry.get("address", "")
                else:
                    company_name, address = extract_address_from_seller(seller_raw)
                gt["company"] = company_name
                gt["date"] = str(header.get("invoice_date", "")).strip()
                gt["address"] = address

            summary = gt_parse.get("summary", {})
            if isinstance(summary, dict):
                raw_total = str(summary.get("total_gross_worth", "")).strip()
                # Strip leading currency symbols
                gt["total"] = re.sub(r"^[\$€£¥₹₩\u20ac\u00a3\u00a5]+", "", raw_total).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
        return gt

    # ── public interface ──────────────────────────────────────────────

    def load(self, split: str = "train") -> list[Sample]:
        """Load Invoices-DONUT and normalize to SROIE schema.

        Uses HF Arrow cache (datasets library) when available; falls back to
        the inline JSONL cache written by _hf_download_dataset_inline().
        """
        dest = self._download()
        samples: list[Sample] = []

        # ── HF Arrow path (preferred) ─────────────────────────────────────────
        hf_cache = self._hf_cache()
        _use_inline = not hf_cache.exists()
        if not _use_inline:
            try:
                from datasets import load_from_disk  # type: ignore

                ds = load_from_disk(str(hf_cache))
                ds_splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
                img_dest_dir = _ensure_dir(dest / "images")
                for ds_split in ds_splits:
                    if ds_split == "test":
                        continue
                    if split != "train" and ds_split != split:
                        continue
                    split_ds = ds[ds_split] if hasattr(ds, "keys") else ds
                    for idx, item in enumerate(split_ds):
                        gt = self._invoices_donut_remap(item.get("ground_truth", "{}"))
                        pil_image = item.get("image")
                        if pil_image is not None:
                            img_path = img_dest_dir / f"{ds_split}_{idx:06d}.jpg"
                            if not img_path.exists():
                                pil_image.convert("RGB").save(img_path, "JPEG")
                            samples.append((img_path, gt))
                _mm.release_hf_dataset(ds)
                _mm.flush_hf_arrow_cache()
            except ImportError:
                _use_inline = True
            except Exception as exc:  # intentional broad catch: HF datasets library
                raise self._fatal(f"Failed to load Arrow cache: {exc}") from exc

        # ── Inline JSONL fallback ─────────────────────────────────────────────
        if _use_inline:
            inline_cache = dest / "hf_cache_inline"
            if not inline_cache.exists():
                raise self._fatal(
                    "Neither HF Arrow cache nor inline cache found. "
                    "Install the `datasets` package or run _hf_download_dataset_inline()."
                )
            for row in _hf_load_jsonl_rows(inline_cache, split="train"):
                if split == "test":
                    continue
                gt = self._invoices_donut_remap(row.get("ground_truth", "{}"))
                img_val = row.get("image")
                if img_val is not None:
                    img_path = Path(str(img_val))
                    if img_path.exists():
                        samples.append((img_path, gt))

        _log_field_coverage(samples, self.name)
        return _validate_samples_nonempty(samples, self.name)

    def validate_cache(self) -> bool:
        """Check that hf_cache exists and first record has gt_parse schema."""
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            return False
        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
            splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            first_split = ds[splits[0]] if hasattr(ds, "keys") else ds
            if len(first_split) == 0:
                return False
            item = first_split[0]
            gt_str = item.get("ground_truth", "")
            if not gt_str:
                return False
            obj = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
            return "gt_parse" in obj
        except Exception:  # intentional broad catch: HF datasets library
            return False

    def clear_cache(self) -> None:
        """Remove the entire Invoices-DONUT cache directory."""
        import shutil

        dest = self._dest_dir()
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
            self._log("Cache cleared.")

    def sample_count(self, split: str = "train") -> int:
        """Return approximate sample count from the HF cache."""
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            return 0
        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
            total = 0
            splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            for s in splits:
                if s == "test":
                    continue
                split_ds = ds[s] if hasattr(ds, "keys") else ds
                total += len(split_ds)
            return total
        except Exception:  # intentional broad catch: HF datasets library
            return 0


# ======================================================================
#  CORDv2Loader — naver-clova-ix/cord-v2 from HuggingFace
# ======================================================================


class CORDv2Loader(BaseDatasetLoader):
    """Download CORD-v2 (naver-clova-ix/cord-v2) from HuggingFace and
    normalize to SROIE schema.

    CORD-v2 contains ~1,000 Indonesian receipts with rich NER annotations.
    Field mapping to SROIE schema:

    - company: ``nm`` field in ``menu``/``store_info.store_name``
    - date:    ``cnt`` in ``payment.date`` or closest date field
    - address: ``store_info.store_addr``
    - total:   ``total.total_price`` or ``subtotal.subtotal_price``

    Coverage note: CORD-v2 has very good company/date/total coverage (>90%).
    Address coverage is lower (~60%) because many receipts omit the address.
    The DatasetNormalizer coverage threshold is set to 0.30 to accommodate this.
    """

    name = "CORD-v2"

    # ── internal paths ────────────────────────────────────────────────

    @property
    def _subdir_name(self) -> str:
        return "cord_v2"

    def _download(self) -> Path:
        dest = self._dest_dir()
        marker = self._marker()
        if marker.exists():
            if (
                not self._hf_cache().exists()
                and not (dest / "hf_cache_inline" / ".done_train").exists()
            ):
                self._warn("Marker present but cache missing — re-downloading.")
                marker.unlink(missing_ok=True)
            else:
                return dest

        _cord_repo = "naver-clova-ix/cord-v2"
        _cord_rev = os.environ.get("HF_CORD_REVISION", None)  # None → latest main

        hf_token = _read_hf_token()
        try:
            from datasets import load_dataset  # type: ignore

            self._log(f"Downloading {_cord_repo} from HuggingFace ...")
            _kw: dict = {"trust_remote_code": False}
            if _cord_rev:
                _kw["revision"] = _cord_rev
            ds = load_dataset(_cord_repo, token=hf_token or None, **_kw)
            ds.save_to_disk(str(self._hf_cache()))
            marker.touch()
            self._log("CORD-v2 download complete.")
        except ImportError:
            self._log("datasets package absent — using inline HF downloader for CORD-v2 ...")
            try:
                _hf_download_dataset_inline(_cord_repo, dest, hf_token, split="train")
                marker.touch()
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                raise self._fatal(
                    f"Inline download failed for {_cord_repo}: {exc}\n"
                    "  Tip: set HF_CORD_REVISION=<commit> or place hf_token.txt for auth."
                ) from exc
        except Exception as exc:  # intentional broad catch: HF datasets library
            raise self._fatal(
                f"CORD-v2 download failed for {_cord_repo}: {exc}\n"
                "  Override via HF_CORD_REVISION=<commit> to pin a known-good version."
            ) from exc
        return dest

    # ── CORD-v2 → SROIE remapping ─────────────────────────────────────

    @staticmethod
    def _cord_extract_company(gt_parsed: dict[str, Any]) -> str:
        """Extract company name from CORD-v2 parsed ground truth.

        Looks in ``store_info.store_name`` first, then falls back to the
        first ``nm`` (name) field from menu items.
        """
        store_info = gt_parsed.get("store_info", {})
        if isinstance(store_info, dict):
            company = str(store_info.get("store_name", "")).strip()
            if company:
                return company

        # Fallback: first nm (name) from menu items
        menu = gt_parsed.get("menu", [])
        if isinstance(menu, list) and menu:
            first_item = menu[0]
            if isinstance(first_item, dict):
                return str(first_item.get("nm", "")).strip()
        return ""

    @staticmethod
    def _cord_extract_date(gt_parsed: dict[str, Any]) -> str:
        """Extract date from CORD-v2 parsed ground truth.

        Searches nested paths for a date pattern, then falls back to
        dedicated date fields in ``payment``.
        """
        _date_re = re.compile(
            r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"
            r"|\b\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}\b"
        )
        # CORD-v2 stores date in various locations
        for date_path in [
            ("store_info", "store_addr"),  # sometimes combined
            ("payment", "date"),
            ("subtotal", "cnt"),
        ]:
            node = gt_parsed
            for key in date_path:
                if isinstance(node, dict):
                    node = node.get(key, {})
            if isinstance(node, str) and node.strip():
                m = _date_re.search(node)
                if m:
                    return m.group()

        # Dedicated date field search
        payment = gt_parsed.get("payment", {})
        if isinstance(payment, dict):
            for k in ("date", "date_time", "receipt_date"):
                val = str(payment.get(k, "")).strip()
                if val:
                    return val
        return ""

    @staticmethod
    def _cord_extract_address(gt_parsed: dict[str, Any], company: str) -> tuple[str, str]:
        """Extract address from CORD-v2 parsed ground truth.

        Returns ``(company_override, address)`` — when the store address
        contains a concatenated company+address and ``company`` is empty,
        :func:`extract_address_from_seller` splits them.
        """
        store_info = gt_parsed.get("store_info", {})
        if isinstance(store_info, dict):
            addr = str(store_info.get("store_addr", "")).strip()
            # Sometimes company and address are concatenated in store_addr
            if addr and not company:
                return extract_address_from_seller(addr)
            return "", addr
        return "", ""

    @staticmethod
    def _cord_extract_total(gt_parsed: dict[str, Any]) -> str:
        """Extract total from CORD-v2 parsed ground truth.

        Checks ``total.total_price`` and related keys first, then falls
        back to ``subtotal.subtotal_price``.
        """
        _currency_re = r"^[\$€£¥₹₩\u20ac\u00a3\u00a5]+"
        total_node = gt_parsed.get("total", {})
        if isinstance(total_node, dict):
            for total_key in ("total_price", "creditcardprice", "cashprice", "emoneyprice"):
                val = str(total_node.get(total_key, "")).strip()
                if val:
                    return re.sub(_currency_re, "", val).strip()

        # Fallback: subtotal_price
        subtotal = gt_parsed.get("subtotal", {})
        if isinstance(subtotal, dict):
            val = str(subtotal.get("subtotal_price", "")).strip()
            if val:
                return re.sub(_currency_re, "", val).strip()
        return ""

    @staticmethod
    def _cord_remap(ground_truth_str: Any) -> dict[str, str]:
        """Parse CORD-v2 ground_truth JSON and remap to SROIE schema.

        CORD-v2 ground_truth structure (top-level keys):
          - ``gt_parse``: dict with ``menu``, ``subtotal``, ``total``
          - ``valid_line``: list of text lines (not used here)
        """
        gt = BaseDatasetLoader._empty_gt_dict()
        try:
            obj = (
                json.loads(ground_truth_str)
                if isinstance(ground_truth_str, str)
                else ground_truth_str
            )
            gt_parsed = obj.get("gt_parse", obj)

            gt["company"] = CORDv2Loader._cord_extract_company(gt_parsed)
            gt["date"] = CORDv2Loader._cord_extract_date(gt_parsed)
            company_override, gt["address"] = CORDv2Loader._cord_extract_address(
                gt_parsed, gt["company"]
            )
            if company_override and not gt["company"]:
                gt["company"] = company_override
            gt["total"] = CORDv2Loader._cord_extract_total(gt_parsed)

        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return gt

    # ── public interface ──────────────────────────────────────────────

    def load(self, split: str = "train") -> list[Sample]:
        """Load CORD-v2 and normalize to SROIE schema."""
        dest = self._download()
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            raise self._fatal("hf_cache/ not found after download.")

        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
        except Exception as exc:  # intentional broad catch: HF datasets library
            raise self._fatal(f"Failed to load CORD-v2 cache: {exc}") from exc

        samples: list[Sample] = []
        splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]

        for ds_split in splits:
            if ds_split == "test":
                continue  # skip test split to avoid contamination

            if split != "train" and ds_split != split:
                continue

            split_ds = ds[ds_split] if hasattr(ds, "keys") else ds
            for idx, item in enumerate(split_ds):
                gt = self._cord_remap(item.get("ground_truth", "{}"))
                pil_image = item.get("image")
                if pil_image is not None:
                    img_dest_dir = _ensure_dir(dest / "images")
                    img_path = img_dest_dir / f"{ds_split}_{idx:06d}.jpg"
                    if not img_path.exists():
                        pil_image.convert("RGB").save(img_path, "JPEG")
                    samples.append((img_path, gt))

        _log_field_coverage(samples, self.name)
        _mm.release_hf_dataset(ds)
        _mm.flush_hf_arrow_cache()
        return _validate_samples_nonempty(samples, self.name)

    def validate_cache(self) -> bool:
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            return False
        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
            splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            first_split = ds[splits[0]] if hasattr(ds, "keys") else ds
            if len(first_split) == 0:
                return False
            item = first_split[0]
            gt_str = item.get("ground_truth", "")
            if not gt_str:
                return False
            obj = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
            return "gt_parse" in obj
        except Exception:  # intentional broad catch: HF datasets library
            return False

    def clear_cache(self) -> None:
        import shutil

        dest = self._dest_dir()
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
            self._log("CORD-v2 cache cleared.")

    def sample_count(self, split: str = "train") -> int:
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            return 0
        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
            total = 0
            splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
            for s in splits:
                if s == "test":
                    continue
                split_ds = ds[s] if hasattr(ds, "keys") else ds
                total += len(split_ds)
            return total
        except Exception:  # intentional broad catch: HF datasets library
            return 0


# ======================================================================
#  Singleton loader instances (lazy)
# ======================================================================

_sroie_loader: SROIELoader | None = None
_wildreceipt_loader: WildReceiptLoader | None = None
_funsd_loader: FUNSDLoader | None = None
_invoices_donut_loader: InvoicesDonutLoader | None = None
_cord_v2_loader: CORDv2Loader | None = None


def _get_sroie_loader() -> SROIELoader:
    global _sroie_loader
    if _sroie_loader is None:
        _sroie_loader = SROIELoader()
    return _sroie_loader


def _get_wildreceipt_loader() -> WildReceiptLoader:
    global _wildreceipt_loader
    if _wildreceipt_loader is None:
        _wildreceipt_loader = WildReceiptLoader()
    return _wildreceipt_loader


def _get_funsd_loader() -> FUNSDLoader:
    global _funsd_loader
    if _funsd_loader is None:
        _funsd_loader = FUNSDLoader()
    return _funsd_loader


def _get_invoices_donut_loader() -> InvoicesDonutLoader:
    global _invoices_donut_loader
    if _invoices_donut_loader is None:
        _invoices_donut_loader = InvoicesDonutLoader()
    return _invoices_donut_loader


def _get_cord_v2_loader() -> CORDv2Loader:
    global _cord_v2_loader
    if _cord_v2_loader is None:
        _cord_v2_loader = CORDv2Loader()
    return _cord_v2_loader


# ======================================================================
#  Backward-compatible module-level functions
# ======================================================================
# These wrap the OOP loaders so that existing call-sites (run_all.py,
# run_experiments.py, etc.) continue to work without modification.


def load_sroie_train() -> list[Sample]:
    """Load SROIE training split — compatibility wrapper for SROIELoader."""
    return _get_sroie_loader().load("train")


def load_sroie_test() -> list[Sample]:
    """Load SROIE test split — compatibility wrapper for SROIELoader.

    Returns an empty list with a warning if test_img/ is absent (matching
    legacy behaviour expected by callers).
    """
    return _get_sroie_loader().load("test")


def load_sroie_val() -> list[Sample]:
    """Load SROIE validation split — compatibility wrapper for SROIELoader.

    Returns an empty list with a warning if val_img/ is absent.
    """
    return _get_sroie_loader().load("val")


def load_wildreceipt() -> list[Sample]:
    """Load WildReceipt — compatibility wrapper for WildReceiptLoader."""
    return _get_wildreceipt_loader().load("train")


def load_funsd() -> list[Sample]:
    """Load FUNSD — compatibility wrapper for FUNSDLoader."""
    return _get_funsd_loader().load("train")


def load_invoices_donut() -> list[Sample]:
    """Load Invoices-DONUT — compatibility wrapper for InvoicesDonutLoader."""
    return _get_invoices_donut_loader().load("train")


def load_cord_v2() -> list[Sample]:
    """Load CORD-v2 — compatibility wrapper for CORDv2Loader."""
    return _get_cord_v2_loader().load("train")


# ======================================================================
#  Combined dataset loader
# ======================================================================

_LOADERS = {
    "sroie": load_sroie_train,
    "wildreceipt": load_wildreceipt,
    "funsd": load_funsd,
    "invoices_donut": load_invoices_donut,
    "cord_v2": load_cord_v2,
}


def get_dataset_loader(name: str):
    """Return the load function for *name*, or ``None`` if not registered.

    This is the public API for looking up a dataset loader by name.
    Prefer this over accessing the private ``_LOADERS`` dict directly.

    Parameters
    ----------
    name : str
        One of: ``"sroie"``, ``"wildreceipt"``, ``"funsd"``,
        ``"invoices_donut"``, ``"cord_v2"``.

    Returns
    -------
    callable or None
    """
    return _LOADERS.get(name)


def split_dataset(
    samples: list[Sample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Split a dataset into train/val/test with a fixed seed.

    Parameters
    ----------
    samples : list of (Path, dict) tuples
    train_ratio, val_ratio, test_ratio : floats summing to 1.0
    seed : int — random seed for reproducibility

    Returns
    -------
    (train, val, test) lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"train_ratio + val_ratio + test_ratio must sum to 1.0, got "
        f"{train_ratio + val_ratio + test_ratio}"
    )
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def _apply_oversampling(
    train_samples: list[Sample],
    sroie_data: list[Sample],
    sroie_oversample: int,
) -> tuple[list[Sample], list[str]]:
    """Apply SROIE oversampling and return extended train samples with source tags.

    SROIE training samples are duplicated *sroie_oversample* times (minimum 1)
    and appended to *train_samples*.  List multiplication creates N references
    to the same immutable Sample tuples — safe and memory-efficient for
    read-only iteration.

    Parameters
    ----------
    train_samples : list[Sample]
        Accumulator list; SROIE samples are appended in-place.
    sroie_data : list[Sample]
        Raw SROIE training samples (not yet oversampled).
    sroie_oversample : int
        Number of times to duplicate SROIE training samples.

    Returns
    -------
    (extended_train, source_tags) where *source_tags* has one ``"sroie"``
    entry per added sample.
    """
    oversample_count = max(1, sroie_oversample)
    oversampled = sroie_data * oversample_count
    train_samples.extend(oversampled)
    return train_samples, ["sroie"] * len(oversampled)


def _split_auxiliary_data(
    aux_samples: list[Sample],
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    """Split auxiliary dataset samples into train and validation portions.

    Uses a 70/15/15 train/val/test split via :func:`split_dataset`.  The
    held-out 15 % test portion is discarded (SROIE test set is used for
    final evaluation).

    Returns
    -------
    (train_split, val_split)
    """
    train_split, val_split, _ = split_dataset(aux_samples, seed=seed)
    return train_split, val_split


def _load_and_merge_datasets(
    dataset_names: list[str],
    sroie_oversample: int,
) -> tuple[list[Sample], list[Sample], list[str], dict[str, int]]:
    """Load all requested datasets and merge into combined train/val lists.

    Iterates over *dataset_names*, loading each via ``_LOADERS``.  SROIE
    samples are oversampled via :func:`_apply_oversampling`; auxiliary
    datasets are split 70/15/15 via :func:`_split_auxiliary_data`.

    Returns
    -------
    (combined_train, combined_val, combined_train_sources, per_loader_counts)
    """
    combined_train: list[Sample] = []
    combined_val: list[Sample] = []
    combined_train_sources: list[str] = []
    per_loader_counts: dict[str, int] = {}

    for name in dataset_names:
        loader = _LOADERS.get(name)
        if loader is None:
            raise ValueError(f"Unknown dataset '{name}'. Valid: {list(_LOADERS)}")
        try:
            data = loader()
        except DatasetLoadError:
            # Re-raise — the error has already been printed to stderr
            raise
        per_loader_counts[name] = len(data)

        if name == "sroie":
            # SROIE: train split → combined_train (optionally oversampled);
            # val split → combined_val (never oversampled).
            _, sroie_sources = _apply_oversampling(combined_train, data, sroie_oversample)
            combined_train_sources.extend(sroie_sources)
            sroie_val = load_sroie_val()
            combined_val.extend(sroie_val)
        else:
            # Auxiliary datasets: 70/15/15 split
            train_split, val_split = _split_auxiliary_data(data, seed=SEED)
            combined_train.extend(train_split)
            combined_train_sources.extend([name] * len(train_split))
            combined_val.extend(val_split)

    return combined_train, combined_val, combined_train_sources, per_loader_counts


def _shuffle_with_sources(
    samples: list[Sample],
    sources: list[str],
    val_samples: list[Sample],
    seed: int,
    return_sources: bool = False,
) -> None:
    """Shuffle *samples* (and optionally *sources*) in place with a fixed seed.

    When *return_sources* is ``True``, samples and sources are shuffled as
    aligned pairs so that ``sources[i]`` still corresponds to ``samples[i]``
    after the shuffle.  *val_samples* is shuffled afterwards using a
    continuation of the same RNG to preserve the original interleaving
    behaviour.
    """
    rng = random.Random(seed)
    if return_sources:
        # Shuffle (sample, source) pairs together to keep alignment.
        paired = list(zip(samples, sources))
        rng.shuffle(paired)
        samples[:] = [p[0] for p in paired]
        sources[:] = [p[1] for p in paired]
    else:
        rng.shuffle(samples)
    # Val shuffle uses a continuation of the same RNG (preserves original behavior).
    rng.shuffle(val_samples)


@overload
def get_combined_dataset(
    dataset_names: list[str],
    sroie_oversample: int = 1,
    return_sources: Literal[False] = False,
) -> tuple[list[Sample], list[Sample]]: ...


@overload
def get_combined_dataset(
    dataset_names: list[str],
    sroie_oversample: int = 1,
    return_sources: Literal[True] = ...,
) -> tuple[list[Sample], list[Sample], list[str]]: ...


def get_combined_dataset(
    dataset_names: list[str],
    sroie_oversample: int = 1,
    return_sources: bool = False,
) -> tuple[list[Sample], list[Sample]] | tuple[list[Sample], list[Sample], list[str]]:
    """Merge multiple datasets into train and validation lists.

    SROIE training samples are added to train as-is (train split from img/).
    SROIE validation samples (from val_img/) are added to the combined val.
    All auxiliary datasets (WildReceipt, Invoices-DONUT) are split
    70/15/15; the 70% goes into combined train, the 15% validation portion
    goes into combined val, and the held-out 15% test portion is discarded.

    After merging, both combined_train and combined_val are shuffled with
    a fixed seed so that samples from different datasets are interleaved.

    Delegates to:
    - :func:`_load_and_merge_datasets` — per-dataset loading and merging
    - :func:`_apply_oversampling` — SROIE duplication
    - :func:`_split_auxiliary_data` — 70/15/15 auxiliary splitting
    - :func:`_shuffle_with_sources` — paired shuffle for source alignment

    Parameters
    ----------
    dataset_names : list of str
        Names of datasets to include.  Valid names: sroie, wildreceipt,
        funsd, invoices_donut, cord_v2.
    sroie_oversample : int, optional
        Number of times to duplicate SROIE training samples (default 1).
        Use 2 or 3 to counteract SROIE field dilution when combining with
        large auxiliary datasets.
    return_sources : bool, optional
        When True, returns a third element: a list of dataset-source strings
        (e.g. "sroie", "wildreceipt") aligned with train_samples.  Used by
        the aux_loss_weight feature in MultiDataset to scale auxiliary losses.
        Default False preserves the original two-tuple return type.

    Returns
    -------
    (train_samples, val_samples) : Tuple[List[Sample], List[Sample]]
        When return_sources is False (default).
    (train_samples, val_samples, train_sources) : Tuple[...] | Tuple[..., List[str]]
        When return_sources is True.  train_sources[i] is the dataset name
        for train_samples[i] (e.g. "sroie" or "wildreceipt").
    """
    combined_train, combined_val, combined_train_sources, _ = _load_and_merge_datasets(
        dataset_names, sroie_oversample
    )

    # Fixed-seed shuffle to interleave samples from different datasets.
    _shuffle_with_sources(
        combined_train, combined_train_sources, combined_val, SEED, return_sources
    )

    if return_sources:
        return combined_train, combined_val, combined_train_sources
    return combined_train, combined_val


# ── preprocess_seller_split ──────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
CACHE_PATH = REPO_ROOT / "seller_split_cache.json"

# ---------------------------------------------------------------------------
# Compiled heuristics (fallback when NER confidence is low)
# ---------------------------------------------------------------------------
# NOTE: _STREET_NUMBER_RE, _PO_BOX_RE, _STATE_ZIP_RE, _STREET_SUFFIX_RE
# are already defined at module level above (in the address-parsing regex
# section). Reuse those definitions — do not redefine here.


def _heuristic_split(seller: str) -> tuple[str, str]:
    """Split seller string into (company, address) using regex heuristics.

    Looks for the earliest address indicator (street number, P.O. Box,
    street suffix, or state+ZIP) that appears *after* the first character.
    Returns (full_string, "") when no address pattern is found.
    """
    seller = seller.strip()
    if not seller:
        return ("", "")

    candidates: list[int] = []
    for pattern in (_PO_BOX_RE, _STREET_NUMBER_RE, _STREET_SUFFIX_RE, _STATE_ZIP_RE):
        m = pattern.search(seller)
        if m and m.start() > 0:
            candidates.append(m.start())

    if not candidates:
        return (seller, "")

    split_at = min(candidates)
    company = seller[:split_at].strip().rstrip(",").strip()
    address = seller[split_at:].strip()
    return (company, address)


# ---------------------------------------------------------------------------
# spaCy-based split
# ---------------------------------------------------------------------------


def _spacy_split(seller: str, nlp) -> tuple[str, str] | None:
    """Use spaCy NER to split seller into (company, address).

    Returns None when NER cannot provide a confident split (no ORG/PERSON
    entities found), so callers can fall back to heuristics.

    Strategy:
    - Find the last ORG or PERSON entity span end as the company boundary.
    - Everything from that boundary onward is the address portion.
    - If no address-type entity (GPE/LOC/FAC) is found after the company
      boundary, fall back to heuristic on the remaining substring.
    """
    doc = nlp(seller)

    # Find the rightmost end of any ORG or PERSON entity.
    company_end: int | None = None
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PERSON") and (company_end is None or ent.end_char > company_end):
            company_end = ent.end_char

    if company_end is None:
        # No clear company entity — cannot split with confidence.
        return None

    company_candidate = seller[:company_end].strip().rstrip(",").strip()
    remainder = seller[company_end:].strip().lstrip(",").strip()

    if not remainder:
        # Entire string is company; check heuristics on full string.
        _, addr = _heuristic_split(seller)
        if addr:
            split_at = seller.find(addr)
            company_candidate = seller[:split_at].strip().rstrip(",").strip()
            return (company_candidate, addr)
        return (seller.strip(), "")

    # Verify the remainder looks like an address via heuristics.
    _, addr_check = _heuristic_split(remainder)
    if addr_check or _STATE_ZIP_RE.search(remainder):
        return (company_candidate, remainder)

    # Remainder doesn't look like an address; use heuristic on full string.
    heur = _heuristic_split(seller)
    return heur


# ---------------------------------------------------------------------------
# LLM fallback (optional)
# ---------------------------------------------------------------------------


def _llm_split(seller: str) -> tuple[str, str] | None:
    """Try OpenAI or Anthropic API to split a hard seller string.

    Returns None when no API key is available or the API call fails.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    prompt = (
        f"Split this seller string into company name and street address.\n"
        f'Input: "{seller}"\n'
        f'Output JSON only (no commentary): {{"company": "...", "address": "..."}}'
    )

    if openai_key:
        try:
            import openai  # type: ignore

            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )
            text = response.choices[0].message.content.strip()
            obj = json.loads(text)
            return (str(obj.get("company", "")), str(obj.get("address", "")))
        except Exception as exc:  # intentional broad catch: third-party OpenAI SDK
            log.debug("OpenAI fallback failed for %r: %s", seller, exc)

    if anthropic_key:
        try:
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            obj = json.loads(text)
            return (str(obj.get("company", "")), str(obj.get("address", "")))
        except Exception as exc:  # intentional broad catch: third-party Anthropic SDK
            log.debug("Anthropic fallback failed for %r: %s", seller, exc)

    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_seller_strings() -> tuple[list[str], str]:
    """Load all seller strings from the HuggingFace dataset.

    Returns (seller_strings, source_description).
    Falls back to a built-in representative list if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset  # type: ignore

        log.info("Loading katanaml-org/invoices-donut-data-v1 from HuggingFace …")
        ds = load_dataset("katanaml-org/invoices-donut-data-v1")
        sellers: list[str] = []
        for split_name, split_ds in ds.items():
            if split_name == "test":
                continue
            for item in split_ds:
                gt_str = item.get("ground_truth", "{}")
                try:
                    obj = json.loads(gt_str) if isinstance(gt_str, str) else gt_str
                    gt_parse = obj.get("gt_parse", obj)
                    header = gt_parse.get("header", {})
                    if isinstance(header, dict):
                        seller = str(header.get("seller", "")).strip()
                        if seller:
                            sellers.append(seller)
                except (json.JSONDecodeError, AttributeError):
                    pass
        log.info("Loaded %d seller strings from HuggingFace.", len(sellers))
        return sellers, "katanaml-org/invoices-donut-data-v1 (HuggingFace)"

    except Exception as exc:  # intentional broad catch: HF datasets library
        log.warning(
            "HuggingFace dataset unavailable (%s). "
            "Falling back to representative built-in examples.",
            exc,
        )

    # Built-in representative examples drawn from the dataset's documented
    # distribution (used when HuggingFace is unreachable in the sandbox).
    builtin = [
        "Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228",
        "Smith & Associates LLC 1200 Oak Avenue Suite 400 Chicago, IL 60601",
        "Johnson Industrial Corp P.O. Box 4521 Houston, TX 77001",
        "Williams Group 888 Maple Street Portland, OR 97201",
        "Brown Consulting Ltd 42 Elm Road Austin, TX 78701",
        "Davis & Partners 3500 North Blvd Denver, CO 80201",
        "Martinez Enterprises 500 Commerce Drive Miami, FL 33101",
        "Anderson Technology Inc 720 Tech Parkway Seattle, WA 98101",
        "Thomas Manufacturing 2100 Industrial Way Detroit, MI 48201",
        "Jackson & Sons 99 River Lane Nashville, TN 37201",
        "White Solutions 1 Corporate Court Boston, MA 02101",
        "Harris Financial Group 555 Wall Street New York, NY 10005",
        "Clark Logistics 3000 Airport Blvd Los Angeles, CA 90001",
        "Lewis Brothers 601 Oak Terrace Atlanta, GA 30301",
        "Robinson Retail Inc 250 Mall Circle Phoenix, AZ 85001",
        "Walker Industries 10 Harbor Drive San Francisco, CA 94101",
        "Hall Healthcare 800 Medical Parkway Dallas, TX 75201",
        "Allen & Baker 404 Research Blvd Philadelphia, PA 19101",
        "Young Law Group 77 Justice Lane Minneapolis, MN 55401",
        "Hernandez Construction 3100 Highway 1 San Antonio, TX 78201",
    ]
    return builtin, "built-in representative examples (HuggingFace unavailable)"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def build_cache(
    sellers: list[str],
    nlp,
    use_llm: bool = True,
) -> dict[str, dict[str, str]]:
    """Process a list of seller strings and return the split cache dict."""
    cache: dict[str, dict[str, str]] = {}
    n_spacy = n_heuristic = n_llm = 0

    unique_sellers = list(dict.fromkeys(sellers))  # deduplicate, preserve order
    log.info("Processing %d unique seller strings …", len(unique_sellers))

    for seller in unique_sellers:
        result = _spacy_split(seller, nlp)
        method = "spacy"

        if result is None and use_llm:
            result = _llm_split(seller)
            if result is not None:
                method = "llm"

        if result is None:
            result = _heuristic_split(seller)
            method = "heuristic"

        company, address = result
        cache[seller] = {"company": company, "address": address}

        if method == "spacy":
            n_spacy += 1
        elif method == "llm":
            n_llm += 1
        else:
            n_heuristic += 1

    log.info(
        "Split breakdown — spaCy: %d, heuristic: %d, LLM: %d",
        n_spacy,
        n_heuristic,
        n_llm,
    )
    return cache


def main() -> int:
    # ── Load spaCy model ──────────────────────────────────────────────────
    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("en_core_web_sm")
            log.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            log.warning("en_core_web_sm not found; downloading …")
            import subprocess

            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                stdout=subprocess.DEVNULL,
            )
            nlp = spacy.load("en_core_web_sm")
    except ImportError:
        log.error("spaCy is not installed. Run: pip install spacy")
        return 1

    # ── Load seller strings ───────────────────────────────────────────────
    sellers, source = _load_seller_strings()

    # ── Build cache ───────────────────────────────────────────────────────
    cache = build_cache(sellers, nlp)

    # ── Write output ──────────────────────────────────────────────────────
    output: dict = {
        "_metadata": {
            "source": source,
            "num_entries": len(cache),
            "note": (
                "Generated by preprocess_seller_split.py using spaCy NER "
                "(en_core_web_sm) with heuristic fallback. "
                "Re-run the script with internet access to regenerate from "
                "the full HuggingFace dataset."
            ),
        }
    }
    output.update(cache)

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log.info("Wrote %d entries to %s", len(cache), CACHE_PATH)
    return 0


# ── dataset_preparation (YOLO + TrOCR data prep) ──────────────────────────

try:
    from PIL import Image as _PrepImage
except ImportError:
    _PrepImage = None  # type: ignore[assignment]

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
YOLO_DIR = DATA_DIR / "yolo"
TROCR_DIR = DATA_DIR / "trocr"

# FIX: Use _get_sroie_dir() at call time (not module-load time) so that
# env vars set by run_all.py after import are respected.  The property-like
# helper is used everywhere instead of the old module-level constant.


def _sroie_data_dir() -> Path:
    """Return SROIE data root, re-reading the env var each call."""
    return _get_sroie_dir()


# Map split names to (img_subdir, key_subdir) in the SROIE tree
SPLIT_MAP = SROIELoader._SPLIT_DIRS  # single source of truth

# Candidate box/ subdirectory names per split (tried in order).
# The SROIE dataset ships one shared "box/" for train; some re-packagings
# use split-specific names.  We probe all candidates.
BOX_DIR_CANDIDATES: dict[str, list[str]] = {
    "train": ["box", "box_train", "train_box"],
    "val": ["box_val", "val_box", "box"],
    "test": ["box_test", "test_box", "box"],
}

# Extensions to probe when resolving OCR box annotation files, in priority order.
# Standard SROIE ships .txt; some distributions use .csv with the identical format.
_BOX_EXTENSIONS = [".txt", ".csv"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_box_dir(split: str) -> Path | None:
    """Return the first existing box annotation directory for a split, or None."""
    for candidate in BOX_DIR_CANDIDATES.get(split, []):
        p = _sroie_data_dir() / candidate
        if p.exists() and any(p.iterdir()):
            return p
    return None


# ── OCR bbox file reader ──────────────────────────────────────────────────────
def _load_ocr_bboxes(box_dir: Path, stem: str) -> list[tuple[list[int], str]]:
    """Load SROIE OCR bounding boxes from the box/ directory.

    Each line has format: x1,y1,x2,y2,x3,y3,x4,y4,text
    Returns list of ([x1,y1,x2,y2], text) tuples using the axis-aligned
    bounding box of the four corner points.

    Diagnostic notes are printed when the file is missing, when lines fail to
    parse, or when the file exists but yields zero valid boxes — helping trace
    stem mismatches and format problems without crashing the pipeline.
    """
    # Probe extensions in priority order: .txt (standard SROIE), then .csv
    box_file: Path | None = None
    for ext in _BOX_EXTENSIONS:
        candidate = box_dir / f"{stem}{ext}"
        if candidate.exists():
            box_file = candidate
            break

    if box_file is None:
        # Try case-insensitive match for both extensions
        ci_candidates = [
            p
            for p in box_dir.iterdir()
            if p.stem.lower() == stem.lower() and p.suffix.lower() in _BOX_EXTENSIONS
        ]
        if ci_candidates:
            box_file = ci_candidates[0]
            print(
                f"  [YOLO] NOTE: using case-insensitive match {box_file.name!r} for stem={stem!r}."
            )
        else:
            print(f"  [YOLO] DEBUG: no box file for stem={stem!r} in {box_dir}")
            return []

    raw_lines = box_file.read_text(encoding="utf-8", errors="replace").splitlines()
    results = []
    parse_errors = 0
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 8)
        if len(parts) < 9:
            parse_errors += 1
            continue
        try:
            coords = [int(p) for p in parts[:8]]
            text = parts[8].strip()
            xs = coords[0::2]
            ys = coords[1::2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            if text and (x2 - x1) > 0 and (y2 - y1) > 0:
                results.append(([x1, y1, x2, y2], text))
        except (ValueError, IndexError):
            parse_errors += 1
            continue

    if parse_errors:
        print(
            f"  [YOLO] WARNING: {parse_errors} line(s) in {box_file.name} failed to parse "
            f"(expected 'x1,y1,x2,y2,x3,y3,x4,y4,text' format)"
        )
    if not results and raw_lines:
        print(
            f"  [YOLO] WARNING: {box_file.name} has {len(raw_lines)} raw line(s) but "
            "yielded 0 valid boxes — check coordinate/text format"
        )
    return results


# ── Line grouping ─────────────────────────────────────────────────────────────
def group_words_into_lines(
    boxes: list[tuple[list[int], str]], row_tol: int = 12
) -> list[tuple[list[int], str]]:
    """Group word-level bboxes into line-level bboxes by y-coordinate proximity.

    Returns list of ([x1,y1,x2,y2], concatenated_text) tuples.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: (b[0][1], b[0][0]))
    lines: list[list[tuple[list[int], str]]] = [[sorted_boxes[0]]]

    for item in sorted_boxes[1:]:
        last_line = lines[-1]
        last_y = last_line[-1][0][1]
        if abs(item[0][1] - last_y) < row_tol:
            last_line.append(item)
        else:
            lines.append([item])

    result = []
    for line in lines:
        xs1 = [b[0][0] for b in line]
        ys1 = [b[0][1] for b in line]
        xs2 = [b[0][2] for b in line]
        ys2 = [b[0][3] for b in line]
        # Sort words left-to-right within a line
        sorted_words = sorted(line, key=lambda b: b[0][0])
        text = " ".join(w[1] for w in sorted_words)
        result.append(([min(xs1), min(ys1), max(xs2), max(ys2)], text))
    return result


# ── Build YOLO dataset ────────────────────────────────────────────────────────
def build_yolo_split(split: str) -> int:
    """Generate YOLO-format labels for one split. Returns sample count.

    YOLO format: one .txt per image with lines: class cx cy w h (normalized).
    Single class 0 = 'text_region'.
    """
    img_subdir, key_subdir = SPLIT_MAP[split]
    img_dir = _sroie_data_dir() / img_subdir

    if not img_dir.exists():
        print(f"  [YOLO] {split}: {img_subdir}/ not found — skipping")
        return 0

    # FIX: probe all candidate box dirs for this split
    box_dir = _find_box_dir(split)

    out_img_dir = YOLO_DIR / "images" / split
    out_lbl_dir = YOLO_DIR / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in _IMAGE_EXTS:
            continue

        img = _PrepImage.open(img_path).convert("RGB")
        W, H = img.size

        # Try to load OCR bboxes for YOLO labels
        boxes = []
        if box_dir is not None:
            raw_boxes = _load_ocr_bboxes(box_dir, img_path.stem)
            boxes = group_words_into_lines(raw_boxes)

        # FIX: When no box/ directory exists, fall back to a single full-image
        # bounding box so YOLO is not trained on pure background images.
        # This produces a degraded but functional text-region detector
        # (vs. the previous behaviour of writing empty label files and training
        # YOLO on 500 "background" images for 50 epochs to no effect).
        if not boxes and box_dir is None:
            # Load key file to confirm the image contains receipt content
            _, key_subdir_local = SPLIT_MAP[split]
            key_dir = _sroie_data_dir() / key_subdir_local
            gt = _load_key_file(key_dir, img_path.stem)
            if any(v for v in gt.values()):
                # 95% of image width/height centred — gives YOLO room to learn
                # that the entire receipt is a "text_region"
                margin = 0.025
                boxes = [
                    (
                        [
                            int(W * margin),
                            int(H * margin),
                            int(W * (1 - margin)),
                            int(H * (1 - margin)),
                        ],
                        "receipt",
                    )
                ]

        # Save image
        dest_img = out_img_dir / img_path.name
        if not dest_img.exists():
            img.save(dest_img)

        # Write YOLO label file
        lbl_path = out_lbl_dir / f"{img_path.stem}.txt"
        with open(lbl_path, "w") as f:
            for bbox, _text in boxes:
                x1, y1, x2, y2 = bbox
                cx = max(0.0, min(1.0, ((x1 + x2) / 2) / W))
                cy = max(0.0, min(1.0, ((y1 + y2) / 2) / H))
                bw = max(0.0, min(1.0, (x2 - x1) / W))
                bh = max(0.0, min(1.0, (y2 - y1) / H))
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        count += 1

    print(f"  [YOLO] {split}: {count} images -> {out_img_dir}")

    # Safety: if every label file is empty after preparation, abort with a
    # clear diagnostic rather than silently training YOLO on nothing.
    label_files = [
        out_lbl_dir / f"{p.stem}.txt"
        for p in sorted(img_dir.iterdir())
        if p.suffix.lower() in _IMAGE_EXTS and (out_lbl_dir / f"{p.stem}.txt").exists()
    ]
    total_instances = sum(1 for lbl in label_files if lbl.stat().st_size > 0)
    if count > 0 and total_instances == 0:
        raise RuntimeError(
            f"[YOLO] build_yolo_split('{split}'): {count} images processed but "
            f"every label file is empty — YOLO would train on pure background. "
            f"Check SROIE_DATA_DIR={_sroie_data_dir()} and that box/ or key/ files exist."
        )
    if box_dir is None and count > 0:
        print(
            f"  [YOLO] {split}: WARNING — no box/ directory found; "
            f"used full-image fallback boxes for {count} images. "
            f"Label quality is degraded; provide box/ annotations for best results."
        )
    return count


# ── Build TrOCR dataset ───────────────────────────────────────────────────────
def build_trocr_split(split: str) -> int:
    """Generate TrOCR line crop images + metadata for one split. Returns crop count.

    Each line crop is a horizontal slice of the receipt image containing one
    text line.  metadata.jsonl has {"file_name": ..., "text": ...} per crop.

    FIX: Previously only the train split had a box_dir, so val/test (and train
    when box/ is missing) produced 0 crops.  Now all splits fall back to
    key-file full-image crops when no box annotations are found, guaranteeing
    non-zero TrOCR training data.
    """
    img_subdir, key_subdir = SPLIT_MAP[split]
    img_dir = _sroie_data_dir() / img_subdir
    key_dir = _sroie_data_dir() / key_subdir

    if not img_dir.exists():
        print(f"  [TrOCR] {split}: {img_subdir}/ not found — skipping")
        return 0

    # FIX: probe all candidate box dirs for this split
    box_dir = _find_box_dir(split)

    out_dir = TROCR_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in _IMAGE_EXTS:
            continue

        img = _PrepImage.open(img_path).convert("RGB")
        W, H = img.size

        # Get line-level bboxes from box annotations if available
        lines: list[tuple[list[int], str]] = []
        if box_dir is not None:
            raw_boxes = _load_ocr_bboxes(box_dir, img_path.stem)
            lines = group_words_into_lines(raw_boxes)

        # FIX: Always fall back to key-file field crops when no box data.
        # This applies to: all val/test images, and any train image whose
        # per-image box file is missing.
        # Generate one entry per field (not one entry with all fields joined)
        # so each label is a short, single-value string — closer to what
        # TrOCR was pretrained on.  The full image is used as the crop in all
        # cases (we have no spatial segmentation without box annotations).
        if not lines:
            gt = _load_key_file(key_dir, img_path.stem)
            for f in FIELDS:
                val = gt.get(f, "").strip()
                if val:
                    lines.append(([0, 0, W, H], val))

        for line_idx, (bbox, text) in enumerate(lines):
            if not text.strip():
                continue
            x1, y1, x2, y2 = bbox
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(W, x2 + pad)
            y2 = min(H, y2 + pad)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = img.crop((x1, y1, x2, y2))
            crop_name = f"{img_path.stem}_{line_idx:03d}.png"
            crop.save(out_dir / crop_name)
            records.append({"file_name": crop_name, "text": text})

    # Write metadata
    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  [TrOCR] {split}: {len(records)} line crops -> {out_dir}")
    return len(records)


# ── Write YOLO dataset.yaml ───────────────────────────────────────────────────
def write_yolo_yaml() -> None:
    """Write the dataset.yaml config required by Ultralytics YOLOv8."""
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_path.write_text(f"""path: {YOLO_DIR.resolve()}
train: images/train
val: images/val
test: images/test
nc: 1
names: ['text_region']
""")
    print(f"  [YOLO] dataset.yaml -> {yaml_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def prepare_all() -> dict[str, int]:
    """Run full dataset preparation for YOLO + TrOCR. Returns counts dict."""
    print("\n=== Dataset Preparation for TrOCR+YOLO Pipeline ===")
    counts = {}

    for split in ["train", "val", "test"]:
        counts[f"yolo_{split}"] = build_yolo_split(split)
        counts[f"trocr_{split}"] = build_trocr_split(split)

    write_yolo_yaml()

    print("\n  Dataset preparation complete.")
    print(f"  YOLO  data -> {YOLO_DIR}")
    print(f"  TrOCR data -> {TROCR_DIR}")
    return counts


def validate_preparation() -> bool:
    """Validate that all dataset preparation outputs exist and are non-empty.

    Returns True if all splits have valid YOLO and TrOCR data, False otherwise.
    """
    issues = []

    # Check YOLO data
    for split in ["train", "val", "test"]:
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            issues.append(f"YOLO {split}: missing directory")
        else:
            img_count = len(list(img_dir.glob("*")))
            lbl_count = len(list(lbl_dir.glob("*.txt")))
            if img_count == 0:
                issues.append(f"YOLO {split}: no images found")
            if img_count != lbl_count:
                issues.append(f"YOLO {split}: {img_count} images but {lbl_count} labels")

    # Check TrOCR data
    for split in ["train", "val", "test"]:
        trocr_split_dir = TROCR_DIR / split
        meta_path = trocr_split_dir / "metadata.jsonl"

        if not trocr_split_dir.exists():
            issues.append(f"TrOCR {split}: missing directory")
        elif not meta_path.exists():
            issues.append(f"TrOCR {split}: no metadata.jsonl")
        else:
            crop_count = len(list(trocr_split_dir.glob("*.png")))
            meta_count = len(meta_path.read_text().splitlines())
            if crop_count == 0:
                issues.append(f"TrOCR {split}: no crop images")
            if crop_count != meta_count:
                issues.append(
                    f"TrOCR {split}: {crop_count} crops but {meta_count} metadata entries"
                )

    if issues:
        print("\nValidation FAILED:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\nValidation PASSED: All datasets prepared correctly")
        return True


if __name__ == "__main__":
    sys.exit(main())
