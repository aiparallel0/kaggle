"""
dataset_loaders.py — Multi-dataset download & normalization module.

Provides an ABC-based loader hierarchy to download each auxiliary dataset
(WildReceipt, FUNSD, Invoices-DONUT) and normalize their annotations to
the SROIE schema:
    {"company": "...", "date": "...", "address": "...", "total": "..."}

Returns lists of (image_path, ground_truth_dict) tuples (type alias: Sample).

Concrete loaders
----------------
- SROIELoader        — local SROIE img/key directories
- WildReceiptLoader  — OpenMMLab tar download
- FUNSDLoader        — HuggingFace nielsr/funsd (replaces CORD)
- InvoicesDonutLoader— HuggingFace katanaml-org/invoices-donut-data-v1

Backward-compatible module-level functions are provided as thin wrappers
around the loader classes so existing call-sites continue to work.
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
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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
except Exception:
    pass

# FIX: Import shared constants from single source of truth (constants.py)
# instead of defining EMPTY_GT and IMAGE_EXTS independently here.
from constants import EMPTY_GT, SEED, _get_sroie_dir
from constants import IMAGE_EXTS as _IMAGE_EXTS_SET

# ── Type alias ────────────────────────────────────────────────────────
Sample = tuple[Path, dict[str, str]]

# ── Shared constants (derived from constants.py) ─────────────────────
_SROIE_FIELDS = frozenset(EMPTY_GT.keys())
_IMAGE_EXTS = _IMAGE_EXTS_SET


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
        except Exception as exc:
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


def _log_field_coverage(samples: list, dataset_name: str) -> None:
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
                    "address": lines[2].strip(),
                    "total": lines[3].strip(),
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

    @abstractmethod
    def clear_cache(self) -> None:
        """Delete any local cached data for this dataset."""
        ...

    @abstractmethod
    def sample_count(self, split: str = "train") -> int:
        """Return the number of samples in *split* without fully loading."""
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
            if gt:
                samples.append((img_path, gt))

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

    def _dest_dir(self) -> Path:
        """Phase 1: Use shared consolidation utility."""
        return self._get_dest_dir("wildreceipt")

    def _inner_dir(self) -> Path:
        """The extracted ``wildreceipt/`` subdirectory inside _dest_dir()."""
        return self._dest_dir() / "wildreceipt"

    def _marker(self) -> Path:
        """Phase 1: Use shared consolidation utility."""
        return self._get_marker_path(self._dest_dir())

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
        except Exception as exc:
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
        except Exception:
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

    def _dest_dir(self) -> Path:
        """Phase 1: Use shared consolidation utility."""
        return self._get_dest_dir("funsd")

    def _hf_cache(self) -> Path:
        """Phase 1: Use shared consolidation utility."""
        return self._get_hf_cache_dir(self._dest_dir())

    def _marker(self) -> Path:
        """Phase 1: Use shared consolidation utility."""
        return self._get_marker_path(self._dest_dir())

    def _download(self) -> Path:
        """Download FUNSD from the HuggingFace datasets hub."""
        dest = self._dest_dir()
        marker = self._marker()
        if marker.exists():
            if not self._hf_cache().exists():
                self._warn("Cache marker present but hf_cache/ missing — re-downloading.")
                marker.unlink(missing_ok=True)
            else:
                return dest

        try:
            from datasets import load_dataset  # type: ignore

            self._log("Downloading nielsr/funsd from HuggingFace ...")
            ds = load_dataset("nielsr/funsd")
            ds.save_to_disk(str(self._hf_cache()))
            marker.touch()
            self._log("Download complete.")
        except Exception as exc:
            raise self._fatal(f"Download failed: {exc}") from exc
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
        except Exception:
            pass
        return gt

    # ── public interface ──────────────────────────────────────────────

    def load(self, split: str = "train") -> list[Sample]:
        """Load FUNSD and normalize to SROIE schema.

        Skips the ``test`` split to prevent contamination.  Uses content
        hash for on-disk filenames to avoid collisions.
        """
        dest = self._download()
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            raise self._fatal("hf_cache/ not found after download.")

        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
        except Exception as exc:
            raise self._fatal(f"Failed to load cache: {exc}") from exc

        samples: list[Sample] = []
        img_dest_dir = _ensure_dir(dest / "images")
        ds_splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]

        for ds_split in ds_splits:
            if ds_split == "test":
                continue  # avoid test contamination
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
        except Exception:
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
        except Exception:
            return 0


class InvoicesDonutLoader(BaseDatasetLoader):
    """Download Invoices-DONUT (katanaml-org/invoices-donut-data-v1) from
    HuggingFace and normalize to SROIE schema."""

    name = "Invoices-DONUT"

    # ── download ──────────────────────────────────────────────────────

    def _dest_dir(self) -> Path:
        """Phase 1: Use shared consolidation utility."""
        return self._get_dest_dir("invoices_donut")

    def _hf_cache(self) -> Path:
        """Phase 1: Use shared consolidation utility."""
        return self._get_hf_cache_dir(self._dest_dir())

    def _marker(self) -> Path:
        """Phase 1: Use shared consolidation utility."""
        return self._get_marker_path(self._dest_dir())

    def _download(self) -> Path:
        """Download Invoices-DONUT from the HuggingFace datasets hub."""
        dest = self._dest_dir()
        marker = self._marker()
        if marker.exists():
            if not self._hf_cache().exists():
                self._warn("Cache marker present but hf_cache/ missing — re-downloading.")
                marker.unlink(missing_ok=True)
            else:
                return dest

        try:
            from datasets import load_dataset  # type: ignore

            self._log("Downloading katanaml-org/invoices-donut-data-v1 from HuggingFace ...")
            ds = load_dataset("katanaml-org/invoices-donut-data-v1")
            ds.save_to_disk(str(self._hf_cache()))
            marker.touch()
            self._log("Download complete.")
        except Exception as exc:
            raise self._fatal(f"Download failed: {exc}") from exc
        return dest

    # ── Invoices-DONUT → SROIE remapping ─────────────────────────────

    @staticmethod
    def _invoices_donut_remap(ground_truth_str: Any) -> dict[str, str]:
        """Parse Invoices-DONUT ground_truth JSON and remap to SROIE schema.

        Field mapping rationale:
        - company: seller name from gt_parse.header.seller
        - date:    invoice date from gt_parse.header.invoice_date
        - address: seller_address from gt_parse.header.seller_address
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
                gt["company"] = str(header.get("seller", "")).strip()
                gt["date"] = str(header.get("invoice_date", "")).strip()
                # seller_address key does not exist in the actual HuggingFace parquet data
                # (field coverage logs confirm address = 0/475 = 0.0%), so leave it empty.
                gt["address"] = ""

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
        """Load Invoices-DONUT and normalize to SROIE schema."""
        dest = self._download()
        hf_cache = self._hf_cache()
        if not hf_cache.exists():
            raise self._fatal("hf_cache/ not found after download.")

        try:
            from datasets import load_from_disk  # type: ignore

            ds = load_from_disk(str(hf_cache))
        except Exception as exc:
            raise self._fatal(f"Failed to load cache: {exc}") from exc

        samples: list[Sample] = []
        splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]

        for ds_split in splits:
            if ds_split == "test":
                continue  # skip test split to avoid contamination

            if split != "train" and ds_split != split:
                continue

            split_ds = ds[ds_split] if hasattr(ds, "keys") else ds
            for idx, item in enumerate(split_ds):
                gt = self._invoices_donut_remap(item.get("ground_truth", "{}"))
                pil_image = item.get("image")
                if pil_image is not None:
                    img_dest_dir = _ensure_dir(dest / "images")
                    img_path = img_dest_dir / f"{ds_split}_{idx:06d}.jpg"
                    if not img_path.exists():
                        pil_image.convert("RGB").save(img_path, "JPEG")
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
        except Exception:
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
        except Exception:
            return 0


# ======================================================================
#  Singleton loader instances (lazy)
# ======================================================================

_sroie_loader: SROIELoader | None = None
_wildreceipt_loader: WildReceiptLoader | None = None
_funsd_loader: FUNSDLoader | None = None
_invoices_donut_loader: InvoicesDonutLoader | None = None


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


# ======================================================================
#  Combined dataset loader
# ======================================================================

_LOADERS = {
    "sroie": load_sroie_train,
    "wildreceipt": load_wildreceipt,
    "funsd": load_funsd,
    "invoices_donut": load_invoices_donut,
}


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


def get_combined_dataset(
    dataset_names: list[str],
    sroie_oversample: int = 1,
) -> tuple[list[Sample], list[Sample]]:
    """Merge multiple datasets into train and validation lists.

    SROIE training samples are added to train as-is (train split from img/).
    SROIE validation samples (from val_img/) are added to the combined val.
    All auxiliary datasets (WildReceipt, Invoices-DONUT) are split
    70/15/15; the 70% goes into combined train, the 15% validation portion
    goes into combined val, and the held-out 15% test portion is discarded.

    After merging, both combined_train and combined_val are shuffled with
    a fixed seed so that samples from different datasets are interleaved.

    Parameters
    ----------
    dataset_names : list of str
        Names of datasets to include.  Valid names: sroie, wildreceipt,
        funsd, invoices_donut.
    sroie_oversample : int, optional
        Number of times to duplicate SROIE training samples (default 1).
        Use 2 or 3 to counteract SROIE field dilution when combining with
        large auxiliary datasets.

    Returns
    -------
    (train_samples, val_samples) : Tuple[List[Sample], List[Sample]]
    """
    combined_train: list[Sample] = []
    combined_val: list[Sample] = []
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
            # List multiplication creates N references to the same immutable
            # Sample tuples — safe and memory-efficient for read-only iteration.
            combined_train.extend(data * max(1, sroie_oversample))
            sroie_val = load_sroie_val()
            combined_val.extend(sroie_val)
        else:
            # Auxiliary datasets: 70/15/15 split
            train_split, val_split, _ = split_dataset(data, seed=SEED)
            combined_train.extend(train_split)
            combined_val.extend(val_split)

    # Fixed-seed shuffle to interleave samples from different datasets
    rng = random.Random(SEED)
    rng.shuffle(combined_train)
    rng.shuffle(combined_val)

    return combined_train, combined_val
