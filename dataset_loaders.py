"""
dataset_loaders.py — Multi-dataset download & normalization module.

Provides functions to download each auxiliary dataset (WildReceipt,
CORD, Invoices-DONUT) and normalize their annotations to the SROIE
schema: {"company": "...", "date": "...", "address": "...", "total": ""}.

Returns lists of (image_path, ground_truth_dict) tuples.
"""

import json
import os
import random
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict

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

# Where auxiliary datasets are stored — respects DONUT_WORKSPACE env var.
def _get_datasets_dir() -> Path:
    return Path(os.environ.get("DONUT_WORKSPACE", "/workspace")) / "datasets"

# BUG C FIX: module-level constant was evaluated once at import time, so
# os.environ["SROIE_DATA_DIR"] set later in run_all.py had no effect.
# Use a function that re-reads the env var on every call instead.
def _get_sroie_dir() -> Path:
    return Path(os.environ.get("SROIE_DATA_DIR", "/workspace/ICDAR-2019-SROIE/data"))

# Type alias
Sample = Tuple[Path, Dict[str, str]]

EMPTY_GT = {"company": "", "date": "", "address": "", "total": ""}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
                        print(f"\r[{pct:5.1f}%] {dl_mb:.1f} / {total_mb:.1f} MB",
                              end="", flush=True)
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
                    f"\n[download] Attempt {attempt + 1} failed: {exc}. "
                    f"Retrying in {wait}s ...",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# SROIE
# ---------------------------------------------------------------------------

def _load_key_file(key_dir: Path, stem: str) -> Dict[str, str]:
    """Load a SROIE key file for the given image stem.

    BUG A FIX: The SROIE repo stores annotations as plain .txt files with
    4 lines (company / date / address / total), NOT as .json files.  The old
    code only tried .json, so key_file.exists() was always False and every
    sample was silently skipped.  We now try .json first (for pre-converted
    data) and fall back to the 4-line .txt format.
    """
    # Try .json first in case a user has pre-converted the files
    key_file_json = key_dir / (stem + ".json")
    if key_file_json.exists():
        try:
            gt = json.loads(key_file_json.read_text(encoding="utf-8"))
            return {k: str(gt.get(k, "")) for k in EMPTY_GT}
        except json.JSONDecodeError:
            pass

    # Fall back to the 4-line .txt format that the repo actually provides
    key_file_txt = key_dir / (stem + ".txt")
    if key_file_txt.exists():
        lines = key_file_txt.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) >= 4:
            return {
                "company": lines[0].strip(),
                "date":    lines[1].strip(),
                "address": lines[2].strip(),
                "total":   lines[3].strip(),
            }

    return {}


def load_sroie_train() -> List[Sample]:
    """Load SROIE training split from the local workspace (img/ + key/)."""
    samples: List[Sample] = []
    sroie_dir = _get_sroie_dir()
    img_dir = sroie_dir / "img"
    key_dir = sroie_dir / "key"
    if not img_dir.exists():
        raise FileNotFoundError(f"SROIE img dir not found: {img_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    for img_path in sorted(p for p in img_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in image_exts):
        gt = _load_key_file(key_dir, img_path.stem)
        if gt:
            samples.append((img_path, gt))
    return samples


def load_sroie_test() -> List[Sample]:
    """Load SROIE test split from the local workspace (test_img/ + test_key/).

    Returns an empty list with a warning if the test split directory
    (test_img/) has not been populated.  This avoids raising on pipelines
    where the test images are not available.
    """
    samples: List[Sample] = []
    sroie_dir = _get_sroie_dir()
    img_dir = sroie_dir / "test_img"
    key_dir = sroie_dir / "test_key"
    if not img_dir.exists():
        print(
            "[SROIE] WARNING: test_img/ directory not found — returning 0 test samples. "
            "Evaluation will be skipped.",
            file=sys.stderr,
        )
        return []

    image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    for img_path in sorted(p for p in img_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in image_exts):
        gt = _load_key_file(key_dir, img_path.stem)
        if gt:
            samples.append((img_path, gt))
    return samples


def load_sroie_val() -> List[Sample]:
    """Load SROIE validation split from the local workspace (val_img/ + val_key/).

    Returns an empty list with a warning if the validation split directory
    (val_img/) has not been populated.
    """
    samples: List[Sample] = []
    sroie_dir = _get_sroie_dir()
    img_dir = sroie_dir / "val_img"
    key_dir = sroie_dir / "val_key"
    if not img_dir.exists():
        print(
            "[SROIE] WARNING: val_img/ directory not found — returning 0 val samples.",
            file=sys.stderr,
        )
        return []

    image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    for img_path in sorted(p for p in img_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in image_exts):
        gt = _load_key_file(key_dir, img_path.stem)
        if gt:
            samples.append((img_path, gt))
    return samples


# ---------------------------------------------------------------------------
# WildReceipt
# ---------------------------------------------------------------------------

# FIX (BUG 1): Old code used load_dataset("Theivaprakasham/wildreceipt") which
# fails with "Dataset scripts are no longer supported" and silently returned [].
# New code downloads the official OpenMMLab tar directly.
_WILDRECEIPT_URL = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"

# WildReceipt label index → SROIE field mapping.
# Label indices from class_list.txt (0-indexed):
#   1 = Store_name_value → company
#   3 = Date_value       → date
#   7 = Total_value      → total
#  10 = Addr_value       → address
_WILDRECEIPT_IDX_TO_FIELD = {
    1: "company",
    3: "date",
    7: "total",
    10: "address",
}


def _download_wildreceipt() -> Path:
    """Download WildReceipt from OpenMMLab tar (confirmed working URL)."""
    dest = _ensure_dir(_get_datasets_dir() / "wildreceipt")
    marker = dest / ".downloaded"
    if marker.exists():
        # Validate cache: ensure train.txt exists inside the extracted directory.
        # A partial download or failed extraction would leave a stale marker.
        if not (dest / "wildreceipt" / "train.txt").exists():
            print(
                "[WildReceipt] Cache marker present but train.txt missing — re-downloading.",
                file=sys.stderr,
            )
            marker.unlink(missing_ok=True)
        else:
            return dest

    url = _WILDRECEIPT_URL
    tar_path = dest / "wildreceipt.tar"
    try:
        print(f"[WildReceipt] Downloading from {url} ...")
        _download_with_progress(url, tar_path)
        print("[WildReceipt] Extracting tar ...")
        with tarfile.open(str(tar_path)) as tf:
            # Use filter='data' on Python 3.12+ to prevent path-traversal attacks.
            if sys.version_info >= (3, 12):
                tf.extractall(str(dest), filter="data")
            else:
                tf.extractall(str(dest))
        tar_path.unlink(missing_ok=True)
        marker.touch()
        print("[WildReceipt] Download and extraction complete.")
    except Exception as exc:
        # FIX (BUG 1): Loud failure instead of silent empty return.
        # Previously the exception was swallowed; now we print a FATAL warning.
        print(
            f"[WildReceipt] FATAL: Download failed from {url}: {exc}. "
            "This experiment will have MISSING DATA.",
            file=sys.stderr,
        )
    return dest


def load_wildreceipt() -> List[Sample]:
    """Load WildReceipt (OpenMMLab format) and normalize to SROIE schema."""
    dest = _download_wildreceipt()
    # INTEGRITY: Only load train.txt to prevent test-set contamination.
    # The tar contains both train.txt and test.txt; test.txt must not be used.
    train_txt = dest / "wildreceipt" / "train.txt"
    if not train_txt.exists():
        print(
            "[WildReceipt] *** WARNING: 'wildreceipt' returned 0 samples! "
            "This experiment's results will NOT reflect this dataset. ***",
            file=sys.stderr,
        )
        return []

    samples: List[Sample] = []
    img_base = dest / "wildreceipt"

    with open(train_txt, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
            for ann in obj.get("annotations", []):
                label_idx = int(ann.get("label", -1))
                field = _WILDRECEIPT_IDX_TO_FIELD.get(label_idx)
                text = str(ann.get("text", "")).strip()
                if field and text:
                    gt[field] = (gt[field] + " " + text).strip() if gt[field] else text

            file_name = obj.get("file_name", "")
            img_path = img_base / file_name
            if img_path.exists():
                samples.append((img_path, gt))

    return samples


# ---------------------------------------------------------------------------
# CORD
# ---------------------------------------------------------------------------

def _download_cord() -> Path:
    """Download CORD from the HuggingFace datasets hub."""
    dest = _ensure_dir(_get_datasets_dir() / "cord")
    marker = dest / ".downloaded"
    if marker.exists():
        # Validate cache: ensure the HF dataset directory exists.
        # A partial download would leave a stale marker without usable data.
        if not (dest / "hf_cache").exists():
            print(
                "[CORD] Cache marker present but hf_cache/ missing — re-downloading.",
                file=sys.stderr,
            )
            marker.unlink(missing_ok=True)
        else:
            return dest

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("naver-clova-ix/cord-v2")
        ds.save_to_disk(str(dest / "hf_cache"))
        marker.touch()
    except Exception as exc:
        print(f"[CORD] Download failed: {exc}. Will return empty dataset.")
    return dest


def _cord_remap(ground_truth_str: str) -> Dict[str, str]:
    """Parse CORD ground_truth JSON and remap to SROIE schema."""
    gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
    try:
        obj = json.loads(ground_truth_str) if isinstance(ground_truth_str, str) else ground_truth_str
        gt_json = obj.get("gt_parse", obj)
        store_info = gt_json.get("store_info", {})
        if isinstance(store_info, dict):
            gt["company"] = str(store_info.get("store_name", "")).strip()
            gt["address"] = str(store_info.get("region", "")).strip()
        total_info = gt_json.get("total", {})
        if isinstance(total_info, dict):
            gt["total"] = str(total_info.get("total_price", "")).strip()
        elif isinstance(total_info, list) and len(total_info) > 0:
            gt["total"] = str(total_info[0].get("total_price", "")).strip()
        # FIX (BUG 7): CORD v2 DOES contain date annotations in gt_parse.
        # Previously hardcoded to "" with incorrect comment "CORD has no standard date field".
        date_info = gt_json.get("date", {})
        if isinstance(date_info, dict):
            gt["date"] = str(date_info.get("date_value", "")).strip()
        elif isinstance(date_info, list) and len(date_info) > 0:
            gt["date"] = str(date_info[0].get("date_value", "")).strip()
        elif isinstance(date_info, str):
            gt["date"] = date_info.strip()
        else:
            gt["date"] = ""
    except (json.JSONDecodeError, AttributeError):
        pass
    return gt


def load_cord() -> List[Sample]:
    """Load CORD and normalize to SROIE schema."""
    dest = _download_cord()
    hf_cache = dest / "hf_cache"
    if not hf_cache.exists():
        print("[CORD] Cache not found — skipping.")
        return []

    try:
        from datasets import load_from_disk  # type: ignore
        ds = load_from_disk(str(hf_cache))
    except Exception as exc:
        print(f"[CORD] Failed to load cache: {exc}")
        return []

    samples: List[Sample] = []
    splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    for split in splits:
        if split == "test":
            continue  # only use train/validation for augmentation
        split_ds = ds[split] if hasattr(ds, "keys") else ds
        for item in split_ds:
            gt = _cord_remap(item.get("ground_truth", "{}"))
            # CORD images are PIL Images in the HF dataset
            pil_image = item.get("image")
            if pil_image is not None:
                # Save to disk so we can use the path-based training pipeline
                img_dest_dir = _ensure_dir(dest / "images")
                idx = len(samples)
                img_path = img_dest_dir / f"{split}_{idx:06d}.jpg"
                if not img_path.exists():
                    pil_image.convert("RGB").save(img_path, "JPEG")
                samples.append((img_path, gt))
    return samples


# ---------------------------------------------------------------------------
# Invoices-DONUT (katanaml-org/invoices-donut-data-v1)
# ---------------------------------------------------------------------------

def _download_invoices_donut() -> Path:
    """Download Invoices-DONUT from the HuggingFace datasets hub."""
    dest = _ensure_dir(_get_datasets_dir() / "invoices_donut")
    marker = dest / ".downloaded"
    if marker.exists():
        if not (dest / "hf_cache").exists():
            print(
                "[Invoices-DONUT] Cache marker present but hf_cache/ missing — re-downloading.",
                file=sys.stderr,
            )
            marker.unlink(missing_ok=True)
        else:
            return dest

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("katanaml-org/invoices-donut-data-v1")
        ds.save_to_disk(str(dest / "hf_cache"))
        marker.touch()
    except Exception as exc:
        print(f"[Invoices-DONUT] Download failed: {exc}. Will return empty dataset.")
    return dest


def _invoices_donut_remap(ground_truth_str: str) -> Dict[str, str]:
    """Parse Invoices-DONUT ground_truth JSON and remap to SROIE schema.

    Field mapping rationale:
    - company: seller name from gt_parse.header.seller (full string; the
      model learns to extract just the company name portion).
    - date:    invoice date from gt_parse.header.invoice_date.
    - address: empty string — invoices don't have a single address field
      that reliably maps to SROIE's "store address".
    - total:   total gross worth from gt_parse.summary.total_gross_worth,
      with any leading currency symbol (e.g. '$') stripped.
    """
    gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
    try:
        obj = json.loads(ground_truth_str) if isinstance(ground_truth_str, str) else ground_truth_str
        gt_parse = obj.get("gt_parse", obj)

        header = gt_parse.get("header", {})
        if isinstance(header, dict):
            # Seller → company: the full seller string is used here because the
            # dataset doesn't split company name from address.  The DONUT decoder
            # will learn to emit only the company name portion during fine-tuning.
            gt["company"] = str(header.get("seller", "")).strip()
            # Invoice date → date
            gt["date"] = str(header.get("invoice_date", "")).strip()

        # address: no reliable single-address field in invoice schema
        gt["address"] = ""

        summary = gt_parse.get("summary", {})
        if isinstance(summary, dict):
            raw_total = str(summary.get("total_gross_worth", "")).strip()
            # Strip any leading currency symbol (e.g. '$', '€', '£', '¥', '₹', '₩')
            # using a regex so all Unicode currency prefixes are handled uniformly.
            gt["total"] = re.sub(r"^[\$€£¥₹₩\u20ac\u00a3\u00a5]+", "", raw_total).strip()
    except (json.JSONDecodeError, AttributeError):
        pass
    return gt


def load_invoices_donut() -> List[Sample]:
    """Load Invoices-DONUT and normalize to SROIE schema."""
    dest = _download_invoices_donut()
    hf_cache = dest / "hf_cache"
    if not hf_cache.exists():
        print("[Invoices-DONUT] Cache not found — skipping.")
        return []

    try:
        from datasets import load_from_disk  # type: ignore
        ds = load_from_disk(str(hf_cache))
    except Exception as exc:
        print(f"[Invoices-DONUT] Failed to load cache: {exc}")
        return []

    samples: List[Sample] = []
    splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    for split in splits:
        if split == "test":
            continue  # skip test split to avoid contamination
        split_ds = ds[split] if hasattr(ds, "keys") else ds
        for idx, item in enumerate(split_ds):
            gt = _invoices_donut_remap(item.get("ground_truth", "{}"))
            # Invoices-DONUT images are PIL Images embedded in the HF dataset
            pil_image = item.get("image")
            if pil_image is not None:
                # Save to disk so the path-based training pipeline can load them
                img_dest_dir = _ensure_dir(dest / "images")
                img_path = img_dest_dir / f"{split}_{idx:06d}.jpg"
                if not img_path.exists():
                    pil_image.convert("RGB").save(img_path, "JPEG")
                samples.append((img_path, gt))

    if not samples:
        print(
            "[Invoices-DONUT] *** WARNING: 'invoices_donut' returned 0 samples! "
            "This experiment's results will NOT reflect this dataset. ***",
            file=sys.stderr,
        )
    return samples


# ---------------------------------------------------------------------------
# Combined dataset loader
# ---------------------------------------------------------------------------

_LOADERS = {
    "sroie": load_sroie_train,
    "wildreceipt": load_wildreceipt,
    "cord": load_cord,
    "invoices_donut": load_invoices_donut,
}


def split_dataset(
    samples: List[Sample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
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
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def get_combined_dataset(
    dataset_names: List[str],
) -> Tuple[List[Sample], List[Sample]]:
    """
    Merge multiple datasets into train and validation lists.

    SROIE training samples are added to train as-is (train split from img/).
    SROIE validation samples (from val_img/) are added to the combined val.
    All auxiliary datasets (WildReceipt, CORD, Invoices-DONUT) are split
    70/15/15; the 70% goes into combined train, the 15% validation portion
    goes into combined val, and the held-out 15% test portion is discarded.

    After merging, both combined_train and combined_val are shuffled with
    a fixed seed so that samples from different datasets are interleaved.

    Parameters
    ----------
    dataset_names : list of str
        Names of datasets to include.  Valid names: sroie, wildreceipt,
        cord, invoices_donut.

    Returns
    -------
    (train_samples, val_samples) : Tuple[List[Sample], List[Sample]]
    """
    combined_train: List[Sample] = []
    combined_val: List[Sample] = []
    per_loader_counts: Dict[str, int] = {}

    for name in dataset_names:
        loader = _LOADERS.get(name)
        if loader is None:
            raise ValueError(f"Unknown dataset '{name}'. Valid: {list(_LOADERS)}")
        print(f"[dataset_loaders] Loading '{name}' ...")
        data = loader()
        per_loader_counts[name] = len(data)
        print(f"[dataset_loaders] '{name}' → {len(data)} samples")
        if len(data) == 0:
            print(
                f"[dataset_loaders] *** WARNING: '{name}' returned 0 samples! "
                f"This experiment's results will NOT reflect this dataset. ***",
                file=sys.stderr,
            )
        if name == "sroie":
            # SROIE: train split goes to combined_train; val split goes to combined_val.
            combined_train.extend(data)
            sroie_val = load_sroie_val()
            combined_val.extend(sroie_val)
            if sroie_val:
                print(f"[dataset_loaders] 'sroie_val' → {len(sroie_val)} val samples")
        else:
            # Auxiliary datasets: 70/15/15 split; hold-out test portion discarded.
            train_split, val_split, _ = split_dataset(data, seed=42)
            combined_train.extend(train_split)
            combined_val.extend(val_split)

    # Fixed-seed shuffle to interleave samples from different datasets.
    rng = random.Random(42)
    rng.shuffle(combined_train)
    rng.shuffle(combined_val)

    counts_summary = ", ".join(f"{n}={c}" for n, c in per_loader_counts.items())
    print(f"[dataset_loaders] Per-loader counts: {counts_summary}")
    print(f"[dataset_loaders] Combined train: {len(combined_train)}  val: {len(combined_val)}")
    return combined_train, combined_val
