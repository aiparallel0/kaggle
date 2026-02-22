"""
dataset_loaders.py — Multi-dataset download & normalization module.

Provides functions to download each auxiliary dataset (WildReceipt, FUNSD,
XFUND, EATEN, CORD, Kaggle Scanned Images) and normalize their annotations
to the SROIE schema: {"company": "...", "date": "...", "address": "...", "total": "..."}.

Returns lists of (image_path, ground_truth_dict) tuples.
"""

import json
import os
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import List, Tuple, Dict

# Where auxiliary datasets are stored
DATASETS_DIR = Path("/workspace/datasets")
SROIE_DIR = Path("/workspace/ICDAR-2019-SROIE/data")

# Type alias
Sample = Tuple[Path, Dict[str, str]]

EMPTY_GT = {"company": "", "date": "", "address": "", "total": ""}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# SROIE
# ---------------------------------------------------------------------------

def load_sroie_train() -> List[Sample]:
    """Load SROIE training split (526 samples) from the local workspace."""
    samples: List[Sample] = []
    img_dir = SROIE_DIR / "img"
    key_dir = SROIE_DIR / "key"
    if not img_dir.exists():
        raise FileNotFoundError(f"SROIE img dir not found: {img_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    for img_path in sorted(p for p in img_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in image_exts):
        key_file = key_dir / (img_path.stem + ".json")
        if key_file.exists():
            try:
                gt = json.loads(key_file.read_text(encoding="utf-8"))
                samples.append((img_path, {k: str(gt.get(k, "")) for k in EMPTY_GT}))
            except json.JSONDecodeError:
                pass
    return samples


def load_sroie_test() -> List[Sample]:
    """Load SROIE test split (100 samples) from the local workspace."""
    samples: List[Sample] = []
    img_dir = SROIE_DIR / "test_img"
    key_dir = SROIE_DIR / "test_key"
    if not img_dir.exists():
        raise FileNotFoundError(f"SROIE test_img dir not found: {img_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    for img_path in sorted(p for p in img_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in image_exts):
        key_file = key_dir / (img_path.stem + ".json")
        if key_file.exists():
            try:
                gt = json.loads(key_file.read_text(encoding="utf-8"))
                samples.append((img_path, {k: str(gt.get(k, "")) for k in EMPTY_GT}))
            except json.JSONDecodeError:
                pass
    return samples


# ---------------------------------------------------------------------------
# WildReceipt
# ---------------------------------------------------------------------------

def _download_wildreceipt() -> Path:
    """Download WildReceipt from the HuggingFace datasets hub."""
    dest = _ensure_dir(DATASETS_DIR / "wildreceipt")
    marker = dest / ".downloaded"
    if marker.exists():
        return dest

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("jinhybr/WildReceipt", trust_remote_code=True)
        ds.save_to_disk(str(dest / "hf_cache"))
        marker.touch()
    except Exception as exc:
        print(f"[WildReceipt] Download failed: {exc}. Will return empty dataset.")
    return dest


# WildReceipt category index → SROIE field mapping (best effort).
# Category indices per the original annotation schema:
#   1 = Store_name_value  → company
#   5 = Date_value        → date
#   9 = Tel_value         (ignored)
#  13 = Tax_value         (ignored)
#  17 = Total_value       → total
# Address is not a standard WildReceipt category; left empty.
_WILDRECEIPT_CAT_TO_FIELD = {
    1: "company",
    5: "date",
    17: "total",
}


def load_wildreceipt() -> List[Sample]:
    """Load WildReceipt and normalize to SROIE schema."""
    dest = _download_wildreceipt()
    hf_cache = dest / "hf_cache"
    if not hf_cache.exists():
        print("[WildReceipt] Cache not found — skipping.")
        return []

    try:
        from datasets import load_from_disk  # type: ignore
        ds = load_from_disk(str(hf_cache))
    except Exception as exc:
        print(f"[WildReceipt] Failed to load cache: {exc}")
        return []

    samples: List[Sample] = []

    # Support both DatasetDict and Dataset
    splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    for split in splits:
        split_ds = ds[split] if hasattr(ds, "keys") else ds
        for item in split_ds:
            gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
            annotations = item.get("annotations", [])
            for ann in annotations:
                cat_id = ann.get("category_id", -1)
                text = str(ann.get("text", "")).strip()
                field = _WILDRECEIPT_CAT_TO_FIELD.get(cat_id)
                if field and text:
                    # Concatenate if multiple boxes for the same field
                    gt[field] = (gt[field] + " " + text).strip() if gt[field] else text

            img_path_str = item.get("file_name", item.get("image_path", ""))
            if img_path_str:
                img_path = Path(img_path_str)
                if not img_path.is_absolute():
                    img_path = dest / img_path_str
                if img_path.exists():
                    samples.append((img_path, gt))

    return samples


# ---------------------------------------------------------------------------
# FUNSD
# ---------------------------------------------------------------------------

def _download_funsd() -> Path:
    """Download FUNSD from the HuggingFace datasets hub."""
    dest = _ensure_dir(DATASETS_DIR / "funsd")
    marker = dest / ".downloaded"
    if marker.exists():
        return dest

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("nielsr/funsd", trust_remote_code=True)
        ds.save_to_disk(str(dest / "hf_cache"))
        marker.touch()
    except Exception as exc:
        print(f"[FUNSD] Download failed: {exc}. Will return empty dataset.")
    return dest


def _funsd_extract_gt(item: dict) -> Dict[str, str]:
    """Extract SROIE-schema GT from a FUNSD annotation item (best effort)."""
    gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
    for ann in item.get("annotations", []):
        label = str(ann.get("label", "")).lower()
        text = str(ann.get("text", "")).strip()
        if not text:
            continue
        # Heuristic: match label keywords to SROIE fields
        if any(kw in label for kw in ("company", "store", "merchant", "vendor", "name")):
            gt["company"] = (gt["company"] + " " + text).strip()
        elif any(kw in label for kw in ("date", "time")):
            gt["date"] = (gt["date"] + " " + text).strip()
        elif any(kw in label for kw in ("address", "addr", "street", "city")):
            gt["address"] = (gt["address"] + " " + text).strip()
        elif any(kw in label for kw in ("total", "amount", "price", "subtotal")):
            gt["total"] = (gt["total"] + " " + text).strip()
    return gt


def load_funsd() -> List[Sample]:
    """Load FUNSD and normalize to SROIE schema."""
    dest = _download_funsd()
    hf_cache = dest / "hf_cache"
    if not hf_cache.exists():
        print("[FUNSD] Cache not found — skipping.")
        return []

    try:
        from datasets import load_from_disk  # type: ignore
        ds = load_from_disk(str(hf_cache))
    except Exception as exc:
        print(f"[FUNSD] Failed to load cache: {exc}")
        return []

    samples: List[Sample] = []
    splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    for split in splits:
        split_ds = ds[split] if hasattr(ds, "keys") else ds
        for item in split_ds:
            gt = _funsd_extract_gt(item)
            img_path_str = item.get("img_path", item.get("image_path", ""))
            if img_path_str:
                img_path = Path(img_path_str)
                if not img_path.is_absolute():
                    img_path = dest / img_path_str
                if img_path.exists():
                    samples.append((img_path, gt))
    return samples


# ---------------------------------------------------------------------------
# XFUND
# ---------------------------------------------------------------------------

def _download_xfund() -> Path:
    """Download XFUND from the HuggingFace datasets hub."""
    dest = _ensure_dir(DATASETS_DIR / "xfund")
    marker = dest / ".downloaded"
    if marker.exists():
        return dest

    try:
        from datasets import load_dataset  # type: ignore
        # XFUND has multiple languages; load all available splits
        for lang in ["zh", "ja", "es", "fr", "it", "de", "pt"]:
            try:
                ds = load_dataset("nielsr/XFUND", lang, trust_remote_code=True)
                ds.save_to_disk(str(dest / f"hf_cache_{lang}"))
            except Exception as exc:
                print(f"[XFUND] lang={lang} failed: {exc}")
        marker.touch()
    except Exception as exc:
        print(f"[XFUND] Download failed: {exc}. Will return empty dataset.")
    return dest


def load_xfund() -> List[Sample]:
    """Load XFUND (all languages) and normalize to SROIE schema."""
    dest = _download_xfund()
    samples: List[Sample] = []

    for lang in ["zh", "ja", "es", "fr", "it", "de", "pt"]:
        hf_cache = dest / f"hf_cache_{lang}"
        if not hf_cache.exists():
            continue
        try:
            from datasets import load_from_disk  # type: ignore
            ds = load_from_disk(str(hf_cache))
        except Exception as exc:
            print(f"[XFUND] lang={lang} load failed: {exc}")
            continue

        splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
        for split in splits:
            split_ds = ds[split] if hasattr(ds, "keys") else ds
            for item in split_ds:
                gt = _funsd_extract_gt(item)  # same schema as FUNSD
                img_path_str = item.get("img_path", item.get("image_path", ""))
                if img_path_str:
                    img_path = Path(img_path_str)
                    if not img_path.is_absolute():
                        img_path = dest / img_path_str
                    if img_path.exists():
                        samples.append((img_path, gt))
    return samples


# ---------------------------------------------------------------------------
# EATEN (Chinese restaurant receipts)
# ---------------------------------------------------------------------------

def _download_eaten() -> Path:
    """Download EATEN from the HuggingFace datasets hub."""
    dest = _ensure_dir(DATASETS_DIR / "eaten")
    marker = dest / ".downloaded"
    if marker.exists():
        return dest

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("jinhybr/EATEN", trust_remote_code=True)
        ds.save_to_disk(str(dest / "hf_cache"))
        marker.touch()
    except Exception as exc:
        print(f"[EATEN] Download failed: {exc}. Will return empty dataset.")
    return dest


def load_eaten() -> List[Sample]:
    """Load EATEN and normalize to SROIE schema."""
    dest = _download_eaten()
    hf_cache = dest / "hf_cache"
    if not hf_cache.exists():
        print("[EATEN] Cache not found — skipping.")
        return []

    try:
        from datasets import load_from_disk  # type: ignore
        ds = load_from_disk(str(hf_cache))
    except Exception as exc:
        print(f"[EATEN] Failed to load cache: {exc}")
        return []

    samples: List[Sample] = []
    splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    for split in splits:
        split_ds = ds[split] if hasattr(ds, "keys") else ds
        for item in split_ds:
            gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
            # EATEN fields: store_name → company, date → date, total → total
            gt["company"] = str(item.get("store_name", "")).strip()
            gt["date"] = str(item.get("date", "")).strip()
            gt["total"] = str(item.get("total", "")).strip()
            gt["address"] = str(item.get("address", "")).strip()

            img_path_str = item.get("file_name", item.get("image_path", ""))
            if img_path_str:
                img_path = Path(img_path_str)
                if not img_path.is_absolute():
                    img_path = dest / img_path_str
                if img_path.exists():
                    samples.append((img_path, gt))
    return samples


# ---------------------------------------------------------------------------
# CORD
# ---------------------------------------------------------------------------

def _download_cord() -> Path:
    """Download CORD from the HuggingFace datasets hub."""
    dest = _ensure_dir(DATASETS_DIR / "cord")
    marker = dest / ".downloaded"
    if marker.exists():
        return dest

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("naver-clova-ix/cord-v2", trust_remote_code=True)
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
        # CORD has no standard date field
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
# Kaggle Scanned Images Dataset
# ---------------------------------------------------------------------------

def _download_kaggle_scanned() -> Path:
    """Download the Kaggle Scanned Receipts dataset via the Kaggle API."""
    dest = _ensure_dir(DATASETS_DIR / "kaggle_scanned")
    marker = dest / ".downloaded"
    if marker.exists():
        return dest

    try:
        import subprocess
        # Dataset identifier: 'jenswalter/receipts' — a public Kaggle scanned receipts dataset.
        # Update this slug if you wish to use a different Kaggle dataset.
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d",
             "jenswalter/receipts", "-p", str(dest), "--unzip"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            marker.touch()
        else:
            print(f"[Kaggle Scanned] Download failed: {result.stderr}")
    except Exception as exc:
        print(f"[Kaggle Scanned] Download failed: {exc}. Will return empty dataset.")
    return dest


def load_kaggle_scanned() -> List[Sample]:
    """Load Kaggle Scanned Images and normalize to SROIE schema (best effort)."""
    dest = _download_kaggle_scanned()
    samples: List[Sample] = []
    image_exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

    # Walk the dataset directory for images and optional annotation files
    for img_path in sorted(dest.rglob("*")):
        if not (img_path.is_file() and img_path.suffix.lower() in image_exts):
            continue
        # Look for a sidecar JSON annotation file
        json_file = img_path.with_suffix(".json")
        gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
        if json_file.exists():
            try:
                raw = json.loads(json_file.read_text(encoding="utf-8"))
                # Generic remapping — adapt keys as found in the actual dataset
                gt["company"] = str(raw.get("company", raw.get("store_name", raw.get("merchant", "")))).strip()
                gt["date"] = str(raw.get("date", raw.get("transaction_date", ""))).strip()
                gt["address"] = str(raw.get("address", raw.get("store_address", ""))).strip()
                gt["total"] = str(raw.get("total", raw.get("total_amount", raw.get("amount", "")))).strip()
            except json.JSONDecodeError:
                pass
        samples.append((img_path, gt))

    return samples


# ---------------------------------------------------------------------------
# Combined dataset loader
# ---------------------------------------------------------------------------

_LOADERS = {
    "sroie": load_sroie_train,
    "wildreceipt": load_wildreceipt,
    "funsd": load_funsd,
    "xfund": load_xfund,
    "eaten": load_eaten,
    "cord": load_cord,
    "kaggle_scanned": load_kaggle_scanned,
}


def get_combined_dataset(dataset_names: List[str]) -> List[Sample]:
    """
    Merge multiple datasets into a single list of (image_path, gt_dict) tuples.

    Parameters
    ----------
    dataset_names : list of str
        Names of datasets to include.  Valid names: sroie, wildreceipt, funsd,
        xfund, eaten, cord, kaggle_scanned.

    Returns
    -------
    List[Sample]
        Combined list of (Path, dict) tuples.
    """
    combined: List[Sample] = []
    for name in dataset_names:
        loader = _LOADERS.get(name)
        if loader is None:
            raise ValueError(f"Unknown dataset '{name}'. Valid: {list(_LOADERS)}")
        print(f"[dataset_loaders] Loading '{name}' ...")
        data = loader()
        print(f"[dataset_loaders] '{name}' → {len(data)} samples")
        combined.extend(data)
    print(f"[dataset_loaders] Total combined samples: {len(combined)}")
    return combined
