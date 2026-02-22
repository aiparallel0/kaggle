"""
dataset_loaders.py — Multi-dataset download & normalization module.

Provides functions to download each auxiliary dataset (WildReceipt,
SROIE-NER, CORD) and normalize their annotations to the SROIE schema:
{"company": "...", "date": "...", "address": "...", "total": "..."}.

Returns lists of (image_path, ground_truth_dict) tuples.
"""

import json
import os
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
        ds = load_dataset("Theivaprakasham/wildreceipt")
        ds.save_to_disk(str(dest / "hf_cache"))
        marker.touch()
    except Exception as exc:
        print(f"[WildReceipt] Download failed: {exc}. Will return empty dataset.")
    return dest


# WildReceipt string label → SROIE field mapping.
# Labels per the Theivaprakasham/wildreceipt annotation schema:
#   Store_name_value  → company
#   Date_value        → date
#   Total_value       → total
# Address is not a standard WildReceipt category; left empty.
_WILDRECEIPT_CAT_TO_FIELD = {
    "Store_name_value": "company",
    "Date_value": "date",
    "Total_value": "total",
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
    img_dest_dir = _ensure_dir(dest / "images")

    # Support both DatasetDict and Dataset
    splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    for split in splits:
        split_ds = ds[split] if hasattr(ds, "keys") else ds
        for item in split_ds:
            gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
            words = item.get("words", [])
            for word in words:
                label = str(word.get("label", ""))
                text = str(word.get("text", "")).strip()
                field = _WILDRECEIPT_CAT_TO_FIELD.get(label)
                if field and text:
                    # Concatenate if multiple tokens for the same field
                    gt[field] = (gt[field] + " " + text).strip() if gt[field] else text

            # Images are PIL objects in this dataset — save to disk
            pil_image = item.get("image")
            if pil_image is not None:
                idx = len(samples)
                img_path = img_dest_dir / f"{split}_{idx:06d}.jpg"
                if not img_path.exists():
                    pil_image.convert("RGB").save(img_path, "JPEG")
                samples.append((img_path, gt))

    return samples


# ---------------------------------------------------------------------------
# SROIE-NER (darentang/sroie — token-level NER format)
# ---------------------------------------------------------------------------

def _download_sroie_ner() -> Path:
    """Download SROIE-NER from the HuggingFace datasets hub."""
    dest = _ensure_dir(DATASETS_DIR / "sroie_ner")
    marker = dest / ".downloaded"
    if marker.exists():
        return dest

    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("darentang/sroie")
        ds.save_to_disk(str(dest / "hf_cache"))
        marker.touch()
    except Exception as exc:
        print(f"[SROIE-NER] Download failed: {exc}. Will return empty dataset.")
    return dest


# NER tag index → (SROIE field, is_begin) mapping.
# Tag scheme: 0=O, 1=B-COMPANY, 2=I-COMPANY, 3=B-DATE, 4=I-DATE,
#             5=B-ADDRESS, 6=I-ADDRESS, 7=B-TOTAL, 8=I-TOTAL
_NER_TAG_TO_FIELD = {
    1: "company",
    2: "company",
    3: "date",
    4: "date",
    5: "address",
    6: "address",
    7: "total",
    8: "total",
}


def load_sroie_ner() -> List[Sample]:
    """Load SROIE-NER (darentang/sroie) and normalize to SROIE schema."""
    dest = _download_sroie_ner()
    hf_cache = dest / "hf_cache"
    if not hf_cache.exists():
        print("[SROIE-NER] Cache not found — skipping.")
        return []

    try:
        from datasets import load_from_disk  # type: ignore
        ds = load_from_disk(str(hf_cache))
    except Exception as exc:
        print(f"[SROIE-NER] Failed to load cache: {exc}")
        return []

    samples: List[Sample] = []
    img_dest_dir = _ensure_dir(dest / "images")

    splits = list(ds.keys()) if hasattr(ds, "keys") else ["train"]
    for split in splits:
        if split == "test":
            continue  # only use train/validation for augmentation
        split_ds = ds[split] if hasattr(ds, "keys") else ds
        for item in split_ds:
            gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
            words = item.get("words", [])
            ner_tags = item.get("ner_tags", [])
            # Aggregate contiguous NER tags back into field-level strings
            for word, tag in zip(words, ner_tags):
                field = _NER_TAG_TO_FIELD.get(tag)
                if field:
                    text = str(word).strip()
                    gt[field] = (gt[field] + " " + text).strip() if gt[field] else text

            # Images are PIL objects in this dataset — save to disk
            pil_image = item.get("image")
            if pil_image is not None:
                idx = len(samples)
                img_path = img_dest_dir / f"{split}_{idx:06d}.jpg"
                if not img_path.exists():
                    pil_image.convert("RGB").save(img_path, "JPEG")
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
# Combined dataset loader
# ---------------------------------------------------------------------------

_LOADERS = {
    "sroie": load_sroie_train,
    "wildreceipt": load_wildreceipt,
    "sroie_ner": load_sroie_ner,
    "cord": load_cord,
}


def get_combined_dataset(dataset_names: List[str]) -> List[Sample]:
    """
    Merge multiple datasets into a single list of (image_path, gt_dict) tuples.

    Parameters
    ----------
    dataset_names : list of str
        Names of datasets to include.  Valid names: sroie, wildreceipt,
        sroie_ner, cord.

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
