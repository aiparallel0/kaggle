"""
dataset_loaders.py — Multi-dataset download & normalization module.

Provides functions to download each auxiliary dataset (WildReceipt,
SROIE-NER, CORD) and normalize their annotations to the SROIE schema:
{"company": "...", "date": "...", "address": "...", "total": "..."}.

Returns lists of (image_path, ground_truth_dict) tuples.
"""

import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict

# Where auxiliary datasets are stored
DATASETS_DIR = Path("/workspace/datasets")
SROIE_DIR = Path(os.environ.get("SROIE_DATA_DIR", "/workspace/ICDAR-2019-SROIE/data"))

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
    dest = _ensure_dir(DATASETS_DIR / "wildreceipt")
    marker = dest / ".downloaded"
    if marker.exists():
        return dest

    url = _WILDRECEIPT_URL
    tar_path = dest / "wildreceipt.tar"
    try:
        print(f"[WildReceipt] Downloading from {url} ...")
        _download_with_progress(url, tar_path)
        print("[WildReceipt] Extracting tar ...")
        with tarfile.open(str(tar_path)) as tf:
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
# SROIE-NER (darentang/sroie — token-level NER format)
# ---------------------------------------------------------------------------

# FIX (BUG 2): Old code used load_dataset("darentang/sroie") which fails with
# "Dataset scripts are no longer supported" and silently returned [].
# New code uses huggingface_hub / datasets API with dynamic parquet discovery.
_SROIE_NER_REPO = "darentang/sroie"

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


def _download_sroie_ner() -> Path:
    """Download SROIE-NER parquet via huggingface_hub (no deprecated script)."""
    dest = _ensure_dir(DATASETS_DIR / "sroie_ner")
    marker = dest / ".downloaded"
    if marker.exists():
        return dest

    parquet_dest = dest / "train.parquet"

    # --- Primary: try hf_hub_download with several known parquet paths -------
    _known_paths = [
        "data/train-00000-of-00001.parquet",
        "train/train-00000-of-00001.parquet",
    ]
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        import shutil

        downloaded = False
        for hf_path in _known_paths:
            try:
                local = hf_hub_download(
                    repo_id=_SROIE_NER_REPO,
                    filename=hf_path,
                    repo_type="dataset",
                )
                shutil.copy2(local, str(parquet_dest))
                downloaded = True
                break
            except Exception:
                pass  # try next known path

        if not downloaded:
            # Discover actual parquet URLs via the HF Datasets-server API
            api_url = (
                "https://datasets-server.huggingface.co/parquet"
                f"?dataset={_SROIE_NER_REPO}"
            )
            try:
                with urllib.request.urlopen(api_url, timeout=30) as resp:
                    info = json.loads(resp.read())
                parquet_files = [
                    pf["url"]
                    for pf in info.get("parquet_files", [])
                    if pf.get("split") == "train"
                ]
                if parquet_files:
                    _download_with_progress(parquet_files[0], parquet_dest)
                    downloaded = True
            except Exception:
                pass  # fall through to load_dataset fallback

        if not downloaded:
            raise RuntimeError("hf_hub_download: no known parquet path succeeded")

    except Exception as exc_primary:
        print(
            f"[SROIE-NER] hf_hub_download failed ({exc_primary}); "
            "falling back to datasets.load_dataset ...",
            file=sys.stderr,
        )
        # --- Fallback: datasets.load_dataset (no trust_remote_code) ----------
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset(_SROIE_NER_REPO, split="train")
            ds.to_parquet(str(parquet_dest))
        except Exception as exc_fallback:
            print(
                f"[SROIE-NER] datasets.load_dataset failed ({exc_fallback}). "
                "Experiments 3, 6, 7 will run without SROIE-NER data.",
                file=sys.stderr,
            )
            return dest

    marker.touch()
    return dest


def load_sroie_ner() -> List[Sample]:
    """Load SROIE-NER (darentang/sroie) and normalize to SROIE schema."""
    dest = _download_sroie_ner()
    parquet_path = dest / "train.parquet"
    if not parquet_path.exists():
        print("[SROIE-NER] Cache not found — skipping.", file=sys.stderr)
        return []

    try:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(str(parquet_path))
    except Exception as exc:
        print(f"[SROIE-NER] Failed to load parquet: {exc}", file=sys.stderr)
        return []

    samples: List[Sample] = []
    img_dest_dir = _ensure_dir(dest / "images")

    for row_idx, row in df.iterrows():
        gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
        words = row.get("words", [])
        ner_tags = row.get("ner_tags", [])
        for word, tag in zip(words, ner_tags):
            field = _NER_TAG_TO_FIELD.get(int(tag))
            if field:
                text = str(word).strip()
                gt[field] = (gt[field] + " " + text).strip() if gt[field] else text

        # Images are stored as bytes in the parquet — save to disk
        image_data = row.get("image")
        if image_data is not None:
            img_path = img_dest_dir / f"row_{row_idx:06d}.jpg"
            if not img_path.exists():
                try:
                    from PIL import Image  # type: ignore
                    import io
                    if isinstance(image_data, dict) and "bytes" in image_data:
                        raw = image_data["bytes"]
                    elif isinstance(image_data, (bytes, bytearray)):
                        raw = image_data
                    else:
                        raw = None
                    if raw:
                        Image.open(io.BytesIO(raw)).convert("RGB").save(
                            img_path, "JPEG"
                        )
                    else:
                        continue
                except Exception:
                    continue
            samples.append((img_path, gt))

    # Prevent test-set leakage: exclude any images that match SROIE test stems
    test_img_dir = SROIE_DIR / "test_img"
    if test_img_dir.exists():
        test_stems = {p.stem for p in test_img_dir.iterdir() if p.is_file()}
        samples = [(p, gt) for p, gt in samples if p.stem not in test_stems]

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
        # FIX (BUG 9): Loud warning when a requested dataset returns 0 samples,
        # so experiments silently degrading to fewer datasets are immediately visible.
        if len(data) == 0:
            print(
                f"[dataset_loaders] *** WARNING: '{name}' returned 0 samples! "
                f"This experiment's results will NOT reflect this dataset. ***",
                file=sys.stderr,
            )
        combined.extend(data)
    print(f"[dataset_loaders] Total combined samples: {len(combined)}")
    return combined
