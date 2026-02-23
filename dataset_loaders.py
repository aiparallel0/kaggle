"""
dataset_loaders.py — Multi-dataset download & normalization module.

Provides functions to download each auxiliary dataset (WildReceipt,
SROIE-NER, CORD, Invoices-DONUT) and normalize their annotations to the
SROIE schema: {"company": "...", "date": "...", "address": "...", "total": ""}.

Returns lists of (image_path, ground_truth_dict) tuples.
"""

import json
import os
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict

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
    """Load SROIE training split (526 samples) from the local workspace."""
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
    """Load SROIE test split (100 samples) from the local workspace."""
    samples: List[Sample] = []
    sroie_dir = _get_sroie_dir()
    img_dir = sroie_dir / "test_img"
    key_dir = sroie_dir / "test_key"
    if not img_dir.exists():
        raise FileNotFoundError(f"SROIE test_img dir not found: {img_dir}")

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
    """Download SROIE-NER parquet from HuggingFace datasets-server API.

    BUG B FIX: The old code used load_dataset("darentang/sroie") which fails
    with "Dataset scripts are no longer supported" on datasets>=4.0.  Even the
    auto-converted parquet from refs/convert/parquet only contains an
    'image_path' string column (not embedded image bytes).

    The fix downloads the NER parquet (words + ner_tags + image_path) from the
    datasets-server API and relies on load_sroie_ner() to resolve image_path
    stems against the local SROIE image directory — no embedded images needed.
    """
    dest = _ensure_dir(_get_datasets_dir() / "sroie_ner")
    marker = dest / ".downloaded"
    parquet_dest = dest / "train.parquet"

    # Validate existing cache: if the parquet exists but is missing ner_tags
    # (stale download), delete it so we re-download the correct file.
    if marker.exists() and parquet_dest.exists():
        try:
            import pyarrow.parquet as pq  # type: ignore
            schema_names = pq.read_schema(str(parquet_dest)).names
            if "ner_tags" in schema_names:
                return dest  # Cache is valid
            print(
                "[SROIE-NER] Stale cache detected (parquet missing 'ner_tags'). Re-downloading.",
                file=sys.stderr,
            )
        except Exception:
            pass
        marker.unlink(missing_ok=True)
        parquet_dest.unlink(missing_ok=True)

    if marker.exists():
        return dest

    # PRIMARY: Download from HuggingFace datasets-server API.
    api_url = "https://datasets-server.huggingface.co/parquet?dataset=darentang/sroie"
    downloaded = False
    try:
        with urllib.request.urlopen(api_url, timeout=30) as resp:
            info = json.loads(resp.read())
        parquet_urls = [
            pf["url"]
            for pf in info.get("parquet_files", [])
            if pf.get("split") == "train"
        ]
        if parquet_urls:
            print("[SROIE-NER] Downloading parquet from datasets-server ...")
            _download_with_progress(parquet_urls[0], parquet_dest)
            downloaded = True
    except Exception as exc:
        print(f"[SROIE-NER] datasets-server download failed: {exc}", file=sys.stderr)

    # FALLBACK: hf_hub_download from refs/convert/parquet branch.
    if not downloaded:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            import shutil
            local = hf_hub_download(
                repo_id="darentang/sroie",
                filename="sroie/train/0000.parquet",
                repo_type="dataset",
                revision="refs/convert/parquet",
            )
            shutil.copy2(local, str(parquet_dest))
            downloaded = True
        except Exception as exc:
            print(f"[SROIE-NER] hf_hub_download fallback failed: {exc}", file=sys.stderr)

    if not downloaded:
        print(
            "[SROIE-NER] FATAL: All download methods failed. "
            "Experiments 3, 6, 7 will run without SROIE-NER data.",
            file=sys.stderr,
        )
        return dest

    # Validate: ensure the parquet has usable NER data before caching.
    try:
        import pandas as pd
        df = pd.read_parquet(str(parquet_dest))
        if "ner_tags" not in df.columns:
            print(
                f"[SROIE-NER] WARNING: Downloaded parquet columns={list(df.columns)} "
                "— missing 'ner_tags'. Not caching.",
                file=sys.stderr,
            )
            return dest
        print(f"[SROIE-NER] Validated: {len(df)} rows with NER tags.")
    except Exception as exc:
        print(f"[SROIE-NER] Validation failed: {exc}", file=sys.stderr)
        return dest

    marker.touch()
    print("[SROIE-NER] Download and validation complete.")
    return dest


def load_sroie_ner() -> List[Sample]:
    """Load SROIE-NER (darentang/sroie) and normalize to SROIE schema.

    BUG B FIX: Reads the NER parquet directly (words + ner_tags + image_path)
    and resolves each image_path stem against the local SROIE image directory.
    The darentang/sroie dataset is the SROIE dataset re-published with NER
    tags, so its images are already present at _get_sroie_dir()/img/.

    This avoids the need for embedded image bytes in the parquet and works with
    datasets>=4.0 (which no longer supports legacy dataset scripts).
    """
    dest = _download_sroie_ner()
    parquet_path = dest / "train.parquet"
    if not parquet_path.exists():
        print("[SROIE-NER] Parquet not found — skipping.", file=sys.stderr)
        return []

    try:
        import pandas as pd
        df = pd.read_parquet(str(parquet_path))
    except Exception as exc:
        print(f"[SROIE-NER] Failed to read parquet: {exc}", file=sys.stderr)
        return []

    sroie_img_dir = _get_sroie_dir() / "img"
    samples: List[Sample] = []
    embedded_img_counter = 0

    for _, row in df.iterrows():
        gt: Dict[str, str] = {k: "" for k in EMPTY_GT}
        raw_words = row.get("words")
        raw_tags = row.get("ner_tags")
        words = raw_words if isinstance(raw_words, (list, tuple)) else []
        ner_tags = raw_tags if isinstance(raw_tags, (list, tuple)) else []
        for word, tag in zip(words, ner_tags):
            field = _NER_TAG_TO_FIELD.get(int(tag))
            if field:
                text = str(word).strip()
                gt[field] = (gt[field] + " " + text).strip() if gt[field] else text

        # Resolve the image from the local SROIE directory using the stem of
        # image_path (e.g. "img/X00016469612.jpg" → stem "X00016469612").
        image_path_val = row.get("image_path") or row.get("image") or ""
        if isinstance(image_path_val, dict):
            # Future-proof: handle embedded-bytes dict if HF ever embeds images.
            image_bytes = image_path_val.get("bytes")
            if image_bytes:
                img_dest_dir = _ensure_dir(dest / "images")
                img_path = img_dest_dir / f"sroie_ner_{embedded_img_counter:06d}.jpg"
                if not img_path.exists():
                    try:
                        import io
                        from PIL import Image  # type: ignore
                        Image.open(io.BytesIO(image_bytes)).convert("RGB").save(img_path, "JPEG")
                    except Exception:
                        continue
                embedded_img_counter += 1
                samples.append((img_path, gt))
            continue

        stem = Path(str(image_path_val)).stem if image_path_val else ""
        if not stem:
            continue
        local_img = sroie_img_dir / (stem + ".jpg")
        if not local_img.exists():
            for ext in (".jpeg", ".png", ".tiff", ".tif"):
                candidate = sroie_img_dir / (stem + ext)
                if candidate.exists():
                    local_img = candidate
                    break
            else:
                continue
        samples.append((local_img, gt))

    if not samples:
        print(
            "[SROIE-NER] WARNING: 0 samples loaded. "
            "The parquet image_path stems may not match local SROIE images. "
            "Deleting cache marker so next run will re-download.",
            file=sys.stderr,
        )
        (dest / ".downloaded").unlink(missing_ok=True)

    # Prevent test-set leakage: exclude images that match SROIE test stems.
    test_img_dir = _get_sroie_dir() / "test_img"
    if test_img_dir.exists():
        test_stems = {p.stem for p in test_img_dir.iterdir() if p.is_file()}
        samples = [(p, gt) for p, gt in samples if p.stem not in test_stems]
    else:
        print(
            "[SROIE-NER] WARNING: test_img/ directory not found — cannot filter test-set images. "
            "Call stage_install() first to create the test split. Returning empty list to "
            "prevent train/test leakage.",
            file=sys.stderr,
        )
        return []

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
    "sroie_ner": load_sroie_ner,
    "cord": load_cord,
    "invoices_donut": load_invoices_donut,
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
    per_loader_counts: Dict[str, int] = {}
    for name in dataset_names:
        loader = _LOADERS.get(name)
        if loader is None:
            raise ValueError(f"Unknown dataset '{name}'. Valid: {list(_LOADERS)}")
        print(f"[dataset_loaders] Loading '{name}' ...")
        data = loader()
        per_loader_counts[name] = len(data)
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
    # Print per-loader summary so users can see exactly which loaders contributed samples
    counts_summary = ", ".join(f"{n}={c}" for n, c in per_loader_counts.items())
    print(f"[dataset_loaders] Per-loader counts: {counts_summary}")
    print(f"[dataset_loaders] Total combined samples: {len(combined)}")
    return combined
