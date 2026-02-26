"""
01_dataset_preparation.py
=========================
Loads the SROIE receipt dataset (or your own), prepares it for:
  - DONUT  → (image, JSON label) pairs
  - YOLO   → (image, bounding-box .txt) pairs
  - TrOCR  → (cropped line image, text) pairs
"""

import os, json, shutil, random
from pathlib import Path
from PIL import Image
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────────────
SEED        = 42
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15
DATA_DIR    = Path("data")
RAW_DIR     = DATA_DIR / "raw"
DONUT_DIR   = DATA_DIR / "donut"
YOLO_DIR    = DATA_DIR / "yolo"
TROCR_DIR   = DATA_DIR / "trocr"

# Key fields to extract (adjust to your receipt schema)
TARGET_KEYS = ["company", "date", "address", "total"]

random.seed(SEED)

# ── 1. Load SROIE from HuggingFace ──────────────────────────────────────────
def load_sroie():
    """
    Uses the 'darentang/sroie' dataset from HuggingFace Hub.
    Each sample has: image, words, bboxes, labels (IOB tags)
    """
    print("Loading SROIE dataset...")
    ds = load_dataset("darentang/sroie", trust_remote_code=True)
    return ds

# ── 2. Build DONUT dataset ───────────────────────────────────────────────────
def build_donut_dataset(ds, split="train"):
    """
    DONUT expects:
      - image: PIL Image
      - ground_truth: JSON string  {"gt_parse": {"company": ..., "date": ..., ...}}
    """
    out_dir = DONUT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    for idx, sample in enumerate(ds[split]):
        img: Image.Image = sample["image"]
        img_path = out_dir / f"{idx:05d}.png"
        img.save(img_path)

        # Build ground-truth JSON from IOB labels
        gt = extract_key_values_from_iob(sample["words"], sample["labels"])
        gt_json = json.dumps({"gt_parse": gt}, ensure_ascii=False)

        metadata.append({
            "file_name": str(img_path.name),
            "ground_truth": gt_json
        })

    # Save metadata.jsonl (DONUT format)
    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"[DONUT] {split}: {len(metadata)} samples → {out_dir}")
    return metadata


def extract_key_values_from_iob(words, labels):
    """Convert IOB-tagged words into a flat key-value dict."""
    result = {k: "" for k in TARGET_KEYS}
    current_key, current_tokens = None, []

    for word, label in zip(words, labels):
        if label.startswith("B-"):
            if current_key:
                result[current_key] = " ".join(current_tokens).strip()
            current_key = label[2:].lower()
            current_tokens = [word]
        elif label.startswith("I-") and current_key:
            current_tokens.append(word)
        else:
            if current_key:
                result[current_key] = " ".join(current_tokens).strip()
            current_key, current_tokens = None, []

    if current_key:
        result[current_key] = " ".join(current_tokens).strip()

    return result


# ── 3. Build YOLO dataset ────────────────────────────────────────────────────
def build_yolo_dataset(ds, split="train"):
    """
    YOLOv8 format:
      images/<split>/image.png
      labels/<split>/image.txt   (one line per box: class cx cy w h  normalised)
    Only one class: 'text_region'
    """
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(ds[split]):
        img: Image.Image = sample["image"]
        W, H = img.size
        img.save(img_dir / f"{idx:05d}.png")

        # Group word-level bboxes into line bboxes (simple row grouping)
        lines = group_words_into_lines(sample["words"], sample["bboxes"], W, H)

        with open(lbl_dir / f"{idx:05d}.txt", "w") as f:
            for (cx, cy, bw, bh) in lines:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    # Save dataset.yaml
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_content = f"""path: {YOLO_DIR.resolve()}
train: images/train
val:   images/val
test:  images/test
nc: 1
names: ['text_region']
"""
    yaml_path.write_text(yaml_content)
    print(f"[YOLO] {split}: saved → {img_dir}")


def group_words_into_lines(words, bboxes, W, H, row_tol=10):
    """
    Cluster words into horizontal lines by y-coordinate proximity.
    Returns list of (cx, cy, bw, bh) normalised to [0,1].
    bboxes are [x1, y1, x2, y2] in pixel coords (SROIE format).
    """
    if not bboxes:
        return []

    # Sort by top-y
    items = sorted(zip(bboxes, words), key=lambda x: x[0][1])
    lines, cur_line = [], [items[0]]

    for item in items[1:]:
        if abs(item[0][1] - cur_line[-1][0][1]) < row_tol:
            cur_line.append(item)
        else:
            lines.append(cur_line)
            cur_line = [item]
    lines.append(cur_line)

    result = []
    for line in lines:
        bbs = [b for b, _ in line]
        x1 = min(b[0] for b in bbs) / W
        y1 = min(b[1] for b in bbs) / H
        x2 = max(b[2] for b in bbs) / W
        y2 = max(b[3] for b in bbs) / H
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        result.append((cx, cy, bw, bh))
    return result


# ── 4. Build TrOCR dataset ───────────────────────────────────────────────────
def build_trocr_dataset(ds, split="train"):
    """
    TrOCR expects (cropped line image, text) pairs.
    We crop each word bbox (or line bbox) from the receipt image.
    """
    out_dir = TROCR_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for idx, sample in enumerate(ds[split]):
        img: Image.Image = sample["image"]
        W, H = img.size

        # Group into lines for better context
        lines = group_words_into_lines_with_text(
            sample["words"], sample["bboxes"], W, H
        )

        for line_idx, (bbox_px, text) in enumerate(lines):
            if not text.strip():
                continue
            x1, y1, x2, y2 = bbox_px
            # Add small padding
            pad = 4
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(W, x2 + pad), min(H, y2 + pad)

            crop = img.crop((x1, y1, x2, y2))
            crop_name = f"{idx:05d}_{line_idx:03d}.png"
            crop.save(out_dir / crop_name)
            records.append({"file_name": crop_name, "text": text})

    meta_path = out_dir / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"[TrOCR] {split}: {len(records)} line crops → {out_dir}")
    return records


def group_words_into_lines_with_text(words, bboxes, W, H, row_tol=10):
    if not bboxes:
        return []
    items = sorted(zip(bboxes, words), key=lambda x: x[0][1])
    lines, cur_line = [], [items[0]]

    for item in items[1:]:
        if abs(item[0][1] - cur_line[-1][0][1]) < row_tol:
            cur_line.append(item)
        else:
            lines.append(cur_line)
            cur_line = [item]
    lines.append(cur_line)

    result = []
    for line in lines:
        bbs  = [b for b, _ in line]
        txts = [t for _, t in line]
        x1 = min(b[0] for b in bbs)
        y1 = min(b[1] for b in bbs)
        x2 = max(b[2] for b in bbs)
        y2 = max(b[3] for b in bbs)
        result.append(((x1, y1, x2, y2), " ".join(txts)))
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = load_sroie()

    for split in ["train", "test"]:
        print(f"\n=== Processing {split} split ===")
        build_donut_dataset(ds, split)
        build_yolo_dataset(ds, split)
        build_trocr_dataset(ds, split)

    print("\nDataset preparation complete.")
    print(f"  DONUT data → {DONUT_DIR}")
    print(f"  YOLO  data → {YOLO_DIR}")
    print(f"  TrOCR data → {TROCR_DIR}")
