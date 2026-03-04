"""
dataset_preparation.py — Prepare YOLO bbox labels + TrOCR line crops.

FIX: The previous version loaded from HuggingFace 'darentang/sroie' which
is broken.  This version uses the EXISTING SROIE split (already downloaded
and split by run_all.py stage_install) to generate YOLO and TrOCR data
from the local img/key directories.  This ensures:
  - Same 500/63/63 train/val/test split used by DONUT
  - No data leakage (test images never appear in training)
  - No redundant re-download

FIX (trocr_train: 0): box_dir was only set for the train split AND only
when SROIE_DATA_DIR/box/ exists.  For val/test splits and when box/
is absent, the code now falls back to key-file full-image crops for ALL splits
so TrOCR always has training data.

Output directories:
  data/yolo/images/{train,val,test}/  + data/yolo/labels/{train,val,test}/
  data/trocr/{train,val,test}/metadata.jsonl + line crop images
"""

import json
import os
from pathlib import Path

from PIL import Image

from constants import FIELDS, IMAGE_EXTS
from dataset_loaders import SROIELoader, _load_key_file

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
YOLO_DIR = DATA_DIR / "yolo"
TROCR_DIR = DATA_DIR / "trocr"

# FIX: Use existing SROIE directories from run_all.py stage_install,
# not a HuggingFace download that is broken upstream.
SROIE_DATA_DIR = Path(os.environ.get("SROIE_DATA_DIR", "/workspace/ICDAR-2019-SROIE/data"))

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


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_box_dir(split: str) -> Path | None:
    """Return the first existing box annotation directory for a split, or None."""
    for candidate in BOX_DIR_CANDIDATES.get(split, []):
        p = SROIE_DATA_DIR / candidate
        if p.exists() and any(p.iterdir()):
            return p
    return None


# ── OCR bbox file reader ──────────────────────────────────────────────────────
def _load_ocr_bboxes(box_dir: Path, stem: str) -> list[tuple[list[int], str]]:
    """Load SROIE OCR bounding boxes from the box/ directory.

    Each line has format: x1,y1,x2,y2,x3,y3,x4,y4,text
    Returns list of ([x1,y1,x2,y2], text) tuples using the axis-aligned
    bounding box of the four corner points.
    """
    box_file = box_dir / f"{stem}.txt"
    if not box_file.exists():
        return []

    results = []
    for line in box_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 8)
        if len(parts) < 9:
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
            continue
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
    img_dir = SROIE_DATA_DIR / img_subdir

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
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Try to load OCR bboxes for YOLO labels
        boxes = []
        if box_dir is not None:
            raw_boxes = _load_ocr_bboxes(box_dir, img_path.stem)
            boxes = group_words_into_lines(raw_boxes)

        # Save image
        dest_img = out_img_dir / img_path.name
        if not dest_img.exists():
            img.save(dest_img)

        # Write YOLO label file (may be empty if no box annotations)
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
    img_dir = SROIE_DATA_DIR / img_subdir
    key_dir = SROIE_DATA_DIR / key_subdir

    if not img_dir.exists():
        print(f"  [TrOCR] {split}: {img_subdir}/ not found — skipping")
        return 0

    # FIX: probe all candidate box dirs for this split
    box_dir = _find_box_dir(split)

    out_dir = TROCR_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img = Image.open(img_path).convert("RGB")
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
            issues.append(f"❌ YOLO {split}: missing directory")
        else:
            img_count = len(list(img_dir.glob("*")))
            lbl_count = len(list(lbl_dir.glob("*.txt")))
            if img_count == 0:
                issues.append(f"❌ YOLO {split}: no images found")
            if img_count != lbl_count:
                issues.append(f"⚠️  YOLO {split}: {img_count} images but {lbl_count} labels")

    # Check TrOCR data
    for split in ["train", "val", "test"]:
        trocr_split_dir = TROCR_DIR / split
        meta_path = trocr_split_dir / "metadata.jsonl"

        if not trocr_split_dir.exists():
            issues.append(f"❌ TrOCR {split}: missing directory")
        elif not meta_path.exists():
            issues.append(f"❌ TrOCR {split}: no metadata.jsonl")
        else:
            crop_count = len(list(trocr_split_dir.glob("*.png")))
            meta_count = len(meta_path.read_text().splitlines())
            if crop_count == 0:
                issues.append(f"❌ TrOCR {split}: no crop images")
            if crop_count != meta_count:
                issues.append(
                    f"⚠️  TrOCR {split}: {crop_count} crops but {meta_count} metadata entries"
                )

    if issues:
        print("\n❌ Validation FAILED:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print("\n✅ Validation PASSED: All datasets prepared correctly")
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare YOLO bbox labels + TrOCR line crops for SROIE dataset"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate existing preparation (do not re-prepare)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-prepare all data even if directories already exist"
    )

    args = parser.parse_args()

    if args.validate:
        validate_preparation()
    else:
        if args.force:
            import shutil

            if YOLO_DIR.exists():
                shutil.rmtree(YOLO_DIR)
            if TROCR_DIR.exists():
                shutil.rmtree(TROCR_DIR)
            print("Cleared existing data directories due to --force flag")

        counts = prepare_all()
        print("\nPreparation summary:")
        for key, val in sorted(counts.items()):
            print(f"  {key:15s} {val:4d}")

        # Auto-validate after preparation
        validate_preparation()
