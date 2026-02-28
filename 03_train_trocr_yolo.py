"""
03_train_trocr_yolo.py — TrOCR+YOLO two-stage training pipeline.

FIX: Previous version was a standalone script that trained a single YOLOv8
and a single TrOCR model.  This version provides functions callable from
run_all.py to run the SAME 8 dataset combinations as the DONUT experiments.
This ensures a fair, matched experimental design for cross-architecture
comparison.

Architecture:
  Stage 1: YOLOv8x detects text regions (~68.2M params)
  Stage 2: TrOCR-large reads text from crops (~558M params)
  Stage 3: Rule-based heuristics assign fields to extracted text

FIX: Added GPU cleanup between experiments.
FIX: Imports constants from shared module.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_scheduler,
)

from constants import FIELDS, SEED, DEVICE, WORKSPACE, _optimal_num_workers, _gpu_cleanup

# ── Config ──────────────────────────────────────────────────────────────────
TROCR_MODEL_ID = "microsoft/trocr-large-printed"
YOLO_BASE = "yolov8x.pt"  # extra-large — leverages available GPU VRAM (~95GB)
YOLO_EPOCHS = 50
YOLO_IMG_SIZE = 640
YOLO_BATCH = 32
TROCR_EPOCHS = 10
TROCR_BATCH = 16
TROCR_LR = 5e-5
TROCR_MAX_LEN = 128
GRAD_ACCUM = 4

RESULTS_DIR = Path("results")
YOLO_DATA_YAML = Path("data/yolo/dataset.yaml")
TROCR_DATA_DIR = Path("data/trocr")

# Pre-compiled regex patterns for field assignment heuristics — compiled once
# at module load instead of on every call to assign_fields_heuristic().

# Date: numeric (DD/MM/YYYY, YYYY-MM-DD, etc.) OR written month names
_DATE_RE = re.compile(
    r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}'          # 25/12/2023, 25-12-23
    r'|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}'            # 2023/12/25
    r'|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{2,4}'  # 25 DEC 2023
    r'|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s+\d{1,2},?\s+\d{4}'  # DEC 25, 2023
    r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}',   # 25 Dec 2023
    re.IGNORECASE,
)
_TOTAL_RE = re.compile(
    r'(?:total|subtotal|amount|sum|due|grand\s*total|nett\s*total|net\s*total)\s*[:\-]?\s*[\$\£\€RM]?\s*\d+[.,]\d{2}',
    re.IGNORECASE,
)
# Matches a standalone monetary amount at end of line (last-resort total finder)
_MONEY_RE = re.compile(r'[\$\£\€RM]?\s*\d+[.,]\d{2}\s*$')
_NUMBER_RE = re.compile(r'[\d]+[.,][\d]{2}')
# Road/address keywords common in Malaysian/SE Asian receipts
_ADDRESS_RE = re.compile(
    r'\b(?:JALAN|JLN|LORONG|LRG|ROAD|STREET|ST|AVENUE|AVE|BOULEVARD|BLVD'
    r'|TAMAN|TMN|BANDAR|PUSAT|KOMPLEKS|NO\.?\s*\d|LOT\s*\d|\d{5}\s+[A-Z])',
    re.IGNORECASE,
)


# ════════════════════════════════════════════════════════════════════════════
# TrOCR Dataset
# ════════════════════════════════════════════════════════════════════════════
class TrOCRReceiptDataset(Dataset):
    """Line crop dataset for TrOCR fine-tuning."""

    def __init__(self, data_dir: Path, processor: TrOCRProcessor, max_length: int):
        self.data_dir = data_dir
        self.processor = processor
        self.max_len = max_length
        self.samples = []

        meta_path = data_dir / "metadata.jsonl"
        if meta_path.exists():
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(self.data_dir / sample["file_name"]).convert("RGB")

        pixel_values = self.processor(
            img, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1: YOLO Training
# ════════════════════════════════════════════════════════════════════════════
def train_yolo(output_dir: Optional[Path] = None) -> Path:
    """Fine-tune YOLOv8 for text-region detection on receipts.

    Returns the path to the best weights file.
    """
    from ultralytics import YOLO

    if output_dir is None:
        output_dir = WORKSPACE / "models" / "yolo_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 1: Fine-tuning YOLOv8 for text-region detection")
    print("=" * 60)

    if not YOLO_DATA_YAML.exists():
        print(f"  YOLO dataset.yaml not found at {YOLO_DATA_YAML}")
        print("  Run 01_dataset_preparation.py first.")
        return output_dir / "run" / "weights" / "best.pt"

    model = YOLO(YOLO_BASE)
    start = time.time()

    model.train(
        data=str(YOLO_DATA_YAML),
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH,
        project=str(output_dir),
        name="run",
        exist_ok=True,
        degrees=5,
        translate=0.1,
        scale=0.3,
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.5,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        patience=15,
        seed=SEED,
    )

    elapsed = time.time() - start
    best_path = output_dir / "run" / "weights" / "best.pt"
    print(f"\nYOLO training complete in {elapsed:.1f}s")
    print(f"Best weights -> {best_path}")

    # FIX: GPU cleanup after YOLO training
    _gpu_cleanup(model)

    return best_path


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2: TrOCR Training
# ════════════════════════════════════════════════════════════════════════════
def train_trocr(output_dir: Optional[Path] = None) -> Dict:
    """Fine-tune TrOCR on line crops from receipts.

    Returns the training history dict.
    """
    if output_dir is None:
        output_dir = WORKSPACE / "models" / "trocr_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 2: Fine-tuning TrOCR on line crops")
    print("=" * 60)

    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = TROCR_MAX_LEN
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    model = model.to(DEVICE)

    train_dir = TROCR_DATA_DIR / "train"
    val_dir = TROCR_DATA_DIR / "val"

    if not train_dir.exists() or not (train_dir / "metadata.jsonl").exists():
        print(f"  TrOCR training data not found at {train_dir}")
        print("  Run 01_dataset_preparation.py first.")
        return {"train_loss": [], "val_loss": []}

    train_ds = TrOCRReceiptDataset(train_dir, processor, TROCR_MAX_LEN)
    # Use val split (not test!) to match DONUT experiment design
    val_ds = TrOCRReceiptDataset(val_dir, processor, TROCR_MAX_LEN)

    if len(train_ds) == 0:
        raise ValueError("TrOCR training dataset is empty — check data paths.")

    train_loader = DataLoader(
        train_ds, batch_size=TROCR_BATCH, shuffle=True, num_workers=_optimal_num_workers()
    )
    val_loader = DataLoader(
        val_ds, batch_size=TROCR_BATCH, shuffle=False, num_workers=_optimal_num_workers()
    ) if len(val_ds) > 0 else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=TROCR_LR)
    total_steps = (len(train_loader) // GRAD_ACCUM) * TROCR_EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_scheduler(
        "linear", optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    start = time.time()

    for epoch in range(TROCR_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"TrOCR Epoch {epoch+1}/{TROCR_EPOCHS}")
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

        avg_train = epoch_loss / len(train_loader)

        # Validation
        avg_val = float("inf")
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(
                        pixel_values=batch["pixel_values"].to(DEVICE),
                        labels=batch["labels"].to(DEVICE),
                    )
                    val_loss += outputs.loss.item()
            avg_val = val_loss / len(val_loader)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        print(f"Epoch {epoch+1}: train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(output_dir / "best")
            processor.save_pretrained(output_dir / "best")
            print(f"  Best TrOCR saved (val_loss={best_val_loss:.4f})")

    model.save_pretrained(output_dir / "final")
    processor.save_pretrained(output_dir / "final")
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start
    print(f"\nTrOCR training complete in {elapsed:.1f}s. Best val_loss={best_val_loss:.4f}")

    # FIX: GPU cleanup after TrOCR training
    _gpu_cleanup(model, optimizer, scheduler)

    return history


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3: TrOCR+YOLO Inference Pipeline
# ════════════════════════════════════════════════════════════════════════════
def _assign_fields_heuristic(ocr_lines: List[Dict]) -> Dict[str, str]:
    """Assign OCR-extracted text lines to SROIE fields using heuristics.

    This is the key weakness of the pipeline approach: rule-based field
    assignment introduces another source of error on top of detection and
    OCR errors (cascading error propagation).

    Heuristic rules:
    - Total: line containing a dollar/number pattern near the bottom
    - Date: line containing a date-like pattern (DD/MM/YYYY, etc.)
    - Company: first non-date, non-total line (typically the store name)
    - Address: remaining lines between company and total
    """
    result = {f: "" for f in FIELDS}

    if not ocr_lines:
        return result

    # Sort lines by vertical position (top to bottom)
    sorted_lines = sorted(ocr_lines, key=lambda x: x.get("y", 0))

    # Date pattern
    date_pattern = _DATE_RE
    # Total pattern: currency symbols or "total" keyword followed by numbers
    total_pattern = _TOTAL_RE
    # Generic money pattern
    money_pattern = _MONEY_RE

    used = set()

    # Find date — scan all lines (date can appear anywhere on receipt)
    for i, line in enumerate(sorted_lines):
        text = line.get("text", "")
        if _DATE_RE.search(text):
            result["date"] = text.strip()
            used.add(i)
            break

    # Find total — prefer explicit keyword match, fall back to last monetary
    # value in the bottom half of the receipt (common receipt layout).
    for i in range(len(sorted_lines) - 1, -1, -1):
        if i in used:
            continue
        text = sorted_lines[i].get("text", "")
        if _TOTAL_RE.search(text):
            numbers = _NUMBER_RE.findall(text)
            result["total"] = numbers[-1] if numbers else text.strip()
            used.add(i)
            break
    if not result["total"]:
        # Fallback: last line in bottom 40% of receipt that contains a money amount
        cutoff = max(0, len(sorted_lines) - max(1, len(sorted_lines) // 5 * 2))
        for i in range(len(sorted_lines) - 1, cutoff - 1, -1):
            if i in used:
                continue
            text = sorted_lines[i].get("text", "")
            if _MONEY_RE.search(text):
                numbers = _NUMBER_RE.findall(text)
                result["total"] = numbers[-1] if numbers else text.strip()
                used.add(i)
                break

    # Company: first 1-2 unused lines before any address/date/total line
    company_parts = []
    for i, line in enumerate(sorted_lines):
        if i not in used and len(company_parts) < 2:
            text = line.get("text", "").strip()
            if text and not _MONEY_RE.search(text) and not _DATE_RE.search(text):
                company_parts.append(text)
                used.add(i)
                # Stop after first line unless second line also looks like a name
                if len(company_parts) == 1 and not _ADDRESS_RE.search(text):
                    break
    result["company"] = " ".join(company_parts)

    # Address: prefer lines with road/postcode keywords; fall back to remaining
    addr_keyword_parts = []
    addr_other_parts = []
    for i, line in enumerate(sorted_lines):
        if i not in used:
            text = line.get("text", "").strip()
            if not text or _MONEY_RE.match(text):
                continue
            if _ADDRESS_RE.search(text):
                addr_keyword_parts.append(text)
            else:
                addr_other_parts.append(text)
    addr_parts = addr_keyword_parts if addr_keyword_parts else addr_other_parts
    result["address"] = " ".join(addr_parts)

    return result


def run_trocr_yolo_inference(
    image_path: Path,
    yolo_model,
    trocr_model,
    trocr_processor: TrOCRProcessor,
) -> Dict[str, str]:
    """Run the full TrOCR+YOLO pipeline on a single image.

    1. YOLO detects text regions
    2. TrOCR reads text from each crop
    3. Heuristic assigns fields

    Returns a dict with SROIE field predictions.
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # Stage 1: YOLO detection
    yolo_results = yolo_model(img, verbose=False)
    ocr_lines = []

    if yolo_results and len(yolo_results[0].boxes) > 0:
        boxes = yolo_results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            # Add padding
            pad = 4
            x1 = max(0, int(x1) - pad)
            y1 = max(0, int(y1) - pad)
            x2 = min(W, int(x2) + pad)
            y2 = min(H, int(y2) + pad)

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            # Stage 2: TrOCR OCR on crop
            crop = img.crop((x1, y1, x2, y2))
            pixel_values = trocr_processor(
                crop, return_tensors="pt"
            ).pixel_values.to(DEVICE)

            with torch.no_grad():
                generated_ids = trocr_model.generate(pixel_values)
            text = trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            if text:
                ocr_lines.append({
                    "text": text,
                    "x": x1,
                    "y": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": float(box.conf[0]) if hasattr(box, "conf") else 1.0,
                })

    # Stage 3: Heuristic field assignment
    return _assign_fields_heuristic(ocr_lines)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["yolo", "trocr", "both"], default="both"
    )
    args = parser.parse_args()

    if args.stage in ("yolo", "both"):
        train_yolo()

    if args.stage in ("trocr", "both"):
        train_trocr()
