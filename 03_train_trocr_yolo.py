"""
03_train_trocr_yolo.py
======================
Stage 1 → Fine-tune YOLOv8 on receipt text-region detection
Stage 2 → Fine-tune TrOCR on cropped line images

Run sequentially:
    python 03_train_trocr_yolo.py --stage yolo
    python 03_train_trocr_yolo.py --stage trocr
    python 03_train_trocr_yolo.py --stage both   (default)
"""

import argparse, json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_scheduler,
)
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ── Config ──────────────────────────────────────────────────────────────────
YOLO_DATA_YAML  = Path("data/yolo/dataset.yaml")
YOLO_BASE       = "yolov8m.pt"          # medium – good balance for text detection
YOLO_OUTPUT     = Path("models/yolo_finetuned")
YOLO_EPOCHS     = 50
YOLO_IMG_SIZE   = 640
YOLO_BATCH      = 8

TROCR_MODEL_ID  = "microsoft/trocr-base-printed"   # use 'large-printed' for more accuracy
TROCR_DATA_DIR  = Path("data/trocr")
TROCR_OUTPUT    = Path("models/trocr_finetuned")
TROCR_EPOCHS    = 10
TROCR_BATCH     = 8
TROCR_LR        = 5e-5
TROCR_MAX_LEN   = 128
GRAD_ACCUM      = 4
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_OUTPUT.mkdir(parents=True, exist_ok=True)
TROCR_OUTPUT.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1: YOLO Fine-tuning
# ════════════════════════════════════════════════════════════════════════════
def train_yolo():
    print("=" * 60)
    print("STAGE 1: Fine-tuning YOLOv8 for text-region detection")
    print("=" * 60)

    model = YOLO(YOLO_BASE)

    results = model.train(
        data      = str(YOLO_DATA_YAML),
        epochs    = YOLO_EPOCHS,
        imgsz     = YOLO_IMG_SIZE,
        batch     = YOLO_BATCH,
        project   = str(YOLO_OUTPUT),
        name      = "run",
        exist_ok  = True,
        # Augmentation tuned for receipts (mostly top-down, small rotation)
        degrees   = 5,
        translate = 0.1,
        scale     = 0.3,
        fliplr    = 0.0,       # receipts are not mirrored
        flipud    = 0.0,
        mosaic    = 0.5,
        # Optimiser
        optimizer = "AdamW",
        lr0       = 1e-3,
        lrf       = 0.01,
        patience  = 15,        # early stopping
    )

    print(f"\nYOLO training complete.")
    print(f"Best weights → {YOLO_OUTPUT}/run/weights/best.pt")
    return results


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2: TrOCR Fine-tuning
# ════════════════════════════════════════════════════════════════════════════
class TrOCRReceiptDataset(Dataset):
    def __init__(self, data_dir: Path, processor: TrOCRProcessor, max_length: int):
        self.data_dir  = data_dir
        self.processor = processor
        self.max_len   = max_length
        self.samples   = []

        with open(data_dir / "metadata.jsonl") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample  = self.samples[idx]
        img     = Image.open(self.data_dir / sample["file_name"]).convert("RGB")

        pixel_values = self.processor(
            img, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            sample["text"],
            padding    = "max_length",
            max_length = self.max_len,
            truncation = True,
            return_tensors = "pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def train_trocr():
    print("=" * 60)
    print("STAGE 2: Fine-tuning TrOCR on line crops")
    print("=" * 60)

    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
    model     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)

    # Decoder config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.max_length             = TROCR_MAX_LEN
    model.config.no_repeat_ngram_size   = 3
    model.config.length_penalty         = 2.0
    model.config.num_beams              = 4

    model = model.to(DEVICE)

    train_ds = TrOCRReceiptDataset(TROCR_DATA_DIR / "train", processor, TROCR_MAX_LEN)
    val_ds   = TrOCRReceiptDataset(TROCR_DATA_DIR / "test",  processor, TROCR_MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=TROCR_BATCH, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=TROCR_BATCH, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TROCR_LR)
    total_steps  = (len(train_loader) // GRAD_ACCUM) * TROCR_EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_scheduler(
        "linear", optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(TROCR_EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"TrOCR Epoch {epoch+1}/{TROCR_EPOCHS} [train]")
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels       = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

        avg_train = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"TrOCR Epoch {epoch+1} [val]"):
                outputs = model(
                    pixel_values = batch["pixel_values"].to(DEVICE),
                    labels       = batch["labels"].to(DEVICE)
                )
                val_loss += outputs.loss.item()

        avg_val = val_loss / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        print(f"Epoch {epoch+1}: train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(TROCR_OUTPUT / "best")
            processor.save_pretrained(TROCR_OUTPUT / "best")
            print(f"  ✓ Best TrOCR saved (val_loss={best_val_loss:.4f})")

    model.save_pretrained(TROCR_OUTPUT / "final")
    processor.save_pretrained(TROCR_OUTPUT / "final")
    with open(TROCR_OUTPUT / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTrOCR training complete. Best val_loss={best_val_loss:.4f}")
    return history


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["yolo", "trocr", "both"], default="both")
    args = parser.parse_args()

    if args.stage in ("yolo", "both"):
        train_yolo()

    if args.stage in ("trocr", "both"):
        train_trocr()
