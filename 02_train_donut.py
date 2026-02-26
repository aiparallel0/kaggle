"""
02_train_donut.py
=================
Fine-tune DONUT (naver-clova-ix/donut-base) on receipt images.
Output: structured JSON with fields  company / date / address / total.

Training strategy:
  - Encoder (Swin): partially frozen (bottom 2 layers frozen)
  - Decoder (BART): fully trainable
  - Task token: <s_receipt>
"""

import os, json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    get_scheduler,
)
from PIL import Image
from tqdm import tqdm
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID       = "naver-clova-ix/donut-base"
DATA_DIR       = Path("data/donut")
OUTPUT_DIR     = Path("models/donut_finetuned")
MAX_LENGTH     = 512          # max decoder tokens
IMAGE_SIZE     = (1280, 960)  # (height, width) – DONUT default
BATCH_SIZE     = 2
GRAD_ACCUM     = 4            # effective batch = 8
EPOCHS         = 10
LR             = 5e-5
WARMUP_RATIO   = 0.1
TASK_TOKEN     = "<s_receipt>"
SAVE_STEPS     = 200
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ──────────────────────────────────────────────────────────────────
class DonutReceiptDataset(Dataset):
    def __init__(self, data_dir: Path, processor: DonutProcessor, max_length: int):
        self.data_dir   = data_dir
        self.processor  = processor
        self.max_length = max_length
        self.samples    = []

        meta_path = data_dir / "metadata.jsonl"
        with open(meta_path) as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.data_dir / sample["file_name"]
        image = Image.open(img_path).convert("RGB")

        # Processor encodes image
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Ground-truth sequence:  <s_receipt><company>...</company>...<s_receipt/>
        gt = json.loads(sample["ground_truth"])["gt_parse"]
        target_sequence = TASK_TOKEN
        for key, val in gt.items():
            target_sequence += f"<s_{key}>{val}</s_{key}>"
        target_sequence += f"</{TASK_TOKEN[1:]}"   # closing tag

        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Mask padding tokens in loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Training loop ────────────────────────────────────────────────────────────
def train():
    print(f"Loading DONUT model: {MODEL_ID}")
    processor = DonutProcessor.from_pretrained(MODEL_ID)
    model     = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

    # Add task-specific tokens
    new_tokens = [TASK_TOKEN, "</s_receipt>",
                  "<s_company>", "</s_company>",
                  "<s_date>",    "</s_date>",
                  "<s_address>", "</s_address>",
                  "<s_total>",   "</s_total>"]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # Decoder config
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.eos_token_id
    model.config.max_length             = MAX_LENGTH

    # Partially freeze encoder (freeze bottom 2 Swin stages)
    for name, param in model.encoder.named_parameters():
        if "layers.0" in name or "layers.1" in name:
            param.requires_grad = False

    model = model.to(DEVICE)

    train_ds = DonutReceiptDataset(DATA_DIR / "train", processor, MAX_LENGTH)
    val_ds   = DonutReceiptDataset(DATA_DIR / "test",  processor, MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    total_steps    = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps   = int(total_steps * WARMUP_RATIO)
    scheduler = get_scheduler(
        "cosine", optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    global_step   = 0
    history       = {"train_loss": [], "val_loss": []}

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
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
                global_step += 1

            pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

        avg_train = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels       = batch["labels"].to(DEVICE)
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()

        avg_val = val_loss / len(val_loader)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        print(f"Epoch {epoch+1}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(OUTPUT_DIR / "best")
            processor.save_pretrained(OUTPUT_DIR / "best")
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

    # Save final + history
    model.save_pretrained(OUTPUT_DIR / "final")
    processor.save_pretrained(OUTPUT_DIR / "final")
    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDONUT training complete. Best val_loss={best_val_loss:.4f}")
    return history


if __name__ == "__main__":
    train()
