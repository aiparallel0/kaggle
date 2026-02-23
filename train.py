import json, random, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import (DonutProcessor, VisionEncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          EarlyStoppingCallback)

# WARNING: This is a legacy standalone script. For the full 8-experiment
# pipeline, use: python run_all.py
# This script is kept for backward compatibility and ad-hoc single-model
# training/evaluation outside the experiment framework.

import os
# BUG C FIX: Use a function so the env var is re-read at call time, not cached
# at import time (run_all.py sets SROIE_DATA_DIR after this module is imported).
def _get_sroie_dir():
    return Path(os.environ.get("SROIE_DATA_DIR", "/workspace/ICDAR-2019-SROIE/data"))

FIELDS = ["company", "date", "address", "total"]
MAX_LENGTH = 512
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

NEW_TOKENS = [
    "<s_sroie>", "</s_sroie>",
    "<s_company>", "</s_company>",
    "<s_date>",    "</s_date>",
    "<s_address>", "</s_address>",
    "<s_total>",   "</s_total>",
]

class SROIEDataset(Dataset):
    def __init__(self, processor, img_dir, key_dir, max_length=MAX_LENGTH):
        self.processor = processor
        self.max_length = max_length
        self.samples = []
        for img_path in sorted(p for p in Path(img_dir).iterdir()
                                if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            # BUG E FIX: Key files are .txt (4-line format), not .json.
            # Try .json first for pre-converted data, then fall back to .txt.
            gt = None
            key_json = Path(key_dir) / (img_path.stem + ".json")
            if key_json.exists():
                try:
                    gt = json.loads(key_json.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            if gt is None:
                key_txt = Path(key_dir) / (img_path.stem + ".txt")
                if key_txt.exists():
                    lines = key_txt.read_text(encoding="utf-8").strip().splitlines()
                    if len(lines) >= 4:
                        gt = {
                            "company": lines[0].strip(),
                            "date":    lines[1].strip(),
                            "address": lines[2].strip(),
                            "total":   lines[3].strip(),
                        }
            if gt is not None:
                self.samples.append((img_path, gt))
        print(f"Loaded {len(self.samples)} samples")

    @classmethod
    def from_samples(cls, processor, samples, max_length=MAX_LENGTH):
        """Create a SROIEDataset from a pre-built list of (path, gt) tuples."""
        obj = cls.__new__(cls)
        Dataset.__init__(obj)
        obj.processor = processor
        obj.max_length = max_length
        obj.samples = list(samples)
        return obj

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        target = "<s_sroie>"
        for f in FIELDS:
            v = gt.get(f, "")
            target += f"<s_{f}>{v}</s_{f}>"
        target += "</s_sroie>"

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            target, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def main():
    from transformers import set_seed as _set_seed
    _set_seed(42)  # Sets random, numpy, torch, and CUDA seeds for reproducibility

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    processor.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]

    model.gradient_checkpointing_enable()

    sroie_dir = _get_sroie_dir()
    full_ds = SROIEDataset(processor, sroie_dir / "img", sroie_dir / "key")
    all_samples = full_ds.samples

    # Use last 15% of samples as validation for early stopping
    shuffled = list(all_samples)
    random.seed(42)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.15))
    train_samples = shuffled[n_val:]
    val_samples = shuffled[:n_val]

    train_ds = SROIEDataset.from_samples(processor, train_samples)
    val_ds = SROIEDataset.from_samples(processor, val_samples)

    workspace = os.environ.get("DONUT_WORKSPACE", "/workspace")
    output_dir = os.path.join(workspace, "donut-sroie-finetuned")
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=30,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print("TRAINING_COMPLETE")

if __name__ == "__main__":
    main()