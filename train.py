import json, torch, os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import (DonutProcessor, VisionEncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

SROIE_DIR = Path("/workspace/ICDAR-2019-SROIE/data")
FIELDS = ["company", "date", "address", "total"]
MAX_LENGTH = 512

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
        for img_path in sorted(Path(img_dir).glob("*.jpg")):
            key_file = Path(key_dir) / (img_path.stem + ".txt")
            if key_file.exists():
                try:
                    gt = json.loads(key_file.read_text(encoding="utf-8"))
                    self.samples.append((img_path, gt))
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        target = "<s_sroie>"
        for f in FIELDS:
            v = gt.get(f, "")
            target += f"<s_{f}>{v}</{f}>"
        target += "</s_sroie>"

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            target, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def main():
    # Start from CORD checkpoint (faster convergence than donut-base)
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    processor.tokenizer.add_special_tokens({"additional_special_tokens": NEW_TOKENS})
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]

    model.gradient_checkpointing_enable()

    # Split: use first 500 images for train, rest for validation
    all_imgs = sorted((SROIE_DIR / "img").glob("*.jpg"))
    split_idx = 500

    train_ds = SROIEDataset(processor, SROIE_DIR / "img", SROIE_DIR / "key")
    # Use all data for training (small dataset), evaluate separately
    
    args = Seq2SeqTrainingArguments(
        output_dir="/workspace/donut-sroie-finetuned",
        num_train_epochs=10,          # 10 epochs from CORD checkpoint is enough
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True,
        logging_steps=20,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
    )

    trainer.train()
    model.save_pretrained("/workspace/donut-sroie-finetuned")
    processor.save_pretrained("/workspace/donut-sroie-finetuned")
    print("TRAINING_COMPLETE")

if __name__ == "__main__":
    main()