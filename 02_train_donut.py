"""
02_train_donut.py — Standalone DONUT fine-tuning (reference implementation).

WARNING: This is a reference script.  For the full 8-experiment pipeline,
use ``python run_all.py``.  This script is provided for ad-hoc training
outside the experiment framework.

FIX: Applied lm_head weight tying fix — sets tie_word_embeddings=False after
resize_token_embeddings() so lm_head and embed_tokens are saved independently.
Without this, the saved checkpoint loses the learned lm_head weights and
reloaded models produce garbage output (F1=0).

FIX: Imports constants from shared constants.py instead of duplicating.

FIX: Delegates training loop to DonutTrainer from train.py instead of
reimplementing it.  DonutReceiptDataset replaced with SROIEDataset.from_samples()
from train.py, which uses the same target-sequence construction logic.

FIX: Added GPU memory cleanup after training.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from transformers import DonutProcessor, VisionEncoderDecoderModel

from constants import BASE_MODEL, MAX_LENGTH, NEW_TOKENS, SEED, _gpu_cleanup

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = BASE_MODEL
DATA_DIR = Path("data/donut")
OUTPUT_DIR = Path("models/donut_finetuned")
IMAGE_SIZE = (1280, 960)  # (height, width) — DONUT default
BATCH_SIZE = 8
GRAD_ACCUM = 2            # effective batch = 16
EPOCHS = 10
LR = 5e-5
WARMUP_RATIO = 0.1
TASK_TOKEN = "<s_sroie>"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class _TrainConfig:
    """Training configuration for standalone 02_train_donut.py."""
    max_epochs: int = EPOCHS
    learning_rate: float = LR
    per_device_train_batch_size: int = BATCH_SIZE
    early_stopping_patience: int = 3
    warmup_steps: int = 0  # set at train time based on dataset size
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = GRAD_ACCUM
    seed: int = SEED
    output_dir: str = str(OUTPUT_DIR / "best")


# ── Training ─────────────────────────────────────────────────────────────────
def train():
    from train import DonutTrainer, SROIEDataset

    print(f"Loading DONUT model: {MODEL_ID}")
    processor = DonutProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

    # Add task-specific tokens
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": NEW_TOKENS}
    )
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # FIX: After resize_token_embeddings(), lm_head and embed_tokens are
    # separate tensors.  Set tie_word_embeddings=False so save_pretrained()
    # saves BOTH weights independently.  Without this, reloading the
    # checkpoint produces garbage output (F1=0).
    model.decoder.config.tie_word_embeddings = False

    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        TASK_TOKEN
    )
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.max_length = MAX_LENGTH

    # Partially freeze encoder (freeze bottom 2 Swin stages)
    for name, param in model.encoder.named_parameters():
        if "layers.0" in name or "layers.1" in name:
            param.requires_grad = False

    # Load data from metadata.jsonl and convert to (Path, gt_dict) tuples
    # for use with SROIEDataset.from_samples().
    def _load_samples(data_dir: Path):
        meta_path = data_dir / "metadata.jsonl"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.jsonl not found at {meta_path}. "
                "Run 01_dataset_preparation.py first."
            )
        samples = []
        with open(meta_path) as fh:
            for line in fh:
                rec = json.loads(line)
                gt = json.loads(rec["ground_truth"])["gt_parse"]
                samples.append((data_dir / rec["file_name"], gt))
        return samples

    train_samples = _load_samples(DATA_DIR / "train")
    val_samples = _load_samples(DATA_DIR / "val")

    if not train_samples:
        raise ValueError("Training dataset is empty — check data paths.")

    train_ds = SROIEDataset.from_samples(processor, train_samples, max_length=MAX_LENGTH)
    val_ds = SROIEDataset.from_samples(processor, val_samples, max_length=MAX_LENGTH)

    # Compute absolute warmup steps for DonutTrainer
    steps_per_epoch = max(1, len(train_ds) // (BATCH_SIZE * GRAD_ACCUM))
    computed_warmup = int(steps_per_epoch * EPOCHS * WARMUP_RATIO)

    config = _TrainConfig(warmup_steps=computed_warmup)
    trainer = DonutTrainer(
        config=config,
        processor=processor,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    result = trainer.train()
    trainer.save(OUTPUT_DIR / "best")

    history = {
        "train_loss": [h.get("loss") for h in result.log_history if "loss" in h],
        "val_loss": [h.get("eval_loss") for h in result.log_history if "eval_loss" in h],
    }
    with open(OUTPUT_DIR / "training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    print("\nDONUT training complete.")

    # FIX: GPU cleanup after training to free VRAM for subsequent stages.
    _gpu_cleanup(model, processor, trainer, train_ds, val_ds)

    return history


if __name__ == "__main__":
    train()
