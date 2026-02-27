# MIT License
#
# Copyright (c) 2024
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
train.py — DONUT SROIE training with OOP interface and legacy compat.

WARNING: This is a legacy standalone script. For the full 8-experiment
pipeline, use: python run_all.py
This script is kept for backward compatibility and ad-hoc single-model
training/evaluation outside the experiment framework.

Critical bug fixes in this version:
  - Sets config.tie_word_embeddings=False after resize_token_embeddings()
    so that lm_head and embed_tokens are saved independently (prevents
    F1=0 on reload caused by tie_weights() destroying learned lm_head).
  - Key file loading: try .txt first, then .json (BUG A/E fix).
  - Zero-sample guard: raises ValueError if training dataset is empty.

Classes
-------
DonutTrainer
    OOP wrapper around Seq2SeqTrainer.  Gets ALL hyperparameters from an
    ExperimentConfig object (no hardcoded epochs / lr / batch size).

TrainingResult
    Dataclass returned by DonutTrainer.train().

SROIEDataset
    PyTorch Dataset that loads SROIE receipt images + key files.

ExperimentConfig will be imported from run_experiments when used via the
pipeline.  For standalone usage, main() defines defaults that match
TRAIN_CONFIG.
"""

import json
import logging
import multiprocessing
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — imported from shared constants.py (eliminates 5x duplication)
# ---------------------------------------------------------------------------

# FIX: Previously FIELDS, MAX_LENGTH, IMAGE_EXTS, NEW_TOKENS, BASE_MODEL,
# SEED were defined independently here and in 4 other files, risking silent
# drift if any file was updated without updating the others.
from constants import FIELDS, MAX_LENGTH, IMAGE_EXTS, NEW_TOKENS, BASE_MODEL, SEED


# BUG C FIX: Use a function so the env var is re-read at call time, not cached
# at import time (run_all.py sets SROIE_DATA_DIR after this module is imported).
def _get_sroie_dir():
    return Path(os.environ.get(
        "SROIE_DATA_DIR",
        "/workspace/ICDAR-2019-SROIE/data",
    ))


def _optimal_num_workers() -> int:
    """Calculate optimal DataLoader num_workers from available CPU cores."""
    num_cpus = multiprocessing.cpu_count()
    return min(8, max(4, num_cpus // 2))


# ---------------------------------------------------------------------------
# TrainingResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Holds outputs from a single training run."""

    log_history: List[Dict[str, Any]] = field(default_factory=list)
    train_samples: int = 0
    val_samples: int = 0
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# SROIEDataset
# ---------------------------------------------------------------------------

class SROIEDataset(Dataset):
    """PyTorch Dataset for SROIE receipt images with ground-truth key files.

    Parameters
    ----------
    processor : DonutProcessor
        HuggingFace processor for DONUT image/text encoding.
    img_dir : str or Path
        Directory containing receipt images.
    key_dir : str or Path
        Directory containing ground-truth key files (.txt or .json).
    max_length : int
        Maximum token length for the decoder target sequence.
    """

    def __init__(
        self,
        processor: DonutProcessor,
        img_dir,
        key_dir,
        max_length: int = MAX_LENGTH,
    ):
        self.processor = processor
        self.max_length = max_length
        self.samples: List[Tuple[Path, Dict[str, str]]] = []

        img_dir = Path(img_dir)
        key_dir = Path(key_dir)

        for img_path in sorted(
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ):
            gt = self._load_ground_truth(key_dir, img_path.stem)
            if gt is not None:
                self.samples.append((img_path, gt))

        print(f"Loaded {len(self.samples)} samples")

    # ------------------------------------------------------------------
    # Ground-truth loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_ground_truth(
        key_dir: Path,
        stem: str,
    ) -> Optional[Dict[str, str]]:
        """Load ground-truth dict from a key directory.

        BUG A/E FIX: Try .txt first (canonical 4-line SROIE format), then
        fall back to .json for pre-converted datasets.
        """
        # Try .txt first (native SROIE format)
        key_txt = key_dir / (stem + ".txt")
        if key_txt.exists():
            gt = SROIEDataset._parse_txt_key(key_txt)
            if gt is not None:
                return gt

        # Fall back to .json
        key_json = key_dir / (stem + ".json")
        if key_json.exists():
            try:
                return json.loads(key_json.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        return None

    @staticmethod
    def _parse_txt_key(path: Path) -> Optional[Dict[str, str]]:
        """Parse a 4-line SROIE key file into a dict."""
        try:
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        except UnicodeDecodeError:
            return None
        if len(lines) < 4:
            return None
        return {
            "company": lines[0].strip(),
            "date":    lines[1].strip(),
            "address": lines[2].strip(),
            "total":   lines[3].strip(),
        }

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_samples(
        cls,
        processor: DonutProcessor,
        samples: List[Tuple[Path, Dict[str, str]]],
        max_length: int = MAX_LENGTH,
    ) -> "SROIEDataset":
        """Create a SROIEDataset from a pre-built list of (path, gt) tuples."""
        obj = cls.__new__(cls)
        Dataset.__init__(obj)
        obj.processor = processor
        obj.max_length = max_length
        obj.samples = list(samples)
        return obj

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, gt = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        target = "<s_sroie>"
        for f in FIELDS:
            v = gt.get(f, "")
            target += f"<s_{f}>{v}</s_{f}>"
        target += "</s_sroie>"

        pixel_values = self.processor(
            image, return_tensors="pt",
        ).pixel_values.squeeze()
        labels = self.processor.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# DonutTrainer
# ---------------------------------------------------------------------------

class DonutTrainer:
    """OOP wrapper around HuggingFace Seq2SeqTrainer for DONUT fine-tuning.

    All hyperparameters come from a config object (typically an
    ``ExperimentConfig`` from ``run_experiments.py``).  This avoids
    duplicating magic numbers across files.

    Parameters
    ----------
    config : object
        An object with the following attributes (duck-typed):
        - ``max_epochs`` (int)
        - ``learning_rate`` (float)
        - ``per_device_train_batch_size`` (int)
        - ``early_stopping_patience`` (int)
        - ``warmup_steps`` (int, optional — default 100)
        - ``weight_decay`` (float, optional — default 0.01)
        - ``seed`` (int, optional — default 42)
        - ``output_dir`` (str or Path)
    processor : DonutProcessor
        Tokenizer + image processor.
    model : VisionEncoderDecoderModel
        The DONUT model to fine-tune.
    train_dataset : Dataset
        Training dataset.
    val_dataset : Dataset or None
        Validation dataset (enables early stopping + eval).

    Raises
    ------
    ValueError
        If ``train_dataset`` is empty (zero samples).
    """

    def __init__(
        self,
        config,
        processor: DonutProcessor,
        model: VisionEncoderDecoderModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        if len(train_dataset) == 0:
            raise ValueError(
                "Training dataset is empty (0 samples). Cannot train on an "
                "empty dataset — check data paths and loaders."
            )

        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self) -> TrainingResult:
        """Run training and return a :class:`TrainingResult`.

        Internally builds ``Seq2SeqTrainingArguments`` from ``self.config``
        and delegates to HuggingFace ``Seq2SeqTrainer``.
        """
        start_time = time.time()

        do_eval = self.val_dataset is not None and len(self.val_dataset) > 0
        optimal_workers = _optimal_num_workers()

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self._output_dir),
            num_train_epochs=self.config.max_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=getattr(self.config, "warmup_steps", 100),
            weight_decay=getattr(self.config, "weight_decay", 0.01),
            save_strategy="epoch",
            eval_strategy="epoch" if do_eval else "no",
            save_total_limit=3,
            load_best_model_at_end=do_eval,
            metric_for_best_model="eval_loss" if do_eval else None,
            greater_is_better=False if do_eval else None,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_steps=20,
            # PERFORMANCE: Optimized DataLoader settings
            dataloader_num_workers=optimal_workers,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=4 if optimal_workers > 0 else None,
            dataloader_persistent_workers=True if optimal_workers > 0 else False,
            remove_unused_columns=False,
            seed=getattr(self.config, "seed", SEED),
        )

        callbacks = []
        if do_eval:
            patience = getattr(
                self.config, "early_stopping_patience", 5,
            )
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=patience,
            ))

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            callbacks=callbacks or None,
        )

        trainer.train()

        duration = time.time() - start_time
        return TrainingResult(
            log_history=trainer.state.log_history,
            train_samples=len(self.train_dataset),
            val_samples=len(self.val_dataset) if self.val_dataset else 0,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

 def save(self, path: Optional[Path] = None) -> None:
    save_dir = Path(path) if path is not None else self._output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── lm_head detach fix ──────────────────────────────────────────────
    # load_best_model_at_end re-loads the best epoch checkpoint via
    # from_pretrained().  Per-epoch checkpoints may not have serialized
    # lm_head independently (even with tie_word_embeddings=False) because
    # safetensors deduplicates tensors sharing a data pointer.  After
    # resize_token_embeddings(), base-vocab rows can still share a pointer
    # with embed_tokens.  We force a deep copy so save_pretrained() writes
    # lm_head as a fully independent tensor with no shared pointer.
    import copy
    decoder = self.model.decoder
    if hasattr(decoder, "lm_head"):
        decoder.lm_head.weight = torch.nn.Parameter(
            decoder.lm_head.weight.data.clone()
        )
    # ────────────────────────────────────────────────────────────────────

    self.model.save_pretrained(str(save_dir))
    self.processor.save_pretrained(str(save_dir))
    logger.info("Model + processor saved → %s", save_dir)
    print(f"Model + processor saved → {save_dir}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _output_dir(self) -> Path:
        """Resolve the output directory from config."""
        return Path(getattr(self.config, "output_dir", "/workspace/donut-sroie-finetuned"))


# ---------------------------------------------------------------------------
# Lightweight config for standalone main()
# ---------------------------------------------------------------------------

@dataclass
class _StandaloneConfig:
    """Minimal config matching TRAIN_CONFIG defaults for legacy main()."""

    max_epochs: int = 30
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 4
    early_stopping_patience: int = 5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    seed: int = SEED
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Legacy standalone entry point
# ---------------------------------------------------------------------------

def main():
    """Standalone training entry point (legacy).

    Uses ``DonutTrainer`` internally with default hyperparameters that
    match ``TRAIN_CONFIG`` from ``run_experiments.py``.
    """
    from transformers import set_seed as _set_seed
    _set_seed(SEED)

    processor = DonutProcessor.from_pretrained(BASE_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": NEW_TOKENS},
    )
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    # After resize, embed_tokens and lm_head are separate tensors with
    # independent random init for the new tokens.  Set tie_word_embeddings=False
    # so save_pretrained() saves BOTH weights independently.  Without this,
    # the saved checkpoint omits lm_head (or tie_weights() overwrites the
    # learned lm_head with embed_tokens), causing F1=0 on reload.
    model.decoder.config.tie_word_embeddings = False

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = (
        processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    )
    model.gradient_checkpointing_enable()

    # Load SROIE data
    sroie_dir = _get_sroie_dir()
    full_ds = SROIEDataset(processor, sroie_dir / "img", sroie_dir / "key")
    all_samples = full_ds.samples

    # Use last 15 % of samples as validation for early stopping
    shuffled = list(all_samples)
    random.seed(SEED)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.15))
    train_samples = shuffled[n_val:]
    val_samples = shuffled[:n_val]

    train_ds = SROIEDataset.from_samples(processor, train_samples)
    val_ds = SROIEDataset.from_samples(processor, val_samples)

    workspace = os.environ.get("DONUT_WORKSPACE", "/workspace")
    output_dir = os.path.join(workspace, "donut-sroie-finetuned")

    config = _StandaloneConfig(output_dir=output_dir)
    trainer = DonutTrainer(
        config=config,
        processor=processor,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )

    result = trainer.train()
    trainer.save()

    print(f"TRAINING_COMPLETE  (duration={result.duration_seconds:.1f}s, "
          f"train={result.train_samples}, val={result.val_samples})")


if __name__ == "__main__":
    main()
