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
  - INDENTATION FIX: save() and _output_dir were indented at 1 space
    (module level) instead of 4 spaces (class body), causing an
    IndentationError on import that crashed ALL 8 DONUT experiments.

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

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    VisionEncoderDecoderModel,
)

# FIX: Previously FIELDS, MAX_LENGTH, IMAGE_EXTS, NEW_TOKENS, BASE_MODEL,
# SEED were defined independently here and in 4 other files, risking silent
# drift if any file was updated without updating the others.
from constants import (
    BASE_MODEL,
    FIELDS,
    MAX_LENGTH,
    NEW_TOKENS,
    SEED,
    _mask_empty_field_labels,
    _optimal_num_workers,
)

__all__ = ["SROIEDataset", "DonutTrainer", "TrainingResult"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LmHeadCloneCallback — prevent safetensors from deduplicating lm_head
# ---------------------------------------------------------------------------


class LmHeadCloneCallback(TrainerCallback):
    """Clone lm_head.weight before every checkpoint save.

    ROOT CAUSE OF F1~0.42:
    After ``resize_token_embeddings()``, ``lm_head.weight`` and
    ``embed_tokens.weight`` share a data pointer for the base-vocab rows.
    ``safetensors`` deduplicates tensors that share a data pointer, so the
    per-epoch checkpoint shard omits ``lm_head.weight`` entirely.  When
    ``Seq2SeqTrainer`` with ``load_best_model_at_end=True`` reloads the best
    epoch checkpoint, ``lm_head.weight`` is randomly re-initialized (missing
    key = random init), collapsing F1 to ~0.42.

    Fix: Force a deep clone of ``lm_head.weight.data`` before every save so
    safetensors sees it as a fully independent tensor and writes it to the
    shard.  This ensures the reloaded checkpoint always has the trained
    lm_head weights.
    """

    def on_save(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            return control
        lm_head = getattr(decoder, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
            logger.debug(
                "LmHeadCloneCallback.on_save: cloned lm_head.weight (epoch %s)",
                state.epoch,
            )
        return control


# ---------------------------------------------------------------------------
# Constants — imported from shared constants.py (eliminates 5x duplication)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TrainingResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Holds outputs from a single training run."""

    log_history: list[dict[str, Any]] = field(default_factory=list)
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
    samples : list of (Path, dict)
        Pre-built list of (image_path, ground_truth_dict) tuples, as
        returned by ``dataset_loaders.load_sroie_train()`` etc.
    max_length : int
        Maximum token length for the decoder target sequence.
    """

    def __init__(
        self,
        processor: DonutProcessor,
        samples: list[tuple[Path, dict[str, str]]],
        max_length: int = MAX_LENGTH,
    ):
        super().__init__()
        self.processor = processor
        self.max_length = max_length
        self.samples = list(samples)
        # Log per-field masking statistics so empty-field dilution is visible.
        if samples:
            for f in FIELDS:
                n = sum(1 for _, gt in samples if not gt.get(f, "").strip())
                if n:
                    logger.info(
                        "Field masking: %d/%d samples will have <%s> masked (%.1f%%)",
                        n, len(samples), f, 100.0 * n / len(samples),
                    )

    # ------------------------------------------------------------------
    # Backward-compatible alternate constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_samples(
        cls,
        processor: DonutProcessor,
        samples: list[tuple[Path, dict[str, str]]],
        max_length: int = MAX_LENGTH,
    ) -> "SROIEDataset":
        """Backward-compatible alias for ``SROIEDataset(processor, samples, max_length)``."""
        return cls(processor, samples, max_length)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, gt = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        target = "<s_sroie>"
        for f in FIELDS:
            v = gt.get(f, "")
            target += f"<s_{f}>{v}</s_{f}>"
        target += "</s_sroie>"

        pixel_values = self.processor(
            image,
            return_tensors="pt",
        ).pixel_values.squeeze()
        labels = self.processor.tokenizer(
            target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # Mask empty-field spans so they contribute no gradient to the loss.
        labels = _mask_empty_field_labels(labels, gt, self.processor.tokenizer)
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
        val_dataset: Dataset | None = None,
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

        # Detect bf16 support (Ampere+ GPUs including Blackwell) — prefer over fp16
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16

        # Cap warmup_steps to ≤10% of total optimizer steps.
        # warmup=500 is correct for large datasets (Exp 8, ~3940 samples, ~1250 opt steps),
        # but exceeds total training for small datasets (Exp 1, ~500 samples, ~320 opt steps),
        # causing the LR to never reach peak value and producing CORD-schema hallucinations.
        _grad_accum = getattr(self.config, "gradient_accumulation_steps", 2)
        _steps_epoch = math.ceil(len(self.train_dataset) / self.config.per_device_train_batch_size)
        _total_opt_steps = math.ceil(_steps_epoch / _grad_accum) * self.config.max_epochs
        _cfg_warmup = getattr(self.config, "warmup_steps", 100)
        _eff_warmup = min(_cfg_warmup, max(10, _total_opt_steps // 10))
        if _eff_warmup != _cfg_warmup:
            logger.warning(
                "warmup_steps capped %d → %d (dataset has only %d opt steps over %d epochs)",
                _cfg_warmup, _eff_warmup, _total_opt_steps, self.config.max_epochs,
            )

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self._output_dir),
            num_train_epochs=self.config.max_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=_grad_accum,
            learning_rate=self.config.learning_rate,
            warmup_steps=_eff_warmup,
            weight_decay=getattr(self.config, "weight_decay", 0.01),
            save_strategy="epoch",
            eval_strategy="epoch" if do_eval else "no",
            save_total_limit=3,
            load_best_model_at_end=do_eval,
            metric_for_best_model="eval_loss" if do_eval else None,
            greater_is_better=False if do_eval else None,
            predict_with_generate=True,
            bf16=use_bf16,
            fp16=use_fp16,
            logging_steps=20,
            # PERFORMANCE: Optimized DataLoader settings
            dataloader_num_workers=optimal_workers,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=4 if optimal_workers > 0 else None,
            dataloader_persistent_workers=optimal_workers > 0,
            remove_unused_columns=False,
            seed=getattr(self.config, "seed", SEED),
        )

        # LmHeadCloneCallback MUST be registered before EarlyStoppingCallback
        # so the clone happens before every checkpoint write (including the
        # best-model checkpoint that load_best_model_at_end reloads).
        callbacks = [LmHeadCloneCallback()]
        if do_eval:
            patience = getattr(
                self.config,
                "early_stopping_patience",
                5,
            )
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                )
            )

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

    def save(self, path: Path | None = None) -> None:
        """Save the fine-tuned model and processor to disk.

        INDENTATION FIX: this method was previously indented with 1 space
        instead of 4, making Python treat it as module-level code and raising
        an IndentationError on import — which crashed all 8 DONUT experiments.
        """
        save_dir = Path(path) if path is not None else self._output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── lm_head detach fix ─────────────────────────────────────────────
        # load_best_model_at_end re-loads the best epoch checkpoint via
        # from_pretrained().  Per-epoch checkpoints may not have serialized
        # lm_head independently (even with tie_word_embeddings=False) because
        # safetensors deduplicates tensors sharing a data pointer.  After
        # resize_token_embeddings(), base-vocab rows can still share a pointer
        # with embed_tokens.  We force a deep copy so save_pretrained() writes
        # lm_head as a fully independent tensor with no shared pointer.
        decoder = self.model.decoder
        if hasattr(decoder, "lm_head"):
            decoder.lm_head.weight = torch.nn.Parameter(decoder.lm_head.weight.data.clone())
        # ──────────────────────────────────────────────────────────────────

        self.model.save_pretrained(str(save_dir))
        self.processor.save_pretrained(str(save_dir))
        logger.info("Model + processor saved → %s", save_dir)

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
    per_device_train_batch_size: int = 16
    early_stopping_patience: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
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
    from constants import set_seed

    set_seed(SEED)

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
    model.decoder.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[
        0
    ]
    model.decoder.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[
        0
    ]
    model.config.use_cache = False  # Required with gradient_checkpointing
    model.decoder.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Load SROIE data using canonical loaders (single source of truth)
    from dataset_loaders import load_sroie_train, load_sroie_val

    train_samples = load_sroie_train()
    val_samples = load_sroie_val()

    train_ds = SROIEDataset(processor, train_samples)
    val_ds = SROIEDataset(processor, val_samples) if val_samples else None

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

    logger.info(
        "TRAINING_COMPLETE  (duration=%.1fs, train=%d, val=%d)",
        result.duration_seconds,
        result.train_samples,
        result.val_samples,
    )


if __name__ == "__main__":
    main()
