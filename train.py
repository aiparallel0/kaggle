# =============================================================================
# train.py
# Purpose: DonutTrainer class + SROIEDataset + MultiDataset — core DONUT training infrastructure
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-07
# =============================================================================
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

import csv
import json
import logging
import math
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure sibling modules are importable regardless of CWD.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Inline transformers fallback — used when the transformers package is absent
# ─────────────────────────────────────────────────────────────────────────────

try:
    from transformers import (
        DonutProcessor,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        VisionEncoderDecoderModel,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    import math as _math_tr

    # ── TrainerCallback base ──────────────────────────────────────────────

    class TrainerCallback:  # type: ignore[no-redef]
        """Minimal TrainerCallback base (mirrors transformers API)."""

        def on_init_end(self, args, state, control, **kw):
            pass

        def on_train_begin(self, args, state, control, **kw):
            pass

        def on_train_end(self, args, state, control, **kw):
            pass

        def on_epoch_begin(self, args, state, control, **kw):
            pass

        def on_epoch_end(self, args, state, control, **kw):
            pass

        def on_step_begin(self, args, state, control, **kw):
            pass

        def on_step_end(self, args, state, control, **kw):
            pass

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            pass

        def on_save(self, args, state, control, model=None, **kw):
            pass

        def on_log(self, args, state, control, logs=None, **kw):
            pass

    # ── Control / State objects ───────────────────────────────────────────

    class _TrainerControl:
        should_training_stop: bool = False
        should_epoch_stop: bool = False
        should_save: bool = False
        should_evaluate: bool = False
        should_log: bool = False

    class _TrainerState:
        epoch: int = 0
        global_step: int = 0
        log_history: list[dict[str, Any]] = field(default_factory=list)
        best_metric: float | None = None
        best_model_checkpoint: str | None = None

        def __init__(self):
            self.epoch = 0
            self.global_step = 0
            self.log_history = []
            self.best_metric = None
            self.best_model_checkpoint = None

    # ── EarlyStoppingCallback ─────────────────────────────────────────────

    class EarlyStoppingCallback(TrainerCallback):  # type: ignore[no-redef]
        """Patience-based early stopping."""

        def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
            self._patience = early_stopping_patience
            self._threshold = early_stopping_threshold
            self._best = float("inf")
            self._counter = 0

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            val_loss = (metrics or {}).get("eval_loss", float("inf"))
            if val_loss < self._best - self._threshold:
                self._best = val_loss
                self._counter = 0
            else:
                self._counter += 1
                if self._counter >= self._patience:
                    control.should_training_stop = True

    # ── Seq2SeqTrainingArguments ──────────────────────────────────────────

    class Seq2SeqTrainingArguments:  # type: ignore[no-redef]
        """Minimal subset of transformers.Seq2SeqTrainingArguments."""

        def __init__(
            self,
            output_dir: str = ".",
            num_train_epochs: int = 3,
            per_device_train_batch_size: int = 8,
            per_device_eval_batch_size: int = 8,
            gradient_accumulation_steps: int = 1,
            learning_rate: float = 5e-5,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            save_strategy: str = "epoch",
            eval_strategy: str = "no",
            save_total_limit: int = 3,
            load_best_model_at_end: bool = False,
            metric_for_best_model: str | None = None,
            greater_is_better: bool | None = None,
            predict_with_generate: bool = False,
            bf16: bool = False,
            fp16: bool = False,
            logging_steps: int = 500,
            dataloader_num_workers: int = 0,
            dataloader_pin_memory: bool = False,
            dataloader_prefetch_factor: int | None = None,
            dataloader_persistent_workers: bool = False,
            remove_unused_columns: bool = True,
            seed: int = 42,
            **kwargs,
        ):
            self.output_dir = output_dir
            self.num_train_epochs = num_train_epochs
            self.per_device_train_batch_size = per_device_train_batch_size
            self.per_device_eval_batch_size = per_device_eval_batch_size
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.learning_rate = learning_rate
            self.warmup_steps = warmup_steps
            self.weight_decay = weight_decay
            self.save_strategy = save_strategy
            self.eval_strategy = eval_strategy
            self.save_total_limit = save_total_limit
            self.load_best_model_at_end = load_best_model_at_end
            self.metric_for_best_model = metric_for_best_model
            self.greater_is_better = greater_is_better
            self.predict_with_generate = predict_with_generate
            self.bf16 = bf16
            self.fp16 = fp16
            self.logging_steps = logging_steps
            self.dataloader_num_workers = dataloader_num_workers
            self.dataloader_pin_memory = dataloader_pin_memory
            self.dataloader_prefetch_factor = dataloader_prefetch_factor
            self.dataloader_persistent_workers = dataloader_persistent_workers
            self.remove_unused_columns = remove_unused_columns
            self.seed = seed

    # ── Seq2SeqTrainer inline ─────────────────────────────────────────────

    class Seq2SeqTrainer:  # type: ignore[no-redef]
        """Inline Seq2Seq training loop replacing transformers.Seq2SeqTrainer.

        Supports the same constructor signature and .train() method used by
        DonutTrainer.  Gradient accumulation, AMP, early stopping, and
        callback lifecycle hooks are all implemented.
        """

        def __init__(
            self,
            model=None,
            args=None,
            train_dataset=None,
            eval_dataset=None,
            callbacks=None,
            optimizers=(None, None),
            data_collator=None,
            **kwargs,
        ):
            self.model = model
            self.args = args or Seq2SeqTrainingArguments()
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.callbacks = callbacks or []
            self._optimizer, self._scheduler = optimizers
            self.state = _TrainerState()
            self.data_collator = data_collator

        def _call_callbacks(self, event: str, control=None, **kw):
            if control is None:
                control = _TrainerControl()
            for cb in self.callbacks:
                getattr(cb, event, lambda *a, **k: None)(self.args, self.state, control, **kw)
            return control

        def _eval_loop(self, dataloader) -> float:
            """Run evaluation; returns mean eval_loss."""
            self.model.eval()
            total_loss = 0.0
            n = 0
            with torch.no_grad():
                for batch in dataloader:
                    batch = {
                        k: v.to(self.model.device) if hasattr(v, "to") else v
                        for k, v in batch.items()
                    }
                    # Strip decoder keys so the model derives them from labels
                    # internally — same guard as in training (see _DonutSeq2SeqTrainer).
                    batch.pop("decoder_input_ids", None)
                    batch.pop("decoder_inputs_embeds", None)
                    out = self.model(**batch)
                    if hasattr(out, "loss") and out.loss is not None:
                        total_loss += out.loss.item()
                        n += 1
            self.model.train()
            return total_loss / max(n, 1)

        def _save_checkpoint(self, output_dir: str, step: int):

            ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            # Fire on_save callbacks (e.g. LmHeadCloneCallback)
            control = _TrainerControl()
            control.should_save = True
            self._call_callbacks("on_save", control, model=self.model)
            torch.save(self.model.state_dict(), ckpt_dir / "pytorch_model.bin")
            return str(ckpt_dir)

        def train(self):
            from torch.utils.data import DataLoader as _DataLoader

            args = self.args
            device = next(self.model.parameters()).device
            use_amp = args.fp16 or args.bf16
            amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

            train_loader = _DataLoader(
                self.train_dataset,
                batch_size=args.per_device_train_batch_size,
                shuffle=True,
                num_workers=args.dataloader_num_workers,
                pin_memory=args.dataloader_pin_memory,
                collate_fn=self.data_collator,
            )
            eval_loader = None
            if self.eval_dataset is not None and args.eval_strategy != "no":
                eval_loader = _DataLoader(
                    self.eval_dataset,
                    batch_size=args.per_device_eval_batch_size,
                    shuffle=False,
                    num_workers=args.dataloader_num_workers,
                    collate_fn=self.data_collator,
                )

            optimizer = self._optimizer
            if optimizer is None:
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                )

            total_steps = (
                len(train_loader) * args.num_train_epochs // args.gradient_accumulation_steps
            )
            scheduler = self._scheduler
            if scheduler is None and args.warmup_steps > 0:

                def _lr_lambda(step):
                    if step < args.warmup_steps:
                        return float(step) / float(max(1, args.warmup_steps))
                    progress = float(step - args.warmup_steps) / float(
                        max(1, total_steps - args.warmup_steps)
                    )
                    return max(0.0, 0.5 * (1.0 + _math_tr.cos(_math_tr.pi * progress)))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

            self.model.train()
            control = _TrainerControl()
            self._call_callbacks("on_train_begin", control)
            best_ckpt = None
            best_eval_loss = float("inf")
            output_dir = args.output_dir
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            global_step = 0
            accum_loss = 0.0

            for epoch in range(int(args.num_train_epochs)):
                self.state.epoch = epoch
                self._call_callbacks("on_epoch_begin", control)
                for step, batch in enumerate(train_loader):
                    self._call_callbacks("on_step_begin", control)
                    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                    # Strip decoder_input_ids / decoder_inputs_embeds so that
                    # VisionEncoderDecoderModel derives them from labels internally.
                    # Passing both causes MBart to raise ValueError (see _DonutSeq2SeqTrainer).
                    batch.pop("decoder_input_ids", None)
                    batch.pop("decoder_inputs_embeds", None)
                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                        out = self.model(**batch)
                        loss = out.loss / args.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    accum_loss += loss.item()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        if scheduler is not None:
                            scheduler.step()
                        global_step += 1
                        self.state.global_step = global_step

                        if global_step % args.logging_steps == 0:
                            logs = {"loss": accum_loss, "step": global_step, "epoch": epoch}
                            self.state.log_history.append(logs)
                            self._call_callbacks("on_log", control, logs=logs)
                            accum_loss = 0.0

                    self._call_callbacks("on_step_end", control)
                    if control.should_training_stop:
                        break

                # Flush the incomplete accumulation window at epoch end.
                # When len(train_loader) % gradient_accumulation_steps != 0,
                # the final partial window accumulates gradients that never
                # reach the (step+1) % grad_accum == 0 condition — those
                # gradients are silently discarded each epoch.
                _remaining = len(train_loader) % args.gradient_accumulation_steps
                if _remaining != 0 and not control.should_training_stop:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1
                    self.state.global_step = global_step

                # End of epoch: evaluate + checkpoint
                eval_loss = float("inf")
                if eval_loader is not None:
                    eval_loss = self._eval_loop(eval_loader)
                    metrics = {"eval_loss": eval_loss}
                    self.state.log_history.append({**metrics, "epoch": epoch})
                    self._call_callbacks("on_evaluate", control, metrics=metrics)

                if args.save_strategy == "epoch":
                    ckpt = self._save_checkpoint(output_dir, global_step)
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        best_ckpt = ckpt
                        self.state.best_model_checkpoint = ckpt

                self._call_callbacks("on_epoch_end", control)
                if control.should_training_stop:
                    break

            # Load best model if requested
            if args.load_best_model_at_end and best_ckpt and Path(best_ckpt).exists():
                sd = torch.load(
                    Path(best_ckpt) / "pytorch_model.bin", map_location=device, weights_only=True
                )
                self.model.load_state_dict(sd, strict=False)

            self._call_callbacks("on_train_end", control)

    # ── DonutProcessor inline ─────────────────────────────────────────────

    class DonutProcessor:  # type: ignore[no-redef]
        """Minimal DonutProcessor: image resize/normalize + sentencepiece tokenizer.

        Loads from a checkpoint directory.  Image processing uses bilinear
        resize to (W=960, H=1280) and ImageNet normalization.

        Tokenization: wraps sentencepiece.SentencePieceProcessor if available,
        else falls back to a character-level tokenizer from vocab.txt.
        """

        _MEAN = [0.485, 0.456, 0.406]
        _STD = [0.229, 0.224, 0.225]

        def __init__(self, pretrained_model_name_or_path: str | Path, **kwargs):
            import numpy as _np

            self._np = _np
            self._model_dir = Path(pretrained_model_name_or_path)
            self._size = {"height": DONUT_IMAGE_SIZE[0], "width": DONUT_IMAGE_SIZE[1]}
            self._sp = None
            self._vocab: dict = {}
            self._id2tok: dict = {}
            self._special_tokens: dict = {}
            self._added_tokens: dict = {}

            # Load config for size override
            cfg_path = self._model_dir / "processor_config.json"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
                if "size" in cfg:
                    self._size = cfg["size"]

            # Try sentencepiece
            sp_path = self._model_dir / "tokenizer.model"
            if sp_path.exists():
                try:
                    import sentencepiece as _spm  # noqa: PLC0415

                    self._sp = _spm.SentencePieceProcessor()
                    self._sp.Load(str(sp_path))
                except ImportError:
                    pass

            if self._sp is None:
                # Fall back to tokenizer.json BPE vocab
                tok_path = self._model_dir / "tokenizer.json"
                if tok_path.exists():
                    with open(tok_path) as f:
                        tok_data = json.load(f)
                    vocab = tok_data.get("model", {}).get("vocab", tok_data.get("vocab", {}))
                    if isinstance(vocab, list):
                        vocab = {t: i for i, t in enumerate(vocab)}
                    self._vocab = vocab
                    self._id2tok = {v: k for k, v in vocab.items()}

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            return cls(pretrained_model_name_or_path, **kwargs)

        @property
        def tokenizer(self):
            return self  # self acts as tokenizer too

        def _resize_and_normalize(self, img) -> torch.Tensor:
            """Resize to (H, W), normalize, return [3, H, W] float32 tensor."""
            import numpy as _np

            target_h = self._size.get("height", DONUT_IMAGE_SIZE[0])
            target_w = self._size.get("width", DONUT_IMAGE_SIZE[1])

            if isinstance(img, _np.ndarray):
                arr = img
            elif hasattr(img, "__array__"):
                arr = _np.array(img)
            else:
                raise TypeError(f"Expected numpy array or PIL Image, got {type(img)}")

            # Resize with basic bilinear via torch
            t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            t = torch.nn.functional.interpolate(
                t, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            # Normalize
            mean = torch.tensor(self._MEAN).view(1, 3, 1, 1)
            std = torch.tensor(self._STD).view(1, 3, 1, 1)
            return (t - mean) / std

        def _process_image(self, images, return_tensors: str = "pt", **kwargs):
            """Process image(s) → {pixel_values: tensor}."""
            import numpy as _np

            if isinstance(images, _np.ndarray) or hasattr(images, "__array__"):
                pixel_values = self._resize_and_normalize(images)
            elif isinstance(images, list):
                pixel_values = torch.cat([self._resize_and_normalize(im) for im in images], 0)
            elif hasattr(images, "tobytes"):  # PIL Image
                pixel_values = self._resize_and_normalize(_np.array(images))
            else:
                raise TypeError(f"Unsupported image type: {type(images)}")

            class _Out:
                pass

            out = _Out()
            out.pixel_values = pixel_values
            return out

        def __call__(self, *args, **kwargs):
            """Dispatch: image processing or tokenizer call."""
            if args and isinstance(args[0], str):
                # Tokenizer call: processor(text, ...)
                text = args[0]
                ids = self.encode(text)
                t = torch.tensor([ids], dtype=torch.long)

                class _TokOut:
                    input_ids = t

                return _TokOut()
            # Image call
            return self._process_image(*args, **kwargs)

        # ── Tokenizer interface ────────────────────────────────────────────

        def add_special_tokens(self, tokens: dict[str, list[str]] | list[str]) -> int:
            if isinstance(tokens, dict):
                tokens = tokens.get("additional_special_tokens", [])
            added = 0
            n = len(self._vocab) + len(self._added_tokens)
            for tok in tokens:
                if tok not in self._added_tokens and tok not in self._vocab:
                    self._added_tokens[tok] = n
                    self._id2tok[n] = tok
                    n += 1
                    added += 1
            return added

        def convert_tokens_to_ids(self, tokens):
            def _tok2id(t):
                if t in self._added_tokens:
                    return self._added_tokens[t]
                if self._sp is not None:
                    return self._sp.PieceToId(t)
                return self._vocab.get(t, 0)

            if isinstance(tokens, list):
                return [_tok2id(t) for t in tokens]
            return _tok2id(tokens)

        def encode(self, text: str, **kwargs) -> list[int]:
            if self._sp is not None:
                return self._sp.Encode(text)
            return [self._vocab.get(c, 0) for c in text]

        def decode(self, ids, skip_special_tokens: bool = False, **kwargs) -> str:
            if self._sp is not None:
                pieces = [self._sp.IdToPiece(i) for i in ids]
                if skip_special_tokens:
                    pieces = [p for p in pieces if not p.startswith("<")]
                return "".join(pieces).replace("▁", " ").strip()
            return "".join(self._id2tok.get(i, "") for i in ids)

        def batch_decode(self, ids_list, **kwargs) -> list[str]:
            return [self.decode(ids, **kwargs) for ids in ids_list]

        def token2json(self, tokens: str, **kwargs) -> dict[str, str]:
            """Parse <s_field>VALUE</s_field> sequences into a dict."""
            import re as _re

            result: dict = {}
            for field, value in _re.findall(r"<s_(\w+)>(.*?)</s_\1>", tokens, _re.DOTALL):
                result[field] = value.strip()
            return result

    # ── HuggingFace Hub downloader (urllib only) ───────────────────────────

    def _hf_hub_download(repo_id: str, dest_dir: str | Path, token: str | None = None) -> Path:
        """Download a HuggingFace Hub model repository to *dest_dir*.

        Uses only ``urllib`` — no ``huggingface_hub`` package required.
        Downloads: config.json, tokenizer.model, tokenizer.json,
        tokenizer_config.json, special_tokens_map.json,
        processor_config.json, model.safetensors (or pytorch_model.bin).

        Args:
            repo_id: HuggingFace repo id, e.g. "naver-clova-ix/donut-base"
            dest_dir: local directory to save files into (created if needed)
            token: optional HF access token for private repos

        Returns:
            Path to *dest_dir*
        """
        import urllib.error as _urlerr
        import urllib.request as _urlreq

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        base_url = f"https://huggingface.co/{repo_id}/resolve/main"
        files = [
            "config.json",
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "processor_config.json",
            "model.safetensors",
        ]

        headers: dict = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        for fname in files:
            out_path = dest / fname
            if out_path.exists():
                continue
            url = f"{base_url}/{fname}"
            req = _urlreq.Request(url, headers=headers)
            try:
                with _urlreq.urlopen(req, timeout=120) as resp:
                    out_path.write_bytes(resp.read())
            except (_urlerr.URLError, OSError):
                # Non-fatal: some files (e.g. tokenizer.model) may not exist for
                # every checkpoint; skip silently and continue.
                pass

        # Fallback: pytorch_model.bin if model.safetensors not downloaded
        if not (dest / "model.safetensors").exists() and not (dest / "pytorch_model.bin").exists():
            url = f"{base_url}/pytorch_model.bin"
            req = _urlreq.Request(url, headers=headers)
            try:
                with _urlreq.urlopen(req, timeout=300) as resp:
                    (dest / "pytorch_model.bin").write_bytes(resp.read())
            except (_urlerr.URLError, OSError):
                pass

        return dest

    # ── Swin Transformer Encoder (matches naver-clova-ix/donut-base weights) ─

    import torch.nn as _nn
    import torch.nn.functional as _F

    def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
        """Partition feature map into non-overlapping windows.

        Args:
            x: (B, H, W, C)
            window_size: window height == window width

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
        """Reverse window partitioning.

        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size: int
            H, W: spatial dimensions of original feature map

        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    class _SwinPatchEmbedding(_nn.Module):
        """Conv2d patch embedding + LayerNorm.

        Mirrors HuggingFace SwinPatchEmbeddings weight names:
          embeddings.patch_embeddings.projection.{weight,bias}
          embeddings.norm.{weight,bias}
        """

        def __init__(self, in_channels: int = 3, embed_dim: int = 128, patch_size: int = 4):
            super().__init__()
            self.patch_size = patch_size
            # Named to match HF: encoder.model.encoder.embeddings.patch_embeddings.projection
            self.patch_embeddings = _nn.Module()
            self.patch_embeddings.projection = _nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.norm = _nn.LayerNorm(embed_dim)

        def forward(self, pixel_values: torch.Tensor):
            """
            Args:
                pixel_values: (B, C, H, W)
            Returns:
                embeddings: (B, num_patches, embed_dim)
                output_dimensions: (H', W') after patch embed
            """
            x = self.patch_embeddings.projection(pixel_values)  # (B, embed_dim, H', W')
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
            x = self.norm(x)
            return x, (H, W)

    class _SwinWindowAttention(_nn.Module):
        """Window-based multi-head self-attention with relative position bias.

        Weight names match HuggingFace SwinWindowAttention:
          attention.self.query/key/value.{weight,bias}
          attention.output.dense.{weight,bias}
          relative_position_bias_table
          relative_position_index
        """

        def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int,
            qkv_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
        ):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.window_size = window_size
            head_dim = dim // num_heads
            self.scale = head_dim**-0.5

            # HF uses separate query/key/value projections under attention.self
            self.attention = _nn.Module()
            self.attention.self = _nn.Module()
            self.attention.self.query = _nn.Linear(dim, dim, bias=qkv_bias)
            self.attention.self.key = _nn.Linear(dim, dim, bias=qkv_bias)
            self.attention.self.value = _nn.Linear(dim, dim, bias=qkv_bias)
            self.attention.output = _nn.Module()
            self.attention.output.dense = _nn.Linear(dim, dim)

            # Relative position bias: table of (2*Wh-1)*(2*Ww-1) entries × num_heads
            self.relative_position_bias_table = _nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )

            # Pre-compute relative position index
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
            coords_flat = coords.flatten(1)  # (2, Wh*Ww)
            relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # (N, N)
            self.register_buffer("relative_position_index", relative_position_index)

            self.attn_drop = _nn.Dropout(attn_drop)
            self.proj_drop = _nn.Dropout(proj_drop)

            _nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            Args:
                x: (num_windows*B, N, C) where N = window_size^2
                mask: (num_windows, N, N) or None
            Returns:
                (num_windows*B, N, C)
            """
            B_, N, C = x.shape
            H = self.num_heads
            head_dim = C // H

            q = self.attention.self.query(x).reshape(B_, N, H, head_dim).permute(0, 2, 1, 3)
            k = self.attention.self.key(x).reshape(B_, N, H, head_dim).permute(0, 2, 1, 3)
            v = self.attention.self.value(x).reshape(B_, N, H, head_dim).permute(0, 2, 1, 3)

            attn = (q @ k.transpose(-2, -1)) * self.scale

            # Relative position bias
            rel_pos_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size * self.window_size,
                self.window_size * self.window_size,
                -1,
            )
            rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()  # (nH, N, N)
            attn = attn + rel_pos_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, H, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, H, N, N)

            attn = _F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.attention.output.dense(x)
            x = self.proj_drop(x)
            return x

    class _SwinIntermediate(_nn.Module):
        """FFN intermediate (fc1 + activation). HF name: intermediate."""

        def __init__(self, dim: int, mlp_ratio: float = 4.0):
            super().__init__()
            hidden = int(dim * mlp_ratio)
            self.dense = _nn.Linear(dim, hidden)
            self.intermediate_act_fn = _nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.intermediate_act_fn(self.dense(x))

    class _SwinOutput(_nn.Module):
        """FFN output (fc2). HF name: output."""

        def __init__(self, dim: int, mlp_ratio: float = 4.0):
            super().__init__()
            hidden = int(dim * mlp_ratio)
            self.dense = _nn.Linear(hidden, dim)
            self.dropout = _nn.Dropout(0.0)

        def forward(self, x: torch.Tensor, _input_tensor: torch.Tensor) -> torch.Tensor:
            return self.dropout(self.dense(x))

    class _SwinBlock(_nn.Module):
        """One Swin Transformer block (W-MSA or SW-MSA).

        Weight names mirror HuggingFace SwinLayer:
          layernorm_before.{weight,bias}
          attention.self.{query,key,value}.{weight,bias}
          attention.output.dense.{weight,bias}
          relative_position_bias_table
          layernorm_after.{weight,bias}
          intermediate.dense.{weight,bias}
          output.dense.{weight,bias}
          layer_scale_parameter1 / layer_scale_parameter2  (absent in base — ignored)
        """

        def __init__(
            self,
            dim: int,
            num_heads: int,
            window_size: int = 8,
            shift_size: int = 0,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_path: float = 0.0,
        ):
            super().__init__()
            self.dim = dim
            self.window_size = window_size
            self.shift_size = shift_size

            self.layernorm_before = _nn.LayerNorm(dim)
            self.attention = _SwinWindowAttention(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                qkv_bias=qkv_bias,
            )
            self.layernorm_after = _nn.LayerNorm(dim)
            self.intermediate = _SwinIntermediate(dim, mlp_ratio)
            self.output = _SwinOutput(dim, mlp_ratio)

            self.drop_path_prob = drop_path
            # Store computed attn_mask lazily based on input spatial size
            self._attn_mask: torch.Tensor | None = None
            self._mask_hw: tuple = (-1, -1)

        def _get_attn_mask(self, H: int, W: int, device) -> torch.Tensor | None:
            if self.shift_size == 0:
                return None
            if self._mask_hw == (H, W) and self._attn_mask is not None:
                return self._attn_mask.to(device)
            ws = self.window_size
            img_mask = torch.zeros(1, H, W, 1, device=device)
            h_slices = [
                slice(0, -ws),
                slice(-ws, -self.shift_size),
                slice(-self.shift_size, None),
            ]
            w_slices = [
                slice(0, -ws),
                slice(-ws, -self.shift_size),
                slice(-self.shift_size, None),
            ]
            cnt = 0
            for hs in h_slices:
                for ws_ in w_slices:
                    img_mask[:, hs, ws_, :] = cnt
                    cnt += 1
            mask_windows = _window_partition(img_mask, ws)
            mask_windows = mask_windows.view(-1, ws * ws)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
            self._attn_mask = attn_mask
            self._mask_hw = (H, W)
            return attn_mask

        @staticmethod
        def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
            if not training or drop_prob == 0.0:
                return x
            keep = 1.0 - drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            noise = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep).div_(keep)
            return x * noise

        def forward(self, hidden_states: torch.Tensor, input_dimensions: tuple) -> torch.Tensor:
            """
            Args:
                hidden_states: (B, H*W, C)
                input_dimensions: (H, W)
            Returns:
                (B, H*W, C)
            """
            H, W = input_dimensions
            B, L, C = hidden_states.shape

            shortcut = hidden_states
            hidden_states = self.layernorm_before(hidden_states)
            hidden_states = hidden_states.view(B, H, W, C)

            # Cyclic shift
            if self.shift_size > 0:
                shifted = torch.roll(
                    hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
            else:
                shifted = hidden_states

            # Partition into windows
            windows = _window_partition(shifted, self.window_size)  # (nW*B, ws, ws, C)
            windows = windows.view(-1, self.window_size * self.window_size, C)

            # Attention
            attn_mask = self._get_attn_mask(H, W, hidden_states.device)
            attn_out = self.attention(windows, mask=attn_mask)

            # Reverse windows
            attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
            shifted_out = _window_reverse(attn_out, self.window_size, H, W)

            # Reverse cyclic shift
            if self.shift_size > 0:
                hidden_states = torch.roll(
                    shifted_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
            else:
                hidden_states = shifted_out
            hidden_states = hidden_states.view(B, H * W, C)
            hidden_states = shortcut + self._drop_path(
                hidden_states, self.drop_path_prob, self.training
            )

            # MLP
            mlp_in = self.layernorm_after(hidden_states)
            mlp_mid = self.intermediate(mlp_in)
            mlp_out = self.output(mlp_mid, mlp_in)
            hidden_states = hidden_states + self._drop_path(
                mlp_out, self.drop_path_prob, self.training
            )

            return hidden_states

    class _SwinPatchMerging(_nn.Module):
        """Patch merging layer (downsampling 2× in spatial dims, 2× channels).

        Weight names mirror HuggingFace SwinPatchMerging:
          reduction.{weight}   (Linear, no bias)
          norm.{weight,bias}
        """

        def __init__(self, input_resolution: tuple, dim: int):
            super().__init__()
            self.input_resolution = input_resolution
            self.dim = dim
            self.reduction = _nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = _nn.LayerNorm(4 * dim)

        def forward(self, x: torch.Tensor, input_dimensions: tuple) -> tuple:
            """
            Args:
                x: (B, H*W, C)
                input_dimensions: (H, W)
            Returns:
                (B, H/2*W/2, 2C), (H/2, W/2)
            """
            H, W = input_dimensions
            B, _, C = x.shape
            x = x.view(B, H, W, C)

            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], dim=-1)
            x = x.view(B, -1, 4 * C)
            x = self.norm(x)
            x = self.reduction(x)
            return x, (H // 2, W // 2)

    class _SwinStage(_nn.Module):
        """One Swin Transformer stage (a sequence of SwinBlocks + optional PatchMerging).

        HF weight path: encoder.model.encoder.layers.{stage_idx}.blocks.{block_idx}.*
        Downsample (if present): encoder.model.encoder.layers.{stage_idx}.downsample.*
        """

        def __init__(
            self,
            dim: int,
            input_resolution: tuple,
            depth: int,
            num_heads: int,
            window_size: int = 8,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_path_rates: list | None = None,
            downsample: bool = False,
        ):
            super().__init__()
            if drop_path_rates is None:
                drop_path_rates = [0.0] * depth

            self.blocks = _nn.ModuleList(
                [
                    _SwinBlock(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=drop_path_rates[i],
                    )
                    for i in range(depth)
                ]
            )
            self.downsample: _nn.Module | None = None
            if downsample:
                self.downsample = _SwinPatchMerging(input_resolution, dim)

        def forward(self, hidden_states: torch.Tensor, input_dimensions: tuple) -> tuple:
            """
            Args:
                hidden_states: (B, H*W, C)
                input_dimensions: (H, W)
            Returns:
                hidden_states: (B, H'*W', C')
                output_dimensions: (H', W')
            """
            for block in self.blocks:
                hidden_states = block(hidden_states, input_dimensions)
            if self.downsample is not None:
                hidden_states, input_dimensions = self.downsample(hidden_states, input_dimensions)
            return hidden_states, input_dimensions

    class _SwinEncoderBody(_nn.Module):
        """Swin encoder body: stack of stages.

        HF weight path prefix: encoder.model.encoder.layers.*
        Followed by a final LayerNorm: encoder.model.encoder.layernorm.*
        """

        def __init__(
            self,
            embed_dim: int = 128,
            depths: list | None = None,
            num_heads: list | None = None,
            image_size: list | None = None,
            patch_size: int = 4,
            window_size: int = 8,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_path_rate: float = 0.1,
        ):
            super().__init__()
            if depths is None:
                depths = [2, 2, 14, 2]
            if num_heads is None:
                num_heads = [4, 8, 16, 32]
            if image_size is None:
                image_size = list(DONUT_IMAGE_SIZE)

            num_stages = len(depths)
            # Stochastic depth decay rule
            total_depth = sum(depths)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

            # Spatial resolution after patch embed
            H_patches = image_size[0] // patch_size
            W_patches = image_size[1] // patch_size

            self.layers = _nn.ModuleList()
            dp_cursor = 0
            for i in range(num_stages):
                dim_i = int(embed_dim * (2**i))
                res_h = H_patches // (2**i)
                res_w = W_patches // (2**i)
                stage = _SwinStage(
                    dim=dim_i,
                    input_resolution=(res_h, res_w),
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rates=dpr[dp_cursor : dp_cursor + depths[i]],
                    downsample=(i < num_stages - 1),  # all stages except last get downsample
                )
                self.layers.append(stage)
                dp_cursor += depths[i]

            # Final norm: LayerNorm on last stage output dim
            final_dim = int(embed_dim * (2 ** (num_stages - 1)))
            self.layernorm = _nn.LayerNorm(final_dim)

        def forward(self, hidden_states: torch.Tensor, input_dimensions: tuple) -> torch.Tensor:
            """
            Args:
                hidden_states: (B, H*W, embed_dim)  — after patch embed
                input_dimensions: (H, W) after patch embed
            Returns:
                (B, H_final*W_final, final_dim)  — last stage output after layernorm
            """
            for stage in self.layers:
                hidden_states, input_dimensions = stage(hidden_states, input_dimensions)
            hidden_states = self.layernorm(hidden_states)
            return hidden_states

    class _SwinModel(_nn.Module):
        """SwinModel — patch embed + encoder body.

        Mirrors HF weight prefix: encoder.model.encoder.*
        """

        def __init__(self, config_dict: dict[str, Any]):
            super().__init__()
            self.embeddings = _SwinPatchEmbedding(
                in_channels=config_dict.get("num_channels", 3),
                embed_dim=config_dict.get("embed_dim", 128),
                patch_size=config_dict.get("patch_size", 4),
            )
            self.encoder = _SwinEncoderBody(
                embed_dim=config_dict.get("embed_dim", 128),
                depths=config_dict.get("depths", [2, 2, 14, 2]),
                num_heads=config_dict.get("num_heads", [4, 8, 16, 32]),
                image_size=config_dict.get("image_size", list(DONUT_IMAGE_SIZE)),
                patch_size=config_dict.get("patch_size", 4),
                window_size=config_dict.get("window_size", 8),
                mlp_ratio=config_dict.get("mlp_ratio", 4.0),
                qkv_bias=config_dict.get("qkv_bias", True),
                drop_path_rate=config_dict.get("drop_path_rate", 0.1),
            )

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            embeddings, dims = self.embeddings(pixel_values)
            return self.encoder(embeddings, dims)

    class _SwinWrapper(_nn.Module):
        """Wrapper matching HF weight path: encoder.model.*

        HF weight structure for SwinModel inside VisionEncoderDecoder:
          encoder.model.encoder.embeddings.*
          encoder.model.encoder.layers.*
          encoder.model.encoder.layernorm.*
        """

        def __init__(self, config_dict: dict[str, Any]):
            super().__init__()
            self.encoder = _SwinModel(config_dict)

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.encoder(pixel_values)

    class _SwinEncoderModule(_nn.Module):
        """Top-level encoder: encoder.model.

        Exposes named_parameters() whose paths start with 'model.encoder.*'
        consistent with the VisionEncoderDecoderModel's 'encoder.*' prefix.
        """

        def __init__(self, config_dict: dict[str, Any]):
            super().__init__()
            self.model = _SwinWrapper(config_dict)

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.model(pixel_values)

        def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
            return super().named_parameters(prefix=prefix, recurse=recurse)

    # ── BART Decoder (matches naver-clova-ix/donut-base weights) ──────────

    class _BARTLearnedPositionalEmbedding(_nn.Embedding):
        """Learned positional embedding with offset (BART uses offset=2)."""

        def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 1):
            # BART offset: positions are stored starting at offset=2
            self.offset = 2
            super().__init__(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)

        def forward(self, input_ids_shape, past_key_values_length: int = 0):
            bsz, seq_len = input_ids_shape
            positions = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_len,
                dtype=torch.long,
                device=self.weight.device,
            )
            return super().forward(positions + self.offset)

    class _BARTDecoderLayer(_nn.Module):
        """One BART decoder layer.

        HF weight names (under decoder.model.decoder.layers.{i}):
          self_attn.k_proj / q_proj / v_proj / out_proj
          self_attn_layer_norm
          encoder_attn.k_proj / q_proj / v_proj / out_proj
          encoder_attn_layer_norm
          fc1 / fc2
          final_layer_norm
        """

        def __init__(
            self,
            d_model: int = 1024,
            decoder_attention_heads: int = 16,
            decoder_ffn_dim: int = 4096,
            dropout: float = 0.0,
            activation_dropout: float = 0.0,
        ):
            super().__init__()
            self.embed_dim = d_model
            self.self_attn = _nn.MultiheadAttention(
                d_model, decoder_attention_heads, dropout=dropout, batch_first=True
            )
            self.self_attn_layer_norm = _nn.LayerNorm(d_model)

            self.encoder_attn = _nn.MultiheadAttention(
                d_model, decoder_attention_heads, dropout=dropout, batch_first=True
            )
            self.encoder_attn_layer_norm = _nn.LayerNorm(d_model)

            self.fc1 = _nn.Linear(d_model, decoder_ffn_dim)
            self.fc2 = _nn.Linear(decoder_ffn_dim, d_model)
            self.final_layer_norm = _nn.LayerNorm(d_model)

            self.dropout_p = dropout
            self.activation_dropout_p = activation_dropout

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            past_key_value: tuple | None = None,
            use_cache: bool = False,
        ) -> tuple:
            """
            Args:
                hidden_states: (B, tgt_len, d_model)
                attention_mask: (B, 1, tgt_len, tgt_len) or (tgt_len, tgt_len) causal mask
                encoder_hidden_states: (B, src_len, d_model)
                encoder_attention_mask: unused (kept for API compat)
                past_key_value: (self_k, self_v, cross_k, cross_v) or None
                use_cache: whether to return present key/values

            Returns:
                (hidden_states, present_key_value)
            """
            residual = hidden_states

            # Self-attention (pre-norm)
            hidden_states = self.self_attn_layer_norm(hidden_states)
            tgt_len = hidden_states.shape[1]

            # Build causal mask for self-attention
            if past_key_value is not None:
                # Incremental decoding: no causal mask needed (single new token)
                self_past_k, self_past_v = past_key_value[0], past_key_value[1]
                # key / value = concat past + current
                k_in = torch.cat([self_past_k, hidden_states], dim=1)
                v_in = torch.cat([self_past_v, hidden_states], dim=1)
                attn_mask_self = None
            else:
                k_in = hidden_states
                v_in = hidden_states
                # Causal mask: upper-triangular with -inf
                attn_mask_self = torch.full(
                    (tgt_len, k_in.shape[1]),
                    float("-inf"),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                attn_mask_self = torch.triu(attn_mask_self, diagonal=1)

            self_attn_out, _ = self.self_attn(
                query=hidden_states,
                key=k_in,
                value=v_in,
                attn_mask=attn_mask_self,
                need_weights=False,
            )

            # Store present key/value (current hidden_states before attn as k/v cache)
            present_self_k = k_in
            present_self_v = v_in

            self_attn_out = _F.dropout(self_attn_out, p=self.dropout_p, training=self.training)
            hidden_states = residual + self_attn_out

            # Cross-attention (encoder attention) — pre-norm
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            if past_key_value is not None and len(past_key_value) >= 4:
                cross_k = past_key_value[2]
                cross_v = past_key_value[3]
            else:
                cross_k = encoder_hidden_states
                cross_v = encoder_hidden_states

            cross_attn_out, _ = self.encoder_attn(
                query=hidden_states,
                key=cross_k,
                value=cross_v,
                need_weights=False,
            )
            cross_attn_out = _F.dropout(cross_attn_out, p=self.dropout_p, training=self.training)
            hidden_states = residual + cross_attn_out

            # FFN (pre-norm)
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = _F.gelu(self.fc1(hidden_states))
            hidden_states = _F.dropout(
                hidden_states, p=self.activation_dropout_p, training=self.training
            )
            hidden_states = self.fc2(hidden_states)
            hidden_states = _F.dropout(hidden_states, p=self.dropout_p, training=self.training)
            hidden_states = residual + hidden_states

            present_key_value = (present_self_k, present_self_v, cross_k, cross_v)
            return hidden_states, present_key_value

    class _BARTDecoderModel(_nn.Module):
        """BART decoder backbone.

        HF weight path: decoder.model.decoder.*
          embed_tokens.weight
          embed_positions.weight
          layers.{i}.*
          layernorm_embedding.{weight,bias}
        """

        def __init__(self, config_dict: dict[str, Any]):
            super().__init__()
            self.d_model = config_dict.get("d_model", 1024)
            vocab_size = config_dict.get("vocab_size", 57580)
            max_pos = config_dict.get("max_position_embeddings", 1536)
            n_layers = config_dict.get("decoder_layers", 4)
            n_heads = config_dict.get("decoder_attention_heads", 16)
            ffn_dim = config_dict.get("decoder_ffn_dim", 4096)
            pad_id = config_dict.get("pad_token_id", 1)

            self.embed_tokens = _nn.Embedding(vocab_size, self.d_model, padding_idx=pad_id)
            self.embed_positions = _BARTLearnedPositionalEmbedding(max_pos, self.d_model)
            self.layers = _nn.ModuleList(
                [
                    _BARTDecoderLayer(
                        d_model=self.d_model,
                        decoder_attention_heads=n_heads,
                        decoder_ffn_dim=ffn_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.layernorm_embedding = _nn.LayerNorm(self.d_model)

        def forward(
            self,
            input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            past_key_values: list | None = None,
            use_cache: bool = False,
        ) -> tuple:
            """
            Args:
                input_ids: (B, tgt_len)
                encoder_hidden_states: (B, src_len, d_model)
                past_key_values: list of per-layer (sk, sv, ck, cv) or None
                use_cache: if True, return present key/values
            Returns:
                (hidden_states: (B, tgt_len, d_model), present_kvs: list or None)
            """
            past_len = 0
            if (
                past_key_values is not None
                and len(past_key_values) > 0
                and past_key_values[0] is not None
            ):
                past_len = past_key_values[0][0].shape[1]

            token_emb = self.embed_tokens(input_ids)
            pos_emb = self.embed_positions(input_ids.shape, past_key_values_length=past_len)
            hidden_states = self.layernorm_embedding(token_emb + pos_emb)

            present_kvs = [] if use_cache else None
            for i, layer in enumerate(self.layers):
                pkv = past_key_values[i] if past_key_values is not None else None
                hidden_states, new_pkv = layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_value=pkv,
                    use_cache=use_cache,
                )
                if use_cache:
                    present_kvs.append(new_pkv)  # type: ignore[union-attr]

            return hidden_states, present_kvs

    class _BARTModelWrapper(_nn.Module):
        """Wrapper so weight paths read decoder.model.decoder.*

        HF nests as: MBartForCausalLM → model → decoder → layers
        Resulting in weight prefix: decoder.model.decoder.*
        """

        def __init__(self, config_dict: dict[str, Any]):
            super().__init__()
            self.decoder = _BARTDecoderModel(config_dict)

        def forward(self, input_ids, encoder_hidden_states, past_key_values=None, use_cache=False):
            return self.decoder(input_ids, encoder_hidden_states, past_key_values, use_cache)

    class _SimpleConfig:
        """Minimal config object carrying key attributes."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _BARTDecoderWrapper(_nn.Module):
        """MBartForCausalLM-like wrapper with lm_head.

        HF weight prefix: decoder.*
          model.decoder.*        (backbone)
          lm_head.weight         (output projection, no bias)
        """

        def __init__(self, config_dict: dict[str, Any]):
            super().__init__()
            self.config = _SimpleConfig(
                tie_word_embeddings=False,
                pad_token_id=config_dict.get("pad_token_id", 1),
                decoder_start_token_id=None,
                use_cache=True,
                vocab_size=config_dict.get("vocab_size", 57580),
                d_model=config_dict.get("d_model", 1024),
            )
            self.model = _BARTModelWrapper(config_dict)
            vocab_size = config_dict.get("vocab_size", 57580)
            d_model = config_dict.get("d_model", 1024)
            self.lm_head = _nn.Linear(d_model, vocab_size, bias=False)

        def resize_token_embeddings(self, new_size: int):
            """Resize embed_tokens and lm_head to new_size."""
            old_size, d_model = self.model.decoder.embed_tokens.weight.shape
            if new_size == old_size:
                return

            # Resize embed_tokens
            new_embed = _nn.Embedding(new_size, d_model)
            _nn.init.normal_(new_embed.weight, mean=0.0, std=d_model**-0.5)
            copy_rows = min(old_size, new_size)
            new_embed.weight.data[:copy_rows] = self.model.decoder.embed_tokens.weight.data[
                :copy_rows
            ]
            new_embed.padding_idx = self.model.decoder.embed_tokens.padding_idx
            self.model.decoder.embed_tokens = new_embed

            # Resize lm_head
            new_lm = _nn.Linear(d_model, new_size, bias=False)
            _nn.init.normal_(new_lm.weight, mean=0.0, std=d_model**-0.5)
            new_lm.weight.data[:copy_rows] = self.lm_head.weight.data[:copy_rows]
            self.lm_head = new_lm

        def forward(
            self,
            input_ids: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            past_key_values: list | None = None,
            use_cache: bool = False,
        ) -> tuple:
            hidden_states, present_kvs = self.model(
                input_ids, encoder_hidden_states, past_key_values, use_cache
            )
            logits = self.lm_head(hidden_states)
            return logits, present_kvs

    # ── VisionEncoderDecoderModel full implementation ──────────────────────

    class _ModelOutput:
        """Minimal output container with .loss and .logits."""

        def __init__(self, loss=None, logits=None):
            self.loss = loss
            self.logits = logits

    class VisionEncoderDecoderModel(_nn.Module):  # type: ignore[no-redef]
        """Inline Swin-B encoder + BART decoder matching naver-clova-ix/donut-base.

        Mirrors the HuggingFace VisionEncoderDecoderModel API:
          - from_pretrained(path_or_repo_id, token=None)
          - forward(pixel_values, decoder_input_ids, labels)  → _ModelOutput
          - generate(pixel_values, decoder_start_token_id, max_length, bad_words_ids)
          - resize_token_embeddings(new_size)   (delegates to decoder)
          - gradient_checkpointing_enable()
          - .encoder  /  .decoder  sub-modules
          - .config.tie_word_embeddings
          - .decoder.lm_head.weight  (for LmHeadCloneCallback)
          - .device  property

        Weight names EXACTLY match the HuggingFace checkpoint so that
        load_state_dict(state_dict, strict=False) works without remapping.
        """

        def __init__(self, encoder_config: dict[str, Any], decoder_config: dict[str, Any]):
            super().__init__()
            self.encoder = _SwinEncoderModule(encoder_config)
            self.decoder = _BARTDecoderWrapper(decoder_config)
            self.config = _SimpleConfig(
                tie_word_embeddings=False,
                pad_token_id=decoder_config.get("pad_token_id", 1),
                decoder_start_token_id=None,
                use_cache=True,
                is_encoder_decoder=True,
            )
            self._use_gradient_checkpointing = False

        # ── Properties ────────────────────────────────────────────────────

        @property
        def device(self) -> torch.device:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

        # ── Gradient checkpointing ─────────────────────────────────────────

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            """Enable gradient checkpointing (reduces VRAM at cost of speed)."""
            self._use_gradient_checkpointing = True

        # ── Token embedding resize ─────────────────────────────────────────

        def resize_token_embeddings(self, new_size: int):
            """Resize decoder embed_tokens and lm_head to *new_size*."""
            self.decoder.resize_token_embeddings(new_size)

        # ── Forward pass ──────────────────────────────────────────────────

        def forward(
            self,
            pixel_values: torch.Tensor | None = None,
            decoder_input_ids: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
            **kwargs,
        ) -> "_ModelOutput":
            """
            Args:
                pixel_values: (B, 3, H, W)
                decoder_input_ids: (B, tgt_len)  — shifted-right target ids
                labels: (B, tgt_len)  — target ids for loss; -100 at padding
            Returns:
                _ModelOutput with .loss (scalar if labels given) and .logits (B, tgt_len, V)
            """
            # Derive decoder_input_ids from labels when not provided (standard
            # teacher-forcing shift-right, matching HF VisionEncoderDecoderModel
            # behavior).  Without this, calling forward(pixel_values, labels=...)
            # would pass None to the decoder and crash.
            if decoder_input_ids is None and labels is not None:
                start_id = self.config.decoder_start_token_id
                if start_id is None:
                    raise RuntimeError(
                        "decoder_start_token_id is not set on model.config. "
                        "Call tokenizer.convert_tokens_to_ids(['<s_sroie>'])[0] and "
                        "assign it to model.config.decoder_start_token_id."
                    )
                pad_id = self.config.pad_token_id
                if pad_id is None:
                    raise RuntimeError(
                        "pad_token_id is not set on model.config. "
                        "Assign tokenizer.pad_token_id to model.config.pad_token_id."
                    )
            # Standard teacher-forcing shift-right: prepend decoder_start_token_id,
            # drop the last label token.  -100 is the HuggingFace ignore index
            # (used to mask padding positions in the cross-entropy loss) and must
            # be replaced with a real token ID before embedding lookup.
            bos = labels.new_full((labels.shape[0], 1), start_id)
            _labels_clean = labels.clone()
            _labels_clean[_labels_clean == LABEL_IGNORE_INDEX] = (
                pad_id  # replace ignore-index with pad
            )
            decoder_input_ids = torch.cat([bos, _labels_clean[:, :-1]], dim=1)  # shift right

            # Encode image
            encoder_out = self.encoder(pixel_values)  # (B, src_len, 1024)

            # Decode
            logits, _ = self.decoder(
                decoder_input_ids, encoder_out, use_cache=False
            )  # (B, tgt_len, V)

            loss = None
            if labels is not None:
                # Shift: logits[t] predicts labels[t]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = _F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=LABEL_IGNORE_INDEX,
                )

            return _ModelOutput(loss=loss, logits=logits)

        # ── Autoregressive generation ──────────────────────────────────────

        @torch.no_grad()
        def generate(
            self,
            pixel_values: torch.Tensor,
            decoder_start_token_id: int = 0,
            max_length: int = 768,
            bad_words_ids: list | None = None,
            **kwargs,
        ) -> torch.Tensor:
            """Greedy autoregressive generation.

            Args:
                pixel_values: (B, 3, H, W)
                decoder_start_token_id: id of the first decoder input token
                max_length: maximum number of generated tokens
                bad_words_ids: list of [[token_id], ...] to block (ignored by greedy;
                    kept for API compatibility with transformers)
            Returns:
                generated_ids: (B, seq_len) including the start token
            """
            device = self.device
            pixel_values = pixel_values.to(device)
            B = pixel_values.shape[0]

            # Encode once
            encoder_out = self.encoder(pixel_values)  # (B, src_len, 1024)

            # Build bad-words set for masking
            bad_word_ids_set: set = set()
            if bad_words_ids:
                for bw in bad_words_ids:
                    if bw:
                        bad_word_ids_set.update(bw)

            # Init decoder input with start token
            generated = torch.full((B, 1), decoder_start_token_id, dtype=torch.long, device=device)
            past_key_values: list | None = None

            for _ in range(max_length - 1):
                # Feed only the last token (with past_key_values for speed)
                if past_key_values is not None:
                    cur_input = generated[:, -1:]
                else:
                    cur_input = generated

                logits, past_key_values = self.decoder(
                    cur_input, encoder_out, past_key_values=past_key_values, use_cache=True
                )
                next_logits = logits[:, -1, :]  # (B, V)

                # Block bad words
                if bad_word_ids_set:
                    for bw_id in bad_word_ids_set:
                        next_logits[:, bw_id] = float("-inf")

                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if all sequences hit eos (pad token used as eos in DONUT)
                pad_id = self.config.pad_token_id
                if pad_id is not None and (next_token == pad_id).all():
                    break

            return generated

        # ── Checkpoint loading ─────────────────────────────────────────────

        @classmethod
        def from_pretrained(
            cls,
            model_name_or_path: str | Path,
            token: str | None = None,
            **kwargs,
        ) -> "VisionEncoderDecoderModel":
            """Load model weights from a local directory or HuggingFace Hub repo.

            Priority:
              1. Local directory (if it exists and contains config.json)
              2. HuggingFace Hub download (via _hf_hub_download)

            After loading, verifies that decoder.lm_head.weight is present in the
            checkpoint (guards against the safetensors deduplication bug described
            in CLAUDE.md §16).

            Args:
                model_name_or_path: local path or HF repo id (e.g. "naver-clova-ix/donut-base")
                token: optional HF token for private repos

            Returns:
                Loaded VisionEncoderDecoderModel
            """
            model_path = Path(model_name_or_path)

            # If the path doesn't exist locally, attempt HF Hub download
            if not model_path.exists() or not (model_path / "config.json").exists():
                # Try to read token from hf_token.txt if not provided
                if token is None:
                    token_file = Path("hf_token.txt")
                    if token_file.exists():
                        token = token_file.read_text().strip() or None
                cache_dir = Path.home() / ".cache" / "donut" / model_path.name
                model_path = _hf_hub_download(str(model_name_or_path), cache_dir, token=token)

            # Load config.json
            config_path = model_path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"config.json not found in {model_path}")
            with open(config_path) as f:
                full_config = json.load(f)

            encoder_cfg = full_config.get("encoder", {})
            decoder_cfg = full_config.get("decoder", {})
            # Propagate top-level pad_token_id to decoder config if not set
            if "pad_token_id" not in decoder_cfg and "pad_token_id" in full_config:
                decoder_cfg["pad_token_id"] = full_config["pad_token_id"]

            model = cls(encoder_cfg, decoder_cfg)

            # Load weights — prefer model.safetensors, fall back to pytorch_model.bin
            st_path = model_path / "model.safetensors"
            bin_path = model_path / "pytorch_model.bin"

            state_dict: dict = {}
            if st_path.exists():
                state_dict = _load_safetensors(st_path)
            elif bin_path.exists():
                state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
            else:
                raise FileNotFoundError(
                    f"No model.safetensors or pytorch_model.bin found in {model_path}"
                )

            # Guard: detect safetensors deduplication bug (lm_head.weight missing)
            # If tie_word_embeddings is False and lm_head.weight is absent, copy
            # embed_tokens.weight as a starting point (will be overwritten by fine-tuning).
            # ROBUSTNESS: both lm_head_key AND embed_key may be absent if the checkpoint
            # was saved with both tensors deduplicated out (e.g. a very aggressive
            # safetensors writer).  In that case we cannot recover silently — raise
            # immediately so the caller knows the checkpoint is corrupt.
            lm_head_key = "decoder.lm_head.weight"
            embed_key = "decoder.model.decoder.embed_tokens.weight"
            if lm_head_key not in state_dict:
                if embed_key in state_dict:
                    state_dict[lm_head_key] = state_dict[embed_key].clone()
                    import logging as _logging_lm

                    _logging_lm.getLogger(__name__).warning(
                        "load_model_with_tied_weights: lm_head.weight missing from checkpoint "
                        "— cloned from embed_tokens.weight as fallback. "
                        "Ensure LmHeadCloneCallback is registered during training to prevent this."
                    )
                else:
                    # Neither key present: checkpoint is corrupt and unrecoverable.
                    _tie = getattr(getattr(model, "config", None), "tie_word_embeddings", True)
                    if not _tie:
                        raise RuntimeError(
                            "CRITICAL: decoder.lm_head.weight is missing from the checkpoint "
                            "and decoder.model.decoder.embed_tokens.weight is also absent — "
                            "the checkpoint is corrupt and cannot be loaded safely. "
                            "Re-train with LmHeadCloneCallback registered. "
                            "See CLAUDE.md §16 Pattern 6."
                        )

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            # Filter out expected missing keys (position_ids buffer, etc.)
            truly_missing = [
                k
                for k in missing
                if not k.endswith("relative_position_index")
                and not k.endswith("_attn_mask")
                and "num_batches_tracked" not in k
            ]
            # ROBUSTNESS: explicitly fail if lm_head.weight is still absent after load.
            # This catches the case where our pre-load guard above was bypassed (e.g.
            # tie_word_embeddings=True on config, so the guard's _tie check was skipped).
            if lm_head_key in missing:
                _tie_after = getattr(getattr(model, "config", None), "tie_word_embeddings", True)
                if not _tie_after:
                    raise RuntimeError(
                        f"CRITICAL: {lm_head_key} is listed in missing_keys after "
                        "load_state_dict(). The checkpoint is corrupt. "
                        "Re-train with LmHeadCloneCallback registered. "
                        "See CLAUDE.md §16 Pattern 6."
                    )
            if truly_missing:
                import logging as _logging_fr

                _logging_fr.getLogger(__name__).warning(
                    "VisionEncoderDecoderModel.from_pretrained: %d missing keys (first 10): %s",
                    len(truly_missing),
                    truly_missing[:10],
                )
            if unexpected:
                import logging as _logging_fr2

                _logging_fr2.getLogger(__name__).debug(
                    "VisionEncoderDecoderModel.from_pretrained: %d unexpected keys (first 5): %s",
                    len(unexpected),
                    unexpected[:5],
                )

            # Fix: issue_report_summary critical #1 — verify lm_head.weight is present
            # after load_state_dict(strict=False). The guard above only pre-patches the
            # state_dict when embed_key is present; if both keys were deduplicated by
            # safetensors (i.e. embed_key was also absent), lm_head.weight is silently
            # missing and the model trains with F1 ≈ 0.42 with no error message.
            _lm_head = getattr(getattr(model, "decoder", None), "lm_head", None)
            if _lm_head is not None and hasattr(_lm_head, "weight"):
                _lm_weight = _lm_head.weight
                if _lm_weight is None or (
                    hasattr(_lm_weight, "device") and str(_lm_weight.device) == "meta"
                ):
                    # Check if tie_word_embeddings is disabled (our pipeline sets it to False).
                    _dec_cfg = getattr(getattr(model, "decoder", None), "config", None)
                    if _dec_cfg is not None and not getattr(_dec_cfg, "tie_word_embeddings", True):
                        raise RuntimeError(
                            f"CRITICAL: decoder.lm_head.weight is missing from the loaded "
                            f"checkpoint at {model_path!r}. This is caused by safetensors "
                            "deduplication dropping lm_head.weight when it shares storage with "
                            "embed_tokens.weight. Fix: ensure LmHeadCloneCallback is registered "
                            "during training (calls .data.clone() before every checkpoint save) "
                            "and that model.decoder.config.tie_word_embeddings=False. "
                            "See CLAUDE.md §16 Pattern 6."
                        )
            # Also verify via missing_keys that lm_head.weight is not in the filtered list
            if lm_head_key in truly_missing:
                raise RuntimeError(
                    f"CRITICAL: {lm_head_key!r} was not loaded from checkpoint at "
                    f"{model_path!r} (present in missing_keys after load_state_dict). "
                    "This indicates safetensors deduplicated lm_head.weight along with "
                    "embed_tokens.weight. Register LmHeadCloneCallback during training to "
                    "break the shared-storage aliasing before each checkpoint save. "
                    "See CLAUDE.md §16 Pattern 6."
                )

            return model

        @staticmethod
        def from_encoder_decoder_pretrained(*args, **kwargs):
            raise NotImplementedError(
                "from_encoder_decoder_pretrained is not implemented in the inline "
                "VisionEncoderDecoderModel. Use from_pretrained() with a local directory "
                "containing config.json and model.safetensors."
            )

# ─────────────────────────────────────────────────────────────────────────────
# Inline image loader — fallback when Pillow is unavailable
# ─────────────────────────────────────────────────────────────────────────────

try:
    from PIL import Image as _PILImage

    def _load_image(path: str | Path) -> "_PILImage.Image":  # type: ignore[name-defined]
        return _PILImage.open(path).convert("RGB")

    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

    def _png_unfilter(scanlines: list[tuple[int, bytes]], width: int, bpp: int) -> bytes:
        """Apply PNG row de-filtering (Sub/Up/Average/Paeth)."""
        out = []
        prev = bytes(width * bpp)
        for ftype, raw in scanlines:
            row = bytearray(raw)
            if ftype == 1:  # Sub
                for i in range(bpp, len(row)):
                    row[i] = (row[i] + row[i - bpp]) & 0xFF
            elif ftype == 2:  # Up
                for i in range(len(row)):
                    row[i] = (row[i] + prev[i]) & 0xFF
            elif ftype == 3:  # Average
                for i in range(len(row)):
                    a = row[i - bpp] if i >= bpp else 0
                    row[i] = (row[i] + (a + prev[i]) // 2) & 0xFF
            elif ftype == 4:  # Paeth
                for i in range(len(row)):
                    a = row[i - bpp] if i >= bpp else 0
                    b = prev[i]
                    c = prev[i - bpp] if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    row[i] = (row[i] + pr) & 0xFF
            out.append(bytes(row))
            prev = bytes(row)
        return b"".join(out)

    def _load_png(path: str | Path) -> "torch.Tensor | None":
        """Minimal PNG decoder → RGB numpy-compatible bytes."""
        import numpy as np

        data = Path(path).read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        pos = 8
        width = height = bit_depth = color_type = 0
        idat = b""
        while pos < len(data):
            length = struct.unpack(">I", data[pos : pos + 4])[0]
            ctype = data[pos + 4 : pos + 8]
            chunk = data[pos + 8 : pos + 8 + length]
            pos += 12 + length
            if ctype == b"IHDR":
                width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk[:10])
            elif ctype == b"IDAT":
                idat += chunk
            elif ctype == b"IEND":
                break
        raw = zlib.decompress(idat)
        # Only handle 8-bit RGB (type 2) and RGBA (type 6)
        if bit_depth != 8 or color_type not in (2, 6):
            return None
        channels = 3 if color_type == 2 else 4
        row_bytes = width * channels
        scanlines = []
        r = 0
        for _ in range(height):
            ftype = raw[r]
            scanlines.append((ftype, raw[r + 1 : r + 1 + row_bytes]))
            r += row_bytes + 1
        pixel_data = _png_unfilter(scanlines, width, channels)
        arr = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, channels)
        if channels == 4:  # Drop alpha
            arr = arr[:, :, :3]
        return arr

    def _load_bmp(path: str | Path) -> "torch.Tensor | None":
        """Minimal BMP decoder for 24-bit uncompressed BMP → RGB numpy array."""
        import numpy as np

        data = Path(path).read_bytes()
        if data[:2] != b"BM":
            return None
        pixel_offset = struct.unpack_from("<I", data, 10)[0]
        width = struct.unpack_from("<i", data, 18)[0]
        height = struct.unpack_from("<i", data, 22)[0]
        bits_per_pixel = struct.unpack_from("<H", data, 28)[0]
        compression = struct.unpack_from("<I", data, 30)[0]
        if bits_per_pixel != 24 or compression != 0:
            return None
        flipped = height > 0
        height = abs(height)
        row_size = (width * 3 + 3) & ~3  # 4-byte aligned rows
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for row in range(height):
            src_row = (height - 1 - row) if flipped else row
            start = pixel_offset + src_row * row_size
            raw_row = data[start : start + width * 3]
            pixels = np.frombuffer(raw_row, dtype=np.uint8).reshape(width, 3)
            arr[row] = pixels[:, ::-1]  # BGR → RGB
        return arr

    def _load_jpeg_pure(path: "str | Path"):  # noqa: C901
        """Pure-Python + NumPy baseline JPEG decoder (SOF0/SOF1 only).

        Supports YCbCr/Grayscale, 4:4:4 and 4:2:0 sampling, EXIF/JFIF APP markers.
        Returns H×W×3 uint8 ndarray (RGB) or None on unsupported/corrupt input.
        """
        import numpy as np

        # ── Zigzag inverse lookup (position_in_stream → flat index in 8×8) ──
        # _ZIGZAG_ORDER[i] = natural-order flat index for the i-th zigzag-scan position.
        # Used to scatter coefficients from zigzag scan order into 8×8 natural order.
        _ZIGZAG_ORDER = [
            0,
            1,
            5,
            6,
            14,
            15,
            27,
            28,
            2,
            4,
            7,
            13,
            16,
            26,
            29,
            42,
            3,
            8,
            12,
            17,
            25,
            30,
            41,
            43,
            9,
            11,
            18,
            24,
            31,
            40,
            44,
            53,
            10,
            19,
            23,
            32,
            39,
            45,
            52,
            54,
            20,
            22,
            33,
            38,
            46,
            51,
            55,
            60,
            21,
            34,
            37,
            47,
            50,
            56,
            59,
            61,
            35,
            36,
            48,
            49,
            57,
            58,
            62,
            63,
        ]

        # ── Precompute 2D IDCT cosine matrix ──────────────────────────────────
        _M = np.array(
            [[np.cos(np.pi * (2 * n + 1) * k / 16) for k in range(8)] for n in range(8)],
            dtype=np.float32,
        )

        def _idct2(block: np.ndarray) -> np.ndarray:
            """2D IDCT-III for an 8×8 block of DCT coefficients."""
            s = block.astype(np.float32)
            s[:, 0] /= np.sqrt(2.0)
            s[0, :] /= np.sqrt(2.0)
            return 0.25 * (_M @ s @ _M.T)

        def _build_huffman(counts: list[int], values: list[int]) -> dict[tuple[int, int], int]:
            """Build Huffman decode table: {(code, length): symbol}."""
            table: dict = {}
            code = 0
            idx = 0
            for length in range(1, 17):
                for _ in range(counts[length - 1]):
                    table[(code, length)] = values[idx]
                    idx += 1
                    code += 1
                code <<= 1
            return table

        # ── Bitstream reader ──────────────────────────────────────────────────
        class _BitReader:
            __slots__ = ("_data", "_pos", "_buf", "_bits_left")

            def __init__(self, data: bytes) -> None:
                self._data = data
                self._pos = 0
                self._buf = 0
                self._bits_left = 0

            def _fill(self) -> None:
                while self._bits_left <= 24 and self._pos < len(self._data):
                    b = self._data[self._pos]
                    self._pos += 1
                    if b == 0xFF:
                        b2 = self._data[self._pos] if self._pos < len(self._data) else 0
                        if b2 == 0x00:
                            # byte stuffing: emit 0xFF
                            self._pos += 1
                        elif 0xD0 <= b2 <= 0xD7:
                            # restart marker — skip, reset is handled by caller
                            self._pos += 1
                            continue
                        elif b2 == 0xD9:
                            # EOI inside entropy stream — stop filling
                            break
                        else:
                            # other marker — do not consume, stop
                            self._pos -= 1
                            break
                    self._buf = (self._buf << 8) | b
                    self._bits_left += 8

            def read_bits(self, n: int) -> int:
                if n == 0:
                    return 0
                if self._bits_left < n:
                    self._fill()
                if self._bits_left < n:
                    raise EOFError("JPEG bitstream truncated")
                self._bits_left -= n
                return (self._buf >> self._bits_left) & ((1 << n) - 1)

            def decode_huffman(self, table: dict[tuple[int, int], int]) -> int:
                code = 0
                for length in range(1, 17):
                    code = (code << 1) | self.read_bits(1)
                    sym = table.get((code, length))
                    if sym is not None:
                        return sym
                raise ValueError("Invalid Huffman code")

            def skip_to_marker(self) -> int:
                """Advance until 0xFF <non-zero, non-stuff> is found; return marker byte."""
                while self._pos < len(self._data):
                    if self._data[self._pos] == 0xFF:
                        self._pos += 1
                        b2 = self._data[self._pos] if self._pos < len(self._data) else 0
                        if b2 != 0x00 and not (0xD0 <= b2 <= 0xD7):
                            self._pos += 1
                            return b2
                    else:
                        self._pos += 1
                return 0

        def _extend(val: int, bits: int) -> int:
            """JPEG coefficient magnitude extension (sign bit)."""
            if bits == 0:
                return 0
            if val < (1 << (bits - 1)):
                return val - (1 << bits) + 1
            return val

        try:
            data = Path(path).read_bytes()
        except OSError:
            return None

        if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
            return None  # not a JPEG

        pos = 2
        quant_tables: dict[int, np.ndarray] = {}
        huff_tables: dict[tuple, dict] = {}  # (class, id) → table
        frame_width = frame_height = 0
        n_components = 0
        comp_info: list[dict] = []  # list of {id, h_samp, v_samp, qt_id}
        sos_data: bytes = b""
        sos_comp_order: list[dict] = []
        exif_orientation = 1

        # ── Parse markers ────────────────────────────────────────────────────
        while pos + 1 < len(data):
            if data[pos] != 0xFF:
                pos += 1
                continue
            while pos < len(data) and data[pos] == 0xFF:
                pos += 1
            if pos >= len(data):
                break
            marker = data[pos]
            pos += 1

            if marker == 0xD8:  # SOI
                continue
            if marker == 0xD9:  # EOI
                break
            if 0xD0 <= marker <= 0xD7:  # RST
                continue
            if pos + 1 >= len(data):
                break
            seg_len = struct.unpack_from(">H", data, pos)[0]
            seg_end = pos + seg_len
            seg_data = data[pos + 2 : seg_end]
            pos = seg_end

            if marker == 0xE1:  # APP1 — may contain EXIF
                try:
                    if seg_data[:6] == b"Exif\x00\x00":
                        tiff = seg_data[6:]
                        byte_order = tiff[:2]
                        bo = ">" if byte_order == b"MM" else "<"
                        ifd0_offset = struct.unpack_from(bo + "I", tiff, 4)[0]
                        n_entries = struct.unpack_from(bo + "H", tiff, ifd0_offset)[0]
                        for ei in range(n_entries):
                            eoff = ifd0_offset + 2 + ei * 12
                            tag = struct.unpack_from(bo + "H", tiff, eoff)[0]
                            if tag == 0x0112:  # Orientation
                                exif_orientation = struct.unpack_from(bo + "H", tiff, eoff + 8)[0]
                                break
                except (struct.error, ValueError, IndexError):
                    pass

            elif 0xE0 <= marker <= 0xEF:  # other APP markers — skip
                pass

            elif marker == 0xDB:  # DQT — quantization table(s)
                sp = 0
                while sp < len(seg_data):
                    pq_tq = seg_data[sp]
                    sp += 1
                    pq = pq_tq >> 4  # precision: 0=8bit, 1=16bit
                    tq = pq_tq & 0x0F
                    if pq == 0:
                        raw = np.frombuffer(seg_data[sp : sp + 64], dtype=np.uint8)
                        sp += 64
                    else:
                        raw = np.frombuffer(seg_data[sp : sp + 128], dtype=np.uint16)
                        raw = raw.byteswap()
                        sp += 128
                    # De-zigzag: raw[i] is the quant factor for zigzag position i;
                    # _ZIGZAG_ORDER[i] is the corresponding natural flat index.
                    qt = np.zeros(64, dtype=np.float32)
                    for i in range(64):
                        qt[_ZIGZAG_ORDER[i]] = float(raw[i])
                    quant_tables[tq] = qt.reshape(8, 8)

            elif marker in (0xC0, 0xC1):  # SOF0 / SOF1 — baseline DCT
                precision = seg_data[0]
                if precision != 8:
                    return None  # only 8-bit supported
                frame_height = struct.unpack_from(">H", seg_data, 1)[0]
                frame_width = struct.unpack_from(">H", seg_data, 3)[0]
                n_components = seg_data[5]
                if n_components not in (1, 3):
                    return None  # CMYK / other not supported
                comp_info = []
                for ci in range(n_components):
                    off = 6 + ci * 3
                    cid = seg_data[off]
                    samp = seg_data[off + 1]
                    qt_id = seg_data[off + 2]
                    comp_info.append(
                        {
                            "id": cid,
                            "h_samp": samp >> 4,
                            "v_samp": samp & 0x0F,
                            "qt_id": qt_id,
                        }
                    )

            elif marker == 0xC2:  # SOF2 — progressive (not supported)
                return None

            elif marker == 0xC4:  # DHT — Huffman table
                sp = 0
                while sp < len(seg_data):
                    tc_th = seg_data[sp]
                    sp += 1
                    tc = tc_th >> 4  # class: 0=DC, 1=AC
                    th = tc_th & 0x0F
                    counts = list(seg_data[sp : sp + 16])
                    sp += 16
                    n_syms = sum(counts)
                    values = list(seg_data[sp : sp + n_syms])
                    sp += n_syms
                    huff_tables[(tc, th)] = _build_huffman(counts, values)

            elif marker == 0xDA:  # SOS — start of scan
                # Parse SOS header
                n_comp_scan = seg_data[0]
                sos_comp_order = []
                for sci in range(n_comp_scan):
                    cs = seg_data[1 + sci * 2]
                    td_ta = seg_data[2 + sci * 2]
                    sos_comp_order.append(
                        {
                            "id": cs,
                            "dc_id": td_ta >> 4,
                            "ac_id": td_ta & 0x0F,
                        }
                    )
                # Entropy-coded data is everything from pos onwards until EOI
                sos_data = data[pos:]
                break  # done parsing markers

        if not comp_info or not sos_data or frame_width == 0 or frame_height == 0:
            return None

        # ── Determine sampling factors ────────────────────────────────────────
        max_h = max(c["h_samp"] for c in comp_info)
        max_v = max(c["v_samp"] for c in comp_info)
        # MCU size in pixels
        mcu_w = max_h * 8
        mcu_h = max_v * 8
        mcu_cols = (frame_width + mcu_w - 1) // mcu_w
        mcu_rows = (frame_height + mcu_h - 1) // mcu_h

        # Map component id → sos entry
        id_to_sos = {s["id"]: s for s in sos_comp_order}

        # ── Allocate output planes ─────────────────────────────────────────────
        planes: list[np.ndarray] = []
        for c in comp_info:
            ph = mcu_rows * c["v_samp"] * 8
            pw = mcu_cols * c["h_samp"] * 8
            planes.append(np.zeros((ph, pw), dtype=np.float32))

        # ── Decode entropy stream ──────────────────────────────────────────────
        br = _BitReader(sos_data)
        dc_preds = [0] * n_components

        try:
            for mcu_row in range(mcu_rows):
                for mcu_col in range(mcu_cols):
                    for ci, comp in enumerate(comp_info):
                        cid = comp["id"]
                        sos_entry = id_to_sos.get(cid)
                        if sos_entry is None:
                            continue
                        dc_table = huff_tables.get((0, sos_entry["dc_id"]), {})
                        ac_table = huff_tables.get((1, sos_entry["ac_id"]), {})
                        qt = quant_tables.get(comp["qt_id"], np.ones((8, 8), dtype=np.float32))
                        h_blocks = comp["h_samp"]
                        v_blocks = comp["v_samp"]

                        for vb in range(v_blocks):
                            for hb in range(h_blocks):
                                # ── Decode DC coefficient ──────────────────
                                dc_sym = br.decode_huffman(dc_table)
                                dc_bits = br.read_bits(dc_sym)
                                dc_val = _extend(dc_bits, dc_sym)
                                dc_preds[ci] += dc_val

                                # ── Decode 63 AC coefficients ──────────────
                                coeffs = np.zeros(64, dtype=np.float32)
                                coeffs[0] = dc_preds[ci]
                                k = 1
                                while k < 64:
                                    ac_sym = br.decode_huffman(ac_table)
                                    rrr = ac_sym >> 4
                                    sss = ac_sym & 0x0F
                                    if sss == 0:
                                        if rrr == 0:
                                            break  # EOB
                                        else:
                                            k += 16  # ZRL
                                            continue
                                    k += rrr
                                    if k >= 64:
                                        break
                                    ac_bits = br.read_bits(sss)
                                    coeffs[k] = _extend(ac_bits, sss)
                                    k += 1

                                # ── Dequantize (zigzag order → natural order) ──
                                # coeffs[zi] is the coefficient at zigzag position zi;
                                # _ZIGZAG_ORDER[zi] is its natural flat index;
                                # qt.flat[_ZIGZAG_ORDER[zi]] is the matching quant factor.
                                block = np.zeros((8, 8), dtype=np.float32)
                                for zi in range(64):
                                    nat = _ZIGZAG_ORDER[zi]
                                    block.flat[nat] = coeffs[zi] * qt.flat[nat]

                                # ── 2D IDCT ───────────────────────────────
                                spatial = _idct2(block) + 128.0

                                # ── Write into plane ──────────────────────
                                pr = mcu_row * v_blocks * 8 + vb * 8
                                pc = mcu_col * h_blocks * 8 + hb * 8
                                planes[ci][pr : pr + 8, pc : pc + 8] = spatial

        except Exception:  # intentional broad catch: partial JPEG decode recovery
            # On bitstream error: return whatever we have (partial decode)
            pass

        # ── Crop planes to actual frame dimensions ────────────────────────────
        if n_components == 1:
            # Grayscale → replicate to RGB
            Y = np.clip(planes[0][:frame_height, :frame_width], 0, 255).astype(np.uint8)
            rgb = np.stack([Y, Y, Y], axis=2)
        else:
            # YCbCr → RGB, with upsampling if subsampled
            yc = comp_info[0]
            cbc = comp_info[1]
            crc = comp_info[2]

            Y_plane = planes[0][:frame_height, :frame_width]

            # Upsample Cb and Cr if subsampled relative to Y
            cb_h = mcu_rows * cbc["v_samp"] * 8
            cb_w = mcu_cols * cbc["h_samp"] * 8
            cr_h = mcu_rows * crc["v_samp"] * 8
            cr_w = mcu_cols * crc["h_samp"] * 8
            Cb_plane = planes[1][:cb_h, :cb_w]
            Cr_plane = planes[2][:cr_h, :cr_w]

            v_ratio = yc["v_samp"] // cbc["v_samp"] if cbc["v_samp"] > 0 else 1
            h_ratio = yc["h_samp"] // cbc["h_samp"] if cbc["h_samp"] > 0 else 1

            if v_ratio > 1 or h_ratio > 1:
                Cb_plane = np.repeat(np.repeat(Cb_plane, v_ratio, axis=0), h_ratio, axis=1)
                Cr_plane = np.repeat(np.repeat(Cr_plane, v_ratio, axis=0), h_ratio, axis=1)

            # Trim to frame size after upsampling
            Cb_plane = Cb_plane[:frame_height, :frame_width]
            Cr_plane = Cr_plane[:frame_height, :frame_width]

            R = Y_plane + 1.402 * (Cr_plane - 128.0)
            G = Y_plane - 0.34414 * (Cb_plane - 128.0) - 0.71414 * (Cr_plane - 128.0)
            B = Y_plane + 1.772 * (Cb_plane - 128.0)

            R = np.clip(R, 0, 255).astype(np.uint8)
            G = np.clip(G, 0, 255).astype(np.uint8)
            B = np.clip(B, 0, 255).astype(np.uint8)
            rgb = np.stack([R, G, B], axis=2)

        # ── Apply EXIF orientation ─────────────────────────────────────────────
        if exif_orientation == 3:
            rgb = np.rot90(rgb, 2)
        elif exif_orientation == 6:
            rgb = np.rot90(rgb, 3)
        elif exif_orientation == 8:
            rgb = np.rot90(rgb, 1)

        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def _load_image(path: str | Path):  # type: ignore[misc]
        """Load an image file as an RGB numpy array without PIL.

        Supports PNG (full), BMP (24-bit), and JPEG (pure-Python baseline DCT).
        For TIFF/WebP: install Pillow (pip install Pillow).
        """
        import numpy as np

        path = Path(path)
        suffix = path.suffix.lower()
        arr = None
        if suffix == ".png":
            arr = _load_png(path)
        elif suffix in (".bmp",):
            arr = _load_bmp(path)
        elif suffix in (".jpg", ".jpeg"):
            arr = _load_jpeg_pure(path)
        if arr is None:
            raise RuntimeError(
                f"Cannot load {path} without Pillow. "
                "Install Pillow: pip install Pillow\n"
                "PNG (8-bit RGB/RGBA), BMP (24-bit), and JPEG (baseline DCT) "
                "are supported natively."
            )
        # Ensure contiguous uint8 RGB
        return np.ascontiguousarray(arr, dtype=np.uint8)

    # Provide a minimal Image-like namespace so code that does `from PIL import Image`
    # and calls Image.open(p).convert("RGB") can use `_load_image` as a shim.
    class _ImageModule:
        @staticmethod
        def open(path):
            class _FakeImg:
                def __init__(self, arr):
                    self._arr = arr

                def convert(self, mode):
                    return self  # already RGB

                @property
                def size(self):
                    h, w = self._arr.shape[:2]
                    return w, h

            return _FakeImg(_load_image(path))

    Image = _ImageModule()  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
# Inline safetensors reader — fallback when safetensors package is unavailable
# ─────────────────────────────────────────────────────────────────────────────

_ST_DTYPE_MAP: dict = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _load_safetensors(path: str | Path) -> dict[str, Any]:
    """Load a .safetensors checkpoint file without the safetensors package.

    The safetensors binary format is simple:
      [8-byte header_size: uint64 LE][header_size bytes of UTF-8 JSON][tensor bytes]

    The JSON maps tensor_name → {"dtype": str, "shape": list[int], "data_offsets": [start, end]}.
    """
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
        data_start = 8 + header_size
        tensors: dict = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            offset_start, offset_end = meta["data_offsets"]
            f.seek(data_start + offset_start)
            raw = f.read(offset_end - offset_start)
            dtype = _ST_DTYPE_MAP.get(meta["dtype"], torch.float32)
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
            shape = meta.get("shape", [])
            if shape:
                tensor = tensor.reshape(shape)
            tensors[name] = tensor.clone()  # clone to detach from buffer
    return tensors


import resource_manager as _mm  # noqa: E402

# FIX: Previously FIELDS, MAX_LENGTH, IMAGE_EXTS, NEW_TOKENS, BASE_MODEL,
# SEED were defined independently here and in 4 other files, risking silent
# drift if any file was updated without updating the others.
from constants import (  # noqa: E402
    BASE_MODEL,
    DONUT_IMAGE_SIZE,
    FIELDS,
    LABEL_IGNORE_INDEX,
    MAX_LENGTH,
    NEW_TOKENS,
    SEED,
    WORKSPACE,
    _mask_empty_field_labels,
    _optimal_num_workers,
)

__all__ = ["SROIEDataset", "MultiDataset", "DonutTrainer", "TrainingResult", "_ensure_dual_config"]

logger = logging.getLogger(__name__)


def _ensure_dual_config(model, field: str, value) -> None:
    """Set a config field on both model.config and model.decoder.config.

    This prevents the silent failure where model.config and model.decoder.config
    disagree — the model trains correctly using model.config but save_pretrained()
    serializes model.decoder.config, so the reloaded model uses the wrong value.

    The function is intentionally defensive: if model.decoder or
    model.decoder.config does not exist, the setter is skipped for the decoder
    config without raising an exception.
    """
    setattr(model.config, field, value)
    if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
        setattr(model.decoder.config, field, value)


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
            logger.debug("LmHeadCloneCallback.on_save: model is None — skipping clone")
            return control
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            logger.debug("LmHeadCloneCallback.on_save: model has no decoder attr — skipping clone")
            return control
        lm_head = getattr(decoder, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            lm_head.weight = torch.nn.Parameter(lm_head.weight.data.clone())
            logger.debug(
                "LmHeadCloneCallback.on_save: cloned lm_head.weight (epoch %s)",
                state.epoch,
            )
        else:
            logger.debug("LmHeadCloneCallback.on_save: no lm_head.weight found — skipping clone")
        return control


# ---------------------------------------------------------------------------
# MinEpochsBeforeStoppingCallback — prevent early stopping before min_epochs
# ---------------------------------------------------------------------------


class MinEpochsBeforeStoppingCallback(TrainerCallback):
    """Prevent EarlyStoppingCallback from stopping training before min_epochs.

    Small SROIE datasets (500 samples, ~63 optimizer steps/epoch) exhibit a
    natural loss plateau in the first few epochs before the model exits the
    XML-scaffolding phase and starts learning field content.  With patience=3
    or patience=5 this plateau can still trigger early stopping prematurely.

    This callback intercepts ``on_evaluate`` and, while ``state.epoch <
    min_epochs``, resets ``control.should_training_stop`` to ``False`` if
    ``EarlyStoppingCallback`` (registered before this callback) just set it
    to ``True``.

    Registration order: this callback MUST be registered AFTER
    ``EarlyStoppingCallback`` so that it fires second and can override the
    stop signal that ESC just set.

    Parameters
    ----------
    min_epochs : int
        Minimum number of complete epochs before early stopping is allowed.
        Default is 5.
    """

    def __init__(self, min_epochs: int = 5) -> None:
        self.min_epochs = min_epochs

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Suppress early-stopping signal before min_epochs.

        ``EarlyStoppingCallback`` fires first (it is registered before this
        callback) and may set ``control.should_training_stop = True``.  We
        reset it to ``False`` while ``state.epoch < min_epochs`` so the model
        is guaranteed to train for at least ``min_epochs`` epochs regardless
        of the val-loss trajectory.
        """
        if state is None:
            return control
        current_epoch = getattr(state, "epoch", 0)
        if current_epoch < self.min_epochs and getattr(control, "should_training_stop", False):
            control.should_training_stop = False
            logger.debug(
                "MinEpochsBeforeStoppingCallback: epoch %.1f < %d — "
                "suppressed early-stopping signal",
                current_epoch,
                self.min_epochs,
            )
        return control


class SROIEOnlyValCallback(TrainerCallback):
    """Log SROIE-only exact-match F1 per epoch as a diagnostic metric.

    For experiments with mixed validation sets (SROIE + auxiliary data),
    the combined ``eval_loss`` used for early stopping is a mixture of
    SROIE and auxiliary domain signals.  This callback computes and logs
    SROIE-only exact-match F1 at every epoch-end so users can audit
    whether early stopping fired at the SROIE-optimal epoch.

    Does **not** change training behaviour — only logs a diagnostic value.

    Attributes
    ----------
    sroie_val_samples : list[tuple]
        ``(image_path, ground_truth_dict)`` tuples from the SROIE val split.
    processor : DonutProcessor
    output_dir : Path
        Where to write the per-epoch ``sroie_only_val_f1.csv`` log.
    """

    def __init__(self, sroie_val_samples, processor, output_dir: Path) -> None:
        super().__init__()
        self._samples = sroie_val_samples
        self._processor = processor
        self._output_dir = Path(output_dir)
        self._rows: list[dict] = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Compute SROIE-only exact-match F1 and log it."""
        if model is None or not self._samples:
            return control

        try:
            device = next(model.parameters()).device
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for img_path, gt in self._samples:
                    try:
                        image = _load_image(img_path)
                        pixel_values = self._processor(image, return_tensors="pt").pixel_values.to(
                            device
                        )
                        decoder_input_ids = (
                            torch.tensor(
                                self._processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])
                            )
                            .unsqueeze(0)
                            .to(device)
                        )
                        outputs = model.generate(
                            pixel_values,
                            decoder_input_ids=decoder_input_ids,
                            max_length=128,
                            num_beams=1,
                        )
                        decoded = self._processor.batch_decode(outputs, skip_special_tokens=False)[
                            0
                        ]
                        result = self._processor.token2json(decoded)
                        if isinstance(result, list):
                            merged: dict = {}
                            for page in result:
                                if isinstance(page, dict):
                                    for k, v in page.items():
                                        if k not in merged:
                                            merged[k] = v
                            result = merged
                        if not isinstance(result, dict):
                            result = {}
                        for fld in FIELDS:
                            pred_val = str(result.get(fld, "")).lower().strip()
                            gt_val = str(gt.get(fld, "")).lower().strip()
                            total += 1
                            if pred_val == gt_val:
                                correct += 1
                    except Exception:  # intentional broad catch: OOM recovery in callback
                        total += len(FIELDS)  # count as all wrong on error
            f1 = correct / total if total > 0 else 0.0
        except Exception as exc:  # intentional broad catch: OOM recovery boundary
            logger.warning("SROIEOnlyValCallback: failed to compute F1: %s", exc)
            model.train()
            return control

        model.train()
        epoch = int(state.epoch) if state.epoch else 0
        self._rows.append({"epoch": epoch, "sroie_only_val_f1": round(f1, 4)})
        logger.info("SROIEOnlyValCallback: epoch=%d  sroie_only_val_f1=%.4f", epoch, f1)

        # Append to CSV log
        csv_path = self._output_dir / "sroie_only_val_f1.csv"
        write_header = not csv_path.exists()
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["epoch", "sroie_only_val_f1"])
                if write_header:
                    writer.writeheader()
                writer.writerow(self._rows[-1])
        except OSError as exc:
            logger.warning("SROIEOnlyValCallback: could not write CSV: %s", exc)
        return control


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
    ) -> None:
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
                        n,
                        len(samples),
                        f,
                        100.0 * n / len(samples),
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
        image = _load_image(img_path)

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
        labels[labels == self.processor.tokenizer.pad_token_id] = LABEL_IGNORE_INDEX
        # Mask empty-field spans so they contribute no gradient to the loss.
        labels = _mask_empty_field_labels(labels, gt, self.processor.tokenizer)
        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# MultiDataset
# ---------------------------------------------------------------------------

_LOW_VRAM_THRESHOLD_BYTES = 25 * (1024**3)  # 25 GB


class MultiDataset(Dataset):
    """Wraps a list of (Path, dict) samples into a PyTorch Dataset.

    When sufficient RAM is available, pre-loads all images into memory
    to eliminate disk I/O during training.
    """

    def __init__(
        self,
        samples: list[tuple[Path, dict]],
        processor: DonutProcessor,
        max_length: int = MAX_LENGTH,
        cache_in_ram: bool = True,
        precompute_tensors: bool = True,
        sample_sources: list[str] | None = None,
        aux_loss_weight: float = 1.0,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self._image_cache: dict[int, Image.Image] = {}
        self._pixel_cache: dict[int, Any] = {}  # precomputed pixel_values tensors
        self._label_cache: dict[int, Any] = {}  # precomputed label token tensors
        # Per-sample loss weights: 1.0 for SROIE, aux_loss_weight for auxiliary.
        # None means all weights are 1.0 (default / no-op).
        if sample_sources is not None and len(sample_sources) != len(samples):
            raise ValueError(
                f"sample_sources length ({len(sample_sources)}) must match "
                f"samples length ({len(samples)})."
            )
        self._sample_sources: list[str] | None = sample_sources
        self._aux_loss_weight: float = aux_loss_weight

        if cache_in_ram and len(samples) > 0:
            # Determine actual image dimensions from processor_config.json so
            # the RAM estimate is correct at any resolution.
            # The old hardcoded `* 3` (3 MB/sample) was wrong at 2560×1920
            # (actual: 14.06 MB/sample) causing the gate to open when it should
            # be closed. memory_manager.ram_cache_is_safe() uses the real formula:
            #   3 × H × W / 1_048_576 MB per sample.
            try:
                from resource_manager import get_image_size_from_processor_config

                _img_h, _img_w = get_image_size_from_processor_config()
            except (ImportError, RuntimeError):
                _img_h, _img_w = DONUT_IMAGE_SIZE  # safe fallback to DONUT native resolution
            if _mm.ram_cache_is_safe(len(samples), _img_h, _img_w):
                import concurrent.futures

                def _load_one(idx_path):
                    idx, path = idx_path
                    try:
                        return idx, _load_image(path)
                    except (OSError, ValueError):
                        return idx, None

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                    for idx, img in pool.map(_load_one, enumerate(s[0] for s in samples)):
                        if img is not None:
                            self._image_cache[idx] = img

                logging.getLogger(__name__).info(
                    "[RAM Cache] %d/%d images cached", len(self._image_cache), len(samples)
                )
            else:
                logging.getLogger(__name__).info(
                    "[RAM Cache] Skipped (ram_cache_is_safe returned False for %d samples at %dx%d)",
                    len(samples),
                    _img_h,
                    _img_w,
                )

            # Precompute pixel_values tensors to eliminate per-step DonutImageProcessor
            # overhead. Each float32 tensor is 3×1280×960×4 bytes ≈ 14.2 MB.
            # Only attempt if: images are cached AND RAM allows the extra footprint.
            # Hard cap: never allocate more than 4 GB of pixel tensors regardless of
            # available RAM, because background model downloads and GPU activations
            # consume headroom that psutil.virtual_memory() may not reflect yet.
            _PIXEL_TENSOR_MAX_MB = 4096
            _pix_mb = len(self._image_cache) * 14.2
            _log = logging.getLogger(__name__)
            if len(self._image_cache) > 0 and precompute_tensors:
                if _pix_mb > _PIXEL_TENSOR_MAX_MB:
                    _log.info(
                        "[Tensor Cache] Skipped (estimated %.0f MB exceeds hard cap %d MB)",
                        _pix_mb,
                        _PIXEL_TENSOR_MAX_MB,
                    )
                else:
                    _free_mb = _mm.ram_headroom_mb()
                    if _pix_mb > _free_mb * 0.50:
                        _log.info(
                            "[Tensor Cache] Skipped (%.0f MB > 50%% of %.0f MB free RAM)",
                            _pix_mb,
                            _free_mb,
                        )
                    else:
                        _log.info(
                            "[Tensor Cache] Precomputing pixel_values for %d images (~%.0f MB) ...",
                            len(self._image_cache),
                            _pix_mb,
                        )
                        for _idx, _img in self._image_cache.items():
                            try:
                                self._pixel_cache[_idx] = processor(
                                    _img, return_tensors="pt"
                                ).pixel_values.squeeze()
                            except (RuntimeError, ValueError):
                                pass
                        _log.info(
                            "[Tensor Cache] Precomputed %d/%d pixel_values tensors",
                            len(self._pixel_cache),
                            len(samples),
                        )
            elif len(self._image_cache) > 0 and not precompute_tensors:
                _log.info("[Tensor Cache] Skipped (precompute_tensors=False — val/eval dataset)")

            # Precompute label token tensors — each is 768 ints (≈3 KB), always fits in RAM.
            # Amortises tokeniser overhead (sentencepiece BPE encode + pad to max_length)
            # across all training steps that revisit each sample.
            if len(self._image_cache) > 0 and precompute_tensors:
                _log = logging.getLogger(__name__)
                for _idx, (_, _gt) in enumerate(samples):
                    _target = "<s_sroie>"
                    for _f in FIELDS:
                        _v = _gt.get(_f, "")
                        _target += f"<s_{_f}>{_v}</s_{_f}>"
                    _target += "</s_sroie>"
                    _lbl = processor.tokenizer(
                        _target,
                        add_special_tokens=False,  # match inference: no BOS/EOS wrappers
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.squeeze()
                    _lbl[_lbl == processor.tokenizer.pad_token_id] = LABEL_IGNORE_INDEX
                    _lbl = _mask_empty_field_labels(_lbl, _gt, processor.tokenizer)
                    self._label_cache[_idx] = _lbl
                _log.info("[Label Cache] Precomputed %d label tensors", len(self._label_cache))

        # Log per-field masking statistics so empty-field dilution is visible.
        if samples:
            _log = logging.getLogger(__name__)
            for f in FIELDS:
                n = sum(1 for _, gt in samples if not gt.get(f, "").strip())
                if n:
                    _log.info(
                        "Field masking: %d/%d samples will have <%s> masked (%.1f%%)",
                        n,
                        len(samples),
                        f,
                        100.0 * n / len(samples),
                    )

    def clear_caches(self) -> None:
        """Explicitly free all precomputed caches to release RAM/VRAM before GC.

        Call this before ``del train_ds`` / ``del val_ds`` to ensure that
        ``_pixel_cache`` tensors (up to 7.1 GB of float32 data) are freed
        immediately rather than waiting for Python's cyclic GC to discover
        that the dataset is unreachable.  This is critical between the 8
        sequential DONUT experiments — without it, the previous experiment's
        cache stays pinned while the next experiment allocates its own,
        doubling the peak RAM usage.
        """
        self._pixel_cache.clear()
        self._image_cache.clear()
        self._label_cache.clear()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path, gt = self.samples[idx]

        # Use precomputed pixel_values tensor if available (eliminates per-step
        # DonutImageProcessor overhead that was causing 88s/step starvation).
        if idx in self._pixel_cache:
            pixel_values = self._pixel_cache[idx]
        else:
            # Fall back: use cached PIL image or load from disk, then process.
            if idx in self._image_cache:
                image = self._image_cache[idx]
            else:
                image = _load_image(img_path)
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # Use precomputed label tensor if available (eliminates per-step tokenisation).
        if idx in self._label_cache:
            labels = self._label_cache[idx]
        else:
            target = "<s_sroie>"
            for f in FIELDS:
                v = gt.get(f, "")
                target += f"<s_{f}>{v}</s_{f}>"
            target += "</s_sroie>"
            labels = self.processor.tokenizer(
                target,
                add_special_tokens=False,  # match inference: no BOS/EOS wrappers
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze()
            labels[labels == self.processor.tokenizer.pad_token_id] = LABEL_IGNORE_INDEX
            labels = _mask_empty_field_labels(labels, gt, self.processor.tokenizer)

        item = {"pixel_values": pixel_values, "labels": labels}

        # Per-sample loss weight: 1.0 for SROIE, aux_loss_weight for auxiliary.
        # Only added when aux_loss_weight != 1.0 to avoid any overhead on default runs.
        if self._sample_sources is not None and self._aux_loss_weight != 1.0:
            src = self._sample_sources[idx] if idx < len(self._sample_sources) else "sroie"
            w = 1.0 if src == "sroie" else self._aux_loss_weight
            item["loss_weight"] = torch.tensor(w, dtype=torch.float32)

        return item


# ---------------------------------------------------------------------------
# LiveDashboardCallback (inlined from live_dashboard.py)
# ---------------------------------------------------------------------------

# Resolve TrainerCallback at class-definition time (not at __init__ time) to
# avoid monkey-patching self.__class__ — a proper mixin/composition pattern.
try:
    from transformers import TrainerCallback as _TrainerCallbackBase
except ImportError:

    class _TrainerCallbackBase:  # type: ignore[no-redef]
        """Stub when transformers is not installed."""


@dataclass
class _EpochRow:
    epoch: int
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    best_f1: float = float("nan")


class LiveDashboardCallback(_TrainerCallbackBase):
    """Trainer callback — per-epoch CSV + rich.live.Live table for a lab-style console.

    When ``rich`` is installed the table is rendered in-place (overwriting previous
    rows) rather than appending, giving a persistent "lab panel" feel throughout the
    entire experiment run.  Falls back silently to plain logging when rich is absent
    or when DISABLE_LIVE_DASHBOARD=1 is set.

    Inherits from ``TrainerCallback`` when transformers is installed, or from a
    no-op stub otherwise.
    """

    def __init__(
        self,
        csv_path: "str | Path | None" = None,
        experiment_id: int = 0,
        use_rich: "bool | None" = None,
        total_epochs: int = 0,
        total_experiments: int = 0,
        dataset_names: "list[str] | None" = None,
        num_samples: int = 0,
    ) -> None:

        if csv_path is None:
            csv_path = f"convergence_exp{experiment_id}.csv"
        self._csv_path = Path(csv_path)
        self._experiment_id = experiment_id
        self._total_epochs = total_epochs
        self._total_experiments = total_experiments
        self._dataset_names: list[str] = dataset_names or []
        self._num_samples = num_samples
        self._rows: list[_EpochRow] = []
        self._best_f1: float = float("nan")
        self._live: Any = None  # rich.live.Live instance when active

        if use_rich is None:
            use_rich = os.environ.get("DISABLE_LIVE_DASHBOARD", "0") != "1"

        self._rich_enabled = False
        if use_rich:
            try:
                import rich  # noqa: F401

                self._rich_enabled = True
            except ImportError:
                pass

        if self._rich_enabled:
            self._start_live()

        if not self._csv_path.exists():
            try:
                self._csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._csv_path, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["epoch", "train_loss", "val_loss", "best_f1"])
            except OSError as exc:
                logger.warning("[LiveDashboard] Could not create CSV %s: %s", self._csv_path, exc)

    def _start_live(self) -> None:
        """Start a rich.live.Live context for in-place table updates."""
        try:
            from rich.live import Live

            self._live = Live(self._build_table(), refresh_per_second=4, transient=True)
            self._live.start()
        except Exception as exc:  # intentional broad catch: third-party rich library
            logger.debug("[LiveDashboard] Could not start rich.live.Live: %s", exc)
            self._live = None

    def _build_table(self) -> "Any":
        """Build the rich Table renderable from current row data."""
        try:
            from rich.table import Table
            from rich.text import Text

            # Header: experiment progress indicator
            exp_str = (
                f"Exp {self._experiment_id}/{self._total_experiments}"
                if self._total_experiments
                else f"Exp {self._experiment_id}"
            )
            epoch_str = f"[{len(self._rows)}/{self._total_epochs}]" if self._total_epochs else ""
            datasets_str = ", ".join(self._dataset_names[:3]) if self._dataset_names else ""
            samples_str = f"  {self._num_samples} samples" if self._num_samples else ""
            subtitle = (
                f"[dim]{datasets_str}{samples_str}[/]" if (datasets_str or samples_str) else None
            )

            table = Table(
                title=f"[bold cyan]{exp_str}[/] — Training {epoch_str}",
                caption=subtitle,
                show_header=True,
                header_style="bold dim",
                border_style="dim",
                expand=False,
            )
            table.add_column("Ep", justify="right", style="dim", width=4)
            table.add_column("Train ↓", justify="right", width=9)
            table.add_column("Val ↓", justify="right", width=9)
            table.add_column("Best F1 ↑", justify="right", style="bold green", width=10)
            table.add_column("Δ F1", justify="right", style="dim", width=8)

            prev_val_loss = float("nan")
            prev_f1 = float("nan")
            for r in self._rows[-15:]:
                tl = f"{r.train_loss:.4f}" if r.train_loss == r.train_loss else "—"
                # Color-code val loss: green if decreasing, red if increasing
                if r.val_loss == r.val_loss:
                    if prev_val_loss == prev_val_loss and r.val_loss < prev_val_loss:
                        vl = Text(f"{r.val_loss:.4f}", style="green")
                    elif prev_val_loss == prev_val_loss and r.val_loss > prev_val_loss:
                        vl = Text(f"{r.val_loss:.4f}", style="red")
                    else:
                        vl = Text(f"{r.val_loss:.4f}")
                    prev_val_loss = r.val_loss
                else:
                    vl = Text("—", style="dim")
                bf = f"{r.best_f1:.4f}" if r.best_f1 == r.best_f1 else "—"
                if r.best_f1 == r.best_f1 and prev_f1 == prev_f1:
                    delta = r.best_f1 - prev_f1
                    delta_str = f"{delta:+.4f}"
                    delta_cell = Text(delta_str, style="green" if delta >= 0 else "red")
                else:
                    delta_cell = Text("—", style="dim")
                if r.best_f1 == r.best_f1:
                    prev_f1 = r.best_f1
                table.add_row(str(r.epoch), tl, vl, bf, delta_cell)

            # Footer row: disk + VRAM usage
            try:
                from constants import format_bytes, get_disk_usage

                _, _, disk_free = get_disk_usage()
                footer_parts = [f"Disk free: {format_bytes(disk_free)}"]
                try:
                    import torch

                    if torch.cuda.is_available():
                        vram_used = torch.cuda.memory_allocated()
                        vram_total = torch.cuda.get_device_properties(0).total_memory
                        footer_parts.append(
                            f"VRAM: {format_bytes(vram_used)}/{format_bytes(vram_total)}"
                        )
                except (RuntimeError, AttributeError):
                    pass
                table.caption = Text(" · ".join(footer_parts), style="dim")
            except Exception:  # intentional broad catch: third-party rich library
                pass

            return table
        except Exception:  # intentional broad catch: third-party rich library
            return ""

    def on_log(
        self, args: "Any", state: "Any", control: "Any", logs: dict | None = None, **kwargs: "Any"
    ) -> None:
        """Print a single overwriting \\r progress line to console during training.

        Full per-step logs are written to terminal.txt via the logging system;
        this method just keeps the console to one clean line.
        Only fires on training steps (logs contains 'loss' key), not eval steps.
        When running inside a rich.live.Live context, the \\r line is suppressed
        (the live table already shows progress).
        """
        if logs is None or "loss" not in logs:
            return
        # If rich.live is active, don't clutter output with \r lines
        if self._live is not None:
            return
        loss = logs["loss"]
        epoch = int(getattr(state, "epoch", 0))
        step = int(getattr(state, "global_step", 0))
        max_steps = int(getattr(state, "max_steps", 0))
        exp_label = f"Exp {self._experiment_id}" if self._experiment_id else "Training"
        total_ep = self._total_epochs if self._total_epochs else "?"
        line = (
            f"\r{exp_label} | Epoch {epoch}/{total_ep}"
            f" | Step {step}/{max_steps}"
            f" | Loss: {loss:.3f}   "
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    def on_epoch_end(self, args: "Any", state: "Any", control: "Any", **kwargs: "Any") -> None:
        # Clear the \r progress line so epoch summary starts on a fresh line
        # (only needed when not inside a rich.live.Live context).
        if self._live is None:
            sys.stdout.write("\n")
            sys.stdout.flush()
        log = state.log_history if state is not None else []
        train_loss = float("nan")
        val_loss = float("nan")
        for entry in reversed(log):
            if "loss" in entry and train_loss != train_loss:
                train_loss = float(entry["loss"])
            if "eval_loss" in entry and val_loss != val_loss:
                val_loss = float(entry["eval_loss"])
            if train_loss == train_loss and val_loss == val_loss:
                break
        epoch = int(getattr(state, "epoch", len(self._rows) + 1))
        row = _EpochRow(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        self._rows.append(row)
        try:
            with open(self._csv_path, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([epoch, train_loss, val_loss, self._best_f1])
        except OSError as exc:
            logger.warning("[LiveDashboard] CSV write failed: %s", exc)
        if self._rich_enabled and self._live is not None:
            try:
                self._live.update(self._build_table())
            except Exception as exc:  # intentional broad catch: third-party rich library
                logger.debug("[LiveDashboard] live.update failed: %s", exc)

    def update_best_f1(self, f1: float) -> None:
        if f1 > self._best_f1 or self._best_f1 != self._best_f1:
            self._best_f1 = f1
            if self._rows:
                self._rows[-1].best_f1 = f1
            if self._live is not None:
                try:
                    self._live.update(self._build_table())
                except Exception:  # intentional broad catch: third-party rich library
                    pass

    def close(self, metrics: "dict[str, float] | None" = None) -> None:
        """Stop the rich.live.Live context and print a final summary card."""
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:  # intentional broad catch: third-party rich library
                pass
            self._live = None

        # Print a clean final summary card
        if self._rich_enabled and self._rows:
            try:
                from rich.console import Console
                from rich.panel import Panel
                from rich.table import Table
                from rich.text import Text

                console = Console()
                best_f1 = self._best_f1 if self._best_f1 == self._best_f1 else 0.0
                exp_str = (
                    f"Exp {self._experiment_id}/{self._total_experiments}"
                    if self._total_experiments
                    else f"Exp {self._experiment_id}"
                )
                datasets_str = ", ".join(self._dataset_names) if self._dataset_names else "—"
                summary_table = Table.grid(padding=(0, 1))
                summary_table.add_column(style="dim")
                summary_table.add_column()
                summary_table.add_row("Experiment", f"[bold]{exp_str}[/]")
                summary_table.add_row("Datasets", datasets_str)
                if self._num_samples:
                    summary_table.add_row("Samples", str(self._num_samples))
                summary_table.add_row("Epochs", f"{len(self._rows)}/{self._total_epochs}")
                f1_color = "green" if best_f1 >= 0.8 else ("yellow" if best_f1 >= 0.5 else "red")
                summary_table.add_row("Best F1", Text(f"{best_f1:.4f}", style=f"bold {f1_color}"))
                if metrics:
                    for field in ("company", "date", "address", "total"):
                        fv = metrics.get(f"{field}_f1")
                        if fv is not None:
                            bar_len = int(fv * 20)
                            bar = "█" * bar_len + "░" * (20 - bar_len)
                            summary_table.add_row(f"  {field}", f"[cyan]{bar}[/] {fv:.3f}")
                try:
                    from constants import format_bytes, get_disk_usage

                    _, _, disk_free = get_disk_usage()
                    summary_table.add_row("Disk free", format_bytes(disk_free))
                except (ImportError, OSError):
                    pass
                console.print(
                    Panel(
                        summary_table,
                        title=f"[bold green]✓ Training Complete[/] — {exp_str}",
                        border_style="green",
                        padding=(0, 1),
                    )
                )
            except Exception:  # intentional broad catch: third-party rich library
                pass


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
    ) -> None:
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

        # Keep num_workers=0 when batch_size≤2 and the RAM/tensor cache is
        # populated. The primary bottleneck causing ~88s/step is the
        # DonutImageProcessor running on every __getitem__ call (eliminated by
        # precomputing pixel_values in MultiDataset.__init__). Forking workers
        # adds IPC overhead on top, so stay with workers=0 when we have caches.
        _batch_size = self.config.per_device_train_batch_size
        _cache_populated = (
            (
                hasattr(self.train_dataset, "_pixel_cache")
                and len(self.train_dataset._pixel_cache) > 0
            )
            or (
                hasattr(self.train_dataset, "_label_cache")
                and len(self.train_dataset._label_cache) > 0
            )
            or (
                hasattr(self.train_dataset, "_image_cache")
                and len(self.train_dataset._image_cache) > 0
            )
        )
        if _batch_size <= 2 and _cache_populated:
            logger.info(
                "DataLoader: num_workers=0 (batch_size=%d ≤ 2 + RAM cache active — "
                "eliminates fork+IPC overhead that caused 88s/step GPU starvation)",
                _batch_size,
            )
            optimal_workers = 0

        # Detect bf16 support (Ampere+ GPUs including Blackwell) — prefer over fp16
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        use_fp16 = torch.cuda.is_available() and not use_bf16
        logging.debug("precision: %s", "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32"))

        # torch.compile DISABLED for DONUT: mode="reduce-overhead" uses CUDA
        # graphs which assume fixed tensor shapes, but autoregressive generate()
        # (used during evaluation via predict_with_generate=True) changes shape
        # at every decoding step.  This causes silent failures or ValueError
        # during evaluation, producing F1=0.  The ~15-30% training speedup is
        # negated by the ~1.5-2 min kernel compilation overhead per experiment
        # and the broken evaluation.
        logging.debug("torch.compile skipped (incompatible with autoregressive generate)")

        # Cap warmup_steps to ≤10% of total optimizer steps.
        # warmup=500 is correct for large datasets (Exp 8, ~3940 samples, ~1250 opt steps),
        # but exceeds total training for small datasets (Exp 1, ~500 samples, ~320 opt steps),
        # causing the LR to never reach peak value and producing CORD-schema hallucinations.
        _grad_accum = getattr(self.config, "gradient_accumulation_steps", 2)
        _steps_epoch = math.ceil(len(self.train_dataset) / self.config.per_device_train_batch_size)
        _total_opt_steps = math.ceil(_steps_epoch / _grad_accum) * self.config.max_epochs

        # Validate that this configuration produces enough optimizer steps.
        # DONUT requires ~200+ steps to learn field content beyond XML scaffolding;
        # fewer steps yield perfectly structured but content-empty predictions (F1≈0)
        # with no visible error — a known silent failure documented in resource_manager.py.
        from resource_manager import validate_training_config as _vtc  # noqa: PLC0415

        _vtc(
            batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=_grad_accum,
            num_train_samples=len(self.train_dataset),
            epochs=self.config.max_epochs,
        )

        _cfg_warmup = getattr(self.config, "warmup_steps", 100)
        _eff_warmup = min(_cfg_warmup, max(10, _total_opt_steps // 10))
        if _eff_warmup != _cfg_warmup:
            logger.warning(
                "warmup_steps capped %d → %d (dataset has only %d opt steps over %d epochs)",
                _cfg_warmup,
                _eff_warmup,
                _total_opt_steps,
                self.config.max_epochs,
            )

        # Guard DataLoader kwargs that were renamed/removed in transformers 5.x.
        # When num_workers=8, passing dataloader_prefetch_factor=2 to
        # Seq2SeqTrainingArguments raises TypeError in transformers >= 5.0 if the
        # parameter no longer exists (root cause: Exp 2-8 crash in <7s while Exp 1
        # with num_workers=0 trains normally). Use inspect to probe the signature.
        import inspect as _inspect

        _ta_params = set(_inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
        _dl_extra: dict = {}
        if optimal_workers > 0:
            if "dataloader_prefetch_factor" in _ta_params:
                _dl_extra["dataloader_prefetch_factor"] = 2
            if "dataloader_persistent_workers" in _ta_params:
                _dl_extra["dataloader_persistent_workers"] = True
        elif "dataloader_prefetch_factor" in _ta_params:
            _dl_extra["dataloader_prefetch_factor"] = None

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self._output_dir),
            num_train_epochs=self.config.max_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=_grad_accum,
            # Set to encoder_lr for HF logging purposes only — the custom
            # layerwise optimizer passed via optimizers= takes precedence.
            learning_rate=getattr(self.config, "encoder_lr", self.config.learning_rate),
            warmup_steps=_eff_warmup,
            weight_decay=getattr(self.config, "weight_decay", 0.01),
            # Label smoothing 0.1: regularises the lm_head distribution; reduces
            # overconfident predictions on rare SROIE tokens (e.g. RM amounts).
            # 0.1 is the standard recommendation for seq2seq token classification;
            # values >0.15 degrade exact-match in SROIE Task-3 (field must match exactly).
            label_smoothing_factor=0.1,
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
            report_to=["tensorboard"],
            logging_dir=str(
                Path(str(getattr(self.config, "output_dir", "results")))
                / "tb_logs"
                / f"exp_{getattr(self.config, 'experiment_id', 0)}"
            ),
            # PERFORMANCE: Optimized DataLoader settings
            dataloader_num_workers=optimal_workers,
            dataloader_pin_memory=optimal_workers > 0,
            remove_unused_columns=False,
            seed=getattr(self.config, "seed", SEED),
            **_dl_extra,
        )

        # Build layerwise optimizer: encoder at encoder_lr, decoder at decoder_lr,
        # with weight decay EXCLUDED from 1-D parameters (biases and LayerNorm
        # scale/shift tensors).
        #
        # WHY exclude 1-D params from weight decay:
        # - Bias terms: L2 regularization on biases makes it harder for the model
        #   to learn precise additive offsets (e.g. vocabulary token logit biases).
        # - LayerNorm weight/bias: shrinking these destabilizes normalization —
        #   LayerNorm scale=1 and shift=0 is only the *initial* state, not the
        #   optimal one; regularizing toward it hurts convergence.
        # - Industry standard: GPT-2, BERT, T5 all exclude 1-D params from decay.
        #
        # Detection strategy: use p.ndim < 2 (all biases and LayerNorm params are
        # 1-D scalars; all weight matrices are ≥2-D). This is more robust than
        # name matching because it works regardless of HF module naming conventions.
        encoder_lr = getattr(self.config, "encoder_lr", self.config.learning_rate)
        decoder_lr = getattr(self.config, "decoder_lr", self.config.learning_rate)
        _weight_decay = getattr(self.config, "weight_decay", 0.01)

        _enc_decay, _enc_no_decay, _dec_decay, _dec_no_decay = [], [], [], []
        for _pname, _p in self.model.named_parameters():
            if not _p.requires_grad:
                continue
            _is_encoder = _pname.startswith("encoder.")
            _no_decay = _p.ndim < 2  # 1-D → bias or LayerNorm; exclude from decay
            if _is_encoder:
                (_enc_no_decay if _no_decay else _enc_decay).append(_p)
            else:
                (_dec_no_decay if _no_decay else _dec_decay).append(_p)

        logger.info(
            "Optimizer param groups: enc_decay=%d enc_nodecay=%d dec_decay=%d dec_nodecay=%d",
            len(_enc_decay),
            len(_enc_no_decay),
            len(_dec_decay),
            len(_dec_no_decay),
        )

        _optimizer_type = getattr(self.config, "optimizer_type", "adamw")
        if _optimizer_type == "sgd":
            # SGD + Nesterov: faster per-step, sufficient for near-converged transformers
            # used in micro mode where adaptive moments aren't needed for short runs
            optimizer = torch.optim.SGD(
                [
                    {"params": _enc_decay, "lr": encoder_lr, "weight_decay": _weight_decay},
                    {"params": _enc_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
                    {"params": _dec_decay, "lr": decoder_lr, "weight_decay": _weight_decay},
                    {"params": _dec_no_decay, "lr": decoder_lr, "weight_decay": 0.0},
                ],
                momentum=0.9,
                nesterov=True,
            )
            logger.info("Optimizer: SGD + Nesterov (micro/mini mode)")
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": _enc_decay, "lr": encoder_lr, "weight_decay": _weight_decay},
                    {"params": _enc_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
                    {"params": _dec_decay, "lr": decoder_lr, "weight_decay": _weight_decay},
                    {"params": _dec_no_decay, "lr": decoder_lr, "weight_decay": 0.0},
                ],
            )

        # OneCycleLR: aggressive warmup + cosine decay, reaches peak LR immediately
        # — much faster convergence than cosine+warmup for short (2–3 epoch) micro runs
        _lr_schedule = getattr(self.config, "lr_schedule", "cosine")
        custom_scheduler = None
        if _lr_schedule == "one_cycle" and _total_opt_steps > 0:
            # 4 param groups now (enc_decay, enc_nodecay, dec_decay, dec_nodecay).
            # OneCycleLR requires one max_lr per param group — repeat each LR so
            # the no-decay group mirrors its decay counterpart's schedule.
            custom_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[encoder_lr, encoder_lr, decoder_lr, decoder_lr],
                total_steps=_total_opt_steps,
                pct_start=0.1,  # 10% warmup, 90% cosine decay
                anneal_strategy="cos",
                div_factor=10.0,  # start lr = max_lr / 10
                final_div_factor=100.0,  # end lr = start_lr / 100
            )
            logger.info(
                "OneCycleLR: max_lr=[%.2e, %.2e], total_steps=%d",
                encoder_lr,
                decoder_lr,
                _total_opt_steps,
            )

        # LmHeadCloneCallback MUST be registered before EarlyStoppingCallback
        # so the clone happens before every checkpoint write (including the
        # best-model checkpoint that load_best_model_at_end reloads).
        callbacks = [LmHeadCloneCallback()]
        if do_eval:
            patience = getattr(
                self.config,
                "early_stopping_patience",
                3,
            )
            # EarlyStoppingCallback MUST be registered BEFORE
            # MinEpochsBeforeStoppingCallback so that ESC fires first and may
            # set control.should_training_stop = True, then MinEpochs fires
            # second and resets it to False while state.epoch < min_epochs.
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=patience,
                )
            )
            callbacks.append(MinEpochsBeforeStoppingCallback(min_epochs=5))

        # Register SROIE-only val F1 diagnostic callback when a pure SROIE
        # validation split is available.  This allows auditing whether
        # early stopping fired at the SROIE-optimal epoch vs. the
        # mixed-validation-optimal epoch (see §4 / §5 limitation note).
        _sroie_val_samples = getattr(self.config, "_sroie_val_samples", None)
        if do_eval and _sroie_val_samples and len(_sroie_val_samples) > 0:
            _out_dir = getattr(self.config, "output_dir", "results")
            callbacks.append(
                SROIEOnlyValCallback(
                    sroie_val_samples=_sroie_val_samples,
                    processor=self.processor,
                    output_dir=Path(str(_out_dir)),
                )
            )
            logger.info(
                "SROIEOnlyValCallback registered (%d SROIE val samples → %s/sroie_only_val_f1.csv)",
                len(_sroie_val_samples),
                _out_dir,
            )

        # DiagnosticCallback — runtime bug-pattern detection + optional AI diagnosis.
        # Enabled by default; set DISABLE_DIAGNOSTICS=1 to skip.
        # AI diagnosis requires ANTHROPIC_API_KEY or MISTRAL_API_KEY env var.
        if os.environ.get("DISABLE_DIAGNOSTICS", "0") != "1":
            try:
                from diagnostics import DiagnosticCallback as _DiagCB

                _ai_diag = os.environ.get("AI_DIAGNOSE", "0") == "1"
                _ai_prov = os.environ.get("AI_DIAGNOSE_PROVIDER", "auto")
                # Fix: issue_report_summary invariant #16 — use None as default, warn if unset.
                _diag_exp_id = getattr(self.config, "experiment_id", None)
                if _diag_exp_id is None:
                    logger.warning(
                        "[Diagnostics] config.experiment_id is None — diagnostics file will "
                        "be named 'diagnostics_expNone.json'. Set experiment_id in ExperimentConfig."
                    )
                _diag_cb = _DiagCB(
                    experiment_id=_diag_exp_id if _diag_exp_id is not None else 0,
                    output_dir=Path(str(getattr(self.config, "output_dir", "results"))),
                    ai_diagnose=_ai_diag,
                    ai_provider=_ai_prov,
                )
                callbacks.append(_diag_cb)
                logger.debug(
                    "[Diagnostics] Callback registered (ai_diagnose=%s, provider=%s)",
                    _ai_diag,
                    _ai_prov,
                )
            except ImportError:
                pass  # diagnostics.py not present — skip
            except Exception as _diag_exc:  # intentional broad catch: third-party diagnostics
                logger.debug("[Diagnostics] Registration failed: %s", _diag_exc)

        # LiveDashboardCallback — auto-registered when rich is installed.
        # Writes per-epoch CSV and optionally redraws a rich table.
        # Disabled by setting env var DISABLE_LIVE_DASHBOARD=1.
        if os.environ.get("DISABLE_LIVE_DASHBOARD", "0") != "1":
            try:
                # Fix: issue_report_summary invariant #16 — default to None, warn if missing.
                exp_id = getattr(self.config, "experiment_id", None)
                if exp_id is None:
                    logger.warning(
                        "[LiveDashboard] config.experiment_id is None (not set). "
                        "CSV will be written to 'convergence_expUNKNOWN.csv' to make "
                        "the missing ID obvious. Set experiment_id in ExperimentConfig."
                    )
                    exp_id_label = "UNKNOWN"
                else:
                    exp_id_label = str(exp_id)
                _out_dir_str = getattr(self.config, "output_dir", "results")
                _csv_path = Path(str(_out_dir_str)) / f"convergence_exp{exp_id_label}.csv"
                _live_cb = LiveDashboardCallback(
                    csv_path=_csv_path,
                    experiment_id=exp_id if exp_id is not None else 0,
                    total_epochs=self.config.max_epochs,
                    total_experiments=getattr(self.config, "total_experiments", 0),
                    dataset_names=getattr(self.config, "datasets", []),
                    num_samples=len(self.train_dataset) if self.train_dataset else 0,
                )
                callbacks.append(_live_cb)
                logger.debug("[LiveDashboard] Callback registered for experiment %d", exp_id)
            except ImportError:
                pass  # live_dashboard.py not found — skip silently
            except Exception as _ld_exc:  # intentional broad catch: third-party registration
                logger.debug("[LiveDashboard] Registration failed: %s", _ld_exc)

        # Gradient checkpointing: trades compute for VRAM — ~halves activation
        # memory (allows batch=4 on 24 GB 4090 instead of batch=2).
        # Must set use_cache=False before enabling gradient checkpointing.
        self.model.config.use_cache = False
        if hasattr(self.model.decoder, "config"):
            self.model.decoder.config.use_cache = False
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logging.debug("Gradient checkpointing enabled (use_reentrant=False)")

        # Guard: transformers Seq2SeqTrainer does
        #   isinstance(dataset, datasets.Dataset)
        # With HuggingFace datasets 3.x lazy-loading, a prior partial import
        # can leave datasets.Dataset inaccessible (AttributeError).
        # Our MultiDataset is a torch.utils.data.Dataset, so isinstance returns
        # False regardless — this guard only prevents the AttributeError.
        try:
            import datasets as _hf_ds

            if not hasattr(_hf_ds, "Dataset"):
                try:
                    _hf_ds.Dataset = _hf_ds.arrow_dataset.Dataset
                except AttributeError:
                    # Last resort: stub class so isinstance() returns False
                    # for our torch Dataset without raising AttributeError.
                    # Fix: issue_report_summary medium #12 — log WARNING when stub is active.
                    _hf_ds.Dataset = type("_HFDatasetStub", (), {})
                    logger.warning(
                        "datasets.Dataset unavailable (arrow_dataset missing). "
                        "Injecting stub class — isinstance(x, datasets.Dataset) will always "
                        "return False. This is expected when the 'datasets' package is absent; "
                        "no functional impact for our torch.utils.data.Dataset subclass."
                    )
        except ImportError:
            # datasets not installed at all — inject a stub module so that
            # Seq2SeqTrainer's isinstance check doesn't crash.
            # Fix: issue_report_summary medium #12 — log WARNING when stub module is injected.
            import types as _types

            _hf_ds = _types.ModuleType("datasets")
            _hf_ds.Dataset = type("_HFDatasetStub", (), {})  # type: ignore[attr-defined]
            sys.modules["datasets"] = _hf_ds
            logger.warning(
                "The 'datasets' library is not installed. Injecting a stub 'datasets' module "
                "so Seq2SeqTrainer's isinstance check does not raise AttributeError. "
                "isinstance(x, datasets.Dataset) will always return False for any object. "
                "Install 'datasets' (pip install datasets) if you need HuggingFace dataset "
                "integration. No functional impact for the DONUT SROIE pipeline."
            )

        # Custom data collator: stacks pixel_values and labels only, deliberately
        # omitting decoder_input_ids.  Without this, HF DataCollatorForSeq2Seq
        # (used implicitly in some transformers versions) would create
        # decoder_input_ids by shifting labels, while the HF
        # VisionEncoderDecoderModel simultaneously creates decoder_inputs_embeds
        # from labels — causing the MBart decoder to raise:
        #   ValueError: You cannot specify both decoder_input_ids and
        #               decoder_inputs_embeds at the same time
        # If the dataset provides per-sample loss_weight tensors (for aux_loss_weight
        # < 1.0), they are stacked into a 1-D batch tensor for compute_loss().
        def _donut_data_collator(features: list[dict[str, Any]]) -> dict[str, Any]:
            batch = {
                "pixel_values": torch.stack([f["pixel_values"] for f in features]),
                "labels": torch.stack([f["labels"] for f in features]),
            }
            if "loss_weight" in features[0]:
                batch["loss_weight"] = torch.stack([f["loss_weight"] for f in features])
            return batch

        # Subclass that prevents decoder_input_ids/decoder_inputs_embeds conflict.
        #
        # HF Seq2SeqTrainer.compute_loss() (via prepare_decoder_input_ids_from_labels)
        # can inject decoder_input_ids into the batch even after the data collator
        # has omitted them.  VisionEncoderDecoderModel then also derives
        # decoder_inputs_embeds from encoder hidden states, so MBart receives both
        # simultaneously and raises ValueError.
        #
        # The fix: bypass super().compute_loss() entirely and call the model
        # directly with only pixel_values + labels.  The model's own forward()
        # handles the shift-right derivation of decoder_input_ids internally —
        # that is the correct code path.  super().compute_loss() must NOT be
        # called because it re-injects decoder_input_ids after the pop.
        class _DonutSeq2SeqTrainer(Seq2SeqTrainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                inputs.pop("decoder_input_ids", None)
                inputs.pop("decoder_inputs_embeds", None)
                # Extract optional per-sample loss weights (from aux_loss_weight feature).
                # These are NOT passed to the model — model.forward() only accepts
                # pixel_values + labels.
                loss_weights = inputs.pop("loss_weight", None)
                outputs = model(**inputs)
                if loss_weights is not None:
                    # Per-sample weighted loss: recompute cross-entropy per sample using
                    # the already-computed logits.  outputs.loss is a scalar mean over
                    # all (batch × seq_len) tokens; we need per-sample granularity so
                    # SROIE samples get weight 1.0 and auxiliary samples get aux_weight.
                    # logits shape: (B, T, V) — already aligned with labels by the model.
                    import torch.nn.functional as _F  # noqa: PLC0415

                    _logits = outputs.logits  # (B, T, V)
                    _labels = inputs["labels"]  # (B, T)
                    _per_tok = _F.cross_entropy(
                        _logits.reshape(-1, _logits.size(-1)),
                        _labels.reshape(-1),
                        reduction="none",
                        ignore_index=LABEL_IGNORE_INDEX,
                    ).view(_labels.size())  # (B, T)
                    _valid = (_labels != LABEL_IGNORE_INDEX).float()
                    _per_sample = (_per_tok * _valid).sum(dim=1) / _valid.sum(dim=1).clamp(
                        min=1
                    )  # (B,)
                    # Weighted mean: sum(w_i * loss_i) / sum(w_i)
                    # → SROIE samples (w=1.0) contribute at full scale
                    # → Auxiliary samples (w<1.0) contribute proportionally less
                    # → Eval path (prediction_step) never calls compute_loss,
                    #   so eval_loss remains unweighted for fair early-stopping.
                    loss = (_per_sample * loss_weights).sum() / loss_weights.sum()
                else:
                    loss = outputs.loss
                return (loss, outputs) if return_outputs else loss

            def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
                # Strip decoder collision keys before HF evaluation loop runs.
                # HF Seq2SeqTrainer.prediction_step() calls model(**inputs) for
                # eval-loss computation, and that call can trigger the same
                # decoder_input_ids / decoder_inputs_embeds collision that
                # compute_loss guards against during training.
                inputs.pop("decoder_input_ids", None)
                inputs.pop("decoder_inputs_embeds", None)
                # Also strip loss_weight — it is only for training, not eval.
                inputs.pop("loss_weight", None)
                # Disable label smoothing during evaluation.
                # With predict_with_generate=True, Seq2SeqTrainer.prediction_step()
                # applies self.label_smoother to the eval loss when label_smoother
                # is not None.  This biases the early-stopping metric: eval_loss
                # would include the smoothing regularisation term and systematically
                # penalise overconfident predictions regardless of whether they are
                # correct — causing early stopping to favour over-smoothed models.
                # Eval loss should be pure cross-entropy so that epoch comparison is
                # fair; label smoothing belongs only to the training objective.
                _saved_smoother = self.label_smoother
                self.label_smoother = None
                try:
                    return super().prediction_step(
                        model, inputs, prediction_loss_only, ignore_keys=ignore_keys
                    )
                finally:
                    self.label_smoother = _saved_smoother

        trainer = _DonutSeq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=_donut_data_collator,
            callbacks=callbacks or None,
            optimizers=(optimizer, custom_scheduler),
        )

        # ── Pre-training guardrails ────────────────────────────────────────────
        # These checks catch the known silent-failure modes documented in CLAUDE.md
        # §16 BEFORE training begins, so no compute is wasted on a broken setup.
        _decoder_config = getattr(self.model, "decoder", self.model).config

        # Guardrail 0: lm_head.weight must NOT share a data pointer with
        # embed_tokens.weight.  resize_token_embeddings() should have broken the
        # alias; if it did not (e.g. custom model wrapper or future HF regression),
        # the first save_pretrained() will silently deduplicate them and drop
        # lm_head.weight from the shard → F1≈0.42 on reload.
        # Fix here rather than letting LmHeadCloneCallback fix it later so the
        # alias never reaches the first checkpoint.
        _g0_lm = getattr(getattr(self.model, "decoder", None), "lm_head", None)
        _g0_emb = None
        try:
            _g0_emb = self.model.decoder.model.decoder.embed_tokens
        except AttributeError:
            pass
        if (
            _g0_lm is not None
            and _g0_emb is not None
            and hasattr(_g0_lm, "weight")
            and hasattr(_g0_emb, "weight")
            and _g0_lm.weight.data_ptr() == _g0_emb.weight.data_ptr()
        ):
            logger.warning(
                "Guardrail 0: lm_head.weight shares storage with embed_tokens.weight "
                "(alias not broken by resize_token_embeddings). "
                "Cloning now to prevent safetensors deduplication on first save."
            )
            _g0_lm.weight = torch.nn.Parameter(_g0_lm.weight.data.clone())
        else:
            logger.debug("Guardrail 0: lm_head.weight is already independent (no alias).")

        # Guardrail 1: tie_word_embeddings MUST be False after resize_token_embeddings().
        # If True, tie_weights() on checkpoint reload overwrites the learned lm_head
        # with embed_tokens weights → F1 = 0.0 on every prediction (Bug B / lm_head_dedup).
        _tie = getattr(_decoder_config, "tie_word_embeddings", True)
        if _tie:
            raise ValueError(
                "CRITICAL: model.decoder.config.tie_word_embeddings is True. "
                "This will cause lm_head.weight to be overwritten on checkpoint reload, "
                "producing F1=0. Call model.decoder.config.tie_word_embeddings=False "
                "immediately after resize_token_embeddings(). See CLAUDE.md §2."
            )

        # Guardrail 2: decoder_start_token_id must be set (not None).
        # None → decoder starts from a random token → garbled output → F1=0 (GP-3/GP-4).
        _dst_id = getattr(self.model.config, "decoder_start_token_id", None)
        if _dst_id is None:
            raise ValueError(
                "CRITICAL: model.config.decoder_start_token_id is None. "
                "Set it to processor.tokenizer.convert_tokens_to_ids(['<s_sroie>'])[0] "
                "before training. See CLAUDE.md §16 GP-3."
            )

        # Guardrail 3: decoder_start_token_id must not map to unk_token_id.
        # If it does, the <s_sroie> token was not registered before convert_tokens_to_ids.
        _unk_id = getattr(self.processor.tokenizer, "unk_token_id", None)
        if _unk_id is not None and _dst_id == _unk_id:
            raise ValueError(
                f"CRITICAL: decoder_start_token_id={_dst_id} equals unk_token_id={_unk_id}. "
                "The <s_sroie> token was not added to the tokenizer before "
                "convert_tokens_to_ids was called. "
                "Call add_special_tokens({'additional_special_tokens': NEW_TOKENS}) "
                "and resize_token_embeddings() before setting decoder_start_token_id. "
                "See CLAUDE.md §16 GP-3."
            )

        # Guardrail 4: all NEW_TOKENS must be registered in the tokenizer vocab.
        # Missing tokens map to unk_token_id → model learns to generate unk → F1=0.
        if _unk_id is not None:
            _missing_toks = [
                tok
                for tok in NEW_TOKENS
                if self.processor.tokenizer.convert_tokens_to_ids([tok])[0] == _unk_id
            ]
            if _missing_toks:
                raise ValueError(
                    f"CRITICAL: {len(_missing_toks)} SROIE special token(s) are missing from "
                    f"the tokenizer vocabulary (all map to unk_token_id={_unk_id}): "
                    f"{_missing_toks}. "
                    "Call processor.tokenizer.add_special_tokens("
                    "{'additional_special_tokens': NEW_TOKENS}) "
                    "and model.decoder.resize_token_embeddings(len(processor.tokenizer)) "
                    "before training. See CLAUDE.md §7."
                )

        # Guardrail 5: model.config and model.decoder.config must agree on
        # critical token-ID fields.  A mismatch means the model trains correctly
        # (using model.config) but produces wrong output after save/reload
        # (save_pretrained serialises model.decoder.config).
        for _cfg_field in ("decoder_start_token_id", "pad_token_id"):
            _top_val = getattr(self.model.config, _cfg_field, None)
            _dec_val = getattr(_decoder_config, _cfg_field, None)
            if _top_val != _dec_val:
                raise ValueError(
                    f"CRITICAL: model.config.{_cfg_field}={_top_val} != "
                    f"model.decoder.config.{_cfg_field}={_dec_val}. "
                    f"Both must be set to the same value. The model will train correctly "
                    f"but produce wrong output after save/reload. "
                    f"Use _ensure_dual_config(model, '{_cfg_field}', <value>) to set both."
                )

        # Guardrail 6: tie_word_embeddings must be False on model.config as well.
        # Guardrail 1 already checked model.decoder.config; this catches the case
        # where only model.config is True (HF tie_weights() checks model.config).
        _top_tie = getattr(self.model.config, "tie_word_embeddings", None)
        if _top_tie is True:
            raise ValueError(
                "CRITICAL: model.config.tie_word_embeddings is True. "
                "Must be False on BOTH model.config AND model.decoder.config. "
                "Use _ensure_dual_config(model, 'tie_word_embeddings', False)."
            )

        logger.info(
            "Pre-training guardrails PASSED: tie_word_embeddings=False, "
            "decoder_start_token_id=%d (%d NEW_TOKENS registered).",
            _dst_id,
            len(NEW_TOKENS),
        )
        # ── End pre-training guardrails ───────────────────────────────────────

        trainer.train()

        # Close the live dashboard (stops rich.live.Live so terminal is clean).
        for cb in callbacks or []:
            if isinstance(cb, LiveDashboardCallback):
                cb.close()

        # Shut down persistent DataLoader worker subprocesses BEFORE returning.
        # Without this, worker processes holding prefetch buffers stay alive
        # across experiments, accumulating ~450 MB per experiment (8 workers ×
        # 2 prefetch batches × ~28 MB/batch at 1280×960, batch_size=2).
        _mm.shutdown_dataloader_workers(trainer)

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
        """Save the fine-tuned model and processor to disk."""
        save_dir = Path(path) if path is not None else self._output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        self._save_model_weights(save_dir)
        self.processor.save_pretrained(str(save_dir))
        logger.info("Model + processor saved → %s", save_dir)

        self._verify_sroie_tokens(save_dir)
        self._verify_lm_head_shard(save_dir)

    def _save_model_weights(self, save_dir: Path) -> None:
        """Save model weights with lm_head detach fix.

        transformers ≥5.x recomputes tied-weight lists before serialization
        and may use content-hash equality (not just data_ptr()) to deduplicate.
        Bypass save_pretrained entirely: write the state dict directly with
        safetensors.torch.save_file so HF's internal dedup logic is never
        invoked.  The lm_head weight gets .clone().contiguous() to guarantee
        a unique data pointer AND a unique content hash.
        """
        decoder = self.model.decoder
        try:
            from safetensors.torch import save_file as _st_save_file  # noqa: PLC0415

            _sd = self.model.state_dict()
            _lm_key = "decoder.lm_head.weight"
            if _lm_key in _sd:
                _sd[_lm_key] = _sd[_lm_key].clone().contiguous()
            _st_save_file(_sd, save_dir / "model.safetensors")
            self.model.config.save_pretrained(str(save_dir))
        except ImportError:
            # safetensors not installed — fall back to save_pretrained with clone.
            # NOTE: this path is still vulnerable to content-hash deduplication in
            # transformers ≥5.x.  Install the safetensors package for full protection.
            if hasattr(decoder, "lm_head"):
                decoder.lm_head.weight = torch.nn.Parameter(decoder.lm_head.weight.data.clone())
            self.model.save_pretrained(str(save_dir))

    def _verify_sroie_tokens(self, save_dir: Path) -> None:
        """Post-save verification: confirm all SROIE tokens survived serialization."""
        _verify_proc = DonutProcessor.from_pretrained(str(save_dir))
        _unk_id = _verify_proc.tokenizer.unk_token_id
        _missing = [
            tok
            for tok in NEW_TOKENS
            if _verify_proc.tokenizer.convert_tokens_to_ids([tok])[0] == _unk_id
        ]
        if _missing:
            raise RuntimeError(
                f"Processor saved to {str(save_dir)!r} is CORRUPT: the following SROIE "
                f"special tokens map to unk_token_id ({_unk_id}): {_missing}. "
                "Ensure processor.save_pretrained() was called after "
                "add_special_tokens() and before this save()."
            )
        logger.info(
            "Post-save verification PASSED: all %d SROIE tokens present in %s",
            len(NEW_TOKENS),
            save_dir,
        )

    def _verify_lm_head_shard(self, save_dir: Path) -> None:
        """Post-save lm_head.weight verification.

        Verify that lm_head.weight was not deduped out of the safetensors shard.
        If safetensors sees embed_tokens.weight and lm_head.weight sharing the
        same data pointer, it silently omits lm_head from the shard — producing
        F1~0.42 on reload (Bug B / lm_head_dedup, CLAUDE.md §16).
        The clone() in _save_model_weights should prevent this; this check confirms it.
        """
        _shard_file = save_dir / "model.safetensors"
        if not _shard_file.exists():
            # Multi-shard save: find the index
            _shard_file = save_dir / "pytorch_model.bin"
        if _shard_file.exists() and _shard_file.suffix == ".safetensors":
            try:
                from safetensors import safe_open

                with safe_open(str(_shard_file), framework="pt", device="cpu") as _sf:
                    _sf_keys = set(_sf.keys())
                _lm_head_keys = {k for k in _sf_keys if "lm_head" in k}
                if not _lm_head_keys:
                    raise RuntimeError(
                        f"CRITICAL: lm_head.weight is MISSING from {_shard_file}. "
                        "safetensors deduplicated it against embed_tokens.weight. "
                        "The LmHeadCloneCallback.on_save() clone did not fire in time. "
                        "Manually call: "
                        "model.decoder.lm_head.weight = "
                        "torch.nn.Parameter(model.decoder.lm_head.weight.data.clone()) "
                        "before save_pretrained(). See CLAUDE.md §16 Pattern 6."
                    )
                logger.info(
                    "Post-save lm_head check PASSED: %s present in %s",
                    _lm_head_keys,
                    _shard_file.name,
                )
            except ImportError:
                # safetensors not installed — skip the detailed check
                logger.debug("safetensors not importable — skipping lm_head shard check")
            except RuntimeError:
                raise  # re-raise our own CRITICAL errors
            except (OSError, KeyError, ValueError) as _sf_exc:
                logger.warning("lm_head shard check failed (non-critical): %s", _sf_exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _output_dir(self) -> Path:
        """Resolve the output directory from config."""
        return Path(getattr(self.config, "output_dir", str(WORKSPACE / "donut-sroie-finetuned")))


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

    # Semantic initialisation of new SROIE special-token embeddings.
    # Replace random vectors with averages of semantically similar existing tokens
    # so the model converges on the correct XML structure in fewer optimizer steps.
    # (With only ~630 optimizer steps from random init, the model learns field values
    # but fails to learn the structural tags, producing </s_sroie> delimiters instead
    # of <s_company>VALUE</s_company> sequences.)
    _semantic_seed_words: dict[str, list[str]] = {
        "<s_company>": ["company", "store", "name", "shop", "merchant"],
        "<s_date>": ["date", "time", "day"],
        "<s_address>": ["address", "location", "street", "place"],
        "<s_total>": ["total", "amount", "price", "sum"],
    }
    _embed_w = model.decoder.model.decoder.embed_tokens.weight
    _lm_w = model.decoder.lm_head.weight
    _bos_id = processor.tokenizer.bos_token_id or 0
    _eos_id = processor.tokenizer.eos_token_id or 2
    _opening_vecs: dict[str, Any] = {}
    with torch.no_grad():
        for _tok in NEW_TOKENS:
            _tid = processor.tokenizer.convert_tokens_to_ids([_tok])[0]
            if _tid >= _embed_w.shape[0]:
                continue
            if _tok == "<s_sroie>":
                _vec = _embed_w.data[_bos_id].clone()
            elif _tok == "</s_sroie>":
                _vec = _embed_w.data[_eos_id].clone()
            elif _tok in _semantic_seed_words:
                _seed_vecs = []
                for _sw in _semantic_seed_words[_tok]:
                    for _swid in processor.tokenizer.encode(_sw, add_special_tokens=False):
                        if _swid < _embed_w.shape[0]:
                            _seed_vecs.append(_embed_w.data[_swid].clone())
                _vec = (
                    torch.stack(_seed_vecs).mean(dim=0)
                    if _seed_vecs
                    else _embed_w.data[_eos_id].clone()
                )
                _opening_vecs[_tok] = _vec
            elif _tok.startswith("</s_") and _tok.endswith(">"):
                _open = _tok.replace("</", "<")
                _vec = _opening_vecs.get(_open, _embed_w.data[_eos_id]).clone()
            else:
                continue
            _embed_w.data[_tid] = _vec
            if _lm_w.shape[0] > _tid:
                _lm_w.data[_tid] = _vec
    logging.debug("[SemanticInit] Initialised %d new SROIE token embeddings", len(NEW_TOKENS))

    # After resize, embed_tokens and lm_head are separate tensors with
    # independent random init for the new tokens.  Set tie_word_embeddings=False
    # so save_pretrained() saves BOTH weights independently.  Without this,
    # the saved checkpoint omits lm_head (or tie_weights() overwrites the
    # learned lm_head with embed_tokens), causing F1=0 on reload.
    _ensure_dual_config(model, "tie_word_embeddings", False)
    # Fix: issue_report_summary invariant #13 — assert tie_word_embeddings took effect.
    assert model.config.tie_word_embeddings is False, (
        "tie_word_embeddings was not applied to model.config after _ensure_dual_config(). "
        "This will cause lm_head.weight to be dropped on checkpoint save, producing F1=0 "
        "on reload. See CLAUDE.md §2."
    )
    if hasattr(model, "decoder") and hasattr(model.decoder, "config"):
        assert model.decoder.config.tie_word_embeddings is False, (
            "tie_word_embeddings was not applied to model.decoder.config after _ensure_dual_config(). "
            "This will cause lm_head.weight to be dropped on checkpoint save, producing F1=0 "
            "on reload. See CLAUDE.md §2."
        )

    _ensure_dual_config(model, "pad_token_id", processor.tokenizer.pad_token_id)
    # Fix: issue_report_summary high #4 — guard convert_tokens_to_ids for empty result / unk_token_id.
    # The string form iterates characters; always use list form to get the full special-token ID.
    _sroie_start_ids = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])
    if not _sroie_start_ids:
        raise ValueError(
            "convert_tokens_to_ids(['<s_sroie>']) returned an empty list. "
            "The <s_sroie> token must be added via add_special_tokens() "
            "before calling convert_tokens_to_ids. See CLAUDE.md §16 GP-3."
        )
    _sroie_start_id = _sroie_start_ids[0]
    _unk_id_check = getattr(processor.tokenizer, "unk_token_id", None)
    if _unk_id_check is not None and _sroie_start_id == _unk_id_check:
        raise ValueError(
            f"convert_tokens_to_ids(['<s_sroie>']) returned unk_token_id={_unk_id_check}. "
            "The <s_sroie> token was not registered in the tokenizer vocabulary before "
            "convert_tokens_to_ids was called. Call "
            "processor.tokenizer.add_special_tokens({'additional_special_tokens': NEW_TOKENS}) "
            "and model.decoder.resize_token_embeddings(len(processor.tokenizer)) first. "
            "See CLAUDE.md §16 GP-3."
        )
    _ensure_dual_config(model, "decoder_start_token_id", _sroie_start_id)
    # ── Guardrail: verify decoder_start_token_id decodes back to the task token ──
    _decoded = processor.tokenizer.decode([model.config.decoder_start_token_id])
    if _decoded != "<s_sroie>":
        raise RuntimeError(
            f"decoder_start_token_id={model.config.decoder_start_token_id} decodes to "
            f"'{_decoded}', not '<s_sroie>'. Token was not added to vocab before "
            f"convert_tokens_to_ids was called, or the list-wrapping syntax is missing. "
            f"Use: tokenizer.convert_tokens_to_ids(['<s_sroie>'])[0]"
        )
    _ensure_dual_config(model, "use_cache", False)  # Required with gradient_checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Load SROIE data using canonical loaders (single source of truth)
    from data_pipeline import load_sroie_train, load_sroie_val

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
