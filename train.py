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
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

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
        log_history: list = field(default_factory=list)
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
            **kwargs,
        ):
            self.model = model
            self.args = args or Seq2SeqTrainingArguments()
            self.train_dataset = train_dataset
            self.eval_dataset = eval_dataset
            self.callbacks = callbacks or []
            self._optimizer, self._scheduler = optimizers
            self.state = _TrainerState()

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
            )
            eval_loader = None
            if self.eval_dataset is not None and args.eval_strategy != "no":
                eval_loader = _DataLoader(
                    self.eval_dataset,
                    batch_size=args.per_device_eval_batch_size,
                    shuffle=False,
                    num_workers=args.dataloader_num_workers,
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
                        optimizer.zero_grad()
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
            self._size = {"height": 1280, "width": 960}
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

            target_h = self._size.get("height", 1280)
            target_w = self._size.get("width", 960)

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

        def add_special_tokens(self, tokens: dict | list) -> int:
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

        def token2json(self, tokens: str, **kwargs) -> dict:
            """Parse <s_field>VALUE</s_field> sequences into a dict."""
            import re as _re

            result: dict = {}
            for field, value in _re.findall(r"<s_(\w+)>(.*?)</s_\1>", tokens, _re.DOTALL):
                result[field] = value.strip()
            return result

    # ── VisionEncoderDecoderModel placeholder ─────────────────────────────

    class VisionEncoderDecoderModel:  # type: ignore[no-redef]
        """Placeholder — requires transformers or the inline Swin+BART implementation.

        Install transformers to use: pip install transformers
        The full inline Swin+BART architecture is planned for a future phase.
        """

        @classmethod
        def from_pretrained(cls, model_name_or_path, *args, **kwargs):
            raise ImportError(
                "transformers >= 4.37.0 is required but not installed.\n"
                "Run: pip install transformers>=4.37.0\n"
                "Or:  pip install -r requirements.txt"
            )

        @staticmethod
        def from_encoder_decoder_pretrained(*args, **kwargs):
            raise ImportError(
                "transformers >= 4.37.0 is required but not installed.\n"
                "Run: pip install transformers>=4.37.0\n"
                "Or:  pip install -r requirements.txt"
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

    def _png_unfilter(scanlines: list, width: int, bpp: int) -> bytes:
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

    def _load_jpeg_ctypes(path: str | Path):
        """Load JPEG via ImageMagick subprocess (system libjpeg fallback)."""
        import numpy as np

        try:
            import subprocess as _sp

            result = _sp.run(
                ["convert", str(path), "-colorspace", "RGB", "ppm:-"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                ppm = result.stdout
                # Parse PPM header
                lines = ppm.split(b"\n")
                if lines[0] == b"P6":
                    dims = lines[1].split()
                    w, h = int(dims[0]), int(dims[1])
                    pixel_data = b"\n".join(lines[3:])
                    arr = np.frombuffer(pixel_data, dtype=np.uint8)
                    if len(arr) >= h * w * 3:
                        return arr[: h * w * 3].reshape(h, w, 3)
        except Exception:
            pass
        return None

    def _load_image(path: str | Path):  # type: ignore[misc]
        """Load an image file as an RGB numpy array without PIL.

        Supports PNG (full), BMP (24-bit), and JPEG (via imagemagick/libjpeg).
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
            arr = _load_jpeg_ctypes(path)
        if arr is None:
            raise RuntimeError(
                f"Cannot load {path} without Pillow. "
                "Install Pillow: pip install Pillow\n"
                "PNG (8-bit RGB/RGBA), BMP (24-bit), and JPEG (via ImageMagick) "
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


def _load_safetensors(path: str | Path) -> dict:
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


import memory_manager as _mm  # noqa: E402

# FIX: Previously FIELDS, MAX_LENGTH, IMAGE_EXTS, NEW_TOKENS, BASE_MODEL,
# SEED were defined independently here and in 4 other files, risking silent
# drift if any file was updated without updating the others.
from constants import (  # noqa: E402
    BASE_MODEL,
    FIELDS,
    MAX_LENGTH,
    NEW_TOKENS,
    SEED,
    WORKSPACE,
    _mask_empty_field_labels,
    _optimal_num_workers,
)

__all__ = ["SROIEDataset", "MultiDataset", "DonutTrainer", "TrainingResult"]

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

    def __init__(self, sroie_val_samples, processor, output_dir: Path):
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
            from constants import FIELDS

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
                    except Exception:
                        total += len(FIELDS)  # count as all wrong on error
            f1 = correct / total if total > 0 else 0.0
        except Exception as exc:
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
        except Exception as exc:
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
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
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
    ):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self._image_cache: dict[int, Image.Image] = {}
        self._pixel_cache: dict[int, Any] = {}  # precomputed pixel_values tensors
        self._label_cache: dict[int, Any] = {}  # precomputed label token tensors

        if cache_in_ram and len(samples) > 0:
            # Determine actual image dimensions from processor_config.json so
            # the RAM estimate is correct at any resolution.
            # The old hardcoded `* 3` (3 MB/sample) was wrong at 2560×1920
            # (actual: 14.06 MB/sample) causing the gate to open when it should
            # be closed. memory_manager.ram_cache_is_safe() uses the real formula:
            #   3 × H × W / 1_048_576 MB per sample.
            try:
                from resource_optimizer import get_image_size_from_processor_config

                _img_h, _img_w = get_image_size_from_processor_config()
            except Exception:
                _img_h, _img_w = 1280, 960  # safe fallback to DONUT native resolution
            if _mm.ram_cache_is_safe(len(samples), _img_h, _img_w):
                import concurrent.futures

                def _load_one(idx_path):
                    idx, path = idx_path
                    try:
                        return idx, _load_image(path)
                    except Exception:
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
                            except Exception:
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
                    _lbl[_lbl == processor.tokenizer.pad_token_id] = -100
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

    def __getitem__(self, idx: int) -> dict:
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
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels = _mask_empty_field_labels(labels, gt, self.processor.tokenizer)

        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# LiveDashboardCallback (inlined from live_dashboard.py)
# ---------------------------------------------------------------------------


@dataclass
class _EpochRow:
    epoch: int
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    best_f1: float = float("nan")


class LiveDashboardCallback:
    """Trainer callback that writes CSV rows and optionally redraws a rich table.

    Compatible with HuggingFace ``TrainerCallback`` interface: the callback
    class inherits from ``transformers.TrainerCallback`` lazily (at __init__
    time) to avoid importing transformers at module level.
    """

    def __init__(
        self,
        csv_path: "str | Path | None" = None,
        experiment_id: int = 0,
        use_rich: "bool | None" = None,
    ) -> None:
        try:
            from transformers import TrainerCallback

            self.__class__ = type(
                "LiveDashboardCallback",
                (self.__class__, TrainerCallback),
                {},
            )
        except ImportError:
            pass

        if csv_path is None:
            csv_path = f"convergence_exp{experiment_id}.csv"
        self._csv_path = Path(csv_path)
        self._experiment_id = experiment_id
        self._rows: list[_EpochRow] = []
        self._best_f1: float = float("nan")

        if use_rich is None:
            use_rich = os.environ.get("DISABLE_LIVE_DASHBOARD", "0") != "1"

        self._rich_enabled = False
        if use_rich:
            try:
                import rich  # noqa: F401

                self._rich_enabled = True
            except ImportError:
                pass

        if not self._csv_path.exists():
            try:
                self._csv_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._csv_path, "w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["epoch", "train_loss", "val_loss", "best_f1"])
            except OSError as exc:
                logger.warning("[LiveDashboard] Could not create CSV %s: %s", self._csv_path, exc)

    def on_epoch_end(self, args: "Any", state: "Any", control: "Any", **kwargs: "Any") -> None:
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
        if self._rich_enabled:
            self._redraw()

    def update_best_f1(self, f1: float) -> None:
        if f1 > self._best_f1 or self._best_f1 != self._best_f1:
            self._best_f1 = f1
            if self._rows:
                self._rows[-1].best_f1 = f1

    def _redraw(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(
                title=f"Experiment {self._experiment_id} — Training Progress",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Epoch", justify="right", style="dim")
            table.add_column("Train Loss", justify="right")
            table.add_column("Val Loss", justify="right")
            table.add_column("Best F1", justify="right", style="green")
            for r in self._rows[-10:]:
                tl = f"{r.train_loss:.4f}" if r.train_loss == r.train_loss else "—"
                vl = f"{r.val_loss:.4f}" if r.val_loss == r.val_loss else "—"
                bf = f"{r.best_f1:.4f}" if r.best_f1 == r.best_f1 else "—"
                table.add_row(str(r.epoch), tl, vl, bf)
            console.print(table)
        except Exception as exc:
            logger.debug("[LiveDashboard] rich redraw failed: %s", exc)

    def close(self) -> None:
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
                _cfg_warmup,
                _eff_warmup,
                _total_opt_steps,
                self.config.max_epochs,
            )

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
            dataloader_pin_memory=optimal_workers > 0,
            dataloader_prefetch_factor=2 if optimal_workers > 0 else None,
            dataloader_persistent_workers=optimal_workers > 0,
            remove_unused_columns=False,
            seed=getattr(self.config, "seed", SEED),
        )

        # Build layerwise AdamW optimizer: encoder at encoder_lr, decoder at decoder_lr.
        # Falls back to a single learning_rate if encoder_lr/decoder_lr are absent.
        encoder_lr = getattr(self.config, "encoder_lr", self.config.learning_rate)
        decoder_lr = getattr(self.config, "decoder_lr", self.config.learning_rate)
        _weight_decay = getattr(self.config, "weight_decay", 0.01)
        encoder_params = [
            p
            for n, p in self.model.named_parameters()
            if n.startswith("encoder.") and p.requires_grad
        ]
        decoder_params = [
            p
            for n, p in self.model.named_parameters()
            if not n.startswith("encoder.") and p.requires_grad
        ]
        _optimizer_type = getattr(self.config, "optimizer_type", "adamw")
        if _optimizer_type == "sgd":
            # SGD + Nesterov: faster per-step, sufficient for near-converged transformers
            # used in micro mode where adaptive moments aren't needed for short runs
            optimizer = torch.optim.SGD(
                [
                    {"params": encoder_params, "lr": encoder_lr},
                    {"params": decoder_params, "lr": decoder_lr},
                ],
                momentum=0.9,
                nesterov=True,
                weight_decay=_weight_decay,
            )
            logger.info("Optimizer: SGD + Nesterov (micro/mini mode)")
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": encoder_params, "lr": encoder_lr},
                    {"params": decoder_params, "lr": decoder_lr},
                ],
                weight_decay=_weight_decay,
            )

        # OneCycleLR: aggressive warmup + cosine decay, reaches peak LR immediately
        # — much faster convergence than cosine+warmup for short (2–3 epoch) micro runs
        _lr_schedule = getattr(self.config, "lr_schedule", "cosine")
        custom_scheduler = None
        if _lr_schedule == "one_cycle" and _total_opt_steps > 0:
            custom_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[encoder_lr, decoder_lr],
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

        # LiveDashboardCallback — auto-registered when rich is installed.
        # Writes per-epoch CSV and optionally redraws a rich table.
        # Disabled by setting env var DISABLE_LIVE_DASHBOARD=1.
        if os.environ.get("DISABLE_LIVE_DASHBOARD", "0") != "1":
            try:
                exp_id = getattr(self.config, "experiment_id", 0)
                _out_dir_str = getattr(self.config, "output_dir", "results")
                _csv_path = Path(str(_out_dir_str)) / f"convergence_exp{exp_id}.csv"
                _live_cb = LiveDashboardCallback(
                    csv_path=_csv_path,
                    experiment_id=exp_id,
                )
                callbacks.append(_live_cb)
                logger.debug("[LiveDashboard] Callback registered for experiment %d", exp_id)
            except ImportError:
                pass  # live_dashboard.py not found — skip silently
            except Exception as _ld_exc:
                logger.debug("[LiveDashboard] Registration failed: %s", _ld_exc)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            callbacks=callbacks or None,
            optimizers=(optimizer, custom_scheduler),
        )

        trainer.train()

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

        # ── Post-save verification: confirm all SROIE tokens survived serialization ──
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
    model.decoder.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        ["<s_sroie>"]
    )[0]
    # ── Guardrail: verify decoder_start_token_id decodes back to the task token ──
    _decoded = processor.tokenizer.decode([model.config.decoder_start_token_id])
    if _decoded != "<s_sroie>":
        raise RuntimeError(
            f"decoder_start_token_id={model.config.decoder_start_token_id} decodes to "
            f"'{_decoded}', not '<s_sroie>'. Token was not added to vocab before "
            f"convert_tokens_to_ids was called, or the list-wrapping syntax is missing. "
            f"Use: tokenizer.convert_tokens_to_ids(['<s_sroie>'])[0]"
        )
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
