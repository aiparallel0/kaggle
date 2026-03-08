# =============================================================================
# train_trocr_yolo.py
# Purpose: YOLO text-region detection + TrOCR OCR pipeline — training and inference
# Project: DONUT Receipt KIE — SROIE Fine-tuning & Benchmarking
# Updated: 2026-03-07
# =============================================================================
"""
train_trocr_yolo.py — TrOCR+YOLO two-stage training pipeline.

FIX: Previous version was a standalone script that trained a single YOLOv8
and a single TrOCR model.  This version provides functions callable from
run_all.py to run the SAME 8 dataset combinations as the DONUT experiments.
This ensures a fair, matched experimental design for cross-architecture
comparison.

Architecture:
  Stage 1: YOLOv8x detects text regions (~68.2M params)
  Stage 2: TrOCR-base reads text from crops (~246M params)
  Stage 3: Rule-based heuristics assign fields to extracted text

FIX: Added GPU cleanup between experiments.
FIX: Imports constants from shared module.
"""

import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_scheduler,
)

from constants import DEVICE, FIELDS, SEED, WORKSPACE, _gpu_cleanup, _optimal_num_workers
from control_suite import CONTROL_SUITE, get_augmentation_transforms

__all__ = [
    "TrOCRReceiptDataset",
    "train_yolo",
    "train_trocr",
    "run_trocr_yolo_inference",
    "_materialize_meta_buffers",
    "_EXPECTED_MISSING_TROCR",
    "_print_trocr_load_report",
    # Patchable constants (micro mode sets these before calling train_yolo/train_trocr)
    "YOLO_BASE",
    "YOLO_EPOCHS",
    "YOLO_IMG_SIZE",
    "YOLO_BATCH",
    "YOLO_OPTIMIZER",
    "YOLO_MOMENTUM",
    "TROCR_EPOCHS",
    "TROCR_BATCH",
    "TROCR_MAX_LEN",
    "TROCR_MINI_MODE",
]

# ── Config ──────────────────────────────────────────────────────────────────
TROCR_MODEL_ID = "microsoft/trocr-base-printed"
YOLO_BASE = "yolov8x.pt"  # extra-large YOLOv8 (~68M params); batch/imgsz kept low to fit in VRAM
YOLO_EPOCHS = 50
YOLO_IMG_SIZE = 512  # reduced from 640 to lower VRAM usage
YOLO_BATCH = 8  # reduced from 32 to prevent CUDA OOM in TaskAlignedAssigner
YOLO_AMP = True  # mixed precision — halves activation memory
YOLO_OPTIMIZER = "AdamW"  # micro mode patches to "SGD" for faster detection convergence
YOLO_MOMENTUM = 0.9  # used when YOLO_OPTIMIZER == "SGD"
TROCR_EPOCHS = 10
TROCR_BATCH = 16
TROCR_LR = 5e-5
TROCR_MAX_LEN = 128
GRAD_ACCUM = 4
TROCR_MINI_MODE = False  # True → SGD+Nesterov+CosineAnnealingLR instead of AdamW+linear

RESULTS_DIR = Path("results")
YOLO_DATA_YAML = WORKSPACE / "data" / "yolo" / "dataset.yaml"
TROCR_DATA_DIR = WORKSPACE / "data" / "trocr"

# Pre-compiled regex patterns for field assignment heuristics — compiled once
# at module load instead of on every call to assign_fields_heuristic().

# Date: numeric (DD/MM/YYYY, YYYY-MM-DD, etc.) OR written month names
_DATE_RE = re.compile(
    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"  # 25/12/2023, 25-12-23
    r"|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}"  # 2023/12/25
    r"|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{2,4}"  # 25 DEC 2023
    r"|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s+\d{1,2},?\s+\d{4}"  # DEC 25, 2023
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}",  # 25 Dec 2023
    re.IGNORECASE,
)
_TOTAL_RE = re.compile(
    r"(?:total|subtotal|amount|sum|due|grand\s*total|nett\s*total|net\s*total)\s*[:\-]?\s*[\$\£\€RM]?\s*\d+[.,]\d{2}",
    re.IGNORECASE,
)
# Matches a standalone monetary amount at end of line (last-resort total finder)
_MONEY_RE = re.compile(r"[\$\£\€RM]?\s*\d+[.,]\d{2}\s*$")
_NUMBER_RE = re.compile(r"[\d]+[.,][\d]{2}")
# Road/address keywords common in Malaysian/SE Asian receipts, plus generic
# English building references and postcode patterns.
_ADDRESS_RE = re.compile(
    r"\b(?:JALAN|JLN|LORONG|LRG|ROAD|STREET|ST|AVENUE|AVE|BOULEVARD|BLVD"
    r"|TAMAN|TMN|BANDAR|PUSAT|KOMPLEKS|NO\.?\s*\d|LOT\s*\d|\d{5}\s+[A-Z]"
    r"|FLOOR|LEVEL|UNIT|BLOCK|BLK)"
    r"|\b\d{5}\b"  # standalone 5-digit postcode
    r"|^\d+\s+[A-Z]",  # line starting with street number + word
    re.IGNORECASE | re.MULTILINE,
)


# ════════════════════════════════════════════════════════════════════════════
# TrOCR Dataset
# ════════════════════════════════════════════════════════════════════════════
class TrOCRReceiptDataset(Dataset):
    """Line crop dataset for TrOCR fine-tuning."""

    def __init__(
        self,
        data_dir: Path,
        processor: TrOCRProcessor,
        max_length: int,
        augmentation=None,  # Optional torchvision transforms pipeline (PIL Image → PIL Image)
    ):
        self.data_dir = data_dir
        self.processor = processor
        self.max_len = max_length
        self.augmentation = augmentation
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

        if self.augmentation is not None:
            img = self.augmentation(img)

        pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze(0)

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
# Meta-device buffer materialisation helper
# ════════════════════════════════════════════════════════════════════════════


def _materialize_meta_buffers(model: torch.nn.Module, device: str) -> int:
    """Walk all modules and force-materialise any remaining meta-device tensors.

    This covers:
      - Persistent buffers  (module._buffers)
      - Non-persistent buffers tracked in module._non_persistent_buffers_set
        (e.g. TrOCR's embed_positions._float_tensor)
      - Any plain tensor attributes that happen to be on 'meta'

    FIX: The existing partial fix (buffer sweep over module._buffers) misses
    non-persistent buffers such as TrOCR's sinusoidal positional embedding
    ``decoder.model.decoder.embed_positions._float_tensor``, which is
    registered via ``register_buffer(..., persistent=False)`` and therefore
    stored as a plain attribute rather than in ``_buffers``.  With
    ``low_cpu_mem_usage=False`` the weight tensors are materialised on CPU,
    but non-persistent buffers may still end up on the meta device causing:
        RuntimeError: Tensor on device meta is not on the expected device cuda:0!

    Returns the count of buffers that were fixed.
    """
    fixed = 0
    for module in model.modules():
        # --- Persistent and non-persistent buffers via _buffers dict ---
        for buf_name, buf in list(module._buffers.items()):
            if buf is not None and buf.device.type == "meta":
                module._buffers[buf_name] = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
                fixed += 1
        # --- Any plain tensor attributes (e.g. _float_tensor set directly) ---
        for attr_name, attr_val in list(vars(module).items()):
            if (
                (not attr_name.startswith("_") or attr_name == "_float_tensor")
                and isinstance(attr_val, torch.Tensor)
                and attr_val.device.type == "meta"
            ):
                try:
                    setattr(
                        module,
                        attr_name,
                        torch.zeros(attr_val.shape, dtype=attr_val.dtype, device=device),
                    )
                    fixed += 1
                except Exception:
                    pass
    return fixed


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1: YOLO Training
# ════════════════════════════════════════════════════════════════════════════
def train_yolo(output_dir: Path | None = None, num_train_samples: int = 0) -> Path:
    """Fine-tune YOLOv8 for text-region detection on receipts.

    Parameters
    ----------
    output_dir:
        Directory for model checkpoints and run artefacts.
    num_train_samples:
        Number of training samples; used to log a freeze-depth recommendation
        when CONTROL_SUITE.yolo.freeze is None (advisory only — behaviour
        is unchanged by the recommendation).

    Returns the path to the best weights file.
    """
    # Defensive GPU cleanup — free any leaked memory from prior stages
    # (e.g. DONUT experiments that may not have fully released VRAM).
    _gpu_cleanup()

    from ultralytics import YOLO

    if output_dir is None:
        output_dir = WORKSPACE / "models" / "yolo_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 1: Fine-tuning YOLOv8 for text-region detection")
    print("=" * 60)

    if not YOLO_DATA_YAML.exists():
        print(f"  YOLO dataset.yaml not found at {YOLO_DATA_YAML}")
        print("  Run dataset_preparation.py first.")
        return output_dir / "run" / "weights" / "best.pt"

    model = YOLO(YOLO_BASE)
    start = time.time()

    _yolo = CONTROL_SUITE.yolo

    # Log recommended freeze depth when freeze is not explicitly set.
    # This is advisory only — behaviour is unchanged (freeze=None = full training).
    if _yolo.freeze is None and num_train_samples > 0:
        _rec_freeze = _yolo.recommended_freeze(num_train_samples)
        if _rec_freeze is not None:
            print(
                f"  [YOLO] NOTICE: {num_train_samples} training samples detected. "
                f"Recommended freeze={_rec_freeze} (see YOLOControlConfig.recommended_freeze). "
                "Currently using freeze=None (full training). "
                "Set CONTROL_SUITE.yolo.freeze to apply."
            )

    model.train(
        data=str(YOLO_DATA_YAML),
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH,
        project=str(output_dir),
        name="run",
        exist_ok=True,
        # ── Augmentation (receipt-domain tuned) ──────────────────────────
        degrees=_yolo.degrees,      # 5° rotation tolerance for tilted receipts
        translate=_yolo.translate,  # 0.1 spatial shift
        scale=_yolo.scale,          # 0.3 zoom range
        fliplr=_yolo.fliplr,        # 0.0 — text direction matters; no horizontal flip
        flipud=_yolo.flipud,        # 0.0 — receipts are always upright
        mosaic=_yolo.mosaic,        # 0.5 — reduced from default 1.0 for document domain
        close_mosaic=_yolo.close_mosaic,  # 10 — disable mosaic for final N epochs (⚠️ critical for mAP)
        mixup=_yolo.mixup,          # 0.0 — disabled (blending receipts confuses layout)
        copy_paste=_yolo.copy_paste,  # 0.0 — disabled; can enable for rare-class boost
        hsv_h=_yolo.hsv_h,
        hsv_s=_yolo.hsv_s,
        hsv_v=_yolo.hsv_v,
        # ── Optimizer ────────────────────────────────────────────────────
        optimizer=YOLO_OPTIMIZER,   # "AdamW" default; micro patches to "SGD" for speed
        momentum=YOLO_MOMENTUM,     # used when optimizer="SGD"
        lr0=_yolo.lr0,              # 1e-3 peak LR
        lrf=_yolo.lrf,              # 0.01 final LR fraction
        weight_decay=_yolo.weight_decay,  # 0.0005 L2 regularisation (⚠️ was missing)
        cos_lr=_yolo.cos_lr,        # False — linear decay (⚠️ was missing)
        warmup_epochs=_yolo.warmup_epochs,    # 3.0 (⚠️ was missing)
        warmup_momentum=_yolo.warmup_momentum,  # 0.8 (⚠️ was missing)
        warmup_bias_lr=_yolo.warmup_bias_lr,    # 0.1 (⚠️ was missing)
        # ── Finetuning ───────────────────────────────────────────────────
        freeze=_yolo.freeze,        # None — no frozen layers (⚠️ CRITICAL: was missing entirely)
        # ── Loss weights ─────────────────────────────────────────────────
        box=_yolo.box,              # 7.5 bbox regression weight (⚠️ was missing)
        cls=_yolo.cls,              # 0.5 classification weight (⚠️ was missing)
        dfl=_yolo.dfl,              # 1.5 focal loss weight (⚠️ was missing)
        # ── Performance ──────────────────────────────────────────────────
        cache=_yolo.cache,          # False — set "ram" to speed up with sufficient memory
        workers=_yolo.workers,      # 8 DataLoader threads (⚠️ was missing)
        fraction=_yolo.fraction,    # 1.0 use full dataset (⚠️ was missing)
        # ── Regularisation ────────────────────────────────────────────────
        dropout=_yolo.dropout,      # 0.0 (⚠️ was missing)
        # ── Other ─────────────────────────────────────────────────────────
        patience=_yolo.patience,    # 15 early stopping
        seed=SEED,
        amp=YOLO_AMP,
        deterministic=_yolo.deterministic,  # True (⚠️ was missing)
    )

    elapsed = time.time() - start
    best_path = output_dir / "run" / "weights" / "best.pt"
    print(f"\nYOLO training complete in {elapsed:.1f}s")
    print(f"Best weights -> {best_path}")

    # FIX: GPU cleanup after YOLO training — delete local reference first
    del model
    _gpu_cleanup()

    return best_path


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2: TrOCR Training
# ════════════════════════════════════════════════════════════════════════════

# Keys that are structurally absent in TrOCR's BEiT vision encoder — the
# generic VisionEncoderDecoderModel wrapper declares an optional
# encoder.pooler submodule, but BEiT (and ViT) checkpoints never include it.
# These two keys will always appear in missing_keys for trocr-base-printed
# and are completely harmless (unused at train AND inference time).
# Filtering them before printing the LOAD REPORT ensures the table shows
# zero unexpected MISSING rows for a clean trocr-base-printed load.
_EXPECTED_MISSING_TROCR: frozenset[str] = frozenset(
    {
        "encoder.pooler.dense.weight",
        "encoder.pooler.dense.bias",
    }
)


def _print_trocr_load_report(model_id: str, loading_info: dict) -> None:
    """Print a LOAD REPORT table for TrOCR, filtering known-benign missing keys.

    encoder.pooler.dense.{weight,bias} are always absent for BEiT-based TrOCR
    models (the pooler is optional in the generic wrapper and unused by BEiT).
    They are excluded before printing so the table shows 0 MISSING for a clean
    trocr-base-printed load.
    """
    raw_missing = loading_info.get("missing_keys", [])
    unexpected = list(loading_info.get("unexpected_keys", []))

    # Filter out structurally-absent BEiT pooler keys before reporting.
    missing = [k for k in raw_missing if k not in _EXPECTED_MISSING_TROCR]

    col_width = max((len(k) for k in missing + unexpected), default=30) + 2
    header = f"{'Key':<{col_width}}| {'Status':<8}|"
    sep = "-" * col_width + "+---------+"

    print(f"\nVisionEncoderDecoderModel LOAD REPORT from: {model_id}")
    print(header)
    print(sep)
    for key in missing:
        print(f"{key:<{col_width}}| {'MISSING':<8}|")
    if not missing and not unexpected:
        print(f"{'(all weights loaded cleanly)':<{col_width}}| {'OK':<8}|")
    print()


def train_trocr(output_dir: Path | None = None) -> dict:
    """Fine-tune TrOCR on line crops from receipts.

    Returns the training history dict.
    """
    # Defensive GPU cleanup — free any leaked memory from prior stages
    # (DONUT experiments, YOLO training, etc.) before loading the 246M-param
    # TrOCR-base model.
    _gpu_cleanup()

    if output_dir is None:
        output_dir = WORKSPACE / "models" / "trocr_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 2: Fine-tuning TrOCR on line crops")
    print("=" * 60)

    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
    # FIX: low_cpu_mem_usage=False forces weight tensors to be materialized on CPU
    # immediately instead of deferred loading via the meta device.
    # FIX 2: After .to(DEVICE), non-persistent buffers (e.g. embed_positions._float_tensor)
    # may still be on the meta device — see buffer sweep below.
    model, loading_info = VisionEncoderDecoderModel.from_pretrained(
        TROCR_MODEL_ID, low_cpu_mem_usage=False, output_loading_info=True
    )
    _print_trocr_load_report(TROCR_MODEL_ID, loading_info)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.generation_config.max_new_tokens = TROCR_MAX_LEN
    model.generation_config.no_repeat_ngram_size = 0  # disabled — harmful for short OCR text
    model.generation_config.length_penalty = 1.0  # neutral — do not penalise short outputs
    model.generation_config.num_beams = 4

    model = model.to(DEVICE)
    # FIX: Non-persistent buffers (e.g. embed_positions._float_tensor in TrOCR's
    # sinusoidal positional embedding) are skipped by model.to() in newer versions
    # of PyTorch and remain on the meta device, causing:
    #   RuntimeError: Tensor on device meta is not on the expected device cuda:0!
    # _materialize_meta_buffers covers persistent buffers, non-persistent buffers,
    # and any plain tensor attributes (including _float_tensor set directly on modules).
    n_fixed = _materialize_meta_buffers(model, DEVICE)
    if n_fixed:
        print(f"  [TrOCR] Materialised {n_fixed} meta-device buffer(s) onto {DEVICE}")

    # Enable gradient checkpointing conditionally based on available VRAM.
    # On high-VRAM cards (> 24 GB), checkpointing adds ~30-40% backward overhead
    # for zero memory benefit — mirror the DONUT path in run_experiments.py.
    # On lower-VRAM cards (≤ 24 GB, e.g. RTX 4090), it is required to fit the
    # 246M-param TrOCR-base backward pass.
    # use_cache must be False when gradient_checkpointing is True (incompatible).
    _trocr = CONTROL_SUITE.trocr
    _TROCR_GRAD_CKPT_THRESHOLD_GB = _trocr.grad_ckpt_vram_threshold_gb
    _trocr_enable_grad_ckpt = True
    if torch.cuda.is_available():
        try:
            _trocr_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if _trocr_vram_gb > _TROCR_GRAD_CKPT_THRESHOLD_GB:
                _trocr_enable_grad_ckpt = False
                print(
                    f"  [TrOCR] GradCkpt disabled — VRAM={_trocr_vram_gb:.1f} GB"
                    f" > {_TROCR_GRAD_CKPT_THRESHOLD_GB:.0f} GB threshold"
                )
            else:
                print(
                    f"  [TrOCR] GradCkpt enabled — VRAM={_trocr_vram_gb:.1f} GB"
                    f" <= {_TROCR_GRAD_CKPT_THRESHOLD_GB:.0f} GB threshold"
                )
        except Exception as _exc:
            print(f"  [TrOCR] GradCkpt VRAM detection failed ({_exc}) — defaulting to enabled")

    if _trocr_enable_grad_ckpt:
        model.config.use_cache = False
        model.decoder.config.use_cache = False
        model.gradient_checkpointing_enable()
    else:
        model.config.use_cache = True
        model.decoder.config.use_cache = True

    # Detect mixed precision dtype — bf16 preferred on Ampere+, fp16 as fallback.
    # This mirrors exactly how DonutTrainer (train.py lines 413-418) handles precision.
    _use_amp = torch.cuda.is_available()
    _amp_dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(_use_amp and _amp_dtype == torch.float16))
    print(
        f"  [TrOCR] AMP enabled: dtype={_amp_dtype}, gradient_checkpointing={_trocr_enable_grad_ckpt}"
        if _use_amp
        else "  [TrOCR] AMP disabled (CPU mode)"
    )

    # VRAM-aware batch size auto-scaling.
    # Reserve accounts for: model weights (~0.9 GiB) + gradients (~0.9 GiB) +
    # AdamW optimizer states (~1.8 GiB) + system overhead (~0.9 GiB) = ~4.5 GiB.
    # Per-item cost with AMP (bf16 activations): ~0.3 GiB for TrOCR-base.
    # Calibration constants are defined on TrOCRControlConfig and shared with
    # effective_batch_size() — both paths always use the same values.
    trocr_batch = TROCR_BATCH
    grad_accum = GRAD_ACCUM
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        max_safe_batch = CONTROL_SUITE.trocr.effective_batch_size(free_gb)
        if max_safe_batch < trocr_batch:
            old_batch = trocr_batch
            trocr_batch = max(1, max_safe_batch)
            # Adjust gradient accumulation to maintain the same effective batch size.
            effective_batch_size = old_batch * GRAD_ACCUM
            grad_accum = max(1, effective_batch_size // trocr_batch)
            print(
                f"  [TrOCR] VRAM-aware batch scaling: {old_batch} → {trocr_batch} "
                f"(free={free_gb:.1f} GiB / {total_gb:.1f} GiB, grad_accum={grad_accum})"
            )
        else:
            print(f"  [TrOCR] VRAM OK: batch_size={trocr_batch}, free={free_gb:.1f} GiB")

    train_dir = TROCR_DATA_DIR / "train"
    val_dir = TROCR_DATA_DIR / "val"

    if not train_dir.exists() or not (train_dir / "metadata.jsonl").exists():
        print(f"  TrOCR training data not found at {train_dir}")
        print("  Run dataset_preparation.py first.")
        return {"train_loss": [], "val_loss": []}

    train_ds = TrOCRReceiptDataset(
        train_dir,
        processor,
        TROCR_MAX_LEN,
        augmentation=get_augmentation_transforms(CONTROL_SUITE.trocr.augmentation_preset),
    )
    # Use val split (not test!) to match DONUT experiment design; no augmentation for val.
    val_ds = TrOCRReceiptDataset(val_dir, processor, TROCR_MAX_LEN)

    if len(train_ds) == 0:
        raise ValueError("TrOCR training dataset is empty — check data paths.")

    train_loader = DataLoader(
        train_ds, batch_size=trocr_batch, shuffle=True, num_workers=_optimal_num_workers()
    )
    val_loader = (
        DataLoader(
            val_ds, batch_size=trocr_batch, shuffle=False, num_workers=_optimal_num_workers()
        )
        if len(val_ds) > 0
        else None
    )

    total_steps = (len(train_loader) // grad_accum) * TROCR_EPOCHS
    if TROCR_MINI_MODE:
        # SGD + Nesterov + CosineAnnealingLR — faster convergence for short micro runs.
        # LR 200× higher than AdamW default: SGD needs larger LR since it lacks adaptive scaling.
        # CosineAnnealingLR decays from TROCR_LR to eta_min over all steps without warmup,
        # reaching useful weights immediately (unlike the linear warmup that barely finishes
        # in 1-epoch micro runs).
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=TROCR_LR * 200,  # e.g. 5e-5 * 200 = 1e-2
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=TROCR_LR * 2,  # floor ≈ 1/100 of initial SGD LR
        )
        print(f"  [TrOCR] Micro mode: SGD+Nesterov+CosineAnnealingLR, lr={TROCR_LR * 200:.2e}")
    else:
        _trocr = CONTROL_SUITE.trocr
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TROCR_LR,
            weight_decay=_trocr.weight_decay,  # 1e-4 (⚠️ was missing — AdamW default is 0)
            betas=(_trocr.adam_beta1, _trocr.adam_beta2),
            eps=_trocr.adam_epsilon,
        )
        warmup_steps = int(total_steps * _trocr.warmup_ratio)
        scheduler = get_scheduler(
            _trocr.lr_scheduler,  # "linear" (was hardcoded, now from control_suite)
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "num_train_samples": 0}
    history["num_train_samples"] = len(train_ds)
    start = time.time()

    try:
        for epoch in range(TROCR_EPOCHS):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"TrOCR Epoch {epoch + 1}/{TROCR_EPOCHS}")
            for step, batch in enumerate(pbar):
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype, enabled=_use_amp):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum
                scaler.scale(loss).backward()
                epoch_loss += outputs.loss.item()  # use unscaled loss for logging

                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix(loss=f"{epoch_loss / (step + 1):.4f}")

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
            print(f"Epoch {epoch + 1}: train={avg_train:.4f}  val={avg_val:.4f}")

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
    finally:
        # Always free GPU memory even if training raised an exception.
        # Without this, a mid-training crash leaves TrOCR (246M params) on the
        # GPU and causes CUDA OOM when the next stage (DONUT) loads its model.
        del model, optimizer, scheduler
        if scaler is not None:
            del scaler
        del train_ds, val_ds, train_loader
        if val_loader is not None:
            del val_loader
        _gpu_cleanup()

    return history


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3: TrOCR+YOLO Inference Pipeline
# ════════════════════════════════════════════════════════════════════════════
def _assign_fields_heuristic(ocr_lines: list[dict]) -> dict[str, str]:
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
    # Total pattern: currency symbols or "total" keyword followed by numbers
    # Generic money pattern

    used = set()

    # Find date — scan all lines (date can appear anywhere on receipt)
    for i, line in enumerate(sorted_lines):
        text = line.get("text", "")
        m = _DATE_RE.search(text)
        if m:
            result["date"] = m.group(0).strip()
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

    # Address: prefer lines with road/postcode keywords; cap at 3 keyword lines
    # or 2 fallback lines (mirrors real receipt address format of 1-3 lines).
    addr_keyword_parts = []
    addr_other_parts = []
    for i, line in enumerate(sorted_lines):
        if i not in used:
            text = line.get("text", "").strip()
            if not text or _MONEY_RE.match(text):
                continue
            if _ADDRESS_RE.search(text):
                if len(addr_keyword_parts) < 3:
                    addr_keyword_parts.append(text)
            else:
                if len(addr_other_parts) < 2:
                    addr_other_parts.append(text)
    addr_parts = addr_keyword_parts if addr_keyword_parts else addr_other_parts
    result["address"] = " ".join(addr_parts)

    return result


def run_trocr_yolo_inference(
    image_path: Path,
    yolo_model,
    trocr_model,
    trocr_processor: TrOCRProcessor,
) -> dict[str, str]:
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
            pixel_values = trocr_processor(crop, return_tensors="pt").pixel_values.to(DEVICE)

            with torch.no_grad():
                generated_ids = trocr_model.generate(
                    pixel_values,
                    num_beams=4,
                    length_penalty=1.0,
                    no_repeat_ngram_size=0,
                )
            text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            if text:
                ocr_lines.append(
                    {
                        "text": text,
                        "x": x1,
                        "y": y1,
                        "x2": x2,
                        "y2": y2,
                        "conf": float(box.conf[0]) if hasattr(box, "conf") else 1.0,
                    }
                )

    # Stage 3: Heuristic field assignment
    return _assign_fields_heuristic(ocr_lines)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["yolo", "trocr", "both"], default="both")
    args = parser.parse_args()

    if args.stage in ("yolo", "both"):
        train_yolo()

    if args.stage in ("trocr", "both"):
        train_trocr()
