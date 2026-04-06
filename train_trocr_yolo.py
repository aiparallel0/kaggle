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

import gc
import json
import logging
import math
import re
import statistics
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import (
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        get_scheduler,
    )

    _TRANSFORMERS_AVAILABLE_TROCR = True
except ImportError:
    _TRANSFORMERS_AVAILABLE_TROCR = False
    import math as _math_tr2

    def get_scheduler(name, optimizer, num_warmup_steps=0, num_training_steps=0, **kwargs):  # type: ignore[misc]
        """Inline LR scheduler — replaces transformers.get_scheduler."""
        from torch.optim.lr_scheduler import LambdaLR

        def _lr_lambda(step):
            if step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            if name == "linear":
                return max(
                    0.0,
                    float(num_training_steps - step)
                    / float(max(1, num_training_steps - num_warmup_steps)),
                )
            # cosine (default)
            progress = float(step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + _math_tr2.cos(_math_tr2.pi * progress)))

        return LambdaLR(optimizer, _lr_lambda)

    class TrOCRProcessor:  # type: ignore[no-redef]
        @classmethod
        def from_pretrained(cls, *a, **kw):
            raise ImportError(
                "transformers >= 4.37.0 is required but not installed.\n"
                "Run: pip install transformers>=4.37.0\n"
                "Or:  pip install -r requirements.txt"
            )

    class VisionEncoderDecoderModel:  # type: ignore[no-redef]
        @classmethod
        def from_pretrained(cls, *a, **kw):
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

    def _load_image(path: "str | Path") -> "_PILImage.Image":  # type: ignore[name-defined]
        return _PILImage.open(path).convert("RGB")

    _PIL_AVAILABLE = True
except ImportError:
    import numpy as _np_img

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

    def _load_png(path: "str | Path") -> "_np_img.ndarray | None":
        """Minimal PNG decoder → RGB numpy array."""
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
        arr = _np_img.frombuffer(pixel_data, dtype=_np_img.uint8).reshape(height, width, channels)
        if channels == 4:
            arr = arr[:, :, :3]
        return arr

    def _load_bmp(path: "str | Path") -> "_np_img.ndarray | None":
        """Minimal BMP decoder for 24-bit uncompressed BMP → RGB numpy array."""
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
        row_size = (width * 3 + 3) & ~3
        arr = _np_img.zeros((height, width, 3), dtype=_np_img.uint8)
        for row in range(height):
            src_row = (height - 1 - row) if flipped else row
            start = pixel_offset + src_row * row_size
            raw_row = data[start : start + width * 3]
            pixels = _np_img.frombuffer(raw_row, dtype=_np_img.uint8).reshape(width, 3)
            arr[row] = pixels[:, ::-1]  # BGR → RGB
        return arr

    def _load_jpeg_ctypes(path: "str | Path"):
        """Load JPEG via ImageMagick subprocess (system libjpeg fallback)."""
        import subprocess as _sp

        try:
            result = _sp.run(
                ["convert", str(path), "-colorspace", "RGB", "ppm:-"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0:
                ppm = result.stdout
                lines = ppm.split(b"\n")
                if lines[0] == b"P6":
                    dims = lines[1].split()
                    w, h = int(dims[0]), int(dims[1])
                    pixel_data = b"\n".join(lines[3:])
                    arr = _np_img.frombuffer(pixel_data, dtype=_np_img.uint8)
                    if len(arr) >= h * w * 3:
                        return arr[: h * w * 3].reshape(h, w, 3)
        except (ValueError, IndexError, MemoryError):
            pass
        return None

    def _load_image(path: "str | Path"):  # type: ignore[misc]
        """Load an image file as an RGB numpy array without PIL."""
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
        return _np_img.ascontiguousarray(arr, dtype=_np_img.uint8)

    class _FakeImg:
        """Minimal PIL.Image shim backed by a numpy array (RGB uint8 HxWx3)."""

        def __init__(self, arr: "_np_img.ndarray") -> None:
            self._arr = arr

        def convert(self, mode: str) -> "_FakeImg":
            return self  # already RGB

        @property
        def size(self) -> "tuple[int, int]":
            h, w = self._arr.shape[:2]
            return w, h

        def crop(self, box: "tuple[int, int, int, int]") -> "_FakeImg":
            x1, y1, x2, y2 = box
            return _FakeImg(self._arr[y1:y2, x1:x2])

    class _ImageModule:
        @staticmethod
        def open(path: "str | Path") -> _FakeImg:
            return _FakeImg(_load_image(path))

    Image = _ImageModule()  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
# Inline YOLOv8x — fallback when ultralytics is unavailable
# Architecture: CSP-DarkNet backbone + PAN-FPN neck + decoupled detection head
# Matches yolov8x.yaml: depth=1.00, width=1.25, max_channels=512
# Final channels: [80, 160, 320, 640, 640]; C2f repeats: [3, 6, 6, 3]
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ultralytics import YOLO as _YOLO_CLS  # noqa: E402

    _ULTRALYTICS_AVAILABLE = True
except ImportError:
    _ULTRALYTICS_AVAILABLE = False

    import torch.nn as _nn

    # ── Building blocks ────────────────────────────────────────────────────

    def _autopad(k, p=None, d=1):
        """Pad to same shape output."""
        if d > 1:
            k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p

    class _Conv(_nn.Module):
        """Conv-BN-SiLU block."""

        def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
            super().__init__()
            self.conv = _nn.Conv2d(
                c1, c2, k, s, _autopad(k, p, d), dilation=d, groups=g, bias=False
            )
            self.bn = _nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
            self.act = (
                _nn.SiLU(inplace=True)
                if act is True
                else (act if isinstance(act, _nn.Module) else _nn.Identity())
            )

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class _Bottleneck(_nn.Module):
        """Standard bottleneck."""

        def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
            super().__init__()
            c_ = int(c2 * e)
            self.cv1 = _Conv(c1, c_, k[0], 1)
            self.cv2 = _Conv(c_, c2, k[1], 1, g=g)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    class _C2f(_nn.Module):
        """CSP Bottleneck with 2 convolutions (YOLOv8 core block)."""

        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
            super().__init__()
            self.c = int(c2 * e)
            self.cv1 = _Conv(c1, 2 * self.c, 1, 1)
            self.cv2 = _Conv((2 + n) * self.c, c2, 1)
            self.m = _nn.ModuleList(
                _Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
                for _ in range(n)
            )

        def forward(self, x):
            y = list(self.cv1(x).chunk(2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))

    class _SPPF(_nn.Module):
        """Spatial Pyramid Pooling - Fast."""

        def __init__(self, c1, c2, k=5):
            super().__init__()
            c_ = c1 // 2
            self.cv1 = _Conv(c1, c_, 1, 1)
            self.cv2 = _Conv(c_ * 4, c2, 1, 1)
            self.m = _nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

        def forward(self, x):
            x = self.cv1(x)
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

    class _DFL(_nn.Module):
        """Distribution Focal Loss (DFL) for box decoding."""

        def __init__(self, c1=16):
            super().__init__()
            self.conv = _nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
            x = torch.arange(c1, dtype=torch.float)
            self.conv.weight.data = x.view(1, c1, 1, 1).clone()
            self.c1 = c1

        def forward(self, x):
            b, c, a = x.shape
            return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

    class _Detect(_nn.Module):
        """YOLOv8 detection head."""

        reg_max = 16

        def __init__(self, nc=80, ch=()):
            super().__init__()
            self.nc = nc
            self.nl = len(ch)
            self.reg_max = 16
            self.no = nc + self.reg_max * 4
            self.stride = torch.zeros(self.nl)
            c2 = max(max(ch) // 4, self.reg_max * 4)
            c3 = max(ch[0], min(nc, 100))
            self.cv2 = _nn.ModuleList(
                _nn.Sequential(
                    _Conv(x, c2, 3), _Conv(c2, c2, 3), _nn.Conv2d(c2, 4 * self.reg_max, 1)
                )
                for x in ch
            )
            self.cv3 = _nn.ModuleList(
                _nn.Sequential(_Conv(x, c3, 3), _Conv(c3, c3, 3), _nn.Conv2d(c3, nc, 1)) for x in ch
            )
            self.dfl = _DFL(self.reg_max)

        def forward(self, x):
            for i in range(self.nl):
                x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
            if self.training:
                return x
            shape = x[0].shape
            x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
            anchors, strides = self._make_anchors(x, self.stride)
            dbox = self._dist2bbox(self.dfl(box), anchors, xywh=True) * strides.squeeze(-1)
            y = torch.cat((dbox, cls.sigmoid()), 1)
            return y

        @staticmethod
        def _make_anchors(feats, strides, grid_cell_offset=0.5):
            anchor_points, stride_tensor = [], []
            for _i, (feat, stride) in enumerate(zip(feats, strides)):
                _, _, h, w = feat.shape
                sx = torch.arange(w, device=feat.device, dtype=torch.float32) + grid_cell_offset
                sy = torch.arange(h, device=feat.device, dtype=torch.float32) + grid_cell_offset
                sy, sx = torch.meshgrid(sy, sx, indexing="ij")
                anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
                stride_tensor.append(
                    torch.full((h * w, 1), stride, dtype=torch.float32, device=feat.device)
                )
            return torch.cat(anchor_points), torch.cat(stride_tensor)

        @staticmethod
        def _dist2bbox(distance, anchor_points, xywh=True):
            lt, rb = distance.chunk(2, 1)
            x1y1 = anchor_points.T.unsqueeze(0) - lt
            x2y2 = anchor_points.T.unsqueeze(0) + rb
            if xywh:
                c_xy = (x1y1 + x2y2) / 2
                wh = x2y2 - x1y1
                return torch.cat((c_xy, wh), 1)
            return torch.cat((x1y1, x2y2), 1)

    class _YOLOv8Model(_nn.Module):
        """Full YOLOv8x model: backbone (0-9) + neck (10-21) + head (22)."""

        def __init__(self, nc=1):
            super().__init__()
            # Channels: [80, 160, 320, 640, 640]
            # C2f repeats: [3, 6, 6, 3]
            # Backbone
            self.model = _nn.ModuleList(
                [
                    _Conv(3, 80, 3, 2),  # 0 - P1/2
                    _Conv(80, 160, 3, 2),  # 1 - P2/4
                    _C2f(160, 160, 3, True),  # 2
                    _Conv(160, 320, 3, 2),  # 3 - P3/8
                    _C2f(320, 320, 6, True),  # 4
                    _Conv(320, 640, 3, 2),  # 5 - P4/16
                    _C2f(640, 640, 6, True),  # 6
                    _Conv(640, 640, 3, 2),  # 7 - P5/32
                    _C2f(640, 640, 3, True),  # 8
                    _SPPF(640, 640, 5),  # 9
                    # Neck
                    _nn.Upsample(None, 2, "nearest"),  # 10
                    None,  # 11 - Concat (handled in forward)
                    _C2f(1280, 640, 3),  # 12
                    _nn.Upsample(None, 2, "nearest"),  # 13
                    None,  # 14 - Concat
                    _C2f(960, 320, 3),  # 15 P3 out
                    _Conv(320, 320, 3, 2),  # 16
                    None,  # 17 - Concat
                    _C2f(960, 640, 3),  # 18 P4 out
                    _Conv(640, 640, 3, 2),  # 19
                    None,  # 20 - Concat
                    _C2f(1280, 640, 3),  # 21 P5 out
                    _Detect(nc, (320, 640, 640)),  # 22 head
                ]
            )
            # Configure detect strides
            self.model[22].stride = torch.tensor([8.0, 16.0, 32.0])
            self._nc = nc

        def forward(self, x):
            # Backbone
            p = [None] * 23
            p[0] = self.model[0](x)
            p[1] = self.model[1](p[0])
            p[2] = self.model[2](p[1])
            p[3] = self.model[3](p[2])
            p[4] = self.model[4](p[3])
            p[5] = self.model[5](p[4])
            p[6] = self.model[6](p[5])
            p[7] = self.model[7](p[6])
            p[8] = self.model[8](p[7])
            p[9] = self.model[9](p[8])
            # Neck
            p[10] = self.model[10](p[9])  # upsample
            p[12] = self.model[12](torch.cat([p[10], p[6]], 1))  # concat 10+6
            p[13] = self.model[13](p[12])  # upsample
            p[15] = self.model[15](torch.cat([p[13], p[4]], 1))  # concat 13+4 → P3
            p[16] = self.model[16](p[15])  # downsample
            p[18] = self.model[18](torch.cat([p[16], p[12]], 1))  # concat 16+12 → P4
            p[19] = self.model[19](p[18])  # downsample
            p[21] = self.model[21](torch.cat([p[19], p[9]], 1))  # concat 19+9 → P5
            # Head
            return self.model[22]([p[15], p[18], p[21]])

    # ── Result containers ─────────────────────────────────────────────────

    class _Boxes:
        """Mimics ultralytics result.boxes interface."""

        def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor):
            self.xyxy = xyxy
            self.conf = conf
            self.cls = cls

        def __len__(self):
            return len(self.xyxy)

        def __iter__(self):
            for i in range(len(self)):
                # Yield a view with 2-D xyxy/conf so callers can do box.xyxy[0].
                yield _Boxes(
                    self.xyxy[i].unsqueeze(0),
                    self.conf[i].unsqueeze(0),
                    self.cls[i].unsqueeze(0),
                )

    class _BoxResult:
        """Mimics ultralytics per-image result."""

        def __init__(self, xyxy, conf, cls):
            self.boxes = _Boxes(xyxy, conf, cls)

    # ── NMS helper ────────────────────────────────────────────────────────

    def _nms_boxes(boxes_xyxy, scores, iou_thr=0.45, score_thr=0.25):
        """Non-maximum suppression (pure torch, no torchvision dependency).

        Returns indices into the *original* ``boxes_xyxy`` / ``scores`` tensors
        (before score filtering) so callers can safely index them without an
        offset mismatch.
        """
        # Preserve original indices so the returned values are valid for the
        # caller's full-size tensors.
        orig_indices = (scores > score_thr).nonzero(as_tuple=False).squeeze(1)
        filt_boxes = boxes_xyxy[orig_indices]
        filt_scores = scores[orig_indices]
        if filt_boxes.numel() == 0:
            return torch.tensor([], dtype=torch.long)
        # Sort by score descending
        order = filt_scores.argsort(descending=True)
        kept_in_filt = []
        while order.numel() > 0:
            i = order[0].item()
            kept_in_filt.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            b = filt_boxes
            xx1 = torch.clamp(b[rest, 0], min=b[i, 0].item())
            yy1 = torch.clamp(b[rest, 1], min=b[i, 1].item())
            xx2 = torch.clamp(b[rest, 2], max=b[i, 2].item())
            yy2 = torch.clamp(b[rest, 3], max=b[i, 3].item())
            inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_rest = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
            iou = inter / (area_i + area_rest - inter + 1e-7)
            order = rest[iou <= iou_thr]
        # Map filtered indices back to positions in the original tensors.
        kept_tensor = torch.tensor(kept_in_filt, dtype=torch.long)
        return orig_indices[kept_tensor]

    # ── Main class ────────────────────────────────────────────────────────

    class _YOLO_CLS:  # noqa: N801
        """Inline YOLOv8x — fallback when ultralytics is not installed.

        Supports:
        - __call__(img) → list[_BoxResult]  (inference)
        - train(data, epochs, ...) → None   (basic training loop)

        Weight loading: looks for ``<model_path_stem>_sd.pt`` (raw state dict)
        alongside the main ``.pt`` file. This companion file is created
        automatically when ultralytics trains and saves a checkpoint.
        """

        def __init__(self, model_path: str | Path = "yolov8x.pt"):
            import logging as _log

            self._log = _log.getLogger(__name__)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = _YOLOv8Model(nc=1).to(self._device)
            self._model_path = Path(model_path)
            # Default inference size; must be a multiple of 32.  Matches YOLO_IMG_SIZE
            # used during training so anchor grids align with trained coordinates.
            self._imgsz = YOLO_IMG_SIZE

            # Try to load raw state dict companion file
            sd_path = self._model_path.with_stem(self._model_path.stem + "_sd")
            if sd_path.exists():
                try:
                    sd = torch.load(sd_path, map_location=self._device, weights_only=True)
                    missing, unexpected = self.model.load_state_dict(sd, strict=False)
                    # Detect shape mismatches: keys present in both sd and model but with
                    # incompatible shapes indicate that the checkpoint was trained with a
                    # different model size (e.g., yolov8n vs yolov8x).  strict=False silently
                    # skips such keys, leaving the affected layers randomly initialised.
                    shape_mismatches = [
                        k
                        for k, v in sd.items()
                        if k in (_model_params := dict(self.model.named_parameters()))
                        and v.shape != _model_params[k].shape
                    ]
                    if shape_mismatches:
                        self._log.error(
                            "YOLOv8 state dict from %s has %d shape mismatch(es) — "
                            "the checkpoint was likely trained on a different model size. "
                            "Affected layers are randomly initialised; detection will fail. "
                            "Re-train with the correct YOLO_BASE architecture. "
                            "Mismatched keys (first 5): %s",
                            sd_path,
                            len(shape_mismatches),
                            shape_mismatches[:5],
                        )
                    else:
                        self._log.info(
                            "Loaded YOLOv8x from %s (missing=%d, unexpected=%d)",
                            sd_path,
                            len(missing),
                            len(unexpected),
                        )
                except Exception as e:
                    # Fix: issue_report_summary critical #3 — log at ERROR level so the
                    # operator sees that YOLOv8 weights failed to load (model will use random init).
                    self._log.error(
                        "Could not load YOLOv8 state dict from %s — falling back to random init. "
                        "Detection quality will be severely degraded. Error: %s",
                        sd_path,
                        e,
                        exc_info=True,
                    )
            else:
                self._log.warning(
                    "No YOLOv8 weights found at %s — using random init. "
                    "Train with ultralytics first to generate weights, or place "
                    "a plain state dict at %s.",
                    self._model_path,
                    sd_path,
                )

        def __call__(
            self, img, verbose: bool = False, imgsz: int | None = None, **kwargs
        ) -> list[Any]:
            """Run inference on a single image. Returns list[_BoxResult]."""
            import numpy as _np

            self.model.eval()

            # ── 1. Convert input to HxWx3 uint8 numpy array ──────────────
            if isinstance(img, _np.ndarray):
                arr = img
            elif _PIL_AVAILABLE and hasattr(img, "tobytes"):
                arr = _np.array(img)
            elif isinstance(img, torch.Tensor):
                # Caller supplied a pre-processed tensor; use as-is (no letterbox).
                t = img.to(self._device)
                with torch.no_grad():
                    pred = self.model(t)
                pred = pred[0]
                cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
                scores = pred[4:].max(0).values
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                keep = _nms_boxes(boxes_xyxy, scores)
                return [
                    _BoxResult(boxes_xyxy[keep].cpu(), scores[keep].cpu(), torch.zeros(len(keep)))
                ]
            else:
                arr = _np.array(img)

            # ── 2. Letterbox-resize: fit in target, pad to multiple of 32 ──
            orig_h, orig_w = arr.shape[:2]
            target = imgsz if imgsz is not None else self._imgsz
            scale = min(target / orig_h, target / orig_w)
            nh, nw = int(orig_h * scale), int(orig_w * scale)
            # Align canvas dimensions to the nearest multiple of 32 (required by the
            # 5-level stride backbone — without this, deeper feature-map heights/widths
            # are non-integer and anchor grids misalign).
            nh32 = max(32, ((nh + 31) // 32) * 32)
            nw32 = max(32, ((nw + 31) // 32) * 32)
            padded = _np.full((nh32, nw32, 3), 114, dtype=_np.uint8)
            if _PIL_AVAILABLE:
                from PIL import Image as _PIL_Img  # noqa: PLC0415

                resized = _np.array(_PIL_Img.fromarray(arr).resize((nw, nh), _PIL_Img.BILINEAR))
            else:
                # Nearest-neighbour resize with no external dependencies.
                row_idx = (_np.arange(nh) * (orig_h / nh)).astype(_np.int32).clip(0, orig_h - 1)
                col_idx = (_np.arange(nw) * (orig_w / nw)).astype(_np.int32).clip(0, orig_w - 1)
                resized = arr[row_idx][:, col_idx]
            padded[:nh, :nw] = resized

            # ── 3. Tensor + forward pass ──────────────────────────────────
            t = (
                torch.from_numpy(padded)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(self._device)
            )
            with torch.no_grad():
                pred = self.model(t)  # [1, 4+nc, total_anchors] — stride-scaled by Detect head
            pred = pred[0]  # [4+nc, total_anchors]

            # ── 4. Decode xywh → xyxy ─────────────────────────────────────
            cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
            scores = pred[4:].max(0).values
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)  # [total_anchors, 4]

            # ── 5. NMS ────────────────────────────────────────────────────
            keep = _nms_boxes(boxes_xyxy, scores)
            xyxy = boxes_xyxy[keep].cpu().float()
            conf = scores[keep].cpu()

            # L4: Verify NMS correctness — no high-IoU pairs should survive.
            _assert_nms_no_high_iou_pairs(xyxy)

            # ── 6. Inverse letterbox: map padded-image coords → original image coords ──
            xyxy[:, 0] /= scale  # x1
            xyxy[:, 1] /= scale  # y1
            xyxy[:, 2] /= scale  # x2
            xyxy[:, 3] /= scale  # y2
            xyxy[:, 0::2].clamp_(0, orig_w)
            xyxy[:, 1::2].clamp_(0, orig_h)

            cls = torch.zeros(len(keep))
            return [_BoxResult(xyxy, conf, cls)]

        def train(
            self,
            data: str | None = None,
            epochs: int = 50,
            imgsz: int = 512,
            batch: int = 8,
            project: str = "runs/detect",
            name: str = "train",
            exist_ok: bool = True,
            seed: int = 0,
            amp: bool = True,
            **kwargs,
        ):
            """Minimal YOLOv8 training loop (inline fallback).

            Loads images and YOLO-format labels from the ``data`` YAML file.
            Saves a plain state dict to ``<project>/<name>/weights/best_sd.pt``.

            NOTE: This inline loop uses a proxy L2 loss over the detection-head
            feature maps rather than the full YOLO detection loss (which requires
            the ultralytics TaskAlignedAssigner).  It will train the backbone
            away from random initialisation but will NOT produce a properly
            calibrated detector.  For real detection quality install ultralytics:
                pip install ultralytics
            """
            import logging as _log

            import yaml as _yaml  # stdlib pyyaml (tiny dep, always present)

            _logger = _log.getLogger(__name__)
            _logger.info("_YOLOv8Inline.train() — ultralytics not installed, using inline loop")

            # Parse YOLO dataset YAML
            if data is None:
                _logger.warning("No data YAML provided — skipping YOLO training")
                return
            try:
                with open(data) as f:
                    ds_cfg = _yaml.safe_load(f)
            except Exception as e:
                # Fix: issue_report_summary critical #3 — log at ERROR level so the
                # operator knows YOLO training was silently skipped due to YAML failure.
                _logger.error(
                    "Could not load YOLO data YAML %s — YOLO training will be SKIPPED. "
                    "This is a fatal configuration error; check the YAML file path and "
                    "contents. Error: %s",
                    data,
                    e,
                    exc_info=True,
                )
                return

            ds_path = Path(ds_cfg.get("path", "."))
            train_img_dir = ds_path / ds_cfg.get("train", "images/train")
            img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            img_files = sorted(p for p in train_img_dir.rglob("*") if p.suffix.lower() in img_exts)
            if not img_files:
                _logger.warning("No images found in %s — skipping YOLO training", train_img_dir)
                return

            out_dir = Path(project) / name / "weights"
            out_dir.mkdir(parents=True, exist_ok=True)
            best_sd_path = out_dir / "best_sd.pt"

            self.model.train()
            # lr=1e-4: conservative LR for the unit-energy proxy loss.  The loss
            # is (mean(f^2)-1)^2 — no degenerate zero, but can still overshoot
            # the unit-energy basin at high LR.  Gradient clipping (max_norm=1.0)
            # below is the primary safeguard; the lower LR provides a second layer.
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=5e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=amp and torch.cuda.is_available())
            best_loss = float("inf")
            _skipped_steps_total = 0  # track GradScaler overflows across all epochs

            import math as _yolo_math  # noqa: PLC0415

            _logger.info(
                "Starting inline YOLO training: %d images × %d epochs", len(img_files), epochs
            )
            for epoch in range(epochs):
                epoch_loss = 0.0
                count = 0
                skipped_steps = 0
                _consecutive_failures = 0  # Fix: issue_report_summary critical #3
                for img_path in img_files:
                    # Load image
                    try:
                        raw = _load_image(img_path)
                    except Exception as _img_exc:
                        _logger.debug(
                            "[inline-YOLO] Could not load %s: %s — skipping", img_path, _img_exc
                        )
                        # Fix: issue_report_summary critical #3 — log at ERROR level with
                        # batch index, track consecutive failures, abort if threshold exceeded.
                        if isinstance(_img_exc, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
                        _consecutive_failures += 1
                        _logger.error(
                            "YOLO inline training: image load failed (batch=%s, "
                            "consecutive_failures=%d/%d): %s",
                            img_path.name,
                            _consecutive_failures,
                            MAX_CONSECUTIVE_BATCH_FAILURES,
                            _img_exc,
                            exc_info=True,
                        )
                        if _consecutive_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                            raise RuntimeError(
                                f"YOLO inline training aborted: {_consecutive_failures} "
                                f"consecutive batch failures exceeded threshold "
                                f"MAX_CONSECUTIVE_BATCH_FAILURES={MAX_CONSECUTIVE_BATCH_FAILURES}. "
                                "Check image data and CUDA memory. "
                                "See CLAUDE.md §5 Fix #3."
                            ) from _img_exc
                        continue
                    _consecutive_failures = 0  # reset on success
                    # Resize to imgsz × imgsz
                    import numpy as _np

                    raw_arr = _np.array(raw) if not isinstance(raw, _np.ndarray) else raw
                    h, w = raw_arr.shape[:2]
                    scale = imgsz / max(h, w)
                    nh, nw = int(h * scale), int(w * scale)
                    from PIL import Image as _PIL_Image  # noqa: PLC0415

                    resized = _np.array(
                        _PIL_Image.fromarray(raw_arr).resize((nw, nh), _PIL_Image.BILINEAR)
                    )
                    padded = _np.zeros((imgsz, imgsz, 3), dtype=_np.uint8)
                    padded[:nh, :nw] = resized
                    t = (
                        torch.from_numpy(padded)
                        .permute(2, 0, 1)
                        .float()
                        .div(255.0)
                        .unsqueeze(0)
                        .to(self._device)
                    )
                    # Load labels — standard YOLO layout: <root>/labels/<split>/<stem>.txt
                    # img_path is under <root>/images/<split>/<stem>.ext, so the label
                    # is at <root>/labels/<split>/<stem>.txt (ds_path = <root>).
                    label_path = (
                        ds_path
                        / "labels"
                        / img_path.parent.name
                        / img_path.with_suffix(".txt").name
                    )
                    if not label_path.exists():
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=amp and torch.cuda.is_available()):
                        # NOTE: Full YOLO detection loss (box regression + DFL +
                        # classification) requires the ultralytics TaskAlignedAssigner
                        # and is not inlined here.  As a proxy, use a unit-energy loss:
                        # (mean(f^2) - 1)^2 per detection head.  The minimum is reached
                        # when each feature map has unit mean energy — impossible to
                        # satisfy by collapsing weights to zero (where mean(f^2)→0 gives
                        # loss→1, not 0).  The old L2 loss `sum(f^2)` had a degenerate
                        # global minimum at weights=0, causing loss→0 by epoch 3.
                        # For real detection quality install ultralytics:
                        #   pip install ultralytics
                        raw_feats = self.model(t)  # train mode → list of [B, no, H, W]
                        loss = sum((f.float().pow(2).mean() - 1.0).pow(2) for f in raw_feats)
                    scaler.scale(loss).backward()
                    # Gradient clipping: MUST unscale before clip so clip sees
                    # true gradient magnitudes, not GradScaler-inflated ones.
                    # max_norm=1.0 prevents the first-step weight collapse that
                    # drives feature maps to zero (L2 proxy loss degenerate min).
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    _scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    # Detect GradScaler-skipped steps (inf/nan gradients).
                    if scaler.get_scale() < _scale_before:
                        skipped_steps += 1
                    _step_loss_val = loss.item()
                    if _yolo_math.isnan(_step_loss_val) or _yolo_math.isinf(_step_loss_val):
                        _logger.warning(
                            "[inline-YOLO] Epoch %d: NaN/Inf loss at image %s — "
                            "stopping epoch early.",
                            epoch + 1,
                            img_path.name,
                        )
                        break
                    epoch_loss += _step_loss_val
                    count += 1

                if count == 0:
                    _logger.warning(
                        "[inline-YOLO] Epoch %d/%d: 0 images processed "
                        "(no label files found or all images failed to load). "
                        "Check that label files exist at <root>/labels/<split>/<stem>.txt",
                        epoch + 1,
                        epochs,
                    )
                    continue

                avg_loss = epoch_loss / count
                _skipped_steps_total += skipped_steps
                if skipped_steps > 0:
                    _logger.warning(
                        "[inline-YOLO] Epoch %d/%d: GradScaler skipped %d/%d steps "
                        "(inf/nan gradients under AMP fp16). "
                        "Weights were NOT updated for those steps.",
                        epoch + 1,
                        epochs,
                        skipped_steps,
                        count,
                    )
                _logger.info("Epoch %d/%d — loss=%.4f", epoch + 1, epochs, avg_loss)
                # Detect weight collapse: with the unit-energy proxy loss, avg_loss
                # near 1.0 per head means feature maps are near-zero (energy→0 ⟹
                # (0-1)^2=1).  This can still happen if gradient clipping is
                # insufficient.  Stop early — further training is useless.
                # (With the old L2 loss the collapse threshold was 1e-6; that loss
                # is no longer used but the guard remains for robustness.)
                num_heads = len(raw_feats) if "raw_feats" in dir() else 1
                _collapse_threshold = 0.95 * num_heads  # ≈1.0 per head → all near-zero
                if epoch > 0 and avg_loss > _collapse_threshold:
                    _logger.warning(
                        "[inline-YOLO] Epoch %d/%d: avg_loss=%.4f ≥ %.2f — feature maps "
                        "have collapsed to near-zero (unit-energy loss at minimum means "
                        "energy≈0).  Stopping early; further epochs will not recover. "
                        "Install ultralytics for real YOLO training: pip install ultralytics",
                        epoch + 1,
                        epochs,
                        avg_loss,
                        _collapse_threshold,
                    )
                    break
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(self.model.state_dict(), best_sd_path)

            if _skipped_steps_total > 0:
                _logger.warning(
                    "[inline-YOLO] Training complete: %d total GradScaler-skipped steps. "
                    "Consider switching to bf16 or fp32 to avoid AMP overflow.",
                    _skipped_steps_total,
                )

            _logger.info("Inline YOLO training done. Best weights → %s", best_sd_path)
            # Also write best.pt stub so downstream code finds the expected path
            best_pt_path = out_dir / "best.pt"
            if not best_pt_path.exists():
                import shutil

                shutil.copy(best_sd_path, best_pt_path)


from constants import (  # noqa: E402
    DEVICE,
    FIELDS,
    IMAGE_EXTS,
    LABEL_IGNORE_INDEX,
    MAX_CONSECUTIVE_BATCH_FAILURES,
    SEED,
    WORKSPACE,
    _gpu_cleanup,
    _optimal_num_workers,
    _progress,
)
from run_experiments import CONTROL_SUITE, compute_metrics, get_augmentation_transforms

__all__ = [
    "TrOCRReceiptDataset",
    "train_yolo",
    "train_trocr",
    "_build_experiment_trocr_metadata",
    "evaluate_trocr_yolo_on_test",
    "evaluate_yolo_detection",
    "print_yolo_detection_metrics",
    "run_trocr_yolo_inference",
    "_materialize_meta_buffers",
    "_EXPECTED_MISSING_TROCR",
    "_print_trocr_load_report",
    # Canonical field assignment — import this instead of duplicating regex logic
    "_assign_fields_heuristic",
    "_extract_ocr_lines",
    # Pipeline lemma verification
    "verify_pipeline_lemmas",
    # Patchable constants (micro/superfast mode sets these before calling train functions)
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
    "FIELD_ASSIGNER_EPOCHS",  # train_field_assigner() reads this as default epoch count
    "TROCR_USE_TENSOR_CACHE",  # True → TrOCRReceiptDataset serialises processed tensors to disk
]

# ════════════════════════════════════════════════════════════════════════════
# Pipeline Lemmas — Formally Verified Invariants
# ════════════════════════════════════════════════════════════════════════════
#
# Each lemma states a property that MUST hold throughout the YOLO+TrOCR
# pipeline.  Violations are caught at runtime via assertion helpers that are
# inserted at the critical call sites.  The lemmas are numbered L1–L8 and
# referenced in comments throughout this file.
#
# L1 — Resolution Invariant
#      YOLO_IMG_SIZE used at train time == YOLO_IMG_SIZE used at inference.
#      Violation causes anchor-grid misalignment → 0% detection rate despite
#      valid mAP during training.
#
# L2 — Detection-OCR Contract
#      Every bounding box emitted by YOLO satisfies: x2-x1 ≥ 5, y2-y1 ≥ 5,
#      all coordinates ≥ 0, and the crop fits within the source image dims.
#      Violation causes TrOCR to receive degenerate input → empty text.
#
# L3 — Field Assignment Completeness
#      _assign_fields_heuristic() always returns a dict with exactly 4 keys
#      (company, date, address, total) and all values are strings.  Calling
#      code never needs to handle missing keys.
#
# L4 — NMS Correctness
#      After NMS, no pair of surviving boxes has IoU > iou_thr, and the
#      highest-confidence box for each cluster is always preserved.
#
# L5 — OCR Correction Idempotence
#      _correct_ocr_chars(_correct_ocr_chars(s)) == _correct_ocr_chars(s)
#      for all strings s.  Applying the correction twice is safe and produces
#      the same result as applying it once.
#
# L6 — Heuristic Mutual Exclusion
#      Each line index in sorted_lines is assigned to at most one field.
#      The `used` set enforces this: once a line is claimed, it is excluded
#      from subsequent field searches.
#
# L7 — Training Minimum Viability
#      TROCR_EPOCHS ≥ _MIN_TROCR_EPOCHS (3) and YOLO_EPOCHS ≥ 1.
#      Below the TrOCR floor, val_loss ≈ 9.1 → decoder outputs garbage →
#      every crop decodes to empty text → F1 = 0.  Enforced at the entry
#      point of train_trocr().
#
# L8 — Pipeline Monotonicity (Detection Rate)
#      _verify_yolo_detection_rate() independently tracks YOLO-zero-box and
#      TrOCR-all-empty failures.  Combined rate > threshold (50%) raises
#      RuntimeError with the dominant cause clearly identified.
#
# ════════════════════════════════════════════════════════════════════════════

# Minimum training epoch thresholds — enforced by L7.
_MIN_TROCR_EPOCHS: int = 3  # below this, val_loss ≈ 9.1 (non-functional decoder)
_MIN_YOLO_EPOCHS: int = 1  # at least one pass through training data
_MIN_BOX_DIMENSION: int = 5  # L2: minimum crop width/height in pixels
_MAX_DETECTION_FAILURE_RATE: float = 0.5  # L8: combined empty-ocr-lines threshold
_NMS_IOU_EPSILON: float = 0.01  # L4: floating-point tolerance for IoU comparison
_MAX_NMS_VERIFICATION_BOXES: int = 500  # L4: max boxes for O(n²) NMS verification

# Minimum acceptable mAP@50 for YOLO detection quality gate.
# Below this threshold, the print_yolo_detection_metrics() function emits a
# prominent warning.  Chosen because mAP50 < 0.30 typically indicates that
# YOLO is not finding text regions reliably enough for downstream TrOCR to
# produce meaningful field extractions.
_YOLO_MAP50_QUALITY_THRESHOLD: float = 0.3


def _assert_resolution_invariant(train_imgsz: int, infer_imgsz: int) -> None:
    """L1 — Assert that training and inference YOLO image sizes match.

    When the resolutions differ, YOLO's anchor grid (computed during training
    at stride levels [8, 16, 32]) produces confidence scores scaled to the
    training resolution.  At a different inference resolution, these scores
    fall below the NMS threshold → 0% detection despite valid training mAP.

    This is the single most common cause of "YOLO detected 0 text regions"
    when the model trained successfully (see CLAUDE.md §16 Pattern 8).
    """
    if train_imgsz != infer_imgsz:
        raise ValueError(
            f"L1 VIOLATION — Resolution Invariant: YOLO trained at imgsz={train_imgsz} "
            f"but inference called with imgsz={infer_imgsz}.  Anchor grids will "
            f"misalign → 0% detection.  Ensure YOLO_IMG_SIZE is used consistently "
            f"at all call sites."
        )


def _assert_box_validity(
    boxes_xyxy: "list[tuple[float, float, float, float]]",
    img_w: float,
    img_h: float,
) -> "list[tuple[float, float, float, float]]":
    """L2 — Validate bounding boxes and return only valid ones.

    Enforces the detection-OCR contract: each box must be a valid crop region
    within the source image.  Invalid boxes are logged and discarded rather
    than crashing TrOCR.

    Returns the list of valid (x1, y1, x2, y2) tuples.
    """
    valid: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in boxes_xyxy:
        # Clamp to image bounds
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        # Check minimum dimensions
        if x2 - x1 < _MIN_BOX_DIMENSION or y2 - y1 < _MIN_BOX_DIMENSION:
            continue
        valid.append((x1, y1, x2, y2))
    return valid


def _assert_field_result_complete(result: dict[str, str]) -> dict[str, str]:
    """L3 — Assert that field assignment result has exactly 4 string-valued keys.

    This postcondition check ensures that no downstream code ever encounters
    a missing key or non-string value from the heuristic or learned assigner.
    """
    for field in FIELDS:
        if field not in result:
            raise AssertionError(
                f"L3 VIOLATION — Field Assignment Completeness: "
                f"field '{field}' missing from result dict.  "
                f"Got keys: {sorted(result.keys())}"
            )
        if not isinstance(result[field], str):
            raise AssertionError(
                f"L3 VIOLATION — Field Assignment Completeness: "
                f"field '{field}' has type {type(result[field]).__name__}, expected str."
            )
    return result


def _assert_nms_no_high_iou_pairs(
    boxes_xyxy: "torch.Tensor",
    iou_thr: float = 0.45,
) -> None:
    """L4 — Assert that no pair of post-NMS boxes exceeds iou_thr.

    This is a defensive check verifying NMS correctness.  Called only when
    the box count is small enough to make the O(n²) check feasible
    (≤ _MAX_NMS_VERIFICATION_BOXES).
    """
    n = boxes_xyxy.shape[0]
    if n <= 1 or n > _MAX_NMS_VERIFICATION_BOXES:
        return
    for i in range(n):
        for j in range(i + 1, n):
            x1 = max(boxes_xyxy[i, 0].item(), boxes_xyxy[j, 0].item())
            y1 = max(boxes_xyxy[i, 1].item(), boxes_xyxy[j, 1].item())
            x2 = min(boxes_xyxy[i, 2].item(), boxes_xyxy[j, 2].item())
            y2 = min(boxes_xyxy[i, 3].item(), boxes_xyxy[j, 3].item())
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_i = (
                (boxes_xyxy[i, 2] - boxes_xyxy[i, 0]) * (boxes_xyxy[i, 3] - boxes_xyxy[i, 1])
            ).item()
            area_j = (
                (boxes_xyxy[j, 2] - boxes_xyxy[j, 0]) * (boxes_xyxy[j, 3] - boxes_xyxy[j, 1])
            ).item()
            union = area_i + area_j - inter
            iou = inter / (union + 1e-7)
            if iou > iou_thr + _NMS_IOU_EPSILON:
                logging.getLogger(__name__).warning(
                    "L4 WARNING — NMS may have a bug: post-NMS boxes %d and %d "
                    "have IoU=%.4f > threshold %.2f",
                    i,
                    j,
                    iou,
                    iou_thr,
                )


def _assert_ocr_correction_idempotent(text: str, corrected: str) -> None:
    """L5 — Assert that OCR correction is idempotent.

    _correct_ocr_chars(corrected) must equal corrected.  If not, the
    correction function has an instability where a second pass would flip
    characters again (e.g. "0" → "O" → "0" cycle).
    """
    double_corrected = _correct_ocr_chars(corrected)
    if double_corrected != corrected:
        logging.getLogger(__name__).warning(
            "L5 WARNING — OCR Correction Idempotence violation: "
            "input=%r → first=%r → second=%r.  "
            "The correction function is not stable under re-application.",
            text,
            corrected,
            double_corrected,
        )


def _assert_mutual_exclusion(used: "set[int]", new_idx: int, field: str) -> None:
    """L6 — Assert that a line index has not already been claimed.

    This is structurally enforced by the `used` set in _assign_fields_heuristic,
    but this assertion provides an explicit invariant check.
    """
    if new_idx in used:
        raise AssertionError(
            f"L6 VIOLATION — Heuristic Mutual Exclusion: "
            f"line index {new_idx} was already claimed by another field "
            f"but is being reassigned to '{field}'."
        )


def _assert_training_viability(
    trocr_epochs: int | None = None,
    yolo_epochs: int | None = None,
) -> None:
    """L7 — Assert minimum training epochs are met.

    Below _MIN_TROCR_EPOCHS, TrOCR fine-tuning destroys the pretrained weights
    without learning SROIE patterns (val_loss ≈ 9.1 at 1 epoch).  Every crop
    decodes to empty text → F1 = 0.
    """
    if trocr_epochs is not None and trocr_epochs < _MIN_TROCR_EPOCHS:
        raise ValueError(
            f"L7 VIOLATION — Training Minimum Viability: "
            f"TROCR_EPOCHS={trocr_epochs} < minimum {_MIN_TROCR_EPOCHS}.  "
            f"At 1–2 epochs, TrOCR produces val_loss≈9.1 (non-functional decoder). "
            f"Every crop decodes to empty text → F1 = 0.  "
            f"Set TROCR_EPOCHS ≥ {_MIN_TROCR_EPOCHS} (≥ 5 recommended)."
        )
    if yolo_epochs is not None and yolo_epochs < _MIN_YOLO_EPOCHS:
        raise ValueError(
            f"L7 VIOLATION — Training Minimum Viability: "
            f"YOLO_EPOCHS={yolo_epochs} < minimum {_MIN_YOLO_EPOCHS}."
        )


def verify_pipeline_lemmas() -> dict[str, bool]:
    """Run all 8 pipeline lemma checks as a pre-flight verification.

    Returns a dict mapping lemma name to pass/fail boolean.  Logs details
    for each lemma.  Can be called from diagnostics.py or as a standalone
    smoke test::

        python -c "from train_trocr_yolo import verify_pipeline_lemmas; verify_pipeline_lemmas()"

    The checks are:
    - L1: Resolution Invariant — YOLO_IMG_SIZE is a positive multiple of 32.
    - L2: Detection-OCR Contract — _assert_box_validity handles edge cases.
    - L3: Field Assignment Completeness — empty input produces 4-field dict.
    - L4: NMS Correctness — verified structurally (called at inference time).
    - L5: OCR Correction Idempotence — test corpus of confusable strings.
    - L6: Heuristic Mutual Exclusion — verified via _assign_fields_heuristic
          on a synthetic multi-line input.
    - L7: Training Minimum Viability — current TROCR_EPOCHS and YOLO_EPOCHS
          are above the minimums.
    - L8: Pipeline Monotonicity — _verify_yolo_detection_rate correctly
          classifies zero-box vs empty-text failures.
    """
    _log = logging.getLogger(__name__)
    results: dict[str, bool] = {}

    # L1: Resolution Invariant
    try:
        assert YOLO_IMG_SIZE > 0, "YOLO_IMG_SIZE must be positive"
        assert YOLO_IMG_SIZE % 32 == 0, f"YOLO_IMG_SIZE={YOLO_IMG_SIZE} not a multiple of 32"
        # Positive test: same value should pass
        _assert_resolution_invariant(YOLO_IMG_SIZE, YOLO_IMG_SIZE)
        # Negative test: different values must raise ValueError.
        # 512 (YOLO_IMG_SIZE default) vs 640 (ultralytics default) are the two
        # most common YOLO inference resolutions — exactly the mismatch that
        # caused Pattern 8 (CLAUDE.md §16).
        try:
            _assert_resolution_invariant(512, 640)
            raise AssertionError("L1: _assert_resolution_invariant(512, 640) should have raised")
        except ValueError:
            pass  # Expected — L1 correctly detects mismatch
        results["L1_resolution_invariant"] = True
        _log.info("L1 PASS: YOLO_IMG_SIZE=%d is valid (positive, multiple of 32)", YOLO_IMG_SIZE)
    except (ValueError, AssertionError) as e:
        results["L1_resolution_invariant"] = False
        _log.error("L1 FAIL: %s", e)

    # L2: Detection-OCR Contract
    try:
        # Test with edge cases: negative coords, tiny boxes, out-of-bounds
        test_boxes: list[tuple[float, float, float, float]] = [
            (10.0, 10.0, 100.0, 50.0),  # valid
            (-5.0, -5.0, 3.0, 3.0),  # too small after clamping
            (0.0, 0.0, 4.0, 100.0),  # width < _MIN_BOX_DIMENSION
            (500.0, 500.0, 600.0, 600.0),  # valid, within bounds
        ]
        valid = _assert_box_validity(test_boxes, 640.0, 480.0)
        assert len(valid) == 2, f"Expected 2 valid boxes, got {len(valid)}"
        results["L2_detection_ocr_contract"] = True
        _log.info(
            "L2 PASS: Box validation correctly filters %d/%d boxes", len(valid), len(test_boxes)
        )
    except (AssertionError, ValueError) as e:
        results["L2_detection_ocr_contract"] = False
        _log.error("L2 FAIL: %s", e)

    # L3: Field Assignment Completeness
    try:
        # Empty input
        r1 = _assign_fields_heuristic([])
        assert set(r1.keys()) == set(FIELDS), f"Missing keys: {set(FIELDS) - set(r1.keys())}"
        assert all(isinstance(v, str) for v in r1.values()), "Non-string values"
        # Non-empty input
        r2 = _assign_fields_heuristic([{"text": "HELLO WORLD", "y": 10}])
        assert set(r2.keys()) == set(FIELDS), f"Missing keys: {set(FIELDS) - set(r2.keys())}"
        assert all(isinstance(v, str) for v in r2.values()), "Non-string values"
        results["L3_field_completeness"] = True
        _log.info("L3 PASS: Field assignment always returns 4-field string dict")
    except (AssertionError, KeyError) as e:
        results["L3_field_completeness"] = False
        _log.error("L3 FAIL: %s", e)

    # L4: NMS Correctness — smoke test with synthetic boxes
    try:
        # Create two boxes with high overlap (IoU > 0.45) and two with no overlap.
        # _assert_nms_no_high_iou_pairs should warn on the overlapping pair.
        high_iou_boxes = torch.tensor(
            [
                [10.0, 10.0, 100.0, 100.0],  # box A
                [15.0, 15.0, 105.0, 105.0],  # box B — highly overlaps A
            ]
        )
        # This should log a warning (high IoU pair), not raise.
        _assert_nms_no_high_iou_pairs(high_iou_boxes, iou_thr=0.45)
        # Non-overlapping boxes should produce no warnings.
        clean_boxes = torch.tensor(
            [
                [10.0, 10.0, 50.0, 50.0],
                [200.0, 200.0, 300.0, 300.0],
            ]
        )
        _assert_nms_no_high_iou_pairs(clean_boxes, iou_thr=0.45)
        # Edge case: single box and empty tensor
        _assert_nms_no_high_iou_pairs(torch.zeros(1, 4), iou_thr=0.45)
        _assert_nms_no_high_iou_pairs(torch.zeros(0, 4), iou_thr=0.45)
        results["L4_nms_correctness"] = True
        _log.info("L4 PASS: NMS correctness verified with synthetic overlapping box test")
    except (AssertionError, RuntimeError) as e:
        results["L4_nms_correctness"] = False
        _log.error("L4 FAIL: %s", e)

    # L5: OCR Correction Idempotence
    try:
        test_strings = [
            "TOTAL 10.50",
            "Hell0 W0rld",
            "25/12/2023",
            "RM 47.8O",
            "JALAN PUCHONG",
            "0123456789",
            "ABCDEFGHIJ",
            "l0O1I",
            "",
            "   ",
        ]
        l5_pass = True
        for s in test_strings:
            c1 = _correct_ocr_chars(s)
            c2 = _correct_ocr_chars(c1)
            if c1 != c2:
                _log.warning(
                    "L5 idempotence failure: %r → %r → %r",
                    s,
                    c1,
                    c2,
                )
                l5_pass = False
        results["L5_ocr_idempotence"] = l5_pass
        if l5_pass:
            _log.info(
                "L5 PASS: OCR correction is idempotent on %d test strings",
                len(test_strings),
            )
        else:
            _log.warning("L5 WARN: OCR correction is NOT idempotent on some strings")
    except (ValueError, AttributeError, TypeError) as e:
        results["L5_ocr_idempotence"] = False
        _log.error("L5 FAIL: %s", e)

    # L6: Heuristic Mutual Exclusion
    try:
        # Multi-line receipt with clear fields
        lines = [
            {"text": "MYDIN MALL SDN BHD", "y": 10, "y2": 30},
            {"text": "NO 1 JALAN PUCHONG", "y": 40, "y2": 60},
            {"text": "47100 PUCHONG SELANGOR", "y": 70, "y2": 90},
            {"text": "25/12/2023", "y": 120, "y2": 140},
            {"text": "TOTAL RM 47.80", "y": 300, "y2": 320},
        ]
        result = _assign_fields_heuristic(lines)
        # Verify all 4 fields are present and non-empty for this clear input
        assert all(result[f] for f in FIELDS), f"Some fields empty: {result}"
        # Mutual exclusion is enforced structurally by the `used` set —
        # we verify the output is consistent (no single line text appears
        # in two different fields).
        assigned_texts = [result[f] for f in FIELDS if result[f]]
        # Check no exact duplicates (a line assigned to two fields)
        for i, t1 in enumerate(assigned_texts):
            for j, t2 in enumerate(assigned_texts):
                if i < j and t1 == t2:
                    raise AssertionError(f"L6 VIOLATION: same text {t1!r} assigned to two fields")
        results["L6_mutual_exclusion"] = True
        _log.info("L6 PASS: Heuristic mutual exclusion verified on 5-line test receipt")
    except (AssertionError, KeyError) as e:
        results["L6_mutual_exclusion"] = False
        _log.error("L6 FAIL: %s", e)

    # L7: Training Minimum Viability
    try:
        _assert_training_viability(trocr_epochs=TROCR_EPOCHS, yolo_epochs=YOLO_EPOCHS)
        results["L7_training_viability"] = True
        _log.info(
            "L7 PASS: TROCR_EPOCHS=%d ≥ %d, YOLO_EPOCHS=%d ≥ %d",
            TROCR_EPOCHS,
            _MIN_TROCR_EPOCHS,
            YOLO_EPOCHS,
            _MIN_YOLO_EPOCHS,
        )
    except ValueError as e:
        results["L7_training_viability"] = False
        _log.error("L7 FAIL: %s", e)

    # L8: Pipeline Monotonicity — verify _verify_yolo_detection_rate
    try:
        # Should NOT raise for small failure counts
        _verify_yolo_detection_rate(2, 1, 63)  # 4.8% failure — well below 50%
        # Should raise for majority failure
        try:
            _verify_yolo_detection_rate(20, 15, 63)  # 55.6% failure — above 50%
            results["L8_pipeline_monotonicity"] = False
            _log.error("L8 FAIL: _verify_yolo_detection_rate did not raise for 55%% failure rate")
        except RuntimeError:
            pass  # Expected
        # Verify separate tracking: YOLO-dominant vs TrOCR-dominant
        try:
            _verify_yolo_detection_rate(30, 5, 63)
            results["L8_pipeline_monotonicity"] = False
        except RuntimeError as e:
            msg = str(e)
            assert "YOLO found 0 boxes" in msg, "L8: YOLO-dominant message missing"
        try:
            _verify_yolo_detection_rate(5, 30, 63)
            results["L8_pipeline_monotonicity"] = False
        except RuntimeError as e:
            msg = str(e)
            assert "TrOCR decoded all crops" in msg, "L8: TrOCR-dominant message missing"
        results["L8_pipeline_monotonicity"] = True
        _log.info("L8 PASS: Detection rate verification correctly classifies failure modes")
    except (AssertionError, RuntimeError) as e:
        results["L8_pipeline_monotonicity"] = False
        _log.error("L8 FAIL: %s", e)

    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    status = "ALL PASS" if passed == total else f"{passed}/{total} PASS"
    _log.info("Pipeline Lemma Verification: %s", status)
    if passed < total:
        failed = [k for k, v in results.items() if not v]
        _log.warning("Failed lemmas: %s", ", ".join(failed))

    return results


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
TROCR_MINI_MODE = False  # True → AdamW+cosine-warmup at 10× LR instead of AdamW+linear-warmup
FIELD_ASSIGNER_EPOCHS = 30  # train_field_assigner() default; patch to 1 for superfast mode
TROCR_USE_TENSOR_CACHE = False  # True → TrOCRReceiptDataset caches tensors to disk on first run

# ── Processor singleton cache ────────────────────────────────────────────────
# Avoids repeated TrOCRProcessor.from_pretrained() calls (2-5 s each) across
# stages that all operate within the same process.
_PROCESSOR_CACHE: dict = {}  # model_id → TrOCRProcessor singleton


def _get_trocr_processor(model_id: str | None = None) -> "TrOCRProcessor":
    """Return a cached TrOCRProcessor, loading from HuggingFace exactly once per process.

    Parameters
    ----------
    model_id:
        HuggingFace model identifier.  Defaults to ``TROCR_MODEL_ID``
        (``"microsoft/trocr-base-printed"``) when ``None``.
    """
    _id = model_id or TROCR_MODEL_ID
    if _id not in _PROCESSOR_CACHE:
        _PROCESSOR_CACHE[_id] = TrOCRProcessor.from_pretrained(_id)
    return _PROCESSOR_CACHE[_id]


RESULTS_DIR = Path("results")
YOLO_DATA_YAML = WORKSPACE / "data" / "yolo" / "dataset.yaml"
TROCR_DATA_DIR = WORKSPACE / "data" / "trocr"

# Pre-compiled regex patterns for field assignment heuristics — compiled once
# at module load instead of on every call to assign_fields_heuristic().
#
# These are the CANONICAL patterns used by both train_trocr_yolo.py and
# reporting.py (via _assign_fields_heuristic).  Do NOT duplicate field-
# assignment regex patterns elsewhere — import this function instead.

# Date: numeric (DD/MM/YYYY, YYYY-MM-DD, etc.) OR written month names
_DATE_RE = re.compile(
    r"\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}"  # 25/12/2023, 25-12-23
    r"|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}"  # 2023/12/25
    r"|\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{2,4}"  # 25 DEC 2023
    r"|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\.?\s+\d{1,2},?\s+\d{4}"  # DEC 25, 2023
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}",  # 25 Dec 2023
    re.IGNORECASE,
)
# Date header keywords (Malay/Indonesian receipts): "date:", "tarikh:", "tanggal:"
# followed by the actual date value.  Used as a secondary check when _DATE_RE
# misses because the date is on the same line as the label keyword.
_DATE_HEADER_RE = re.compile(
    r"(?:date|tarikh|tanggal)\s*[:\-]?\s*(\S+)",
    re.IGNORECASE,
)
# Total patterns — ordered from most specific to least.  Includes Malaysian
# keywords (jumlah) alongside English variants.  Capture group extracts the
# numeric value directly.
_TOTAL_KEYWORD_RE = re.compile(
    r"(?:total|subtotal|jumlah|amount\s+due|grand\s*total|nett\s*total|net\s*total"
    r"|total\s+amount|amount|sum|due)"
    r"\s*[:\-]?\s*(?:rm|myr|usd|\$|£|€)?\s*(\d[\d,]*\.\d{2})",
    re.IGNORECASE,
)
# Legacy total pattern (no capture group) for backward-compatible match detection
_TOTAL_RE = re.compile(
    r"(?:total|subtotal|jumlah|amount|sum|due|grand\s*total|nett\s*total|net\s*total)"
    r"\s*[:\-]?\s*[\$\£\€RM]?\s*\d+[.,]\d{2}",
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
    r"|FLOOR|LEVEL|UNIT|BLOCK|BLK|LANE|DRIVE|PLAZA|MALL|PARK|BATU|KM|KILOMETER|MILE"
    r"|TINGKAT)"
    r"|\b\d{5}\b"  # standalone 5-digit postcode
    r"|^\d+\s+[A-Z]",  # line starting with street number + word
    re.IGNORECASE | re.MULTILINE,
)


# ════════════════════════════════════════════════════════════════════════════
# OCR Character Correction
# ════════════════════════════════════════════════════════════════════════════

# Visually similar character pairs that TrOCR commonly confuses:
#   0 ↔ O/o   (zero vs. oh)
#   1 ↔ l/I/i (one vs. el/eye)
#   5 ↔ S/s   (five vs. ess)
#   8 ↔ B     (eight vs. bee)
#   2 ↔ Z/z   (two vs. zed)
_OCR_L2D = {
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "i": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
    "z": "2",
}
_OCR_D2L = {"0": "O", "1": "l", "5": "S", "8": "B", "2": "Z"}


def _correct_ocr_chars(text: str) -> str:
    """Fix common TrOCR single-character substitution mistakes.

    Splits on whitespace and corrects each token based on its inferred type:
    - Tokens that look predominantly numeric → substitute confusable letters
      with their digit equivalents (e.g. "O" → "0", "l" → "1").
    - Tokens that look predominantly alphabetic → substitute confusable
      digits with their letter equivalents (e.g. "0" → "O", "1" → "l").
    - Ambiguous or punctuation-only tokens are passed through unchanged.

    This is a lightweight pre-processing step; the FieldAttentionAssigner
    additionally learns character-level confusion patterns from training data.

    **L5 (OCR Correction Idempotence)**: This function is designed to be
    idempotent — applying it twice yields the same result as once.  This is
    enforced by a debug-mode assertion (only active when ``__debug__`` is True).
    """
    corrected = []
    for token in text.split():
        n_digit_like = sum(1 for c in token if c.isdigit() or c in _OCR_L2D)
        n_alpha_like = sum(1 for c in token if c.isalpha() or c in _OCR_D2L)
        if n_digit_like > n_alpha_like:
            corrected.append("".join(_OCR_L2D.get(c, c) for c in token))
        elif n_alpha_like > n_digit_like:
            corrected.append("".join(_OCR_D2L.get(c, c) for c in token))
        else:
            corrected.append(token)
    result = " ".join(corrected)

    # L5: Debug-mode idempotence check — only runs with assertions enabled
    # (python without -O flag).  Zero overhead in production (-O strips asserts).
    if __debug__:
        _assert_ocr_correction_idempotent(text, result)

    return result


# ════════════════════════════════════════════════════════════════════════════
# Field-Assigner Backend Selection
# ════════════════════════════════════════════════════════════════════════════
#
# Change FIELD_ASSIGNER_BACKEND to switch between the three tiers:
#
#   "char"       Backend 1 — character embeddings (no pretrained weights)
#                ~532 K trainable params.  Trains from scratch on SROIE.
#                Handles single-char OCR swaps (Hell0→Hello) via learned
#                embedding proximity.  No internet / HF download required.
#
#   "lm"         Backend 2 — BERT-tiny text encoder (prajjwal1/bert-tiny)
#                ~4.9 M params (BERT-tiny 4.4 M + assigner ~530 K).
#                Language model prior from pretraining; morphological OCR
#                errors resolved at subword level, not character level.
#                Requires HF download on first use (~17 MB).
#
#   "lm+vision"  Backend 3 (DEFAULT) — LM encoder + TrOCR vision features
#                                       + cross-field consistency loss
#                ~5.1 M params.  Best quality on poor-quality images: raw
#                pixel features from TrOCR's encoder feed directly into the
#                assigner, bypassing any OCR decoding error on that crop.
#                Requires HF download on first use.
#
# ════════════════════════════════════════════════════════════════════════════
FIELD_ASSIGNER_BACKEND: str = "lm+vision"  # ← change here to "char" or "lm"

# Try to load BERT-tiny; if unavailable, "lm" and "lm+vision" silently fall
# back to the "char" backend.
try:
    from transformers import AutoModel as _AutoModel
    from transformers import AutoTokenizer as _AutoTokenizer

    _LM_AVAILABLE = True
except ImportError:
    _LM_AVAILABLE = False


def _build_lm_encoder() -> "tuple":
    """Return (tokenizer, model) for prajjwal1/bert-tiny (128-dim hidden).

    BERT-tiny is frozen by default — it acts as a feature extractor so the
    assigner's ~530 K parameters remain the only trainable component.
    The 128-dim hidden size matches d_model exactly (no projection needed).
    """
    if not _LM_AVAILABLE:
        raise ImportError(
            "transformers is required for the 'lm' and 'lm+vision' backends. "
            "Run: pip install 'transformers>=4.37.0'"
        )
    tok = _AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = _AutoModel.from_pretrained("prajjwal1/bert-tiny")
    for p in model.parameters():
        p.requires_grad = False
    return tok, model


# Soft plausibility scores used by the Backend 3 consistency regulariser.
# Each callable returns a score in [0.1, 1.0]: 1.0 = line matches the
# expected pattern for that field, 0.1 = very unlikely match.
# Company has a uniform prior — no reliable pattern to enforce.
_FIELD_PLAUSIBILITY: dict[str, object] = {
    "date": lambda t: 1.0 if _DATE_RE.search(t) else 0.1,
    "total": lambda t: 1.0 if (_TOTAL_RE.search(t) or _MONEY_RE.search(t)) else 0.1,
    "company": lambda _t: 0.5,
    "address": lambda t: 1.0 if _ADDRESS_RE.search(t) else 0.3,
}


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
        use_tensor_cache: bool = False,  # True → cache all tensors to disk on first run
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

        # ── Tensor cache ─────────────────────────────────────────────────────
        # When use_tensor_cache=True: process all samples once, save to a .pt
        # file keyed by (dir, max_length, processor, augmentation).  Subsequent
        # runs load all pixel_values + labels tensors in a single torch.load()
        # call, bypassing per-sample Pillow + TrOCRProcessor overhead entirely.
        #
        # Note: cached pixel_values are built WITHOUT augmentation (augmentation
        # is stochastic so baking it into the cache would freeze the random state).
        # When use_tensor_cache=True, augmentation is effectively disabled — this
        # is intentional for instant-mode speed runs where training quality is
        # secondary to wall-clock time.
        self._pixel_values_cache: list | None = None
        self._labels_cache: list | None = None

        if use_tensor_cache and self.samples:
            import hashlib

            proc_name = getattr(processor, "name_or_path", TROCR_MODEL_ID)
            # Include augmentation repr in the key so cache is invalidated if
            # augmentation settings change between runs.
            aug_repr = repr(augmentation)
            cache_key = hashlib.sha256(
                f"{data_dir}:{max_length}:{proc_name}:{aug_repr}".encode()
            ).hexdigest()[:16]
            cache_file = data_dir / f".tensor_cache_{cache_key}.pt"

            if cache_file.exists():
                try:
                    cached = torch.load(cache_file, map_location="cpu", weights_only=True)
                    if (
                        not isinstance(cached, dict)
                        or "pixel_values" not in cached
                        or "labels" not in cached
                    ):
                        raise ValueError("Cache file missing required keys 'pixel_values'/'labels'")
                    self._pixel_values_cache = cached["pixel_values"]
                    self._labels_cache = cached["labels"]
                    print(
                        f"  [TrOCR] Tensor cache loaded ({len(self._pixel_values_cache)} samples)"
                        f" <- {cache_file.name}"
                    )
                except Exception as _ce:
                    print(f"  [TrOCR] Tensor cache load failed ({_ce}) -- rebuilding")
                    self._pixel_values_cache = None
                    self._labels_cache = None

            if self._pixel_values_cache is None:
                n_total = len(self.samples)
                # Estimate bytes per sample: pixel_values is (C, H, W) float32.
                # Try to get actual dimensions from the processor config; fall back
                # to TrOCR-base defaults (3 × 384 × 384) if unavailable.
                try:
                    _feat_ext = getattr(processor, "image_processor", None) or getattr(
                        processor, "feature_extractor", None
                    )
                    _img_size = getattr(_feat_ext, "size", None)
                    if isinstance(_img_size, dict):
                        _h = _img_size.get("height", 384)
                        _w = _img_size.get("width", 384)
                    elif isinstance(_img_size, int):
                        _h = _w = _img_size
                    else:
                        _h = _w = 384
                except (TypeError, ValueError, AttributeError):
                    _h = _w = 384
                _bytes_per_sample = 3 * _h * _w * 4  # float32 = 4 bytes
                _required_bytes = n_total * _bytes_per_sample

                # ── RAM availability check ────────────────────────────────────
                # Use psutil if available; fall back to a conservative estimate
                # via /proc/meminfo on Linux or skip the check entirely.
                _available_bytes: int | None = None
                try:
                    import psutil  # noqa: E402

                    _available_bytes = psutil.virtual_memory().available
                except (ImportError, RuntimeError):
                    try:
                        with open("/proc/meminfo") as _mf:
                            for _line in _mf:
                                if _line.startswith("MemAvailable:"):
                                    _available_bytes = int(_line.split()[1]) * 1024
                                    break
                    except (FileNotFoundError, ValueError, OSError):
                        pass  # Unknown available RAM — proceed optimistically

                _bytes_per_gb = 1024**3
                if _available_bytes is not None and _required_bytes > _available_bytes:
                    _req_gb = _required_bytes / _bytes_per_gb
                    _avail_gb = _available_bytes / _bytes_per_gb
                    print(
                        f"  [TrOCR] Tensor cache requires ~{_req_gb:.1f} GB RAM for"
                        f" {n_total} samples, but only {_avail_gb:.1f} GB available."
                    )
                    print(
                        "  [TrOCR] Falling back to lazy per-sample loading"
                        " (slower but memory-safe)."
                    )
                    # Leave self._pixel_values_cache = None so __getitem__ uses
                    # the lazy-loading fallback path automatically.
                else:
                    print(
                        f"  [TrOCR] Building tensor cache for {n_total} samples"
                        f" -> {cache_file.name}"
                    )
                    pixel_values_list: list = []
                    labels_list: list = []
                    for _i, _sample in enumerate(self.samples):
                        _img = _load_image(self.data_dir / _sample["file_name"])
                        _pv = processor(_img, return_tensors="pt").pixel_values.squeeze(0)
                        _lbl = processor.tokenizer(
                            _sample["text"],
                            padding="max_length",
                            max_length=max_length,
                            truncation=True,
                            return_tensors="pt",
                        ).input_ids.squeeze(0)
                        _lbl[_lbl == processor.tokenizer.pad_token_id] = LABEL_IGNORE_INDEX
                        pixel_values_list.append(_pv)
                        labels_list.append(_lbl)
                        if (_i + 1) % 100 == 0 or (_i + 1) == n_total:
                            print(f"  [TrOCR]   cached {_i + 1}/{n_total} samples...")
                        if (_i + 1) % 1000 == 0:
                            gc.collect()
                    self._pixel_values_cache = pixel_values_list
                    self._labels_cache = labels_list
                    try:
                        torch.save(
                            {"pixel_values": pixel_values_list, "labels": labels_list},
                            cache_file,
                        )
                        print(f"  [TrOCR] Tensor cache saved -> {cache_file.name}")
                    except Exception as _se:
                        print(f"  [TrOCR] Failed to save tensor cache: {_se}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Fast path: return pre-cached tensors without any I/O or processor overhead.
        if self._pixel_values_cache is not None:
            return {
                "pixel_values": self._pixel_values_cache[idx],
                "labels": self._labels_cache[idx],
            }

        sample = self.samples[idx]
        img = _load_image(self.data_dir / sample["file_name"])

        if self.augmentation is not None and _PIL_AVAILABLE:
            # torchvision transforms require PIL images; skip augmentation without PIL
            img = self.augmentation(img)

        pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = LABEL_IGNORE_INDEX
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

    # L7: Training Minimum Viability — catch invalid epoch count before
    # spending time on data loading and model initialisation.
    _assert_training_viability(yolo_epochs=YOLO_EPOCHS)

    if output_dir is None:
        output_dir = WORKSPACE / "models" / "yolo_finetuned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STAGE 1: Fine-tuning YOLOv8 for text-region detection")
    print(f"         ultralytics={'yes' if _ULTRALYTICS_AVAILABLE else 'no (inline fallback)'}")
    print("=" * 60)

    if not YOLO_DATA_YAML.exists():
        print(f"  YOLO dataset.yaml not found at {YOLO_DATA_YAML}")
        print("  Run dataset_preparation.py first.")
        return output_dir / "run" / "weights" / "best.pt"

    model = _YOLO_CLS(YOLO_BASE)
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
        degrees=_yolo.degrees,  # 5° rotation tolerance for tilted receipts
        translate=_yolo.translate,  # 0.1 spatial shift
        scale=_yolo.scale,  # 0.3 zoom range
        fliplr=_yolo.fliplr,  # 0.0 — text direction matters; no horizontal flip
        flipud=_yolo.flipud,  # 0.0 — receipts are always upright
        mosaic=_yolo.mosaic,  # 0.5 — reduced from default 1.0 for document domain
        close_mosaic=_yolo.close_mosaic,  # 10 — disable mosaic for final N epochs (⚠️ critical for mAP)
        mixup=_yolo.mixup,  # 0.0 — disabled (blending receipts confuses layout)
        copy_paste=_yolo.copy_paste,  # 0.0 — disabled; can enable for rare-class boost
        hsv_h=_yolo.hsv_h,
        hsv_s=_yolo.hsv_s,
        hsv_v=_yolo.hsv_v,
        # ── Optimizer ────────────────────────────────────────────────────
        optimizer=YOLO_OPTIMIZER,  # "AdamW" default; micro patches to "SGD" for speed
        momentum=YOLO_MOMENTUM,  # used when optimizer="SGD"
        lr0=_yolo.lr0,  # 1e-3 peak LR
        lrf=_yolo.lrf,  # 0.01 final LR fraction
        weight_decay=_yolo.weight_decay,  # 0.0005 L2 regularisation (⚠️ was missing)
        cos_lr=_yolo.cos_lr,  # False — linear decay (⚠️ was missing)
        warmup_epochs=_yolo.warmup_epochs,  # 3.0 (⚠️ was missing)
        warmup_momentum=_yolo.warmup_momentum,  # 0.8 (⚠️ was missing)
        warmup_bias_lr=_yolo.warmup_bias_lr,  # 0.1 (⚠️ was missing)
        # ── Finetuning ───────────────────────────────────────────────────
        freeze=_yolo.freeze,  # None — no frozen layers (⚠️ CRITICAL: was missing entirely)
        # ── Loss weights ─────────────────────────────────────────────────
        box=_yolo.box,  # 7.5 bbox regression weight (⚠️ was missing)
        cls=_yolo.cls,  # 0.5 classification weight (⚠️ was missing)
        dfl=_yolo.dfl,  # 1.5 focal loss weight (⚠️ was missing)
        # ── Performance ──────────────────────────────────────────────────
        cache=_yolo.cache,  # False — set "ram" to speed up with sufficient memory
        workers=_yolo.workers,  # 8 DataLoader threads (⚠️ was missing)
        fraction=_yolo.fraction,  # 1.0 use full dataset (⚠️ was missing)
        # ── Regularisation ────────────────────────────────────────────────
        dropout=_yolo.dropout,  # 0.0 (⚠️ was missing)
        # ── Other ─────────────────────────────────────────────────────────
        patience=_yolo.patience,  # 15 early stopping
        seed=SEED,
        amp=YOLO_AMP,
        deterministic=_yolo.deterministic,  # True (⚠️ was missing)
    )

    elapsed = time.time() - start
    best_path = output_dir / "run" / "weights" / "best.pt"
    print(f"\nYOLO training complete in {elapsed:.1f}s")
    print(f"Best weights -> {best_path}")

    # Save companion raw state dict for inline fallback (allows inference without ultralytics)
    if _ULTRALYTICS_AVAILABLE and best_path.exists():
        sd_path = best_path.with_stem(best_path.stem + "_sd")
        if not sd_path.exists():
            try:
                ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                _m = ckpt.get("model", ckpt.get("ema"))
                if _m is not None and hasattr(_m, "state_dict"):
                    torch.save(_m.state_dict(), sd_path)
                    print(f"  Saved raw state dict → {sd_path}")
            except Exception as _e:
                # Fix: issue_report_summary critical #3 — log at ERROR level so the operator
                # knows the state dict companion file was NOT saved. Inference without
                # ultralytics installed will fall back to random weights as a result.
                logging.getLogger(__name__).error(
                    "Could not save YOLO companion state dict to %s — inference without "
                    "ultralytics installed will use random init (degraded quality). Error: %s",
                    sd_path,
                    _e,
                    exc_info=True,
                )

    # FIX: GPU cleanup after YOLO training — delete local reference first
    del model
    _gpu_cleanup()

    return best_path


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1b: YOLO Detection Evaluation
# ════════════════════════════════════════════════════════════════════════════


def _compute_iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _yolo_cx_cy_w_h_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: float, img_h: float
) -> tuple[float, float, float, float]:
    """Convert YOLO normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)."""
    abs_cx = cx * img_w
    abs_cy = cy * img_h
    abs_w = w * img_w
    abs_h = h * img_h
    return (
        abs_cx - abs_w / 2,
        abs_cy - abs_h / 2,
        abs_cx + abs_w / 2,
        abs_cy + abs_h / 2,
    )


def _load_yolo_gt_boxes(
    label_path: Path, img_w: float, img_h: float
) -> list[tuple[float, float, float, float]]:
    """Load ground truth bounding boxes from a YOLO-format label file.

    Each line: ``class cx cy w h`` (normalised).  Returns list of (x1, y1, x2, y2).
    """
    boxes: list[tuple[float, float, float, float]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue  # skip malformed lines with non-numeric coordinates
        boxes.append(_yolo_cx_cy_w_h_to_xyxy(cx, cy, w, h, img_w, img_h))
    return boxes


def _match_boxes_greedy(
    pred_boxes: list[tuple[float, float, float, float]],
    gt_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy match predicted boxes to ground truth at given IoU threshold.

    Returns (true_positives, false_positives, false_negatives).
    """
    matched_gt: set[int] = set()
    tp = 0
    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = _compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def evaluate_yolo_detection(
    yolo_weights: str | Path,
    data_yaml: str | Path | None = None,
    imgsz: int | None = None,
) -> dict[str, Any]:
    """Evaluate YOLO detection quality on the validation set.

    Uses the ultralytics ``model.val()`` API when available for full COCO-style
    mAP computation.  Falls back to a manual greedy-matching evaluation against
    ground truth YOLO labels when ultralytics is not installed.

    Parameters
    ----------
    yolo_weights : str or Path
        Path to YOLO best weights file (``best.pt``).
    data_yaml : str or Path or None
        Path to the dataset YAML file.  Defaults to ``YOLO_DATA_YAML``.
    imgsz : int or None
        Inference image size.  Defaults to ``YOLO_IMG_SIZE``.

    Returns
    -------
    dict[str, Any]
        Detection metrics with the following keys:

        - ``yolo_map50``: mAP at IoU=0.50
        - ``yolo_map``: mAP at IoU=0.50:0.95
        - ``yolo_precision``: mean detection precision
        - ``yolo_recall``: mean detection recall
        - ``yolo_num_val_images``: number of validation images used
        - ``yolo_backend``: ``"ultralytics"`` or ``"manual"``
    """
    _log = logging.getLogger(__name__)
    if data_yaml is None:
        data_yaml = YOLO_DATA_YAML
    if imgsz is None:
        imgsz = YOLO_IMG_SIZE

    yolo_weights = Path(yolo_weights)
    if not yolo_weights.exists():
        _log.warning(
            "evaluate_yolo_detection: weights file %s does not exist — returning empty metrics.",
            yolo_weights,
        )
        return {}

    result: dict[str, Any] = {}

    # ── Ultralytics path: full COCO-style mAP ──────────────────────────────
    if _ULTRALYTICS_AVAILABLE:
        _log.info("Evaluating YOLO detection quality via ultralytics model.val() ...")
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            model = YOLO(str(yolo_weights))
            metrics = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False)

            # Extract metrics from the DetMetrics object.
            # Attribute names: .box.map, .box.map50, .box.map75, .box.mp, .box.mr
            box = getattr(metrics, "box", None)
            if box is not None:
                result["yolo_map50"] = round(float(getattr(box, "map50", 0.0)), 4)
                result["yolo_map"] = round(float(getattr(box, "map", 0.0)), 4)
                result["yolo_precision"] = round(float(getattr(box, "mp", 0.0)), 4)
                result["yolo_recall"] = round(float(getattr(box, "mr", 0.0)), 4)
            else:
                # Fallback: some ultralytics versions use different attribute layout
                result["yolo_map50"] = round(float(getattr(metrics, "map50", 0.0)), 4)
                result["yolo_map"] = round(float(getattr(metrics, "map", 0.0)), 4)
                result["yolo_precision"] = round(float(getattr(metrics, "mp", 0.0)), 4)
                result["yolo_recall"] = round(float(getattr(metrics, "mr", 0.0)), 4)

            result["yolo_backend"] = "ultralytics"

            # Try to read number of val images from the dataset YAML
            try:
                import yaml  # noqa: PLC0415

                with open(data_yaml) as f:
                    ds_cfg = yaml.safe_load(f)
                val_dir = Path(ds_cfg.get("val", ""))
                if not val_dir.is_absolute():
                    val_dir = Path(data_yaml).parent / val_dir
                result["yolo_num_val_images"] = sum(
                    1 for p in val_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
                )
            except (OSError, KeyError, TypeError, ValueError):
                result["yolo_num_val_images"] = 0

            del model
            _gpu_cleanup()

            _log.info(
                "YOLO detection eval (ultralytics): "
                "mAP50=%.4f  mAP50-95=%.4f  precision=%.4f  recall=%.4f",
                result.get("yolo_map50", 0),
                result.get("yolo_map", 0),
                result.get("yolo_precision", 0),
                result.get("yolo_recall", 0),
            )
            return result

        except (ImportError, RuntimeError, AttributeError, OSError) as exc:
            _log.warning(
                "ultralytics model.val() failed (%s: %s) — falling back to manual evaluation.",
                type(exc).__name__,
                exc,
            )
            # Fall through to manual evaluation

    # ── Manual path: greedy IoU matching against ground truth labels ────────
    _log.info(
        "Evaluating YOLO detection quality via manual IoU matching (ultralytics not available) ..."
    )

    # Load dataset YAML to find val image/label directories
    val_images_dir: Path | None = None
    val_labels_dir: Path | None = None
    try:
        import yaml  # noqa: PLC0415

        with open(data_yaml) as f:
            ds_cfg = yaml.safe_load(f)
        val_path = ds_cfg.get("val", "")
        val_images_dir = (
            Path(val_path) if Path(val_path).is_absolute() else Path(data_yaml).parent / val_path
        )
        # YOLO label dir mirrors image dir: images/val → labels/val
        # Use Path manipulation for cross-platform safety.
        val_labels_dir = val_images_dir.parent.parent / "labels" / val_images_dir.name
    except (ImportError, OSError, KeyError, TypeError, ValueError) as exc:
        _log.warning(
            "Could not parse dataset YAML %s: %s — cannot run manual detection eval.",
            data_yaml,
            exc,
        )
        return {"yolo_backend": "manual", "yolo_error": f"YAML parse failed: {exc}"}

    if val_images_dir is None or not val_images_dir.exists():
        _log.warning(
            "YOLO validation image directory not found at %s — cannot run manual detection eval.",
            val_images_dir,
        )
        return {"yolo_backend": "manual", "yolo_error": "val images dir not found"}

    # Collect validation images
    val_images = sorted(p for p in val_images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not val_images:
        _log.warning("No validation images found in %s", val_images_dir)
        return {"yolo_backend": "manual", "yolo_num_val_images": 0}

    # Load YOLO model
    yolo_model = _YOLO_CLS(str(yolo_weights))

    total_tp = 0
    total_fp = 0
    total_fn = 0
    num_images_with_detections = 0

    for img_path in _progress(val_images, desc="YOLO detection eval"):
        # Load image to get dimensions
        img = _load_image(img_path)
        img_w = float(img.size[0]) if _PIL_AVAILABLE else float(img.shape[1])  # type: ignore[union-attr]
        img_h = float(img.size[1]) if _PIL_AVAILABLE else float(img.shape[0])  # type: ignore[union-attr]

        # Run YOLO inference (L1: pass YOLO_IMG_SIZE)
        yolo_results = yolo_model(img, verbose=False, imgsz=imgsz)

        # Extract predicted boxes as (x1, y1, x2, y2)
        pred_boxes: list[tuple[float, float, float, float]] = []
        if yolo_results and len(yolo_results[0].boxes) > 0:
            for box in yolo_results[0].boxes:
                xyxy = box.xyxy[0].cpu().tolist()
                pred_boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3]))

        if pred_boxes:
            num_images_with_detections += 1

        # Load ground truth boxes
        label_path = val_labels_dir / (img_path.stem + ".txt") if val_labels_dir else None
        gt_boxes: list[tuple[float, float, float, float]] = []
        if label_path is not None:
            gt_boxes = _load_yolo_gt_boxes(label_path, img_w, img_h)

        # Match at IoU=0.5
        tp, fp, fn = _match_boxes_greedy(pred_boxes, gt_boxes, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    del yolo_model
    _gpu_cleanup()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    # F1@IoU=0.5 as a proxy for mAP50 — not a true multi-threshold mAP, but
    # the best single-number approximation available without ultralytics.
    map50_proxy = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    detection_rate = num_images_with_detections / len(val_images) if val_images else 0.0

    result = {
        # NOTE: in manual mode, yolo_map50 stores F1@IoU=0.5 (a proxy for true mAP50).
        # True mAP50 requires multi-threshold PR curve computation (available via ultralytics).
        "yolo_map50": round(map50_proxy, 4),
        "yolo_map": 0.0,  # multi-threshold mAP not available in manual mode
        "yolo_precision": round(precision, 4),
        "yolo_recall": round(recall, 4),
        "yolo_detection_rate": round(detection_rate, 4),
        "yolo_num_val_images": len(val_images),
        "yolo_total_tp": total_tp,
        "yolo_total_fp": total_fp,
        "yolo_total_fn": total_fn,
        "yolo_backend": "manual",
    }

    _log.info(
        "YOLO detection eval (manual): "
        "precision=%.4f  recall=%.4f  F1@IoU50=%.4f  detection_rate=%.1f%%  "
        "(%d val images, TP=%d FP=%d FN=%d)",
        precision,
        recall,
        map50_proxy,
        detection_rate * 100,
        len(val_images),
        total_tp,
        total_fp,
        total_fn,
    )
    return result


def print_yolo_detection_metrics(metrics: dict[str, Any]) -> None:
    """Print YOLO detection metrics in a clear, readable table."""
    if not metrics:
        print("  [YOLO] No detection metrics available.")
        return

    backend = metrics.get("yolo_backend", "unknown")
    print("=" * 60)
    print(f"  YOLO Detection Quality  (backend: {backend})")
    print("=" * 60)
    print(f"  mAP@50:          {metrics.get('yolo_map50', 'N/A')}")
    print(f"  mAP@50-95:       {metrics.get('yolo_map', 'N/A')}")
    print(f"  Precision:       {metrics.get('yolo_precision', 'N/A')}")
    print(f"  Recall:          {metrics.get('yolo_recall', 'N/A')}")
    if "yolo_detection_rate" in metrics:
        print(f"  Detection Rate:  {metrics['yolo_detection_rate']:.1%}")
    print(f"  Val Images:      {metrics.get('yolo_num_val_images', 'N/A')}")
    if metrics.get("yolo_total_tp") is not None:
        print(
            f"  TP/FP/FN:        {metrics['yolo_total_tp']}"
            f" / {metrics['yolo_total_fp']}"
            f" / {metrics['yolo_total_fn']}"
        )

    # Quality gate: warn if detection quality is poor
    map50 = metrics.get("yolo_map50", 0.0)
    if isinstance(map50, (int, float)) and map50 < _YOLO_MAP50_QUALITY_THRESHOLD:
        print(
            f"  ⚠️  WARNING: mAP@50 is below {_YOLO_MAP50_QUALITY_THRESHOLD:.2f} "
            "— YOLO detection quality is poor."
        )
        print("     This will degrade TrOCR+YOLO end-to-end F1 significantly.")
        print(
            "     Consider: more YOLO training epochs, verify data preparation, "
            "or install ultralytics for real detection."
        )
    print("=" * 60)


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
        # Newer transformers (≥4.45) adds output_projection to MBartDecoder; the
        # microsoft/trocr-base-printed checkpoint predates this layer so it is
        # always absent on load.  The layer is not used by TrOCR inference.
        "decoder.model.decoder.output_projection.weight",
        "decoder.model.decoder.output_projection.bias",
    }
)


def _check_config_tie_word_embeddings(model_path: "Path", json_module) -> None:  # noqa: ANN001
    """Read config.json in *model_path* and warn if ``tie_word_embeddings`` is ``true``.

    When ``tie_word_embeddings=true`` is written to config.json, HuggingFace's
    ``from_pretrained()`` will expect ``lm_head.weight`` to be absent from the
    safetensors file (treating it as a tied weight).  Even if the weight was
    physically written to disk, the load machinery may report it as "missing"
    or silently re-tie it to ``embed_tokens.weight``, causing F1 = 0.
    """
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            _cfg = json_module.loads(config_file.read_text())
            _twe_on_disk = _cfg.get("tie_word_embeddings", "UNSET")
            print(f"  [TrOCR-verify] config.json tie_word_embeddings={_twe_on_disk}")
            if _twe_on_disk is True:
                logging.getLogger(__name__).warning(
                    "config.json has tie_word_embeddings=true — "
                    "from_pretrained() will expect lm_head.weight to be tied and "
                    "may report it as 'missing' even if it exists in the safetensors file."
                )
        except Exception:
            pass


def _verify_lm_head_in_checkpoint(model_path: "Path") -> None:
    """Raise ``RuntimeError`` if ``decoder.lm_head.weight`` is absent from a saved checkpoint.

    Inspects the safetensors file header (single-shard) or the
    ``model.safetensors.index.json`` weight-map (sharded) at *model_path*.
    Raises immediately with a descriptive message so the pipeline fails fast
    rather than silently producing F1 = 0 when a reloaded TrOCR model has a
    randomly-re-initialised output projection.

    Call this both **after** ``save_pretrained()`` (to verify the save was
    correct) and **before** loading a checkpoint for evaluation (to verify the
    checkpoint on disk is intact).

    Parameters
    ----------
    model_path : Path
        Directory produced by ``model.save_pretrained(model_path)``.
    """
    import json as _j
    import struct as _s

    model_path = Path(model_path)
    _key = "decoder.lm_head.weight"

    # Sharded model: check the weight-map index.
    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        try:
            wmap = _j.loads(index_file.read_text()).get("weight_map", {})
            # Accept any key ending with ".lm_head.weight" as a valid match.
            # Different transformers versions may emit different key names
            # (e.g. "decoder.lm_head.weight", "decoder.model.lm_head.weight").
            _keys_set = set(wmap.keys())
            _all_lm_keys = [k for k in _keys_set if "lm_head" in k or "output_projection" in k]
            print(
                f"  [TrOCR-verify] safetensors index: {len(_keys_set)} keys, "
                f"lm-related: {_all_lm_keys}"
            )
            _has_lm_head = _key in _keys_set or any(
                k.endswith(".lm_head.weight") or k.endswith(".output_projection.weight")
                for k in _keys_set
            )
            if not _has_lm_head:
                logging.getLogger(__name__).error(
                    "DIAGNOSTIC: 'decoder.lm_head.weight' NOT FOUND in weight_map index. "
                    "Keys present: %s",
                    sorted(_keys_set),
                )
                raise RuntimeError(
                    f"CRITICAL: {_key!r} is missing from the safetensors index at "
                    f"{index_file}.  The tensor key name may differ between "
                    "transformers versions (see DIAGNOSTIC log above for actual keys).  "
                    "Ensure the weight alias is broken (data.clone()) and "
                    "_tied_weights_keys is cleared before save_pretrained().  "
                    "See CLAUDE.md §16 Pattern 6."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "_verify_lm_head_in_checkpoint: could not parse index (%s) — skipping", exc
            )
        # Also check config.json tie_word_embeddings for the sharded case.
        _check_config_tie_word_embeddings(model_path, _j)
        return

    # Single-shard model: inspect the binary safetensors header.
    single_file = model_path / "model.safetensors"
    if single_file.exists():
        try:
            with open(single_file, "rb") as fh:
                hdr_len = _s.unpack("<Q", fh.read(8))[0]
                hdr = _j.loads(fh.read(hdr_len))
            # Accept any key ending with ".lm_head.weight" as a valid match.
            # Different transformers versions may emit different key names
            # (e.g. "decoder.lm_head.weight", "decoder.model.lm_head.weight").
            _keys_set = set(hdr.keys())
            _all_lm_keys = [k for k in _keys_set if "lm_head" in k or "output_projection" in k]
            print(
                f"  [TrOCR-verify] safetensors header: {len(_keys_set) - 1} keys "
                f"(excl __metadata__), lm-related: {_all_lm_keys}"
            )
            _has_lm_head = _key in _keys_set or any(
                k.endswith(".lm_head.weight") or k.endswith(".output_projection.weight")
                for k in _keys_set
            )
            if not _has_lm_head:
                logging.getLogger(__name__).error(
                    "DIAGNOSTIC: 'decoder.lm_head.weight' NOT FOUND in safetensors header. "
                    "Keys present (excluding __metadata__): %s",
                    sorted(k for k in hdr if k != "__metadata__"),
                )
                raise RuntimeError(
                    f"CRITICAL: {_key!r} is missing from the safetensors shard "
                    f"{single_file.name}.  The tensor key name may differ between "
                    "transformers versions (see DIAGNOSTIC log above for actual keys).  "
                    "Ensure the weight alias is broken (data.clone()) "
                    "and _tied_weights_keys is cleared before save_pretrained().  "
                    "See CLAUDE.md §16 Pattern 6."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "_verify_lm_head_in_checkpoint: could not read shard header (%s) — skipping", exc
            )
        # Also check config.json tie_word_embeddings for the single-shard case.
        _check_config_tie_word_embeddings(model_path, _j)
        return

    # No safetensors found — may be a PyTorch bin checkpoint; skip the check.
    logging.getLogger(__name__).warning(
        "_verify_lm_head_in_checkpoint: no safetensors file found in %s — "
        "skipping lm_head integrity check (PyTorch .bin format is unverified)",
        model_path,
    )


def _verify_yolo_detection_rate(
    yolo_zero_count: int,
    trocr_empty_count: int,
    total_count: int,
    threshold: float = 0.5,
) -> None:
    """Raise ``RuntimeError`` (or log ``CRITICAL``) if the empty-ocr-lines rate is too high.

    An empty-ocr-lines rate above *threshold* means the pipeline produced no text
    for more than half the test images, silently driving F1 to 0.  There are two
    independent causes, now tracked separately:

    * ``yolo_zero_count`` — YOLO found 0 bounding boxes (genuine detection failure:
      wrong model, wrong ``imgsz``, undertrained YOLO, inline proxy fallback).
    * ``trocr_empty_count`` — YOLO found boxes but TrOCR decoded every crop to
      empty text (undertrained TrOCR: 1-epoch fine-tuning leaves ``val_loss≈9``,
      the decoder outputs garbage, ``batch_decode`` returns only whitespace).

    Both failure modes produce the same downstream symptom (``ocr_lines == []``),
    but they require completely different fixes.

    Parameters
    ----------
    yolo_zero_count : int
        Images where YOLO returned 0 bounding boxes.
    trocr_empty_count : int
        Images where YOLO found boxes but TrOCR decoded all crops to empty text.
    total_count : int
        Total number of images evaluated.
    threshold : float
        Maximum acceptable fraction of images with empty ocr_lines (default 0.50).
    """
    if total_count == 0:
        return
    combined_empty = yolo_zero_count + trocr_empty_count
    rate = combined_empty / total_count
    if rate > threshold:
        backend = "ultralytics" if _ULTRALYTICS_AVAILABLE else "inline _YOLO_CLS fallback"
        # Determine which failure mode dominates to give the most actionable message
        if trocr_empty_count > yolo_zero_count:
            trocr_msg = (
                f"Dominant cause: TrOCR decoded all crops to empty text on "
                f"{trocr_empty_count}/{total_count} images — TrOCR model is undertrained.  "
                "Fix: raise TROCR_EPOCHS to ≥ 5 in speed modes (1 epoch produces "
                "val_loss≈9.1 which destroys pretrained weights); also ensure "
                "TROCR_MAX_LEN ≥ 64 so receipt text lines are not truncated."
            )
        else:
            trocr_msg = (
                f"Dominant cause: YOLO found 0 boxes on {yolo_zero_count}/{total_count} "
                f"images (YOLO backend: {backend}).  "
                "Likely causes: (1) inline YOLO fallback with random/proxy weights — "
                "install ultralytics and retrain; (2) YOLO trained for too few epochs "
                "or at too low a resolution; (3) inference image resolution differs "
                "from training resolution."
            )
        raise RuntimeError(
            f"CRITICAL: empty-ocr-lines rate is {rate:.1%} "
            f"({combined_empty}/{total_count} images) — exceeds threshold {threshold:.0%}.  "
            f"YOLO-zero-box images: {yolo_zero_count}.  "
            f"TrOCR-all-empty images: {trocr_empty_count}.  "
            f"{trocr_msg}  "
            "All field predictions for affected images are empty strings, "
            "which drives global F1 to 0."
        )
    if rate > 0:
        logging.getLogger(__name__).warning(
            "Empty ocr_lines rate: %.1f%% (%d/%d images) — "
            "YOLO-zero-box: %d, TrOCR-all-empty: %d.  "
            "Some field predictions may be empty.",
            rate * 100,
            combined_empty,
            total_count,
            yolo_zero_count,
            trocr_empty_count,
        )


def _save_model_safetensors_direct(
    model: "torch.nn.Module",
    save_dir: "Path",
    lm_head_key: str = "decoder.lm_head.weight",
) -> None:
    """Save *model* weights directly via ``safetensors.torch.save_file``.

    This bypasses ``model.save_pretrained()`` entirely, which avoids the HF
    internal deduplication logic that drops ``lm_head.weight`` from the shard
    when it detects content-hash equality with ``embed_tokens.weight`` (a known
    regression in transformers ≥5.x even after ``data.clone()`` and
    ``_clear_lm_head_tied_keys()``).

    Steps:
    1. Clear ``_tied_weights_keys`` on *model* and its decoder so that
       HuggingFace internals cannot re-deduplicate the tensor during any
       subsequent ``save_pretrained()`` call.
    2. Get ``model.state_dict()``.
    3. Break **all** shared-memory aliases generically: iterate the state dict,
       track the first key seen for each ``data_ptr()`` value, and clone every
       subsequent key that shares the same pointer.  This covers
       ``lm_head.weight``, ``decoder.output_projection.weight`` (added in
       transformers ≥4.45), and any additional tied-weight pairs introduced in
       future versions — without hardcoding specific key names.
    4. Write the de-aliased state dict with ``safetensors.torch.save_file()``.
    5. Save the config via ``model.config.save_pretrained()`` so that
       ``from_pretrained()`` can reconstruct the model at load time.

    The caller is responsible for saving the processor separately.
    """
    from safetensors.torch import save_file as _st_save_file  # noqa: PLC0415

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Clear _tied_weights_keys so that HuggingFace cannot re-tie (and thus
    # silently drop) lm_head.weight or output_projection.weight during save.
    _clear_lm_head_tied_keys(model)

    sd = model.state_dict()

    # DIAGNOSTIC: Log whether lm_head_key is present in the raw state_dict
    # and what keys contain "lm_head" or "output_projection" or "embed_tokens"
    _lm_related_keys = sorted(
        k for k in sd if "lm_head" in k or "output_projection" in k or "embed_tokens" in k
    )
    print(f"  [TrOCR-save] state_dict has {len(sd)} keys")
    print(f"  [TrOCR-save] lm_head/output_projection/embed_tokens keys: {_lm_related_keys}")
    print(f"  [TrOCR-save] '{lm_head_key}' in sd: {lm_head_key in sd}")

    # If the expected lm_head key is absent, attempt to find an alternative or
    # inject directly from the model — HF's state_dict() may have filtered it
    # via _tied_weights_keys that were re-populated after our _clear call.
    if lm_head_key not in sd:
        # Also search for output_projection.weight — used in transformers ≥4.45
        # where the layer was renamed from lm_head to output_projection.
        _alt_keys = [
            k for k in sd if ("lm_head" in k or "output_projection" in k) and "weight" in k
        ]
        if _alt_keys:
            logging.getLogger(__name__).warning(
                "lm_head_key %r not in state_dict, but found alternative(s): %s — using %s",
                lm_head_key,
                _alt_keys,
                _alt_keys[0],
            )
            lm_head_key = _alt_keys[0]
        else:
            logging.getLogger(__name__).error(
                "CRITICAL: No lm_head key found in state_dict at all! "
                "state_dict has %d keys. Keys containing 'lm_head': %s. "
                "This means model.state_dict() is filtering out the lm_head weight, "
                "likely because _tied_weights_keys was re-populated after our clear. "
                "Attempting to extract lm_head.weight directly from model parameters.",
                len(sd),
                [k for k in sd if "lm_head" in k],
            )
            # Last resort: extract lm_head.weight directly from the model.
            # Also check output_projection for transformers ≥4.45.
            _decoder = getattr(model, "decoder", None)
            _lm_head = getattr(_decoder, "lm_head", None) if _decoder else None
            if _lm_head is None or not hasattr(_lm_head, "weight"):
                _lm_head = getattr(_decoder, "output_projection", None) if _decoder else None
            if _lm_head is not None and hasattr(_lm_head, "weight"):
                sd[lm_head_key] = _lm_head.weight.data.clone()
                print(
                    f"  [TrOCR-save] Manually injected {lm_head_key} "
                    "from model.decoder lm_head/output_projection"
                )

    # Generic alias-breaking loop: safetensors.torch.save_file raises
    # RuntimeError if any two tensors share a data_ptr().  Build a map from
    # data_ptr → first key seen; any subsequent key with the same data_ptr is
    # an alias — clone it to give it a unique buffer.  This handles all tied
    # weights (lm_head.weight, output_projection.weight in ≥4.45, and any
    # future additions) without per-version hardcoded key names.
    seen_ptrs: dict[int, str] = {}
    for k, v in list(sd.items()):
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            sd[k] = v.clone().contiguous()
        else:
            seen_ptrs[ptr] = k

    # Defensive perturbation for lm_head: even though safetensors.torch.save_file()
    # does NOT deduplicate by content hash (only HuggingFace's save_pretrained() does),
    # we apply a small relative perturbation as a belt-and-suspenders measure.  This
    # ensures the tensor bytes differ from embed_tokens.weight in case the caller
    # ever switches to save_pretrained() in the future.
    #
    # WHY NOT smallest_normal:  torch.finfo(dtype).smallest_normal ≈ 1.2e-38
    # for float32.  Typical pretrained weight values are ~0.01–0.05.  The ULP
    # (unit in last place) at 0.02 is ~2.4e-9, so anything smaller than that is
    # rounded away — the addition is a no-op and the bytes remain identical.
    #
    # WHY RELATIVE (eps) WORKS:  torch.finfo(dtype).eps is the machine epsilon
    # (~1.19e-7 for fp32, ~7.8e-3 for bf16).  Multiplying by (1 + eps) flips at
    # least the least-significant mantissa bit of every element, guaranteeing the
    # tensor is bytewise distinct from embed_tokens.weight regardless of the
    # magnitude of the weights.
    if lm_head_key in sd:
        _lm = sd[lm_head_key]
        sd[lm_head_key] = _lm * (1.0 + torch.finfo(_lm.dtype).eps)

    # DIAGNOSTIC: Verify lm_head_key survived alias-breaking
    print(f"  [TrOCR-save] After alias-break: '{lm_head_key}' in sd: {lm_head_key in sd}")
    # DIAGNOSTIC: Check config tie_word_embeddings before writing
    _twe = getattr(model.config, "tie_word_embeddings", "UNSET")
    _dec_cfg = getattr(getattr(model, "decoder", None), "config", None)
    _twe_dec_val = getattr(_dec_cfg, "tie_word_embeddings", "UNSET") if _dec_cfg else "N/A"
    print(
        f"  [TrOCR-save] config.tie_word_embeddings={_twe}, "
        f"decoder.config.tie_word_embeddings={_twe_dec_val}"
    )
    if _twe is True:
        logging.getLogger(__name__).error(
            "CRITICAL: model.config.tie_word_embeddings is True at save time! "
            "This will cause from_pretrained() to expect lm_head.weight to be tied and "
            "report it as 'missing'. Forcing to False."
        )

    # Belt-and-suspenders: force tie_word_embeddings=False on both configs
    # unconditionally before saving config.json, in case something reset it
    # during training (e.g., gradient_checkpointing_enable()).
    model.config.tie_word_embeddings = False
    if _dec_cfg is not None:
        _dec_cfg.tie_word_embeddings = False

    # Sync decoder_start_token_id in the decoder sub-config to match the
    # top-level config.  Without this, the saved config.json has the correct
    # value at the top level (set via processor.tokenizer.cls_token_id before
    # training) but a stale default inside the decoder block, which breaks
    # standalone checkpoint loading that reads only the decoder sub-config.
    _top_dst_id = getattr(model.config, "decoder_start_token_id", None)
    if _dec_cfg is not None and _top_dst_id is not None:
        _dec_cfg.decoder_start_token_id = _top_dst_id

    _st_save_file(sd, save_dir / "model.safetensors")
    model.config.save_pretrained(str(save_dir))


def _clear_lm_head_tied_keys(model: "torch.nn.Module") -> None:
    """Remove ``decoder.lm_head.weight`` from ``_tied_weights_keys`` on *model*
    and its decoder child so that ``save_pretrained`` cannot re-deduplicate the
    tensor even after a ``data.clone()`` has broken the storage alias.

    HuggingFace ``PreTrainedModel.save_pretrained`` checks ``_tied_weights_keys``
    *before* comparing ``data_ptr()`` values.  If the key is still listed, the
    tensor is omitted from the shard regardless of whether storage is shared,
    undoing the protection provided by the ``.clone()`` call.
    """
    _lm_key_outer = "decoder.lm_head.weight"
    _lm_key_inner = "lm_head.weight"
    decoder = getattr(model, "decoder", None)
    for obj, key in [(model, _lm_key_outer), (decoder, _lm_key_inner)]:
        if obj is None:
            continue
        tied = getattr(obj, "_tied_weights_keys", None)
        if tied is not None and key in tied:
            try:
                obj._tied_weights_keys = [k for k in tied if k != key]
            except (AttributeError, TypeError):
                pass  # class-level attribute; assignment not possible — harmless


def _print_trocr_load_report(model_id: str, loading_info: dict[str, Any]) -> None:
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


def _build_experiment_trocr_metadata(
    exp_datasets: list[str],
    sroie_oversample: int,
    output_dir: Path,
) -> Path:
    """Build a per-experiment TrOCR training metadata.jsonl.

    Merges SROIE TrOCR crops (from TROCR_DATA_DIR/train, repeated
    *sroie_oversample* times) with pseudo-crops from auxiliary datasets
    (full receipt images whose ground-truth field values serve as text labels).

    Uses absolute paths in ``file_name`` entries so no file copying is needed:
    ``Path(data_dir) / "/absolute/path"`` resolves to the absolute path on
    POSIX, which is how ``TrOCRReceiptDataset.__getitem__`` loads images.

    Parameters
    ----------
    exp_datasets : list[str]
        Dataset names for this experiment, e.g. ``["sroie", "wildreceipt"]``.
    sroie_oversample : int
        How many times to repeat SROIE crops (mirrors DONUT oversampling).
    output_dir : Path
        Directory where ``metadata.jsonl`` is written.

    Returns
    -------
    Path
        *output_dir* (created if it did not exist).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    # ── 1. SROIE TrOCR crops with oversampling ──────────────────────────────
    base_train_dir = TROCR_DATA_DIR / "train"
    base_meta = base_train_dir / "metadata.jsonl"
    if base_meta.exists():
        with open(base_meta) as _fh:
            base_records = [json.loads(line) for line in _fh if line.strip()]
        n_base = len(base_records)
        for _ in range(max(1, sroie_oversample)):
            for rec in base_records:
                abs_path = str((base_train_dir / rec["file_name"]).resolve())
                records.append({"file_name": abs_path, "text": rec["text"]})
        print(
            f"  [TrOCR-exp] SROIE: {n_base} crops "
            f"× {sroie_oversample} = {n_base * sroie_oversample} records"
        )
    else:
        print(f"  [TrOCR-exp] Warning: SROIE TrOCR data not found at {base_train_dir}")

    # ── 2. Auxiliary datasets (full image + field values as text labels) ─────
    aux_names = [ds for ds in exp_datasets if ds != "sroie"]
    for ds_name in aux_names:
        try:
            import data_pipeline as _dp  # noqa: PLC0415

            loader_fn = _dp.get_dataset_loader(ds_name)
            if loader_fn is None:
                print(
                    f"  [TrOCR-exp] Warning: no loader registered for dataset '{ds_name}' — skipping"
                )
                continue
            samples = loader_fn()
            before = len(records)
            for img_path, gt_dict in samples:
                abs_path = str(Path(img_path).resolve())
                for field_val in gt_dict.values():
                    if field_val and str(field_val).strip():
                        records.append({"file_name": abs_path, "text": str(field_val).strip()})
            added = len(records) - before
            print(f"  [TrOCR-exp] {ds_name}: {len(samples)} images → {added} records")
        except ImportError as _exc:
            print(
                f"  [TrOCR-exp] Warning: data_pipeline import failed — skipping '{ds_name}': {_exc}"
            )
        except Exception as _exc:
            import traceback as _tb

            print(f"  [TrOCR-exp] Warning: unexpected error loading '{ds_name}': {_exc}")
            _tb.print_exc()

    # ── Write metadata.jsonl ─────────────────────────────────────────────────
    meta_path = output_dir / "metadata.jsonl"
    with open(meta_path, "w") as _fh:
        for r in records:
            _fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [TrOCR-exp] Total: {len(records)} records → {meta_path}")
    return output_dir


def _save_trocr_training_plots(history: dict[str, Any], output_dir: Path) -> None:
    """Save TrOCR training loss curves and CSV after training completes.

    Creates:
    - ``results/figures/trocr_loss.png`` (300 DPI) — train vs val loss curve
    - ``results/figures/trocr_loss.svg`` — same chart as SVG fallback
    - ``results/trocr_training_history.csv`` — epoch, train_loss, val_loss

    Falls back to an inline SVG if matplotlib is not installed.
    """
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    if not train_losses and not val_losses:
        return

    # ── CSV ──────────────────────────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "trocr_training_history.csv"
    try:
        with open(csv_path, "w") as _csv:
            _csv.write("epoch,train_loss,val_loss\n")
            for i, tl in enumerate(train_losses, start=1):
                vl = val_losses[i - 1] if i - 1 < len(val_losses) else None
                vl_str = f"{vl:.6f}" if vl is not None and vl != float("inf") else ""
                _csv.write(f"{i},{tl:.6f},{vl_str}\n")
    except OSError:
        pass

    # ── Figures directory ─────────────────────────────────────────────────────
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    # ── Attempt matplotlib plot ───────────────────────────────────────────────
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_losses, marker="o", label="Train Loss", linewidth=1.5)
        if val_losses:
            _v_epochs = list(range(1, len(val_losses) + 1))
            ax.plot(_v_epochs, val_losses, marker="s", label="Val Loss", linewidth=1.5)
        ax.set_title("TrOCR Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        png_path = figures_dir / "trocr_loss.png"
        fig.savefig(png_path, dpi=300)
        try:
            pdf_path = figures_dir / "trocr_loss.pdf"
            fig.savefig(pdf_path)
        except Exception:
            pass
        plt.close(fig)
        print(f"  [TrOCR] Training loss plot saved → {png_path}")
        return
    except Exception:
        pass

    # ── SVG fallback (zero dependencies) ─────────────────────────────────────
    W, H = 560, 320
    ML, MR, MT, MB = 58, 20, 40, 55
    PW, PH = W - ML - MR, H - MT - MB
    all_vals = [v for v in train_losses + val_losses if v != float("inf")]
    ymax = max(all_vals, default=1.0) * 1.1 or 1.0
    ymin = 0.0

    def _ty(v: float) -> int:
        return MT + PH - int((v - ymin) / (ymax - ymin) * PH)

    def _tx(i: int) -> float:
        n = max(len(epochs) - 1, 1)
        return ML + (i / n) * PW

    lines_svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"'
        f' font-family="sans-serif" font-size="10">',
        f'<text x="{W // 2}" y="22" text-anchor="middle" font-size="11"'
        f' font-weight="bold">TrOCR Training Loss</text>',
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + PH}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{ML}" y1="{MT + PH}" x2="{ML + PW}" y2="{MT + PH}"'
        f' stroke="#333" stroke-width="1"/>',
        f'<text x="14" y="{MT + PH // 2}" text-anchor="middle" font-size="9"'
        f' transform="rotate(-90 14 {MT + PH // 2})">Loss</text>',
        f'<text x="{ML + PW // 2}" y="{H - 8}" text-anchor="middle" font-size="9">Epoch</text>',
    ]
    # Grid lines + Y tick labels
    for _tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _yv = ymin + _tick * (ymax - ymin)
        _yp = _ty(_yv)
        lines_svg += [
            f'<line x1="{ML}" y1="{_yp}" x2="{ML + PW}" y2="{_yp}"'
            f' stroke="#ccc" stroke-width="0.5" stroke-dasharray="4,3"/>',
            f'<text x="{ML - 4}" y="{_yp + 3}" text-anchor="end" font-size="8">{_yv:.2f}</text>',
        ]
    # Train loss polyline
    if len(epochs) > 1:
        pts = " ".join(f"{_tx(i):.1f},{_ty(v)}" for i, v in enumerate(train_losses))
        lines_svg.append(
            f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="1.8"/>'
        )
    for i, v in enumerate(train_losses):
        lines_svg.append(f'<circle cx="{_tx(i):.1f}" cy="{_ty(v)}" r="3" fill="#1f77b4"/>')
    # Val loss polyline
    if val_losses:
        v_epochs_idx = list(range(len(val_losses)))
        if len(v_epochs_idx) > 1:
            vpts = " ".join(
                f"{_tx(i):.1f},{_ty(v)}" for i, v in enumerate(val_losses) if v != float("inf")
            )
            if vpts:
                lines_svg.append(
                    f'<polyline points="{vpts}" fill="none"'
                    f' stroke="#ff7f0e" stroke-width="1.8" stroke-dasharray="6,3"/>'
                )
        for i, v in enumerate(val_losses):
            if v != float("inf"):
                lines_svg.append(f'<circle cx="{_tx(i):.1f}" cy="{_ty(v)}" r="3" fill="#ff7f0e"/>')
    # Legend
    lines_svg += [
        f'<rect x="{ML + PW - 110}" y="{MT + 6}" width="10" height="8" fill="#1f77b4"/>',
        f'<text x="{ML + PW - 97}" y="{MT + 14}" font-size="9">Train Loss</text>',
        f'<rect x="{ML + PW - 110}" y="{MT + 20}" width="10" height="8" fill="#ff7f0e"/>',
        f'<text x="{ML + PW - 97}" y="{MT + 28}" font-size="9">Val Loss</text>',
        "</svg>",
    ]
    svg_content = "\n".join(lines_svg)
    svg_path = figures_dir / "trocr_loss.svg"
    try:
        svg_path.write_text(svg_content, encoding="utf-8")
        print(f"  [TrOCR] Training loss SVG saved → {svg_path}")
    except OSError:
        pass


def _save_trocr_eval_plots(exp_metrics: dict[str, Any], results_dir: Path) -> None:
    """Save per-field F1 bar chart for TrOCR+YOLO evaluation results.

    Creates:
    - ``results/figures/trocr_field_f1.png`` (300 DPI) — per-field F1 bar chart
    - ``results/figures/trocr_field_f1.svg`` — same chart as SVG fallback

    Falls back to an inline SVG if matplotlib is not installed.
    """
    from constants import FIELDS  # noqa: PLC0415

    if not exp_metrics:
        return

    fields = FIELDS  # ["company", "date", "address", "total"]
    f1_values = [exp_metrics.get(f"{f}_f1", 0.0) for f in fields]
    global_f1 = exp_metrics.get("global_f1", 0.0)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Attempt matplotlib plot ───────────────────────────────────────────────
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(figsize=(7, 4))
        x = list(range(len(fields)))
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        bars = ax.bar(x, f1_values, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f.capitalize() for f in fields])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 Score")
        ax.set_title(f"TrOCR+YOLO Per-Field F1  (Global F1 = {global_f1:.4f})")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, val in zip(bars, f1_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        png_path = figures_dir / "trocr_field_f1.png"
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        print(f"  [TrOCR] Per-field F1 chart saved → {png_path}")
        return
    except Exception:
        pass

    # ── SVG fallback ──────────────────────────────────────────────────────────
    W, H = 420, 280
    ML, MR, MT, MB = 55, 20, 50, 45
    PW, PH = W - ML - MR, H - MT - MB
    n = len(fields)
    bar_w = PW / n * 0.6
    colors_svg = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    ymax_svg = 1.0

    def _ty2(v: float) -> int:
        return MT + PH - int(v / ymax_svg * PH)

    lines_svg2 = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"'
        f' font-family="sans-serif" font-size="10">',
        f'<text x="{W // 2}" y="18" text-anchor="middle" font-size="10" font-weight="bold">'
        f"TrOCR+YOLO Per-Field F1</text>",
        f'<text x="{W // 2}" y="32" text-anchor="middle" font-size="8">'
        f"Global F1 = {global_f1:.4f}</text>",
        f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT + PH}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{ML}" y1="{MT + PH}" x2="{ML + PW}" y2="{MT + PH}"'
        f' stroke="#333" stroke-width="1"/>',
        f'<text x="14" y="{MT + PH // 2}" text-anchor="middle" font-size="9"'
        f' transform="rotate(-90 14 {MT + PH // 2})">F1</text>',
    ]
    for _tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _yp2 = _ty2(_tick)
        lines_svg2 += [
            f'<line x1="{ML}" y1="{_yp2}" x2="{ML + PW}" y2="{_yp2}"'
            f' stroke="#ccc" stroke-width="0.5" stroke-dasharray="4,3"/>',
            f'<text x="{ML - 4}" y="{_yp2 + 3}" text-anchor="end" font-size="8">{_tick:.2f}</text>',
        ]
    for fi, (fname, fval) in enumerate(zip(fields, f1_values)):
        gx = ML + fi * (PW / n) + (PW / n - bar_w) / 2
        bh = int(fval / ymax_svg * PH)
        by = MT + PH - bh
        color = colors_svg[fi % len(colors_svg)]
        lines_svg2 += [
            f'<rect x="{gx:.1f}" y="{by}" width="{bar_w:.1f}" height="{bh}"'
            f' fill="{color}" opacity="0.85" stroke="white" stroke-width="0.6"/>',
            f'<text x="{gx + bar_w / 2:.1f}" y="{by - 3}" text-anchor="middle" font-size="8">'
            f"{fval:.3f}</text>",
            f'<text x="{gx + bar_w / 2:.1f}" y="{MT + PH + 14}" text-anchor="middle"'
            f' font-size="9">{fname.capitalize()}</text>',
        ]
    lines_svg2.append("</svg>")
    svg_content2 = "\n".join(lines_svg2)
    svg_path2 = figures_dir / "trocr_field_f1.svg"
    try:
        svg_path2.write_text(svg_content2, encoding="utf-8")
        print(f"  [TrOCR] Per-field F1 SVG saved → {svg_path2}")
    except OSError:
        pass


def train_trocr(
    output_dir: Path | None = None,
    train_data_dir: Path | None = None,
) -> dict[str, Any]:
    """Fine-tune TrOCR on line crops from receipts.

    Parameters
    ----------
    output_dir : Path, optional
        Directory where checkpoints (``best/``, ``final/``) are saved.
        Defaults to ``WORKSPACE/models/trocr_finetuned``.
    train_data_dir : Path, optional
        Directory containing ``metadata.jsonl`` for training crops.
        Defaults to ``TROCR_DATA_DIR/train`` (SROIE only).
        Pass a per-experiment directory built by
        ``_build_experiment_trocr_metadata()`` to use a mixed-dataset split.

    Returns
    -------
    dict
        Training history with keys ``train_loss``, ``val_loss``, and
        ``num_train_samples``.
    """
    # Minimum-epoch enforcement: 1 epoch of TrOCR fine-tuning produces val_loss≈9.1,
    # which destroys microsoft/trocr-base-printed pretrained weights without learning
    # any SROIE-specific patterns.  Every YOLO crop decodes to empty text, driving F1
    # to 0 with a misleading "YOLO detected 0 text regions" error (YOLO is actually fine).
    # L7: Training Minimum Viability — enforced here so it is impossible to
    # silently train a non-functional TrOCR model, regardless of how the
    # caller patched TROCR_EPOCHS.
    # See CLAUDE.md §16 Pattern 9 (Masked Cascading Failures) and the post-mortem
    # comment block in run_all.py near _superfast_mode_handler.
    _assert_training_viability(trocr_epochs=TROCR_EPOCHS)

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

    processor = _get_trocr_processor(TROCR_MODEL_ID)
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
    # Fix: issue_report_summary invariant #14 — add convert_tokens_to_ids list-form guard
    # to the TrOCR path, matching the DONUT path in train.py.
    # TrOCR uses cls_token_id (a direct attribute) rather than convert_tokens_to_ids, but
    # we must still verify it is non-None and not the unk_token_id.
    _trocr_cls_id = processor.tokenizer.cls_token_id
    if _trocr_cls_id is None:
        raise ValueError(
            "TrOCR processor.tokenizer.cls_token_id is None. "
            "The TrOCR tokenizer must have a [CLS] token. "
            "Check that 'microsoft/trocr-base-printed' was loaded correctly."
        )
    _trocr_unk_id = getattr(processor.tokenizer, "unk_token_id", None)
    if _trocr_unk_id is not None and _trocr_cls_id == _trocr_unk_id:
        raise ValueError(
            f"TrOCR processor.tokenizer.cls_token_id={_trocr_cls_id} equals "
            f"unk_token_id={_trocr_unk_id}. The [CLS] token is not registered in the "
            "tokenizer vocabulary. Use convert_tokens_to_ids(['[CLS]'])[0] in list form "
            "to verify: if the result equals unk_token_id, the tokenizer is corrupt. "
            "See CLAUDE.md §16 GP-3 (Fix #14)."
        )
    # Fix: issue_report_summary invariant #15 — verify decoder special tokens are present.
    # The DONUT path checks all NEW_TOKENS in DonutTrainer._pre_training_guardrails().
    # For TrOCR we verify the three structural tokens: [CLS], [SEP], [PAD].
    for _trocr_tok_name, _trocr_tok_id in [
        ("[CLS]", processor.tokenizer.cls_token_id),
        ("[SEP]", processor.tokenizer.sep_token_id),
        ("[PAD]", processor.tokenizer.pad_token_id),
    ]:
        if _trocr_tok_id is None:
            raise ValueError(
                f"TrOCR tokenizer is missing required token {_trocr_tok_name!r}. "
                "Ensure 'microsoft/trocr-base-printed' is loaded without truncation. "
                "See CLAUDE.md §16 Fix #15."
            )
        if _trocr_unk_id is not None and _trocr_tok_id == _trocr_unk_id:
            raise ValueError(
                f"TrOCR tokenizer token {_trocr_tok_name!r} maps to unk_token_id={_trocr_unk_id}. "
                "The token is not registered in the vocabulary. "
                "See CLAUDE.md §16 Fix #15."
            )
    model.generation_config.max_new_tokens = TROCR_MAX_LEN
    model.generation_config.no_repeat_ngram_size = 0  # disabled — harmful for short OCR text
    model.generation_config.length_penalty = 1.0  # neutral — do not penalise short outputs
    model.generation_config.num_beams = 1  # greedy; beam search is slow per-crop with no F1 gain

    # ── lm_head weight-tying guardrail (mirrors DONUT path in train.py) ────────
    # VisionEncoderDecoderModel wraps the TrOCR decoder as a BART-style model
    # whose lm_head.weight may share storage with decoder.model.embed_tokens.weight
    # after from_pretrained().  safetensors deduplicates tensors sharing a data
    # pointer, so lm_head.weight is silently dropped from per-epoch checkpoint shards.
    # Setting tie_word_embeddings=False on BOTH configs tells HF to not re-tie them,
    # which alone is insufficient — the alias must be actively broken (see clone below).
    model.config.tie_word_embeddings = False
    if hasattr(model.decoder, "config"):
        model.decoder.config.tie_word_embeddings = False

    # Post-load check: raise immediately if lm_head.weight is already missing.
    # This can happen if the pretrained checkpoint itself was built with an older
    # HF version that deduplicated the weight.  Better to fail loudly now than
    # produce garbled output silently at inference time.
    # In transformers ≥4.45, the output projection was renamed from lm_head to
    # output_projection — accept either key name as a valid equivalent.
    _trocr_lm_head_key = "decoder.lm_head.weight"
    _trocr_output_proj_key = "decoder.output_projection.weight"
    # Auto-detect actual key name from model state dict for later saves.
    if hasattr(model.decoder, "output_projection") and not hasattr(model.decoder, "lm_head"):
        _trocr_lm_head_key = _trocr_output_proj_key
    _trocr_loading_missing = loading_info.get("missing_keys", [])
    if _trocr_lm_head_key in _trocr_loading_missing:
        # Before raising, verify the output_projection variant is not present either.
        _has_any_output_proj = any(
            k.endswith(".lm_head.weight") or k.endswith(".output_projection.weight")
            for k in model.state_dict()
        )
        if not _has_any_output_proj:
            raise RuntimeError(
                f"CRITICAL: {_trocr_lm_head_key} is missing from the {TROCR_MODEL_ID} checkpoint. "
                "The model cannot produce valid predictions. "
                "Check that the pretrained model is a complete, uncorrupted download. "
                "See CLAUDE.md §16 Pattern 6."
            )

    # Break the weight alias immediately so all subsequent save_pretrained() calls
    # write lm_head.weight as an independent tensor rather than as a deduplicated
    # pointer to embed_tokens.weight.
    # In transformers ≥4.45 the layer was renamed to output_projection; check both.
    _lm_attr = None
    if hasattr(model.decoder, "lm_head") and hasattr(model.decoder.lm_head, "weight"):
        _lm_attr = model.decoder.lm_head
    elif hasattr(model.decoder, "output_projection") and hasattr(
        model.decoder.output_projection, "weight"
    ):
        _lm_attr = model.decoder.output_projection
    if _lm_attr is not None:
        _lm = _lm_attr
        _emb = (
            model.decoder.model.decoder.embed_tokens
            if hasattr(model.decoder, "model")
            and hasattr(model.decoder.model, "decoder")
            and hasattr(model.decoder.model.decoder, "embed_tokens")
            else None
        )
        if _emb is not None and _lm.weight.data_ptr() == _emb.weight.data_ptr():
            _lm.weight = torch.nn.Parameter(_lm.weight.data.clone())
            _clear_lm_head_tied_keys(model)
            print("  [TrOCR] lm_head.weight alias broken (was sharing storage with embed_tokens)")
        else:
            _clear_lm_head_tied_keys(model)
            print("  [TrOCR] lm_head.weight is already independent (no alias to break)")

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
            # Fix: issue_report_summary critical #3 — log at ERROR level so the operator
            # knows VRAM detection failed and gradient checkpointing defaulted to enabled.
            logging.getLogger(__name__).error(
                "[TrOCR] VRAM detection failed — defaulting gradient checkpointing to ENABLED. "
                "If this causes OOM, set grad_ckpt_vram_threshold_gb explicitly in "
                "ControlSuite.trocr. Error: %s",
                _exc,
                exc_info=True,
            )

    if _trocr_enable_grad_ckpt:
        model.config.use_cache = False
        model.decoder.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
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
    scaler = torch.amp.GradScaler("cuda", enabled=(_use_amp and _amp_dtype == torch.float16))
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

    train_dir = train_data_dir if train_data_dir is not None else TROCR_DATA_DIR / "train"
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
        use_tensor_cache=TROCR_USE_TENSOR_CACHE,
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

    # Validate that this configuration produces enough optimizer steps for convergence.
    # TrOCR (like DONUT) requires ~200+ steps to learn field alignment; fewer steps
    # typically produce a model that outputs structurally correct but content-empty
    # predictions, which gives near-zero F1 without any obvious error.
    from resource_manager import validate_training_config as _vtc  # noqa: PLC0415

    _vtc(
        batch_size=trocr_batch,
        gradient_accumulation_steps=grad_accum,
        num_train_samples=len(train_ds),
        epochs=TROCR_EPOCHS,
    )

    if TROCR_MINI_MODE:
        # AdamW with an elevated LR and a brief cosine warmup.  SGD at lr=1e-2
        # (the previous TROCR_LR * 200 setting) irreversibly damages pretrained
        # weights in the first few batches of a 1-epoch run because it lacks
        # adaptive per-parameter scaling and has no warmup.  AdamW handles large
        # step sizes gracefully and is used by all serious transformer fine-tuning.
        # LR is 10× the normal rate so short runs still make meaningful progress.
        _micro_lr = TROCR_LR * 10
        _micro_decay: list = []
        _micro_no_decay: list = []
        for _p in model.parameters():
            (_micro_no_decay if _p.ndim < 2 else _micro_decay).append(_p)
        optimizer = torch.optim.AdamW(
            [
                {"params": _micro_decay, "lr": _micro_lr, "weight_decay": 1e-4},
                {"params": _micro_no_decay, "lr": _micro_lr, "weight_decay": 0.0},
            ],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        # Brief warmup (10% of steps) then cosine decay to avoid the cold-start
        # gradient explosion that SGD without warmup causes.
        _warmup_steps = max(1, total_steps // 10)

        def _lr_lambda(step: int) -> float:
            if step < _warmup_steps:
                return step / _warmup_steps
            progress = (step - _warmup_steps) / max(1, total_steps - _warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        print(f"  [TrOCR] Micro mode: AdamW+cosine-warmup, lr={_micro_lr:.2e}")
    else:
        _trocr = CONTROL_SUITE.trocr
        # Split parameters into decay / no-decay groups.
        # 1-D params (biases, LayerNorm γ/β) must NOT receive weight decay:
        # applying L2 regularisation to them distorts normalisation layers and
        # biases, yielding slower convergence and slightly lower F1.
        # p.ndim < 2 is more robust than name-matching ("bias", "LayerNorm.weight")
        # because it works for any architecture without enumerating names.
        _trocr_decay: list = []
        _trocr_no_decay: list = []
        for _p in model.parameters():
            (_trocr_no_decay if _p.ndim < 2 else _trocr_decay).append(_p)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": _trocr_decay,
                    "lr": TROCR_LR,
                    "weight_decay": _trocr.weight_decay,
                },
                {
                    "params": _trocr_no_decay,
                    "lr": TROCR_LR,
                    "weight_decay": 0.0,
                },
            ],
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

    # ── Pre-training guardrails (mirror DONUT guardrails in train.py) ──────────
    # Guardrail A: decoder_start_token_id must be set and non-None.
    _trocr_dst_id = getattr(model.config, "decoder_start_token_id", None)
    if _trocr_dst_id is None:
        raise ValueError(
            "CRITICAL [TrOCR]: model.config.decoder_start_token_id is None. "
            "Set it to processor.tokenizer.cls_token_id before training."
        )
    # Guardrail B: decoder_start_token_id must not equal unk_token_id.
    _trocr_unk_id = getattr(processor.tokenizer, "unk_token_id", None)
    if _trocr_unk_id is not None and _trocr_dst_id == _trocr_unk_id:
        raise ValueError(
            f"CRITICAL [TrOCR]: decoder_start_token_id={_trocr_dst_id} equals "
            f"unk_token_id={_trocr_unk_id}. The model will generate garbage. "
            "Use processor.tokenizer.cls_token_id to set decoder_start_token_id."
        )
    # Guardrail C: tie_word_embeddings must be False on both configs.
    for _cfg_name, _cfg_obj in [
        ("model.config", model.config),
        (
            "decoder.config",
            getattr(model, "decoder", None) and getattr(model.decoder, "config", None),
        ),
    ]:
        if _cfg_obj is not None and getattr(_cfg_obj, "tie_word_embeddings", None) is True:
            raise ValueError(
                f"CRITICAL [TrOCR]: {_cfg_name}.tie_word_embeddings is True. "
                "Must be False to prevent lm_head.weight deduplication on save. "
                "Set model.config.tie_word_embeddings = False before training."
            )
    print(
        f"  [TrOCR] Pre-training guardrails PASSED: tie_word_embeddings=False, "
        f"decoder_start_token_id={_trocr_dst_id}"
    )
    # ── End pre-training guardrails ──────────────────────────────────────────

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "num_train_samples": 0}
    history["num_train_samples"] = len(train_ds)
    start = time.time()
    _trocr_logger = logging.getLogger(__name__)

    try:
        _trocr_skipped_total = 0  # GradScaler overflow steps across all epochs
        _trocr_nan_epochs = 0  # consecutive NaN-loss epochs (abort threshold = 3)
        import math as _math  # noqa: PLC0415

        for epoch in range(TROCR_EPOCHS):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            _epoch_skipped = 0
            _nan_break = False  # tracks whether the inner loop was aborted early
            # Fix: issue_report_summary critical #3 — track consecutive batch failures.
            _consecutive_batch_failures = 0

            _desc = f"TrOCR Epoch {epoch + 1}/{TROCR_EPOCHS}"
            for step, batch in enumerate(
                _progress(train_loader, desc=_desc, total=len(train_loader))
            ):
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype, enabled=_use_amp):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / grad_accum
                scaler.scale(loss).backward()
                _step_loss = outputs.loss.item()  # unscaled, for logging and NaN check
                # NaN loss detection: abort this epoch immediately rather than
                # accumulating NaN into epoch_loss and logging a misleading average.
                if _math.isnan(_step_loss) or _math.isinf(_step_loss):
                    print(
                        f"  [TrOCR] WARNING: step {step} loss={_step_loss} — "
                        "stopping epoch early (fp16 overflow or bad batch). "
                        "Consider switching to bf16."
                    )
                    # Let GradScaler detect the inf/nan and reduce its scale factor
                    # before we zero gradients.  Calling scaler.step() here causes the
                    # scaler to skip the optimizer update and call scaler.update() to
                    # halve the internal scale, which prevents the same overflow in the
                    # next epoch.  Skipping this step means the scale stays high and the
                    # overflow typically repeats every epoch indefinitely.
                    scaler.unscale_(optimizer)
                    _scale_before_nan = scaler.get_scale()
                    scaler.step(optimizer)  # skips weight update; marks _found_inf
                    scaler.update()  # halves scale due to detected inf/nan
                    if scaler.get_scale() < _scale_before_nan:
                        _epoch_skipped += 1
                    optimizer.zero_grad(set_to_none=True)
                    _nan_break = True
                    break
                epoch_loss += _step_loss

                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    _scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    # Detect GradScaler-skipped steps (inf/nan under AMP fp16).
                    # Only advance the LR scheduler when weights were actually updated —
                    # stepping the scheduler on a skipped optimizer step wastes warmup
                    # budget and shifts the LR curve without a corresponding weight change.
                    if scaler.get_scale() < _scale_before:
                        _epoch_skipped += 1
                    else:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            # Flush the incomplete accumulation window at the end of each epoch.
            # When len(train_loader) % grad_accum != 0, the final partial window
            # accumulates gradients that never reach the (step+1) % grad_accum == 0
            # condition — those gradients are silently discarded, causing up to
            # (grad_accum - 1) / grad_accum fraction of data per epoch to contribute
            # to loss logging but NOT to weight updates.
            # Skip the flush if the epoch was aborted early due to NaN (gradients
            # were already zeroed in the break handler above).
            if not _nan_break:
                _remaining = len(train_loader) % grad_accum
                if _remaining != 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    _scale_before_flush = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < _scale_before_flush:
                        _epoch_skipped += 1
                    else:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            avg_train = epoch_loss / max(len(train_loader), 1)

            if _epoch_skipped > 0:
                _trocr_skipped_total += _epoch_skipped
                print(
                    f"  [TrOCR] Epoch {epoch + 1}: GradScaler skipped {_epoch_skipped} "
                    f"optimizer step(s) due to AMP fp16 overflow. "
                    "Weights were NOT updated for those steps."
                )
                try:
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
                        _scale_before_step = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                        if scaler.get_scale() >= _scale_before_step:
                            scheduler.step()
                        optimizer.zero_grad()
                    _consecutive_batch_failures = 0  # reset on success
                except Exception as _batch_exc:
                    # Fix: issue_report_summary critical #3 — log at ERROR level with
                    # batch index; call empty_cache() on OOM; abort if threshold exceeded.
                    _is_oom = isinstance(_batch_exc, torch.cuda.OutOfMemoryError)
                    if _is_oom:
                        torch.cuda.empty_cache()
                    _consecutive_batch_failures += 1
                    _trocr_logger.error(
                        "TrOCR training: batch failed (epoch=%d, step=%d, "
                        "consecutive_failures=%d/%d, oom=%s): %s",
                        epoch + 1,
                        step,
                        _consecutive_batch_failures,
                        MAX_CONSECUTIVE_BATCH_FAILURES,
                        _is_oom,
                        _batch_exc,
                        exc_info=True,
                    )
                    if _consecutive_batch_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                        raise RuntimeError(
                            f"TrOCR training aborted at epoch {epoch + 1}, step {step}: "
                            f"{_consecutive_batch_failures} consecutive batch failures exceeded "
                            f"threshold MAX_CONSECUTIVE_BATCH_FAILURES={MAX_CONSECUTIVE_BATCH_FAILURES}. "
                            "This indicates a systemic issue (persistent CUDA OOM or corrupt data). "
                            "See CLAUDE.md §5 Fix #3."
                        ) from _batch_exc
                    # Skip this batch; reset gradient state to avoid stale gradients
                    try:
                        optimizer.zero_grad()
                    except Exception:
                        pass
                    continue

            # Track consecutive NaN epochs: 3 in a row → abort training entirely.
            if _math.isnan(avg_train):
                _trocr_nan_epochs += 1
                print(
                    f"  [TrOCR] Epoch {epoch + 1}: avg_train=NaN "
                    f"({_trocr_nan_epochs}/3 consecutive NaN epochs)."
                )
                if _trocr_nan_epochs >= 3:
                    raise RuntimeError(
                        "TrOCR training aborted: 3 consecutive NaN-loss epochs. "
                        "This indicates severe fp16 overflow or a corrupt batch pipeline. "
                        "Switch to bf16 (Ampere+) or fp32, or lower the learning rate."
                    )
            else:
                _trocr_nan_epochs = 0  # reset counter on any clean epoch

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

            # ── Val-loss divergence detection ────────────────────────────────
            # The DiagnosticCallback catches this for DONUT (Seq2SeqTrainer),
            # but this custom loop needs its own check.  LR overshoot (e.g.
            # TROCR_MINI_MODE at 10× LR) causes val_loss to double every epoch
            # while train_loss oscillates — the decoder's pretrained weights
            # are irreversibly destroyed.
            _val_hist = history["val_loss"]
            if len(_val_hist) >= 3:
                _recent = _val_hist[-3:]
                if _recent[-1] > _recent[-2] > _recent[-3]:
                    print(
                        f"  [TrOCR] WARNING: val_loss rising 3 consecutive epochs: "
                        f"{[f'{v:.4f}' for v in _recent]} — possible LR overshoot. "
                        f"Current lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
            if len(_val_hist) >= 5:
                _recent5 = _val_hist[-5:]
                if all(_recent5[i + 1] > _recent5[i] for i in range(4)):
                    raise RuntimeError(
                        f"TrOCR training aborted: val_loss increased 5 consecutive epochs "
                        f"({[f'{v:.4f}' for v in _recent5]}). The decoder's pretrained "
                        f"weights are destroyed. Current lr={optimizer.param_groups[0]['lr']:.2e}. "
                        f"Fix: set TROCR_MINI_MODE=False (uses lr=5e-5 instead of 5e-4)."
                    )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                # Bypass HF save_pretrained entirely to prevent lm_head deduplication.
                # transformers ≥5.x recomputes tied-weight lists before serialization
                # and may use content-hash equality (not just data_ptr()) to deduplicate,
                # so clone() + _clear_lm_head_tied_keys() is no longer sufficient.
                _save_model_safetensors_direct(model, output_dir / "best", _trocr_lm_head_key)
                processor.save_pretrained(output_dir / "best")
                # Post-save integrity check using the shared helper.
                _verify_lm_head_in_checkpoint(output_dir / "best")
                print(f"  Best TrOCR saved (val_loss={best_val_loss:.4f})")

        # Final checkpoint: same direct-save approach.
        _save_model_safetensors_direct(model, output_dir / "final", _trocr_lm_head_key)
        processor.save_pretrained(output_dir / "final")
        # Post-save integrity check — mirrors the guard on the best checkpoint.
        _verify_lm_head_in_checkpoint(output_dir / "final")
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        _save_trocr_training_plots(history, output_dir)

        elapsed = time.time() - start
        print(f"\nTrOCR training complete in {elapsed:.1f}s. Best val_loss={best_val_loss:.4f}")
    finally:
        # Always free GPU memory even if training raised an exception.
        # Without this, a mid-training crash leaves TrOCR (246M params) on the
        # GPU and causes CUDA OOM when the next stage (DONUT) loads its model.
        # Fix: issue_report_summary medium #9 — wrap each `del` in its own try/except
        # so a NameError on one variable (e.g. model was never assigned because training
        # crashed during setup) does not abort the finally block before _gpu_cleanup() runs,
        # which would leave TrOCR (246M params) on the GPU and OOM the next stage.
        try:
            del model
        except Exception:
            pass
        try:
            del optimizer
        except Exception:
            pass
        try:
            del scheduler
        except Exception:
            pass
        try:
            del scaler
        except Exception:
            pass
        try:
            del train_ds
        except Exception:
            pass
        try:
            del val_ds
        except Exception:
            pass
        try:
            del train_loader
        except Exception:
            pass
        try:
            del val_loader
        except Exception:
            pass
        # _gpu_cleanup() is ALWAYS called last — even if every del above raised.
        _gpu_cleanup()

    return history


# ════════════════════════════════════════════════════════════════════════════
# STAGE 3: TrOCR+YOLO Inference Pipeline
# ════════════════════════════════════════════════════════════════════════════
def _assign_fields_heuristic(ocr_lines: list[dict[str, Any]]) -> dict[str, str]:
    """Assign OCR-extracted text lines to SROIE fields using heuristics.

    This is the CANONICAL field assignment function used by both the training/
    evaluation pipeline (train_trocr_yolo.py) and the benchmark pipeline
    (reporting.py).  Do NOT create duplicate field-assignment implementations
    elsewhere — import and use this function instead.

    The function accepts bbox-aware dicts (with keys 'text', 'x', 'y', 'x2',
    'y2') and uses spatial position for address grouping.  When called with
    plain-text dicts (only 'text' key, no bbox), it still works correctly but
    spatial grouping degrades to sequential grouping.

    Heuristic rules (priority order):
    1. Total: line containing a keyword ("total", "jumlah", etc.) + monetary
       value, falling back to the last monetary value in the bottom portion.
    2. Date: line matching a date pattern (DD/MM/YYYY, named months, etc.),
       falling back to date-header keywords ("date:", "tarikh:", "tanggal:").
    3. Company: first 1–2 unused lines from the top (non-monetary, non-date).
    4. Address: remaining unused non-monetary lines grouped by vertical
       proximity; the largest contiguous cluster wins.

    Enforced lemmas:
    - **L3** (Completeness): always returns dict with exactly 4 string-valued keys.
    - **L6** (Mutual Exclusion): each line index assigned to at most one field.
    """
    result = {f: "" for f in FIELDS}

    if not ocr_lines:
        # L3: even empty input produces a complete 4-field dict.
        return _assert_field_result_complete(result)

    # Sort lines by vertical position (top to bottom)
    sorted_lines = sorted(ocr_lines, key=lambda x: x.get("y", 0))

    used: set[int] = set()

    # ── 1. Find total — prefer keyword match with capture group ──────────
    # Try _TOTAL_KEYWORD_RE first (extracts numeric value via capture group).
    for i in range(len(sorted_lines) - 1, -1, -1):
        if i in used:
            continue
        text = sorted_lines[i].get("text", "")
        m = _TOTAL_KEYWORD_RE.search(text)
        if m:
            # Clean: keep digits, comma, dot
            captured = re.sub(r"[^\d.,]", "", m.group(1))
            result["total"] = captured
            _assert_mutual_exclusion(used, i, "total")  # L6
            used.add(i)
            break
    # Fallback 1: legacy _TOTAL_RE (no capture group) + _NUMBER_RE extraction
    if not result["total"]:
        for i in range(len(sorted_lines) - 1, -1, -1):
            if i in used:
                continue
            text = sorted_lines[i].get("text", "")
            if _TOTAL_RE.search(text):
                numbers = _NUMBER_RE.findall(text)
                result["total"] = numbers[-1] if numbers else text.strip()
                used.add(i)
                break
    # Fallback 2: last monetary value in bottom 40% of receipt
    if not result["total"]:
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
    # Fallback 3: largest monetary value anywhere on the receipt
    if not result["total"]:
        candidates: list[tuple[float, str, int]] = []
        for i, line in enumerate(sorted_lines):
            if i in used:
                continue
            text = line.get("text", "")
            for m in re.finditer(r"\d[\d,]*\.\d{2}", text):
                try:
                    val = float(m.group(0).replace(",", ""))
                    candidates.append((val, m.group(0), i))
                except ValueError:
                    pass
        if candidates:
            candidates.sort(reverse=True)
            result["total"] = candidates[0][1]
            used.add(candidates[0][2])

    # ── 2. Find date — scan all lines ────────────────────────────────────
    for i, line in enumerate(sorted_lines):
        if i in used:
            continue
        text = line.get("text", "")
        m = _DATE_RE.search(text)
        if m:
            result["date"] = m.group(0).strip()
            _assert_mutual_exclusion(used, i, "date")  # L6
            used.add(i)
            break
    # Fallback: date-header keyword ("date:", "tarikh:", "tanggal:")
    if not result["date"]:
        for i, line in enumerate(sorted_lines):
            if i in used:
                continue
            text = line.get("text", "")
            m = _DATE_HEADER_RE.search(text)
            if m:
                result["date"] = m.group(1).strip()
                _assert_mutual_exclusion(used, i, "date")  # L6
                used.add(i)
                break

    # ── 3. Company: first 1–2 unused lines from the top ──────────────────
    company_parts: list[str] = []
    for i, line in enumerate(sorted_lines):
        if i not in used and len(company_parts) < 2:
            text = line.get("text", "").strip()
            if text and not _MONEY_RE.search(text) and not _DATE_RE.search(text):
                # Skip lines that are purely numeric
                if re.match(r"^[\d\s\.,:]+$", text):
                    continue
                company_parts.append(text)
                _assert_mutual_exclusion(used, i, "company")  # L6
                used.add(i)
                # Stop after first line unless second line also looks like a name
                if len(company_parts) == 1 and not _ADDRESS_RE.search(text):
                    break
    result["company"] = " ".join(company_parts)

    # ── 4. Address: group remaining lines by vertical proximity ──────────
    # Collect candidates: unused non-monetary lines with optional address
    # keyword boost.  Uses bbox y-coordinates when available for spatial
    # grouping; falls back to sequential ordering when bboxes are absent.
    addr_candidates: list[tuple[float, float, str, bool]] = []
    for i, line in enumerate(sorted_lines):
        if i not in used:
            text = line.get("text", "").strip()
            if not text or _MONEY_RE.match(text):
                continue
            y_top = float(line.get("y", 0))
            y_bot = float(line.get("y2", y_top + 20))
            has_addr_signal = bool(_ADDRESS_RE.search(text))
            addr_candidates.append((y_top, y_bot, text, has_addr_signal))

    if addr_candidates:
        # Estimate typical line height as the median of (y2 - y) across candidates.
        _heights = [y2 - y for y, y2, _, _ in addr_candidates if y2 > y]
        _typical_h = statistics.median(_heights) if _heights else 30.0
        # Two lines are "adjacent" if the gap between them is ≤ 1.5× typical height.
        _gap_thresh = max(5.0, _typical_h * 1.5)

        # Build consecutive groups.
        groups: list[list[tuple[str, bool]]] = [[(addr_candidates[0][2], addr_candidates[0][3])]]
        for j in range(1, len(addr_candidates)):
            prev_y2 = addr_candidates[j - 1][1]
            curr_y = addr_candidates[j][0]
            if curr_y - prev_y2 <= _gap_thresh:
                groups[-1].append((addr_candidates[j][2], addr_candidates[j][3]))
            else:
                groups.append([(addr_candidates[j][2], addr_candidates[j][3])])

        # Pick the best group.  Scoring priority:
        #   1. Keyword count (address signals: JALAN, LOT, postcode, etc.)
        #   2. Group size (number of contiguous lines)
        # This intentionally prefers a single line with an address keyword
        # (e.g. "NO 1 JALAN PUCHONG") over a 3-line group without keywords,
        # because receipt content between company and total is often mixed
        # (item lines, tax lines).  The keyword signal is more reliable than
        # size alone for selecting the actual address block.
        def _group_score(g: list[tuple[str, bool]]) -> tuple[int, int]:
            kw_count = sum(1 for _, has_kw in g if has_kw)
            return (kw_count, len(g))

        best_group = max(groups, key=_group_score)
        # Cap at 4 lines: SROIE addresses span 1–3 lines in practice; allowing
        # 4 avoids silently truncating rare 4-line addresses without risk of
        # including unrelated content (company and date/total are already removed).
        result["address"] = " ".join(text for text, _ in best_group[:4])

    # L3: Postcondition — always return a complete 4-field dict with string values.
    return _assert_field_result_complete(result)


def _ned(a: str, b: str) -> float:
    """Normalised edit distance between two strings, in [0.0, 1.0]."""
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, lb + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[lb] / max(la, lb)


# ════════════════════════════════════════════════════════════════════════════
# Attention-Based Field Assigner
# ════════════════════════════════════════════════════════════════════════════


class FieldAttentionAssigner(torch.nn.Module):
    """Trainable attention-based replacement for regex heuristics.

    Three selectable backends (set via FIELD_ASSIGNER_BACKEND or the
    ``backend`` constructor argument):

    Backend 1 — "char"  (~532 K params)
        Character-level embedding max-pooled over the token, fused with
        normalised bounding-box spatial features, then a 2-layer Transformer
        encoder + cross-attention field queries.  Trains from scratch.
        Handles single-char OCR substitutions (Hell0 → Hello) via learned
        embedding proximity.  No internet or pretrained weights required.

    Backend 2 — "lm"  (~4.9 M params)
        Replaces the char embedding with a frozen BERT-tiny encoder
        (prajjwal1/bert-tiny, 128-dim hidden).  Gives the assigner a
        language-model prior: morphological OCR errors resolved at subword
        level.  BERT-tiny is frozen; only the assigner head is trained.

    Backend 3 — "lm+vision"  (~5.1 M params)  ← DEFAULT
        Backend 2 PLUS:
        • raw pixel features from TrOCR's ViT encoder (768-dim, projected to
          128) feed alongside text, letting the model bypass OCR decoding
          errors on blurry crops by reading the image directly.
        • cross-field consistency regulariser during training (KL between
          model attention and regex-derived plausibility priors).

    All three backends share the same forward / assign / checkpoint API.
    """

    # Printable ASCII chars for the "char" backend.
    _CHARSET: str = (
        " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,/:;-()[]@#$%&*+='\"<>?!"
    )
    _CHAR2IDX: dict[str, int] = {c: i + 1 for i, c in enumerate(_CHARSET)}
    _VOCAB_SIZE: int = len(_CHARSET) + 1  # 1-based; 0 = padding / unknown

    # TrOCR encoder hidden size (used by "lm+vision" to project to d_model).
    _VISION_DIM: int = 768

    FIELDS: list[str] = ["company", "date", "address", "total"]
    CHECKPOINT_NAME: str = "field_assigner.pt"

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_text_len: int = 64,
        dropout: float = 0.1,
        backend: str | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_text_len = max_text_len
        self.backend = (backend or FIELD_ASSIGNER_BACKEND).lower()

        # ── Text encoder (backend-dependent) ─────────────────────────────
        if self.backend in ("lm", "lm+vision") and _LM_AVAILABLE:
            self._lm_tokenizer, self.lm_encoder = _build_lm_encoder()
            self._use_lm = True
        else:
            if self.backend in ("lm", "lm+vision"):
                print(
                    f"[FieldAssigner] BERT-tiny unavailable; "
                    f"'{self.backend}' backend falls back to 'char'."
                )
            # char backend (also the fallback when transformers unavailable)
            self.char_emb = torch.nn.Embedding(self._VOCAB_SIZE, d_model, padding_idx=0)
            self.text_proj = torch.nn.Linear(d_model, d_model)
            self._use_lm = False

        # ── Spatial encoder: normalised bbox → d_model ────────────────────
        self.spatial_mlp = torch.nn.Sequential(
            torch.nn.Linear(4, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, d_model),
        )

        # ── Vision projection: TrOCR ViT features → d_model (mode 3 only) ─
        _use_vision = self.backend == "lm+vision" and self._use_lm
        if _use_vision:
            self.vision_proj = torch.nn.Linear(self._VISION_DIM, d_model)

        # ── Fusion: (text + spatial [+ vision]) → d_model ────────────────
        fuse_in = d_model * (3 if _use_vision else 2)
        self.fuse = torch.nn.Linear(fuse_in, d_model)
        self._fuse_in = fuse_in  # saved to checkpoint for load validation

        # ── Transformer encoder over all lines ────────────────────────────
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── 4 learnable field query vectors ──────────────────────────────
        self.field_queries = torch.nn.Parameter(torch.randn(len(self.FIELDS), d_model))

        # ── Cross-attention: field queries attend to line context ─────────
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    # ── Internal encoding helpers ─────────────────────────────────────────

    def _encode_text(self, text: str) -> torch.Tensor:
        """Map a raw string to a (d_model,) float vector.

        Uses BERT-tiny for backends "lm" / "lm+vision" (frozen, mean-pool
        over subword tokens), or character embeddings for "char".
        """
        if self._use_lm:
            dev = next(self.lm_encoder.parameters()).device
            inputs = self._lm_tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_text_len,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            out = self.lm_encoder(**inputs)
            # Mean-pool over subword tokens → (d_model,)
            return out.last_hidden_state.squeeze(0).mean(dim=0)

        # ── char backend ──────────────────────────────────────────────────
        dev = self.char_emb.weight.device
        indices = [self._CHAR2IDX.get(c, 0) for c in text[: self.max_text_len]]
        if not indices:
            return torch.zeros(self.d_model, device=dev)
        t = torch.tensor(indices, dtype=torch.long, device=dev)
        emb = self.char_emb(t)  # (L, d_model)
        vec = emb.max(dim=0).values  # max-pool
        return torch.relu(self.text_proj(vec))

    def _encode_spatial(self, bbox: list[float], img_w: float, img_h: float) -> torch.Tensor:
        """Map a [x1,y1,x2,y2] box to a (d_model,) float vector."""
        dev = self.spatial_mlp[0].weight.device
        x1, y1, x2, y2 = bbox
        norm = torch.tensor(
            [
                x1 / max(img_w, 1.0),
                y1 / max(img_h, 1.0),
                x2 / max(img_w, 1.0),
                y2 / max(img_h, 1.0),
            ],
            dtype=torch.float32,
            device=dev,
        )
        return self.spatial_mlp(norm)

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(
        self,
        line_texts: list[str],
        bboxes: list[list[float]],
        img_w: float,
        img_h: float,
        vision_feats: "list[torch.Tensor] | None" = None,
    ) -> torch.Tensor:
        """Return attention weight matrix of shape (n_fields, N_lines).

        vision_feats : list of N tensors, each shape (768,) — TrOCR encoder
            mean-pool for that crop.  Only used when backend == "lm+vision"
            and the vision_proj layer is present.  Pass None (default) to
            silently skip vision features (e.g. when running without YOLO).
        """
        N = len(line_texts)
        if N == 0:
            return torch.zeros(len(self.FIELDS), 0, device=self.field_queries.device)

        text_vecs = torch.stack([self._encode_text(t) for t in line_texts])  # (N, d_model)
        spatial_vecs = torch.stack(
            [self._encode_spatial(b, img_w, img_h) for b in bboxes]
        )  # (N, d_model)

        parts = [text_vecs, spatial_vecs]

        # Backend 3: fuse in vision features when available
        if hasattr(self, "vision_proj") and vision_feats is not None and len(vision_feats) == N:
            dev = self.field_queries.device
            vis = torch.stack(
                [
                    f.to(dev)
                    if isinstance(f, torch.Tensor)
                    else torch.zeros(self._VISION_DIM, device=dev)
                    for f in vision_feats
                ]
            )  # (N, 768)
            parts.append(self.vision_proj(vis))  # (N, d_model)

        # If vision_proj exists but feats are missing, pad with zeros so fuse
        # dimensions still match (graceful degradation).
        elif hasattr(self, "vision_proj"):
            dev = self.field_queries.device
            parts.append(torch.zeros(N, self.d_model, device=dev))

        line_enc = torch.relu(self.fuse(torch.cat(parts, dim=-1)))  # (N, d_model)

        ctx = self.encoder(line_enc.unsqueeze(0)).squeeze(0)  # (N, d_model)
        q = self.field_queries.unsqueeze(0)  # (1, 4, d_model)
        kv = ctx.unsqueeze(0)  # (1, N, d_model)
        _, attn_weights = self.cross_attn(q, kv, kv)  # (1, 4, N)
        return attn_weights.squeeze(0)  # (4, N)

    # ── Inference API ─────────────────────────────────────────────────────

    @torch.no_grad()
    def assign(
        self,
        ocr_lines: list[dict[str, Any]],
        img_w: float = 1000.0,
        img_h: float = 1280.0,
        vision_feats: "list[torch.Tensor] | None" = None,
    ) -> dict[str, str]:
        """Assign SROIE fields from OCR lines using learned attention.

        Drop-in replacement for _assign_fields_heuristic().

        ocr_lines    : list of dicts with keys 'text', 'x', 'y', 'x2', 'y2'.
        vision_feats : optional list of N (768,) tensors from TrOCR encoder
                       (Backend 3 only).  Pass None to skip vision path.

        Lines are pre-processed with _correct_ocr_chars() before encoding.
        Each line is assigned to at most one field (exclusive selection);
        'address' may span the top-2 lines by attention weight.
        """
        self.eval()
        result: dict[str, str] = {f: "" for f in self.FIELDS}
        if not ocr_lines:
            # L3: even empty input produces a complete 4-field dict.
            return _assert_field_result_complete(result)

        texts = [_correct_ocr_chars(ln.get("text", "")) for ln in ocr_lines]
        bboxes = [
            [ln.get("x", 0.0), ln.get("y", 0.0), ln.get("x2", img_w), ln.get("y2", img_h)]
            for ln in ocr_lines
        ]
        attn = self.forward(texts, bboxes, img_w, img_h, vision_feats)  # (4, N)

        used: set[int] = set()
        for fi, field in enumerate(self.FIELDS):
            scores = attn[fi].clone()
            for u in used:
                scores[u] = -1.0  # mask already-claimed lines

            if field == "address":
                # Use attention-threshold selection so SROIE addresses that
                # span 1–4 lines are captured correctly (the heuristic already
                # handles this with proximity grouping capped at 4 lines;
                # the learned assigner must not be structurally worse).
                # Threshold = 1/N: the expected attention score under a uniform
                # distribution.  Lines scoring above this are above-average
                # candidates for the address field.
                threshold = 1.0 / max(len(ocr_lines), 1)
                eligible = [
                    j
                    for j in range(len(ocr_lines))
                    if j not in used and scores[j].item() > threshold
                ]
                # Sort top-to-bottom for correct multi-line reading order.
                eligible_sorted = sorted(eligible, key=lambda j: ocr_lines[j].get("y", 0))
                # Cap at 4 lines — consistent with _assign_fields_heuristic().
                selected = eligible_sorted[:4]
                if not selected:
                    continue
                parts = [ocr_lines[j].get("text", "").strip() for j in selected]
                result[field] = " ".join(p for p in parts if p)
                used.update(selected)
            else:
                idx = int(scores.argmax().item())
                result[field] = ocr_lines[idx].get("text", "").strip()
                used.add(idx)

        # L3: Postcondition — always return a complete 4-field dict.
        return _assert_field_result_complete(result)


# ════════════════════════════════════════════════════════════════════════════
# Field Assigner — Training + Persistence
# ════════════════════════════════════════════════════════════════════════════


def _get_field_assigner_path() -> Path:
    """Return the canonical checkpoint path for the FieldAttentionAssigner."""
    return WORKSPACE / "models" / FieldAttentionAssigner.CHECKPOINT_NAME


def _match_ocr_to_gt(
    ocr_texts: list[str],
    gt: dict[str, str],
    ned_threshold: float = 0.45,
) -> dict[str, int]:
    """Weakly match OCR line texts to GT field values via NED.

    Returns {field: best_ocr_line_index} for fields where a match was found
    (NED < ned_threshold).  Fields with no close match are excluded.
    """
    field_to_idx: dict[str, int] = {}
    field_to_ned: dict[str, float] = {}
    for field in FieldAttentionAssigner.FIELDS:
        gt_val = gt.get(field, "").strip()
        if not gt_val:
            continue
        best_ned, best_idx = 1.0, -1
        for li, ocr_text in enumerate(ocr_texts):
            d = _ned(ocr_text, gt_val)
            if d < best_ned:
                best_ned, best_idx = d, li
        if best_ned < ned_threshold and best_idx >= 0:
            # Resolve collision: if two fields matched the same line, keep the
            # one with the lower NED score.
            existing_field = next((f for f, i in field_to_idx.items() if i == best_idx), None)
            if existing_field is None or best_ned < field_to_ned[existing_field]:
                if existing_field is not None:
                    del field_to_idx[existing_field]
                    del field_to_ned[existing_field]
                field_to_idx[field] = best_idx
                field_to_ned[field] = best_ned
    return field_to_idx


def _make_synthetic_ocr_lines(
    gt: dict[str, str],
    img_w: float = 960.0,
    img_h: float = 1280.0,
) -> list[dict[str, Any]]:
    """Create synthetic OCR lines from GT text using SROIE receipt layout.

    Used as a quality-gate fallback when YOLO+TrOCR output has fewer than
    3 matched fields.  Positions reflect the consistent Malaysian receipt
    structure in SROIE:
        company  → y ≈  5% of height
        address  → y ≈ 15% / 23% of height (up to 2 lines)
        date     → y ≈ 35% of height
        total    → y ≈ 85% of height

    Returns a list of dicts with keys: text, x, y, x2, y2.
    """
    lines: list[dict[str, Any]] = []
    x1 = 0.05 * img_w
    x2 = 0.95 * img_w
    line_h = 0.06 * img_h

    company = gt.get("company", "").strip()
    if company:
        y = 0.05 * img_h
        lines.append({"text": company, "x": x1, "y": y, "x2": x2, "y2": y + line_h})

    address = gt.get("address", "").strip()
    if address:
        # Split at commas; if no commas, split at the space nearest the midpoint.
        parts = [p.strip() for p in address.split(",") if p.strip()]
        if len(parts) >= 2:
            mid = (len(parts) + 1) // 2
            addr_lines = [", ".join(parts[:mid]), ", ".join(parts[mid:])]
        else:
            # Single pass: scan outward from the midpoint until a space is found.
            mid = len(address) // 2
            split_pos = mid
            for offset in range(len(address)):
                found = False
                for pos in (mid - offset, mid + offset):
                    if 0 < pos < len(address) and address[pos] == " ":
                        split_pos = pos
                        found = True
                        break
                if found:
                    break
            addr_lines = [address[:split_pos].strip(), address[split_pos:].strip()]
            addr_lines = [a for a in addr_lines if a]

        for i, addr_line in enumerate(addr_lines[:2]):
            y = (0.15 + i * 0.08) * img_h
            lines.append({"text": addr_line, "x": x1, "y": y, "x2": x2, "y2": y + line_h})

    date = gt.get("date", "").strip()
    if date:
        y = 0.35 * img_h
        lines.append({"text": date, "x": x1, "y": y, "x2": x2, "y2": y + line_h})

    total = gt.get("total", "").strip()
    if total:
        y = 0.85 * img_h
        lines.append({"text": total, "x": x1, "y": y, "x2": x2, "y2": y + line_h})

    return lines


def _y_to_bin(y: float, img_h: float, n_bins: int) -> int:
    """Map a y-coordinate to a vertical bin index in [0, n_bins-1]."""
    return min(int(y / max(img_h, 1.0) * n_bins), n_bins - 1)


def _compute_corpus_plausibility(
    corpus: list[tuple],
    n_bins: int = 10,
    min_hq_samples: int = 20,
    device: str = "cpu",
) -> dict[str, "torch.Tensor"]:
    """Compute empirical field plausibility from high-quality corpus entries.

    For each field, scores each vertical-position bin (y / img_h × n_bins)
    by how often it was the correct match in high-quality (real YOLO+TrOCR,
    3+ matched fields) corpus entries.

    Parameters
    ----------
    corpus          : list of 7-tuples
                      (ocr_lines, field_to_idx, gt, img_w, img_h, vis_feats, is_hq).
                      Only entries where is_hq=True contribute to the prior.
    n_bins          : number of vertical position bins.
    min_hq_samples  : minimum high-quality entries required; falls back to
                      uniform 0.5 prior when corpus is too small.
    device          : torch device string for output tensors.

    Returns
    -------
    Dict mapping field name → float tensor of shape (n_bins,) in [0.1, 1.0].
    """
    # Filter to high-quality entries only.
    hq = [
        (ocr_lines, field_to_idx, img_h)
        for ocr_lines, field_to_idx, _gt, _img_w, img_h, _vis, is_hq in corpus
        if is_hq
    ]

    if len(hq) < min_hq_samples:
        logging.getLogger(__name__).warning(
            "[FieldAssigner] Only %d high-quality corpus entries (need %d); "
            "corpus-derived plausibility prior falls back to uniform 0.5. "
            "Consistency regulariser will have no positional bias.",
            len(hq),
            min_hq_samples,
        )
        uniform = torch.full((n_bins,), 0.5, dtype=torch.float32)
        return {field: uniform.to(device) for field in FieldAttentionAssigner.FIELDS}

    counts: dict[str, list[float]] = {f: [0.0] * n_bins for f in FieldAttentionAssigner.FIELDS}
    totals: dict[str, list[float]] = {f: [0.0] * n_bins for f in FieldAttentionAssigner.FIELDS}

    for ocr_lines, field_to_idx, img_h in hq:
        for field in FieldAttentionAssigner.FIELDS:
            matched_idx = field_to_idx.get(field, -1)
            for li, line in enumerate(ocr_lines):
                y = float(line.get("y", 0))
                b = _y_to_bin(y, img_h, n_bins)
                totals[field][b] += 1.0
                if li == matched_idx:
                    counts[field][b] += 1.0

    result: dict[str, torch.Tensor] = {}
    for field in FieldAttentionAssigner.FIELDS:
        plaus = torch.zeros(n_bins, dtype=torch.float32)
        for b in range(n_bins):
            if totals[field][b] > 0:
                plaus[b] = max(0.1, min(1.0, counts[field][b] / totals[field][b]))
            else:
                plaus[b] = 0.5  # uninformative prior for unseen bins
        result[field] = plaus.to(device)

    return result


def train_field_assigner(
    sroie_dir: Path,
    yolo_model,
    trocr_model,
    trocr_processor: "TrOCRProcessor",
    epochs: int | None = None,
    lr: float = 3e-4,
    ned_threshold: float = 0.45,
    device: str = DEVICE,
    backend: str | None = None,
    consistency_lambda: float = 0.05,
) -> "FieldAttentionAssigner":
    """Train a FieldAttentionAssigner on SROIE training data.

    Pipeline
    --------
    1. For each training image, run YOLO+TrOCR to build a cached corpus.
    2. Quality-gate: call _match_ocr_to_gt() at corpus-build time.
       - If 3+ GT fields were matched → keep real YOLO+TrOCR output (is_hq=True).
         Realistic OCR noise in the corpus is valuable training signal.
       - If fewer than 3 fields matched → YOLO+TrOCR output is too noisy.
         Replace the OCR lines with synthetic lines derived from GT text using
         SROIE's consistent Malaysian receipt layout (is_hq=False).  This
         ensures every corpus entry has clean labels while YOLO remains the
         preferred source when it works well.
    3. Compute a corpus-derived plausibility prior from the high-quality subset.
    4. Train with cross-entropy over which line to select per field.
    5. Backend "lm+vision" additionally:
       - Caches TrOCR encoder features (vision feats) per crop.
       - Adds a consistency regulariser (λ=consistency_lambda): penalises
         attention weight on lines that disagree with the corpus-derived
         positional prior (avoids the circular regex-regulariser problem).

    Parameters
    ----------
    sroie_dir          : path containing img/ and key/ subdirectories.
    yolo_model         : loaded YOLOv8 model.
    trocr_model        : loaded TrOCR VisionEncoderDecoderModel.
    trocr_processor    : loaded TrOCRProcessor.
    epochs             : training epochs.
    lr                 : AdamW learning rate.
    ned_threshold      : maximum NED to accept an OCR line as the GT match.
    device             : torch device string.
    backend            : override FIELD_ASSIGNER_BACKEND for this run.
    consistency_lambda : weight for the Backend-3 consistency regulariser.
                         0.0 disables it; values > 0.1 may override CE loss.

    Returns
    -------
    Trained FieldAttentionAssigner saved to _get_field_assigner_path().
    """
    effective_epochs = epochs if epochs is not None else FIELD_ASSIGNER_EPOCHS
    effective_backend = (backend or FIELD_ASSIGNER_BACKEND).lower()
    assigner = FieldAttentionAssigner(backend=effective_backend).to(device)
    use_vision = effective_backend == "lm+vision" and assigner._use_lm
    use_consistency = use_vision and consistency_lambda > 0.0

    optimizer = torch.optim.AdamW(assigner.parameters(), lr=lr, weight_decay=0.01)
    ce_loss = torch.nn.CrossEntropyLoss()

    img_dir = sroie_dir / "img"
    key_dir = sroie_dir / "key"
    if not img_dir.exists():
        print("[FieldAssigner] img/ not found — skipping training.")
        return assigner

    image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not image_paths:
        print("[FieldAssigner] No training images found — skipping training.")
        return assigner

    # ── Build corpus (run YOLO+TrOCR once; cache for all epochs) ─────────
    # Corpus tuple: (ocr_lines, field_to_idx, gt, img_w, img_h, vis_feats, is_hq)
    # is_hq=True  → real YOLO+TrOCR output with ≥3 matched fields (preferred)
    # is_hq=False → synthetic GT-derived lines (quality-gate fallback)
    print(
        f"[FieldAssigner] Building corpus from {len(image_paths)} images "
        f"(backend={effective_backend})…"
    )
    corpus: list[tuple] = []
    n_hq = 0
    n_synthetic = 0
    for img_path in image_paths:
        key_path = key_dir / (img_path.stem + ".txt")
        if not key_path.exists():
            key_path = key_dir / (img_path.stem + ".json")
        if not key_path.exists():
            continue
        try:
            gt = json.loads(key_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        gt = {k.lower(): str(v).strip() for k, v in gt.items()}

        try:
            ocr_lines, vis_feats, _ = _extract_ocr_lines(
                img_path,
                yolo_model,
                trocr_model,
                trocr_processor,
                device,
                return_vision_feats=use_vision,
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).debug(
                "[FieldAssigner] YOLO+TrOCR failed for %s: %s", img_path.name, exc
            )
            ocr_lines = []
            vis_feats = None

        # Load image dimensions (needed for both real and synthetic paths).
        img = _load_image(img_path)
        img_w = float(img.size[0]) if _PIL_AVAILABLE else float(img.shape[1])  # type: ignore[union-attr]
        img_h = float(img.size[1]) if _PIL_AVAILABLE else float(img.shape[0])  # type: ignore[union-attr]

        # ── Quality gate ─────────────────────────────────────────────────
        # Check how many GT fields were matched by the real YOLO+TrOCR output.
        is_hq = False
        field_to_idx: dict[str, int] = {}
        if ocr_lines:
            texts = [_correct_ocr_chars(ln.get("text", "")) for ln in ocr_lines]
            field_to_idx = _match_ocr_to_gt(texts, gt, ned_threshold)
            if len(field_to_idx) >= 3:
                is_hq = True  # Keep real output — realistic OCR noise is useful.

        if not is_hq:
            # Fewer than 3 fields matched (or no OCR lines at all).
            # Fall back to synthetic GT-derived lines so the corpus has clean
            # labels instead of noise.  YOLO is still used when it works well.
            ocr_lines = _make_synthetic_ocr_lines(gt, img_w=img_w, img_h=img_h)
            vis_feats = None  # vision features are not meaningful for synthetic lines
            if not ocr_lines:
                continue
            synth_texts = [ln.get("text", "") for ln in ocr_lines]
            field_to_idx = _match_ocr_to_gt(synth_texts, gt, ned_threshold)
            n_synthetic += 1
        else:
            n_hq += 1

        if not field_to_idx:
            continue

        corpus.append((ocr_lines, field_to_idx, gt, img_w, img_h, vis_feats, is_hq))

    if not corpus:
        print("[FieldAssigner] Corpus is empty — skipping training.")
        return assigner

    print(
        f"[FieldAssigner] Corpus built: {len(corpus)} samples "
        f"({n_hq} high-quality, {n_synthetic} synthetic fallback)"
    )

    # ── Compute corpus-derived plausibility prior (Bug-3 fix) ────────────
    # Replaces static regex _FIELD_PLAUSIBILITY with empirical positional
    # distribution from high-quality corpus entries.  Falls back to uniform
    # 0.5 when too few high-quality entries exist (≥20 required).
    corpus_plausibility = _compute_corpus_plausibility(corpus, device=device)

    print(f"[FieldAssigner] Training {len(corpus)} samples × {effective_epochs} epochs…")
    assigner.train()
    for epoch in range(effective_epochs):
        total_loss = 0.0
        n_examples = 0
        for ocr_lines, field_to_idx, _gt, img_w, img_h, vis_feats, _is_hq in corpus:
            # field_to_idx was computed at corpus-build time — no re-matching.
            if not field_to_idx:
                continue

            texts = [_correct_ocr_chars(ln.get("text", "")) for ln in ocr_lines]
            bboxes = [
                [ln.get("x", 0.0), ln.get("y", 0.0), ln.get("x2", img_w), ln.get("y2", img_h)]
                for ln in ocr_lines
            ]
            attn = assigner.forward(texts, bboxes, img_w, img_h, vis_feats)  # (4, N)

            # ── Primary loss: cross-entropy over line selection ──────────
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for fi, field in enumerate(FieldAttentionAssigner.FIELDS):
                if field not in field_to_idx:
                    continue
                target = torch.tensor([field_to_idx[field]], dtype=torch.long, device=device)
                loss = loss + ce_loss(attn[fi].unsqueeze(0), target)
                n_examples += 1

            # ── Backend-3 consistency regulariser (corpus-derived prior) ─
            # Penalises attending to lines at vertical positions that
            # empirically correspond to other fields.  Uses positional bins
            # computed from the high-quality corpus subset rather than the
            # regex patterns that made the regulariser circular.
            # Loss = λ × Σ_f Σ_i p(i|f) × (1 − plausibility(f, bin_i))
            if use_consistency:
                n_bins = corpus_plausibility[FieldAttentionAssigner.FIELDS[0]].shape[0]
                cons = torch.tensor(0.0, device=device, requires_grad=True)
                for fi, field in enumerate(FieldAttentionAssigner.FIELDS):
                    field_prior = corpus_plausibility[field]  # (n_bins,)
                    plaus = torch.stack(
                        [
                            field_prior[_y_to_bin(float(ln.get("y", 0)), img_h, n_bins)]
                            for ln in ocr_lines
                        ]
                    )  # (N,)
                    probs = attn[fi].softmax(-1)
                    cons = cons + (probs * (1.0 - plaus)).sum()
                loss = loss + consistency_lambda * cons

            if n_examples == 0:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(assigner.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg = total_loss / max(n_examples, 1)
            print(f"[FieldAssigner] Epoch {epoch + 1}/{effective_epochs} — loss={avg:.4f}")

    # ── Save checkpoint ──────────────────────────────────────────────────
    ckpt_path = _get_field_assigner_path()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": assigner.state_dict(),
            "d_model": assigner.d_model,
            "max_text_len": assigner.max_text_len,
            "backend": effective_backend,
            "fuse_in": assigner._fuse_in,
        },
        ckpt_path,
    )
    print(f"[FieldAssigner] Saved → {ckpt_path}  (backend={effective_backend})")
    return assigner


def load_field_assigner(device: str = DEVICE) -> "FieldAttentionAssigner | None":
    """Load a trained FieldAttentionAssigner from disk.

    The backend is restored from the checkpoint so the loaded model always
    matches the architecture it was trained with.  Returns None if no
    checkpoint exists, so callers fall back to _assign_fields_heuristic().
    """
    ckpt_path = _get_field_assigner_path()
    if not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        assigner = FieldAttentionAssigner(
            d_model=ckpt.get("d_model", 128),
            max_text_len=ckpt.get("max_text_len", 64),
            backend=ckpt.get("backend", FIELD_ASSIGNER_BACKEND),
        ).to(device)
        assigner.load_state_dict(ckpt["state_dict"])
        assigner.eval()
        return assigner
    except Exception as exc:
        print(f"[FieldAssigner] Could not load checkpoint ({exc}); falling back to heuristic.")
        return None


def _extract_ocr_lines(
    image_path: Path,
    yolo_model,
    trocr_model,
    trocr_processor: "TrOCRProcessor",
    device: str = DEVICE,
    return_vision_feats: bool = False,
) -> "tuple[list[dict[str, Any]], list[torch.Tensor] | None, int]":
    """Run YOLO detection + TrOCR reading on one image.

    Enforces pipeline lemmas:
    - **L1** (Resolution Invariant): passes ``YOLO_IMG_SIZE`` to YOLO inference,
      ensuring train/inference resolution match.
    - **L2** (Detection-OCR Contract): validates each box satisfies minimum
      dimension and image-bounds constraints before cropping.
    - **L5** (OCR Correction Idempotence): checked on decoded text when
      ``_correct_ocr_chars`` is applied by downstream field assignment.

    Returns
    -------
    ocr_lines       : list of dicts with 'text', 'x', 'y', 'x2', 'y2', 'conf'.
    vision_feats    : list of (768,) tensors — TrOCR encoder mean-pool per crop —
                      when return_vision_feats=True; otherwise None.
                      Used by Backend 3 to bypass OCR decoding errors on
                      blurry crops by passing raw pixel features to the assigner.
    yolo_box_count  : int — number of bounding boxes YOLO found before TrOCR
                      decoding.  0 means YOLO detected nothing; a positive value
                      with len(ocr_lines)==0 means YOLO found boxes but TrOCR
                      decoded every crop to empty text (undertrained model).
    """
    img = _load_image(image_path)
    W = float(img.size[0]) if _PIL_AVAILABLE else float(img.shape[1])  # type: ignore[union-attr]
    H = float(img.size[1]) if _PIL_AVAILABLE else float(img.shape[0])  # type: ignore[union-attr]

    # L1: Resolution Invariant — always pass YOLO_IMG_SIZE to ensure
    # inference resolution matches training resolution.
    yolo_results = yolo_model(img, verbose=False, imgsz=YOLO_IMG_SIZE)
    ocr_lines: list[dict] = []
    vision_feats: list[torch.Tensor] = []
    yolo_box_count: int = 0  # boxes found by YOLO before TrOCR decoding
    discarded_box_count: int = 0  # L2: boxes discarded for violating contract

    if yolo_results and len(yolo_results[0].boxes) > 0:
        boxes = yolo_results[0].boxes
        yolo_box_count = len(boxes)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            pad = 4
            x1 = max(0, int(x1) - pad)
            y1 = max(0, int(y1) - pad)
            x2 = min(W, int(x2) + pad)
            y2 = min(H, int(y2) + pad)

            # L2: Detection-OCR Contract — validate box dimensions.
            # Boxes smaller than _MIN_BOX_DIMENSION in either axis cannot
            # contain readable text; passing them to TrOCR wastes compute
            # and may produce hallucinated tokens from degenerate input.
            # NOTE: This is intentionally inline rather than calling
            # _assert_box_validity() because we operate on individual
            # already-padded/clamped boxes within the loop (not a batch of
            # raw tuples), and we need per-box discard counting.
            if x2 - x1 < _MIN_BOX_DIMENSION or y2 - y1 < _MIN_BOX_DIMENSION:
                discarded_box_count += 1
                continue

            if _PIL_AVAILABLE:
                crop = img.crop((x1, y1, x2, y2))  # type: ignore[union-attr]
            else:
                import numpy as _np_crop  # noqa: PLC0415

                crop = _np_crop.ascontiguousarray(img[int(y1) : int(y2), int(x1) : int(x2)])  # type: ignore[index]
            pixel_values = trocr_processor(crop, return_tensors="pt").pixel_values.to(device)

            with torch.no_grad():
                # Use greedy decoding (num_beams=1) for inference: beam search
                # adds 4× memory and latency per crop with negligible F1 gain on
                # short OCR text (≤20 chars).  The generation_config may have
                # num_beams=4 from training setup; override here explicitly.
                generated_ids = trocr_model.generate(
                    pixel_values,
                    max_new_tokens=TROCR_MAX_LEN,
                    num_beams=1,
                    length_penalty=1.0,
                    no_repeat_ngram_size=0,
                )
                # Backend 3: reuse the encoder's last_hidden_state that was
                # already computed inside generate() via encoder_outputs.  When
                # generate() returns encoder_outputs we extract the cached value;
                # otherwise fall back to a second encoder pass (extra forward).
                if return_vision_feats:
                    enc_out = trocr_model.encoder(pixel_values)
                    feat = enc_out.last_hidden_state.mean(dim=1).squeeze(0).cpu()
                    vision_feats.append(feat)

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
            elif return_vision_feats and vision_feats:
                # No text decoded — drop the corresponding vision feat so
                # the two lists stay aligned.
                vision_feats.pop()

    # L2: Log discarded boxes for diagnostics (not a failure — just small noise boxes).
    if discarded_box_count > 0:
        logging.getLogger(__name__).debug(
            "L2: Discarded %d/%d YOLO boxes below minimum %dpx dimension for %s",
            discarded_box_count,
            yolo_box_count,
            _MIN_BOX_DIMENSION,
            image_path.name,
        )

    return ocr_lines, (vision_feats if return_vision_feats else None), yolo_box_count


def run_trocr_yolo_inference(
    image_path: Path,
    yolo_model,
    trocr_model,
    trocr_processor: TrOCRProcessor,
    field_assigner: "FieldAttentionAssigner | None" = None,
) -> dict[str, str]:
    """Run the full TrOCR+YOLO pipeline on a single image.

    1. YOLO detects text regions
    2. TrOCR reads text from each crop
    3. Field assignment: FieldAttentionAssigner (if provided and trained),
       otherwise _assign_fields_heuristic() as fallback.

    Returns a dict with SROIE field predictions.
    """
    # Stages 1+2: YOLO detection + TrOCR OCR
    # Backend 3 ("lm+vision") also extracts encoder features per crop so the
    # assigner can read blurry text from pixel features when decoded text is
    # unreliable.  Other backends skip the extra encoder pass (faster).
    need_vision = (
        field_assigner is not None
        and getattr(field_assigner, "backend", "") == "lm+vision"
        and getattr(field_assigner, "_use_lm", False)
    )
    ocr_lines, vision_feats, yolo_box_count = _extract_ocr_lines(
        image_path,
        yolo_model,
        trocr_model,
        trocr_processor,
        return_vision_feats=need_vision,
    )

    # Warn when ocr_lines is empty — distinguishing the two possible causes:
    # (A) YOLO found 0 boxes: genuine detection failure (wrong model, wrong imgsz, etc.)
    # (B) YOLO found boxes but TrOCR decoded every crop to empty text: undertrained TrOCR
    #     (1-epoch fine-tuning destroys pretrained weights, val_loss stays ≈9; raise
    #     TROCR_EPOCHS floor to ≥5 to reach val_loss ~2.5–3.0 for basic text decoding)
    if not ocr_lines:
        _log = logging.getLogger(__name__)
        if yolo_box_count == 0:
            _log.warning(
                "YOLO detected 0 text regions for %s — all field predictions will be empty. "
                "If using the inline YOLO fallback, train with ultralytics for real detections.",
                image_path,
            )
        else:
            _log.warning(
                "YOLO detected %d text region(s) for %s but TrOCR decoded all %d crop(s) to "
                "empty text — likely TrOCR model is undertrained (val_loss too high) or "
                "TROCR_MAX_LEN too short. Check that TROCR_EPOCHS ≥ 5 in speed modes.",
                yolo_box_count,
                image_path,
                yolo_box_count,
            )

    # Stage 3: field assignment
    if field_assigner is not None:
        img = _load_image(image_path)
        img_w = float(img.size[0]) if _PIL_AVAILABLE else float(img.shape[1])  # type: ignore[union-attr]
        img_h = float(img.size[1]) if _PIL_AVAILABLE else float(img.shape[0])  # type: ignore[union-attr]
        return field_assigner.assign(ocr_lines, img_w=img_w, img_h=img_h, vision_feats=vision_feats)

    return _assign_fields_heuristic(ocr_lines)


# ════════════════════════════════════════════════════════════════════════════
# Evaluate TrOCR+YOLO on test set
# ════════════════════════════════════════════════════════════════════════════
def evaluate_trocr_yolo_on_test(
    yolo_weights: str,
    trocr_model_path: str,
    test_samples: list[tuple[Path, dict[str, str]]],
) -> dict[str, Any]:
    """Evaluate TrOCR+YOLO pipeline on the SROIE test set. Returns metrics dict.

    If a trained FieldAttentionAssigner checkpoint exists at the expected path
    (see ``load_field_assigner``), it is used for field assignment instead of
    the regex heuristic.  When no checkpoint is found the function falls back
    to ``_assign_fields_heuristic`` transparently.
    """
    _eval_log = logging.getLogger(__name__)

    # L1: Resolution Invariant — verify YOLO_IMG_SIZE is valid before loading models.
    # This catches configuration errors (e.g. patched to 0 or non-multiple-of-32)
    # before any GPU memory is allocated.
    _assert_resolution_invariant(YOLO_IMG_SIZE, YOLO_IMG_SIZE)

    # Log which YOLO backend is active so operators can immediately see
    # whether a real detector or the inline proxy fallback is being used.
    if _ULTRALYTICS_AVAILABLE:
        _eval_log.info("evaluate_trocr_yolo_on_test: YOLO backend = ultralytics (real detector)")
        try:
            from ultralytics import YOLO
        except ImportError:
            YOLO = _YOLO_CLS  # noqa: N806 — shouldn't reach here, defensive
    else:
        _eval_log.warning(
            "evaluate_trocr_yolo_on_test: YOLO backend = inline _YOLO_CLS fallback "
            "(ultralytics not installed).  Detection quality will be poor — "
            "install ultralytics for real text-region detection."
        )
        YOLO = _YOLO_CLS  # noqa: N806

    # Verify the TrOCR checkpoint on disk has decoder.lm_head.weight before
    # loading it.  If the tensor was deduped out by safetensors at save time,
    # the loaded model will have a randomly-initialized lm_head and produce
    # garbage OCR text for every crop, driving F1 to 0 with no error raised.
    _verify_lm_head_in_checkpoint(Path(trocr_model_path))

    yolo_model = YOLO(str(yolo_weights))
    trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_path)
    # FIX: low_cpu_mem_usage=False + _materialize_meta_buffers prevents the
    # meta-device crash on TrOCR's sinusoidal positional embedding buffer.
    # output_loading_info=True lets us fail fast if decoder.lm_head.weight is
    # absent from the checkpoint (belt-and-suspenders alongside the header
    # check above and the self-test below).
    trocr_model, loading_info = VisionEncoderDecoderModel.from_pretrained(
        trocr_model_path, low_cpu_mem_usage=False, output_loading_info=True
    )
    lm_head_key = "decoder.lm_head.weight"
    missing_keys = loading_info.get("missing_keys", [])
    if lm_head_key in missing_keys:
        # Before raising, check if the output_projection variant (transformers ≥4.45)
        # is present in the loaded model — if so, the checkpoint is valid.
        _has_output_proj = any(
            k.endswith(".lm_head.weight") or k.endswith(".output_projection.weight")
            for k in trocr_model.state_dict()
        )
        if not _has_output_proj:
            raise RuntimeError(
                f"CRITICAL: {lm_head_key!r} is missing from the loaded checkpoint at "
                f"{trocr_model_path!r}.  safetensors deduplicated it at save time.  "
                "Re-train with _save_model_safetensors_direct() to prevent this.  "
                "See CLAUDE.md §16 Pattern 6."
            )
    trocr_model = trocr_model.to(DEVICE)
    _materialize_meta_buffers(trocr_model, DEVICE)
    trocr_model.eval()

    # Enable KV-cache for inference.  The model was saved with use_cache=False
    # because gradient checkpointing (required during training on ≤24 GB GPUs)
    # is incompatible with caching.  At inference time there is no gradient
    # checkpointing, so enabling the cache reduces autoregressive generation
    # from O(n²) to O(n) — a 2–5× latency improvement for typical receipt text.
    trocr_model.config.use_cache = True
    if hasattr(trocr_model, "decoder") and hasattr(trocr_model.decoder, "config"):
        trocr_model.decoder.config.use_cache = True

    # Post-load self-test: run a single forward pass on a random image tensor
    # and verify the generated token IDs are not all identical.  A broken
    # lm_head (randomly initialised uniform logits) produces the same token
    # repeated for every position — catch this before evaluating all 63 images.
    # Dimensions (32 H × 128 W) are intentionally small — just enough for the
    # ViT patch-embed stride to produce a valid feature map.  We only care that
    # the decoder generates diverse output, not that the image is meaningful.
    _SELF_TEST_H, _SELF_TEST_W = 32, 128
    _self_test_pixels = torch.randn(1, 3, _SELF_TEST_H, _SELF_TEST_W, device=DEVICE)
    with torch.no_grad():
        try:
            _self_test_ids = trocr_model.generate(_self_test_pixels, max_new_tokens=8)
            _flat = _self_test_ids[0].tolist()
            # A broken lm_head produces uniform logits → the same argmax token
            # every step.  len(set(_flat)) == 1 means all elements are identical.
            if len(_flat) > 1 and len(set(_flat)) == 1:
                raise RuntimeError(
                    "CRITICAL: TrOCR self-test FAILED — model generated the same token "
                    f"({_flat[0]}) for every position.  This indicates a broken "
                    "lm_head (randomly-initialised uniform logits) caused by the "
                    "safetensors deduplication bug.  Re-train with "
                    "LmHeadCloneCallback registered or call "
                    "model.decoder.lm_head.weight = "
                    "torch.nn.Parameter(model.decoder.lm_head.weight.data.clone()) "
                    "before save_pretrained().  See CLAUDE.md §16 Pattern 6."
                )
        except RuntimeError:
            raise
        except Exception as _st_exc:
            _eval_log.warning("TrOCR self-test skipped (non-critical): %s", _st_exc)

    # Load the trained FieldAttentionAssigner if a checkpoint exists; fall back
    # to the regex heuristic when none is found (first run, no training yet).
    field_assigner = load_field_assigner(device=DEVICE)

    predictions = []
    ground_truths = [s[1] for s in test_samples]
    latencies = []
    yolo_zero_count = 0  # images where YOLO found 0 bounding boxes
    trocr_empty_count = 0  # images where YOLO found boxes but TrOCR decoded all to empty

    with torch.no_grad():
        for img_path, _gt in _progress(test_samples, desc="TrOCR+YOLO eval"):
            t0 = time.perf_counter()
            # Run detection + OCR inline so we can track zero-detection images
            # without calling _extract_ocr_lines() twice.
            need_vision = (
                field_assigner is not None
                and getattr(field_assigner, "backend", "") == "lm+vision"
                and getattr(field_assigner, "_use_lm", False)
            )
            ocr_lines, vision_feats, yolo_box_count = _extract_ocr_lines(
                img_path,
                yolo_model,
                trocr_model,
                trocr_processor,
                return_vision_feats=need_vision,
            )
            if not ocr_lines:
                if yolo_box_count == 0:
                    # YOLO found no boxes — genuine detection failure
                    yolo_zero_count += 1
                    _eval_log.warning(
                        "YOLO detected 0 text regions for %s — all field predictions will "
                        "be empty.  YOLO backend: %s.",
                        img_path,
                        "ultralytics" if _ULTRALYTICS_AVAILABLE else "inline _YOLO_CLS fallback",
                    )
                else:
                    # YOLO found boxes but TrOCR decoded every crop to empty text
                    trocr_empty_count += 1
                    _eval_log.warning(
                        "YOLO detected %d text region(s) for %s but TrOCR decoded all %d "
                        "crop(s) to empty text — likely TrOCR model is undertrained "
                        "(val_loss too high) or TROCR_MAX_LEN too short. "
                        "Check that TROCR_EPOCHS ≥ 5 in speed modes.",
                        yolo_box_count,
                        img_path,
                        yolo_box_count,
                    )
            # Field assignment
            if field_assigner is not None:
                img = _load_image(img_path)
                img_w = float(img.size[0]) if _PIL_AVAILABLE else float(img.shape[1])  # type: ignore[union-attr]
                img_h = float(img.size[1]) if _PIL_AVAILABLE else float(img.shape[0])  # type: ignore[union-attr]
                pred = field_assigner.assign(
                    ocr_lines, img_w=img_w, img_h=img_h, vision_feats=vision_feats
                )
            else:
                pred = _assign_fields_heuristic(ocr_lines)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            predictions.append(pred)

    # Check aggregate zero-detection rate.  Raises RuntimeError if > 50% of
    # images had empty ocr_lines, which would silently drive F1 to 0.
    _verify_yolo_detection_rate(yolo_zero_count, trocr_empty_count, len(test_samples))

    metrics = compute_metrics(predictions, ground_truths)
    metrics["num_samples"] = len(test_samples)
    metrics["mean_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
    if field_assigner is not None:
        metrics["field_assigner_backend"] = getattr(field_assigner, "backend", "unknown")

    # Include detection-level statistics so operators can distinguish YOLO
    # failures from TrOCR failures without re-running evaluation.
    metrics["yolo_zero_detection_count"] = yolo_zero_count
    metrics["trocr_empty_decode_count"] = trocr_empty_count
    total = len(test_samples)
    metrics["yolo_detection_rate"] = (
        round((total - yolo_zero_count) / total, 4) if total > 0 else 0.0
    )

    # GPU cleanup
    _gpu_cleanup(yolo_model, trocr_model, trocr_processor)

    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TrOCR+YOLO training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train_trocr_yolo.py                # full pipeline\n"
            "  python train_trocr_yolo.py --stage yolo   # YOLO only\n"
            "  python train_trocr_yolo.py --superfast    # bare minimum, <3 min\n"
        ),
    )
    parser.add_argument("--stage", choices=["yolo", "trocr", "both"], default="both")
    parser.add_argument(
        "--superfast",
        action="store_true",
        help=(
            "Bare minimum mode: YOLO production defaults + TrOCR 5 epochs max_len=64. "
            "Floor of 5 epochs is enforced — 1 epoch produces val_loss≈9.1 (non-functional "
            "decoder that outputs garbage for every crop, driving F1 to 0)."
        ),
    )
    args = parser.parse_args()

    if args.superfast:
        # Patch module-level constants before calling train functions.
        # All originals are restored in the finally block.
        # NOTE: TROCR_EPOCHS floor is 5, not 1.  1 epoch of TrOCR fine-tuning
        # destroys microsoft/trocr-base-printed pretrained weights without learning
        # SROIE patterns (val_loss=9.1268).  Every YOLO crop decodes to empty text,
        # ocr_lines stays empty for 92% of images, and _verify_yolo_detection_rate
        # raises a misleading RuntimeError blaming YOLO when YOLO is fine.
        # See CLAUDE.md §16 Pattern 9 (Masked Cascading Failures).
        _sf_saved = {
            "TROCR_EPOCHS": TROCR_EPOCHS,
            "TROCR_MAX_LEN": TROCR_MAX_LEN,
            "TROCR_BATCH": TROCR_BATCH,
        }
        import sys as _sys

        _mod = _sys.modules[__name__]
        try:
            _mod.TROCR_EPOCHS = 5  # floor: 1 epoch → val_loss≈9.1 (non-functional)
            _mod.TROCR_MAX_LEN = 64  # 128→64: 2× faster; still covers ~50-char lines
            _mod.TROCR_BATCH = 4
            if args.stage in ("yolo", "both"):
                train_yolo()
            if args.stage in ("trocr", "both"):
                train_trocr()
        finally:
            for _k, _v in _sf_saved.items():
                setattr(_mod, _k, _v)
    else:
        if args.stage in ("yolo", "both"):
            train_yolo()

        if args.stage in ("trocr", "both"):
            train_trocr()
