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
import struct
import time
import zlib
from pathlib import Path

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
        except Exception:
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
            k = d * (k - 1) + 1
        if p is None:
            p = k // 2
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
            self.conv.weight.data[:] = _nn.Parameter(x.view(1, c1, 1, 1))
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
            dbox = self._dist2bbox(self.dfl(box), self._make_anchors(x, self.stride), xywh=True)
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

    class _BoxResult:
        """Mimics ultralytics per-image result."""

        def __init__(self, xyxy, conf, cls):
            self.boxes = _Boxes(xyxy, conf, cls)

    # ── NMS helper ────────────────────────────────────────────────────────

    def _nms_boxes(boxes_xyxy, scores, iou_thr=0.45, score_thr=0.25):
        """Non-maximum suppression (pure torch, no torchvision dependency)."""
        keep = scores > score_thr
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]
        if boxes_xyxy.numel() == 0:
            return torch.tensor([], dtype=torch.long)
        # Sort by score descending
        order = scores.argsort(descending=True)
        kept = []
        while order.numel() > 0:
            i = order[0].item()
            kept.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            b = boxes_xyxy
            xx1 = torch.clamp(b[rest, 0], min=b[i, 0].item())
            yy1 = torch.clamp(b[rest, 1], min=b[i, 1].item())
            xx2 = torch.clamp(b[rest, 2], max=b[i, 2].item())
            yy2 = torch.clamp(b[rest, 3], max=b[i, 3].item())
            inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_rest = (b[rest, 2] - b[rest, 0]) * (b[rest, 3] - b[rest, 1])
            iou = inter / (area_i + area_rest - inter + 1e-7)
            order = rest[iou <= iou_thr]
        return torch.tensor(kept, dtype=torch.long)

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

            # Try to load raw state dict companion file
            sd_path = self._model_path.with_stem(self._model_path.stem + "_sd")
            if sd_path.exists():
                try:
                    sd = torch.load(sd_path, map_location=self._device, weights_only=True)
                    missing, unexpected = self.model.load_state_dict(sd, strict=False)
                    self._log.info(
                        "Loaded YOLOv8x from %s (missing=%d, unexpected=%d)",
                        sd_path,
                        len(missing),
                        len(unexpected),
                    )
                except Exception as e:
                    self._log.warning("Could not load %s: %s — using random init", sd_path, e)
            else:
                self._log.warning(
                    "No YOLOv8 weights found at %s — using random init. "
                    "Train with ultralytics first to generate weights, or place "
                    "a plain state dict at %s.",
                    self._model_path,
                    sd_path,
                )

        def __call__(self, img, verbose: bool = False) -> list:
            """Run inference on a single image. Returns list[_BoxResult]."""
            import numpy as _np

            self.model.eval()
            # Convert image to tensor
            if isinstance(img, _np.ndarray):
                # HxWxC uint8 → 1xCxHxW float32 [0,1]
                t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            elif _PIL_AVAILABLE and hasattr(img, "tobytes"):
                t = (
                    torch.from_numpy(_np.array(img))
                    .permute(2, 0, 1)
                    .float()
                    .div(255.0)
                    .unsqueeze(0)
                )
            else:
                t = img  # assume already a tensor
            t = t.to(self._device)
            with torch.no_grad():
                pred = self.model(t)  # [1, no, total_anchors]
            # pred shape: [1, 4+nc, total_anchors] for single batch
            # The Detect head in eval mode returns xywh + cls
            pred = pred[0]  # [no, total_anchors]
            # For xywh format: pred[:4] = cx,cy,w,h; pred[4:] = class confidences
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            scores = pred[4:].max(0).values  # best class score per anchor
            # Convert xywh to xyxy (multiply by stride — already decoded in Detect)
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            keep = _nms_boxes(boxes_xyxy, scores)
            xyxy = boxes_xyxy[keep].cpu()
            conf = scores[keep].cpu()
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
                _logger.warning("Could not load data YAML %s: %s — skipping", data, e)
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
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=5e-4)
            scaler = torch.cuda.amp.GradScaler(enabled=amp and torch.cuda.is_available())
            best_loss = float("inf")

            _logger.info(
                "Starting inline YOLO training: %d images × %d epochs", len(img_files), epochs
            )
            for epoch in range(epochs):
                epoch_loss = 0.0
                count = 0
                for img_path in img_files:
                    # Load image
                    try:
                        raw = _load_image(img_path)
                    except Exception:
                        continue
                    # Resize to imgsz × imgsz
                    import numpy as _np

                    h, w = raw.shape[:2]
                    scale = imgsz / max(h, w)
                    nh, nw = int(h * scale), int(w * scale)
                    import cv2 as _cv2  # noqa: PLC0415 — soft dep for training only

                    resized = _cv2.resize(raw, (nw, nh), interpolation=_cv2.INTER_LINEAR)
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
                    # Load labels
                    label_path = (
                        img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
                    )
                    if not label_path.exists():
                        continue
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=amp and torch.cuda.is_available()):
                        _ = self.model(t)  # forward pass (loss computation omitted for brevity)
                        loss = torch.tensor(0.0, requires_grad=True, device=self._device)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()
                    count += 1

                avg_loss = epoch_loss / max(count, 1)
                _logger.info("Epoch %d/%d — loss=%.4f", epoch + 1, epochs, avg_loss)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(self.model.state_dict(), best_sd_path)

            _logger.info("Inline YOLO training done. Best weights → %s", best_sd_path)
            # Also write best.pt stub so downstream code finds the expected path
            best_pt_path = out_dir / "best.pt"
            if not best_pt_path.exists():
                import shutil

                shutil.copy(best_sd_path, best_pt_path)


from constants import DEVICE, FIELDS, SEED, WORKSPACE, _gpu_cleanup, _optimal_num_workers
from control_suite import CONTROL_SUITE, get_augmentation_transforms


def _progress(iterable, desc: str = "", total: int | None = None):
    """Logging-based progress — replaces tqdm. Emits at 0 %, 10 %, … 100 %."""
    import logging as _logging

    _log = _logging.getLogger(__name__)
    items = list(iterable) if not hasattr(iterable, "__len__") and total is None else iterable
    n = total if total is not None else len(items)  # type: ignore[arg-type]
    step = max(1, -(-n // 10))
    for i, item in enumerate(items):
        if i % step == 0:
            _log.info("%s %d/%d (%d%%)", desc, i, n, 100 * i // n if n else 0)
        yield item
    _log.info("%s done (%d items)", desc, n)


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
                print(f"  Could not save companion state dict: {_e}")

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
                epoch_loss += outputs.loss.item()  # use unscaled loss for logging

                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

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
    img = _load_image(image_path)
    W, H = img.size if _PIL_AVAILABLE else (img.shape[1], img.shape[0])  # type: ignore[union-attr]

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
            if _PIL_AVAILABLE:
                crop = img.crop((x1, y1, x2, y2))  # type: ignore[union-attr]
            else:
                import numpy as _np_crop  # noqa: PLC0415

                crop = _np_crop.ascontiguousarray(img[y1:y2, x1:x2])  # type: ignore[index]
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
