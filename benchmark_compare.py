"""
benchmark_compare.py
====================
Head-to-head evaluation: DONUT  vs.  YOLOv8 + TrOCR-base-printed + Regex

Given a folder of receipt JPG images and a parallel folder of .txt label
files (4-line format: company, date, address, total — the SROIE Task-3
schema), this script:

  1. Runs BOTH pipelines on every image
  2. Computes per-field F1, exact-match Accuracy, NED, and wall-clock Speed
  3. Prints a rich terminal report
  4. Saves results/benchmark_results.json  (machine-readable)
  5. Saves figures/benchmark_comparison.pdf + .png  (journal-ready 2D plots)

Label file format (.txt, UTF-8, one field per line):
  Line 1 → company  (e.g. "WATSON'S SODA & SNACKS")
  Line 2 → date     (e.g. "25/12/2023")
  Line 3 → address  (e.g. "1 ORCHARD ROAD #B1-01 SINGAPORE")
  Line 4 → total    (e.g. "12.50")

Usage
-----
  python benchmark_compare.py \\
      --images_dir  /data/sroie/test_img \\
      --labels_dir  /data/sroie/test_key \\
      --yolo_model  /models/best.pt \\
      --donut_model naver-clova-ix/donut-base-finetuned-cord-v2 \\
      [--max_samples 50]   # optional: limit for quick smoke-test

Dependencies (pip install):
  torch torchvision transformers ultralytics pillow
  editdistance matplotlib tqdm numpy

Author: aiparallel0/kaggle contributors
License: MIT
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["SampleResult", "BenchmarkResult", "compare_all", "main"]

import numpy as np

# Defer heavy imports to avoid import-time crashes when deps are missing
try:
    import editdistance
    import matplotlib
    import torch
    from PIL import Image
    from tqdm import tqdm

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(
        f"FATAL: missing dependency — {e}\n"
        "Run: pip install torch torchvision transformers "
        "ultralytics pillow editdistance matplotlib tqdm numpy"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Constants — imported from single source of truth (constants.py)
# ─────────────────────────────────────────────────────────────────────────────
from constants import BASE_MODEL, FIELDS, IMAGE_EXTS, MAX_LENGTH


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SampleResult:
    image_name: str
    ground_truth: dict[str, str]
    prediction: dict[str, str]
    inference_time_ms: float


@dataclass
class BenchmarkResult:
    method: str  # "DONUT" | "YOLOv8+TrOCR+Regex"
    samples: list[SampleResult] = field(default_factory=list)

    # Aggregated metrics (filled by compute_metrics)
    per_field_f1: dict[str, float] = field(default_factory=dict)
    per_field_accuracy: dict[str, float] = field(default_factory=dict)
    global_f1: float = 0.0
    global_accuracy: float = 0.0
    global_ned: float = 0.0  # Normalised Edit Distance (lower = better)
    mean_inference_ms: float = 0.0
    median_inference_ms: float = 0.0
    p95_inference_ms: float = 0.0


# ──────────────────────────────���──────────────────────────────────────────────
# Text normalisation — shared by both pipelines
# ─────────────────────────────────────────────────────────────────────────────
def _normalise(text: str) -> str:
    """
    Normalise a field value for fair comparison:
      - Unicode NFC
      - lowercase
      - collapse whitespace
      - strip leading/trailing whitespace
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_f1(pred: str, gold: str) -> float:
    """
    Token-level F1 (SROIE Task-3 official metric).
    Both strings are normalised before comparison.
    Returns 0.0 when both are empty (defined as perfect match = 1.0
    only if BOTH are empty simultaneously — edge case handled below).
    """
    pred = _normalise(pred)
    gold = _normalise(gold)
    if pred == gold:
        return 1.0  # handles both-empty correctly
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    precision = sum(pred_tokens.count(t) for t in common) / len(pred_tokens)
    recall = sum(gold_tokens.count(t) for t in common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _ned(pred: str, gold: str) -> float:
    """
    Normalised Edit Distance.  Lower is better.
    NED = edit_distance(pred, gold) / max(len(pred), len(gold))
    Returns 0.0 when both are empty.
    """
    pred = _normalise(pred)
    gold = _normalise(gold)
    if pred == gold:
        return 0.0
    maxlen = max(len(pred), len(gold))
    if maxlen == 0:
        return 0.0
    return editdistance.eval(pred, gold) / maxlen


def _exact(pred: str, gold: str) -> float:
    """Binary exact match (after normalisation). Returns 1.0 or 0.0."""
    return float(_normalise(pred) == _normalise(gold))


# ─────────────────────────────────────────────────────────────────────────────
# Label loader — reads SROIE .txt key files
# ─────────────────────────────────────────────────────────────────────────────
def load_label_txt(path: Path) -> dict[str, str]:
    """
    Parse a SROIE-style .txt key file.

    Supported formats
    -----------------
    Format A (4-line plain text):
        WATSON'S SODA & SNACKS
        25/12/2023
        1 ORCHARD ROAD
        12.50

    Format B (key:value JSON):
        {"company": "...", "date": "...", "address": "...", "total": "..."}

    Format C (multiline key: value):
        company: WATSON'S SODA & SNACKS
        date: 25/12/2023
        address: 1 ORCHARD ROAD
        total: 12.50
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return {f: "" for f in FIELDS}

    # --- Format B: JSON ---
    if text.startswith("{"):
        try:
            data = json.loads(text)
            return {f: str(data.get(f, "")) for f in FIELDS}
        except json.JSONDecodeError:
            pass

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # --- Format C: key: value ---
    kv = {}
    for line in lines:
        for fld in FIELDS:
            if line.lower().startswith(fld + ":"):
                kv[fld] = line[len(fld) + 1 :].strip()
    if len(kv) >= 3:
        return {f: kv.get(f, "") for f in FIELDS}

    # --- Format A: positional 4-line ---
    padded = (lines + ["", "", "", ""])[:4]
    return dict(zip(FIELDS, padded))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────
def find_pairs(
    images_dir: Path, labels_dir: Path, max_samples: int | None = None
) -> list[tuple[Path, dict]]:
    """
    Find (image_path, ground_truth_dict) pairs.
    Matches image stem → label stem (case-insensitive).
    """
    label_index: dict[str, Path] = {}
    for ext in (".txt", ".json"):
        for lp in labels_dir.rglob(f"*{ext}"):
            label_index[lp.stem.lower()] = lp

    pairs = []
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_path.stem.lower()
        if stem not in label_index:
            continue  # no matching label, skip
        gt = load_label_txt(label_index[stem])
        pairs.append((img_path, gt))

    if max_samples:
        pairs = pairs[:max_samples]

    if not pairs:
        sys.exit(
            f"FATAL: No image+label pairs found.\n"
            f"  images_dir: {images_dir}\n"
            f"  labels_dir: {labels_dir}\n"
            "Check that image stems match label file stems."
        )
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# ██████  PIPELINE A — DONUT
# ─────────────────────────────────────────────────────────────────────────────
class DonutPipeline:
    """
    End-to-end DONUT inference.
    Loads VisionEncoderDecoderModel + DonutProcessor.
    Decodes XML tags and parses them into a dict.

    Critical: tie_word_embeddings must be False after loading a
    fine-tuned checkpoint that was saved with this flag.
    """

    def __init__(self, model_id_or_path: str, device: str = "auto"):
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )

        print(f"[DONUT] Loading model: {model_id_or_path}  →  {self.device}")
        self.processor = DonutProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)

        # Guard: ensure lm_head weight is present
        lm_head_w = getattr(self.model.decoder, "lm_head", None)
        if lm_head_w is None or lm_head_w.weight is None:
            print("[DONUT] WARNING: lm_head weight missing — attempting weight tie fix")
            self.model.decoder.lm_head.weight = self.model.decoder.model.decoder.embed_tokens.weight

        self.model.to(self.device).eval()
        self.task_prompt = "<s_sroie>"

    def _parse_output(self, token_str: str) -> dict[str, str]:
        """
        Parse DONUT XML output into a SROIE field dict.

        Expected format:
            <s_sroie><s_company>...</s_company><s_date>...</s_date>...</s_sroie>
        Falls back to empty dict on any parse failure.
        """
        result = {f: "" for f in FIELDS}
        for fld in FIELDS:
            pattern = rf"<s_{fld}>(.*?)</s_{fld}>"
            m = re.search(pattern, token_str, re.DOTALL)
            if m:
                result[fld] = m.group(1).strip()
        return result

    def predict(self, image: Image.Image) -> tuple[dict[str, str], float]:
        """
        Run inference on one PIL image.
        Returns (field_dict, inference_time_ms).
        """
        t0 = time.perf_counter()
        pixel_values = self.processor(image.convert("RGB"), return_tensors="pt").pixel_values.to(
            self.device
        )

        decoder_input_ids = self.processor.tokenizer(
            self.task_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=MAX_LENGTH,
                # FIX: early_stopping=True is invalid/deprecated with num_beams=1
                # (greedy decoding). Removed to eliminate Transformers deprecation
                # warnings. early_stopping is only meaningful for beam search.
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        token_str = self.processor.batch_decode(outputs.sequences)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._parse_output(token_str), elapsed_ms

    def run_benchmark(self, pairs: list[tuple[Path, dict]], desc: str = "DONUT") -> BenchmarkResult:
        result = BenchmarkResult(method="DONUT")
        for img_path, gt in tqdm(pairs, desc=desc, unit="img"):
            img = Image.open(img_path).convert("RGB")
            pred, ms = self.predict(img)
            result.samples.append(
                SampleResult(
                    image_name=img_path.name,
                    ground_truth=gt,
                    prediction=pred,
                    inference_time_ms=ms,
                )
            )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# ██████  PIPELINE B — YOLOv8 + TrOCR-base-printed + Regex
# ─────────────────────────────────────────────────────────────────────────────

# ── Regex patterns (pre-compiled, comprehensive) ────────────────────────────

# Date patterns — covers DD/MM/YYYY, YYYY-MM-DD, D MMM YYYY, etc.
_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b"),
    re.compile(r"\b\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}\b"),
    re.compile(
        r"\b\d{1,2}\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)\s*\d{2,4}\b",
        re.IGNORECASE,
    ),
    # SROIE date header keywords
    re.compile(r"(?:date|tarikh|tanggal)\s*[:\-]?\s*(\S+)", re.IGNORECASE),
]

# Total patterns — most specific first
_TOTAL_PATTERNS = [
    re.compile(
        r"(?:total|jumlah|amount\s+due|grand\s+total|total\s+amount)"
        r"\s*[:\-]?\s*(?:rm|myr|usd|\$|£|€)?\s*(\d[\d,]*\.\d{2})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:total|jumlah)\s*[:\-]?\s*(\d[\d,]*\.\d{2})",
        re.IGNORECASE,
    ),
    # Fallback: largest monetary value on the page
    re.compile(r"(?:rm|myr|\$|£|€)\s*(\d[\d,]*\.\d{2})"),
    re.compile(r"(\d[\d,]*\.\d{2})\s*(?:rm|myr)?"),
]

# Address heuristics — detect lines that look like addresses
_STREET_WORDS = re.compile(
    r"\b(?:jalan|jln|lorong|lot|no\.?|blok|block|level|floor|tingkat|"
    r"road|street|avenue|lane|drive|boulevard|plaza|mall|park|"
    r"batu|km|kilometer|mile)\b",
    re.IGNORECASE,
)
_POSTCODE_RE = re.compile(r"\b\d{5}\b")


class TrOCRYOLOPipeline:
    """
    Two-stage KIE pipeline:
      1. YOLOv8  — detects text region bounding boxes in the receipt image
      2. TrOCR   — reads text from each cropped region
      3. Regex   — assigns each text line to one of 4 SROIE fields

    For best.pt (custom-trained YOLO): pass yolo_model_path.
    TrOCR uses 'microsoft/trocr-base-printed' (no fine-tuning required
    for clean printed receipts).
    """

    def __init__(
        self,
        yolo_model_path: str = "yolov8n.pt",
        trocr_model_id: str = "microsoft/trocr-base-printed",
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from ultralytics import YOLO

        from train_trocr_yolo import _materialize_meta_buffers

        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        print(f"[YOLO]  Loading model: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)

        print(f"[TrOCR] Loading model: {trocr_model_id}  →  {self.device}")
        self.trocr_processor = TrOCRProcessor.from_pretrained(trocr_model_id)
        # FIX: low_cpu_mem_usage=False + _materialize_meta_buffers prevents the
        # meta-device crash on TrOCR's sinusoidal positional embedding buffer.
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
            trocr_model_id, low_cpu_mem_usage=False
        ).to(self.device)
        _materialize_meta_buffers(self.trocr_model, self.device)
        self.trocr_model.eval()

    # ── YOLO: detect text bounding boxes ───────────────────────────────────
    def _detect_boxes(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        """
        Run YOLOv8 on the image.
        Returns list of (x1, y1, x2, y2) boxes sorted top-to-bottom.
        Falls back to the full image as a single box if no detections.
        """
        results = self.yolo(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                # Sanity: skip degenerate boxes
                if (x2 - x1) < 5 or (y2 - y1) < 3:
                    continue
                boxes.append((x1, y1, x2, y2))

        # Sort top-to-bottom (reading order)
        boxes.sort(key=lambda b: b[1])

        if not boxes:
            # Fallback: treat full image as single text region
            w, h = image.size
            boxes = [(0, 0, w, h)]

        return boxes

    # ── TrOCR: read text from a crop ───────────────────────────────────────
    def _read_crop(self, crop: Image.Image) -> str:
        """Run TrOCR on a single cropped PIL image. Returns decoded string."""
        pixel_values = self.trocr_processor(
            crop.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated = self.trocr_model.generate(
                pixel_values,
                max_new_tokens=128,
            )

        text = self.trocr_processor.batch_decode(generated, skip_special_tokens=True)[0]
        return text.strip()

    # ── Regex: assign lines to fields ──────────────────────────────────────
    def _assign_fields(self, lines: list[str]) -> dict[str, str]:
        """
        Rule-based field assignment.

        Priority order (highest specificity wins):
          1. TOTAL   — explicit keyword + monetary value, or largest number
          2. DATE    — date-shaped token
          3. ADDRESS — street-word heuristic or postcode
          4. COMPANY — first non-empty line that isn't date/total/address

        Returns dict with all 4 FIELDS keys always present.
        """
        result = {f: "" for f in FIELDS}
        used: set = set()

        full_text = "\n".join(lines)

        # ── 1. TOTAL ────────────────────────────────────────────────────────
        for pat in _TOTAL_PATTERNS:
            m = pat.search(full_text)
            if m:
                # Extract the numeric part
                captured = m.group(1) if m.lastindex else m.group(0)
                # Clean: keep digits, comma, dot
                captured = re.sub(r"[^\d.,]", "", captured)
                result["total"] = captured
                # Mark the line containing this match as used
                for i, line in enumerate(lines):
                    if pat.search(line):
                        used.add(i)
                        break
                if result["total"]:
                    break

        # Fallback total: largest monetary value not yet assigned
        if not result["total"]:
            candidates = []
            for i, line in enumerate(lines):
                if i in used:
                    continue
                for m in re.finditer(r"\d[\d,]*\.\d{2}", line):
                    try:
                        val = float(m.group(0).replace(",", ""))
                        candidates.append((val, m.group(0), i))
                    except ValueError:
                        pass
            if candidates:
                candidates.sort(reverse=True)
                result["total"] = candidates[0][1]
                used.add(candidates[0][2])

        # ── 2. DATE ─────────��───────────────────────────────────────────────
        for i, line in enumerate(lines):
            if i in used:
                continue
            for pat in _DATE_PATTERNS:
                m = pat.search(line)
                if m:
                    # If pattern has a capture group, use it
                    captured = m.group(1) if m.lastindex else m.group(0)
                    result["date"] = captured.strip()
                    used.add(i)
                    break
            if result["date"]:
                break

        # ── 3. ADDRESS ──────────────────────────────────────────────────────
        # Collect lines with address signals (may span multiple lines)
        address_lines = []
        for i, line in enumerate(lines):
            if i in used:
                continue
            has_street = bool(_STREET_WORDS.search(line))
            has_postcode = bool(_POSTCODE_RE.search(line))
            # Address heuristic: 2+ words, contains digit or street word
            has_digit = bool(re.search(r"\d", line))
            looks_like_addr = (has_street or has_postcode) or (has_digit and len(line.split()) >= 3)
            if looks_like_addr:
                address_lines.append((i, line))

        if address_lines:
            # Take a run of consecutive address-like lines
            result["address"] = " ".join(ln for _, ln in address_lines[:4])
            for idx, _ in address_lines[:4]:
                used.add(idx)

        # ── 4. COMPANY ──────────────────────────────────────────────────────
        # Heuristic: company is usually the first prominent text line
        # (often all-caps, relatively short, appears near the top)
        # We prefer lines from the top third of the image
        top_third_limit = max(1, len(lines) // 3)
        for i, line in enumerate(lines):
            if i in used:
                continue
            if not line.strip():
                continue
            # Prefer lines from the top (header area) that are not purely numeric
            if re.match(r"^[\d\s\.,:]+$", line):
                continue
            result["company"] = line.strip()
            used.add(i)
            if i < top_third_limit:
                break  # confident it's the company header

        return result

    # ── Full pipeline for one image ─────────────────────────────────────────
    def predict(self, image: Image.Image) -> tuple[dict[str, str], float]:
        """
        Run the full YOLO→TrOCR→Regex pipeline on one image.
        Returns (field_dict, inference_time_ms).
        """
        t0 = time.perf_counter()

        # Stage 1: detect text boxes
        boxes = self._detect_boxes(image)

        # Stage 2: read text from each crop
        lines = []
        img_arr = image.convert("RGB")
        for x1, y1, x2, y2 in boxes:
            # Add a small padding to each crop
            pad = 4
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(img_arr.width, x2 + pad)
            y2p = min(img_arr.height, y2 + pad)
            crop = img_arr.crop((x1p, y1p, x2p, y2p))
            text = self._read_crop(crop)
            if text:
                lines.append(text)

        # Stage 3: regex field assignment
        pred = self._assign_fields(lines)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return pred, elapsed_ms

    def run_benchmark(
        self, pairs: list[tuple[Path, dict]], desc: str = "YOLO+TrOCR+Regex"
    ) -> BenchmarkResult:
        result = BenchmarkResult(method="YOLOv8+TrOCR+Regex")
        for img_path, gt in tqdm(pairs, desc=desc, unit="img"):
            img = Image.open(img_path).convert("RGB")
            pred, ms = self.predict(img)
            result.samples.append(
                SampleResult(
                    image_name=img_path.name,
                    ground_truth=gt,
                    prediction=pred,
                    inference_time_ms=ms,
                )
            )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(bench: BenchmarkResult) -> BenchmarkResult:
    """
    Fill bench.per_field_f1, per_field_accuracy, global_f1, global_ned,
    mean/median/p95 inference time in-place. Returns the same object.
    """
    if not bench.samples:
        return bench

    field_f1s: dict[str, list[float]] = {f: [] for f in FIELDS}
    field_exacts: dict[str, list[float]] = {f: [] for f in FIELDS}
    all_neds: list[float] = []
    times: list[float] = []

    for s in bench.samples:
        for fld in FIELDS:
            pred_val = s.prediction.get(fld, "")
            gt_val = s.ground_truth.get(fld, "")
            field_f1s[fld].append(_token_f1(pred_val, gt_val))
            field_exacts[fld].append(_exact(pred_val, gt_val))
            all_neds.append(_ned(pred_val, gt_val))
        times.append(s.inference_time_ms)

    bench.per_field_f1 = {f: float(np.mean(field_f1s[f])) for f in FIELDS}
    bench.per_field_accuracy = {f: float(np.mean(field_exacts[f])) for f in FIELDS}
    bench.global_f1 = float(np.mean([v for vals in field_f1s.values() for v in vals]))
    bench.global_accuracy = float(np.mean([v for vals in field_exacts.values() for v in vals]))
    bench.global_ned = float(np.mean(all_neds))
    bench.mean_inference_ms = float(np.mean(times))
    bench.median_inference_ms = float(np.median(times))
    bench.p95_inference_ms = float(np.percentile(times, 95))
    return bench


# ─────────────────────────────────────────────────────────────────────────────
# Terminal report
# ─��───────────────────────────────────────────────────────────────────────────
def print_report(results: list[BenchmarkResult], n_samples: int) -> None:
    """Print a formatted comparison table to stdout."""
    SEP = "-" * 72

    print(f"\n{'=' * 72}")
    print(f"  BENCHMARK REPORT -- {n_samples} samples")
    print(f"{'=' * 72}")

    # ── Summary table ────────────────────────────────────────────────────────
    header = f"{'Metric':<28}" + "".join(f"{r.method:>20}" for r in results)
    print(f"\n{header}")
    print(SEP)

    def row(name, vals, fmt=".4f"):
        return f"{name:<28}" + "".join(f"{v:>20{fmt}}" for v in vals)

    print(row("Global F1  [UP]", [r.global_f1 for r in results]))
    print(row("Global Accuracy  [UP]", [r.global_accuracy for r in results]))
    print(row("NED  [DN=better]", [r.global_ned for r in results]))
    print(row("Mean inference (ms)  [DN]", [r.mean_inference_ms for r in results], ".1f"))
    print(row("Median inference (ms)", [r.median_inference_ms for r in results], ".1f"))
    print(row("P95 inference (ms)", [r.p95_inference_ms for r in results], ".1f"))

    # -- Per-field breakdown -------------------------------------------------
    print(f"\n{'-' * 72}")
    print("  Per-Field F1")
    print(SEP)
    for fld in FIELDS:
        vals = [r.per_field_f1.get(fld, 0.0) for r in results]
        print(row(f"  {fld:<26}", vals))

    print(f"\n{'-' * 72}")
    print("  Per-Field Exact-Match Accuracy")
    print(SEP)
    for fld in FIELDS:
        vals = [r.per_field_accuracy.get(fld, 0.0) for r in results]
        print(row(f"  {fld:<26}", vals))

    print(f"\n{'=' * 72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation — journal-ready 2D plots
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.4,
        "figure.dpi": 300,
    }
)

_METHOD_COLORS = {
    "DONUT": "#4C72B0",
    "YOLOv8+TrOCR+Regex": "#DD8452",
}


def plot_results(results: list[BenchmarkResult], out_dir: Path) -> None:
    """
    Generate 3 publication-quality plots:
      Fig 1 — Per-field F1 grouped bar chart
      Fig 2 — Speed distribution (violin)
      Fig 3 — Radar / spider chart (F1, Accuracy, 1-NED, Speed-normalised)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Fig 1: Per-field F1 grouped bar chart ────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.0))
    x = np.arange(len(FIELDS))
    n = len(results)
    width = 0.72 / n
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    for i, res in enumerate(results):
        vals = [res.per_field_f1.get(f, 0.0) for f in FIELDS]
        color = _METHOD_COLORS.get(res.method, f"C{i}")
        bars = ax1.bar(
            x + offsets[i],
            vals,
            width * 0.92,
            label=res.method,
            color=color,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f.capitalize() for f in FIELDS])
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("Token F1 (↑)")
    ax1.set_title("(a) Per-Field F1: DONUT vs. YOLOv8+TrOCR+Regex")
    ax1.legend(loc="upper right", frameon=False)
    ax1.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    _save(fig1, out_dir / "fig1_field_f1")

    # ── Fig 2: Speed distribution — violin plot ───────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
    times_data = [[s.inference_time_ms for s in r.samples] for r in results]
    labels_list = [r.method for r in results]
    colors_list = [_METHOD_COLORS.get(r.method, f"C{i}") for i, r in enumerate(results)]

    parts = ax2.violinplot(
        times_data, positions=range(1, len(results) + 1), showmedians=True, showextrema=True
    )
    for pc, color in zip(parts["bodies"], colors_list):
        pc.set_facecolor(color)
        pc.set_alpha(0.72)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax2.set_xticks(range(1, len(results) + 1))
    ax2.set_xticklabels(labels_list, rotation=12, ha="right")
    ax2.set_ylabel("Inference time (ms) ↓")
    ax2.set_title("(b) Inference Speed Distribution")
    ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    _save(fig2, out_dir / "fig2_speed")

    # ── Fig 3: Radar chart — 4 axes ──────────────────────────────────────────
    # Axes: F1, Accuracy, 1-NED (higher=better), Speed-score (1 - norm_time)
    max_time = max(r.mean_inference_ms for r in results) or 1.0
    radar_labels = ["F1", "Accuracy", "1-NED", "Speed"]

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig3, ax3 = plt.subplots(figsize=(3.8, 3.8), subplot_kw={"projection": "polar"})
    ax3.set_theta_offset(np.pi / 2)
    ax3.set_theta_direction(-1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(radar_labels, size=8)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=6)
    ax3.grid(linestyle=":", linewidth=0.6, alpha=0.7)
    ax3.set_title("(c) Multi-Metric Radar", pad=14, size=9)

    for i, res in enumerate(results):
        speed_score = 1.0 - (res.mean_inference_ms / max_time)
        values = [
            res.global_f1,
            res.global_accuracy,
            max(0.0, 1.0 - res.global_ned),
            speed_score,
        ]
        values += values[:1]
        color = _METHOD_COLORS.get(res.method, f"C{i}")
        ax3.plot(angles, values, color=color, linewidth=1.6, label=res.method)
        ax3.fill(angles, values, color=color, alpha=0.18)

    ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=False, fontsize=7)

    _save(fig3, out_dir / "fig3_radar")

    print(f"\n[Plots] Saved to: {out_dir}")


def _save(fig, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialiser
# ─────────────────────────────────────────────────────────────────────────────
def save_json(results: list[BenchmarkResult], out_path: Path) -> None:
    """Save all benchmark results to a machine-readable JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in results:
        payload.append(
            {
                "method": r.method,
                "global_f1": r.global_f1,
                "global_accuracy": r.global_accuracy,
                "global_ned": r.global_ned,
                "mean_inference_ms": r.mean_inference_ms,
                "median_inference_ms": r.median_inference_ms,
                "p95_inference_ms": r.p95_inference_ms,
                "per_field_f1": r.per_field_f1,
                "per_field_accuracy": r.per_field_accuracy,
                "n_samples": len(r.samples),
                "samples": [
                    {
                        "image": s.image_name,
                        "gt": s.ground_truth,
                        "pred": s.prediction,
                        "inference_ms": round(s.inference_time_ms, 2),
                    }
                    for s in r.samples
                ],
            }
        )
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[JSON]  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark DONUT vs YOLOv8+TrOCR+Regex on receipt KIE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--images_dir",
        type=Path,
        required=True,
        help="Directory containing receipt JPG/PNG images",
    )
    p.add_argument(
        "--labels_dir",
        type=Path,
        required=True,
        help="Directory containing .txt label files (4-line SROIE format)",
    )
    p.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8n.pt",
        help="Path to YOLO .pt weights (e.g. best.pt from your training)",
    )
    p.add_argument(
        "--donut_model",
        type=str,
        default=BASE_MODEL,
        help="DONUT model ID (HuggingFace) or local path",
    )
    p.add_argument(
        "--trocr_model",
        type=str,
        default="microsoft/trocr-base-printed",
        help="TrOCR model ID (HuggingFace) or local path",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (useful for quick tests)",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory for JSON output and figures",
    )
    p.add_argument(
        "--skip_donut",
        action="store_true",
        help="Skip DONUT pipeline (run only YOLO+TrOCR)",
    )
    p.add_argument(
        "--skip_yolo",
        action="store_true",
        help="Skip YOLO+TrOCR pipeline (run only DONUT)",
    )
    p.add_argument(
        "--yolo_conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold",
    )
    p.add_argument(
        "--yolo_iou",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold",
    )
    return p.parse_args()


def compare_all(
    results_dir: Path = Path("results"), figures_dir: Path = Path("figures")
) -> None:
    """Generate cross-architecture comparison plots from saved benchmark results.

    Called by run_all.py stage_comparison.  Gracefully no-ops when result
    files are not yet available (e.g. first --yolo-only run with no DONUT data).
    """
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)

    benchmark_path = results_dir / "benchmark_results.json"
    all_exp_path = results_dir / "all_experiments.json"

    if not benchmark_path.exists() and not all_exp_path.exists():
        print(
            "  [compare_all] No result files found "
            f"({benchmark_path}, {all_exp_path}) — skipping comparison plots."
        )
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark results if available
    bench_data = {}
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            bench_data = json.load(f)

    # Load DONUT experiment results if available
    donut_data = {}
    if all_exp_path.exists():
        with open(all_exp_path) as f:
            donut_data = json.load(f)

    # Generate a simple comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Cross-Architecture Comparison (DONUT vs TrOCR+YOLO+Regex)")
    ax.set_xlabel("Architecture / Experiment")
    ax.set_ylabel("F1 Score")

    labels: list[str] = []
    values: list[float] = []

    # Add DONUT experiment F1s
    if isinstance(donut_data, dict):
        for exp_key, exp_val in donut_data.items():
            if isinstance(exp_val, dict):
                f1 = exp_val.get("metrics", {}).get("global_f1")
                if f1 is not None:
                    labels.append(f"DONUT {exp_key}")
                    values.append(float(f1))

    # Add benchmark pipeline F1s
    if isinstance(bench_data, list):
        for entry in bench_data:
            method = entry.get("method", "Unknown")
            f1 = entry.get("global_f1") or entry.get("metrics", {}).get("global_f1")
            if f1 is not None:
                labels.append(method)
                values.append(float(f1))

    if labels:
        ax.bar(range(len(labels)), values, tick_label=labels)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path = figures_dir / "cross_arch_comparison.png"
        fig.savefig(str(out_path), dpi=150)
        print(f"  [compare_all] Saved comparison plot → {out_path}")
    else:
        print("  [compare_all] No F1 metrics found in result files — skipping plot.")

    plt.close(fig)


def main() -> None:
    args = parse_args()

    # ── Validate inputs ──────────────────────────────────────────────────────
    if not args.images_dir.exists():
        sys.exit(f"FATAL: images_dir not found: {args.images_dir}")
    if not args.labels_dir.exists():
        sys.exit(f"FATAL: labels_dir not found: {args.labels_dir}")

    print(f"\n{'=' * 60}")
    print("  RECEIPT KIE BENCHMARK")
    print(f"  DONUT model  : {args.donut_model}")
    print(f"  YOLO model   : {args.yolo_model}")
    print(f"  TrOCR model  : {args.trocr_model}")
    print(f"  images_dir   : {args.images_dir}")
    print(f"  labels_dir   : {args.labels_dir}")
    if args.max_samples:
        print(f"  max_samples  : {args.max_samples}")
    print(f"{'=' * 60}\n")

    # ── Discover paired data ─────────────────────────────────────────────────
    pairs = find_pairs(args.images_dir, args.labels_dir, args.max_samples)
    print(f"Found {len(pairs)} image+label pairs.\n")

    all_results: list[BenchmarkResult] = []

    # ── Run DONUT ───────────────────────���────────────────────────────────────
    if not args.skip_donut:
        print("--- Pipeline A: DONUT -----------------------------------------------")
        donut = DonutPipeline(model_id_or_path=args.donut_model)
        donut_result = donut.run_benchmark(pairs, desc="DONUT inference")
        donut_result = compute_metrics(donut_result)
        all_results.append(donut_result)
        # Free GPU memory before next pipeline
        del donut
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()

    # ── Run YOLOv8 + TrOCR + Regex ──────────────────────────────────────────
    if not args.skip_yolo:
        print("\n--- Pipeline B: YOLOv8 + TrOCR-base-printed + Regex ----------")
        yolo_trocr = TrOCRYOLOPipeline(
            yolo_model_path=args.yolo_model,
            trocr_model_id=args.trocr_model,
            conf_threshold=args.yolo_conf,
            iou_threshold=args.yolo_iou,
        )
        yolo_result = yolo_trocr.run_benchmark(pairs, desc="YOLO+TrOCR+Regex inference")
        yolo_result = compute_metrics(yolo_result)
        all_results.append(yolo_result)
        del yolo_trocr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()

    if not all_results:
        sys.exit("FATAL: Both pipelines were skipped. Nothing to report.")

    # ── Report ───────────────────────────────────────────────────────────────
    print_report(all_results, n_samples=len(pairs))

    # ── Save JSON ────────────────────────────────────────────────────────────
    json_path = args.output_dir / "benchmark_results.json"
    save_json(all_results, json_path)

    # ── Save plots ───────────────────────────────────────────────────────────
    plot_results(all_results, out_dir=args.output_dir / "figures")


if __name__ == "__main__":
    main()
